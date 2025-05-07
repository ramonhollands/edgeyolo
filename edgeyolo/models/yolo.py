import argparse
from loguru import logger
import sys
from copy import deepcopy

sys.path.append('./')  # to run '$ python *.py' files in subdirectories


import torch
from .common import *
from .experimental import *
from ..utils2.autoanchor import check_anchor_order
from ..utils2.general import make_divisible, check_file, set_logging
from ..utils2.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr
from ..utils2.loss import SigmoidBin

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Identity(nn.Module):
    """
    for pretrain
    """

    stride = [8, 16, 32]

    def __init__(self, out_channel, act="act", input_chs=()):
        super(Identity, self).__init__()
        act = {
            True: True,
            "silu": True,
            "relu": nn.ReLU(),
            "leakyrelu": nn.LeakyReLU(0.1),
            "gelu": nn.GELU(),
            "sigmoid": nn.Sigmoid(),
            None: None
        }[act.lower()]
        self.out = nn.ModuleList(RepConv(in_c, out_channel, k=3, s=1, act=act) for in_c in input_chs)
        # print(out_channel, input_chs)

    def forward(self, x):
        return [out(x[i]) for i, out in enumerate(self.out)]


class YOLOXDetect(nn.Module):

    stride = [8, 16, 32]    # strides computed during build
    export = False          # onnx export
    is_fused = False
    divide_x = None  # for tflite export
    divide_y = None
    divide_w = None
    divide_h = None
    no_decode_layer = True

    def __init__(self, nc=80, anchors=(), conv=Conv, ch=(),
                 no_decode_layer=True,
                 divide_x=None, divide_y=None, divide_w=None, divide_h=None):

        self.divide_x = divide_x
        self.divide_y = divide_y
        self.divide_w = divide_w
        self.divide_h = divide_h

        self.no_decode_layer = True

        super(YOLOXDetect, self).__init__()
        self.ch = ch
        self.n_anchors = len(anchors[0]) // 2
        self.n_layers = len(anchors)

        self.grid = [torch.zeros(1)] * self.n_layers  # init grid

        self.num_classes = nc
        self.decode_in_inference = True  # for deploy, set to False
        # print(ch)
        x = self.hide_ch = int(ch[0] * 1)
        self.stems = nn.ModuleList(Conv(y, x, k=1, s=1, act=True) for y in ch)
        self.cls_convs = nn.ModuleList(conv(x, x, k=3, s=1, act=True) for y in ch)
        self.reg_convs = nn.ModuleList(conv(x, x, k=3, s=1, act=True) for y in ch)
        self.cls_preds = nn.ModuleList(nn.Conv2d(x, self.n_anchors * self.num_classes, 1, 1, 0) for _ in ch)
        self.reg_preds = nn.ModuleList(nn.Conv2d(x, 4, 1, 1, 0) for _ in ch)
        self.obj_preds = nn.ModuleList(nn.Conv2d(x, 1, 1, 1, 0) for _ in ch)

        self.ia = nn.ModuleList(ImplicitA(x) for _ in ch)
        self.im = nn.ModuleList(ImplicitM((self.num_classes + 5) * self.n_anchors) for _ in ch)
        self.rego_preds = None

    @staticmethod
    def _make_grid(nx=20, ny=20, n_anchors=1):

        print('Generating grid', nx, ny, n_anchors)

        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])

        if n_anchors == 1:
            return torch.stack((xv, yv), 2).view((1, ny, nx, 2)).float()
        else:
            return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def forward(self, x):
        z = []
        export = True # torch.onnx.is_in_onnx_export()        
        
        for i in range(self.n_layers):

            out = self.stems[i](x[i])
            cls_x = out
            reg_x = out

            cls_feat = self.cls_convs[i](cls_x)
            reg_feat = self.reg_convs[i](reg_x)

            if not self.is_fused:
                x_cls = self.cls_preds[i](self.ia[i](cls_feat))
                x_reg = self.reg_preds[i](self.ia[i](reg_feat))
                x_obj = self.obj_preds[i](self.ia[i](reg_feat))
                x[i] = torch.cat([x_reg, x_obj, x_cls], dim=1)
                x[i] = self.im[i](x[i])
            else:
                x_cls = self.cls_preds[i](cls_feat)
                x_rego = self.rego_preds[i](reg_feat)
                x[i] = torch.cat([x_rego, x_cls], dim=1)

            if export:
                bs, _, ny, nx = x[i].shape  # x(bs, 85, 20, 20)
                x[i] = x[i].view(bs, self.num_classes + 5, ny, nx).permute(0, 2, 3, 1).contiguous()
                self.grid[i] = self._make_grid(nx, ny, self.n_anchors).to(x[i].device)
                y = x[i]

                print('self.stride[i]', self.stride[i])
                xy, wh, conf = y.split((2, 2, self.num_classes + 1), 3)
                conf = conf.sigmoid()
                xy = (xy + self.grid[i]) * self.stride[i]  # new xy
                stride_tensor = torch.full_like(conf[..., :1], self.stride[i])
                y = torch.cat((xy, wh, conf, stride_tensor), 3)
                z.append(y.view(bs, -1, self.num_classes + 6))
            else:
                print('EXPORT=False')

        # return torch.cat(z, 1)
        print('(self.training or self.no_decode_layer)', self.training, self.no_decode_layer)
        return x if (self.training or self.no_decode_layer) else torch.cat(z, 1)

    def fuse(self):
        print("YOLOXDetect.fuse")
        with torch.no_grad():
            for i in range(self.n_layers):
                try:

                    c1_, c2_, _, _ = self.ia[i].implicit.shape

                    c1, c2, _, _ = self.cls_preds[i].weight.shape
                    self.cls_preds[i].bias += torch.matmul(self.cls_preds[i].weight.reshape(c1, c2),
                                                           self.ia[i].implicit.reshape(c2_, c1_)).squeeze(1)
                    c1, c2, _, _ = self.reg_preds[i].weight.shape
                    self.reg_preds[i].bias += torch.matmul(self.reg_preds[i].weight.reshape(c1, c2),
                                                           self.ia[i].implicit.reshape(c2_, c1_)).squeeze(1)
                    c1, c2, _, _ = self.obj_preds[i].weight.shape
                    self.obj_preds[i].bias += torch.matmul(self.obj_preds[i].weight.reshape(c1, c2),
                                                           self.ia[i].implicit.reshape(c2_, c1_)).squeeze(1)
                except:
                    print(i)
                    raise

            # fuse ImplicitM and Convolution
            for i in range(self.n_layers):
                c1, c2, _, _ = self.im[i].implicit.shape   # c2 = self.no * self.na = 85 * 1

                b = self.im[i].implicit.reshape(c2)
                w = self.im[i].implicit.transpose(0, 1)
                self.cls_preds[i].bias *= b[5:]
                self.cls_preds[i].weight *= w[5:]

                self.reg_preds[i].bias *= b[:4]
                self.reg_preds[i].weight *= w[:4]

                self.obj_preds[i].bias *= b[4:5]
                self.obj_preds[i].weight *= w[4:5]

            self.rego_preds = nn.ModuleList(nn.Conv2d(self.hide_ch, 5, 1) for x in self.ch)
            for i in range(self.n_layers):
                self.rego_preds[i].bias.requires_grad = False
                self.rego_preds[i].weight.requires_grad = False

                self.rego_preds[i].bias[:4] = self.reg_preds[i].bias
                self.rego_preds[i].bias[4:5] = self.obj_preds[i].bias

                self.rego_preds[i].weight[:4] = self.reg_preds[i].weight
                self.rego_preds[i].weight[4:5] = self.obj_preds[i].weight

            self.is_fused = True


class IBin(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=(), bin_count=21):  # detection layer
        super(IBin, self).__init__()
        self.nc = nc  # number of classes
        self.bin_count = bin_count

        self.w_bin_sigmoid = SigmoidBin(bin_count=self.bin_count, min=0.0, max=4.0)
        self.h_bin_sigmoid = SigmoidBin(bin_count=self.bin_count, min=0.0, max=4.0)
        # classes, x,y,obj
        self.no = nc + 3 + \
            self.w_bin_sigmoid.get_length() + self.h_bin_sigmoid.get_length()   # w-bce, h-bce
            # + self.x_bin_sigmoid.get_length() + self.y_bin_sigmoid.get_length()
        
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

    def forward(self, x):

        #self.x_bin_sigmoid.use_fw_regression = True
        #self.y_bin_sigmoid.use_fw_regression = True
        self.w_bin_sigmoid.use_fw_regression = True
        self.h_bin_sigmoid.use_fw_regression = True
        
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                #y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                

                #px = (self.x_bin_sigmoid.forward(y[..., 0:12]) + self.grid[i][..., 0]) * self.stride[i]
                #py = (self.y_bin_sigmoid.forward(y[..., 12:24]) + self.grid[i][..., 1]) * self.stride[i]

                pw = self.w_bin_sigmoid.forward(y[..., 2:24]) * self.anchor_grid[i][..., 0]
                ph = self.h_bin_sigmoid.forward(y[..., 24:46]) * self.anchor_grid[i][..., 1]

                #y[..., 0] = px
                #y[..., 1] = py
                y[..., 2] = pw
                y[..., 3] = ph
                
                y = torch.cat((y[..., 0:4], y[..., 46:]), dim=-1)
                
                z.append(y.view(bs, -1, y.shape[-1]))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolor-csp-c.yaml', ch=3, nc=None, anchors=None, divide_x=1, divide_y=1, divide_h=1, divide_w=1, no_decode_layer=False, is_file=True):  # models, input channels, number of classes
        super(Model, self).__init__()
        self.traced = False
        if isinstance(cfg, dict):
            self.yaml = cfg  # models dict
        else:  # is *.yaml
            import yaml     # for torch hub
            if is_file:
                self.yaml_file = Path(cfg).name
                with open(cfg) as f:
                    self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # models dict
            else:
                self.yaml = yaml.load(cfg, Loader=yaml.SafeLoader)  # models dict
        # print(self.yaml)

        # logger.info(self.yaml)
        # logger.info(is_file)

        # Define models
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels

        # print(ch)
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding models.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding models.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # models, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (YOLOXDetect, Identity)):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.stride = [8, 16, 32]
            self.stride = m.stride
            m.divide_x = divide_x
            m.divide_y = divide_y
            m.divide_h = divide_h
            m.divide_w = divide_w
            m.no_decode_layer = no_decode_layer
            m.training = False

            # print('Strides: %s' % m.stride.tolist())
        if isinstance(m, IBin):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases_bin()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        # logger.info('')

    def forward(self, x, augment=False, profile=False):
        # x /= 255.0
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if not hasattr(self, 'traced'):
                self.traced=False

            if profile:
                c = isinstance(m, (Detect, IBin))
                o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                for _ in range(10):
                    m(x.copy() if c else x)
                t = time_synchronized()
                for _ in range(10):
                    m(x.copy() if c else x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_aux_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, mi2, s in zip(m.m, m.m2, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            b2 = mi2.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b2.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b2.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi2.bias = torch.nn.Parameter(b2.view(-1), requires_grad=True)

    def _initialize_biases_bin(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Bin() module
        bc = m.bin_count
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            old = b[:, (0,1,2,bc+3)].data
            obj_idx = 2*bc+4
            b[:, :obj_idx].data += math.log(0.6 / (bc + 1 - 0.99))
            b[:, obj_idx].data += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, (obj_idx+1):].data += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            b[:, (0,1,2,bc+3)].data = old
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_biases_kpt(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.models.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def reparameterize(self):

        return self.fuse()

    def fuse2(self):
        print("HERE.fuse2")

    def fuse(self):  # fuse models Conv2d() + BatchNorm2d() layers
        print('Reparameterizing models...')
        # print('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, RepConv):
                #print(f" fuse_repvgg_block")
                m.fuse_repvgg_block()
            elif isinstance(m, RepConv_OREPA):
                #print(f" switch_to_deploy")
                m.switch_to_deploy()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
            elif isinstance(m, (YOLOXDetect)):
                m.fuse()
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap models
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print models information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    # logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, RobustConv, RobustConv2, DWConv, GhostConv, RepConv, RepConv_OREPA, DownC, 
                 SPP, SPPF, SPPCSPC, GhostSPPCSPC, MixConv2d, Focus, Stem, GhostStem, CrossConv, 
                 Bottleneck, BottleneckCSPA, BottleneckCSPB, BottleneckCSPC, 
                 RepBottleneck, RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,  
                 Res, ResCSPA, ResCSPB, ResCSPC, 
                 RepRes, RepResCSPA, RepResCSPB, RepResCSPC, 
                 ResX, ResXCSPA, ResXCSPB, ResXCSPC, 
                 RepResX, RepResXCSPA, RepResXCSPB, RepResXCSPC, 
                 Ghost, GhostCSPA, GhostCSPB, GhostCSPC,
                 SwinTransformerBlock, STCSPA, STCSPB, STCSPC,
                 SwinTransformer2Block, ST2CSPA, ST2CSPB, ST2CSPC]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [DownC, SPPCSPC, GhostSPPCSPC, 
                     BottleneckCSPA, BottleneckCSPB, BottleneckCSPC, 
                     RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC, 
                     ResCSPA, ResCSPB, ResCSPC, 
                     RepResCSPA, RepResCSPB, RepResCSPC, 
                     ResXCSPA, ResXCSPB, ResXCSPC, 
                     RepResXCSPA, RepResXCSPB, RepResXCSPC,
                     GhostCSPA, GhostCSPB, GhostCSPC,
                     STCSPA, STCSPB, STCSPC,
                     ST2CSPA, ST2CSPB, ST2CSPC]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Chuncat:
            c2 = sum([ch[x] for x in f])
        elif m is Shortcut:
            c2 = ch[f[0]]
        elif m is Foldcut:
            c2 = ch[f] // 2
        elif m in [IBin, YOLOXDetect, Identity]:
            args.append([ch[x] for x in f])
            if not m == Identity:
                if isinstance(args[1], int):  # number of anchors
                    args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is ReOrg:
            c2 = ch[f] * 4
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        # print(args)
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        # logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolor-csp-c.yaml', help='models.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile models speed')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create models
    model = Model(opt.cfg).to(device)
    model.train()
    
    # if opt.profile:
    #     img = torch.rand(1, 3, 640, 640).to(device)
    #     y = model(img, profile=True)

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = models(img, profile=True)

    # Tensorboard
    # from torch.utils2.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(models.models, img)  # add models to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add models to tensorboard
