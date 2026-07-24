import argparse
from loguru import logger
import sys
from copy import deepcopy
from pathlib import Path

sys.path.append('./')  # to run '$ python *.py' files in subdirectories


import torch
from .common import *
from ..utils2.general import make_divisible, check_file, set_logging
from ..utils2.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

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
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])

        if n_anchors == 1:
            return torch.stack((xv, yv), 2).view((1, ny, nx, 2)).float()
        else:
            return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def forward(self, x):
        z = []
        self.training |= self.export
        export = torch.onnx.is_in_onnx_export()


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

            if self.n_anchors != 1:
                exit('Not supported')

            if export and self.n_anchors == 1:
                bs, _, ny, nx = x[i].shape  # x(bs, 85, 20, 20)
                x[i] = x[i].view(bs, self.num_classes + 5, ny, nx).permute(0, 2, 3, 1).contiguous()

                self.grid[i] = self._make_grid(nx, ny, self.n_anchors).to(x[i].device)
                y = x[i]

                xy, wh, conf = y.split((2, 2, self.num_classes + 1), 3)

                conf = conf.sigmoid()

                xy = (xy + self.grid[i]) * self.stride[i]  # new xy

                wh = torch.exp(wh) * self.stride[i]  # new wh

                xy = xy / torch.tensor([self.divide_x, self.divide_y])
                wh = wh / torch.tensor([self.divide_w, self.divide_h])

                # export friendly wh.clamp_(min=0.0, max=1.0)
                wh = torch.relu(wh)
                wh = 1 - torch.relu(1 - wh)
                xy = torch.relu(xy)
                xy = 1 - torch.relu(1 - xy)

                y = torch.cat((xy, wh, conf), 3)

                # Flatten the tensor
                z.append(y.view(bs, -1, self.num_classes + 5))
            else:
                bs, _, ny, nx = x[i].shape  # x(bs,85,20,20) to x(bs,1,20,20,85)
                x[i] = x[i].view(bs, self.n_anchors, self.num_classes + 5, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

                if not self.training and not self.no_decode_layer:  # inference
                    # if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny, self.n_anchors).to(x[i].device)

                    y = x[i]
                    if not export:
                        y[..., 4:] = y[..., 4:].sigmoid()
                        y[..., 0:2] = (y[..., 0:2] + self.grid[i]) * self.stride[i]  # xy
                        y[..., 2:4] = torch.exp(y[..., 2:4]) * self.stride[i]  # wh
                    else:
                        xy, wh, conf = y.split((2, 2, self.num_classes + 1), 4)  # y.tensor_split((2, 4, 5), 4)# torch 1.8.0
                        conf = conf.sigmoid()

                        xy = (xy + self.grid[i]) * self.stride[i]  # new xy
                        wh = torch.exp(wh) * self.stride[i]  # new wh

                        if self.divide_x:
                            xy = xy / torch.tensor([self.divide_x, self.divide_y])
                            wh = wh / torch.tensor([self.divide_w, self.divide_h])
                            wh.clamp_(min=0.0, max=1.0)

                        y = torch.cat((xy, wh, conf), 4)

                    z.append(y.view(bs, -1, self.num_classes + 5))

        return x if (self.training or self.no_decode_layer) else torch.cat(z, 1)

    def fuse(self):
        # print("YOLOXDetect.fuse")
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


class Model(nn.Module):
    def __init__(self, cfg='yolor-csp-c.yaml', ch=3, nc=None, anchors=None, divide_x=1, divide_y=1, divide_h=1, divide_w=1,
                 no_decode_layer=False, is_file=True):  # models, input channels, number of classes
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

        # Build strides for Boxify / YOLOX head
        m = self.model[-1]
        if isinstance(m, (YOLOXDetect, Identity)):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.divide_x = divide_x
            m.divide_y = divide_y
            m.divide_h = divide_h
            m.divide_w = divide_w
            m.no_decode_layer = no_decode_layer

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

            if self.traced:
                if isinstance(m, YOLOXDetect):
                    break

            if profile:
                c = isinstance(m, YOLOXDetect)
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






    def reparameterize(self):

        return self.fuse()

    def fuse(self):  # fuse models Conv2d() + BatchNorm2d() layers
        print('Reparameterizing models...')
        # print('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
            elif isinstance(m, YOLOXDetect):
                m.fuse()
        self.info()
        return self


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
        if m in [nn.Conv2d, Conv, DWConv, RepConv, SPPCSPC]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m is SPPCSPC:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Shortcut:
            c2 = ch[f[0]]
        elif m is MP or m is SP:
            c2 = ch[f]
        elif m in [YOLOXDetect, Identity]:
            args.append([ch[x] for x in f])
            if not m == Identity:
                if isinstance(args[1], int):  # number of anchors
                    args[1] = [list(range(args[1] * 2))] * len(f)
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
