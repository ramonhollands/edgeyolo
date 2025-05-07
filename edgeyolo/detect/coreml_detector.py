from ..data.data_augment import preproc
from ..utils import postprocess
import numpy as np
from loguru import logger
import os
from time import time
import coremltools as ct

class CoreMlDetector:

    strides = [8, 16, 32]

    def __init__(self, weight_file, conf_thres, nms_thres, *args, **kwargs):
        
        logger.info(f"loading weights from {weight_file}")
        self.mlmodel = ct.models.MLModel('lego.mlpackage')

        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        
        self.use_decoder = kwargs.get("use_decoder") or False
        
        self.class_names = ['Lego']
        self.input_size = [160, 160]
        if isinstance(self.input_size, int):
            self.input_size = [self.input_size] * 2
        
        logger.info("Coreml model loaded")
        

    # def __preprocess(self, imgs):
    #     pad_ims = []
    #     rs = []
    #     for img in imgs:
    #         pad_im, r = preproc(img, self.input_size)
    #         pad_ims.append(torch.from_numpy(pad_im).unsqueeze(0))
    #         rs.append(r)
    #     assert len(pad_ims) == self.batch_size, "batch size not match!"
    #     self.t0 = time()
    #     ret_ims = pad_ims[0] if len(pad_ims) == 1 else torch.cat(pad_ims)
    #     return ret_ims.float(), rs

    def __postprocess(self, results, rs=None):
        # print(results, results.shape)
        import torch
        results_as_tensor = torch.tensor(results)

        outs = postprocess(results_as_tensor, len(self.class_names), self.conf_thres, self.nms_thres, True)

        if rs is not None:
            for i, r in enumerate(rs):
                if outs[i] is not None:
                    outs[i][..., :4] /= r
                    outs[i] = outs[i].cpu()
        return outs

    def __call__(self, imgs, legacy=False):
        self.t0 = time()
        from PIL import Image
        pil_img = Image.fromarray(imgs[0])
        pil_img = pil_img.convert('RGB')
        pil_img = pil_img.resize((160, 160))
        
        net_outputs = self.mlmodel.predict({"image": pil_img})['output']

        outputs = self.__postprocess(net_outputs)
        self.dt = time() - self.t0
        
        return outputs
