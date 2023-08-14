#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>

import os

from saicinpainting.evaluation.refinement import refine_predict
from saicinpainting.evaluation.utils import move_to_device

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from hydra import compose, initialize
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
from PIL import Image

from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_img_to_modulo
from loguru import logger

class LamaInterface:
    def __init__(self, ckpt_path: str, train_config_path: str):
        self.device = torch.device('cuda:0')

        initialize(config_path="./configs/prediction", job_name="lama_app")
        self.predict_config = compose(config_name="default.yaml")
        self.pad_out_to_modulo = 8
        self.predict_config.refine = False

        self.build_model(ckpt_path, train_config_path)

    def build_model(self, ckpt_path: str, train_config_path: str):
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        model = load_checkpoint(train_config, ckpt_path, strict=False, map_location='cpu')
        model.freeze()
        if not self.predict_config.get('refine', False):
            model.to(self.device)
        self.model = model

        logger.info('build Lama finish')

    def post_proc_image(self, result):
        result = np.clip(result*255, 0, 255).astype('uint8')
        result = Image.fromarray(result)
        return result

    def convert_image(self, image: Image.Image, mode='RGB', return_orig=False):
        img = np.array(image.convert(mode))
        if img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))
        out_img = img.astype('float32')/255
        if return_orig:
            return out_img, img
        else:
            return out_img

    def pre_proc_image(self, image: Image.Image, mask: Image.Image):
        image = self.convert_image(image, mode='RGB')
        mask = self.convert_image(mask, mode='L')
        result = dict(image=image, mask=mask[None, ...])

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo>1:
            result['unpad_to_size'] = result['image'].shape[1:]
            result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
            result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)
        return result

    @torch.no_grad()
    def infer(self, image: Image.Image, mask: Image.Image):
        result = self.pre_proc_image(image, mask)
        batch = default_collate([result])

        # cur_res = refine_predict(batch, self.model, **self.predict_config.refiner)
        # cur_res = cur_res[0].permute(1, 2, 0).detach().cpu().numpy()
        batch = move_to_device(batch, self.device)
        batch['mask'] = (batch['mask']>0)*1
        batch = self.model(batch)
        cur_res = batch[self.predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
        unpad_to_size = batch.get('unpad_to_size', None)
        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]

        result = self.post_proc_image(cur_res)
        return result

    def infer_refine(self, image: Image.Image, mask: Image.Image):
        result = self.pre_proc_image(image, mask)
        batch = default_collate([result])

        cur_res = refine_predict(batch, self.model, **self.predict_config.refiner)
        cur_res = cur_res[0].permute(1, 2, 0).detach().cpu().numpy()

        result = self.post_proc_image(cur_res)
        return result

if __name__ == '__main__':
    lama = LamaInterface('big_lama/models/best.ckpt', 'big_lama/config.yaml')
    img = Image.open('../DIS/imgs/imgs/1.png')
    mask = Image.open('../DIS/imgs/mask/1_2.png')
    pred = lama.infer(img, mask)
    pred.save('1.png')
