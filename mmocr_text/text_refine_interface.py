from mmocr.apis import TextDetInferencer
import cv2
import numpy as np
import torch
from PIL import Image
from loguru import logger
from utils import edge_blur, style_transfer, get_infer_map

class TextRefineInterface:
    def __init__(self, ckpt_path: str='mmocr/ckpts/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015_20221101_124139-4ecb39ac.pth'):
        self.device = torch.device('cuda:0')
        self.inferencer = TextDetInferencer(model='DBNetPP', weights=ckpt_path)

    @staticmethod
    def post_proc_image(bgr_image):
        return Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))

    @staticmethod
    def make_mask(size, poly_list, min_area=50):
        """
        根据文本检测模型输出结果创建mask
        :param size: mask尺寸，和原图保持一致
        :param poly_list: 文本区域描述
        :return: mask
        """
        mask = np.zeros(size, dtype=np.uint8)
        for poly in poly_list:
            poly = np.array(poly).round().astype(np.int32).reshape(-1, 2)
            area = cv2.contourArea(poly, True)
            if area>min_area: # 过滤太小的文字
                cv2.fillPoly(mask, [poly], 255, 8, 0)
        return mask

    def aline_text_color(self, image: np.ndarray, image_ref: np.ndarray, mask: np.ndarray):
        """
        和sd生成的图对齐文本颜色
        :param image:
        :param image_ref:
        :param mask:
        :return:
        """
        # Lab颜色空间中，ab通道混合
        img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        ref_lab = cv2.cvtColor(image_ref, cv2.COLOR_BGR2LAB)
        ref_lab[mask, 1:] = img_lab[mask, 1:]
        image_ref = cv2.cvtColor(ref_lab, cv2.COLOR_LAB2BGR)

        # 风格对齐
        infer_map = get_infer_map(image)
        image_ref = style_transfer(image_ref, infer_map)
        return image_ref

    @torch.no_grad()
    def infer(self, image: Image.Image, image_ref: Image.Image) -> Image.Image:
        """
        根据文本清晰的参考图修复文本
        :param image: sd生成后文本畸变的图像
        :param image_ref: 具有清晰文本的参考图像(mixpipe出来的图)
        :return: 修复后的图像
        """
        image = image.convert('RGB')
        image_ref = image_ref.convert('RGB')

        w, h = image.size
        # 强制参考图像和被替换图像相同尺寸
        image_ref = image_ref.resize((w, h), Image.ANTIALIAS)

        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        image_ref = cv2.cvtColor(np.asarray(image_ref), cv2.COLOR_RGB2BGR)

        # 检测文本区域
        res = self.inferencer(image_ref)
        # 创建mask并适当膨胀
        mask = self.make_mask((h, w), res['predictions'][0]['polygons'])
        kernel = np.ones((7, 7), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, 3)
        # mask_b = mask>10
        cv2.imwrite('mask.png', mask)

        image_ref = self.aline_text_color(image, image_ref, mask>127)

        # cv2.imshow('mask', mask)
        # cv2.waitKey()
        # 边缘虚化
        image_ref_a = np.concatenate((image_ref, mask[:,:,None]), axis=2)
        image_ref_a = edge_blur(image_ref_a)
        mask = (image_ref_a[:,:,3]/255)[:,:,None]

        image = (image*(1-mask) + image_ref*mask).astype(np.uint8)

        pred = self.post_proc_image(image)
        return pred

if __name__ == '__main__':
    td = TextRefineInterface('ckpts/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015_20221101_124139-4ecb39ac.pth')
    img = Image.open('../imgs/text1.png')
    img_ref = Image.open('../imgs/text_ref.png')
    pred = td.infer(img, img_ref)
    pred.save('../imgs/td1.png')