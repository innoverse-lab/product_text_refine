from mmocr_text.text_refine_interface import TextRefineInterface
from PIL import Image

if __name__ == '__main__':
    td = TextRefineInterface('mmocr_text/ckpts/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015_20221101_124139-4ecb39ac.pth')
    name = '1690952943765_1'

    img = Image.open(f'./imgs/{name}.png')
    img_ref = Image.open(f'./imgs/{name}_ref.png')
    pred = td.infer(img, img_ref)
    pred.save(f'./imgs/{name}_refine1.png')