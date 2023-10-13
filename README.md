# 文本识别+替换

现在有两种图像，图像A是一个文本细节崩坏的图像，文本难以辨识。提供一个文本完好的参考图像A_ref，移除掉A中文本崩坏的区域，并将A_ref中文本完好的区域粘贴到A的对应位置。

需要实现的功能如下:
1. 通过DBNet++检测文本区域 (提供封装接口)
2. 通过 [SAM模型](https://github.com/facebookresearch/segment-anything) 获取更为精准的文本分割结果
3. 使用 [Lama模型](https://github.com/advimman/lama) 填充A中崩坏的文本区域
4. 将完好的文本粘贴到对应区域


mmocr和lama依赖库的安装，参考各自对应的```Readme```说明文档。

# 提供的封装接口

## 文本检测

使用以下接口实现文本检测:
```python
from mmocr.apis import TextDetInferencer
inferencer = TextDetInferencer(model='DBNetPP', weights=ckpt_path)
poly_list = inferencer(image_ref)['predictions'][0]['polygons']
for poly in poly_list:
    #poly: [x1,y1,x2,y2,...] 文字区域包围框的4个顶点
```

## 图像填充(Inpaint)

使用以下接口实现Inpaint:
```python
from lama.lama_interface import LamaInterface
from PIL import Image

lama = LamaInterface('big_lama/models/best.ckpt', 'big_lama/config.yaml')
img = Image.open('imgs/1.png')
mask = Image.open('mask/1.png') # 黑白mask，填充白色区域
pred = lama.infer(img, mask)
pred.save('1.png')
```