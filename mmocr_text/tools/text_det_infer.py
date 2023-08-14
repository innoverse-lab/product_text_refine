from mmocr.apis import TextDetInferencer
# Load models into memory
inferencer = TextDetInferencer(model='DBNetPP', weights='ckpts/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015_20221101_124139-4ecb39ac.pth')
# Inference
res = inferencer('demo/2.png', show=True)
print(res)