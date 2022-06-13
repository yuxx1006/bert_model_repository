# Bert_classification_model
# Desc：bert model for product classification in nlp
#      the service is deployed on Triton server
## 1. 路径说明
```html
bert_model_repository --所有文件都放在此文件夹下
    flow --bert python backend前后处理流
        1 --模型版本
            model.py -- 模型调用文件 
            pytorch_pretrained -- 调用berttokenizer
            triton_python_backend_utils.py -- pu_utils
        config.pbtxt -- 模型配置
    bert_classifier
        1 -- 模型版本
            model.onnx --onnx模型文件
        config.pbtxt -- 分类模型配置
    
注：所有txt file放在/opt/tritonsever下面，可直接读
    - catid3_and_catname3 （id2name对应关系）
    - class.txt （1927 商品类别）
    - vocab.txt （berttokenizer）
```
