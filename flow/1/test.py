# import os
# class_list = []
# class_path = './config/class.txt'
# assert os.path.exists(class_path), "class.txt does not exist"
# with open(class_path, "rb") as f:
#     lines = f.readlines()
#     for line in lines:
#         line = line.strip()
#         class_list.append(line)
# print(class_list)
import json
import triton_python_backend_utils as pb_utils
from model import TritonPythonModel
import numpy as np
args = {}
model = TritonPythonModel()
model.initialize(args)
print(model.num_classes)

text = '我想买一辆自行车'
input_ids, attention_mask = model.preprocess(text)
print(input_ids, attention_mask)
output = np.ones((1, 1297))

process_result_dic = model.postprocess(output)
process_result_json = json.dumps(
    process_result_dic, ensure_ascii=False)
print(process_result_json)


# model_name = "bert_classifier"
# output_name = "output"
# infer_request = pb_utils.InferenceRequest(
#     model_name=model_name,
#     requested_output_names=[output_name],
#     inputs=[input_ids, attention_mask])
# infer_response = infer_request.exec()
# response_tensor = pb_utils.get_output_tensor_by_name(
#     infer_response, output_name)
# pytorch_tensor = from_dlpack(response_tensor.to_dlpack())
