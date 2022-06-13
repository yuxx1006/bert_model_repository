# *_*coding:utf-8 *_*
"""
@author: Xiaoxiao Yu
@time: 2022/6/9 11:11 AM
@desc: bert model for product classification in nlp
"""
import os
import json
import time
import heapq

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.dlpack import from_dlpack
from pytorch_pretrained import BertTokenizer
import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        self.class_list_path = 'class.txt'
        assert os.path.exists(self.class_list_path), "class.txt does not exist"
        self.class_list = [x.strip() for x in open(
            self.class_list_path).readlines()]    # 类别名单
        # self.save_path = dataset + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 128                                           # mini-batch大小
        # 每句话处理成的长度(短填长切)
        self.pad_size = 15
        self.learning_rate = 5e-5                                       # 学习率
        # self.bert_path = './bert_model'
        self.vocab_path = 'vocab.txt'
        assert os.path.exists(self.vocab_path), "vocab.txt does not exist"
        self.tokenizer = BertTokenizer.from_pretrained(self.vocab_path)
        self.hidden_size = 768

    def execute(self, requests):
        responses = []
        first_time = time.time()
        preprocess_cost = 0
        infer_cost = 0
        postprocess_cost = 0
        for request in requests:
            #query0 = pb_utils.get_input_tensor_by_name(
            #    request, "INPUT").as_numpy().tolist()
            #print(query0)
            query = [t.decode('UTF-8') for t in pb_utils.get_input_tensor_by_name(request, "INPUT").as_numpy().tolist()]
            # preprocess
            input_ids, attention_mask = self.preprocess(query[0])
            input0 = pb_utils.Tensor('input_ids', input_ids)
            input1 = pb_utils.Tensor('attention_mask', attention_mask)

            preprocess_time = time.time()
            preprocess_cost += int((preprocess_time-first_time)*1000)

            # infer
            model_name = "bert_classifier"
            output_name = "output"
            infer_request = pb_utils.InferenceRequest(
                model_name=model_name,
                requested_output_names=[output_name],
                inputs=[input0, input1])
            infer_response = infer_request.exec()
            response_tensor = pb_utils.get_output_tensor_by_name(
                infer_response, output_name)
            pytorch_tensor = from_dlpack(response_tensor.to_dlpack())
            #print(pytorch_tensor)
            infer_time = time.time()
            infer_cost += int((infer_time-preprocess_time)*1000)

            # postprocess
            if isinstance(pytorch_tensor, torch.Tensor):
                pytorch_tensor = pytorch_tensor.numpy()
            process_result_dic = self.postprocess(pytorch_tensor)

            process_result_json = json.dumps(
                process_result_dic, ensure_ascii=False)

            out_tensor_0 = pb_utils.Tensor("OUTPUT", np.array(
                process_result_json, dtype=np.object_))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            responses.append(inference_response)

            last_time = time.time()
            postprocess_cost += int((last_time-infer_time)*1000)

            print(f"total cost: {int((last_time-first_time)*1000)}, "
                  f"preprocess cost: {preprocess_cost}, "
                  f"infer cost: {infer_cost}, "
                  f"postprocess cost: {postprocess_cost}, ")
            
            return responses

    # helper: tokenizer
    def fine_grade_tokenize(self, text):
        CLS = '[CLS]'
        PAD_SIZE = self.pad_size
        token = self.tokenizer.tokenize(text)
        token = [CLS] + token
        tokens = self.tokenizer.convert_tokens_to_ids(token)
        seq_len = len(token)

        if seq_len < PAD_SIZE:
            mask = [1] * seq_len + [0] * (PAD_SIZE - seq_len)
            tokens += ([0] * (PAD_SIZE - seq_len))
        else:
            mask = [1] * PAD_SIZE
            tokens = tokens[:PAD_SIZE]
        return (tokens, mask)

    # preprocess for text input
    def preprocess(self, text):
        tokens, mask = self.fine_grade_tokenize(text)
        input_feed = {'input_ids': [tokens],
                      'attention_mask': [mask]}
        input_ids = np.array(input_feed['input_ids'], dtype=np.int64)
        attention_mask = np.array(input_feed['attention_mask'], dtype=np.int64)

        return (input_ids, attention_mask)

    # helper read file
    def id2name(self):
        with open("catid3_and_catname3", encoding="utf-8") as f:
            cat = {}
            for line in f.readlines():
                cat[line.split()[0]] = line.split()[1]
            return cat

    # postprocess
    def postprocess(self, output):
        res = {}
        label = F.softmax(torch.LongTensor(output)/2, 1).cpu()
        tmp = zip(range(len(label.data[0])), label.data[0].numpy())
        max_five = heapq.nlargest(5, tmp, key=lambda x: x[1])
        cat = self.id2name()
        for i in max_five:
            res[cat[self.class_list[i[0]]]] = "{:.4f}".format(i[1])
        return res
