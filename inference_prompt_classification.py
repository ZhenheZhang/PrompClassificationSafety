#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
@File   :   inference_prompt_classification.py
@Author :   Zhenhe Zhang
@Date   :   2023/10/24
@Notes  :   Inference
"""

import os
import json
import torch
import time
from peft import PeftModel
from transformers import AutoTokenizer, AutoModel


def read_from_json(file: str):
    with open(file, 'r', encoding='utf-8') as f:
        return [json.loads(l.strip('\n')) for l in f.readlines()]


def write2json(result, file):
    with open(file, 'w', encoding='utf-8') as f:
        for res in result:
            f.write(json.dumps(res, ensure_ascii=False)+'\n')


def inference(data_dir, file, model_dir):
    tokenizer = AutoTokenizer.from_pretrained('./chatGLM', trust_remote_code=True)
    model = AutoModel.from_pretrained("./chatGLM", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, model_dir)

    model = model.half().cuda()
    model = model.eval()

    infos = read_from_json(os.path.join(data_dir, file))
    out_file = '{}_infer_ans.json'.format(file[:-5])
    system_prompt = "请判断问题是否涉及相关安全风控领域，直接回答是或者否，如果回答是的话，请再给出具体类别。用户的问题是：@"

    answers = []
    t1 = time.time()
    with torch.no_grad():
        for idx, item in enumerate(infos):
            if isinstance(item['prompt'], list):
                prompt = item['prompt'][0]
            else:
                prompt = item['prompt']
            input_text = system_prompt.replace('@', prompt)

            ids = tokenizer.encode(input_text)
            input_ids = torch.LongTensor([ids]).cuda()
            out = model.generate(
                input_ids=input_ids,
                max_length=1024,
                do_sample=False,
                temperature=0
            )
            out_text = tokenizer.decode(out[0])
            item['out_text'] = out_text
            answer = out_text[len(input_text):].lstrip(' ')
            answer = answer.replace(',', '，')
            item["infer_answer"] = answer
            answers.append({'index': idx, **item})
            print(idx, prompt, answer)

            if idx % 100 == 0:
                t2 = time.time()
                print("time", (t2 - t1) / (idx + 1))
                write2json(answers, os.path.join(data_dir, out_file))

    write2json(answers, os.path.join(data_dir, out_file))


def eval(data_dir: str, file: str):
    infos = read_from_json(os.path.join(data_dir, file))
    correct = 0
    for info in infos:
        if info['infer_answer'] == info['content']:
            correct += 1
    print("ACC ALL:", correct / len(infos))


if __name__ == "__main__":
    pass
