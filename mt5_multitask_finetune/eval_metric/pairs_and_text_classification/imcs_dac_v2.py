# encoding:utf-8
from multiprocessing import Manager
from tqdm import tqdm
import numpy as np
import os
from functools import partial
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import requests
import json
from typing import Tuple
export_model_batch_size = 1


def test_data_prepare(path: str, type_1="imcs_dac_v2") -> Tuple[list, list]:
    "generate the test dataset from path in ncbi_blue dataset"
    try:
        with open(path, "r") as f:
            content = json.load(f)
    except:
        raise Exception("the file need to be #json# file type!")
    # 每个template 单独 计算 inference metric
    print("{} test dataset length is {}".format(path, len(content)))
    input_list = []
    label_list = []
    for i in content:
        tmp = i
        if len(tmp) <= 1:
            print(tmp)
            raise Exception("input sentence spliting exception!")
        # require len(tmp)>=2
        input = tmp[0]
        if type_1 == "ner":
            label = tmp[-1].split(",")
        elif type_1 == "sentence_pairs":
            label = float(tmp[-1])
        elif type_1 == "imcs_dac_v2":
            label = tmp[-1]
        input_list.append(input)
        # ,
        label_list.append(label)
    return input_list, label_list


def batch_data(inputs_data, batch_size=export_model_batch_size):
    "generator for batch test data"
    for start_idx in range(0, len(inputs_data), batch_size):
        excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs_data[excerpt]


class SelfEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self, obj)


def ner_one_template(input_batch: list, model_name="dac_v2_3b", port="8501"):
    '''input:batch_list []
        output:predicitons
    '''
    input = np.array(input_batch)
    input_data = {
        "signature_name": "",
        "instances": input}
    data = json.dumps(input_data, cls=SelfEncoder, indent=2)
    root_url = f"http://10.100.45.205:{port}"
    url = f"{root_url}/v1/models/{model_name}:predict"
    result = requests.post(url, data=data)
    tmp = eval(result.content)
    return_list = [i["outputs"] for i in tmp["predictions"]]
    return return_list


manager = Manager()
all_templates_prediction = manager.dict()


def gen_predict(all_param, basepath):
    pool = ThreadPool(20)
    tem_name, port = all_param
    input, _ = test_data_prepare(os.path.join(basepath, tem_name))
    all_input = list(batch_data(input))[:1]
    ner_one_template_1 = partial(ner_one_template, port=port)
    tmp_prediction = list(tqdm(
        pool.imap(ner_one_template_1, all_input), total=len(all_input), desc=tem_name))
    pool.close()
    pool.join()
    with open(os.path.join(basepath, f"result/val_imcs_dac_v2_{len(tmp_prediction)}_"+tem_name), "w") as writer:
        json.dump(tmp_prediction, writer)

    all_templates_prediction[tem_name] = tmp_prediction
    print("template {} has been finished".format(tem_name))


def ner_dataset_test(path: str):
    '''param: already prepared dataset_dir-->path
       func: collect and return all templates prediction result and label
    '''
    import os

    template_file_pth_list = []
    for _, _, files, in os.walk(path):
        for template_name in files:
            if template_name.endswith("json") and "val" not in template_name:
                template_file_pth_list.append(template_name)

    _, label = test_data_prepare(os.path.join(path, template_file_pth_list[0]))

    # collect all the templates applied test dataset's prediction of this ner dataset
    c_num = list(zip(template_file_pth_list, ["8501", "8503", "8505", "8507"]))
    print(c_num)
    gpus_pool = Pool(4)
    gen_predict1 = partial(gen_predict, basepath=path)
    _ = gpus_pool.map(gen_predict1, c_num)
    gpus_pool.close()
    gpus_pool.join()
    # examine the prediction
    assert len(template_file_pth_list) == len(all_templates_prediction.keys())
    # 返回一个label
    return all_templates_prediction, label


def metric_one_prompt(prediction, label):
    # peasrso 系数, 判断是否线性相关??
    # pair task for biosses
    right_n = 0
    prediction_len = 0
    label_len = 0
    assert len(label[:1]) == len(prediction)
    prediction = [i for i in prediction]
    label = [j for j in label[:1]]
    for i, j in zip(prediction, label):
        if i[0].strip() == j.strip():
            right_n += 1
    acc = float(right_n)/len(label)
    return {"acc": acc, "length_prediction": len(prediction)}


def merge(predictions, label):
    all_prediction = []
    print(predictions)
    for idx in range(len(label[:1])):
        tmp = []
        for prompt in predictions.keys():
            tmp.extend(predictions[prompt][idx])
        all_prediction.append(list(set(tmp)))
    return all_prediction


def vote(predictions, label, threhold=2):
    from collections import Counter
    result_prediction = []
    for idx in range(len(label[:1])):
        tmp = []
        for prompt in predictions.keys():
            tmp.extend(predictions[prompt][idx])
        tmp_c = Counter(tmp)
        if idx == 0:
            print(tmp_c)
        # vote strategy >half prompts num
        f_tmp = [i[0] for i in tmp_c.items() if i[1] >= threhold]
        if idx == 0:
            print(f_tmp)
        result_prediction.append(f_tmp)
    return result_prediction


THREHOLD = 2  # the threshold num of votes of templates


def predict_result(predictions, label):
    predict_result = {}
    for i in predictions.keys():
        predict_result[i] = metric_one_prompt(predictions[i], label)
    all_predict = merge(predictions, label)
    vote_predict = vote(predictions, label, threhold=2)
    predict_result["add_all_templates"] = metric_one_prompt(all_predict, label)
    predict_result["vote_by_all_templates"] = metric_one_prompt(
        vote_predict, label)
    return predict_result


test_dir = "/raid/yiptmp/nlp_prepare_dataset/med0_dataset/test_cblue_v2/{}"
test_dataset_path = {}
with open("ner_dataset_names", "r") as f:
    for i in f:
        test_dataset_path[i.strip()] = test_dir.format(i.strip())


model_name = "mt5_3b_imcs_dac_v2"
result_dir = "/raid/zyftest/project/Med0/t5_multitasks_finetune/eval_metric/test_result"
for dataset_name, path in test_dataset_path.items():
    predictions, label_result = ner_dataset_test(path)
    tmp = predict_result(predictions, label_result)
    save_pth = os.path.join(result_dir, dataset_name+"_"+model_name)
    print(tmp)
    with open(save_pth, "w") as f:
        json.dump(tmp, f, indent=2, ensure_ascii=False)
