from typing import Tuple
import json
def test_data_prepare(path:str, type="ner")->Tuple[list,list]:
    "generate the test dataset from path in ncbi_blue dataset"
    try:
        with open(path,"r") as f:
            content = json.load(f)
    except:
        raise Exception("the file need to be #json# file type!")
    # 每个template 单独 计算 inference metric
    print("{} test dataset length is {}".format(path, len(content)))
    input_list = []
    label_list = []
    for i in content:   
        tmp = i.split("\t")    
        if len(tmp)<=1:
            raise Exception("input sentence spliting exception!")
        # require len(tmp)>=2 
        input = " ".join(tmp[:-1])
        if type=="ner":
            label = tmp[-1].split(",")
        elif type=="sentence_pairs":
            label = float(tmp[-1])
        input_list.append(input)
        # ,
        label_list.append(label)
    return input_list, label_list

def batch_data(inputs_data,batch_size=32):
    "generator for batch test data"
    for start_idx in range(0, len(inputs_data), batch_size):
        excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs_data[excerpt]

import numpy as np
import json
import requests
class SelfEncoder(json.JSONEncoder):  
    def default(self, obj):  
        if isinstance(obj, np.ndarray):  
            return obj.tolist()  
        elif isinstance(obj, np.floating):  
            return float(obj)  
        elif isinstance(obj, bytes):  
            return str(obj, encoding='utf-8');  
        return json.JSONEncoder.default(self, obj)
def ner_one_template(input_batch:list):
    '''input:batch_list []
        output:predicitons
    '''
    input = np.array(input_batch)
    input_data = {  
    "signature_name": "",  
    "instances":input}
    data = json.dumps(input_data, cls=SelfEncoder, indent=2)
    root_url = "http://10.100.45.205:8501"
    url = "%s/v1/models/medicine_prompt_t5_small:predict" % root_url
    result = requests.post(url, data=data)
    tmp = eval(result.content)
    return_list = [i["outputs"].split(",") for i in tmp["predictions"]]
    return return_list 
import os
def ner_dataset_test(path:str):
    '''param: already prepared dataset_dir-->path
       func: collect and return all templates prediction result and label
    '''
    template_file_pth_list = []
    for _,_, files, in os.walk(path):
        for template_name in files:
            template_file_pth_list.append(template_name)
    all_templates_prediction= {}
    dataset_label = []

    for i in template_file_pth_list:
        tmp_prediction = []
        template_path = os.path.join(path, i)
        input, label = test_data_prepare(template_path)
        # get label
        if dataset_label==[]:
            dataset_label=label
    
        for input_batch in batch_data(input):
            tmp_prediction.extend(ner_one_template(input_batch))
        # collect all the templates applied test dataset's prediction of this ner dataset
        all_templates_prediction[i] = tmp_prediction
        print("template {} has been finished".format(i))
    # examine the prediction
    assert len(template_file_pth_list) == len(all_templates_prediction.keys())
    # 
    return all_templates_prediction, label

def metric_one_prompt(prediction, label):
    right_n = 0
    prediction_len = 0
    label_len = 0
    for predict_i,label_i in zip(prediction,label):
        for entity in predict_i:
            if entity in label_i:
                right_n+=1
        prediction_len+=len(predict_i)
        label_len += len(label_i)

    acc = float(right_n) / float(prediction_len)*100
    recall = float(right_n) / float(label_len)*100
    f1 = 2*(acc*recall)/(acc+recall)
    return {"f1":f1,"recall":recall, "acc":acc}