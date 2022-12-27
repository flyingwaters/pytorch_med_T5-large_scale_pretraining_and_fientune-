import bminf
import torch
from transformers import MT5ForConditionalGeneration, MT5Config

config = MT5Config.from_pretrained("google/mt5-large")
model = MT5ForConditionalGeneration(config)


model.load_state_dict(torch.load(
    "/raid/zyftest/project/med_T5/bmtrain_pretrain/mt5_xxl_ckpt-1-810.pt"))

with torch.cuda.device(0):
    bminf.wrapper(model)
