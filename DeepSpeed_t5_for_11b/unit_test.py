from codes import PyTorchDataModule
from transformers import T5Tokenizer
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
tokenizer = T5Tokenizer.from_pretrained("t5-3b")
test_path =  "/raid/yiptmp/nlp_prepare_dataset/med0_dataset/dataset_tsv/all_test.tsv"
import pandas as pd
test_df = pd.read_csv(test_path, sep="\t",names=["source_text", "target_text"], low_memory=False)


gdata = PyTorchDataModule(test_df, tokenizer, 512,512)
# data_loader = DataLoader(gdata,batch_size =8,sampler= WeightedRandomSampler([0.1,0.1,0.5,0.5],num_samples=20, replacement=True))
# for i in data_loader:
#     print(i)
#     break
len(gdata)
print(gdata[1])