# encoding:utf-8
from model_center.dataset import DistributedDataLoader
from model_center.utils import print_inspect
from model_center.dataset.t5dataset import DATASET
from dataset_test import truthful
from model_center.tokenizer import T5Tokenizer
from model_center.model import T5
from model_center import get_args
from sklearn.metrics import accuracy_score, f1_score
import torch
import bmtrain as bmt
from datasets import load_dataset


bmt.init_distributed(seed=0, zero_level=3)
model = T5.from_pretrained("mt5-large")
bmt.synchronize()

tokenizer = T5Tokenizer.from_pretrained("mt5-large")
script_pth = "/raid/yiptmp/nlp_prepare_dataset/med0_dataset/dataset_process/ye_jin_guo_process/truthfulqa.py"

train_dataset = truthful(path=script_pth, split="train", rank=bmt.rank(
), world_size=bmt.world_size(), tokenizer=tokenizer, max_encoder_length=512, max_decoder_length=128)
dev_dataset = truthful(path=script_pth, split="validation", rank=bmt.rank(
), world_size=bmt.world_size(), tokenizer=tokenizer, max_encoder_length=512, max_decoder_length=128)
bmt.synchronize()
# get the memory usage
bmt.print_rank("Model mem\n", torch.cuda.memory_summary())
batch_size = 16
# resample  wrapper this DistributedDataLoader
#
train_dataloader = DistributedDataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DistributedDataLoader(
    dev_dataset, batch_size=batch_size, shuffle=False)

optimizer = bmt.optim.AdamOffloadOptimizer(model.parameters(), lr=1e-3)
lr_scheduler = bmt.lr_scheduler.Noam(
    optimizer,
    start_lr=1e-5,
    warmup_iter=100,
    end_iter=-1)
loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)
# before backward scale the loss up to 1024*old_loss and after backward scale the grad down to grad/1024
# in this way, to avoid the underflow during the update of optimizer

optim_manager = bmt.optim.OptimManager(loss_scale=1024)
optim_manager.add_optimizer(optimizer, lr_scheduler)

for epoch in range(50000000):
    model.train()
    for step_n, data in enumerate(train_dataloader):
        enc_input = data["enc_input"]
        enc_length = data["enc_length"]
        dec_input = data["dec_input"]
        dec_length = data["dec_length"]
        targets = data["targets"]
        index = data["index"]

        # model forward
        # model in model_center has a unique input form
        # logits = model(enc_input, enc_length, dec_input, dec_length, output_logits=True).logits
        #
        output = model(enc_input, enc_length, dec_input,
                       dec_length, output_logits=True, return_dict=True)
        # calculate loss
        logits = output.logits
        # logits = logits.index_select(dim=-1, index=targets)
        # logits = logits[torch.where(index == 1)]
        # print(tuple(output.logits.shape))
        # print(tuple(targets.shape))
        loss = loss_func(
            logits.view(-1, logits.shape[-1]), targets.view(-1))
        # use bmt.sum_loss(loss) to gather all loss information from all distributed processes
        global_loss = bmt.sum_loss(loss).item()

        # like the function of pytorch why ? manager?
        # zero grad
        optim_manager.zero_grad()
        # scale loss before backward to avoid precision underflow of fp16
        optim_manager.backward(loss)

        # clip gradient norm
        grad_norm = optim_manager.clip_grad_norm(
            optimizer.param_groups, max_norm=10.0, norm_type=2)

        # step for all optimizer inside optim_manager
        optim_manager.step()

        # print information only on rank 0 when distributed training
        bmt.print_rank(
            "loss: {:.4f} | lr: {:.4e} | grad_norm: {:.4f} | steps: {}".format(
                global_loss,
                lr_scheduler.current_lr,
                grad_norm,
                step_n
            )
        )

    # evaluate model
    model.eval()
    with torch.no_grad():
        pd = []  # prediction
        gt = []  # ground_truth
        for it, data in enumerate(dev_dataloader):
            enc_input = data["enc_input"]
            enc_length = data["enc_length"]
            dec_input = data["dec_input"]
            dec_length = data["dec_length"]
            targets = data["targets"]
            index = data["index"]
            #
            output = model(enc_input, enc_length, dec_input,
                           dec_length, output_logits=True, return_dict=True)
            logits = output.logits
            loss = loss_func(
                logits.view(-1, logits.shape[-1]), targets.view(-1))

            logits = logits.argmax(dim=-1)

            pd.extend(logits.cpu().tolist())
            gt.extend(targets.cpu().tolist())

        # gather results from all distributed processes
        pd = bmt.gather_result(torch.tensor(pd).int()).cpu().tolist()
        gt = bmt.gather_result(torch.tensor(gt).int()).cpu().tolist()

        # calculate metric
        bmt.print_rank(pd[0][:2], gt[0][:2])
