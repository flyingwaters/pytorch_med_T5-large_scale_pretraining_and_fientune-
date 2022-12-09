import time

import psutil
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_info


class CUDACallback(Callback):

    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_batch_end(self, trainer, pl_module) -> None:
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20

        virt_mem = psutil.virtual_memory()
        virt_mem = round((virt_mem.used / (1024 ** 3)), 2)
        pl_module.log('Peak CUDA Memory (GiB)', max_memory / 1000, prog_bar=True, on_step=True, sync_dist=True)
        pl_module.log(f"Average Virtual memory (GiB)", virt_mem, prog_bar=True, on_step=True, sync_dist=True)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time
        virt_mem = psutil.virtual_memory()
        virt_mem = round((virt_mem.used / (1024 ** 3)), 2)
        swap = psutil.swap_memory()
        swap = round((swap.used / (1024 ** 3)), 2)

        max_memory = trainer.training_type_plugin.reduce(max_memory)
        epoch_time = trainer.training_type_plugin.reduce(epoch_time)
        virt_mem = trainer.training_type_plugin.reduce(virt_mem)
        swap = trainer.training_type_plugin.reduce(swap)

        rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
        rank_zero_info(f"Average Peak CUDA memory {max_memory:.2f} MiB")
        rank_zero_info(f"Average Peak Virtual memory {virt_mem:.2f} GiB")
        rank_zero_info(f"Average Peak Swap memory {swap:.2f} Gib")
