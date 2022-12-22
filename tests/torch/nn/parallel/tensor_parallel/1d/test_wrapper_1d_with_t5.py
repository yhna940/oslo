import argparse
import time

import torch
import torch.distributed as dist
import wandb
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BartConfig, BartForConditionalGeneration

import oslo
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.parallel.tensor_parallel import TensorParallel

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--memory_priority", action="store_true", default=False)
args = parser.parse_args()

tp_size = 2
batch_size = 16
seq_length = 128
model_name = "facebook/bart-base"

# parallel context 생성
parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=tp_size,
    tensor_parallel_mode=ParallelMode.TENSOR_1D,
)

torch.set_printoptions(sci_mode=False)
torch.manual_seed(0)

# 토크나이저 생성
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 모델 생성 및 병렬화 수행
model_no_tp = BartForConditionalGeneration(
    BartConfig.from_pretrained(model_name)
).cuda()
model_tp = BartForConditionalGeneration(BartConfig.from_pretrained(model_name))
wrapper_tp = TensorParallel(
    model_tp, parallel_context, memory_priority=args.memory_priority
)

oslo.ready(model_tp, parallel_context)

if dist.get_rank() == 0:
    print(wrapper_tp)

# 옵티마이저 생성
optimizer_tp = Adam(wrapper_tp.parameters(), lr=3e-5)
optimizer_no_tp = Adam(model_no_tp.parameters(), lr=3e-5)

# 데이터셋 생성
datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(sample) for sample in datasets[:500]]
dataloader = DataLoader(datasets, batch_size=batch_size)

# 모니터링 생성
if dist.get_rank() == 0:
    wandb.init(project="oslo", name=f"{model_name}_nprocs{tp_size}_tp1d_bs{batch_size}")
    cur = time.time()

# 모니터링 생성 대기
dist.barrier()

# 학습 시작
for data in dataloader:
    optimizer_tp.zero_grad()
    optimizer_no_tp.zero_grad()

    inputs = tokenizer(
        data,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_length,
    ).to("cuda")

    fw_start = time.time()
    loss_no_tp = model_no_tp(**inputs, labels=inputs["input_ids"]).loss
    fw_time = time.time() - fw_start

    fw_start_tp = time.time()
    loss_tp = wrapper_tp(**inputs, labels=inputs["input_ids"]).loss
    fw_time_tp = time.time() - fw_start_tp

    bw_start = time.time()
    loss_no_tp.backward()
    optimizer_no_tp.step()
    bw_time = time.time() - bw_start

    bw_start_tp = time.time()
    loss_tp.backward()
    optimizer_tp.step()
    bw_time_tp = time.time() - bw_start_tp

    if dist.get_rank() == 0:
        print(f"[tp/notp loss]: {loss_tp:.4f}, {loss_no_tp:.4f}")
        wandb.log(
            {
                "tp_loss": loss_tp,
                "notp_loss": loss_no_tp,
                "tp_fw_time": fw_time_tp,
                "notp_fw_time": fw_time,
                "tp_bw_time": bw_time_tp,
                "notp_bw_time": bw_time,
            }
        )

dist.barrier()
