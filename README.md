# CPM

## Setup

1. Install Python dependencies.
```bash
pip install -r requirements.txt
```

2. Install Apex.
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

3. Preprocess the dataset and save it to `data`.

## Finetune CPM2

On each host, run the following commands in the `cpm` directory.
```bash
export NCCL_SOCKET_IFNAME=NETWORK_INTERFACE_TO_USE
# set $t to the total number of gpus, $x to the total number of hosts, $y to the rank of each host (e.g., four hosts with node rank 0, 1, 2, 3), $z to the number of gpus per host and $addr to the master address
python -m launch --nnodes $x --node_rank $y --nproc_per_node $z main_with_runtime.py --data_dir data --master_addr $addr --module cpm2_4 --checkpoint_dir output --partition cpm2_4/gpipe.json --sync_mode asp --distributed_backend nccl -b 12 --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 100 --verbose 0 --num_ranks_in_server $z --config_path cpm2_4/mp_conf.json
```

## Run with 32 GPUs

For 4 hosts each holding 8 GPUs, we partition the CPM2 model into 4 stages, and each stage is placed on one host. Inside a host, a stage is split to 8 GPUs using model parallelism.



```bash
# Host 0
python3 -m launch --nnodes 4 --node_rank 0 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr $addr  --module cpm2_4 --checkpoint_dir output --partition cpm2_4/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path cpm2_4/mp_conf.json -b 8

# Host 1
python3 -m launch --nnodes 4 --node_rank 1 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr $addr  --module cpm2_4 --checkpoint_dir output --partition cpm2_4/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path cpm2_4/mp_conf.json -b 8

# Host 2
python3 -m launch --nnodes 4 --node_rank 2 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr $addr  --module cpm2_4 --checkpoint_dir output --partition cpm2_4/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path cpm2_4/mp_conf.json -b 8

# Host 3
python3 -m launch --nnodes 4 --node_rank 3 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr $addr  --module cpm2_4 --checkpoint_dir output --partition cpm2_4/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path cpm2_4/mp_conf.json -b 8
```