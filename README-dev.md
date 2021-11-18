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

## Finetune CPM

On each host, run the following commands in the `cpm` directory.
```bash
export GLOO_SOCKET_IFNAME=NETWORK_INTERFACE_TO_USE
# set $t to the total number of gpus, $x to the total number of hosts, $y to the rank of each host (e.g., four hosts with node rank 0, 1, 2, 3), $z to the number of gpus per host and $addr to the master address
python -m launch --nnodes $x --node_rank $y --nproc_per_node $z main_with_runtime.py --data_dir data --master_addr $addr --module medium_$t --checkpoint_dir output --partition medium_$t/vpipe.json --sync_mode asp --distributed_backend gloo -b 2 --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 100 --verbose 0 --num_ranks_in_server $z --config_path medium_$t/mp_conf.json
```

## Hybrid Parallelism 

Change the --config_path to medium_$t/dp_conf.json.

In dp_conf.json, map a stage to multiple GPU ranks, in order to apply data parallelism to per stage on multiple GPUs.

For example, in medium_4/dp_conf.json, we use 4 hosts with each holding 4 GPUs. We partition a CPM Medium model into 4 stages, and 
each stage is placed on one host. Inside a host, a stage is 
replicated on 4 GPUs with data parallelism. 

Note that, in dp_conf.json, the rank mapping is per GPU rank. And the node rank $y is per host rank. Thus, GPUs on node rank 0 is with GPU rank 0,1,2,3; GPUs on node rank 1 is with GPU rank 4,5,6,7. By configuring dp_conf.json, users can configure various hybrid topologies of DP+PP.

<!-- python3 -m launch --nnodes 1 --node_rank 0 --nproc_per_node 4 main_with_runtime.py --data_dir data --master_addr localhost --module medium_4 --checkpoint_dir output --partition medium_4/vpipe.json --sync_mode asp --distributed_backend gloo -b 2 --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 100 --verbose 0 --num_ranks_in_server 4 --config_path medium_4/mp_conf.json -->


python3 -m launch --nnodes 1 --node_rank 0 --nproc_per_node 4 main_with_runtime.py --data_dir data --master_addr localhost --module medium_dp --checkpoint_dir output --partition medium_dp/gpipe_dp.json --sync_mode asp --distributed_backend nccl -b 1 --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 100 --verbose 0 --num_ranks_in_server 4 --config_path medium_dp/dp_conf.json

python test.py --master_addr localhost --rank 0 --intra_server_broadcast --backend nccl --use_helper_threads --send

python test.py --master_addr localhost --rank 1 --intra_server_broadcast --backend nccl --use_helper_threads

## scp data
host 0
 scp -i "../shixiong.pem" -r data ubuntu@ec2-34-215-233-18.us-west-2.compute.amazonaws.com:/home/ubuntu

host 1
  scp -i "../shixiong.pem" -r data ubuntu@ec2-34-213-46-50.us-west-2.compute.amazonaws.com:/home/ubuntu


## scp cpm_baseline
host 0
 scp -i "cpm_github/shixiong.pem" -r cpm_baseline ubuntu@ec2-52-34-93-207.us-west-2.compute.amazonaws.com:/home/ubuntu

host 1
 scp -i "cpm_github/shixiong.pem" -r cpm_baseline ubuntu@ec2-34-213-46-50.us-west-2.compute.amazonaws.com:/home/ubuntu



## host 0
ssh -i "shixiong.pem" ubuntu@ec2-52-34-93-207.us-west-2.compute.amazonaws.com

## host 1
ssh -i "shixiong.pem" ubuntu@ec2-34-213-46-50.us-west-2.compute.amazonaws.com


## run docker 

nvidia-docker run -it -v $PWD:/workspace --net=host --ipc=host dmye/cpm:v0



export NCCL_SOCKET_IFNAME=ens5


##  4 GPU per host  1-1-2  batch size 3 
python3 -m launch --nnodes 1 --node_rank 0 --nproc_per_node 4 main_with_runtime.py --data_dir data --master_addr localhost --module medium_dp --checkpoint_dir output --partition medium_dp/gpipe_dp.json --sync_mode asp --distributed_backend nccl -b 3 --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 4 --config_path medium_dp/dp_conf.json


## 4 GPU per host  1-1-1-1 medium_4 
python3 -m launch --nnodes 1 --node_rank 0 --nproc_per_node 4 main_with_runtime.py --data_dir data --master_addr localhost --module medium_4 --checkpoint_dir output --partition medium_4/gpipe.json --sync_mode asp --distributed_backend nccl -b 6 --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 4 --config_path medium_4/mp_conf.json




###  8 GPU per host  DP 2 x PP 4 batch size 6  epoch 4.55hr see 8-2-4-8.log
python3 -m launch --nnodes 1 --node_rank 0 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr localhost --module medium_4 --checkpoint_dir output --partition medium_4/vpipe.json --sync_mode asp --distributed_backend nccl -b 6 --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path medium_4/8dp_conf.json

### 8 GPU per host pp8 bs 8  epoch 4.5 hr 
python3 -m launch --nnodes 1 --node_rank 0 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr localhost --module medium_8 --checkpoint_dir output --partition medium_8/vpipe.json --sync_mode asp --distributed_backend nccl -b 8 --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path medium_8/mp_conf.json


### 2 x 8 GPU  dp 4 pp 4   11.5/4 = 2.7


host 0

python3 -m launch --nnodes 2 --node_rank 0 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr localhost --module medium_4 --checkpoint_dir output --partition medium_4/vpipe.json --sync_mode asp --distributed_backend nccl -b 6 --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path medium_4/dp_conf.json


host 1

python3 -m launch --nnodes 2 --node_rank 1 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr 172.31.7.136  --module medium_4 --checkpoint_dir output --partition medium_4/vpipe.json --sync_mode asp --distributed_backend nccl -b 6 --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path medium_4/dp_conf.json

## 32 GPU 8dp 4 pp

host 0
python3 -m launch --nnodes 4 --node_rank 0 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr localhost --module medium_4 --checkpoint_dir output --partition medium_4/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path medium_4/32dp_conf.json -b 8

host 1
python3 -m launch --nnodes 4 --node_rank 1 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr 172.31.7.136 --module medium_4 --checkpoint_dir output --partition medium_4/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path medium_4/32dp_conf.json -b 8

host 2

python3 -m launch --nnodes 4 --node_rank 2 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr 172.31.7.136 --module medium_4 --checkpoint_dir output --partition medium_4/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path medium_4/32dp_conf.json -b 8

host 3

python3 -m launch --nnodes 4 --node_rank 3 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr 172.31.7.136 --module medium_4 --checkpoint_dir output --partition medium_4/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path medium_4/32dp_conf.json -b 8


## 32 dp 8 pp4 large 

host 0
python3 -m launch --nnodes 4 --node_rank 0 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr localhost --module large_4 --checkpoint_dir output --partition large_4/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path large_4/32dp_conf.json -b 1

host 1
python3 -m launch --nnodes 4 --node_rank 1 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr 172.31.7.136 --module large_4 --checkpoint_dir output --partition large_4/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path large_4/32dp_conf.json -b 1

host 2

python3 -m launch --nnodes 4 --node_rank 2 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr 172.31.7.136 --module large_4 --checkpoint_dir output --partition large_4/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path large_4/32dp_conf.json -b 1

host 3

python3 -m launch --nnodes 4 --node_rank 3 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr 172.31.7.136 --module large_4 --checkpoint_dir output --partition large_4/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path large_4/32dp_conf.json -b 1

## 32 dp 4 pp 8 large 


host 0
python3 -m launch --nnodes 4 --node_rank 0 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr localhost --module large_8 --checkpoint_dir output --partition large_8/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path large_8/32dp_conf.json -b 1

host 1
python3 -m launch --nnodes 4 --node_rank 1 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr 172.31.7.136 --module large_8 --checkpoint_dir output --partition large_8/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path large_8/32dp_conf.json -b 1

host 2

python3 -m launch --nnodes 4 --node_rank 2 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr 172.31.7.136 --module large_8 --checkpoint_dir output --partition large_8/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path large_8/32dp_conf.json -b 1

host 3

python3 -m launch --nnodes 4 --node_rank 3 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr 172.31.7.136 --module large_8 --checkpoint_dir output --partition large_8/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path large_8/32dp_conf.json -b 1
## DeepSpeed 

DeepSpeed

Commands for machine 181/185/184/183:

cd cpm_baseline
nvidia-docker run -it -v $HOME/.ssh:/home/deepspeed/.ssh -v $PWD:/cpm --net=host --ipc=host dmye/cpm:v0
sudo mkdir -p /run/sshd; sudo /usr/sbin/sshd -p 2222

Commands for machine 181:
vim .deepspeed_env
insert NCCL_SOCKET_IFNAME=enp216s0
cd cpm
bash fine_tune_chid_*_*.sh


sudo vim +280 /usr/local/lib/python3.6/dist-packages/deepspeed/launcher/runner.py

sudo vim +49 /usr/local/lib/python3.6/dist-packages/deepspeed/launcher/multinode_runner.py


environment['PDSH_SSH_ARGS_APPEND'] = '-p2222'




## results
                  8 x 16      2.9 epoch hour
16 GPU batch size 32 x 16 2.36 epoch hour 


8 GPU batch size 32 x 8 4.44 epoch hour


4.5hr


### add tc 

 sudo tc qdisc add dev ens5 root tbf rate 40000mbit buffer 40m latency 400ms


### show tc 

 tc qdisc show dev ens5

show bandwdith

 sudo apt install nload

 nload


### VPipe mp 16  4 stage
 
python3 -m launch --nnodes 2 --node_rank 0 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr localhost --module large_4 --checkpoint_dir output --partition large_4/gpipe.json --s
ync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print
-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path large_4/mp_conf.json -b 2


python3 -m launch --nnodes 2 --node_rank 1 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr 172.31.7.136  --module large_4 --checkpoint_dir output --partition large_4/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path large_4/mp_conf.json -b 1


# vpipe mp 16 8 stage
python3 -m launch --nnodes 2 --node_rank 0 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr localhost --module large_8 --checkpoint_dir output --partition large_8/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path large_8/mp_conf.json -b 1


python3 -m launch --nnodes 2 --node_rank 1 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr 172.31.7.136  --module large_8 --checkpoint_dir output --partition large_8/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path large_8/mp_conf.json -b 1

# vpipe mp 32 8 stage

python3 -m launch --nnodes 4 --node_rank 0 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr localhost --module large_8 --checkpoint_dir output --partition large_8/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path large_8/mp_conf.json -b 4

python3 -m launch --nnodes 4 --node_rank 1 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr 172.31.7.136  --module large_8 --checkpoint_dir output --partition large_8/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path large_8/mp_conf.json -b 4

python3 -m launch --nnodes 4 --node_rank 2 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr 172.31.7.136  --module large_8 --checkpoint_dir output --partition large_8/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path large_8/mp_conf.json -b 4

python3 -m launch --nnodes 4 --node_rank 3 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr 172.31.7.136  --module large_8 --checkpoint_dir output --partition large_8/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path large_8/mp_conf.json -b 4