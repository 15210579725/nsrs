#!/bin/bash

# 打开第一个终端并执行 visdom
gnome-terminal -- bash -c "visdom; exec bash"

# 打开第二个终端，执行 tmux 然后执行指定的 Python 脚本
gnome-terminal -- bash -c "tmux new-session -d 'env CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0 xvfb-run -a -s \"-screen 0 1400x900x24\" python nsrs/experiments/gym/run_se_control.py'; exec bash"

# 打开第三个终端，执行 tmux 然后执行指定的 Python 脚本
gnome-terminal -- bash -c "tmux new-session -d 'env CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=2 xvfb-run -a -s \"-screen 0 1400x900x24\" python nsrs/experiments/gym/run_se_control.py'; exec bash"

# 打开第四个终端并执行 nvitop
gnome-terminal -- bash -c "nvitop; exec bash"

# 打开第五个终端并执行 htop
gnome-terminal -- bash -c "htop; exec bash"
