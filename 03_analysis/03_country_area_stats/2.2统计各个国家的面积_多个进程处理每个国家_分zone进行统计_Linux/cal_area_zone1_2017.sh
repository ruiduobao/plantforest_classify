#!/bin/bash
#SBATCH --job-name=zone1_2017     # 作业名称
#SBATCH --output=raster_%j.log      # 输出文件，%j会被替换为作业ID
#SBATCH --error=raster_%j.err       # 错误日志文件
#SBATCH --nodes=1                   # 请求节点数
#SBATCH --ntasks-per-node=1         # 每个节点的任务数
#SBATCH --cpus-per-task=40         # 每个任务使用的CPU核心数
#SBATCH --time=24:00:00             # 最大运行时间
#SBATCH --partition=tyhcnormal         # 分区名称（根据实际情况修改）
# 加载conda环境
source ~/.bashrc
conda activate geodetector
# 设置Python脚本的工作目录（根据实际情况修改）
cd /work/home/chengrui1075/SEA_TREE/CODE/3.analy/calarea
# 运行Python脚本
python ./2.2.cal_ZONE_NUMBER_LINUX.py





















