# 环境配置
## 创建环境
### 方式一
conda env create -f environment.yml
### 方式二
conda create -n Transformer python=3.10.18 -y
pip install -r requirements.txt
## 加载环境
conda activate transformer

# 实验命令
## 全部
bash scripts/run_all.sh
## 基础模型
bash scripts/run_base.sh
## 参数搜索（考虑到运行时间因此对模型以及搜索域做了简化处理）
bash scripts/search_hyperparams.sh
## 消融实验（考虑到运行时间因此对模型做了简化处理）
bash scripts/ablate.sh

## 单头效果比多头好的补充实验
bash scripts/ablate2.sh


## 测试所有训练模型
bash scripts/test.sh

# 会话命令
## 创建会话
screen -S transformer
## 查看会话
screen -r transformer
## 删除会话
screen -S transformer -X quit
