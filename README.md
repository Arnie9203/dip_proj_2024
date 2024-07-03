# 数字图像处理课程项目

## 相关文件

Face文件夹：包括第二部分面部处理所需要的模型

neural-style-pt文件夹：包括实现第三部分图像风格迁移的所有文件

configs文件夹：包括启动扩散模型的配置文件

stable_diffusion文件夹：stable diffusion的官方文件

checkpoints文件夹：用于存放扩散模型的checkpoint

environment.yaml：用于部署conda环境

task1.py：第一部分的实现代码

task2+3.py：第二部分的实现代码和第三部分图像风格迁移的代码实现

task3.py：第三部分基于InstructPix2Pix的交互界面的代码实现

## 环境部署

```sh
conda env create -f environment.yaml
conda activate dip_proj
```

## 界面启动

```sh
python task1.py
python task2+3.py
python task3.py
```

