# LeNet-5 卷积神经网络实验报告（MNIST 手写数字 0-9）

> **作业对齐清单（你需要交付/截图的点）**  
> - 代码（文档里展示代码时建议在 Word/PDF 里设置 **23 号及以上**；本项目代码在 `src/`）  
> - 运行结果截图（训练日志 + 生成的图：`runs/*/metrics.png`、`confusion_matrix.png`、`samples.png`）  
> - 项目结构截图（PowerShell 运行 `tree /f` 后截图）  
> - CNN 架构图（本文档已提供 LeNet-5 结构示意图，可直接使用；也可补充你自己的绘图截图）

---

## 1. 实验目标

- 使用 **LeNet-5** 卷积神经网络实现 **MNIST**（0-9）手写数字识别。
- 完成：依赖配置、数据集下载、模型搭建、训练、验证（测试集评估）、结果可视化与保存。

---

## 2. 环境与依赖配置（Windows / PowerShell）

### 2.1 使用 conda 创建环境（推荐：你已安装 conda）

#### 方案 A：用 `environment.yml` 一键安装（最可复现）

在项目根目录执行：

```bash
conda env create -f environment.yml
conda activate cnn-lenet
```

如果你电脑**没有 NVIDIA 显卡**或不想装 CUDA，把 `environment.yml` 里这行：

- `pytorch-cuda=12.1`

删除后重新创建环境即可（或直接使用下面“方案 B”的 CPU 安装命令）。

#### 方案 B：手动创建 conda 环境 + 安装依赖（更灵活）

```bash
conda create -n cnn-lenet python=3.10 -y
conda activate cnn-lenet
```

安装 PyTorch（两选一）：

- **GPU（NVIDIA + CUDA 12.1）**：

```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

- **CPU**：

```bash
conda install pytorch torchvision cpuonly -c pytorch -y
```

再安装其余依赖（本项目的绘图/评估等库）：

```bash
pip install -r requirements.txt
```

> 建议你截图：`conda activate cnn-lenet` 后运行 `python -c "import torch; print(torch.__version__)"` 的结果，作为环境配置证明。

### 2.2 使用 venv 创建虚拟环境（可选）

在项目根目录 `d:\code\CNN` 打开 PowerShell：

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

> 说明：  
> - `torch/torchvision`：模型与 MNIST 数据集加载  
> - `matplotlib`：训练曲线/混淆矩阵/样例图保存  
> - `scikit-learn`：混淆矩阵  
> - `tqdm`：训练进度条

---

## 3. 数据集下载与准备（MNIST）

本实验使用 `torchvision.datasets.MNIST`，第一次运行会 **自动下载** 到 `data/` 目录（已在 `.gitignore` 忽略）。

- **下载位置**：`data/`  
- **输入尺寸**：\(28 \times 28\)，单通道灰度图（shape：`1x28x28`）  
- **归一化**：均值 `0.1307`，方差 `0.3081`（MNIST 常用配置）

---

## 4. LeNet-5 网络结构说明（对齐作业“画结构图”要求）

### 4.0 架构图（直接使用）

![LeNet-5 架构图](./assets/lenet5-architecture.png)

### 4.1 结构示意（与题图对应）

下面给出按输入 \(28\times 28\times 1\) 推导的张量尺寸变化（MNIST 版本 LeNet-5）：

- **输入**：\(1\times 28\times 28\)  
- **Conv1**：核 \(5\times 5\)，6 个通道，padding=0 → \(6\times 24\times 24\)  
- **AvgPool1**：\(2\times 2\) → \(6\times 12\times 12\)  
- **Conv2**：核 \(5\times 5\)，16 个通道 → \(16\times 8\times 8\)  
- **AvgPool2**：\(2\times 2\) → \(16\times 4\times 4\)  
- **Flatten**：\(16\cdot 4\cdot 4=256\)  
- **FC1**：256 → 120  
- **FC2**：120 → 84  
- **FC3**：84 → 10（对应 0-9）

---

## 5. 代码实现说明（CNN / 训练 / 验证）

### 5.1 代码文件结构（用于“项目结构截图”）

建议在 PowerShell 运行并截图：

```bash
tree /f
```

本项目结构（你运行 `tree /f` 的输出应类似）：

```text
CNN
│  .gitignore
│  REPORT.md
│  requirements.txt
│
└─src
      model.py
      train.py
      utils.py
```

### 5.2 核心实现位置

- **模型（LeNet-5）**：`src/model.py`
- **训练+验证+产物保存**：`src/train.py`
- **评估/画图/混淆矩阵/样例保存**：`src/utils.py`

---

## 6. 训练模型（含验证/测试评估）

### 6.1 启动训练（会自动下载 MNIST）

在项目根目录执行：

```bash
python .\src\train.py --epochs 5 --batch-size 128 --lr 1e-3
```

如果你机器没有 CUDA 或想强制用 CPU：

```bash
python .\src\train.py --epochs 5 --cpu
```

> 如果你在某些 PowerShell/IDE 终端里 `conda activate cnn-lenet` 后运行仍提示找不到 `torch`，直接用下面这种方式启动训练（**不依赖激活**）最稳：
>
> ```bash
> conda run -n cnn-lenet python .\src\train.py --epochs 5 --cpu
> ```

### 6.2 训练过程中你会看到什么（用于“运行结果截图”）

命令行会打印每个 epoch 的 loss/acc（进度条形式）。  
训练结束后会输出：

- `Done. Artifacts saved to: runs\YYYYMMDD-HHMMSS`

### 6.3 训练产物（用于“运行结果截图/插图”）

每次运行会生成一个新的目录：`runs/时间戳/`，其中包括：

- **`config.json`**：本次训练超参数与设备信息
- **`model.txt`**：网络结构（可截图当作结构补充）
- **`metrics.jsonl`**：每个 epoch 的 train/val 指标（便于追溯）
- **`metrics.png`**：Loss/Accuracy 曲线图（建议截图或直接插入报告）
- **`confusion_matrix.png`**：测试集混淆矩阵
- **`samples.png`**：测试集样例预测可视化（对/错一眼可见）
- **`checkpoints/best.pt`**：验证准确率最高的模型权重

> 提交时建议至少截图：训练命令行 + `metrics.png` + `confusion_matrix.png` + `samples.png`。

---

## 7. 验证（测试集评估）过程说明

本项目把 **MNIST 测试集**当作验证/测试集合（作业通常允许这样写法；你也可以在训练集里再划分验证集）。

- **验证时机**：每个 epoch 训练结束后，在测试集上计算 `val_loss/val_acc`
- **指标定义**：
  - `Accuracy = 正确预测数 / 总样本数`
  - `CrossEntropyLoss` 为分类常用损失函数
- **可视化验证**：
  - 混淆矩阵：观察哪些数字更容易被混淆（例如 4/9、3/5 等）
  - 样例预测图：随机/前若干张图片展示真实标签与预测标签

---

## 8. 结果记录与分析（你可以按实际运行结果填写）

把你运行后某次 `runs/时间戳/metrics.jsonl` 的最终结果抄到这里（示例占位）：

- **训练轮数**：5  
- **Batch size**：128  
- **学习率**：1e-3  
- **最终测试集准确率**：_____%  
- **最好测试集准确率**：_____%（对应 `checkpoints/best.pt`）

简要分析（可写 5~10 行）：

- 曲线是否收敛？是否出现过拟合（train acc 高但 val acc 下降）？
- 哪些类别容易混淆？结合混淆矩阵举 1~2 个例子说明原因。

---

## 9.（可选加分/对齐题目第 2 条）自建手写数字数据集流程

> 题目第 2 条要求：自己构建数据集（纸上手写→裁剪为 28×28），并注意各类别数量均衡。  
> 本节给出推荐规范，你可以按你们班实际要求补充“收集数量、命名规则、发送邮箱”等信息。

### 9.1 数据采集建议

- **采集方式**：白纸上手写 0-9，多人多风格更好；用手机拍照或扫描。
- **数量建议**：每类至少 50 张（越多越稳），尽量各类数量一致。
- **文件夹结构**（推荐）：

```text
my_digits
├─0
├─1
├─2
...
└─9
```

### 9.2 图像预处理要点（把原图变成 28×28）

- 灰度化、二值化（可选）
- 找到数字区域（外接矩形），裁剪后做 **等比缩放**，再 padding 成 28×28
- 背景/前景颜色尽量与 MNIST 类似（黑底白字或白底黑字都可以，但训练时需一致）

> 若老师要求你必须在代码里实现自建数据集，你可以在此项目基础上新增 `src/custom_dataset.py`，并在 `train.py` 里把 MNIST loader 替换成自建数据集 loader。

---

## 10. 复现步骤（最短路径）

在 `d:\code\CNN`：

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\src\train.py --epochs 5
```

然后到最新的 `runs/时间戳/` 目录中截图/取图，完成提交材料。

