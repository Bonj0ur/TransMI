# TransMI: a Transfer-learning method for generalized Map Information measurement
Pytorch implementation of our paper “TransMI: a Transfer-learning method for generalized Map Information measurement”. We collect a Subjective dataset for deep learning in Information Theory of Cartography named SITC and propose a novel approach, dubbed TransMI, to measure the quantity of map information, which is faster, more unified, and more subjective.

![Pipeline](assets\Pipeline.png)

## 1. Requirements
- pytorch
- torchvision
- numpy
- pandas

## 2. Dataset
We provide the [Subjective dataset for deep learning in Information Theory of Cartography (SITC)](./data/) in this repository. Due to capacity limitation, we provide the data download address [[TransMI (figshare.com)](https://figshare.com/s/eb6665ae65b7cf9a93f6)].

| File Name / Folder            | Content                           |
| ----------------------------- | --------------------------------- |
| ./data/pre_train/data         | Relative SITC dataset             |
| ./data/pre_train/clsLabel.csv | Relative SITC labels              |
| ./data/fine_tune/data         | Absolute SITC dataset             |
| ./data/fine_tune/MOS.csv      | Absolute SITC mean opinion scores |
| ./data/fine_tune/ckpt         | pretrained model                  |

### 2.1 Relative SITC (ReSITC)

[ReSITC](./data/pre_train/) contains 2970 maps, among which there are 20 kinds of relative relationships as shown in the table below.

| Label |              Class               |
| :---: | :------------------------------: |
|   0   |     Increase in Polygon Type     |
|   1   |      Increase in Line Type       |
|   2   |      Increase in Point Type      |
|   3   |      Increase in Point Num       |
|   4   |     Increase in Polygon Num      |
|   5   |       Increase in Line Num       |
|   6   | Distribution Tends to be Chaotic |
|   7   |     Increase in Point Level      |
|   8   |  Increase in Polygon Complexity  |
|   9   |         Change in Color          |
|  10   |     Change in Point Meaning      |
|  11   |     Decrease in Polygon Type     |
|  12   |      Decrease in Line Type       |
|  13   |      Decrease in Point Type      |
|  14   |      Decrease in Point Num       |
|  15   |     Decrease in Polygon Num      |
|  16   |       Decrease in Line Num       |
|  17   | Distribution Tends to be Orderly |
|  18   |     Decrease in Point Level      |
|  19   |  Decrease in Polygon Complexity  |

### 2.2 Absolute SITC (AbSITC)

[AbSITC](./data/fine_tune/) contains 330 maps, and each map is scored by 5 participants according to the quantity of map information.

## 3. Reproducibility of the results

You can reproduce the results in the paper according to the following instructions. 

### 3.1 Pre-training Stage

![Pre-training Results](assets\Pretraining_Results.png)

To reproduce the k-fold cross validation results in pre-training stage:

```bash
cd pre_train
python train.py
```

To generate the pretrained model for fine-tuning stage:

```bash
cd ..
cd generate_ckpoint
python train.py
```

### 3.2 Fine-tuning Stage

![Fine-tuning Results](assets\Finetuning_Results.png)

To reproduce the k-fold cross validation results in fine-tuning stage:

```bash
cd ..
cd fine_tune
python train.py
```

## 4. Visualization

Here, we provide some prediction results of our model.

![Visualization](assets\Visualization.png)
