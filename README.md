# Triple-Cross-domain-Attention-for-HAR
![Image text](https://github.com/yinntag/Triple-Cross-domain-Attention-for-HAR/blob/main/Model/TA.jpg)
# Abstract
Efficiently identifying activities of daily living (ADL) provides very important contextual information that is able to improve the effectiveness of various sports tracking and healthcare applications. Recently, attention mechanism that selectively focuses on time series signals has been widely adopted in sensor based human activity recognition (HAR), which can enhance interesting target activity and ignore irrelevant background activity. Several attention mechanisms have been investigated, which achieve remarkable performance in HAR scenario. Despite their success, prior these attention methods ignore the cross-interaction between different dimensions. In the paper, in order to avoid above shortcoming, we present a triplet cross-dimension attention for sensor-based activity recognition task, where three attention branches are built to capture the cross-interaction between sensor dimension, temporal dimension and channel dimension. The effectiveness of triplet attention method is validated through extensive experiments on four public HAR dataset namely UCI-HAR, PAMAP2, WISDM and UNIMIB-SHAR as well as the weakly labeled HAR dataset. Extensive experiments show consistent improvements in classification performance with various backbone models such as plain CNN and ResNet, demonstrating a good generality ability of the triplet attention. Visualization analysis is provided to support our conclusion, and actual implementation is evaluated on a Raspberry Pi platform.
# Requirements
- python 3.7
- pytorch >= 1.1.0
- torchvision
- numpy 1.21.2
# Usage
All datasets used in this paper can be download from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).
Run `python main.py` to train and test on several HAR datasets. 
# Contributing
We appreciate all contributions. Please do not hesitate to let me know if you have any problems during the reproduction.
# Citation
```
@article{tang2022triple,
  title={Triple Cross-Domain Attention on Human Activity Recognition Using Wearable Sensors},
  author={Tang, Yin and Zhang, Lei and Teng, Qi and Min, Fuhong and Song, Aiguo},
  journal={IEEE Transactions on Emerging Topics in Computational Intelligence},
  year={2022},
  publisher={IEEE}
}
```
