# ST-CGAN: Stacked Conditional Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal with PyTorch

This repository is unofficial implementation of  [Stacked Conditional Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal](https://arxiv.org/abs/1712.02478) [Wang+, **CVPR** 2018] with PyTorch.

**Official Dataset and Code(coming soon...) is [here](https://github.com/DeepInsight-PCALab/ST-CGAN).**

## Requirements
* Python3.x
* PyTorch 1.5.0
* pillow
* matplotlib

## Usage
* Set datasets under ```./dataset```. You can Download datasets from [here](https://github.com/DeepInsight-PCALab/ST-CGAN).

Then,
```
python3 train.py
```

## Results
Here are some results from validation set.
### Shadow Detection
![](https://github.com/IsHYuhi/ST-CGAN_Stacked_Conditional_Generative_Adversarial_Networks/blob/master/result/detected_shadow.jpg)
(Top to bottom: ground truth, shadow detection)

### Shadow Removal
![](https://github.com/IsHYuhi/ST-CGAN_Stacked_Conditional_Generative_Adversarial_Networks/blob/master/result/shadow_removal.jpg)
(Top to bottom: input, ground truth, shadow removal)

## References
* Stacked Conditional Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal, Jifeng Wang<sup>∗</sup>, Xiang Li<sup>∗</sup>, Le Hui, Jian Yang, **Nanjing University of Science and Technology**, [[arXiv]](https://arxiv.org/abs/1712.02478)
