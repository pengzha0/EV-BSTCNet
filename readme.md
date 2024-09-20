This repository contains the PyTorch implementation for 
"Event-Based Binary Neural Networks for Efficient and Accurate Lip Reading" 

## Running environment

Training: Pytorch 1.8.1 with cuda 11.1 in an Ubuntu 20.04 system <br>

*other versions may also work~ :)*

## Dataset

We use the links in [DVS-Lip](https://drive.google.com/file/d/1dBEgtmctTTWJlWnuWxFtk8gfOdVVpkQ0/view) (really thanks for that). You should download the dataset and put it in the folder "data/"

## training
```python
python main.py --gpus=0 --back_type='TCN' --evaluate=None
```

## test
```python
python main.py --gpus=0 --back_type='TCN' --evaluate="path to your .pth"
```

## result
| Model                  | Acc1   | Acc2   | Acc |
|------------------------|-------|-------|-----|
| [EV-STCNet](https://drive.google.com/file/d/1y-yo73M79YDeUWODT7jTO4vkPjHcsBU1/view?usp=drive_link)        | 65.86 | 87.43 | 76.62 |
| [EV-BSTCNet](https://drive.google.com/file/d/1PxLPkKZ0X9FmlOm2FhCR-hXDie_IBrD_/view?usp=drive_link)         | 66.03 | 86.26 | 76.15  |
