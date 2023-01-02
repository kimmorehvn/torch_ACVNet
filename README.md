# Moreh AI Framework validated
Original repo: https://github.com/gangweiX/ACVNet

## Install on MAF v22.9.2 with Pytorch, conda (use this repo or clone and update code as guided)
```bash
cd ml-workbench
cd ACVNet
mkdir pretrained_model
pip install -r requirements.txt
```

## Data preparation (train on KITTI dataset for quickly)
(if the link of .zip files not works in the future, see original README on how to download)
```bash
mkdir KITTI_2012 KITTI_2015
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip
unzip -d data_stereo_flow.zip KITTI_2012
unzip -d data_scene_flow.zip KITTI_2015
```

## Prepare Pretrained Model

Download from [Scene Flow Drive folder](https://drive.google.com/drive/folders/1oY472efAgwCCSxtewbbA2gEtee-dlWSG?usp=sharing) and put the file sceneflow.ckpt into ./pretrained_model

## Training
```bash
python main_kitti.py --kitti15_datapath ./KITTI_2015 --kitti12_datapath ./KITTI_2012 2>&1 | tee acvNet-maf.log
```

## Note some errors may occur and how to fix by code changes (already fix in this ml-workbench repo)
- Not supported device from moreh.driver (however on A100 Torch 1.7.1, work normally without cmt .cuda() lines):
```bash
File "main_kitti.py", line 182, in test_sample
    image_outputs["errormap"] = [disp_error_image_func.apply(disp_est, disp_gt) for disp_est in disp_ests]
  File "main_kitti.py", line 182, in <listcomp>
    image_outputs["errormap"] = [disp_error_image_func.apply(disp_est, disp_gt) for disp_est in disp_ests]
  File "/home/ubuntu/.conda/envs/pytorch/lib/python3.8/site-packages/moreh/driver/pytorch/torch/autograd/__init__.py", line 173, in apply
    outputs = frontend.register_custom_operation_exit(
  File "/home/ubuntu/.conda/envs/pytorch/lib/python3.8/site-packages/moreh/driver/pytorch/torch/common/frontend.py", line 793, in register_custom_operation_exit
    device_idx = get_device_index_from_tensor(interim_output_tensors[0])
  File "/home/ubuntu/.conda/envs/pytorch/lib/python3.8/site-packages/moreh/driver/pytorch/torch/common/frontend.py", line 129, in get_device_index_from_tensor
    return get_device_index(tensor.device)
  File "/home/ubuntu/.conda/envs/pytorch/lib/python3.8/site-packages/moreh/driver/pytorch/torch/common/frontend.py", line 198, in get_device_index
    raise exception.NotSupportedDeviceException
moreh.driver.pytorch.torch.common.exception.NotSupportedDeviceException: Not supported device.
```
-> Fix: In main_kitti.py, comment out these lines:
Lines 146-148 (train_sample) and 173-175 (test_sample):
```bash
imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
# imgL = imgL.cuda()
# imgR = imgR.cuda()
# disp_gt = disp_gt.cuda()
```

- Dataloader's workers are out of shared memory
```bash
RuntimeError: DataLoader worker (pid 20027) is killed by signal: Bus error. It is possible that dataloaders workers are out of shared memory. Please try to raise your shared memory limit.
...
RuntimeError: DataLoader worker (pid(s) 20027) exited unexpectedly
```
-> Fix: Line 61, 62 of main_kitti.py, set num_workers of into 0

```bash
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=0, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=0, drop_last=False)
```

# Original README:
------------

# Fast-ACVNet
Our significant extension version of ACV, named Fast-ACV, will be soon available at https://github.com/gangweiX/Fast-ACVNet

| Method | Scene Flow <br> (EPE) | KITTI 2012 <br> (3-all) | KITTI 2015 <br> (D1-all) | Runtime (ms) |
|---|---|---|---|---|
| Fast-ACVNet+ | 0.59 | 1.85 % | 2.01 % | 45 |
| HITNet | - | 1.89 % |1.98 % | 54 |
| CoEx | 0.69 | 1.93 % | 2.13 % | 33 |
| BGNet+ |  - | 2.03 % | 2.19 % | 35 |
| AANet |  0.87 | 2.42 % | 2.55 % | 62 |
| DeepPrunerFast | 0.97 | - | 2.59 % | 50 |

Our Fast-ACVNet+ outperforms all the published real-time methods on Scene Flow, KITTI 2012 and KITTI 2015


# ACVNet (CVPR 2022)
This is the implementation of the paper: [Attention Concatenation Volume for Accurate and Efficient Stereo Matching](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Attention_Concatenation_Volume_for_Accurate_and_Efficient_Stereo_Matching_CVPR_2022_paper.pdf), CVPR 2022, Gangwei Xu, Junda Cheng, Peng Guo, Xin Yang

## Introduction

An informative and concise cost volume representation is vital for stereo matching of high accuracy and efficiency. In this paper, we present a novel cost volume construction method which generates attention weights from correlation clues to suppress redundant information and enhance matching-related information in the concatenation volume. To generate reliable attention weights, we propose multi-level adaptive patch matching to improve the distinctiveness of the matching cost at different disparities even for textureless regions.

![image](https://github.com/gangweiX/ACVNet/blob/main/imgs/acv_network.png)

# How to use

## Environment
* Python 3.8
* Pytorch 1.10

## Install

### Create a virtual environment and activate it.

```
conda create -n acvnet python=3.8
conda activate acvnet
```
### Dependencies

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install tqdm
```

## Data Preparation
Download [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo), [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

## Train
Use the following command to train ACVNet on Scene Flow

Firstly, train attention weights generation network for 64 epochs,
```
python main.py --attention_weights_only True
```
Secondly, freeze attention weights generation network parameters, train the remaining network for another 64 epochs,
```
python main.py --freeze_attention_weights True
```
Finally, train the complete network for 64 epochs,
```
python main.py
```

Use the following command to train ACVNet on KITTI (using pretrained model on Scene Flow)
```
python main_kitti.py
```

## Test
```
python test_sceneflow.py
```



### Pretrained Model

[Scene Flow](https://drive.google.com/drive/folders/1oY472efAgwCCSxtewbbA2gEtee-dlWSG?usp=sharing)

## Results on KITTI 2015 leaderboard
[Leaderboard Link](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

| Method | D1-bg (All) | D1-fg (All) | D1-all (All) | Runtime (s) |
|---|---|---|---|---|
| ACVNet | 1.37 % | 3.07 % | 1.65 % | 0.20 |
| LEAStereo | 1.40 % | 2.91 % | 1.65 % | 0.30 |
| GwcNet | 1.74 % | 3.93 % | 2.11 % | 0.32 |
| PSMNet | 1.86 % | 4.62 % | 2.32 % | 0.41 |

## Qualitative results on Scene Flow Datasets, KITTI 2012 and KITTI 2015

### The left column is left image, and the right column is results of our ACVNet.

![image](https://github.com/gangweiX/ACVNet/blob/main/imgs/acv_result.png)

# Citation

If you find this project helpful in your research, welcome to cite the paper.

```
@inproceedings{xu2022attention,
  title={Attention Concatenation Volume for Accurate and Efficient Stereo Matching},
  author={Xu, Gangwei and Cheng, Junda and Guo, Peng and Yang, Xin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12981--12990},
  year={2022}
}

```

# Acknowledgements

Thanks to Xiaoyang Guo for opening source of his excellent work GwcNet. Our work is inspired by this work and part of codes are migrated from [GwcNet](https://github.com/xy-guo/GwcNet).
