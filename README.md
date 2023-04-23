# **Hand Avatar: Free-Pose Hand Animation and Rendering from Monocular Video (CVPR 2023)**
[Project Page](https://seanchenxy.github.io/HandAvatarWeb/?utm_source=catalyzex.com) | [Paper](https://arxiv.org/abs/2211.12782)

## Prerequisite
+ Create a new enviornment by
    ```
    conda env create -f env.yml
    ```
    or (recommend)
    ```bash
    conda create -n handavatar python=3.9
    conda activate handavatar
    pip install -r requirements.txt
    ```

## Download
+ Download offical [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/).
+ You should accept MANO LICENCE and download MANO model from official website. 
+ Link floder
    ```bash
    ln -s /path/to/mano_v1_2 mano
    cd mano
    ln -s models/MANO_RIGHT.pkl MANO_RIGHT.pkl
    cd .. && mkdir data && cd data
    ln -s /path/to/InterHand2.6M InterHand
    ```
+ Hand segmentation (Unnecessary, If only runing inference): MANO meshes are projected to generate hand masks.
    ```bash
    python segment/seg_interhand2.6m_from_mano.py
    ```
    > Set `subject` in the file for different subjects.
+ Download pretrained model and pre-processed data from [Google Drive](https://drive.google.com/drive/folders/19X0XOPWCrTPx4IAs2jpj34qbO0bC2Pew?usp=sharing).
    > The data can also be pre-processed by `python handavatar/core/data/interhand/train.py` after generating hand segmentation mask.

+ The overall floder is as follows
    ```
    ROOT
        ├──data
            ├──InterHand
                ├──5
                    ├──InterHand2.6M_5fps_batch1
                        ├──images
                        ├──preprocess
                        ├──masks_removeblack
                    |──annotations
        ├──handavatar
            ├──out/handavatar/interhand
                ├──test_Capture0_ROM04_RT_Occlusion
                    ├──pretrained_model/latest.tar
                ├──test_Capture1_ROM04_RT_Occlusion
                    ├──pretrained_model/latest.tar
                ├──val_Capture0_ROM04_RT_Occlusion
                    ├──pretrained_model/latest.tar
        ├──smplx
            ├──out/pretrained_lbs_weights
                ├──lbs_weights.pth
        ├──pairof
            ├──out/pretrained_pairof
                ├──pairof.ckpt
        ├──mano
    ```

## Inference
```
./handavatar/scripts/run_hand.sh
```
+ Results are saved in the same floder of pretrained model.

## Metrics
Set the following configs in the config file
```yaml
phase: 'val'
experiment: 'pretrained_model'
resume: True
```
then run
```
./handavatar/scripts/train_hand.sh
```

## Training
Set the following configs in the config file
```yaml
phase: 'train'
experiment: 'your_exp_name'
resume: False
```
then run
```
./handavatar/scripts/train_hand.sh 
```
+ Tensorbard can be to track training process.
+ The training need one GPU and >32G memory. You can reduce `patch: N_patches` or `patch: size` to reduce GPU usage . In the mean time, incearsing `train: accum_iter` can control batch size (i.e., `batch_size=N_patches*accum_iter`) in an optimization step.

## Reference
```tex
@inproceedings{bib:handavatar,
  title={Hand Avatar: Free-Pose Hand Animation and Rendering from Monocular Video},
  author={Chen, Xingyu and Wang, Baoyuan and Shum, Heung-Yeung},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```

## Acknowledgement
Our implementation is based on [COAP](https://github.com/markomih/COAP), [SMLPX](https://github.com/vchoutas/smplx), and [HumanNeRF](https://github.com/chungyiweng/humannerf). We thank them for inspiring implementations.
