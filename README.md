# **Hand Avatar: Free-Pose Hand Animation and Rendering from Monocular Video (CVPR 2023)**
[Project Page](https://seanchenxy.github.io/HandAvatarWeb/?utm_source=catalyzex.com) | [Paper](https://arxiv.org/abs/2211.12782)

## Updata
+ 2023-4-20 inference code is released

## Prerequisite
Create a new enviornment by
```
conda env create -f env.yml
```
or
```bash
conda create -n handavatar python=3.9
conda activate handavatar
pip install -r requirements.txt
```

## Download
+ Download pretrained model and preprocessed data from [Google Drive](https://drive.google.com/drive/folders/19X0XOPWCrTPx4IAs2jpj34qbO0bC2Pew?usp=sharing).
+ Download offical [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/).

### Link
```bash
cd data
ln -s /path/to/InterHand2.6M InterHand
```
+ The floder is as follows
    ```
    ROOT
        ├──data
            ├──InterHand
                ├──5
                    ├──InterHand2.6M_5fps_batch1
                        ├──images
                        ├──preprocess
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
    ```

### Data preprocessing
Comming soon. You can use our released data now.


## Inference
```
./handavatar/scripts/run_hand.sh
```
+ Results are saved in the same floder of pretrained model.

## Metrics
Set `phase: 'val'` in the config file, then run
```
./handavatar/scripts/train_hand.sh
```

## Training
Set `phase: 'train'` in the config file, then run
```
./handavatar/scripts/train_hand.sh 
```
+ Tensorbard is used to track training process.
+ The training need >32G GPU memory. You can reduce `patch: N_patches` or `patch: size` to reduce GPU usage . In the mean time, incearsing `train: accum_iter` can control batch size in an optimization step.
