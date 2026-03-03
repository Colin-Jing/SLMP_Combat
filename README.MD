# Spherical Latent Motion Prior for Physics-Based Simulated Humanoid Control 

[[website]](https://colin-jing.github.io/SLMP/)

<div float="center">
  <img src="assets/SLMP_Combat.gif" />
</div>


## TODOs

- [ ] Release dataset.

- [ ] Release SLMP prior training code.

- [x] Release SLMP combat code. 


## Intallation 

1. Create new conda environment and install pytroch:

```
conda create -n isaac python=3.8
[install pytorch]
pip install -r requirement.txt
```

2. Install [isaacgym](https://docs.nvidia.com/isaac/isaacgym/doc/setup.html)

3. Download SMPL paramters from [SMPL](https://smpl.is.tue.mpg.de/) and [SMPLX](https://smpl-x.is.tue.mpg.de/download.php). Put them in the `data/smpl` folder, unzip them into 'data/smpl' folder. For SMPL, please download the v1.1.0 version, which contains the neutral humanoid. Rename the files `basicmodel_neutral_lbs_10_207_0_v1.1.0`, `basicmodel_m_lbs_10_207_0_v1.1.0.pkl`, `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` to `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl` and `SMPL_FEMALE.pkl`. For SMPLX, please download the v1.1 version. Rename The file structure should look like this:

```

|-- data
    |-- smpl
        |-- SMPL_FEMALE.pkl
        |-- SMPL_NEUTRAL.pkl
        |-- SMPL_MALE.pkl
        |-- SMPLX_FEMALE.pkl
        |-- SMPLX_NEUTRAL.pkl
        |-- SMPLX_MALE.pkl

```

4. Download the pre-trained prior models from [Google Drive](https://drive.google.com/drive/folders/16BxgFYTnuyvzuHAs2eiVgBVnLpdyzPjx?usp=sharing) and put them in the `output` folder. The file structure should look like this:

```
|-- output
    |-- HumanoidIm
        |-- g1_combat_prior
            |-- Humanoid_00007000.pth
        |-- pm01_combat_prior
            |-- Humanoid.pth
        |-- smpl_combat_prior
            |-- Humanoid_00017700.pth
```

# Commands

Training and inference scripts are in the `scripts` folder:

- **SMPL self-play**: `bash scripts/train_smpl_combat.sh` / `bash scripts/eval_smpl_combat.sh`
- **G1 self-play**: `bash scripts/train_g1_combat.sh` / `bash scripts/eval_g1_combat.sh`
- **PM01 self-play**: `bash scripts/train_pm01_combat.sh` / `bash scripts/eval_pm01_combat.sh`
- **G1 vs PM01 combat**: `bash scripts/train_g1_pm01_mixed_combat.sh` / `bash scripts/eval_g1_pm01_mixed_combat.sh`



## Acknowledgement
This project is built on top of [SMPLOlympics](https://smplolympics.github.io/SMPLOlympics/).
