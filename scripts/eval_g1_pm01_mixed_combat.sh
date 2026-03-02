export OMP_NUM_THREADS=1
python phc/run_hydra.py \
       project_name=SLPM_combat num_agents=2 \
       learning=self_play exp_name=combat_g1_pm01_mixed \
       env=env_combat_mixed_g1_pm01 \
       env.num_envs=1 \
       sim=robot_sim \
       headless=False \
       no_virtual_display=True \
       epoch=-1 test=True
