export OMP_NUM_THREADS=1
python phc/run_hydra.py \
       project_name=SLPM_combat num_agents=2 \
       learning=self_play exp_name=combat_g1_pm01_mixed \
       env=env_combat_mixed_g1_pm01 \
       env.num_envs=2048 \
       sim=robot_sim \
       learning.params.network.space.continuous.sigma_init.val=-1.7 \
       headless=True \
       learning.params.config.save_frequency=500 \
       learning.params.config.switch_frequency=250
