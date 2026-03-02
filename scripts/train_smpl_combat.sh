export OMP_NUM_THREADS=1
python phc/run_hydra.py  \
       project_name=SLPM_combat num_agents=2     \
       learning=self_play exp_name=combat_smpl \
       env=env_default \
       env.num_envs=3072 \
       robot=smpl_humanoid \
       headless=True \
       learning.params.config.switch_frequency=250 \
       env.self_obs_v=1
