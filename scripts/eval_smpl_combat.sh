python phc/run_hydra.py  \
       project_name=SLPM_combat num_agents=2     \
       learning=self_play exp_name=combat_smpl \
       env=env_default \
       robot=smpl_humanoid \
       learning.params.config.switch_frequency=250 \
       env.self_obs_v=1 \
       no_virtual_display=True epoch=-1 test=True env.num_envs=4 headless=False
