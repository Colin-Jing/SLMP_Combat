python phc/run_hydra.py \
       project_name=SLPM_combat num_agents=2 \
       learning=self_play exp_name=combat_g1 \
       env=env_combat_g1 \
       robot=g1_29dof \
       sim=robot_sim \
       learning.params.config.switch_frequency=250 \
       no_virtual_display=True epoch=-1 test=True env.num_envs=1 headless=False
