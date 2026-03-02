from phc.env.tasks.humanoid_combat_g1 import HumanoidCombatG1, HumanoidCombatG1Z


PM01_STRIKE_BODY_NAMES = [
    "link_knee_pitch_l", "link_ankle_pitch_l", "link_ankle_roll_l",
    "link_knee_pitch_r", "link_ankle_pitch_r", "link_ankle_roll_r",
]
PM01_FOOT_NAMES = [
    "link_ankle_pitch_l", "link_ankle_roll_l",
    "link_ankle_pitch_r", "link_ankle_roll_r",
]
PM01_HAND_NAMES = [
    "link_elbow_yaw_l", "link_elbow_yaw_r",
    "link_knee_pitch_l", "link_ankle_roll_l",
    "link_knee_pitch_r", "link_ankle_roll_r",
]
PM01_TARGET_NAMES = [
    "link_base", "link_torso_yaw", "link_head_yaw", "link_hip_yaw_l", "link_hip_yaw_r"
]


class _PM01CombatMixin:
    def _get_combat_body_names(self):
        return PM01_STRIKE_BODY_NAMES, PM01_FOOT_NAMES, PM01_HAND_NAMES, PM01_TARGET_NAMES

    def get_task_obs_size(self):
        if self._enable_task_obs:
            num_bodies = len(self._body_names)
            num_dofs = len(self._dof_names)
            num_hand = len(PM01_HAND_NAMES)
            num_target = len(PM01_TARGET_NAMES)
            obs_size = (3 + 6 + 3 + 3)
            obs_size += num_dofs + num_dofs
            obs_size += num_bodies * 3 + num_bodies * 6
            obs_size += num_bodies + num_bodies
            obs_size += num_hand * num_target * 3
            return obs_size
        return 0


class HumanoidCombatPM01(_PM01CombatMixin, HumanoidCombatG1):
    pass


class HumanoidCombatPM01Z(_PM01CombatMixin, HumanoidCombatG1Z):
    pass
