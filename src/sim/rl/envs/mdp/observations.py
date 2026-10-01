"""DreamWaQ critic 특권 관측 항목."""


def disturbance_force(env, asset_cfg):
    """몸통 링크에 가해진 외란 힘 d_t를 링크 좌표계로 반환한다."""
    robot = env.scene[asset_cfg.name]
    return robot.permanent_wrench_composer.composed_force_as_torch[:, asset_cfg.body_ids[0]]
