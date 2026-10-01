"""actuator 특성의 도메인 랜덤화 이벤트."""


def randomize_motor_strength(env, env_ids, factor_range, actuator_names):
    """선택한 actuator의 출력 배율을 환경별로 샘플링한다."""
    robot = env.scene["robot"]
    for name in actuator_names:
        robot.actuators[name].randomize_strength(env_ids, factor_range)
