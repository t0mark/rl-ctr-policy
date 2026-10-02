"""actuator 특성의 도메인 랜덤화와 로봇 관절 구성 검사 이벤트."""


def randomize_motor_strength(env, env_ids, factor_range, actuator_names):
    """선택한 actuator의 출력 배율을 환경별로 샘플링한다."""
    robot = env.scene["robot"]
    for name in actuator_names:
        robot.actuators[name].randomize_strength(env_ids, factor_range)


def check_joint_names(env, env_ids, joint_names):
    """로봇 YAML의 관절 목록이 USD 로봇의 실제 관절 목록과 같은지 확인한다."""
    actual = set(env.scene["robot"].joint_names)
    expected = set(joint_names)
    if actual != expected:
        raise ValueError(f"로봇 YAML joints가 USD 관절과 다릅니다. "
                         f"YAML에만 있음: {sorted(expected - actual)}, USD에만 있음: {sorted(actual - expected)}")
