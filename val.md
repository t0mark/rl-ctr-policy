# 추가 검증 후보

우선순위: Critic → 보상 함수 → 학습 환경

- Critic
- PPO 손실
- 보상 함수
- 관측·행동
- 학습 환경
- 2단계 전환·운영


참고: 이번에 확인한 DreamWaQ Table I의 다른 값들
앞선 검증에서 애매했던 부분들이 정리됩니다.

항목	DreamWaQ	Two-Phase	우리 코드
Orientation	|g|² −0.2	|g|² +1.0	−1.0
Joint accelerations	−2.5e−7	−1e−6	−1e−6 (논문)
Action rate / Smoothness	−0.01 / −0.01	−0.01 / −0.01	동일 ✅
Ang. velocity tracking	0.5	1.1	1.1 (논문)
Power distribution	var(τ·θ̇)² −1e−5	없음	없음
특히 Orientation은 DreamWaQ가 음수(−0.2) 를 씁니다. Two-Phase 표의 +1.0이 표기 오류라는 앞선 판단을 뒷받침합니다. 코드의 −1.0은 방향이 맞고, 크기는 Two-Phase 값을 따르고 있습니다.

Power distribution(모터 간 파워 분산 벌점)은 DreamWaQ에만 있고 Two-Phase Table 4에는 없습니다. 현재 코드에도 없으니 Two-Phase 기준으로는 맞습니다.



시계열에서 관찰된 것
기준 동작 재생 중 dof_acc_l2가 착지 순간마다 −0.5 ~ −1.0의 스파이크를 만듭니다. 평균(−0.18)보다 훨씬 큽니다. 착지 충격에서 관절 가속도가 급증하므로 정상입니다.

다만 rollout 시작 직후 −6.2의 큰 스파이크가 한 번 나타납니다. 리셋 직후 공중에서 떨어져 첫 접지할 때 발생합니다. 학습 초기에는 넘어져서 리셋이 잦으므로 이 스파이크가 반복적으로 들어갑니다. 에피소드가 짧을수록 평균 보상에서 차지하는 비중이 커지니, 학습 초기 지표를 볼 때 감안하시면 됩니다.