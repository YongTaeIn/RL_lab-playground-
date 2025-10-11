### 강화학습 기본 컨셉
- 현재의 state(상태)에서 어떤 action(행동)을 취하는 것이 최적인지 학습하는 것이며, 행동을 취할 때마다 외부 environment(환경)에서 reward(보상)가 주어지는데, 이러한 보상을 최대화 하는 방향으로 학습이 진행됨.



### 강화학습의 목표 
- Optimal Policy (π*) 를 찾자 .
    -> 에이전트가 장기적으로 받을 **누적 보상(return)** 을 최대화하는 정책(optimal policy) π* 를 찾는 것
'''
    수식 작성
'''




### 강화학습 기본 용어 설명.

1. State
2. action
3. reward
4. episode
5. policy
6. Q-function (action-value function)
7. agent


### RL agnet 의 중요한 3가지 요소
1. policy (π) : 정책
    - 정의 : Agent의 행동을 결정해주는 함수. 즉 State(현재 상태)에서 Action(행동)을 할 수 있도록 정해줌.
    - 쉬운 이해 : 에이전트가 행동을 결정하는 전략
    - 수식 : π(a|s)
        - Deterministic policy : a = π(s)
        - Stochasitc policy : π(a|s) = P[A_t =a|S_t =s]
    - 수식 의미 : 상태 s에서 행동 a를 선택할 확률 또는 규칙
2. value function
    - 정의 : 미래에 받는 보상에 대한 함수, 취할 State가 좋은지 나쁜지 평가 해줌.
    - 쉬운 이해 : 현재 policy가 얼마나 잘하고 있는지를 수치로 평가해주는 함수
    - 특이 사항 : reward랑 다름, reward는 다음 스텝갈 때 주는 것.
    - 수식 : 
        - State Value function : V^π(s) = Eπ​[t=0∑∞​γtrt​∣s0​=s] ='수식을 풀어서도 작성해줘'
        - State - Action Value function(= Q-Function) : Q^π(s,a) =Eπ​[t=0∑∞​γtrt​∣s0​=s,a0​=a] ='수식을 풀어서도 작성해줘'
3. model
    - 정의 : agent's representation of the environment (환경이 1. next state (새로운 상태)를 내놓고, reward(보상)을 준다. 즉 모델을 알고 있으면 next state, reward를 알 수 있다. 
    - 쉬운 이해 : 에이전트가 환경의 규칙을 머릿속에 표현한 내적 시뮬레이터
    - 수식 : 
        - Pss′a​=Pr[St+1​=s′∣St​=s,At​=a] (의미 : s에서 a를 했을 때 s′로 이동할 확률)
        - Rsa​=E[Rt+1​∣St​=s,At​=a] (의미 : 상태 s에서 행동 a를 했을 때 받을 평균 보상)