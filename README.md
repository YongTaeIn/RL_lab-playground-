# 강화학습 (Reinforcement Learning) 입문 가이드

> 처음 강화학습을 배우는 사람을 위한 핵심 개념 정리

---

## 🎯 강화학습이란?

**강화학습(Reinforcement Learning, RL)** 은 현재의 **State(상태)** 에서 어떤 **Action(행동)** 을 취하는 것이 최적인지 학습하는 방법입니다.

- 행동을 취할 때마다 외부 **Environment(환경)** 에서 **Reward(보상)** 이 주어집니다
- 이러한 보상을 **최대화** 하는 방향으로 학습이 진행됩니다

```
Agent(에이전트) ⟷ Environment(환경)
    ↓ Action           ↓ State, Reward
  (오른쪽 이동)        (현재 위치 : 5, 보상 : +1)
```

---

## 🏆 강화학습의 목표

### Optimal Policy (π<sup>\*</sup>) 찾기

에이전트가 장기적으로 받을 **누적 보상(Return)** 을 최대화하는 **최적 정책(Optimal Policy)** π<sup>\*</sup> 를 찾는 것이 강화학습의 목표입니다.

**수식:**

$$\pi^{*}(s) = \arg\max_a Q^{*}(s,a)$$

**의미:**
- $\pi^*(s)$ : 상태 s에서의 최적 정책 (최적의 행동)
- $\arg\max_a$ : 가장 큰 값을 만드는 행동 a를 선택
- $Q^*(s,a)$ : 상태 s에서 행동 a의 최적 가치
- $*$ : Optimal(최적의) 라는 뜻
- **해석**: 각 상태에서 Q값이 가장 높은 행동을 선택하는 것이 최적 정책!

---

## 📚 강화학습 기본 용어

### 1. State (상태, S)

**정의:**
- 환경에서 에이전트가 현재 위치한 상태
- 에이전트가 관찰할 수 있는 환경의 정보

**쉬운 이해:**
- 게임 캐릭터의 현재 위치
- 체스판의 현재 배치
- 자동차의 현재 속도와 위치

**예시:**
```python
current_state = 0  # 16개 상태 중 0번 상태
```

---

### 2. Action (행동, A)

**정의:**
- 특정 상태에서 에이전트가 수행할 수 있는 선택지

**쉬운 이해:**
- 게임에서 선택할 수 있는 버튼 (점프, 공격, 방어)
- 미로에서 움직일 방향 (상, 하, 좌, 우)

**예시:**
```python
n_actions = 4  # 상, 하, 좌, 우
action = 0  # "상" 선택
```

---

### 3. Reward (보상, R)

**정의:**
- 행동을 수행한 후 환경으로부터 받는 피드백 신호
- 에이전트의 행동이 얼마나 좋았는지를 나타내는 수치

**쉬운 이해:**
- 게임에서 적을 물리치면 +10점
- 벽에 부딪히면 -1점
- 목표에 도달하면 +100점

**특징:**
- **즉각적인 보상**: 행동 직후 바로 받는 보상
- Value Function과는 다름 (Value는 미래의 누적 보상)

**예시:**
```python
reward = 1 if next_state == goal_state else 0
```

---

### 4. Episode (에피소드)

**정의:**
- 시작 상태에서 종료 상태(목표 또는 실패)까지 도달하는 일련의 과정

**쉬운 이해:**
- 게임 한 판 (시작 → 게임 오버 또는 승리)
- 미로 찾기 한 번 (출발 → 목표 도달 또는 포기)

**예시:**
```python
for episode in range(1000):  # 1000번의 에피소드
    state = initial_state
    while not done:
        action = select_action(state)
        next_state, reward, done = env.step(action)
```

---

### 5. Agent (에이전트)

**정의:**
- 환경과 상호작용하며 학습하는 주체
- 상태를 관찰하고 행동을 선택하는 학습자

**쉬운 이해:**
- 게임을 플레이하는 AI
- 미로를 탈출하려는 로봇
- 체스를 두는 컴퓨터

---
### 6. Return ()

**정의:**
- 추가예정 (G_t)

**쉬운 이해:**
- 추가예정

---
## 🧠 RL Agent의 3가지 핵심 요소

강화학습 에이전트는 다음 3가지 요소로 구성됩니다:

---

### 1️⃣ Policy (정책, π)

**정의:**
- Agent의 행동을 결정해주는 함수
- State(현재 상태)에서 Action(행동)을 선택하도록 정해줌

**쉬운 이해:**
- 에이전트가 행동을 결정하는 **전략** 또는 **규칙**
- "이 상황에서는 이렇게 행동해!" 라는 지침

**수식:**

$$\pi(a|s) = P[A_t = a \mid S_t = s]$$

$\pi(a|s) = P[A_t = a \mid S_t = s]$

**수식 의미:**
- 상태 s에서 행동 a를 선택할 확률

**종류:**

1. **Deterministic Policy (결정적 정책)**
   
   $$a = \pi(s)$$
   
   - 상태 s가 주어지면 항상 같은 행동 a를 선택
   - 예: "출구가 오른쪽에 있으면 항상 오른쪽으로 이동"

2. **Stochastic Policy (확률적 정책)**
   
   $$\pi(a|s) = P[A_t = a \mid S_t = s]$$
   
   - 상태 s에서 여러 행동 중 확률적으로 선택
   - 예: "70% 확률로 오른쪽, 30% 확률로 왼쪽"

**코드 예시:**
```python
# Epsilon-Greedy Policy
if random.random() < epsilon:
    action = random.choice(actions)  # 탐색
else:
    action = argmax(Q_table[state])  # 활용
```

---

### 2️⃣ Value Function (가치 함수)

**정의:**
- 미래에 받을 보상을 평가하는 함수
- 현재 State(상태) 또는 Action(행동)이 얼마나 좋은지 평가

**쉬운 이해:**
- 현재 policy가 얼마나 잘하고 있는지를 **수치로 평가**해주는 함수
- "이 상태에서 미래까지 얼마나 많은 보상을 받을 수 있을까?"

**⚠️ 중요:**
- **Reward ≠ Value Function**
- Reward는 **즉각적인** 보상 (다음 스텝에서 받는 것)
- Value는 **장기적인** 누적 보상의 기댓값

---

#### 2-1. State Value Function (V<sup>π</sup>(s))

상태 s가 얼마나 좋은지 평가

**수식:**

$$V^{\pi}(s) = \mathbb{E}_{\pi}[ G_t \mid S_t = s ]$$

**수식 의미:**
- $G_t$ : Return (누적 보상) = $\sum \gamma^t \times r_t$
- $S_t = s$ : 시간 t에서 상태가 s일 때
- "상태 s에서의 가치는 그 상태에서 얻을 수 있는 누적 보상의 기댓값"

**풀어 쓰면:**

$$V^{\pi}(s) = \mathbb{E}_{\pi}\left[ \sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s \right]$$

$$V^{\pi}(s) = \mathbb{E}_{\pi}\left[ r_0 + \gamma r_1 + \gamma^2 r_2 + \gamma^3 r_3 + \cdots \mid s_0 = s \right]$$

**의미:**
- 상태 s에서 시작해서 정책 π를 따를 때
- 앞으로 받을 모든 보상의 할인된 합의 기댓값
- **쉽게 말하면**: "이 상태에 있으면 미래에 얼마나 좋을까?"

---

#### 2-2. Action-Value Function (Q<sup>π</sup>(s,a)) = Q-Function

상태 s에서 행동 a를 했을 때 얼마나 좋은지 평가

**수식:**

$$q^{\pi}(s,a) = \mathbb{E}_{\pi}[ G_t \mid S_t = s, A_t = a ]$$

**수식 의미:**
- $G_t$ : Return (누적 보상) = $\sum \gamma^t \times r_t$
- $S_t = s$ : 시간 t에서 상태가 s일 때
- $A_t = a$ : 시간 t에서 행동이 a일 때
- "상태 s에서 행동 a를 했을 때의 가치는 그때부터 얻을 수 있는 누적 보상의 기댓값"

**풀어 쓰면:**

$$Q^{\pi}(s,a) = \mathbb{E}_{\pi}\left[ \sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s, a_0 = a \right]$$

$$Q^{\pi}(s,a) = \mathbb{E}_{\pi}\left[ r_0 + \gamma r_1 + \gamma^2 r_2 + \gamma^3 r_3 + \cdots \mid s_0 = s, a_0 = a \right]$$

**의미:**
- 상태 s에서 행동 a를 선택하고
- 이후 정책 π를 따를 때 받을 모든 보상의 할인된 합의 기댓값
- **쉽게 말하면**: "이 상태에서 이 행동을 하면 미래에 얼마나 좋을까?"

**Q-Learning 업데이트 수식:**

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]$$

**각 항의 의미:**
- $Q(s,a)$ : 현재 추정치 (지금까지의 추정)
- $\alpha$ : 학습률 (Learning Rate, 얼마나 빨리 업데이트할지)
- $r$ : 즉각적인 보상 (Immediate Reward)
- $\gamma \max_{a'} Q(s',a')$ : 다음 상태에서 얻을 수 있는 최대 가치
- $\left[ \cdots \right]$ : TD Error (시간차 오차) = 실제 - 예측

---

### 3️⃣ Model (모델)

**정의:**
- Agent's representation of the environment
- 환경이 어떻게 동작하는지에 대한 에이전트의 내부 표현
- 환경의 규칙을 알고 있으면 **다음 상태(next state)**와 **보상(reward)**을 예측 가능

**쉬운 이해:**
- 에이전트가 환경의 규칙을 머릿속에 표현한 **내적 시뮬레이터**
- "이 행동을 하면 다음에 어떤 상태가 될까?" 를 예측하는 능력

---

#### Model의 두 가지 구성 요소

**1. Transition Model (전이 모델):**

$$P^a_{ss'} = \Pr[ S_{t+1} = s' \mid S_t = s, A_t = a ]$$

**의미:** 상태 s에서 행동 a를 했을 때 상태 s'로 이동할 확률

**예시:**
- 미로에서 "오른쪽"을 선택하면 90% 확률로 실제로 오른쪽으로 이동
- 10% 확률로 미끄러져서 다른 방향으로 이동

**2. Reward Model (보상 모델):**

$$R^a_s = \mathbb{E}[ R_{t+1} \mid S_t = s, A_t = a ]$$

**의미:** 상태 s에서 행동 a를 했을 때 받을 평균 보상

**예시:**
- 상태 5에서 "앞으로" 행동하면 평균 +2점의 보상을 받음

---


## 💡 학습 팁

### 초보자를 위한 학습 순서

1. **개념 이해**
   - State, Action, Reward, Policy 개념 확실히 이해
   - MDP (Markov Decision Process) 개념 학습

2. **간단한 환경에서 실습**
   - Grid World, FrozenLake 같은 간단한 환경
   - Q-Learning으로 시작 (Q-Table 시각화 가능)

3. **심화 학습**
   - DQN (Deep Q-Network)
   - Policy Gradient 방법
   - Actor-Critic 방법

4. **복잡한 환경으로 확장**
   - Atari Games (DQN)
   - Continuous Control (PPO, SAC)
   - Multi-Agent RL

---

## 🔑 핵심 요약

### 강화학습 = 시행착오를 통한 학습

```
┌─────────────────────────────────────────────┐
│  목표: 최적 정책 π* 찾기                        │
│                                             │
│  방법:                                       │
│  1. 행동 → 보상 관찰                           │
│  2. 좋은 행동 → 더 자주 선택                     │
│  3. 나쁜 행동 → 덜 선택                         │
│  4. 반복하며 최적 정책 학습                       │
└─────────────────────────────────────────────┘
```

### Agent의 3대 요소

1. **Policy (π)**: 어떻게 행동할까? (전략)
2. **Value Function (V, Q)**: 이 상태/행동은 얼마나 좋을까? (평가)
3. **Model (P, R)**: 환경이 어떻게 반응할까? (예측)

---

## 모호한 것
1. 교수님 ppt2장 MRP, MDP 반영하기 -> 나머지 부분도 기존 readme.md에 반영하기


## 추가 예정.
0. Q-learning 내용 순서 알맞게 다시 작성해서 반영하기
1. MDP
2. Q-learning
3. DQN
4. PPO


---

**마지막 업데이트:** 2025-10-12

**작성자:** Taein Yong
