# Q-Learning & DQN (Deep Q-Network)

> Value-Based 강화학습의 핵심: Q-Learning에서 DQN까지

---

## 📚 목차

1. [Q-Learning 기본 개념](#-q-learning-기본-개념)
2. [Q-Learning의 한계점](#-q-learning의-한계점)
3. [DQN의 등장](#-dqn의-등장)
4. [Q-Learning vs DQN 비교](#-q-learning-vs-dqn-비교)
5. [실습 코드](#-실습-코드)

---

## 🎯 Q-Learning 기본 개념

### 정의

**Q-Learning**은 **Model-Free**, **Off-Policy**, **Value-Based** 강화학습 알고리즘입니다.

- **Model-Free**: 환경의 모델(전이 확률, 보상 함수)을 몰라도 학습 가능
- **Off-Policy**: 행동하는 정책과 학습하는 정책이 다를 수 있음
- **Value-Based**: Q-Function(Action-Value Function)을 학습

### Q-Table이란?

**Q-Table (Q-테이블)** 은 모든 **(상태, 행동) 쌍에 대한 Q값(기댓값)** 을 저장하는 표입니다.

**구조:**
- **행(Row)**: 각 상태 (State)
- **열(Column)**: 각 행동 (Action)  
- **셀 값**: Q(s, a) = 상태 s에서 행동 a를 했을 때의 **기대 누적 reward**

**수식:**
$$Q(s, a) = \mathbb{E}[R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots \mid S_t = s, A_t = a]$$

**의미:**
- 상태 s에서 행동 a를 선택한 후, 계속 정책을 따랐을 때 받을 **총 누적 보상의 기댓값**

**예시:**
```
Q(상태=3, 행동=오른쪽) = 0.8
→ "상태 3에서 오른쪽으로 가면 앞으로 평균 0.8의 보상을 받을 것"
```

### 핵심 아이디어

**"모든 (상태, 행동) 쌍의 가치를 테이블에 저장하고, 경험을 통해 업데이트한다"**

```
Q-Table:
         Action 0  Action 1  Action 2  Action 3
State 0    0.5       0.3       0.7       0.2
State 1    0.8       0.6       0.4       0.9
State 2    0.3       0.7       0.5       0.6
...
```

### Q-Learning 알고리즘

**1. Q-Table 초기화**
```python
Q(s,a) = 0  # 모든 상태-행동 쌍에 대해
```

**2. 에피소드 반복:**

```python
for episode in range(num_episodes):
    state = env.reset()
    
    while not done:
        # (1) ε-greedy로 행동 선택
        if random() < epsilon:
            action = random_action()  # 탐색 (Exploration)
        else:
            action = argmax(Q[state])  # 활용 (Exploitation)
        
        # (2) 행동 실행
        next_state, reward, done = env.step(action)
        
        # (3) Q-Table 업데이트 (핵심!)
        Q[state, action] = Q[state, action] + α * [
            reward + γ * max(Q[next_state]) - Q[state, action]
        ]
        
        state = next_state
```

### Bellman Optimality Equation 기반 업데이트

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ \underbrace{r + \gamma \max_{a'} Q(s',a')}_{\text{TD Target}} - Q(s,a) \right]$$

**각 항의 의미:**
- $Q(s,a)$ : 현재 Q값 (업데이트 대상)
- $\alpha$ : 학습률 (Learning Rate, 0 < α ≤ 1)
- $r$ : 즉각적 보상 (Immediate Reward)
- $\gamma$ : 할인율 (Discount Factor, 0 ≤ γ < 1)
- $\max_{a'} Q(s',a')$ : 다음 상태에서의 최대 Q값
- $\left[ \cdots \right]$ : **TD Error** (Temporal Difference Error)

**TD Error의 의미:**
- TD Target - 현재 추정치
- "예상보다 좋았으면" → 양수 → Q값 증가
- "예상보다 나빴으면" → 음수 → Q값 감소

### Q-Learning의 특징

#### ✅ 장점

1. **구현이 간단함**
   - Q-Table만 있으면 됨
   - 이해하기 쉬운 알고리즘

2. **Model-Free**
   - 환경의 전이 확률을 몰라도 학습 가능
   - 실제 경험만으로 학습

3. **Off-Policy**
   - 탐색과 활용을 분리 가능
   - 다른 정책의 경험으로도 학습 가능

4. **수렴 보장**
   - 충분한 탐색과 적절한 학습률 조건 하에
   - 최적 Q-Function으로 수렴 보장

#### 📊 동작 예시

**Grid World 예시:**
```
┌─────┬─────┬─────┬─────┐
│  S  │     │     │  G  │  S: Start, G: Goal
├─────┼─────┼─────┼─────┤
│     │  X  │     │     │  X: 장애물
├─────┼─────┼─────┼─────┤
│     │     │     │     │
└─────┴─────┴─────┴─────┘

Actions: ↑(0), ↓(1), ←(2), →(3)
Rewards: Goal=+1, 장애물=-1, 나머지=0
```

**학습 과정:**
```
Episode 1: S → → ↑ X (-1 보상) → Q-Table 업데이트
Episode 2: S → ↓ → → G (+1 보상) → Q-Table 업데이트
Episode 3: S → → → G (+1 보상) → Q-Table 업데이트
...
Episode 1000: 최적 경로 학습 완료!
```

---

## ⚠️ Q-Learning의 한계점

### 1️⃣ **상태 공간 폭발 (State Space Explosion)**

**문제:**
- Q-Table은 모든 (상태, 행동) 쌍을 저장해야 함
- 상태가 많아지면 메모리 사용량이 기하급수적으로 증가

**예시:**

| 환경 | 상태 수 | 행동 수 | Q-Table 크기 |
|------|---------|---------|--------------|
| Grid World (4×4) | 16 | 4 | 64 |
| Atari Pong (84×84 grayscale) | 256^(84×84) ≈ 10^14000 | 6 | 💥 불가능 |
| 바둑 (19×19) | 3^361 ≈ 10^172 | 361 | 💥 불가능 |

```python
# Grid World는 괜찮음
Q_table = np.zeros((16, 4))  # 64개 엔트리

# Atari는?
Q_table = np.zeros((256^7056, 6))  # 😱 메모리 부족!
```

### 2️⃣ **연속 상태 공간 처리 불가**

**문제:**
- Q-Table은 이산(discrete) 상태만 다룰 수 있음
- 연속(continuous) 상태는 무한개이므로 테이블로 표현 불가능

**예시:**
```python
# 로봇 팔 제어
state = [
    joint_angle_1,      # 0.0 ~ 360.0도 (연속)
    joint_angle_2,      # 0.0 ~ 360.0도 (연속)
    joint_velocity_1,   # -∞ ~ +∞ (연속)
    joint_velocity_2,   # -∞ ~ +∞ (연속)
]

# 이걸 Q-Table로? 😱
# → 무한한 조합, 메모리에 저장 불가능
```

**임시 해결책: Discretization (이산화)**
```python
# 각도를 10도 단위로 나누기
joint_angle_1 = round(angle / 10) * 10  # 0, 10, 20, ..., 360

# 문제:
# - 정밀도 손실 (30.5도 → 30도로 근사)
# - 여전히 조합 폭발 가능
```

### 3️⃣ **일반화 능력 부족 (No Generalization)**

**문제:**
- 비슷한 상태를 독립적으로 학습
- 경험을 공유하지 못함

**예시:**
```python
# Grid World에서
state_1 = (3, 4)  # Q값 학습됨
state_2 = (3, 5)  # 비슷한 상태인데 처음부터 학습해야 함

# 사람은 알 수 있음:
# "아, (3,4)랑 비슷하네? 비슷하게 행동하면 되겠구나!"
# 하지만 Q-Learning은 이걸 모름 😭
```

**이미지 예시:**
```
Atari Pong에서:
프레임 t:   공이 왼쪽 위에 있음 → Q값 학습
프레임 t+1: 공이 왼쪽 위에서 1픽셀 움직임
            → 완전히 다른 상태로 취급!
            → 처음부터 다시 학습 😱
```

### 4️⃣ **샘플 효율성 낮음**

**문제:**
- 모든 상태를 충분히 방문해야 함
- 학습에 많은 에피소드 필요

**예시:**
```python
# 16개 상태 × 4개 행동 = 64개 Q값
# 각각 충분히 업데이트하려면?
# → 수천 ~ 수만 에피소드 필요

# Atari라면?
# → 10^14000개 상태-행동 쌍... 
# → 우주의 나이보다 오래 걸림 💀
```

### 5️⃣ **고차원 입력 처리 불가**

**문제:**
- 이미지, 센서 데이터 등 복잡한 입력을 직접 사용할 수 없음
- Feature Engineering이 필요

**예시:**
```python
# Atari 게임 화면: 84×84×3 RGB 이미지
# Q-Learning: 이걸 어떻게 상태로? 🤔

# 방법 1: 픽셀 자체를 상태로?
state = image  # 256^(84×84×3) 가지 상태 → 불가능

# 방법 2: 수동으로 특징 추출?
state = [ball_x, ball_y, paddle_x, paddle_y]  # 힘듦, 한계 있음
```

---

## 🚀 DQN의 등장

### "Deep Q-Network: 딥러닝 + Q-Learning"

**핵심 아이디어:**
> **Q-Table을 신경망(Neural Network)으로 대체하자!**

```
Q-Learning:  Q-Table[state, action] → Q값
                      ↓
DQN:        Neural Network(state) → [Q(s,a₀), Q(s,a₁), ..., Q(s,aₙ)]
```

### DQN의 기본 구조

#### 1. Q-Network (신경망)

**입력:** 상태 (State)  
**출력:** 모든 행동의 Q값

```python
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)  # [Q(s,a₀), Q(s,a₁), ..., Q(s,aₙ)]
        return q_values
```

**예시:**
```python
state = [0.5, 0.3, 0.8, 0.2]  # 4차원 상태
q_values = dqn(state)         # [0.7, 0.3, 0.9, 0.4]
                              # ↑ 각 행동의 Q값
action = argmax(q_values)     # action = 2 (Q값이 0.9로 최대)
```

#### 2. Loss Function

**목표:** Neural Network가 Bellman Equation을 만족하도록 학습

$$\text{Loss} = \mathbb{E}\left[ \left( \underbrace{r + \gamma \max_{a'} Q(s', a'; \theta^-)}_{\text{TD Target}} - \underbrace{Q(s, a; \theta)}_{\text{Prediction}} \right)^2 \right]$$

**의미:**
- TD Target: "정답"에 해당 (Bellman Equation)
- Prediction: 현재 네트워크의 예측
- Loss: 둘의 차이를 최소화

```python
# TD Target 계산
with torch.no_grad():
    next_q_values = target_net(next_state)
    td_target = reward + gamma * torch.max(next_q_values)

# Current Q값
current_q = q_net(state)[action]

# Loss
loss = F.mse_loss(current_q, td_target)
```

### DQN의 혁신적 기법

#### 🧠 Experience Replay (경험 재생)

1️⃣ **문제 (Problem)**
- 에이전트가 환경과 상호작용하며 얻는 경험은 **시간적으로 연속되어 상관관계가 높음**
- 데이터를 순서대로 학습하면 **과적합(Overfitting)** 및 **학습 불안정성(Unstable Learning)** 발생
- 과거 경험(**희귀한 경험**)을 한 번만 사용하므로 **데이터 효율(Data Efficiency)** 이 낮음

---

2️⃣ **해결책 (Solution)**
- **Replay Buffer**(또는 Experience Replay)에 경험을 저장하여, 과거 데이터를 재사용  
    $(s_t, a_t, r_t, s_{t+1}, done)$
- 학습 시, 저장된 경험을 **무작위(mini-batch)** 로 샘플링 → **상관관계(correlation)** 감소  
- 이로써 **안정적이고 효율적인 학습** 가능



```python
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        # 경험을 저장할 버퍼 (최대 크기 지정)
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # 하나의 경험을 버퍼에 추가
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size=32):
        # 무작위로 경험 샘플링 → 시간적 상관관계 제거
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
```

**장점:**
- ✅ 샘플 효율성 향상 (Sample Efficiency) — 과거 경험을 재사용 가능
- ✅ 시간적 상관성 제거 (Decorrelation) — 안정적 학습 가능
- ✅ Off-policy 학습 지원 — 과거 정책으로 수집된 데이터도 학습에 활용 가능
- ✅ Offline Learning 가능 — 환경 없이 저장된 경험으로도 학습 가능

#### 2️⃣ **Target Network (타겟 네트워크)**

**문제:**
- TD Target을 계산할 때 같은 네트워크 사용
- Target이 계속 움직임 → "Moving Target Problem"
- 학습 불안정 (발산 가능)

```python
# 문제가 되는 경우:
td_target = reward + gamma * max(Q_net(next_state))  # Q_net이 계속 변함
loss = (td_target - Q_net(state)[action])^2          # 목표가 움직임!
```

**해결책:**
- **Target Network** 사용 (가중치 고정)
- 일정 주기마다만 업데이트

```python
# Q-Network (계속 학습됨)
q_net = DQN(state_dim, action_dim)

# Target Network (가끔 업데이트)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(q_net.state_dict())  # 복사

# 학습
for step in range(num_steps):
    # TD Target은 Target Network로 계산 (고정된 목표)
    td_target = reward + gamma * max(target_net(next_state))
    
    # Q-Network만 업데이트
    loss = (td_target - q_net(state)[action])^2
    loss.backward()
    optimizer.step()
    
    # C 스텝마다 Target Network 업데이트
    if step % C == 0:
        target_net.load_state_dict(q_net.state_dict())
```

**장점:**
- ✅ 학습 안정화 (고정된 목표)
- ✅ 발산 방지

#### 3️⃣ **Convolutional Neural Network (CNN)**

**이미지 입력 처리:**
- Atari 게임처럼 고차원 이미지 입력
- CNN으로 특징 자동 추출

```python
class DQN_CNN(nn.Module):
    def __init__(self, action_dim):
        super(DQN_CNN, self).__init__()
        # 입력: 84×84×4 (4프레임 stacked)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, action_dim)
    
    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values
```

### DQN 알고리즘 전체 흐름

```python
# 1. 초기화
q_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(q_net.state_dict())
replay_buffer = ReplayBuffer(capacity=10000)

# 2. 학습 루프
for episode in range(num_episodes):
    state = env.reset()
    
    for t in range(max_steps):
        # (1) ε-greedy로 행동 선택
        if random() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = q_net(state)
            action = torch.argmax(q_values)
        
        # (2) 행동 실행
        next_state, reward, done, _ = env.step(action)
        
        # (3) 경험 저장 (Experience Replay)
        replay_buffer.push(state, action, reward, next_state, done)
        
        # (4) 미니배치 샘플링 & 학습
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            
            # TD Target 계산 (Target Network)
            with torch.no_grad():
                next_q = target_net(next_states)
                td_target = rewards + gamma * torch.max(next_q, dim=1)[0]
            
            # Current Q값
            current_q = q_net(states).gather(1, actions)
            
            # Loss & 업데이트
            loss = F.mse_loss(current_q, td_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # (5) Target Network 업데이트
        if t % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())
        
        state = next_state
        if done:
            break
```

---

## 📊 Q-Learning vs DQN 비교

### 종합 비교표

| 특징 | Q-Learning | DQN |
|------|-----------|-----|
| **Q값 저장 방식** | Q-Table (딕셔너리/배열) | Neural Network |
| **상태 공간** | 이산, 작은 공간만 가능 | 연속, 고차원 가능 |
| **메모리 사용** | 상태 수에 비례 (기하급수) | 네트워크 크기 (고정) |
| **일반화 능력** | ❌ 없음 (각 상태 독립) | ✅ 있음 (비슷한 상태 공유) |
| **입력 타입** | 숫자 (이산 상태) | 이미지, 센서 데이터 등 |
| **학습 안정성** | ✅ 안정적 | ⚠️ 불안정 (기법 필요) |
| **샘플 효율성** | 낮음 | 중간 (Replay Buffer) |
| **수렴 보장** | ✅ 이론적 보장 | ⚠️ 보장 없음 (실험적) |
| **구현 난이도** | 쉬움 ⭐ | 어려움 ⭐⭐⭐ |
| **계산 비용** | 낮음 | 높음 (GPU 필요) |
| **대표 적용** | Toy Problems | Atari, 로봇 제어 |

### 상세 비교

#### 1. **상태 표현 (State Representation)**

**Q-Learning:**
```python
# 이산 상태만 가능
state = 5  # 상태 번호
Q_table[5][2]  # 상태 5에서 행동 2의 Q값
```

**DQN:**
```python
# 연속, 고차원 모두 가능
state = np.array([0.5, 0.3, 0.8])  # 연속 값
state = image  # 84×84×3 이미지
q_values = dqn_net(state)  # Neural Network로 처리
```

#### 2. **학습 방식**

**Q-Learning:**
```python
# 직접 업데이트 (테이블 값 변경)
Q[s, a] = Q[s, a] + α * (td_target - Q[s, a])
```

**DQN:**
```python
# Gradient Descent로 학습
loss = (td_target - q_net(s)[a])^2
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

#### 3. **메모리 사용량**

**Q-Learning:**
```python
# Grid World 10×10, 4 actions
Q_table = np.zeros((100, 4))  # 400 floats = 1.6 KB

# Atari (불가능)
Q_table = np.zeros((256^7056, 6))  # 💥 메모리 부족
```

**DQN:**
```python
# 상태 크기와 무관하게 네트워크 크기만 필요
dqn_net = DQN(state_dim, action_dim)
# 파라미터 수: 수만 ~ 수백만 개 (고정)
# 메모리: ~10MB (GPU VRAM 포함)
```

#### 4. **일반화 능력**

**Q-Learning:**
```python
# 상태 (3, 4)를 학습했어도
Q[3, 4][0] = 0.8  # 학습됨

# 상태 (3, 5)는 처음부터
Q[3, 5][0] = 0.0  # 초기값 그대로
```

**DQN:**
```python
# 비슷한 상태는 비슷한 Q값
state_1 = [3, 4, ...]
state_2 = [3, 5, ...]  # 비슷한 입력

q_net(state_1)  # [0.8, 0.3, ...]
q_net(state_2)  # [0.75, 0.35, ...]  # 비슷한 출력!
```




---

## 💡 핵심 요약

### Q-Learning → DQN 진화 과정

```
Q-Learning의 한계:
├─ ❌ 상태 공간 폭발
├─ ❌ 연속 상태 처리 불가
├─ ❌ 일반화 능력 부족
└─ ❌ 고차원 입력 처리 불가
         ↓
    "신경망으로 해결하자!"
         ↓
DQN의 혁신:
├─ ✅ Q-Table → Neural Network
├─ ✅ Experience Replay
├─ ✅ Target Network
└─ ✅ CNN (이미지 처리)
```

### 핵심 메시지

> **Q-Learning은 강화학습의 기초이자 본질을 담고 있습니다.**  
> **DQN은 Q-Learning을 실제 복잡한 문제에 적용 가능하게 만든 혁신입니다.**

---

## 🔗 실습 코드

이 폴더의 Jupyter Notebook 파일들을 참고하세요:

- `1.Q-Learning.ipynb`: Q-Learning 기초 구현
- `2.DQN.ipynb`: DQN 구현 및 Atari 적용

---


**작성일:** 2025-10-23
**작성자:** Taein Yong


