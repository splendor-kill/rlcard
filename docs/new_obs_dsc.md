# OBS 设计意见&建议

在 new_obs.pdf 中，新的 observation 设计已经远超原始的 rlcard 方案，主要提升点包括：

* 加入玩家状态信息（如玩家人数、当前行动玩家）

* 增加筹码信息（所有玩家的投入）

* 增强动作空间建模（合法动作 mask）

* 记录完整打牌过程（以 (4,5,9,5) 的张量存储）

## 进一步优化分析
针对该观测空间，我们可以从表示能力、压缩优化、多智能体适应性 3 个方面进一步优化。

1. 表示能力优化

**问题：**

* 当前设计 (984 维) 仍然包含 大量 0 值填充，尤其是 动作序列 (4,5,9,5)

* 手牌信息 (0..51) 是 one-hot 方式，如果用 embedding 方式能降低维度。

**优化建议：**

- 稀疏存储动作序列：使用 player_id, stage, action_value 组合，而不是固定 4×5×9×5 格式。

- 手牌 embedding：将 0..51 映射到一个 d=16 的低维空间，减少维度：

  ```python
  hand_embedding = nn.Embedding(52, 16)
  hand_feature = hand_embedding(hand_index)
  ```

* 相对筹码归一化：用 当前筹码 / 最大筹码 而不是原始 BB 单位表示。



2. 压缩优化

**问题：**

* 目前是 984 维 1D 向量，但很多信息是高维结构化数据，比如 (4,5,9,5) 是时间序列，可以用 Transformer 处理，而不是展平。

**优化建议：**

- 变长存储 (Variable-Length Encoding)：只存 非零动作 而不是填充 900 维数据。

- 阶段拆分 (Stage-Split)：

  将 (preflop, flop, turn, river) 各阶段分离，而不是统一展平到 1 维数组。

  例如，分成 (4, action_dim)，不同阶段分别编码：

  ```python
  stage_embed = nn.Embedding(4, 8)
  stage_feature = stage_embed(stage_index)
  ```

* 时序处理 (LSTM/Transformer)：对打牌过程 (900 维) 用 Transformer，而不是 simple MLP。



3. 多智能体适应性

**问题：**

* 目前 obs 主要是单玩家视角，但 MARL 训练中，需要建模 对手策略。

**优化建议：**

* 共享公共信息（如公共牌、底池、局势阶段）。

* 每个玩家有独立观察空间（手牌、筹码）。

* 增加对手建模 (Opponent Modeling)，引入对手的历史行为：

  ```python
  opponent_action_hist = LSTM(past_actions)
  ```

## 伪码

```python
import numpy as np
import torch
import torch.nn as nn

class PokerObservation:
    def __init__(self, num_players=6):
        self.num_players = num_players
        self.action_dim = 5  # 5种合法动作
        self.stage_dim = 4   # 4个阶段 (PREFLOP, FLOP, TURN, RIVER)
            # Embedding for hand representation
    self.hand_embedding = nn.Embedding(52, 16)
    self.stage_embedding = nn.Embedding(self.stage_dim, 8)

    # Opponent modeling (LSTM for action history)
    self.opponent_modeling = nn.LSTM(input_size=self.action_dim, hidden_size=32, batch_first=True)

def encode_observation(self, player, public_cards, all_players):
    """
    Encode observation for a given player in a multi-agent RL environment.
    """

    # 1. 手牌 + 公共牌 (Embedding)
    hand_feature = self.hand_embedding(torch.tensor(player.hand, dtype=torch.long))  # (2, 16)
    public_feature = self.hand_embedding(torch.tensor(public_cards, dtype=torch.long))  # (5, 16)

    # 2. 玩家状态信息
    num_players_mask = np.zeros(self.num_players)
    num_players_mask[:len(all_players)] = 1  # 例如6人局，mask=[1,1,1,1,1,1,0,0,0]

    current_player_mask = np.zeros(self.num_players)
    current_player_mask[player.id] = 1  # 当前玩家的位置

    # 3. 筹码信息
    chip_bets = np.array([p.in_chips for p in all_players]) / 100  # 归一化筹码信息
    stack_sizes = np.array([p.remained_chips for p in all_players]) / 100

    # 4. 处理合法动作
    legal_actions_mask = np.zeros(self.action_dim)
    for action in player.legal_actions:
        legal_actions_mask[action] = 1

    # 5. 动作历史 (用 LSTM 进行建模)
    action_history = np.zeros((self.stage_dim, self.num_players, self.action_dim))  # (4, 6, 5)
    for s, stage in enumerate(player.history):  # 遍历每个阶段
        for p, past_action in enumerate(stage.actions):  # 遍历每个玩家的历史行动
            action_history[s, p, past_action] = 1  # 记录玩家的动作
    
    # LSTM 处理对手行为
    opponent_action_hist = torch.tensor(action_history, dtype=torch.float32).unsqueeze(0)  # (1, 4, 6, 5)
    _, (opponent_feature, _) = self.opponent_modeling(opponent_action_hist)  # (1, 32)

    # 6. 拼接所有特征
    observation = np.concatenate([
        hand_feature.flatten(),
        public_feature.flatten(),
        num_players_mask,
        current_player_mask,
        chip_bets,
        stack_sizes,
        legal_actions_mask,
        opponent_feature.detach().numpy().flatten()  # 对手建模特征
    ])

    return observation
```




## 改进点总结
✅ 减少无效填充

采用 紧凑存储 取代 4x5x9x5 的固定格式。

✅ 采用 Embedding 代替 One-Hot

让手牌 & 阶段信息变为可学习的低维特征，减少维度。

✅ 使用 LSTM 处理对手历史信息

让智能体能预测对手的策略，提高对抗性。

✅ 采用 共享+个体 信息分离

共享底池、公共牌，个体特征包括筹码、手牌。