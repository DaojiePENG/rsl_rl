#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    '''
    这个类封装了PyTorch神经网络相关的操作。包括：
    定义神经网络形状；
    指定激活函数类型；

    '''
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions)) # 最后一级网络将输出形状映射到跟动作所需驱动节点维度相同；
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation) # 在最后一级之前操作为将上一层网络的输出形状映射到下一网络的输出形状，并添加一个激活层；
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = [] # 不太懂这个 critic_layer 惩罚层是什么意思，是在模仿学习的时候会用到的吗?
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        '''
        获取动作分布的熵，
        '''
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        '''
        这个函数是调用神经网络更新下一步动作的函数。
        它调用神经网络计算出了平均的动作，然后将动作结果添加正态分布的噪声。
        这个结果会在 PPO 中使用到。
        这就是我一直想找到的在神经网络优化过程所调用神经网络计算动作输出的地方。

        注：这里只是生成了一个动作的正态分布函数定义，具体作用到节点上的值需要经过下面的 act 函数采样得到；
        '''
        mean = self.actor(observations) # 这里调用了神经网络计算除了平均的动作
        self.distribution = Normal(mean, mean * 0.0 + self.std) # 这里用神经网络输出的动作结果添加了正态分布的噪声；

    def act(self, observations, **kwargs):
        '''
        每当 act 的时候就用 actor 神经网络来更新一下 distribution 即有正态分布噪声的生成动作。

        简单来说就是采样正态分布的某个值作为动作命令。
        也就是说添加噪声只需要控制正态分布的 std 变量就可以了。
        '''
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations): # 输入 observations 是观测结果；输出是经过神经网络处理后的动作；
        actions_mean = self.actor(observations) # 这个就是将神经网络的调用封装了一层接口，没做什么别的操作；
        return actions_mean # 以后在这个库里，调用 act_inference 就相当于调用了模型计算最终输出结果；

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
