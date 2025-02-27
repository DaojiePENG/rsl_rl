# Copyright (c) 2021-2025, Daojie Peng, HKUST(GZ)
# All rights reserved.
#
# email: Daojie.Peng@qq.com

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl_lm.utils import resolve_nn_activation


class ActorTransformer(nn.Module):
    def __init__(self, num_actor_obs, d_actor, nhead_actor, num_layers_actor, dim_feedforward_actor, dropout_actor, activation, num_actions):
        super().__init__()
        self.num_actor_obs = num_actor_obs
        self.input_embedding = nn.Linear(num_actor_obs, d_actor)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_actor, nhead=nhead_actor, dim_feedforward=dim_feedforward_actor, dropout=dropout_actor, activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers_actor)
        self.actor_output_layer = nn.Linear(d_actor, num_actions)

    def forward(self, observations):
        embedded_obs = self.input_embedding(observations.unsqueeze(1))
        transformer_output = self.transformer_encoder(embedded_obs)
        actions = self.actor_output_layer(transformer_output.squeeze(1))
        return actions

class CriticTransformer(nn.Module):
    def __init__(self, num_critic_obs, d_critic, nhead_critic, num_layers_critic, dim_feedforward_critic, dropout_critic, activation):
        super().__init__()
        self.num_critic_obs = num_critic_obs
        self.input_embedding = nn.Linear(num_critic_obs, d_critic)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_critic, nhead=nhead_critic, dim_feedforward=dim_feedforward_critic, dropout=dropout_critic, activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers_critic)
        self.critic_output_layer = nn.Linear(d_critic, 1)

    def forward(self, observations):
        embedded_obs = self.input_embedding(observations.unsqueeze(1))
        transformer_output = self.transformer_encoder(embedded_obs)
        value = self.critic_output_layer(transformer_output.squeeze(1))
        return value

class ActorCriticTransformerV0(nn.Module):
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
        d_actor=16,
        nhead_actor=4,
        num_layers_actor=2,
        dim_feedforward_actor=16,
        dropout_actor=0.1,
        d_critic=16,
        nhead_critic=4,
        num_layers_critic=2,
        dim_feedforward_critic=16,
        dropout_critic=0.1,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        # Policy (Actor)
        self.actor = ActorTransformer(num_actor_obs, d_actor, nhead_actor, num_layers_actor, dim_feedforward_actor, dropout_actor, activation, num_actions)

        # Value function (Critic)
        self.critic = CriticTransformer(num_critic_obs, d_critic, nhead_critic, num_layers_critic, dim_feedforward_critic, dropout_critic, activation)

        print(f"Actor TransformerV0: {self.actor}")
        print(f"Critic TransformerV0: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

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
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        return self.actor(observations)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)
    

