import os
import uuid
import types
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import bullet_safety_gym  # noqa
import dsrl
import gymnasium as gym  # noqa
import numpy as np
import pyrallis
import torch
from dsrl.infos import DENSITY_CFG
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from fsrl.utils import WandbLogger
from torch.utils.data import DataLoader
from tqdm.auto import trange  # noqa
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from examples.configs.cql_configs import CQL_DEFAULT_CONFIG, CQLTrainConfig
from myosrl.algorithms import CQL, CQLTrainer
from myosrl.common import Mimic3_Vent_Dataset
from osrl.common.exp_util import auto_name, seed_all


@pyrallis.wrap()
def train(args: CQLTrainConfig):
    # Update config
    global gym
    print("Gym version:", gym.__version__)
    cfg, old_cfg = asdict(args), asdict(CQLTrainConfig())
    differing_values = {key: cfg[key] for key in cfg.keys() if cfg[key] != old_cfg[key]}
    cfg = asdict(CQL_DEFAULT_CONFIG[args.task]())
    cfg.update(differing_values)
    args = types.SimpleNamespace(**cfg)

    # Setup logger
    default_cfg = asdict(CQL_DEFAULT_CONFIG[args.task]())
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = args.task + "-cost-" + str(int(args.cost_limit))
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.group, args.name)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    logger.save_config(cfg, verbose=args.verbose)

    # Set seed
    seed_all(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    # Load the train data dictionary
    current_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data'))
    pkl_file = os.path.join(data_dir, 'processed_mimic3_data.pkl')
    with open(pkl_file, 'rb') as file:
        df = pd.read_pickle(pkl_file)
    data = df

    # Load the evaluate trajectory data list
    pkl_file = os.path.join(data_dir, 'processed_mimic3_episodes.pkl')
    with open(pkl_file, 'rb') as file:
        df_episodes = pd.read_pickle(pkl_file)

    if args.density != 1.0:
        density_cfg = DENSITY_CFG[args.task + "_density" + str(args.density)]

    # Model & optimizer setup
    model = CQL(
        state_dim=38,
        action_dim=3,
        max_action=1.0,
        a_hidden_sizes=args.a_hidden_sizes,
        c_hidden_sizes=args.c_hidden_sizes,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        num_q=args.num_q,
        device=args.device,
    )
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    def checkpoint_fn():
        return {"model_state": model.state_dict()}

    logger.setup_checkpoint_fn(checkpoint_fn)

    trainer = CQLTrainer(model,
                         logger=logger,
                         actor_lr=args.actor_lr,
                         critic_lr=args.critic_lr,
                         device=args.device)

    # Initialize PyTorch DataLoader
    dataset = Mimic3_Vent_Dataset(data,
                                  reward_scale=args.reward_scale,
                                  cost_scale=args.cost_scale)
    trainloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    trainloader_iter = iter(trainloader)

    # For saving the best
    best_reward = -np.inf
    best_cost = np.inf
    best_idx = 0

    # Training
    for step in trange(args.update_steps, desc="Training"):
        batch = next(trainloader_iter)

        observations, next_observations, actions, rewards, done = [
            b.to(args.device) for b in batch
        ]
        trainer.train_one_step(observations, next_observations, actions, rewards, done)

        # Evaluation
        if (step + 1) % args.eval_every == 0 or step == args.update_steps - 1:
            ret, cost, length = trainer.evaluate(args.eval_episodes, df_episodes)

            average_reward = ret / length
            logger.store(tab="eval", Cost=cost, Average_Qvalue=average_reward, Length=length)

            # Save the current weight
            logger.save_checkpoint()
            # Save the best weight
            if cost < best_cost or (cost == best_cost and ret > best_reward):
                best_cost = cost
                best_reward = ret
                best_idx = step
                logger.save_checkpoint(suffix="best")

            logger.store(tab="train", best_idx=best_idx)
            logger.write(step, display=False)
        else:
            logger.write_without_reset(step)


if __name__ == "__main__":
    train()
