from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from pyrallis import field


@dataclass
class CQLTrainConfig:
    # wandb params
    project: str = "OSRL_for_Vent"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "CQL"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs"
    verbose: bool = True
    # dataset params
    outliers_percent: float = None
    noise_scale: float = None
    inpaint_ranges: Tuple[Tuple[float, float, float, float], ...] = None
    epsilon: float = None
    density: float = 1.0
    # training params
    task: str = "mimic3"
    dataset: str = None
    seed: int = 0
    device: str = "cpu"
    threads: int = 4
    reward_scale: float = 0.1
    cost_scale: float = 1
    # actor_lr: float = 0.0001
    # critic_lr: float = 0.001

    # actor_lr: float = 0.00001
    # critic_lr: float = 0.00001

    actor_lr: float = 0.000001
    critic_lr: float = 0.00001

    alpha_lr: float = 0.0001
    vae_lr: float = 0.001
    # alpha_lr: float = 0.000001
    # vae_lr: float = 0.00001

    # cost_limit: int = 10
    cost_limit: float = 0.10
    episode_len: int = 10
    # batch_size: int = 64
    batch_size: int = 512
    update_steps: int = 35_000
    num_workers: int = 8
    # model params
    a_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    c_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    vae_hidden_sizes: int = 800
    sample_action_num: int = 10
    gamma: float = 0.99
    # gamma: float = 0.95
    tau: float = 0.005
    # tau: float = 0.005

    beta: float = 0.1
    num_q: int = 2
    num_qc: int = 2
    qc_scalar: float = 1.5
    # evaluation params
    eval_episodes: int = 1000
    # eval_every: int = 2500
    eval_every: int = 1000


@dataclass
class CQLmimic3Config(CQLTrainConfig):
    # training params
    task: str = "mimic3"
    episode_len: int = 18
    update_steps: int = 50_000


CQL_DEFAULT_CONFIG = {
"mimic3":CQLmimic3Config
}