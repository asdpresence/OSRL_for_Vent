from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from pyrallis import field


@dataclass
class BCQLTrainConfig:
    # wandb params
    project: str = "OSRL_for_Vent"
    group: str = "BCQL"
    name: Optional[str] = None
    prefix: Optional[str] = "BCQL"
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
    # reward_scale: float = 0.1
    reward_scale: float = 0.1
    cost_scale: float = 1

    # actor_lr: float = 0.001
    # critic_lr: float = 0.001
    actor_lr: float = 0.00001
    critic_lr: float = 0.00001

    vae_lr: float = 0.0001
    # vae_lr: float = 0.00001
    phi: float = 0.05
    lmbda: float = 0.75
    beta: float = 0.5

    cost_limit: float = 0.10
    # 修改
    episode_len: int = 18
    # batch_size: int = 64
    batch_size: int = 512
    # update_steps: int = 50_000
    update_steps: int = 10_000
    num_workers: int = 16
    # model params
    a_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    c_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    vae_hidden_sizes: int = 400
    sample_action_num: int = 10
    gamma: float = 0.99
    tau: float = 0.005
    num_q: int = 2
    num_qc: int = 2
    PID: List[float] = field(default=[0.1, 0.003, 0.001], is_mutable=True)
    # evaluation params
    eval_episodes: int = 1000
    eval_every: int = 1000



@dataclass
class BCQLmimic3Config(BCQLTrainConfig):
    # training params
    task: str = "mimic3"
    episode_len: int = 18
    # update_steps: int = 200_000

BCQL_DEFAULT_CONFIG = {

    "mimic3":BCQLmimic3Config
}