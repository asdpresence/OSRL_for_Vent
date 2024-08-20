from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from pyrallis import field


@dataclass
class BCTrainConfig:
    # wandb params
    project: str = "OSRL_for_Vent"
    group: str = "BC"
    name: Optional[str] = None
    prefix: Optional[str] = "BC"
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
    actor_lr: float = 0.001
    cost_limit: float = 0.10
    episode_len: int = 300
    batch_size: int = 512
    update_steps: int = 50_000
    num_workers: int = 8
    bc_mode: str = "all"
    # model params
    a_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    gamma: float = 1.0
    # evaluation params
    eval_episodes: int = 1000
    eval_every: int = 1000


@dataclass
class BCmimic3Config(BCTrainConfig):
    # training params
    task: str = "mimic3"
    episode_len: int = 18



BC_DEFAULT_CONFIG = {
    "mimic3":BCmimic3Config
}