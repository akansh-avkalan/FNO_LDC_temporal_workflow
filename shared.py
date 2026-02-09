from dataclasses import dataclass
from typing import Tuple, Optional

TASK_QUEUE = "fno_pipeline_queue"

@dataclass
class DataSetConfig:
    repo_id: str
    folder: str
    output_dir: str


@dataclass
class TrainConfig: 
    dataset_path : str
    batch_size : int
    train_ratio : float
    val_ratio : float
    test_ratio : float
    seed : int
    num_workers : int
    fno_width : int   
    modes1 : int     
    modes2 : int
    in_channels : int
    out_channels : int
    epochs: int
    log_dir : str
    checkpoint_frequency : int
    learning_rate : float 
    
    
@dataclass
class EvaluateConfig: 
    model_path: str
    train_config : TrainConfig


@dataclass 
class MLPipelineInput:
    dataset_config : DataSetConfig
    train_config : TrainConfig
    evaluate_config : Optional[EvaluateConfig] = None  # ✅ Made optional with default None
