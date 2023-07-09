from typing import Optional, List, Dict
from dataclasses import dataclass, field

@dataclass
class RunningArguments:
    """
    Arguments for running this project.
    """
    r_type: str = field(
        default=None,
        metadata={"help": "Retrieval model type: sparse / dense"},
    )
    s_index: str = field(
        default=None,
        metadata={"help": "Path to sparse indexes"}
    )
    d_index: Optional[str] = field(
        default=None, metadata={"help": "Path to dense indexes"}
    )
    reader_path: str = field(
        default=None, metadata={"help": "Path to reader model (sentence transformer)"}
    )
