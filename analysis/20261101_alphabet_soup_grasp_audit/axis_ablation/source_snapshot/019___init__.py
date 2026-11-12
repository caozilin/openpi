"""Block optimization for optional receding-horizon preset execution."""

from .controller import BlockSQPIK
from .objective import BlockObjective
from .types import BlockSQPDiagnostics, BlockSQPSettings, BlockStageReference

__all__ = (
    "BLOCK_SQP_MAINTAINED",
    "BlockObjective",
    "BlockSQPDiagnostics",
    "BlockSQPIK",
    "BlockSQPSettings",
    "BlockStageReference",
)

BLOCK_SQP_MAINTAINED = True
