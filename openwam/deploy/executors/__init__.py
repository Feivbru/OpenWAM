"""Inference executors: how engine-generated action chunks reach the robot.

Interchangeable mechanisms behind one interface
(``predict_action(conditions)`` / ``reset()`` / ``shutdown()``):
:class:`SyncInferenceExecutor` (default; buffer-and-replan),
:class:`AsyncInferenceExecutor` (threaded prefetch), and
:class:`VrtcSyncExecutor` (cube-pool VRTC, when ``vrtc.enabled``).
"""

from openwam.deploy.executors.async_executor import (
    EXECUTION_CLI_NUMERIC_OVERRIDES,
    AsyncInferenceExecutor,
    ExecutionConfig,
    apply_execution_cli_overrides,
    normalize_execution_config,
    resolve_execution_config,
)
from openwam.deploy.executors.sync_executor import SyncInferenceExecutor
from openwam.deploy.executors.vrtc_executor import VrtcSyncExecutor, build_vrtc_executor

__all__ = [
    "SyncInferenceExecutor",
    "AsyncInferenceExecutor",
    "VrtcSyncExecutor",
    "build_vrtc_executor",
    "ExecutionConfig",
    "EXECUTION_CLI_NUMERIC_OVERRIDES",
    "apply_execution_cli_overrides",
    "normalize_execution_config",
    "resolve_execution_config",
]
