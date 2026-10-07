"""Official 24 RoboCasa GR1 tabletop env ids for OpenWAM evaluation.

Source: robocasa-gr1-tabletop-tasks README (Isaac-GR00T simulation_service list).
Gym may register many more ``gr1_unified/*`` variants; evaluation uses only these.
"""

from __future__ import annotations

OFFICIAL_ENV_IDS: tuple[str, ...] = (
    "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env",
)


def env_short_name(env_id: str) -> str:
    """Strip suite prefix / Env suffix for log filenames."""
    name = env_id
    if "/" in name:
        name = name.split("/", 1)[1]
    if name.endswith("_Env"):
        name = name[: -len("_Env")]
    # Drop the long embodiment suffix for readable paths.
    marker = "_GR1ArmsAndWaistFourierHands"
    if marker in name:
        name = name.split(marker, 1)[0]
    return name


def smoke_env_ids() -> tuple[str, ...]:
    """1–2 tasks for multi-card topology smoke."""
    return OFFICIAL_ENV_IDS[:2]


def split_official_env_ids(n_groups: int = 2) -> list[tuple[str, ...]]:
    """Partition the 24 official tasks into ``n_groups`` contiguous chunks."""
    if n_groups < 1:
        raise ValueError(f"n_groups must be >= 1, got {n_groups}")
    ids = list(OFFICIAL_ENV_IDS)
    n = len(ids)
    base, rem = divmod(n, n_groups)
    groups: list[tuple[str, ...]] = []
    cursor = 0
    for i in range(n_groups):
        cnt = base + (1 if i < rem else 0)
        groups.append(tuple(ids[cursor : cursor + cnt]))
        cursor += cnt
    return groups


__all__ = ["OFFICIAL_ENV_IDS", "env_short_name", "smoke_env_ids", "split_official_env_ids"]
