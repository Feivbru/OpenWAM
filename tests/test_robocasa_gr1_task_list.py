from __future__ import annotations

from benchmarks.robocasa_gr1.task_list import OFFICIAL_ENV_IDS, split_official_env_ids


def test_split_official_env_ids_two_groups_are_contiguous_halves():
    groups = split_official_env_ids(2)
    assert len(groups) == 2
    assert len(groups[0]) == 12
    assert len(groups[1]) == 12
    assert groups[0] + groups[1] == OFFICIAL_ENV_IDS
    assert len(set(groups[0]) & set(groups[1])) == 0
