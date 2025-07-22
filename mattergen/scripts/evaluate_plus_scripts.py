# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from pathlib import Path
from typing import Literal

import fire
import numpy as np

from mattergen.common.utils.eval_utils import load_structures
from mattergen.common.utils.globals import get_device
from mattergen.evaluation.evaluate import evaluate
from mattergen.evaluation.reference.reference_dataset_serializer import LMDBGZSerializer
from mattergen.evaluation.utils.structure_matcher import (
    DefaultDisorderedStructureMatcher,
    DefaultOrderedStructureMatcher,
)


def main(
    structures_path: str,
    relax: bool = True,
    energies_path: str | None = None,
    structure_matcher: Literal["ordered", "disordered"] = "disordered",
    save_as: str | None = None,
    per_structure_save_as: str | None = None, # 添加此行
    potential_load_path: (
        Literal["MatterSim-v1.0.0-1M.pth", "MatterSim-v1.0.0-5M.pth"] | None
    ) = None,
    reference_dataset_path: str | None = None,
    device: str = str(get_device()),
    structures_output_path: str | None = None,
):
    structures = load_structures(Path(structures_path))
    energies = np.load(energies_path) if energies_path else None
    structure_matcher = (
        DefaultDisorderedStructureMatcher()
        if structure_matcher == "disordered"
        else DefaultOrderedStructureMatcher()
    )
    reference = None
    if reference_dataset_path:
        reference = LMDBGZSerializer().deserialize(reference_dataset_path)

    per_structure_metrics, summary_metrics = evaluate(
        structures=structures,
        relax=relax,
        energies=energies,
        structure_matcher=structure_matcher,
        potential_load_path=potential_load_path,
        reference=reference,
        device=device,
        structures_output_path=structures_output_path,
    )
    if save_as:
        with open(save_as, "w") as f:
            json.dump(summary_metrics, f, indent=2)

    # 如果指定了 per_structure_save_as 路径，则保存每个结构的指标
    if per_structure_save_as:
        with open(per_structure_save_as, "w") as f:
            json.dump(per_structure_metrics, f, indent=4)

    print(json.dumps(summary_metrics, indent=2)) # 保持打印汇总指标


def _main():
    fire.Fire(main)


if __name__ == "__main__":
    _main()
