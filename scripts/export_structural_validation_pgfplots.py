#!/usr/bin/env python3
"""Export compact, numeric PGFPlots tables for structural validation."""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "ours"
OUTPUT = ROOT / "paper" / "Figure" / "structural_validation_data"
BACKBONES = {
    "Llama-3.2-3B": ("3b", 28),
    "Llama-3.1-8B": ("8b", 32),
}


def read_rows(name: str) -> list[dict[str, str]]:
    with (SOURCE / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(
    name: str,
    fieldnames: list[str],
    rows: list[dict[str, object]],
    *,
    line_terminator: str | None = None,
) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / name).open("w", encoding="utf-8", newline="") as handle:
        kwargs = {} if line_terminator is None else {"lineterminator": line_terminator}
        writer = csv.DictWriter(handle, fieldnames=fieldnames, **kwargs)
        writer.writeheader()
        writer.writerows(rows)


def write_group_points(name: str, color: str, rows: list[dict[str, object]]) -> None:
    """Write TikZ point commands so marker sizes remain data-driven."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / name).open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("% Generated from data/ours/group_analysis.csv; do not edit.\n")
        for row in rows:
            handle.write(
                "\\StructuralGroupPoint"
                f"{{{color}}}{{{row['marker_size']}}}"
                f"{{{row['x']}}}{{{row['y']}}}\n"
            )


def export_pairs() -> None:
    rows = read_rows("pair_analysis.csv")
    for backbone, (short_name, _) in BACKBONES.items():
        selected = [
            {"x": row["C_i_to_j"], "y": row["C_j_to_i"]}
            for row in rows
            if row["backbone"] == backbone
        ]
        if not selected:
            raise RuntimeError(f"pair_analysis.csv has no rows for {backbone}")
        write_rows(f"pair_{short_name}.csv", ["x", "y"], selected)


def export_groups() -> None:
    rows = read_rows("group_analysis.csv")
    for backbone, (short_name, _) in BACKBONES.items():
        selected = []
        for row in rows:
            if row["backbone"] != backbone:
                continue
            marker_size = 0.5 * math.sqrt(8.0 * float(row["group_size"]))
            selected.append(
                {
                    "x": row["Delta"],
                    "y": row["delta"],
                    "marker_size": f"{marker_size:.6f}",
                }
            )
        if not selected:
            raise RuntimeError(f"group_analysis.csv has no rows for {backbone}")
        write_rows(
            f"group_{short_name}.csv",
            ["x", "y", "marker_size"],
            selected,
        )
        write_group_points(
            f"group_{short_name}_points.tex",
            "threeblue" if short_name == "3b" else "eightred",
            selected,
        )


def export_joint() -> None:
    rows = read_rows("joint_analysis.csv")
    for backbone, (short_name, layer_count) in BACKBONES.items():
        selected = []
        for row in sorted(
            (item for item in rows if item["backbone"] == backbone),
            key=lambda item: int(item["nominal_target"]),
        ):
            target = int(row["nominal_target"])
            x_value = float(row["Delta_max"])
            y_value = float(row["C_joint"]) / layer_count
            offsets = {
                ("3b", 15): (0.030, -0.035),
                ("3b", 20): (0.030, -0.035),
                ("3b", 25): (0.030, 0.030),
                ("8b", 15): (-0.035, 0.035),
                ("8b", 20): (-0.035, 0.030),
                ("8b", 25): (-0.035, 0.030),
            }
            label_dx, label_dy = offsets[(short_name, target)]
            marker_area = 260.0 * float(row["Pure_drop"]) + 18.0
            selected.append(
                {
                    "x": f"{x_value:.12g}",
                    "y": f"{y_value:.12g}",
                    "marker_size": f"{0.5 * math.sqrt(marker_area):.6f}",
                    "label_x": f"{x_value + label_dx:.12g}",
                    "label_y": f"{y_value + label_dy:.12g}",
                    "label": f"{target}\\%",
                }
            )
        if len(selected) != 3:
            raise RuntimeError(
                f"joint_analysis.csv expected three operating points for {backbone}"
            )
        write_rows(
            f"joint_{short_name}.csv",
            ["x", "y", "marker_size", "label_x", "label_y", "label"],
            selected,
            line_terminator="\n",
        )


def main() -> None:
    export_pairs()
    export_groups()
    export_joint()
    print(f"wrote PGFPlots data to {OUTPUT}")


if __name__ == "__main__":
    main()
