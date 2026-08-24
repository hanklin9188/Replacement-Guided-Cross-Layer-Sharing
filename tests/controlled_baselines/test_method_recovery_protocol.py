from pathlib import Path

import torch

from icassp27.controlled_baselines.method_recovery import (
    _factor_parameter_names,
    _lr_at_step,
    _set_basis_full_trainable,
    _set_svd_factor_trainable,
    load_method_config,
)
from icassp27.controlled_baselines.modeling import FactorizedLinear


class TinyFactors(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(4, 4)
        self.first = FactorizedLinear(4, 5, 2)
        self.block = torch.nn.ModuleDict({"second": FactorizedLinear(5, 3, 2)})


def test_basis_full_ft_activates_every_existing_parameter():
    model = TinyFactors()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    names = _set_basis_full_trainable(model)
    assert set(names) == {name for name, _ in model.named_parameters()}
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_svd_stages_use_exact_u_then_v_factors_only():
    model = TinyFactors()
    all_factors = _factor_parameter_names(model)
    assert all(".u_proj." in name or ".v_proj." in name for name in all_factors)
    u_names = _set_svd_factor_trainable(model, "u")
    assert u_names and all(".u_proj." in name for name in u_names)
    assert {name for name, parameter in model.named_parameters() if parameter.requires_grad} == set(u_names)
    v_names = _set_svd_factor_trainable(model, "v")
    assert v_names and all(".v_proj." in name for name in v_names)
    assert {name for name, parameter in model.named_parameters() if parameter.requires_grad} == set(v_names)
    assert set(u_names).isdisjoint(v_names)


def test_continuous_warmup_cosine_schedule():
    assert _lr_at_step(1.0, 1, 100, 10, 0.01) == 0.1
    assert _lr_at_step(1.0, 10, 100, 10, 0.01) == 1.0
    assert _lr_at_step(1.0, 100, 100, 10, 0.01) == 0.01
    assert _lr_at_step(1.0, 50, 100, 10, 0.01) < 1.0


def test_locked_method_config_is_four_h200_and_decision_ce():
    source = Path(__file__).resolve().parents[2] / "configs/method_recovery_4h200.example.yaml"
    cfg = load_method_config(source)
    assert cfg["slurm"]["gpus_per_job"] == 4
    assert cfg["recovery"]["loss_scope"] == "decision"
    assert cfg["recovery"]["selection_metric"] == "decision_ce"
    assert cfg["recovery"]["svd_stage_order"] == ["u", "v"]
    assert cfg["matrix"]["seeds"] == [42, 43, 44]
