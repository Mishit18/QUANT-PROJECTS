from src.robustness.deploy_gates import capacity_cost_grid, deploy_reject_gate


def test_capacity_grid_and_deploy_gate():
    grid = capacity_cost_grid(4.0, 1.2, [1_000_000, 5_000_000], [1, 3, 5])
    gate = deploy_reject_gate(ic_mean=0.01, ic_ir=0.3, turnover=1.0, net_return=0.05)
    reject = deploy_reject_gate(ic_mean=-0.01, ic_ir=0.1, turnover=5.0, net_return=-0.02)

    assert len(grid) == 6
    assert gate["verdict"] == "deploy_candidate"
    assert reject["verdict"] == "reject"

