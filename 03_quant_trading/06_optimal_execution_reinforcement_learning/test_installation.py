"""
Installation and smoke-test verification.

Run from repository root:
    python test_installation.py
"""

import sys

import numpy as np


def ok(message: str) -> None:
    print(f"[OK] {message}")


def fail(message: str) -> None:
    print(f"[FAILED] {message}")


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from analysis.metrics import execution_cost_metrics  # noqa: F401
        from env.execution_env import ExecutionEnv  # noqa: F401
        from env.impact_models import LinearImpact  # noqa: F401
        from env.liquidity_models import MeanRevertingLiquidity  # noqa: F401
        from models.almgren_chriss import AlmgrenChrissLinear  # noqa: F401
        from models.bcq import BCQ  # noqa: F401
        from models.replay_buffer import ReplayBuffer  # noqa: F401
        from models.td3_bc import TD3_BC  # noqa: F401
        from models.twap import TWAP  # noqa: F401
        from models.vwap import VWAP  # noqa: F401

        ok("All imports successful")
    except ImportError as exc:
        fail(f"Import failed: {exc}")
        raise AssertionError("required project imports failed") from exc


def test_almgren_chriss():
    """Test Almgren-Chriss implementation."""
    print("\nTesting Almgren-Chriss...")

    try:
        from models.almgren_chriss import AlmgrenChrissLinear

        ac = AlmgrenChrissLinear(
            initial_inventory=1000.0,
            num_steps=20,
            volatility=0.02,
            eta=0.01,
            gamma=0.001,
            risk_aversion=0.5,
        )

        trades = ac.get_trades()
        assert len(trades) == 20, "Wrong number of trades"
        assert abs(np.sum(trades) - 1000.0) < 10.0, "Trades do not sum to inventory"

        cost = ac.expected_cost()
        variance = ac.cost_variance()
        assert cost > 0, "Cost should be positive"
        assert variance > 0, "Variance should be positive"

        ok(f"AC test passed (cost={cost:.2f}, var={variance:.2f})")
    except Exception as exc:
        fail(f"AC test failed: {exc}")
        raise


def test_environment():
    """Test execution environment."""
    print("\nTesting execution environment...")

    try:
        from env.execution_env import ExecutionEnv
        from env.impact_models import LinearImpact
        from env.liquidity_models import MeanRevertingLiquidity

        env = ExecutionEnv(
            initial_inventory=1000.0,
            num_steps=20,
            volatility=0.02,
            liquidity_process=MeanRevertingLiquidity(),
            impact_model=LinearImpact(eta=0.01),
            seed=42,
        )

        state, _ = env.reset()
        assert state.shape == (6,), f"Wrong state shape: {state.shape}"

        action = np.array([0.1])
        next_state, reward, _, _, info = env.step(action)

        assert next_state.shape == (6,), "Wrong next state shape"
        assert isinstance(reward, (int, float)), "Reward should be scalar"
        assert "total_cost" in info, "Missing total_cost in info"

        ok("Environment test passed")
    except Exception as exc:
        fail(f"Environment test failed: {exc}")
        raise


def test_twap():
    """Test TWAP strategy."""
    print("\nTesting TWAP strategy...")

    try:
        from env.execution_env import ExecutionEnv
        from models.twap import TWAP

        env = ExecutionEnv(initial_inventory=1000.0, num_steps=20, seed=42)
        twap = TWAP(initial_inventory=1000.0, num_steps=20)

        state, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < 25:
            action = np.array([twap.get_action(state)])
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

        summary = env.get_execution_summary()
        assert summary["completion_rate"] > 0.90, "TWAP did not complete execution"

        ok(f"TWAP test passed (completion={summary['completion_rate']:.2%})")
    except Exception as exc:
        fail(f"TWAP test failed: {exc}")
        raise


def test_rl_models():
    """Test RL model instantiation."""
    print("\nTesting RL models...")

    try:
        from models.bcq import BCQ
        from models.td3_bc import TD3_BC

        bcq = BCQ(state_dim=6, action_dim=1, max_action=0.3, device="cpu")
        td3_bc = TD3_BC(state_dim=6, action_dim=1, max_action=0.3, device="cpu")

        state = np.random.randn(6)
        bcq_action = bcq.select_action(state)
        td3_action = td3_bc.select_action(state)

        assert bcq_action.shape == (1,), "Wrong BCQ action shape"
        assert td3_action.shape == (1,), "Wrong TD3+BC action shape"

        ok("RL models test passed")
    except Exception as exc:
        fail(f"RL models test failed: {exc}")
        raise


def test_replay_buffer():
    """Test replay buffer."""
    print("\nTesting replay buffer...")

    try:
        from models.replay_buffer import ReplayBuffer

        buffer = ReplayBuffer(state_dim=6, action_dim=1, max_size=1000, device="cpu")

        for _ in range(100):
            state = np.random.randn(6)
            action = np.random.randn(1)
            next_state = np.random.randn(6)
            reward = np.random.randn()
            done = False
            buffer.add(state, action, next_state, reward, done)

        assert buffer.size == 100, f"Wrong buffer size: {buffer.size}"

        batch = buffer.sample(32)
        assert len(batch) == 5, "Wrong batch structure"

        ok("Replay buffer test passed")
    except Exception as exc:
        fail(f"Replay buffer test failed: {exc}")
        raise


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("INSTALLATION VERIFICATION")
    print("=" * 60)

    tests = [
        test_imports,
        test_almgren_chriss,
        test_environment,
        test_twap,
        test_rl_models,
        test_replay_buffer,
    ]

    results = []
    for test in tests:
        try:
            test()
            results.append(True)
        except Exception:
            results.append(False)

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {sum(results)}/{len(results)}")

    if all(results):
        ok("Installation successful")
        print("\nUsage:")
        print("  python experiments/train_rl.py --agent bcq")
        print("  python experiments/run_benchmarks.py")
        print("  python experiments/stress_tests.py")
        return 0

    fail("Some tests failed")
    print("\nTroubleshooting:")
    print("  pip install -r requirements.txt")
    print("  Verify Python >= 3.8")
    print("  Check PyTorch installation")
    return 1


if __name__ == "__main__":
    sys.exit(main())
