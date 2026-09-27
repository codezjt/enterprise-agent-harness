import pytest

from enterprise_harness.observability import (
    CostTracker,
    ModelPricing,
)


def test_cost_tracker_calculates_cost():
    tracker = CostTracker(
        ModelPricing(
            input_per_1k=0.001,
            output_per_1k=0.002,
        )
    )

    record = tracker.record(
        input_tokens=2000,
        output_tokens=500,
    )

    assert record.input_tokens == 2000
    assert record.output_tokens == 500

    assert record.input_cost == pytest.approx(0.002)
    assert record.output_cost == pytest.approx(0.001)
    assert record.total_cost == pytest.approx(0.003)

    assert tracker.input_tokens == 2000
    assert tracker.output_tokens == 500
    assert tracker.total_tokens == 2500
    assert tracker.estimated_cost == pytest.approx(0.003)


def test_cost_tracker_accumulates():
    tracker = CostTracker(
        ModelPricing(
            input_per_1k=0.001,
            output_per_1k=0.002,
        )
    )

    tracker.record(
        input_tokens=1000,
        output_tokens=1000,
    )

    tracker.record(
        input_tokens=2000,
        output_tokens=500,
    )

    assert tracker.input_tokens == 3000
    assert tracker.output_tokens == 1500
    assert tracker.total_tokens == 4500

    assert tracker.estimated_cost == pytest.approx(0.006)


def test_cost_tracker_snapshot():
    tracker = CostTracker(
        ModelPricing(
            input_per_1k=0.001,
            output_per_1k=0.002,
        )
    )

    tracker.record(
        input_tokens=1000,
        output_tokens=500,
    )

    assert tracker.snapshot() == {
        "input_tokens": 1000,
        "output_tokens": 500,
        "total_tokens": 1500,
        "estimated_cost": pytest.approx(0.002),
    }


def test_cost_tracker_reset():
    tracker = CostTracker(
        ModelPricing(
            input_per_1k=0.001,
            output_per_1k=0.002,
        )
    )

    tracker.record(
        input_tokens=1000,
        output_tokens=1000,
    )

    tracker.reset()

    assert tracker.input_tokens == 0
    assert tracker.output_tokens == 0
    assert tracker.total_tokens == 0
    assert tracker.estimated_cost == 0.0


def test_negative_tokens_are_rejected():
    tracker = CostTracker(
        ModelPricing(
            input_per_1k=0.001,
            output_per_1k=0.002,
        )
    )

    with pytest.raises(ValueError):
        tracker.record(
            input_tokens=-1,
            output_tokens=0,
        )

    with pytest.raises(ValueError):
        tracker.record(
            input_tokens=0,
            output_tokens=-1,
        )


def test_negative_pricing_is_rejected():
    with pytest.raises(ValueError):
        ModelPricing(
            input_per_1k=-0.001,
            output_per_1k=0.002,
        )

    with pytest.raises(ValueError):
        ModelPricing(
            input_per_1k=0.001,
            output_per_1k=-0.002,
        )