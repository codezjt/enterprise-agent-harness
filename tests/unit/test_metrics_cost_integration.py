import pytest

from enterprise_harness.observability import (
    CostTracker,
    MetricCollector,
    ModelPricing,
)


def test_metric_collector_records_cost_with_tokens():
    cost_tracker = CostTracker(
        ModelPricing(
            input_per_1k=0.001,
            output_per_1k=0.002,
        )
    )

    collector = MetricCollector(
        cost_tracker=cost_tracker,
    )

    collector.record_tokens(
        input_tokens=2000,
        output_tokens=500,
    )

    snapshot = collector.snapshot()

    assert snapshot.input_tokens == 2000
    assert snapshot.output_tokens == 500
    assert snapshot.total_tokens == 2500
    assert snapshot.estimated_cost == pytest.approx(0.003)

    assert cost_tracker.input_tokens == 2000
    assert cost_tracker.output_tokens == 500
    assert cost_tracker.total_tokens == 2500
    assert cost_tracker.estimated_cost == pytest.approx(0.003)


def test_metric_collector_accumulates_cost():
    cost_tracker = CostTracker(
        ModelPricing(
            input_per_1k=0.001,
            output_per_1k=0.002,
        )
    )

    collector = MetricCollector(
        cost_tracker=cost_tracker,
    )

    collector.record_tokens(
        input_tokens=1000,
        output_tokens=1000,
    )

    collector.record_tokens(
        input_tokens=2000,
        output_tokens=500,
    )

    snapshot = collector.snapshot()

    assert snapshot.input_tokens == 3000
    assert snapshot.output_tokens == 1500
    assert snapshot.total_tokens == 4500
    assert snapshot.estimated_cost == pytest.approx(0.006)

    assert cost_tracker.estimated_cost == pytest.approx(0.006)


def test_metric_collector_can_work_without_cost_tracker():
    collector = MetricCollector()

    collector.record_tokens(
        input_tokens=1000,
        output_tokens=500,
    )

    snapshot = collector.snapshot()

    assert snapshot.input_tokens == 1000
    assert snapshot.output_tokens == 500
    assert snapshot.total_tokens == 1500
    assert snapshot.estimated_cost == 0.0