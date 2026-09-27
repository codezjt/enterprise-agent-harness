from enterprise_harness.observability import (
    MetricCollector,
    MetricSnapshot,
)


def test_metric_snapshot_rates():
    snapshot = MetricSnapshot(
        agent_runs=10,
        agent_successes=8,
        task_runs=20,
        task_successes=15,
        tool_calls=10,
        tool_successes=9,
    )

    assert snapshot.agent_success_rate == 0.8
    assert snapshot.task_success_rate == 0.75
    assert snapshot.tool_success_rate == 0.9


def test_metric_collector():
    collector = MetricCollector()

    collector.record_agent_run(success=True)
    collector.record_agent_run(success=False)

    collector.record_task(success=True)

    collector.record_tool(success=True)
    collector.record_tool(success=False)

    collector.record_tool_retry()
    collector.record_replan()
    collector.record_human_approval()
    collector.record_policy_denied()

    collector.record_latency(100)
    collector.record_latency(200)
    collector.record_latency(300)

    collector.record_tokens(
        input_tokens=1000,
        output_tokens=500,
    )

    collector.record_cost(0.01)

    snapshot = collector.snapshot()

    assert snapshot.agent_runs == 2
    assert snapshot.agent_successes == 1

    assert snapshot.task_runs == 1
    assert snapshot.task_successes == 1

    assert snapshot.tool_calls == 2
    assert snapshot.tool_successes == 1

    assert snapshot.tool_retries == 1
    assert snapshot.replans == 1
    assert snapshot.human_approvals == 1
    assert snapshot.policy_denied == 1

    assert snapshot.input_tokens == 1000
    assert snapshot.output_tokens == 500
    assert snapshot.total_tokens == 1500

    assert snapshot.estimated_cost == 0.01

    assert snapshot.average_latency_ms == 200


def test_metric_percentiles():
    snapshot = MetricSnapshot(
        latencies_ms=list(range(1, 101)),
    )

    assert snapshot.p95_latency_ms == 95.05
    assert snapshot.p99_latency_ms == 99.01


def test_metric_collector_reset():
    collector = MetricCollector()

    collector.record_agent_run(success=True)
    collector.record_tool(success=True)

    collector.reset()

    snapshot = collector.snapshot()

    assert snapshot.agent_runs == 0
    assert snapshot.tool_calls == 0
    assert snapshot.total_tokens == 0
    assert snapshot.estimated_cost == 0.0


def test_metric_validation():
    collector = MetricCollector()

    try:
        collector.record_latency(-1)
        assert False
    except ValueError:
        pass

    try:
        collector.record_cost(-1)
        assert False
    except ValueError:
        pass
