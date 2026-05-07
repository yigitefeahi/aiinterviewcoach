from types import SimpleNamespace

from app.reporting import build_regression_snapshot


def test_build_regression_snapshot_pass():
    sessions = [
        SimpleNamespace(
            result_json={
                "status": "completed",
                "average_score": 72,
                "final_summary": {"score": 74, "confidence_score": 70, "red_flags": ["a"]},
            }
        ),
        SimpleNamespace(
            result_json={
                "status": "completed",
                "average_score": 69,
                "final_summary": {"score": 71, "confidence_score": 66, "red_flags": []},
            }
        ),
    ]
    result = build_regression_snapshot(sessions)
    assert result["status"] == "pass"
    assert result["sample_size"] == 2


def test_build_regression_snapshot_insufficient_data():
    sessions = [SimpleNamespace(result_json={"status": "in_progress"})]
    result = build_regression_snapshot(sessions)
    assert result["status"] == "insufficient_data"
