from app.interview_config import normalize_config, target_question_count
from app.interview_evaluation import score_reliability


def test_normalize_config_applies_defaults():
    cfg = normalize_config({})
    assert cfg["difficulty"] == "Junior"
    assert cfg["mode"] == "text"
    assert cfg["interview_length"] == "10 Questions"
    assert cfg["focus_area"] == "Mixed"


def test_target_question_count_maps_lengths():
    assert target_question_count("5 Questions") == 5
    assert target_question_count("10 Questions") == 10
    assert target_question_count("15 Questions") == 15
    assert target_question_count("20 Minutes") == 6
    assert target_question_count("30 Minutes") == 8


def test_score_reliability_high_consistency():
    rel = score_reliability([72, 73, 72, 74])
    assert rel["runs"] == 4
    assert rel["consistency_label"] in {"high", "moderate"}
    assert rel["std_dev"] <= 1.0


def test_score_reliability_low_consistency():
    rel = score_reliability([40, 75, 55, 90])
    assert rel["runs"] == 4
    assert rel["consistency_percent"] < 80
