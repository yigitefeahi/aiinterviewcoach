from app.rate_limit import RateLimitConfig, is_rate_limited


def test_rate_limit_blocks_after_threshold():
    config = RateLimitConfig(max_requests=2, window_seconds=60)
    key = "test-rate-limit-key"
    assert is_rate_limited(key, config) is False
    assert is_rate_limited(key, config) is False
    assert is_rate_limited(key, config) is True
