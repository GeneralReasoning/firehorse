import json

from firehorse.mcp.convert import strip_or_reward_marker


class TestStripOrRewardMarker:
    def test_raw_form(self):
        text = 'Result text\n[OR_REWARD:{"r":1.0,"f":true}]'
        assert strip_or_reward_marker(text) == "Result text"

    def test_no_marker(self):
        assert strip_or_reward_marker("plain output") == "plain output"

    def test_empty(self):
        assert strip_or_reward_marker("") == ""

    def test_json_escaped_form(self):
        # Marker as it appears inside a raw stream-json line
        raw_line = json.dumps(
            {"content": 'Result text\n[OR_REWARD:{"r":1.0,"f":true}]'}
        )
        cleaned = strip_or_reward_marker(raw_line)
        assert "OR_REWARD" not in cleaned
        # The tool result text is preserved (decoded JSON still valid).
        assert json.loads(cleaned) == {"content": "Result text"}

    def test_multiple_markers(self):
        text = 'x[OR_REWARD:{"r":0,"f":false}]y[OR_REWARD:{"r":1,"f":true}]'
        assert strip_or_reward_marker(text) == "xy"

    def test_negative_reward(self):
        text = 'fail [OR_REWARD:{"r":-0.5,"f":false}]'
        assert strip_or_reward_marker(text) == "fail"

    def test_null_reward(self):
        text = 'pending [OR_REWARD:{"r":null,"f":false}]'
        assert strip_or_reward_marker(text) == "pending"

    def test_preserves_episode_complete(self):
        text = (
            'done [OR_REWARD:{"r":1.0,"f":true}]\n\n'
            "[EPISODE COMPLETE] The environment has signaled ..."
        )
        cleaned = strip_or_reward_marker(text)
        assert "OR_REWARD" not in cleaned
        assert "[EPISODE COMPLETE]" in cleaned
