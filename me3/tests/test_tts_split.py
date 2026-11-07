from agent.processors.mcp_speech import split_for_speed


def test_short_sentence_untouched():
    s = "Sure, try Atomic Habits by James Clear."
    assert split_for_speed(s) == [s]


def test_long_sentence_split_at_comma():
    s = "Tornadoes are violent rotating columns of air, and they form when warm moist air meets cold dry air."
    head, tail = split_for_speed(s)
    assert head == "Tornadoes are violent rotating columns of air,"
    assert tail.startswith("and they form")
    assert f"{head} {tail}" == s


def test_no_split_point_keeps_sentence():
    s = "This is a fairly long sentence without any commas that keeps going on and on."
    assert split_for_speed(s) == [s]


def test_comma_too_early_or_late_is_skipped():
    s = "Well, this sentence has a comma very early and one near the very end, ok."
    assert split_for_speed(s) == [s]
