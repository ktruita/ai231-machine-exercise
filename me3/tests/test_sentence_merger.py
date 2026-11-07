import pytest
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.tests.utils import run_test

from agent.processors.conversation import ShortSentenceMerger

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def pieces(*chunks: str) -> list[str]:
    frames = [LLMFullResponseStartFrame(), *(LLMTextFrame(c) for c in chunks), LLMFullResponseEndFrame()]
    down, _ = await run_test(ShortSentenceMerger(), frames_to_send=frames)
    return [f.text for f in down if isinstance(f, LLMTextFrame)]


async def test_short_opener_is_joined_to_next_sentence():
    out = await pieces("Sure", "! Try", " Atomic Habits by James Clear.", " It's practical and easy to read.")
    assert "".join(out) == "Sure, Try Atomic Habits by James Clear. It's practical and easy to read."
    assert out[0].startswith("Sure, Try Atomic Habits")


async def test_two_short_sentences_in_a_row():
    out = await pieces("Oh! Sure. Let me tell you about books.")
    assert "".join(out) == "Oh, Sure, Let me tell you about books."


async def test_long_sentences_untouched_and_question_kept():
    text = "Atomic Habits is a great place to start. Want more recommendations?"
    assert "".join(await pieces(text)) == text


async def test_lone_short_reply_still_spoken():
    assert "".join(await pieces("Sure!")) == "Sure!"


async def test_markdown_list_becomes_speech():
    out = "".join(await pieces("Sure, here are two great books:\n", "1. *Atomic", " Habits* by James Clear.\n",
                               "2. **Mindset** by Carol Dweck."))
    assert "*" not in out and "1." not in out and "2." not in out
    assert "Atomic Habits by James Clear." in out and "Mindset by Carol Dweck." in out


async def test_long_reply_capped_after_whole_sentence():
    frames = [LLMFullResponseStartFrame(),
              *(LLMTextFrame(c) for c in ["One two three four five six. ", "Seven eight nine ten eleven. ",
                                          "Twelve thirteen fourteen fifteen. "]),
              LLMFullResponseEndFrame()]
    down, _ = await run_test(ShortSentenceMerger(max_words=8), frames_to_send=frames)
    text = "".join(f.text for f in down if isinstance(f, LLMTextFrame))
    assert text.startswith("One two three four five six. Seven eight nine ten eleven.")
    assert "Twelve" not in text


async def test_quotes_removed():
    assert "\"" not in "".join(await pieces('Try "Atomic Habits" by James Clear today.'))
