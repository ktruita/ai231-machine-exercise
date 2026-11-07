import pytest
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.tests.utils import run_test

from agent.processors.conversation import AddressFilter

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def spoken_text(*chunks: str) -> str:
    frames = [LLMFullResponseStartFrame(), *(LLMTextFrame(c) for c in chunks), LLMFullResponseEndFrame()]
    down, _ = await run_test(AddressFilter(), frames_to_send=frames)
    return "".join(f.text for f in down if isinstance(f, LLMTextFrame))


async def test_marker_in_one_chunk_is_silenced():
    assert await spoken_text("<silent>") == ""


async def test_marker_split_across_tokens_is_silenced():
    assert await spoken_text("<", "sil", "ent", ">") == ""


async def test_marker_with_leading_space_and_trailing_text_is_silenced():
    assert await spoken_text(" <silent>", " ok") == ""


async def test_truncated_marker_is_silenced():
    assert await spoken_text("<sil", "ent") == ""


async def test_normal_answer_passes_unchanged():
    assert await spoken_text("It's ", "3 PM", ".") == "It's 3 PM."


async def test_answer_starting_with_angle_bracket_passes():
    assert await spoken_text("<", "3 degrees colder than yesterday.") == "<3 degrees colder than yesterday."


async def test_each_response_decided_separately():
    frames = [LLMFullResponseStartFrame(), LLMTextFrame("<silent>"), LLMFullResponseEndFrame(),
              LLMFullResponseStartFrame(), LLMTextFrame("Sure."), LLMFullResponseEndFrame()]
    down, _ = await run_test(AddressFilter(), frames_to_send=frames)
    assert "".join(f.text for f in down if isinstance(f, LLMTextFrame)) == "Sure."
