import pytest

from agent.processors.conversation import addressed_to_someone_else


@pytest.mark.parametrize("text", [
    "Derek traffic was insanity today?",          # ASR dropped the comma
    "Yo, Mark. Meetings bump to 03:00.",
    "Laura, lights flickered again. Storm maybe.",
    "Roomy fridge is empty again. Groceries?",
    "Rita, pass the soy sauce, please.",
    "Dana, these shoes are killing me.",
    "Chris, your laptop's still on the table, man.",
    "(faint, distant voice) Hey Anna, dinner is ready.",
])
def test_side_talk_detected(text):
    assert addressed_to_someone_else(text)


@pytest.mark.parametrize("text", [
    "Mute the TV.",
    "Mark this reminder as done please",
    "Open the front door camera feed.",
    "Set an alarm for 06:30AM on weekdays.",
    "Hey, what's the weather like?",
    "Hey assistant, play some music.",
    "Will it rain tomorrow?",
    "May I ask you something?",
    "How long should I steep green tea?",
    "Play some jazz please",
])
def test_requests_not_side_talk(text):
    assert not addressed_to_someone_else(text)


# --- the gate: resume after being cut off, otherwise stay silent without calling the LLM ----
from pipecat.frames.frames import (  # noqa: E402
    BotStartedSpeakingFrame,
    InterruptionFrame,
    LLMContextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.tests.utils import run_test  # noqa: E402

from agent.processors.conversation import BackchannelGate  # noqa: E402


def _ctx(text):
    return LLMContext(messages=[{"role": "system", "content": "s"},
                                {"role": "assistant", "content": "Atomic Habits is"},
                                {"role": "user", "content": text}])


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_side_talk_while_idle_is_dropped(anyio_backend):
    ctx = _ctx("Yo Mark, meeting's bumped to three.")
    down, _ = await run_test(BackchannelGate(side_talk_check=True), frames_to_send=[LLMContextFrame(ctx)])
    assert not any(isinstance(f, LLMContextFrame) for f in down)  # no LLM call
    assert "Ignored" in ctx.get_messages()[-1]["content"]


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_side_talk_that_cut_us_off_resumes(anyio_backend):
    ctx = _ctx("Hey Jamie, you fed the dog already, yeah?")
    frames = [BotStartedSpeakingFrame(), InterruptionFrame(), LLMContextFrame(ctx)]
    down, _ = await run_test(BackchannelGate(side_talk_check=True), frames_to_send=frames)
    assert any(isinstance(f, LLMContextFrame) for f in down)
    assert "Continue your previous answer" in ctx.get_messages()[-1]["content"]


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_gate_stamps_current_time_once(anyio_backend):
    ctx = _ctx("What time is it?")
    gate = BackchannelGate(timezone="Asia/Manila")
    await run_test(gate, frames_to_send=[LLMContextFrame(ctx)])
    await run_test(gate, frames_to_send=[LLMContextFrame(ctx)])  # a second turn must not stack lines
    system = ctx.get_messages()[0]["content"]
    assert system.startswith("s") and system.count("Current local time:") == 1
    assert "(Asia/Manila)" in system


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_tool_rule_stays_last_after_time_stamp(anyio_backend):
    from agent.processors.conversation import TOOL_RULE
    ctx = LLMContext(messages=[{"role": "system", "content": "prompt" + TOOL_RULE},
                               {"role": "user", "content": "Lights on."}])
    gate = BackchannelGate(timezone="Asia/Manila")
    for _ in range(2):
        await run_test(gate, frames_to_send=[LLMContextFrame(ctx)])
    system = ctx.get_messages()[0]["content"]
    assert system.endswith(TOOL_RULE) and system.count(TOOL_RULE) == 1
    assert system.count("Current local time:") == 1
    assert system.index("Current local time:") < system.index("TOOLS ARE MANDATORY")
