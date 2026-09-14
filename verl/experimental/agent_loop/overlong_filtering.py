"""Overlong filtering + full-tool-call-fidelity fix for multi-domain-RL.

Drafted here for review; intended to be placed under
`Cascade2/verl-version/verl/verl/experimental/agent_loop/` (alongside the existing,
simpler `overlong_filtering_agent.py`) and imported from that package's `__init__.py` so
the `@register` decorators below actually fire at trainer startup -- same mechanism as
`tool_agent`/`single_turn_agent` (`tool_agent_loop.py:99`, `single_turn_agent_loop.py:28`).
Not wired yet: dropping this file in-place under `verl/verl/` without importing it from
`__init__.py` leaves the decorators inert, exactly like the current, unwired
`multi_domain_agent.py` in that same directory. (2026-08-19: the pinned `verl/` checkout
this docstring originally referred to has been retired in favor of the editable-installed
v0.9.0 checkout at `Internship/env/verl` -- the tool-agent-loop mechanics this file overrides
are byte-identical between the two, confirmed diff, so nothing below changes, but wiring
this file in means importing it from *that* checkout's `agent_loop/__init__.py` now.)

Registered names (`tool_agent_overlong_filtering`, `single_turn_agent_overlong_filtering`)
are deliberately distinct from the existing `overlong_filtering_tool` /
`overlong_filtering_single_turn` in `overlong_filtering_agent.py`, so this can sit
alongside that file without clobbering it -- rename to reuse those names instead if the
intent is to replace it. `prepare_data.py`'s `agent_name` for workplace rows already
points at `tool_agent_overlong_filtering` to match.

Two distinct mechanisms, for two distinct rollout shapes:

- `OverlongFilteringSingleTurnAgentLoop` (MCQA, structured outputs): one `generate()` call
  per rollout. "Did the response hit `response_length` without emitting EOS" is a complete
  truncation test here -- there's no other way for the sequence to end.

- `OverlongFilteringToolAgentLoopSimple` (workplace assistant, naive): the same "last token
  != eos" test as the single-turn class above, applied to the whole multi-turn rollout's
  final response. This is what DAPO's original recipe and NeMo RL's own `overlong_filtering`
  actually do -- neither is turn-boundary-aware either. Known limitation, deliberately
  accepted for a first experiment: `ToolAgentLoop`'s state machine stops *cleanly at a turn
  boundary* whenever the token budget is exceeded (`tool_agent_loop.py:286`, `:387`/`:433`),
  so the last emitted assistant turn still ends in a real EOS/turn-end token even when the
  episode was genuinely cut short -- this test won't catch that specific case. Start here;
  measure whether it matters (e.g. via `num_turns/max` pinning at the cap, or the
  `truncated_by_budget` rate from the class below) before reaching for the more complex one.

- `OverlongFilteringToolAgentLoop` (workplace assistant, budget/turn-cap-aware): records
  *why* the state machine terminated, authoritatively, at the state transition itself, so it
  also catches the turn-boundary case above -- while still deliberately excluding turns lost
  to `max_assistant_turns`/`max_user_turns` (task-incompleteness, not decoding truncation;
  masking that would throw away real training signal, since the strict workplace reward
  already penalizes an incomplete action trace on its own). 2026-08-22: exposes this as a
  single categorical `extra_fields["exit_reason"]` (`response_length_budget` /
  `max_assistant_turns` / `max_user_turns` / `natural_stop`) instead of a lone boolean,
  matching dragon-agentic's `meta["exit_reason"]` pattern (`dragon/harness/default.py:153,158`)
  -- diagnostic only, doesn't change masking behavior by itself.

- `AllToolCallsAgentLoop` (2026-08-19, workplace assistant, both variants above build on this
  now): fixes the `max_parallel_calls` truncation bug documented in `multi-domain-RL/AGENTS.md`
  §5.1/§5.2. Upstream `ToolAgentLoop._handle_processing_tools_state` (`tool_agent_loop.py:315`)
  only executes `agent_data.tool_calls[:self.max_parallel_calls]` (`max_parallel_calls=1` here)
  -- but nothing on the generation side stops the model at one call per turn (`HermesToolParser`
  has no stop token for it, and the Hermes/Qwen tool-use system prompt itself says "you may call
  one or more functions"), so whenever the model emits 2+ `<tool_call>` blocks in one turn, only
  the first is executed, only the first gets a tool response, and only the first is ever recorded
  into `predicted_actions` (populated inside `WorkplaceTool.execute()`, which only runs for calls
  that reach `_call_tool`). Confirmed against the real NeMo-Gym reference (`NVIDIA-NeMo/Gym` @
  `5675816c`, `resources_servers/workplace_assistant/app.py`'s `verify()`): the reference
  reconstructs its scored action list from *every* `function_call` item the model emitted, with
  no equivalent truncation -- so this repo's reward computation was not measuring what the
  reference protocol measures, specifically for episodes where the model batches calls.

  Fix: execute every call in `agent_data.tool_calls`, not just the first `max_parallel_calls` --
  and execute them *sequentially* (awaited one at a time), not concurrently via `asyncio.gather`
  like the upstream method does for its (always <= max_parallel_calls-sized) task list. This is
  deliberate, not an oversight: `WorkplaceTool.execute()`'s own docstring notes a per-rollout
  `asyncio.Lock` would be needed "if you allowed multiple tool calls from the same assistant turn
  to execute concurrently against the same environment" -- true concurrent execution here would
  mean multiple coroutines mutating the same shared, non-thread-safe pandas DataFrames
  (`agent_data._workplace_env`) at once, and `predicted_actions.append(...)` (also inside
  `WorkplaceTool.execute()`) would race too, silently reordering the action trace that
  `execute_actions_and_reset_state` later replays "sequentially, in exact recorded order" --
  order matters there, so a reordered trace can turn a correct multi-action episode into an
  incorrectly-scored one even without any tool actually failing. Sequential awaiting keeps the
  original one-environment-access-at-a-time invariant while still processing every call.

  `_handle_processing_tools_state` has no factored extension point at the loop that builds
  `tasks`/`tool_call_names` (it's inline in a ~140-line method), so this necessarily copies the
  whole method rather than hooking a smaller piece -- diff against `tool_agent_loop.py:307-451`
  if upstream changes that method; the only intended difference is the removed `[:max_parallel_calls]`
  slice and the `asyncio.gather` -> sequential-await swap. `_build_assistant_message` is also
  overridden the same way for consistency (drops the same slice) -- currently dead code in this
  repo since `enable_continuous_token=False`, but would silently reintroduce the identical
  call/record mismatch if that flag is ever turned on, so it's fixed here too rather than left
  as a latent trap.

  `OverlongFilteringToolAgentLoopSimple` and `OverlongFilteringToolAgentLoop` both now inherit
  from `AllToolCallsAgentLoop` instead of bare `ToolAgentLoop` -- their own method bodies are
  unchanged, `super()._handle_processing_tools_state(...)` in the advanced variant now resolves
  to the fixed implementation automatically, so both registered names get the fix for free.
"""

import logging
import os
import sys
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import HermesToolParser, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.tools.schemas import OpenAIFunctionCallSchema, OpenAIFunctionParsedSchema, ToolResponse
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent_overlong_filtering")
class OverlongFilteringSingleTurnAgentLoop(SingleTurnAgentLoop):
    async def run(self, sampling_params: dict[str, Any], *args: Any, **kwargs: Any) -> AgentLoopOutput:
        output = await super().run(sampling_params, *args, **kwargs)

        is_overlong = (
            len(output.response_ids) >= self.response_length
            and (len(output.response_ids) == 0 or output.response_ids[-1] != self.tokenizer.eos_token_id)
        )

        if output.extra_fields is None:
            output.extra_fields = {}
        output.extra_fields["overlong_filtering"] = is_overlong

        if is_overlong:
            output.response_mask = [0] * len(output.response_mask)

        return output


class AllToolCallsAgentLoop(ToolAgentLoop):
    """Records every tool call the model emits in a turn for reward-replay purposes (never just
    the first `max_parallel_calls`), and executes up to `max_parallel_calls` of them live --
    2026-08-22, revised to match dragon-agentic's actual mechanism (`dragon/harness/default.py`,
    verified by reading it, not assumed): dragon-agentic does **not** restrict the model's
    *generation* to N calls either -- `chat.completions.create` there carries no
    `parallel_tool_calls` or stop-token constraint, same as this file's default. What it does is
    cap *execution*: calls past `max_tool_calls_per_turn` get an explicit
    `"error: tool-call budget exceeded"` message instead of being silently dropped, and -- because
    dragon-agentic's reward reads message history -- that refusal message is automatically what a
    reward function sees for that call, no separate bookkeeping needed.

    This repo's workplace reward doesn't read message history; it replays a separately-tracked
    `predicted_actions` list (`reward.py::execute_actions_and_reset_state`) into a fresh
    environment, decoupled from whatever happened live. That gives a strictly better option than
    dragon-agentic has available: **cap live execution using the existing `max_parallel_calls`
    knob (no new config needed) for compute/session-safety, while recording every emitted call
    into `predicted_actions` regardless of whether it was actually run** -- so reward-replay
    fidelity to NeMo-Gym's `verify()` (§5.2, `multi-domain-RL/AGENTS.md`) holds even when the cap
    binds, and only the model's *live* feedback for that turn is capped, not what it's scored on.
    With `max_parallel_calls=1` (this repo's current setting) every 2nd+ call in a batch gets
    refused-and-recorded rather than executed-and-recorded; bump `max_parallel_calls` in
    `launch.sh` if richer live feedback for legitimate multi-action batches (dataset max: 8) is
    wanted -- it no longer silently drops anything past the config value either way.

    Both tool-agent classes below build on this instead of on bare `ToolAgentLoop`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """2026-08-24 diagnostic: a val_only debug run showed the model's prompt missing the
        entire tools block (no `# Tools` section, no schemas -- confirmed by diffing against a
        known-working run's `input` field), even with `format=hermes` (plain, not
        `hermes_single_tool_call`) -- ruling out the parser as the cause. Traced the mechanism:
        `self.tools`/`self.tool_schemas` here come from the `tools=ToolListWrap(self.tools)`
        kwarg the worker passes in (`agent_loop.py:705`), which is `load_all_tools(tool_config_
        path=...)` (`tool_registry.py:83`) run once per worker -- and that function silently
        returns `[]` with no exception, no log line, if `tool_config_path` is falsy at the point
        the *worker* evaluates it, even though the driver's CLI override sets it. This print
        settles whether that's what happened here, without touching vendored verl/. Remove once
        root-caused -- see multi-domain-RL/AGENTS.md §8.6.

        2026-08-24 update: a first attempt at this used a bare `print(...)` and produced no
        output at all in a real run, despite every other explanation (wrong file, shadowed
        install, stale bytecode, duplicate registration) being ruled out -- so this may be an
        output-buffering/capture issue with plain stdout from inside a Ray actor, not proof this
        constructor never ran. Switched to `logger.warning` (this module already creates
        `logger` at WARN level) plus an explicit `sys.stderr` write with `flush=True`, so at
        least one of the two should survive if the other gets swallowed the same way.
        """
        super().__init__(*args, **kwargs)
        message = f"[AllToolCallsAgentLoop.__init__] loaded {len(self.tools)} tool(s): {sorted(self.tools.keys())}"
        logger.warning(message)
        print(message, file=sys.stderr, flush=True)

    async def _refuse_tool_call(self, tool_call: Any, agent_data: AgentData) -> tuple[ToolResponse, float, dict]:
        """A call past `max_parallel_calls` this turn: never dispatched live (protects live
        compute/shared-environment access from a pathological over-generation -- see class
        docstring), but its action record is still captured for reward-replay -- `WorkplaceTool.
        execute()` never runs for it, so nothing else appends it to `predicted_actions`; this does.
        Message wording mirrors dragon-agentic's exact refusal text (`default.py:140`) and its
        "always answer every call" principle.

        Workplace-assistant-specific by convention (`predicted_actions` is that domain's reward
        bookkeeping key, same as `WorkplaceTool.execute()` uses) -- harmless no-op key for any
        other tool-calling domain that might reuse this class.
        """
        agent_data.extra_fields.setdefault("predicted_actions", []).append(
            {"name": tool_call.name, "arguments": tool_call.arguments}
        )
        message = f"error: tool-call budget exceeded ({self.max_parallel_calls} per turn) — call not run."
        return ToolResponse(text=message), 0.0, {}


    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
        if state == AgentState.TERMINATED:
            # Mirrors the exact predicate + evaluation order of tool_agent_loop.py:286-305:
            # whichever of these is true first is always why the loop actually stopped here,
            # regardless of whether a later condition was also simultaneously true.
            if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
                agent_data.extra_fields["exit_reason"] = "response_length_budget"
            elif self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
                agent_data.extra_fields["exit_reason"] = "max_assistant_turns"
            elif self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
                agent_data.extra_fields["exit_reason"] = "max_user_turns"
            else:
                # No tool calls in the last generation -- the model chose to stop on its own.
                agent_data.extra_fields["exit_reason"] = "natural_stop"
                agent_data.extra_fields["final_answer_text"] = (   # named final_answer_text, NOT
                    # last_assistant_turn -- that name collides with stock ToolAgentLoop's own
                    # per-turn write of it (a raw TokenOutput object, not text) for every multi-
                    # turn agent loop built on this base (AllToolCallsAgentLoop and DeepSearch's
                    # SearchAgentLoop/SuPOAgentLoop) -- reading it back as text in the other 3
                    # termination branches (response_length_budget/max_assistant_turns/
                    # max_user_turns, where THIS write never runs) would hand extract_boxed() a
                    # TokenOutput instead of a str -> AttributeError.
                    agent_data.messages[-1].get("content", "")
                    if agent_data.messages and agent_data.messages[-1].get("role") == "assistant"
                    else ""
                )

            # tool_history: the full, ordered list of every tool call this episode actually
            # dispatched, generic across any tool-using agent loop built on this base (not just
            # DeepSearch's search tool -- deliberately NOT named "search_calls", which is
            # DeepSearch's own reward.py's terminology for its no-search guard; that project maps
            # this generic key onto that name itself, the way multi-domain-RL already does with
            # its own "predicted_actions"). Placed OUTSIDE the natural_stop-only branch above (all
            # four termination reasons get it, not just natural_stop) -- unlike final_answer_text,
            # this doesn't need the current turn to have been freshly parsed: agent_data.messages
            # already holds every PRIOR turn's tool_calls regardless of how the FINAL turn ends, so
            # deriving the whole episode's history from it is complete (or very nearly so -- only a
            # final turn that got cut off mid-generation by a budget limit could be missing,
            # the same edge case final_answer_text's own natural_stop-only scoping exists to avoid
            # for its own field, not one this derivation needs to worry about).
            agent_data.extra_fields["tool_history"] = [
                tc
                for msg in agent_data.messages
                if msg.get("role") == "assistant"
                for tc in (msg.get("tool_calls") or [])
            ]

        return state
    

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Copy of `ToolAgentLoop._handle_processing_tools_state` (`tool_agent_loop.py:307-451`)
        with three changes: every call in `agent_data.tool_calls` is looked at (not just
        `[:self.max_parallel_calls]`); only the first `max_parallel_calls` of them are actually
        dispatched (`_call_tool`), the rest are refused-but-recorded (`_refuse_tool_call`, see
        class docstring); and dispatched calls are awaited sequentially instead of via
        `asyncio.gather` (see module docstring for why). Keep this in sync with upstream if that
        method changes.
        # The only TERMINATED return from this method (tool_agent_loop.py:387, :433) is
                    # "appending the next tool response would cross the token budget" -- there is no
                    # turn-cap or natural-stop path through this method, so this is unconditional.
        """
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []
        previous_messages = list(agent_data.messages)

        tasks = []
        tool_call_names = []
        for index, tool_call in enumerate(agent_data.tool_calls):
            if index < self.max_parallel_calls:
                tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            else:
                tasks.append(self._refuse_tool_call(tool_call, agent_data))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            # Sequential, not asyncio.gather(*tasks): every dispatched call here can mutate the
            # same shared, non-thread-safe agent_data._workplace_env, and predicted_actions
            # ordering (appended inside WorkplaceTool.execute() or _refuse_tool_call) must match
            # emission order for the reward's replay to be meaningful. See module docstring.
            responses = []
            for task in tasks:
                responses.append(await task)

        for tool_index, (tool_response, tool_reward, _) in enumerate(responses):
            tool_call = agent_data.tool_calls[tool_index]
            if tool_response.image or tool_response.video:
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                message = {"role": "tool", "content": tool_response.text or ""}
            if tool_call.tool_call_id is not None:
                message["tool_call_id"] = tool_call.tool_call_id

            add_messages.append(message)

            if tool_response.image:
                if isinstance(tool_response.image, list):
                    for img in tool_response.image:
                        if img is not None:
                            new_images_this_turn.append(img)
                else:
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)

            if tool_response.video:
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages)

        if self.enable_continuous_token and not new_images_this_turn:
            schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
            merge_result, response_mask, response_logprobs = await self.ct_merge_non_assistant_msg(
                previous_messages,
                agent_data.messages,
                agent_data.prompt_ids,
                agent_data.response_mask,
                agent_data.response_logprobs if agent_data.response_logprobs else None,
                tools=schemas,
            )
            if len(response_mask) >= self.response_length:
                agent_data.extra_fields["exit_reason"] = "response_length_budget"
                return AgentState.TERMINATED
            agent_data.prompt_ids = merge_result.token_ids
            agent_data.response_mask = response_mask
            if agent_data.response_logprobs:
                agent_data.response_logprobs = response_logprobs or []
            agent_data.user_turns += 1
            return AgentState.GENERATING
        elif self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        elif self.tool_parser_name == "gemma4":
            parts = []
            for msg, name in zip(add_messages, tool_call_names, strict=True):
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = "".join([item.get("text", "") for item in content if item.get("type") == "text"])
                parts.append(f'<|tool_response>response:{name}{{value:<|"|>{content}<|"|>}}<tool_response|>')
            tool_response_text = "".join(parts)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            images = new_images_this_turn if new_images_this_turn else None
            videos = None
            response_ids = await self.apply_chat_template(
                add_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )
            response_ids = self.turn_separator + response_ids

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            agent_data.extra_fields["exit_reason"] = "response_length_budget"
            return AgentState.TERMINATED

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    def _build_assistant_message(self, content: str, agent_data: AgentData) -> dict[str, Any]:
        """Same fix as above, for the continuous-token path (`tool_agent_loop.py:452-475`) --
        NOT dead code whenever a caller sets `enable_continuous_token=True` (e.g. DeepSearch's
        `data.continuous_token.enable=True`) -- kept in sync so turning that flag on doesn't
        silently reintroduce the truncation.

        2026-09-01: the `raise ValueError` this used to have on `has_decode_error` crashed the
        WHOLE trajectory (`_run_agent_loop` -> uncaught -> the episode is lost) the first time a
        real, undertrained policy emitted a syntactically-valid-JSON-but-non-object `arguments`
        string (observed live: `"arguments": "some plain string"` instead of
        `"arguments": {"query": "some plain string"}`) -- an entirely expected, recoverable kind of
        mistake early in RL training, not a wiring bug worth losing a whole rollout over. `_call_tool`
        (this class's own `_handle_processing_tools_state`, inherited `_call_tool`) already handles
        this exact case gracefully -- `tool.execute(...)` raising (e.g. a plain string has no
        `.get()`) is caught by its own broad `except Exception`, producing a normal "Error
        executing tool" observation the model can react to and keep going. This method's job is
        only to build the RECORDED message for continuous-token mode; it has no business being
        stricter than the path that actually dispatches the call.

        First attempt at this fix SKIPPED the malformed call entirely (`continue`, no warning-only
        fallback) -- wrong, caught by re-comparing against the base `ToolAgentLoop` version this
        overrides: `OpenAIFunctionCallSchema.from_openai_function_parsed_schema` ALREADY returns a
        safe, valid `function_call` on decode failure (`arguments` coerced to `{}`), `has_decode_error`
        is just a heads-up flag alongside it, not "there is no object." Dropping the call meant this
        method's recorded `tool_calls` list and `_handle_processing_tools_state`'s later per-call
        tool-response messages (which iterate the UNTOUCHED `agent_data.tool_calls`, one response
        per entry, each carrying its own `tool_call_id` regardless of what got recorded here) would
        disagree -- a "tool" message referencing a `tool_call_id` absent from the preceding
        assistant message's own `tool_calls`, an incoherent transcript. Fixed: keep using the
        already-safe `function_call` (empty-dict arguments) instead of skipping, so the recorded
        `tool_calls` list always has exactly one entry per `agent_data.tool_calls` entry, matching
        what the tool-response messages will reference -- just log a warning, don't raise, don't drop.
        """
        message: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if not agent_data.tool_calls:
            return message

        tool_calls = []
        for tool_call in agent_data.tool_calls:
            function_call, has_decode_error = OpenAIFunctionCallSchema.from_openai_function_parsed_schema(
                OpenAIFunctionParsedSchema(name=tool_call.name, arguments=tool_call.arguments)
            )
            if has_decode_error:
                logger.warning(
                    f"Invalid tool call arguments for '{tool_call.name}': expected a JSON object "
                    f"string, got {tool_call.arguments!r} -- recording it with empty arguments "
                    "instead of raising or dropping it (dropping would desync this message's "
                    "tool_calls from _handle_processing_tools_state's later per-call tool-response "
                    "messages, which reference every agent_data.tool_calls entry unconditionally)."
                )
            tool_call_message = {
                "type": "function",
                "function": function_call.model_dump(),
            }
            if tool_call.tool_call_id is not None:
                tool_call_message["id"] = tool_call.tool_call_id
            tool_calls.append(tool_call_message)
        message["tool_calls"] = tool_calls
        return message


@register("tool_agent_overlong_filtering")
class OverlongFilteringToolAgentLoop(AllToolCallsAgentLoop):
    """Tracks *why* the state machine terminated as a single categorical `exit_reason`
    (2026-08-22, replaces the earlier `truncated_by_budget`-only boolean with the same
    mechanism generalized) -- dragon-agentic's `meta["exit_reason"] = "max_turns"` pattern
    (`dragon/harness/default.py:153,158`) is a single clean field for exactly this, versus
    scattered booleans; adopting the same shape here. Diagnostic only unless something reads
    `extra_fields["exit_reason"]` downstream -- doesn't change any training-affecting behavior
    by itself.
    """

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        output = await super().run(sampling_params, **kwargs)

        exit_reason = (output.extra_fields or {}).get("exit_reason", "natural_stop")
        truncated_by_budget = exit_reason == "response_length_budget"
        truncated_mid_generation = (
            len(output.response_ids) >= self.response_length
            and (len(output.response_ids) == 0 or output.response_ids[-1] != self.tokenizer.eos_token_id)
        )
        is_overlong = truncated_by_budget or truncated_mid_generation

        if output.extra_fields is None:
            output.extra_fields = {}
        output.extra_fields["exit_reason"] = exit_reason
        output.extra_fields["overlong_filtering"] = is_overlong

        if is_overlong:
            output.response_mask = [0] * len(output.response_mask)

        return output