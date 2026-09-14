# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score


@register("dapo_overlong_penalty")
class DAPORewardManagerNemotron(RewardManagerBase):
    """DAPO Reward Manager."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)

        # DAPO Reward Config
        overlong_penalty_cfg = config.reward.get("reward_kwargs", {}).get("overlong_penalty", None)
        self.overlong_penalty_cfg = overlong_penalty_cfg
        self.max_resp_len = config.reward.get("reward_kwargs", {}).get("max_resp_len", None)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer 
        self.eos_id = self.tokenizer.eos_token_id
        assert type(self.eos_id) == int

        if self.overlong_penalty_cfg is not None and self.overlong_penalty_cfg.enable:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_penalty_cfg=}, but got None"
            )
        
        print("DAPO OVERLONG NEMOTRON MANAGER")

    async def run_single(self, data: DataProto) -> dict:
        data = data[-1:]  # for multi-sequence outputs, we only compute reward based on the last sequence
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        # for workplace agent (tool calling)
        tool_extra_fields = (
            data_item.non_tensor_batch.get("tool_extra_fields") or {}
        )
        extra_info = dict(
            data_item.non_tensor_batch.get("extra_info") or {}
        )
        extra_info["predicted_actions"] = tool_extra_fields.get(
            "predicted_actions", []
        )
        extra_info["last_assistant_turn"] = tool_extra_fields.get(
            "last_assistant_turn", ""
        )
        # DeepSearch's own key -- NOT a rename of the line above, a separate addition:
        # "last_assistant_turn" collides with stock ToolAgentLoop's own per-turn write of that
        # name (a raw TokenOutput, not text) for multi-turn agent loops (AllToolCallsAgentLoop
        # and DeepSearch's SearchAgentLoop/SuPOAgentLoop, which build on it) -- multi-domain-RL's
        # own single_turn_agent_overlong_filtering doesn't go through that code path, so its use
        # of "last_assistant_turn" above is unaffected and left as-is. DeepSearch's write side
        # uses "final_answer_text" instead (overlong_filtering.py's
        # AllToolCallsAgentLoop._handle_generating_state).
        extra_info["final_answer_text"] = tool_extra_fields.get(
            "final_answer_text", ""
        )
        # Same idea as final_answer_text above -- generic key ("tool_history", not
        # "search_calls", which is DeepSearch's own reward.py's own no-search-guard terminology;
        # DeepSearch maps this generic key onto that name itself). Written by
        # overlong_filtering.py's AllToolCallsAgentLoop._handle_generating_state on every
        # termination (all 4 reasons, not just natural_stop -- see that file's own comment for
        # why this one doesn't need the same scoping final_answer_text does).
        extra_info["tool_history"] = tool_extra_fields.get(
            "tool_history", []
        )
        overlong_filtering = extra_info.get("overlong_filtering", [])


        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )
        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        reward_extra_info = {}
        """
        The reward_extra_info → val-core/custom-metric pathway 
        (this is the mechanism that produces things like the per-data-source 
        wildguard/helpsteer2/helpsteer3 panels, and the "workplace assistant 
        reward" style metrics I mentioned earlier). It only picks up keys that 
        are nested inside extra_fields["reward_extra_info"], specifically because 
        reward_extra_keys is built from extra_fields["reward_extra_info"].keys() 
        (agent_loop.py:1082-1083).
        """

        score: float
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        reward = score

        if self.overlong_penalty_cfg is not None and self.overlong_penalty_cfg.enable:
            valid_response_ids = valid_response_ids.tolist() if hasattr(valid_response_ids, "tolist") else list(valid_response_ids)
            is_overlong = (
                valid_response_length >= self.max_resp_len
                and (len(valid_response_ids) == 0 or valid_response_ids[-1] != self.eos_id)
            )
                     
            if self.overlong_penalty_cfg.log:
                reward_extra_info["overlong_reward"] = -1*reward if is_overlong else 0.0
                reward_extra_info["overlong"] = is_overlong
            
            if is_overlong:   
                reward = 0.0

        ability = data_item.non_tensor_batch.get("ability")
        if ability:
            for tracked_domain in self.track_domains:
                is_domain = ability == tracked_domain
                reward_extra_info[f"{tracked_domain} reward"] = (
                    float(reward) if is_domain else 0.0
                )
                reward_extra_info[f"{tracked_domain} count"] = float(is_domain)

        if overlong_filtering:
            reward_extra_info["overlong_filtering"] = 1.

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
