import asyncio

import numpy as np
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager


class _Tokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        del token_ids, skip_special_tokens
        return "response"


def test_naive_reward_manager_passes_rollout_identity_in_extra_info():
    captured = {}

    def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
        del data_source, solution_str, ground_truth, kwargs
        captured.update(extra_info)
        return {"score": 1.0}

    data = DataProto.from_dict(
        {
            "responses": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }
    )
    data.non_tensor_batch = {
        "data_source": np.array(["test"], dtype=object),
        "reward_model": np.array([{"ground_truth": "answer"}], dtype=object),
        "extra_info": np.array([{"existing": "kept"}], dtype=object),
        "uid": np.array(["prompt-7"], dtype=object),
        "session_id": np.array([3], dtype=np.int64),
    }
    async def run_reward():
        manager = NaiveRewardManager(DictConfig({}), _Tokenizer(), compute_score)
        manager.loop = asyncio.get_running_loop()
        return await manager.run_single(data)

    result = asyncio.run(run_reward())

    assert result["reward_score"] == 1.0
    assert captured == {
        "existing": "kept",
        "uid": "prompt-7",
        "session_id": 3,
        "num_turns": None,
        "rollout_reward_scores": {},
    }
