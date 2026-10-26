"""Pure Torch inference for the exact ScaleTrack BFM checkpoint, without Isaac Sim.

The checkpoint contains actor weights, not a TorchScript deployment wrapper.
The observation transform is shared with ScaleBridge's BFM adapter, while the
actor and task embedder are loaded strictly from ScaleTrack's training code.
"""

from __future__ import annotations

import sys
from pathlib import Path


class RawScaleTrackBFM:
    def __init__(self, *, checkpoint: Path, metadata: dict, scalebridge: Path, device: str = "cuda"):
        import torch
        from omegaconf import OmegaConf

        if Path(metadata["checkpoint"]).resolve() != Path(checkpoint).resolve():
            raise ValueError("raw ScaleTrack metadata identifies a different checkpoint")
        if (
            len(metadata["joint_names"]) != 29
            or len(metadata["default_joint_pos"]) != 29
            or len(metadata["action_scale"]) != 29
        ):
            raise ValueError("raw ScaleTrack BFM requires 29 ordered body joints, defaults, and action scales")
        scaletrack = Path(metadata.get("scaletrack_root", Path(checkpoint).resolve().parents[4]))
        sys.path.insert(0, str(scaletrack / "source/my_rsl_rl"))
        sys.path.insert(0, str(scalebridge.resolve()))
        from my_rsl_rl.networks import HumanoidTransformer, TaskEmbedder
        from scalebridge.agent.bfm_agent import BFMAgent

        self.torch = torch
        self.device = device
        self.metadata = metadata
        self.feature_adapter = BFMAgent.__new__(BFMAgent)
        self.feature_adapter.config = OmegaConf.create({"future_idx": [0, 1, 2, 3, 4, 32]})
        self.actor = HumanoidTransformer(64, 29, 29, embed_dim=384, num_heads=6, ff_dim=384, num_layers=6).to(device)
        self.embedder = TaskEmbedder(267, 384).to(device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
        self.actor.load_state_dict(
            {k.removeprefix("actor."): v for k, v in state.items() if k.startswith("actor.")}, strict=True
        )
        self.embedder.load_state_dict(
            {
                k.removeprefix("actor_task_embedder."): v
                for k, v in state.items()
                if k.startswith("actor_task_embedder.")
            },
            strict=True,
        )
        self.actor.eval()
        self.embedder.eval()
        self.default = torch.tensor(metadata["default_joint_pos"], dtype=torch.float32, device=device)
        self.scale = torch.tensor(metadata["action_scale"], dtype=torch.float32, device=device)

    def get_action(self, observation: dict, mode: int):
        torch = self.torch
        # Isaac Lab's mdp.joint_pos_rel subtracts the fixed nominal G1 pose.
        obs = dict(observation)
        obs["dof_pos_buffer"] = observation["dof_pos_buffer"] - self.default
        prop = self.feature_adapter._policy_obs_for_engine(obs)
        task = self.feature_adapter._policy_task_for_engine(obs)
        # ScaleTrack's act_inference appends all-one mode features; control
        # selection changes the reference mask during training, but its play
        # path intentionally bypasses that mask.
        mode_feature = torch.ones((*task.shape[:-1], 14), dtype=task.dtype, device=task.device)
        task = torch.cat((task, mode_feature), dim=-1)
        if prop.shape != (1, 3, 64) or task.shape != (1, 6, 267):
            raise ValueError(f"wrong ScaleTrack feature shapes: prop={tuple(prop.shape)}, task={tuple(task.shape)}")
        with torch.inference_mode():
            raw = self.actor(prop, observation["actions_buffer"], self.embedder(task))
            target = raw * self.scale + self.default
        return target, raw
