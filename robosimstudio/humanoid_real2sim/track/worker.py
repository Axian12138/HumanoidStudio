"""ScaleBridge BFM inference worker. Run with ScaleBridge's Python, never RSS's.

Only the policy executes here. All physical stepping remains in RoboSimStudio.
Each request contains a fresh observation from the current RSS state, so pause
and resume do not advance hidden simulator time in this process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scalebridge", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.scalebridge.resolve()))

    import torch
    from omegaconf import OmegaConf
    from scalebridge.agent.bfm_agent import BFMAgent

    metadata = json.loads(args.metadata.read_text())
    raw_scaletrack = metadata.get("policy_kind") == "raw_scaletrack_bfm"
    if raw_scaletrack:
        from raw_bfm import RawScaleTrackBFM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = RawScaleTrackBFM(checkpoint=args.checkpoint, metadata=metadata, scalebridge=args.scalebridge, device=device)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("BFM Torch-TensorRT requires CUDA")
        device = "cuda"
        agent = BFMAgent(
            OmegaConf.create(
                {
                    "checkpoint": str(args.checkpoint.resolve()),
                    "metadata_path": str(args.metadata.resolve()),
                    "deployment_metadata_path": str(args.metadata.resolve()),
                    "future_idx": [0, 1, 2, 3, 4, 32],
                    "control_mode": 7,
                }
            ),
            device=device,
        )
    print("@ready " + json.dumps({"joint_names": metadata["joint_names"], "body_names": metadata["selected_body_names"]}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "close":
                return
            if request.get("command") != "infer":
                raise ValueError("expected infer")
            mode = int(request["mode"])
            if mode not in (4, 7):
                raise ValueError("interactive editor supports VR-5 (4) and WholeBody-14 (7)")
            if not raw_scaletrack:
                agent.config.control_mode = mode
                agent.control_mode = torch.tensor([mode], dtype=torch.long, device=device)
            observation = {
                key: torch.as_tensor(
                    value,
                    device=device,
                    dtype=torch.long if key == "future_time_offsets" else torch.float32,
                )
                for key, value in request["observation"].items()
            }
            with torch.inference_mode():
                target, raw = agent.get_action(observation, mode) if raw_scaletrack else agent.get_action(observation)
            print("@result " + json.dumps({"target": target.detach().cpu().numpy().reshape(-1).tolist(), "action": raw.detach().cpu().numpy().reshape(-1).tolist()}), flush=True)
        except Exception as exc:
            print("@error " + json.dumps({"type": type(exc).__name__, "message": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
