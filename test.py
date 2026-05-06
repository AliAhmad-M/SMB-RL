import torch
import warnings
import numpy as np
import cv2
import json
import platform
import sys
from pathlib import Path
from datetime import datetime, timezone

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

from environments.smb_env import make_smb_env
from agents.mario import Mario


# Config
NUM_RUNS   = 100
SAVE_DIR   = Path("runs") / datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
CKPT_PATH  = Path("output") / "2026-05-05T23-04-06" / "mario_final.chkpt"
VIDEO_OUT  = SAVE_DIR / "best_run.mp4"
META_OUT   = SAVE_DIR / "eval_metadata.json"
FPS        = 60
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

ACTIONS = [["right", "B"], ["right", "A", "B"]]
SKIP    = 4


# Raw visual environment
def make_raw_env():
    warnings.filterwarnings("ignore", category=UserWarning, module="gym")
    env = gym_super_mario_bros.make(
        "SuperMarioBros-2-1-v0",
        render_mode="rgb_array",
        apply_api_compatibility=True,
    )
    env = JoypadSpace(env, ACTIONS)
    return env


# Agent setup
def load_agent(action_dim: int) -> Mario:
    mario = Mario(
        state_dim=(4, 12, 12),
        action_dim=action_dim,
        save_dir=SAVE_DIR,
        device=DEVICE,
    )
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    mario.net.load_state_dict(ckpt["model"])
    mario.exploration_rate = 0.05
    return mario


# Single episode runner
def run_episode(mario: Mario, agent_env, raw_env):
    state, _ = agent_env.reset() if hasattr(agent_env.reset(), '__len__') else (agent_env.reset(), {})
    raw_env.reset()

    if isinstance(state, tuple):
        state = state[0]

    total_reward = 0.0
    frames       = []
    done         = False
    flag_get     = False
    step_count   = 0

    while not done:
        action = mario.act(state)
        step_count += 1

        result = agent_env.step(action)
        if len(result) == 5:
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            next_state, reward, done, info = result

        total_reward += reward

        for _ in range(SKIP):
            raw_result = raw_env.step(action)
            raw_frame  = raw_env.render()
            if raw_frame is not None:
                frames.append(raw_frame.copy())
            if len(raw_result) == 5:
                _, _, rt, ru, _ = raw_result
                if rt or ru:
                    break
            else:
                _, _, rd, _ = raw_result
                if rd:
                    break

        if info.get("flag_get", False):
            flag_get = True
            done     = True

        state = next_state

    return total_reward, frames, flag_get, step_count


# Video writer
def save_video(frames: list, path: Path, fps: int = FPS):
    if not frames:
        print("[ERROR]: No frames to save.")
        return
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"\tVideo saved -> {path}  ({len(frames)} frames, {len(frames)/fps:.1f}s)")


# Metadata writer
def save_metadata(results: list, best_run: int, best_reward: float, best_frames: list, path: Path):
    rewards = [r["reward"] for r in results]
    flags   = [r for r in results if r["flag_get"]]

    metadata = {
        "eval_timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(CKPT_PATH),
        "device": DEVICE,
        "num_runs": NUM_RUNS,
        "fps": FPS,
        "skip_frames": SKIP,
        "actions": ACTIONS,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,

        "summary": {
            "best_run": best_run,
            "best_reward": round(best_reward, 2),
            "best_video": str(VIDEO_OUT),
            "mean_reward": round(float(np.mean(rewards)), 2),
            "std_reward": round(float(np.std(rewards)), 2),
            "min_reward": round(float(np.min(rewards)), 2),
            "max_reward": round(float(np.max(rewards)), 2),
            "median_reward": round(float(np.median(rewards)), 2),
            "flag_capture_count": len(flags),
            "flag_capture_rate": round(len(flags) / NUM_RUNS, 4),
            "best_run_frame_count": len(best_frames),
            "best_run_duration_s": round(len(best_frames) / FPS, 2),
        },

        "runs": results,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\tMetadata saved -> {path}")


# Main
def main():
    eval_start = datetime.now(timezone.utc)

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Running {NUM_RUNS} evaluation episodes...\n")

    _tmp = make_smb_env(render_mode="rgb_array")
    action_dim = _tmp.action_space.n
    _tmp.close()

    mario = load_agent(action_dim)

    best_reward = -np.inf
    best_frames = []
    best_run    = -1

    results = []

    for run_idx in range(1, NUM_RUNS + 1):
        print(f"Run {run_idx}/{NUM_RUNS}...", end=" ", flush=True)
        run_start = datetime.now(timezone.utc)

        agent_env = make_smb_env(render_mode="rgb_array")
        raw_env   = make_raw_env()

        try:
            reward, frames, flag_get, steps = run_episode(mario, agent_env, raw_env)
        finally:
            agent_env.close()
            raw_env.close()

        elapsed = (datetime.now(timezone.utc) - run_start).total_seconds()
        flag_str = "FLAG captured!" if flag_get else ""
        print(f"reward={reward:.1f}  frames={len(frames)}  steps={steps}  {flag_str}")

        results.append({
            "run": run_idx,
            "reward": round(float(reward), 2),
            "frame_count": len(frames),
            "step_count": steps,
            "duration_s": round(elapsed, 2),
            "flag_get": flag_get,
        })

        if reward > best_reward:
            best_reward = reward
            best_frames = frames
            best_run    = run_idx

    # Summary
    print("\nResults")
    for r in results:
        marker = "< BEST" if r["run"] == best_run else ""
        print(f"\tRun {r['run']}: reward={r['reward']:.1f}  frames={r['frame_count']}  flag={r['flag_get']}  {marker}")

    total_elapsed = round((datetime.now(timezone.utc) - eval_start).total_seconds(), 2)
    print(f"\nTotal eval time: {total_elapsed}s")
    print(f"\nSaving best run (run {best_run}, reward={best_reward:.1f})...")
    save_video(best_frames, VIDEO_OUT)
    save_metadata(results, best_run, best_reward, best_frames, META_OUT)


if __name__ == "__main__":
    main()