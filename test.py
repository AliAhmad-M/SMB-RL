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
NUM_RUNS   = 25
SAVE_DIR   = Path("runs") / datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
CKPT_PATH  = Path("output") / "placeholder_path" / "mario_final.chkpt"
VIDEO_OUT  = SAVE_DIR / "best_run.mp4"
META_OUT   = SAVE_DIR / "eval_metadata.json"
FPS        = 60
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
UPSCALE    = 3  

ACTIONS = [["right", "B"], ["right", "A", "B"]]
SKIP    = 4

# Color palette
COL_WHITE       = (255, 255, 255)
COL_BLACK       = (0, 0, 0)
COL_GRAY_NODE   = (128, 128, 128)
COL_CYAN_NODE   = (255, 255, 0)
COL_PINK_LINE   = (203, 105, 255)
COL_SUCCESS     = (76, 220, 115)

# Overlay layout
N_HIDDEN    = 12
NODE_R      = 5    
NES_BUTTONS = ["U", "D", "L", "R", "A", "B"]
ACTION_BUTTONS = [
    {"R", "B"},
    {"R", "A", "B"},
]

def draw_nn_overlay(state: np.ndarray, activations: dict, chosen_action: int) -> np.ndarray:
    canvas_h, canvas_w = 240, 240
    panel = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    # Observation Grid
    grid_res = state.shape[1] 
    cell_size = 8
    grid_size = grid_res * cell_size
    gx, gy = (canvas_w - grid_size) // 2, 20
    
    raw = state[-1].astype(np.uint8)
    for row in range(grid_res):
        for col in range(grid_res):
            val = int(raw[row, col])
            color = (val, val, val) 
            x1, y1 = gx + col * cell_size, gy + row * cell_size
            x2, y2 = x1 + cell_size, y1 + cell_size
            cv2.rectangle(panel, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(panel, (x1, y1), (x2, y2), (230, 230, 230), 1)

    # Hidden Layer
    hidden_y = 150
    fc_vals = activations.get("fc", np.zeros(256))
    idx = np.linspace(0, len(fc_vals) - 1, N_HIDDEN, dtype=int)
    nodes_x = np.linspace(30, canvas_w - 30, N_HIDDEN).astype(int)
    
    anchor_grid = (gx + grid_size // 2, gy + grid_size)
    for nx in nodes_x:
        cv2.line(panel, anchor_grid, (nx, hidden_y - NODE_R), COL_PINK_LINE, 1, cv2.LINE_AA)
    
    for i, nx in enumerate(nodes_x):
        v = fc_vals[idx[i]]
        color = COL_CYAN_NODE if v > 0.5 else COL_GRAY_NODE
        cv2.circle(panel, (nx, hidden_y), NODE_R, color, -1)
        cv2.circle(panel, (nx, hidden_y), NODE_R, COL_BLACK, 1)

    # Output Buttons
    output_y = 210
    btn_x_pos = np.linspace(40, canvas_w - 40, len(NES_BUTTONS)).astype(int)
    pressed = ACTION_BUTTONS[chosen_action] if chosen_action < len(ACTION_BUTTONS) else set()

    for i, (btn_name, bx) in enumerate(zip(NES_BUTTONS, btn_x_pos)):
        active = btn_name in pressed
        line_col = (255, 120, 120) if active else (220, 220, 220)
        for nx in nodes_x:
            cv2.line(panel, (nx, hidden_y + NODE_R), (bx, output_y - NODE_R), line_col, 1)

        btn_fill = COL_SUCCESS if active else COL_WHITE
        cv2.circle(panel, (bx, output_y), NODE_R, btn_fill, -1)
        cv2.circle(panel, (bx, output_y), NODE_R, COL_BLACK, 1)
        cv2.putText(panel, btn_name, (bx - 4, output_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, COL_BLACK, 1, cv2.LINE_AA)

    return panel

def make_raw_env():
    warnings.filterwarnings("ignore", category=UserWarning, module="gym")
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode="rgb_array", apply_api_compatibility=True)
    env = JoypadSpace(env, ACTIONS)
    return env

def load_agent(action_dim: int) -> Mario:
    mario = Mario(state_dim=(4, 12, 12), action_dim=action_dim, save_dir=SAVE_DIR, device=DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    mario.net.load_state_dict(ckpt["model"])
    mario.exploration_rate = 0.05
    return mario

class ActivationCache:
    def __init__(self, net):
        self._cache = {}
        self._hooks = []
        def hook(module, inp, out):
            with torch.no_grad():
                self._cache["fc"] = out.detach().float()[0].cpu().numpy()
        for name, module in net.named_modules():
            if isinstance(module, torch.nn.Linear):
                self._hooks.append(module.register_forward_hook(hook))
                break
    def get(self) -> dict: return dict(self._cache)
    def remove(self):
        for h in self._hooks: h.remove()

def run_episode(mario: Mario, agent_env, raw_env, act_cache: ActivationCache):
    state, _ = agent_env.reset() if hasattr(agent_env.reset(), '__len__') else (agent_env.reset(), {})
    raw_env.reset()
    if isinstance(state, tuple): state = state[0]

    total_reward, frames, done, flag_get, step_count = 0.0, [], False, False, 0

    while not done:
        action = mario.act(state)
        step_count += 1
        with torch.no_grad():
            state_t = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            _ = mario.net(state_t, model="online")

        activations = act_cache.get()
        result = agent_env.step(action)
        next_state, reward, terminated, truncated, info = result if len(result) == 5 else (*result, {})
        done = terminated or truncated
        total_reward += reward

        for _ in range(SKIP):
            raw_result = raw_env.step(action)
            raw_frame = raw_env.render()
            if raw_frame is not None:
                viz_panel = draw_nn_overlay(np.array(state), activations, action)
                gp_up = cv2.resize(raw_frame, (256 * UPSCALE, 240 * UPSCALE), interpolation=cv2.INTER_NEAREST)
                vz_up = cv2.resize(viz_panel, (240 * UPSCALE, 240 * UPSCALE), interpolation=cv2.INTER_NEAREST)
                combined = np.hstack((vz_up, gp_up))
                frames.append(combined)
            if (raw_result[2] if len(raw_result) < 5 else raw_result[2] or raw_result[3]): break

        if info.get("flag_get", False):
            flag_get, done = True, True
        state = next_state

    return total_reward, frames, flag_get, step_count

def save_video(frames: list, path: Path, fps: int = FPS):
    if not frames: return
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()

def save_metadata(results: list, best_run: int, best_reward: float, path: Path):
    rewards = [r["reward"] for r in results]
    metadata = {
        "summary": {
            "best_run": best_run,
            "best_reward": round(best_reward, 2),
            "flag_rate": round(sum(1 for r in results if r["flag_get"]) / NUM_RUNS, 2)
        },
        "runs": results
    }
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)

def main():
    eval_start = datetime.now(timezone.utc)
    print(f"\n[EVAL START] Device: {DEVICE} | CKPT: {CKPT_PATH.name}")
    
    _tmp = make_smb_env(render_mode="rgb_array")
    action_dim = _tmp.action_space.n
    _tmp.close()

    mario = load_agent(action_dim)
    act_cache = ActivationCache(mario.net)
    best_reward, best_frames, best_run, results = -np.inf, [], -1, []

    for run_idx in range(1, NUM_RUNS + 1):
        print(f"Run {run_idx}/{NUM_RUNS}...", end=" ", flush=True)
        run_start = datetime.now(timezone.utc)
        agent_env, raw_env = make_smb_env(render_mode="rgb_array"), make_raw_env()
        
        try:
            reward, frames, flag_get, steps = run_episode(mario, agent_env, raw_env, act_cache)
        finally:
            agent_env.close()
            raw_env.close()

        elapsed = (datetime.now(timezone.utc) - run_start).total_seconds()
        flag_status = "SUCCESS" if flag_get else "FAIL"
        print(f"R:{reward:.1f} | S:{steps} | {flag_status} | {elapsed:.1f}s")

        results.append({
            "run": run_idx, 
            "reward": round(float(reward), 2), 
            "steps": steps, 
            "flag_get": flag_get
        })
        
        if reward > best_reward:
            best_reward, best_frames, best_run = reward, frames, run_idx

    print(f"\n[SAVING] Best Run: {best_run} | Video: {VIDEO_OUT.name}")
    save_video(best_frames, VIDEO_OUT)
    save_metadata(results, best_run, best_reward, META_OUT)
    act_cache.remove()
    print(f"[FINISHED] Total Time: {(datetime.now(timezone.utc) - eval_start).total_seconds():.1f}s\n")

if __name__ == "__main__":
    main()