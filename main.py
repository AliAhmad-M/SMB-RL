#!/usr/bin/env python3
"""
Mario RL Studio - one-file GUI for training and testing the Mario DQN agent.

Put this file in the project root (next to train.py / eval.py) and run:

    pip install PySide6 matplotlib
    python mario_gui.py

Training / testing run in a child process (this same file launched with
--worker), so the window never freezes and Stop always works. The GUI process
itself never imports torch, so it opens instantly.
"""

import argparse
import json
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUTPUT_DIR = BASE / "output"
RUNS_DIR = BASE / "runs"
SETTINGS_FILE = BASE / ".mario_gui_settings.json"
JOB_FILE = BASE / ".mario_gui_job.json"
STOP_FILE = BASE / ".mario_gui_stop"
EVENT_PREFIX = "@@"

ACTIONS = [["right", "B"], ["right", "A", "B"]]
SKIP = 4

DEFAULT_AGENT = dict(
    eps_start=1.0, eps_decay=0.9993, eps_min=0.1,
    gamma=0.95,
    lr0=2.5e-4, lrf=1.0e-5, lr_gamma=0.9991,
    batch_size=64, learn_every=3, sync_every=10000, burnin=10000,
    buffer_size=75000,
)


# =============================================================================
#  WORKER SIDE  (runs in the child process; heavy imports happen lazily here)
# =============================================================================

def emit(**kw):
    print(EVENT_PREFIX + json.dumps(kw), flush=True)


def build_env(world, stage, render_mode="rgb_array"):
    """Same wrapper stack as environments.smb_env.make_smb_env, but with a level."""
    import warnings
    import gym_super_mario_bros
    from gym.wrappers import FrameStack
    from nes_py.wrappers import JoypadSpace
    from environments.wrappers import (SkipFrame, TileObservation, GrayScaleObservation,
                                       CropObservation, ResizeObservation)

    warnings.filterwarnings("ignore", category=UserWarning, module="gym")
    env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v0",
                                    render_mode=render_mode, apply_api_compatibility=True)
    env = JoypadSpace(env, ACTIONS)
    env = SkipFrame(env, skip=SKIP)
    env = TileObservation(env)
    env = CropObservation(env, top=48, bottom=16, left=32, right=32)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=12)
    env = FrameStack(env, num_stack=4)
    return env


def build_raw_env(world, stage):
    import warnings
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace

    warnings.filterwarnings("ignore", category=UserWarning, module="gym")
    env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v0",
                                    render_mode="rgb_array", apply_api_compatibility=True)
    return JoypadSpace(env, ACTIONS)


def pick_device(choice, torch):
    if choice == "CPU":
        return "cpu"
    if choice == "CUDA" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available - using CPU")
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_all(seed, torch):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_agent(a, action_dim, save_dir, device):
    """Create a Mario agent and apply the GUI's agent settings on top of its defaults."""
    import torch
    from agents.mario import Mario

    m = Mario(state_dim=(4, 12, 12), action_dim=action_dim, save_dir=save_dir, device=device)
    m.exploration_rate = a["eps_start"]
    m.exploration_rate_decay = a["eps_decay"]
    m.exploration_rate_min = a["eps_min"]
    m.discount = a["gamma"]
    m.batch_size = int(a["batch_size"])
    m.learn_every = int(a["learn_every"])
    m.sync_every = int(a["sync_every"])
    m.burnin = int(a["burnin"])
    m.lr0, m.lrf = a["lr0"], a["lrf"]

    for pg in m.optimizer.param_groups:
        pg["lr"] = m.lr0
        pg["initial_lr"] = m.lr0
    m.scheduler = torch.optim.lr_scheduler.ExponentialLR(m.optimizer, gamma=a["lr_gamma"])

    if int(a["buffer_size"]) != 75000:
        from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
        m.memory = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(int(a["buffer_size"]), device=torch.device("cpu")))
    return m


# ------------------------------ TRAINING ------------------------------------

def worker_train(cfg):
    import random
    import torch
    from metrics.logger import MetricLogger

    run, acfg = cfg["run"], cfg["agent"]
    device = pick_device(run["device"], torch)
    if run["seed"] is not None:
        seed_all(run["seed"], torch)

    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir = OUTPUT_DIR / (stamp + (f"_{run['name']}" if run["name"] else ""))
    save_dir.mkdir(parents=True, exist_ok=True)
    stop_file = Path(cfg["stop_file"])

    levels = [tuple(l) for l in cfg["levels"]]
    level_mode = cfg["level_mode"]
    envs = {}

    def get_env(lv):
        if lv not in envs:
            print(f"[env] building level {lv[0]}-{lv[1]}")
            envs[lv] = build_env(*lv)
        return envs[lv]

    def pick_level(i):
        if level_mode == "random":
            return random.choice(levels)
        if level_mode == "cycle":
            return levels[i % len(levels)]
        return levels[0]

    mario = build_agent(acfg, get_env(levels[0]).action_space.n, save_dir, device)
    original_burnin = mario.burnin

    S = dict(best=float("-inf"), rewards=[], flags=0, steps=0)

    def save_ckpt(tag, episode):
        path = save_dir / f"mario_{tag}.chkpt"
        payload = {
            "model": mario.net.state_dict(),
            "optimizer": mario.optimizer.state_dict(),
            "exploration_rate": mario.exploration_rate,
            "step": mario.curr_step,
            "episode": episode,
            "best_mean_reward": S["best"],
            "episode_rewards": S["rewards"],
            "flags_captured": S["flags"],
            "total_steps": S["steps"],
        }
        if hasattr(mario, "scheduler"):
            payload["scheduler"] = mario.scheduler.state_dict()
        torch.save(payload, path)
        return path

    # ---- starting point: new / resume / transfer ----
    start_episode = 0
    if cfg["start_mode"] in ("resume", "transfer"):
        ck_path = OUTPUT_DIR / cfg["checkpoint"]
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        mario.net.load_state_dict(ck["model"])
        full = "step" in ck and "optimizer" in ck
        if cfg["start_mode"] == "resume" and full:
            mario.exploration_rate = ck["exploration_rate"]
            mario.curr_step = ck["step"]
            mario.optimizer.load_state_dict(ck["optimizer"])
            if "scheduler" in ck:
                mario.scheduler.load_state_dict(ck["scheduler"])
            start_episode = ck.get("episode", 0) + 1
            mario.curr_episode = start_episode
            S.update(best=ck.get("best_mean_reward", float("-inf")),
                     rewards=list(ck.get("episode_rewards", [])),
                     flags=ck.get("flags_captured", 0),
                     steps=ck.get("total_steps", 0))
            # The replay buffer is not stored in checkpoints. Without a fresh burn-in the
            # agent would try to sample from an empty buffer right after resuming.
            mario.burnin = mario.curr_step + original_burnin
            print(f"Resumed {cfg['checkpoint']} at step {mario.curr_step}, episode {start_episode} "
                  f"(refilling replay buffer for {original_burnin} steps first)")
        else:
            if cfg["start_mode"] == "resume":
                print("[warn] Checkpoint has no training state (weights only) - "
                      "loading weights and starting at episode 0.")
            else:
                print(f"Transferred weights from {cfg['checkpoint']} (episode counter reset)")

    n_eps = run["episodes"]
    end_episode = start_episode + n_eps
    logger = MetricLogger(save_dir, checkpoint_every=run["log_every"], window=5)

    print(f"Device: {device} | levels: {', '.join(f'{w}-{s}' for w, s in levels)} ({level_mode})")
    print(f"Output: {save_dir}")
    emit(type="init", total=n_eps, run_dir=str(save_dir), device=device)

    stopped = False
    last_done = start_episode - 1

    for e in range(start_episode, end_episode):
        lv = pick_level(e - start_episode)
        env = get_env(lv)
        state = env.reset()
        ep_reward, ep_flag, ep_steps = 0.0, False, 0

        while True:
            action = mario.act(state)
            logger.update_epsilon(mario.exploration_rate)

            next_state, reward, done, trunc, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            S["steps"] += 1

            flag_get = bool(info.get("flag_get", False))
            ep_flag = ep_flag or flag_get

            mario.cache(state, next_state, action, reward, done or trunc)
            q, loss = mario.learn()

            grad_norm = None
            if loss is not None:
                grad_norm = sum(p.grad.data.norm(2).item() ** 2
                                for p in mario.net.parameters() if p.grad is not None) ** 0.5
            logger.log_step(reward, loss, q, grad_norm=grad_norm)

            state = next_state
            if S["steps"] % 200 == 0 and stop_file.exists():
                stopped = True
                break
            if done or trunc or flag_get:
                break

        if stopped:
            break

        S["flags"] += int(ep_flag)
        S["rewards"].append(ep_reward)
        logger.log_episode()
        mario.on_episode_end()
        last_done = e

        if e >= run["warmup"]:
            window = min(100, len(S["rewards"]))
            mean_reward = sum(S["rewards"][-window:]) / window
            if mean_reward > S["best"]:
                S["best"] = mean_reward
                path = save_ckpt("best", e)
                print(f"\tNew best mean reward: {S['best']:.3f} (ep {e + 1}) -> {path.name}")

        flag_rate = S["flags"] / (e + 1) * 100
        emit(type="episode", i=e - start_episode + 1, total=n_eps, ep=e + 1,
             reward=round(float(ep_reward), 2), flag=ep_flag, steps=ep_steps,
             level=f"{lv[0]}-{lv[1]}", eps=round(mario.exploration_rate, 4),
             lr=mario.get_current_lr(), flag_rate=round(flag_rate, 2))

        if (e - start_episode) % run["log_every"] == 0 or e == end_episode - 1:
            print(f"\tEpisode {e + 1} | level {lv[0]}-{lv[1]} | reward {ep_reward:.1f} | "
                  f"flags {S['flags']}/{e + 1} ({flag_rate:.1f}%) | eps {mario.exploration_rate:.3f} | "
                  f"lr {mario.get_current_lr():.6f}")
            logger.record_checkpoint(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

        if (e - start_episode + 1) % run["save_every"] == 0:
            print(f"\tCheckpoint saved -> {save_ckpt(f'ep{e + 1}', e).name}")

    save_ckpt("final", last_done)
    mario.save()
    for env in envs.values():
        env.close()

    done_eps = last_done - start_episode + 1
    print(f"\n{'Training stopped early' if stopped else 'Training complete'}.\n"
          f"  Episodes run  : {done_eps}\n"
          f"  Total steps   : {S['steps']}\n"
          f"  Flags captured: {S['flags']}\n"
          f"  Best mean reward (100-ep window): {S['best']:.3f}\n"
          f"  Saved: {save_dir / 'mario_final.chkpt'}")
    emit(type="done", stopped=stopped)


# ------------------------------- TESTING ------------------------------------

COL_WHITE, COL_BLACK = (255, 255, 255), (0, 0, 0)
COL_GRAY_NODE, COL_CYAN_NODE = (128, 128, 128), (255, 255, 0)
COL_PINK_LINE, COL_SUCCESS = (203, 105, 255), (76, 220, 115)
N_HIDDEN, NODE_R = 12, 5
NES_BUTTONS = ["U", "D", "L", "R", "A", "B"]
ACTION_BUTTONS = [{"R", "B"}, {"R", "A", "B"}]


def draw_nn_overlay(state, activations, chosen_action):
    import numpy as np
    import cv2

    canvas_h, canvas_w = 240, 240
    panel = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    grid_res = state.shape[1]
    cell = 8
    grid_size = grid_res * cell
    gx, gy = (canvas_w - grid_size) // 2, 20

    raw = state[-1].astype(np.uint8)
    for row in range(grid_res):
        for col in range(grid_res):
            val = int(raw[row, col])
            x1, y1 = gx + col * cell, gy + row * cell
            cv2.rectangle(panel, (x1, y1), (x1 + cell, y1 + cell), (val, val, val), -1)
            cv2.rectangle(panel, (x1, y1), (x1 + cell, y1 + cell), (230, 230, 230), 1)

    hidden_y = 150
    fc_vals = activations.get("fc", np.zeros(256))
    idx = np.linspace(0, len(fc_vals) - 1, N_HIDDEN, dtype=int)
    nodes_x = np.linspace(30, canvas_w - 30, N_HIDDEN).astype(int)

    anchor = (gx + grid_size // 2, gy + grid_size)
    for nx in nodes_x:
        cv2.line(panel, anchor, (nx, hidden_y - NODE_R), COL_PINK_LINE, 1, cv2.LINE_AA)
    for i, nx in enumerate(nodes_x):
        color = COL_CYAN_NODE if fc_vals[idx[i]] > 0.5 else COL_GRAY_NODE
        cv2.circle(panel, (nx, hidden_y), NODE_R, color, -1)
        cv2.circle(panel, (nx, hidden_y), NODE_R, COL_BLACK, 1)

    output_y = 210
    btn_x = np.linspace(40, canvas_w - 40, len(NES_BUTTONS)).astype(int)
    pressed = ACTION_BUTTONS[chosen_action] if chosen_action < len(ACTION_BUTTONS) else set()
    for name, bx in zip(NES_BUTTONS, btn_x):
        active = name in pressed
        line_col = (255, 120, 120) if active else (220, 220, 220)
        for nx in nodes_x:
            cv2.line(panel, (nx, hidden_y + NODE_R), (bx, output_y - NODE_R), line_col, 1)
        cv2.circle(panel, (bx, output_y), NODE_R, COL_SUCCESS if active else COL_WHITE, -1)
        cv2.circle(panel, (bx, output_y), NODE_R, COL_BLACK, 1)
        cv2.putText(panel, name, (bx - 4, output_y + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, COL_BLACK, 1, cv2.LINE_AA)
    return panel


class ActivationCache:
    def __init__(self, net):
        import torch
        self._cache, self._hooks = {}, []

        def hook(module, inp, out):
            with torch.no_grad():
                self._cache["fc"] = out.detach().float()[0].cpu().numpy()

        for _, module in net.named_modules():
            if isinstance(module, torch.nn.Linear):
                self._hooks.append(module.register_forward_hook(hook))
                break

    def get(self):
        return dict(self._cache)

    def remove(self):
        for h in self._hooks:
            h.remove()


def _raw_done(res):
    return bool(res[2]) or (len(res) > 4 and bool(res[3]))


def eval_episode(mario, env, raw_env, cache, o):
    import numpy as np
    import torch
    import cv2

    out = env.reset()
    state = out[0] if isinstance(out, tuple) else out
    record = o["record"]
    if record:
        raw_env.reset()

    total, frames, flag, steps, done = 0.0, [], False, 0, False
    up = o["upscale"]

    while not done and steps < o["max_steps"]:
        action = mario.act(state)
        steps += 1

        activations = {}
        if record and o["overlay"]:
            st = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(mario.device)
            mario.net(st, model="online")
            activations = cache.get()

        res = env.step(action)
        if len(res) == 5:
            nxt, reward, term, trunc, info = res
            done = term or trunc
        else:
            nxt, reward, done, info = res
        total += reward

        if record:
            panel = None
            if o["overlay"]:
                panel = cv2.resize(draw_nn_overlay(np.array(state), activations, action),
                                   (240 * up, 240 * up), interpolation=cv2.INTER_NEAREST)
            for _ in range(SKIP):
                rr = raw_env.step(action)
                frame = raw_env.render()
                if frame is not None:
                    game = cv2.resize(frame, (256 * up, 240 * up), interpolation=cv2.INTER_NEAREST)
                    frames.append(np.hstack((panel, game)) if panel is not None else game)
                if _raw_done(rr):
                    break

        if info.get("flag_get", False):
            flag, done = True, True
        state = nxt

    return total, frames, flag, steps


def save_video(frames, path, fps):
    import cv2
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def worker_test(cfg):
    import numpy as np
    import torch

    device = pick_device(cfg["device"], torch)
    if cfg["seed"] is not None:
        seed_all(cfg["seed"], torch)

    world, stage = cfg["level"]
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir = RUNS_DIR / f"{stamp}_L{world}-{stage}"
    save_dir.mkdir(parents=True, exist_ok=True)
    stop_file = Path(cfg["stop_file"])
    n_runs = cfg["runs"]
    record = cfg["record"]

    print(f"[EVAL] Device: {device} | checkpoint: {cfg['checkpoint']} | level {world}-{stage}")
    emit(type="init", total=n_runs, run_dir=str(save_dir), device=device)

    env = build_env(world, stage)
    raw_env = build_raw_env(world, stage) if record else None

    agent_cfg = dict(DEFAULT_AGENT, buffer_size=1000)
    mario = build_agent(agent_cfg, env.action_space.n, save_dir, device)
    ck = torch.load(OUTPUT_DIR / cfg["checkpoint"], map_location=device, weights_only=False)
    mario.net.load_state_dict(ck["model"])
    mario.net.eval()
    mario.exploration_rate = cfg["epsilon"]

    cache = ActivationCache(mario.net) if (record and cfg["overlay"]) else None
    opts = dict(record=record, overlay=cfg["overlay"], upscale=cfg["upscale"], max_steps=cfg["max_steps"])

    results, best_reward, best_frames, best_run = [], -np.inf, [], -1
    t_start = time.time()

    with torch.no_grad():
        for i in range(1, n_runs + 1):
            if stop_file.exists():
                print("[EVAL] Stopped by user")
                break
            t0 = time.time()
            reward, frames, flag, steps = eval_episode(mario, env, raw_env, cache, opts)
            elapsed = time.time() - t0
            print(f"Run {i}/{n_runs} | R:{reward:.1f} | S:{steps} | "
                  f"{'SUCCESS' if flag else 'FAIL'} | {elapsed:.1f}s")
            results.append(dict(run=i, reward=round(float(reward), 2), steps=steps, flag_get=flag))
            emit(type="run", i=i, total=n_runs, reward=round(float(reward), 2), steps=steps,
                 flag=flag, elapsed=round(elapsed, 1))
            if reward > best_reward:
                best_reward, best_run = reward, i
                best_frames = frames  # old best frames are released here

    if cache:
        cache.remove()
    env.close()
    if raw_env:
        raw_env.close()

    video_path = None
    if results:
        n = len(results)
        rewards = [r["reward"] for r in results]
        meta = {
            "checkpoint": cfg["checkpoint"],
            "level": f"{world}-{stage}",
            "epsilon": cfg["epsilon"],
            "summary": {
                "runs": n,
                "best_run": best_run,
                "best_reward": round(float(best_reward), 2),
                "mean_reward": round(float(np.mean(rewards)), 2),
                "flag_rate": round(sum(r["flag_get"] for r in results) / n, 3),
            },
            "runs": results,
        }
        with open(save_dir / "eval_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        if record and best_frames:
            video_path = save_dir / "best_run.mp4"
            print(f"[SAVING] Best run {best_run} -> {video_path}")
            save_video(best_frames, video_path, cfg["fps"])
            emit(type="video", path=str(video_path))
        s = meta["summary"]
        print(f"\n[FINISHED] {n} runs | flag rate {s['flag_rate'] * 100:.0f}% | "
              f"mean reward {s['mean_reward']} | best {s['best_reward']} (run {best_run}) | "
              f"{time.time() - t_start:.1f}s\nSaved to {save_dir}")
    emit(type="done", stopped=len(results) < n_runs)


def worker_main(mode, config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    try:
        (worker_train if mode == "train" else worker_test)(cfg)
    except Exception:
        traceback.print_exc()
        sys.exit(1)



# =============================================================================
#  GUI SIDE  (PySide6)
# =============================================================================

def run_gui():
    from string import Template
    from PySide6.QtCore import Qt, QProcess, QProcessEnvironment, QTimer, QUrl, Signal
    from PySide6.QtGui import QColor, QDesktopServices, QPalette
    from PySide6.QtWidgets import (
        QApplication, QButtonGroup, QCheckBox, QComboBox,
        QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
        QSizePolicy, QSplitter, QTabWidget, QVBoxLayout, QWidget)

    try:
        import numpy as np
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        has_mpl = True
    except Exception:
        has_mpl = False

    NO_CKPT = "(no checkpoints in output/)"

    AGENT_FIELDS = [
        ("Epsilon start", "eps_start", "Exploration at the start of a new/transfer run"),
        ("Epsilon decay / ep", "eps_decay", "Multiplied in after every episode"),
        ("Epsilon min", "eps_min", ""),
        ("Discount (gamma)", "gamma", ""),
        ("Learning rate", "lr0", "Initial LR (a resumed run keeps the checkpoint's LR)"),
        ("Min learning rate", "lrf", "LR floor"),
        ("LR decay / ep", "lr_gamma", "ExponentialLR gamma per episode"),
        ("Batch size", "batch_size", ""),
        ("Learn every N steps", "learn_every", ""),
        ("Target sync (steps)", "sync_every", ""),
        ("Burn-in (steps)", "burnin", "Steps collected before learning starts"),
        ("Replay buffer size", "buffer_size", "Transitions kept in memory"),
    ]
    INT_FIELDS = ("batch_size", "learn_every", "sync_every", "burnin", "buffer_size")

    THEMES = {
        "Dark": dict(window="#202124", base="#2b2d31", card="#2b2d31", text="#e8eaed", muted="#9aa0a6",
                     button="#35373c", hover="#41444a", border="#43464c", accent="#3b8ed0",
                     disabled="#2a2b2e", dtext="#6b6f76", good="#2fa572", bad="#d9534f"),
        "Light": dict(window="#f3f4f6", base="#ffffff", card="#ffffff", text="#1f2328", muted="#6b7280",
                      button="#e6e8eb", hover="#d9dce0", border="#cfd3d8", accent="#2b6cb0",
                      disabled="#e3e5e8", dtext="#9aa0a6", good="#2fa572", bad="#d9534f"),
    }

    QSS = Template("""
        QWidget { font-size: 13px; }
        QMainWindow, QDialog { background: $window; }
        QGroupBox { border: 1px solid $border; border-radius: 8px; margin-top: 12px;
                    padding: 14px 8px 8px 8px; font-weight: 600; }
        QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
        QLineEdit, QComboBox { padding: 5px 8px; border: 1px solid $border; border-radius: 6px;
                               background: $base; min-height: 20px; }
        QLineEdit:focus, QComboBox:focus { border-color: $accent; }
        QComboBox::drop-down { border: none; width: 22px; }
        QComboBox QAbstractItemView { background: $base; color: $text; border: 1px solid $border;
                                      selection-background-color: $accent; selection-color: white; outline: 0; }
        QPushButton { padding: 6px 12px; border-radius: 6px; background: $button; border: 1px solid $border; }
        QPushButton:hover { background: $hover; }
        QPushButton:disabled { color: $dtext; background: $disabled; }
        QPushButton#start, QPushButton#stop { color: white; font-weight: 700; font-size: 14px;
                                              border: none; padding: 11px; }
        QPushButton#start { background: #2fa572; }
        QPushButton#start:hover { background: #248a5d; }
        QPushButton#stop { background: #c0392b; }
        QPushButton#stop:hover { background: #992d22; }
        QPushButton#start:disabled, QPushButton#stop:disabled { background: $disabled; color: $dtext; }
        QPushButton[seg="true"] { border-radius: 0; padding: 5px 12px; margin: 0; }
        QPushButton[seg="true"]:checked { background: $accent; color: white; border-color: $accent; }
        QFrame#card { background: $card; border: 1px solid $border; border-radius: 10px; }
        QLabel#muted { color: $muted; font-size: 11px; }
        QLabel#value { font-size: 19px; font-weight: 700; }
        QLabel#title { font-size: 22px; font-weight: 700; }
        QProgressBar { border: none; border-radius: 5px; background: $card; max-height: 10px; min-height: 10px; }
        QProgressBar::chunk { background: $accent; border-radius: 5px; }
        QPlainTextEdit { font-family: Consolas, "Cascadia Mono", monospace; font-size: 12px;
                         background: $base; border: 1px solid $border; border-radius: 8px; }
        QTabWidget::pane { border: none; }
        QTabBar::tab { padding: 8px 24px; border-radius: 6px; margin-right: 4px; background: transparent; }
        QTabBar::tab:selected { background: $accent; color: white; }
        QScrollArea { border: none; background: transparent; }
        QScrollArea > QWidget > QWidget { background: transparent; }
    """)

    def apply_theme(app, name):
        t = THEMES[name]
        app.setStyle("Fusion")
        pal = QPalette()
        R = QPalette.ColorRole
        for role, key in [(R.Window, "window"), (R.WindowText, "text"), (R.Base, "base"),
                          (R.AlternateBase, "card"), (R.Text, "text"), (R.Button, "button"),
                          (R.ButtonText, "text"), (R.ToolTipBase, "base"), (R.ToolTipText, "text"),
                          (R.Highlight, "accent"), (R.PlaceholderText, "muted")]:
            pal.setColor(role, QColor(t[key]))
        pal.setColor(R.HighlightedText, QColor("white"))
        for role in (R.Text, R.ButtonText, R.WindowText):
            pal.setColor(QPalette.ColorGroup.Disabled, role, QColor(t["dtext"]))
        app.setPalette(pal)
        app.setStyleSheet(QSS.safe_substitute(t))

    def fmt_time(sec):
        sec = int(max(0, sec))
        h, r = divmod(sec, 3600)
        m, s = divmod(r, 60)
        return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")

    def open_path(p):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    # ------------------------------------------------------------------ widgets
    class Segmented(QWidget):
        changed = Signal(str)

        def __init__(self, values):
            super().__init__()
            lay = QHBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)
            self.group = QButtonGroup(self)
            self.group.setExclusive(True)
            self.buttons = {}
            for v in values:
                b = QPushButton(v)
                b.setCheckable(True)
                b.setProperty("seg", True)
                b.setCursor(Qt.CursorShape.PointingHandCursor)
                b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                b.clicked.connect(lambda _=False, val=v: self.changed.emit(val))
                lay.addWidget(b)
                self.group.addButton(b)
                self.buttons[v] = b
            self.buttons[values[0]].setChecked(True)

        def currentText(self):
            for v, b in self.buttons.items():
                if b.isChecked():
                    return v
            return ""

        def setCurrentText(self, text):
            if text in self.buttons:
                self.buttons[text].setChecked(True)

    class StatCard(QFrame):
        def __init__(self):
            super().__init__()
            self.setObjectName("card")
            lay = QVBoxLayout(self)
            lay.setContentsMargins(12, 8, 12, 8)
            lay.setSpacing(0)
            self.title = QLabel("")
            self.title.setObjectName("muted")
            self.value = QLabel("-")
            self.value.setObjectName("value")
            lay.addWidget(self.title)
            lay.addWidget(self.value)

        def set(self, title=None, value=None):
            if title is not None:
                self.title.setText(title)
            if value is not None:
                self.value.setText(value)

    # ------------------------------------------------------------------ window
    class MainWindow(QMainWindow):
        def __init__(self, settings):
            super().__init__()
            self.setWindowTitle("Mario RL Studio")
            self.resize(1400, 900)
            self.setMinimumSize(1120, 720)

            self.settings = settings
            self.theme = settings.get("appearance") if settings.get("appearance") in THEMES else "Dark"
            self.getters = {}
            self.widgets = {}
            self.ckpt_pickers = []
            self.ckpt_index = {}
            self.device_segs = []
            self.proc = None
            self._buf = ""
            self.mode = "train"
            self.stopping = False
            self.run_dir = None
            self.last_video = None
            self.t_start = 0.0
            self.d = {}
            self.plot_dirty = False

            self._build_ui()
            self._refresh_ckpts()
            self._update_start_widgets()
            self._reset_dashboard("train")
            if has_mpl:
                self._redraw()

            self.timer = QTimer(self)
            self.timer.timeout.connect(self._tick)
            self.timer.start(700)

        # --------------------------------------------------------- settings
        def _save_settings(self):
            data = {k: g() for k, g in self.getters.items()}
            data["appearance"] = self.theme
            for entry in self.ckpt_pickers:
                rel = self._ckpt_rel(entry)
                if rel:
                    data[entry[0]] = rel
            try:
                SETTINGS_FILE.write_text(json.dumps(data, indent=2))
            except Exception:
                pass

        def _reg(self, key, w, default):
            self.widgets[key] = w
            saved = self.settings.get(key, default)
            if isinstance(w, QLineEdit):
                w.setText(str(saved))
                self.getters[key] = w.text
            elif isinstance(w, QComboBox):
                i = w.findText(str(saved))
                if i >= 0:
                    w.setCurrentIndex(i)
                self.getters[key] = w.currentText
            elif isinstance(w, Segmented):
                w.setCurrentText(str(saved))
                self.getters[key] = w.currentText
            else:  # QCheckBox / checkable QGroupBox
                w.setChecked(bool(saved))
                self.getters[key] = w.isChecked
            return w

        def val(self, key):
            return self.getters[key]()

        # --------------------------------------------------------- UI build
        def _build_ui(self):
            split = QSplitter(Qt.Orientation.Horizontal)
            self.setCentralWidget(split)

            # ---------------- left: settings
            left = QWidget()
            lv = QVBoxLayout(left)
            lv.setContentsMargins(14, 14, 8, 14)

            head = QHBoxLayout()
            title = QLabel("Mario RL Studio")
            title.setObjectName("title")
            head.addWidget(title)
            head.addStretch(1)
            self.theme_seg = Segmented(list(THEMES))
            self.theme_seg.setCurrentText(self.theme)
            self.theme_seg.setFixedWidth(160)
            self.theme_seg.changed.connect(self._set_theme)
            head.addWidget(self.theme_seg)
            lv.addLayout(head)

            self.tabs = QTabWidget()
            self.tabs.addTab(self._scroll(self._build_train_tab()), "Train")
            self.tabs.addTab(self._scroll(self._build_test_tab()), "Test")
            lv.addWidget(self.tabs, 1)

            bar = QHBoxLayout()
            self.btn_start = QPushButton("▶  Start")
            self.btn_start.setObjectName("start")
            self.btn_start.clicked.connect(self.start)
            self.btn_stop = QPushButton("■  Stop")
            self.btn_stop.setObjectName("stop")
            self.btn_stop.setEnabled(False)
            self.btn_stop.clicked.connect(self.stop)
            bar.addWidget(self.btn_start)
            bar.addWidget(self.btn_stop)
            lv.addLayout(bar)
            self.status = QLabel("Idle")
            self.status.setObjectName("muted")
            lv.addWidget(self.status)
            split.addWidget(left)

            # ---------------- right: dashboard
            right = QWidget()
            rv = QVBoxLayout(right)
            rv.setContentsMargins(8, 14, 14, 14)

            cards = QHBoxLayout()
            self.cards = []
            for _ in range(6):
                c = StatCard()
                cards.addWidget(c)
                self.cards.append(c)
            rv.addLayout(cards)

            self.progress = QProgressBar()
            self.progress.setRange(0, 1000)
            self.progress.setTextVisible(False)
            rv.addSpacing(6)
            rv.addWidget(self.progress)
            self.prog_label = QLabel("")
            self.prog_label.setObjectName("muted")
            rv.addWidget(self.prog_label)

            if has_mpl:
                self.fig = Figure(figsize=(6, 3), dpi=100)
                self.ax1 = self.fig.add_subplot(121)
                self.ax2 = self.fig.add_subplot(122)
                self.fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.88, wspace=0.2)
                self.canvas = FigureCanvasQTAgg(self.fig)
                self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                rv.addWidget(self.canvas, 3)
            else:
                lbl = QLabel("Install matplotlib for live charts:  pip install matplotlib")
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                rv.addWidget(lbl, 3)

            tools = QHBoxLayout()
            tools.addWidget(QLabel("<b>Console</b>"))
            tools.addStretch(1)
            b_folder = QPushButton("Open run folder")
            b_folder.clicked.connect(self._open_run_dir)
            self.btn_video = QPushButton("Play last video")
            self.btn_video.setEnabled(False)
            self.btn_video.clicked.connect(lambda: open_path(self.last_video))
            b_clear = QPushButton("Clear")
            b_clear.clicked.connect(lambda: self.log.clear())
            for b in (b_folder, self.btn_video, b_clear):
                tools.addWidget(b)
            rv.addLayout(tools)

            self.log = QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumBlockCount(4000)
            self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
            rv.addWidget(self.log, 2)
            split.addWidget(right)

            split.setStretchFactor(0, 0)
            split.setStretchFactor(1, 1)
            split.setSizes([480, 920])

        @staticmethod
        def _scroll(widget):
            sa = QScrollArea()
            sa.setWidgetResizable(True)
            sa.setWidget(widget)
            sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            return sa

        # ---- form helpers
        def _group(self, title, layout):
            g = QGroupBox(title)
            form = QFormLayout(g)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
            layout.addWidget(g)
            return g, form

        @staticmethod
        def _row(form, label, widget, hint=None):
            if hint:
                box = QWidget()
                v = QVBoxLayout(box)
                v.setContentsMargins(0, 0, 0, 0)
                v.setSpacing(2)
                v.addWidget(widget)
                h = QLabel(hint)
                h.setObjectName("muted")
                h.setWordWrap(True)
                v.addWidget(h)
                widget = box
            if label is None:
                form.addRow(widget)
            else:
                form.addRow(label, widget)

        def _entry(self, form, label, key, default, hint=None, width=150):
            e = QLineEdit()
            self._reg(key, e, default)
            self._row(form, label, e, hint)

        def _level_picker(self, form, key):
            box = QWidget()
            h = QHBoxLayout(box)
            h.setContentsMargins(0, 0, 0, 0)
            cw, cs = QComboBox(), QComboBox()
            cw.addItems([str(i) for i in range(1, 9)])
            cs.addItems([str(i) for i in range(1, 5)])
            self._reg(f"{key}_world", cw, "1")
            self._reg(f"{key}_stage", cs, "1")
            h.addWidget(QLabel("World"))
            h.addWidget(cw)
            h.addSpacing(8)
            h.addWidget(QLabel("Stage"))
            h.addWidget(cs)
            h.addStretch(1)
            form.addRow("Level", box)
            return cw, cs

        def _ckpt_rows(self, form, key, hint=None):
            run, file = QComboBox(), QComboBox()
            for c in (run, file):
                c.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                c.setMinimumContentsLength(14)
                c.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            box = QWidget()
            h = QHBoxLayout(box)
            h.setContentsMargins(0, 0, 0, 0)
            btn = QPushButton("⟳")
            btn.setFixedWidth(36)
            btn.setToolTip("Rescan output/ for checkpoints")
            btn.clicked.connect(self._refresh_ckpts)
            h.addWidget(run, 1)
            h.addWidget(btn)
            run.currentTextChanged.connect(lambda _t, r=run, f=file: self._fill_files(r, f))
            form.addRow("Run folder", box)
            self._row(form, "Checkpoint", file, hint)
            entry = (key, run, file)
            self.ckpt_pickers.append(entry)
            return entry

        def _device_row(self, form):
            seg = Segmented(["Auto", "CUDA", "CPU"])
            seg.setCurrentText(self.settings.get("device", "Auto"))
            seg.changed.connect(self._sync_device)
            self.device_segs.append(seg)
            self.getters["device"] = lambda: self.device_segs[0].currentText()
            form.addRow("Device", seg)

        def _sync_device(self, v):
            for s in self.device_segs:
                s.setCurrentText(v)

        # ---- train tab
        def _build_train_tab(self):
            page = QWidget()
            lay = QVBoxLayout(page)
            lay.setContentsMargins(2, 6, 8, 6)

            _, f = self._group("Level", lay)
            self._level_picker(f, "train_level")

            _, f = self._group("Run", lay)
            self._entry(f, "Episodes to train", "episodes", 4000)
            self._entry(f, "Checkpoint every (eps)", "save_every", 20)
            self._entry(f, "Log every (eps)", "log_every", 20)
            self._entry(f, "Best-model warmup (eps)", "warmup", 80)
            self._entry(f, "Seed (blank = random)", "seed", "")
            self._entry(f, "Run name (optional)", "run_name", "", width=220)
            self._device_row(f)

            _, f = self._group("Start from", lay)
            self.start_mode = Segmented(["New", "Resume", "Transfer"])
            self._reg("start_mode", self.start_mode, "New")
            self.start_mode.changed.connect(lambda _: self._update_start_widgets())
            f.addRow("Mode", self.start_mode)
            self.train_ckpt = self._ckpt_rows(f, "train_ckpt",
                                             "Resume continues the run; Transfer loads weights only.")

            g = QGroupBox("Additional settings (agent configuration)")
            g.setCheckable(True)
            gl = QVBoxLayout(g)
            self.adv = QWidget()
            af = QFormLayout(self.adv)
            af.setContentsMargins(0, 0, 0, 0)
            af.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            for label, key, hint in AGENT_FIELDS:
                self._entry(af, label, f"agent_{key}", DEFAULT_AGENT[key], hint or None)
            reset = QPushButton("Reset to defaults")
            reset.clicked.connect(self._reset_agent)
            af.addRow(reset)
            gl.addWidget(self.adv)
            self._reg("show_adv", g, False)
            g.toggled.connect(self.adv.setVisible)
            self.adv.setVisible(g.isChecked())
            lay.addWidget(g)

            lay.addStretch(1)
            return page

        def _reset_agent(self):
            for _, key, _h in AGENT_FIELDS:
                self.widgets[f"agent_{key}"].setText(str(DEFAULT_AGENT[key]))

        def _update_start_widgets(self):
            on = self.start_mode.currentText() != "New"
            self.train_ckpt[1].setEnabled(on)
            self.train_ckpt[2].setEnabled(on)

        # ---- test tab
        def _build_test_tab(self):
            page = QWidget()
            lay = QVBoxLayout(page)
            lay.setContentsMargins(2, 6, 8, 6)

            _, f = self._group("Model", lay)
            self.test_ckpt = self._ckpt_rows(f, "test_ckpt")
            self._device_row(f)

            _, f = self._group("Evaluation", lay)
            self._level_picker(f, "test_level")
            self._entry(f, "Number of runs", "test_runs", 25)
            self._entry(f, "Epsilon during test", "test_eps", 0.05)
            self._entry(f, "Max steps per run", "test_max_steps", 4000, "Ends runs where Mario gets stuck.")
            self._entry(f, "Seed (blank = random)", "test_seed", "")

            _, f = self._group("Video", lay)
            self.rec = QCheckBox("Record video of the best run")
            self._reg("test_record", self.rec, True)
            f.addRow(self.rec)
            self.overlay = QCheckBox("Show neural-net overlay")
            self._reg("test_overlay", self.overlay, True)
            f.addRow(self.overlay)
            self._entry(f, "Upscale", "test_upscale", 3, width=90)
            self._entry(f, "FPS", "test_fps", 60, width=90)
            note = QLabel("Turning recording off makes testing much faster.")
            note.setObjectName("muted")
            f.addRow(note)

            lay.addStretch(1)
            return page

        # --------------------------------------------------------- helpers
        def _set_theme(self, name):
            self.theme = name
            apply_theme(QApplication.instance(), name)
            self.plot_dirty = True

        def _scan_ckpts(self):
            found = {}
            if OUTPUT_DIR.exists():
                for p in OUTPUT_DIR.rglob("*.chkpt"):
                    found.setdefault(p.parent.relative_to(OUTPUT_DIR).as_posix(), []).append(p)

            def fkey(p):
                n = p.stem
                if n == "mario_final":
                    return (0, 0, n)
                if n == "mario_best":
                    return (1, 0, n)
                m = re.fullmatch(r"mario_ep(\d+)", n)
                if m:
                    return (2, -int(m.group(1)), n)
                m = re.fullmatch(r"mario_net_(\d+)", n)
                if m:
                    return (3, -int(m.group(1)), n)
                return (4, 0, n)

            folders = sorted(found, key=lambda f: max(p.stat().st_mtime for p in found[f]), reverse=True)
            self.ckpt_index = {f: [p.name for p in sorted(found[f], key=fkey)] for f in folders}

        def _fill_files(self, run, file):
            file.blockSignals(True)
            file.clear()
            file.addItems(self.ckpt_index.get(run.currentText(), []))
            file.blockSignals(False)

        def _ckpt_rel(self, entry):
            _, run, file = entry
            folder, name = run.currentText(), file.currentText()
            if folder not in self.ckpt_index or not name:
                return ""
            return name if folder == "." else f"{folder}/{name}"

        def _refresh_ckpts(self):
            self._scan_ckpts()
            for entry in self.ckpt_pickers:
                key, run, file = entry
                prev = self._ckpt_rel(entry) or self.settings.get(key, "")
                folder, _, name = prev.rpartition("/")
                folder = folder or "."
                run.blockSignals(True)
                run.clear()
                if self.ckpt_index:
                    run.addItems(list(self.ckpt_index))
                    i = run.findText(folder)
                    run.setCurrentIndex(i if i >= 0 else 0)
                else:
                    run.addItem(NO_CKPT)
                run.blockSignals(False)
                self._fill_files(run, file)
                j = file.findText(name)
                if j >= 0:
                    file.setCurrentIndex(j)

        def _num(self, key, label, cast=float, lo=None, hi=None):
            raw = self.val(key).strip()
            try:
                v = float(raw)
                if cast is int:
                    v = int(v)
            except ValueError:
                raise ValueError(f"{label}: '{raw}' is not a valid number")
            if lo is not None and v < lo:
                raise ValueError(f"{label} must be at least {lo}")
            if hi is not None and v > hi:
                raise ValueError(f"{label} must be at most {hi}")
            return v

        def _seed(self, key):
            raw = self.val(key).strip()
            try:
                return int(raw) if raw else None
            except ValueError:
                raise ValueError(f"Seed: '{raw}' is not a whole number")

        def _ckpt_name(self, entry):
            name = self._ckpt_rel(entry)
            if not name or not (OUTPUT_DIR / name).is_file():
                raise ValueError("Pick a checkpoint from the output folder first.")
            return name

        def _collect_train(self):
            levels = [[int(self.val("train_level_world")), int(self.val("train_level_stage"))]]
            agent = {}
            for _, key, _h in AGENT_FIELDS:
                agent[key] = self._num(f"agent_{key}", key, int if key in INT_FIELDS else float, lo=0)
            if agent["batch_size"] < 1 or agent["learn_every"] < 1 or agent["sync_every"] < 1:
                raise ValueError("Batch size, learn-every and sync-every must be at least 1.")
            if agent["buffer_size"] < agent["batch_size"]:
                raise ValueError("Replay buffer must be at least as large as the batch size.")
            start = self.start_mode.currentText().lower()
            return dict(
                run=dict(episodes=self._num("episodes", "Episodes", int, lo=1),
                         save_every=self._num("save_every", "Checkpoint every", int, lo=1),
                         log_every=self._num("log_every", "Log every", int, lo=1),
                         warmup=self._num("warmup", "Warmup", int, lo=0),
                         seed=self._seed("seed"),
                         name=re.sub(r"[^A-Za-z0-9_-]", "", self.val("run_name")),
                         device=self.val("device")),
                level_mode="single", levels=levels, start_mode=start,
                checkpoint=self._ckpt_name(self.train_ckpt) if start != "new" else "",
                agent=agent, stop_file=str(STOP_FILE))

        def _collect_test(self):
            return dict(
                checkpoint=self._ckpt_name(self.test_ckpt),
                level=[int(self.val("test_level_world")), int(self.val("test_level_stage"))],
                runs=self._num("test_runs", "Number of runs", int, lo=1),
                epsilon=self._num("test_eps", "Epsilon", float, lo=0, hi=1),
                max_steps=self._num("test_max_steps", "Max steps", int, lo=10),
                record=self.rec.isChecked(), overlay=self.overlay.isChecked(),
                upscale=self._num("test_upscale", "Upscale", int, lo=1, hi=8),
                fps=self._num("test_fps", "FPS", int, lo=1, hi=240),
                seed=self._seed("test_seed"), device=self.val("device"),
                stop_file=str(STOP_FILE))

        # --------------------------------------------------------- run control
        def start(self):
            if self.proc is not None:
                return
            mode = "train" if self.tabs.currentIndex() == 0 else "test"
            try:
                cfg = self._collect_train() if mode == "train" else self._collect_test()
            except ValueError as e:
                QMessageBox.critical(self, "Invalid settings", str(e))
                return
            self._save_settings()
            JOB_FILE.write_text(json.dumps(cfg))
            if STOP_FILE.exists():
                STOP_FILE.unlink()

            self.mode, self.stopping, self._buf = mode, False, ""
            self._reset_dashboard(mode)
            self.log.clear()
            self.t_start = time.time()

            proc = QProcess(self)
            env = QProcessEnvironment.systemEnvironment()
            env.insert("PYTHONIOENCODING", "utf-8")
            env.insert("PYTHONUTF8", "1")
            proc.setProcessEnvironment(env)
            proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            proc.setWorkingDirectory(str(BASE))
            proc.readyReadStandardOutput.connect(self._on_stdout)
            proc.finished.connect(self._on_finished)
            proc.errorOccurred.connect(self._on_error)
            self.proc = proc
            proc.start(sys.executable, ["-u", str(Path(__file__).resolve()),
                                        "--worker", mode, "--config", str(JOB_FILE)])
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_stop.setText("■  Stop")
            self._set_status("Starting…")

        def stop(self):
            if self.proc is None:
                return
            if not self.stopping:
                self.stopping = True
                STOP_FILE.write_text("stop")
                self.btn_stop.setText("Force kill")
                self._set_status("Stopping… saving state (click again to force kill)")
            else:
                self.proc.kill()

        def _on_error(self, err):
            if err == QProcess.ProcessError.FailedToStart:
                QMessageBox.critical(self, "Could not start", "Failed to launch the Python worker process.")

        def _on_stdout(self):
            data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
            self._buf += data
            *lines, self._buf = self._buf.split("\n")
            for line in lines:
                self._handle_line(line.rstrip("\r"))

        def _handle_line(self, line):
            if line.startswith(EVENT_PREFIX):
                try:
                    self._on_event(json.loads(line[len(EVENT_PREFIX):]))
                except Exception:
                    pass
            else:
                self.log.appendPlainText(line)

        def _on_finished(self, code, _status):
            if self._buf.strip():
                self._handle_line(self._buf.strip())
            self._buf = ""
            self.proc = None
            if STOP_FILE.exists():
                STOP_FILE.unlink()
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.btn_stop.setText("■  Stop")
            if self.stopping:
                self._set_status("Stopped")
            elif code != 0:
                self._set_status("Failed - see console")
            else:
                self._set_status(f"Finished in {fmt_time(time.time() - self.t_start)}")
                self.progress.setValue(1000)
            self._refresh_ckpts()
            self.plot_dirty = True

        def _set_status(self, text):
            self.status.setText(text)

        # --------------------------------------------------------- events
        def _on_event(self, ev):
            t, c = ev.get("type"), self.cards
            if t == "init":
                self.run_dir = ev["run_dir"]
                self._set_status(f"{'Training' if self.mode == 'train' else 'Testing'} on {ev['device'].upper()}…")
            elif t == "episode":
                d = self.d
                d["rewards"].append(ev["reward"])
                d["flag_rate"].append(ev["flag_rate"])
                last = d["rewards"][-20:]
                c[0].set(value=f"{ev['i']} / {ev['total']}")
                c[1].set(value=f"{ev['reward']:.0f}" + ("  🏁" if ev["flag"] else ""))
                c[2].set(value=f"{sum(last) / len(last):.0f}")
                c[3].set(value=f"{ev['flag_rate']:.1f}%")
                c[4].set(value=f"{ev['eps']:.3f}")
                c[5].set(value=f"{ev['lr']:.2e}")
                self._progress(ev["i"], ev["total"], f"Level {ev['level']}")
                self.plot_dirty = True
            elif t == "run":
                runs = self.d["runs"]
                runs.append(ev)
                n = len(runs)
                rewards = [r["reward"] for r in runs]
                wins = sum(r["flag"] for r in runs)
                c[0].set(value=f"{ev['i']} / {ev['total']}")
                c[1].set(value=f"{ev['reward']:.0f}" + ("  🏁" if ev["flag"] else ""))
                c[2].set(value=f"{sum(rewards) / n:.0f}")
                c[3].set(value=f"{wins / n * 100:.0f}%  ({wins}/{n})")
                c[4].set(value=f"{max(rewards):.0f}")
                c[5].set(value=f"{sum(r['steps'] for r in runs) / n:.0f}")
                self._progress(ev["i"], ev["total"], "")
                self.plot_dirty = True
            elif t == "video":
                self.last_video = ev["path"]
                self.btn_video.setEnabled(True)

        def _progress(self, i, total, extra):
            frac = i / max(1, total)
            self.progress.setValue(int(frac * 1000))
            elapsed = time.time() - self.t_start
            eta = elapsed / max(frac, 1e-9) * (1 - frac)
            self.prog_label.setText(f"{i}/{total}  ({frac * 100:.1f}%)    elapsed {fmt_time(elapsed)}"
                                    f"    ETA {fmt_time(eta)}    {extra}")

        def _reset_dashboard(self, mode):
            titles = (["Episode", "Reward", "Avg reward (20)", "Flag rate", "Epsilon", "Learning rate"]
                      if mode == "train" else
                      ["Run", "Reward", "Mean reward", "Flag rate", "Best reward", "Avg steps"])
            for card, title in zip(self.cards, titles):
                card.set(title=title, value="-")
            self.d = dict(rewards=[], flag_rate=[], runs=[])
            self.progress.setValue(0)
            self.prog_label.setText("")
            self.plot_dirty = True

        # --------------------------------------------------------- plotting
        def _tick(self):
            if has_mpl and self.plot_dirty:
                self._redraw()

        def _redraw(self):
            self.plot_dirty = False
            t = THEMES[self.theme]
            self.fig.set_facecolor(t["base"])
            for ax in (self.ax1, self.ax2):
                ax.clear()
                ax.set_facecolor(t["base"])
                ax.tick_params(colors=t["text"], labelsize=8)
                for sp in ax.spines.values():
                    sp.set_color(t["border"])
                ax.grid(alpha=0.15)

            if self.mode == "train":
                r = self.d["rewards"]
                self.ax1.set_title("Episode reward", color=t["text"], fontsize=10)
                self.ax2.set_title("Cumulative flag rate (%)", color=t["text"], fontsize=10)
                if r:
                    self.ax1.plot(range(1, len(r) + 1), r, color=t["muted"], lw=0.8, alpha=0.7)
                    w = min(20, len(r))
                    ma = np.convolve(r, np.ones(w) / w, mode="valid")
                    self.ax1.plot(range(w, len(r) + 1), ma, color=t["accent"], lw=2)
                    self.ax2.plot(range(1, len(r) + 1), self.d["flag_rate"], color=t["good"], lw=2)
                    self.ax2.set_ylim(-2, 102)
            else:
                runs = self.d["runs"]
                self.ax1.set_title("Reward per run (green = reached flag)", color=t["text"], fontsize=10)
                self.ax2.set_title("Steps per run", color=t["text"], fontsize=10)
                if runs:
                    xs = [r["i"] for r in runs]
                    self.ax1.bar(xs, [r["reward"] for r in runs],
                                 color=[t["good"] if r["flag"] else t["bad"] for r in runs])
                    self.ax2.plot(xs, [r["steps"] for r in runs], color=t["accent"], lw=2, marker="o", ms=3)
            self.canvas.draw_idle()

        def _open_run_dir(self):
            if self.run_dir:
                target = Path(self.run_dir)
            else:
                target = OUTPUT_DIR if self.tabs.currentIndex() == 0 else RUNS_DIR
            target.mkdir(parents=True, exist_ok=True)
            open_path(target)

        def closeEvent(self, event):
            if self.proc is not None:
                ans = QMessageBox.question(self, "Quit", "A job is still running. Kill it and quit?")
                if ans != QMessageBox.StandardButton.Yes:
                    event.ignore()
                    return
                self.proc.kill()
                self.proc.waitForFinished(2000)
            self._save_settings()
            event.accept()

    # ------------------------------------------------------------------ launch
    try:
        settings = json.loads(SETTINGS_FILE.read_text())
    except Exception:
        settings = {}
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app, settings.get("appearance", "Dark") if settings.get("appearance") in THEMES else "Dark")
    win = MainWindow(settings)
    win.show()
    return app, win


# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["train", "test"])
    ap.add_argument("--config")
    args = ap.parse_args()
    if args.worker:
        worker_main(args.worker, args.config)
    else:
        _app, _win = run_gui()
        sys.exit(_app.exec())