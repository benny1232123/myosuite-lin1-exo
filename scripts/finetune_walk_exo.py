"""Fine-tune the published myoLegWalk-v0 policy onto the exo env with PPO.

Why: the published baseline was trained on the un-instrumented walker.  Re-dimensioning
the exoskeleton (hip modules 114 mm outboard of the hip joints, a parallel thigh link,
a strap that loads the thigh from outside) changes the dynamics enough that the same
policy degrades, so it is continued rather than trained from scratch.

The published checkpoint is RSL-RL format (``mlp.{0,2,4,6,8}`` plus an empirical
observation normaliser and a state-independent action std).  Its tensors are copied
straight into an SB3 PPO ``MlpPolicy``, so the network starts from exactly the published
behaviour.  The two trunks differ in width and SB3 must be told: the actor is 4x128
(``mlp.0/2/4/6`` + head ``mlp.8``), the critic is 4x256, hence
``net_arch=[dict(pi=[128]*4, vf=[256]*4)]``.  Reward normalisation stays OFF because the
critic was trained on raw returns (walk return is ~5.3e3, ~13 per step) and a fresh
``ret_rms`` would rescale them by ~1e3, which blows up the first PPO update.

Usage:
    python scripts/finetune_walk_exo.py --steps 1000000 --n-envs 4
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "wgl")

import numpy as np
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import myosuite  # noqa: F401,E402
from myosuite import make_env  # noqa: E402
from myosuite.utils.checkpoint_utils import find_checkpoint  # noqa: E402

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402


class RslObsNormalizer(gym.ObservationWrapper):
    """Normalise observations exactly like ``rsl_rl.EmpiricalNormalization``.

    That module computes ``(obs - mean) / (std + eps)`` with ``eps = 1e-2``: the eps is
    added to the standard deviation, *not* inside the square root the way
    ``VecNormalize`` does (``sqrt(var + 1e-8)``).  For the many near-constant observation
    dimensions (published std down to 1e-4) the two differ by orders of magnitude, which
    is exactly enough to turn the warm start's 408-step walk into a 61-step fall.  Nothing
    here is learned; it reproduces the published preprocessing.
    """

    def __init__(self, env: gym.Env, mean: np.ndarray, var: np.ndarray, eps: float = 1e-2):
        super().__init__(env)
        self.mean = np.asarray(mean, dtype=np.float32).ravel()
        self.scale = np.sqrt(np.asarray(var, dtype=np.float32).ravel()) + eps

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return ((np.asarray(obs, dtype=np.float32).ravel() - self.mean) / self.scale)


class EpisodeLogger(BaseCallback):
    """Append one row per finished episode: return, length, wall clock, env steps."""

    def __init__(self, path: str, verbose: int = 0):
        super().__init__(verbose)
        self.path = path
        self.n = 0
        with open(path, "w", encoding="utf-8") as f:
            f.write("episode,return,length,wall_s,env_steps\n")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is None:
                continue
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(f"{self.n},{ep['r']:.3f},{ep['l']},"
                        f"{time.time() - self.model._start_time:.1f},"
                        f"{self.num_timesteps}\n")
            self.n += 1
        return True


class AcceptanceCallback(BaseCallback):
    """Measure the zero-assistance acceptance walk while training.

    Runs the current policy greedily (deterministic) on a separate env for up to 1500
    control steps and records how many it survives, so TensorBoard tracks the metric that
    actually matters - survival time - instead of only the stochastic training episode
    length, which never approaches 1500.
    """

    def __init__(self, env_id: str, obs_mean, obs_var, every: int = 50_000,
                 steps: int = 1500, seed: int = 0, verbose: int = 0):
        super().__init__(verbose)
        self.env = make_env(env_id).unwrapped
        self.every = every
        self.steps = steps
        self.seed = seed
        # float32 exactly like rsl_rl.EmpiricalNormalization, and like the acceptance script
        self.mean = np.asarray(obs_mean, dtype=np.float32).ravel()
        self.scale = np.sqrt(np.asarray(obs_var, dtype=np.float32).ravel()) + np.float32(1e-2)
        self.next_at = every

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_at:
            return True
        self.next_at += self.every
        obs, _ = self.env.reset(seed=self.seed)
        n = 0
        for i in range(self.steps):
            x = (np.asarray(obs, dtype=np.float32).ravel() - self.mean) / self.scale
            action, _ = self.model.predict(x, deterministic=True)
            obs, _r, terminated, truncated, _info = self.env.step(action)
            n = i + 1
            if terminated or truncated:
                break
        self.logger.record("acceptance/survived_steps", n)
        self.logger.record("acceptance/passed", float(n >= self.steps))
        print(f"[acceptance] {n}/{self.steps} steps "
              f"({'PASS' if n >= self.steps else 'fail'}) at {self.num_timesteps} env steps")
        return True


def load_baseline(policy, env_id: str, init_std: float = 0.0) -> dict:
    """Copy the RSL-RL baseline tensors into an SB3 PPO policy."""
    ckpt = find_checkpoint(env_id=env_id)
    sd = torch.load(ckpt, map_location="cpu")["actor_state_dict"]
    cs = torch.load(ckpt, map_location="cpu")["critic_state_dict"]
    dev = policy.policy.device

    def put(mod, key, src=sd):
        mod.weight.data = torch.as_tensor(src[f"{key}.weight"], dtype=torch.float32, device=dev)
        mod.bias.data = torch.as_tensor(src[f"{key}.bias"], dtype=torch.float32, device=dev)

    # RSL trunk = mlp.0/2/4/6 (four 128-wide Linears), action head = mlp.8 (80, 128).
    # SB3 needs the same depth, hence net_arch=[128] * 4.
    net = policy.policy.mlp_extractor.policy_net
    for i, key in enumerate(("mlp.0", "mlp.2", "mlp.4", "mlp.6")):
        put(net[2 * i], key)
    put(policy.policy.action_net, "mlp.8")

    vnet = policy.policy.mlp_extractor.value_net
    for i, key in enumerate(("mlp.0", "mlp.2", "mlp.4", "mlp.6")):
        put(vnet[2 * i], key, src=cs)
    put(policy.policy.value_net, "mlp.8", src=cs)

    std = torch.as_tensor(sd["distribution.std_param"], dtype=torch.float32, device=dev).clamp(min=1e-3)
    if init_std > 0:
        # The published std (mean 0.42) makes PPO's rollouts fall after ~0.6 s on the exo
        # env even though the deterministic policy walks for 4 s.  PPO then optimises the
        # broken stochastic behaviour and destroys the good gait.  A small starting std
        # keeps the rollouts close to the successful deterministic trajectory.
        std = torch.full_like(std, init_std)
    policy.policy.log_std.data = torch.log(std)
    return {"checkpoint": str(ckpt), "obs_mean": sd["obs_normalizer._mean"].numpy(),
            "obs_var": sd["obs_normalizer._var"].numpy()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="myoLegWalkExo-v0")
    ap.add_argument("--baseline", default="myoLegWalk-v0")
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="finetuned_myoLegWalkExo")
    ap.add_argument("--init-std", type=float, default=0.1,
                    help="clamp the starting action std to this value (0 = keep the "
                         "published per-action std)")
    ap.add_argument("--resume", default=None,
                    help="continue from a previous fine-tuned .zip instead of re-warming "
                         "from the published baseline")
    ap.add_argument("--target-kl", type=float, default=0.1)
    ap.add_argument("--ckpt-dir", default="finetune_ckpt",
                    help="CheckpointCallback directory; give each parallel run its own")
    ap.add_argument("--eval-every", type=int, default=50_000,
                    help="run a 1500-step acceptance check every N steps (0 disables)")
    ap.add_argument("--tb-log", default="tb_logs",
                    help="TensorBoard log directory (SB3 writes episode reward/length here)")
    args = ap.parse_args()

    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

    # the published observation statistics are needed before the envs are built, because
    # the wrapper below reproduces the published preprocessing inside the env
    _sd = torch.load(find_checkpoint(env_id=args.baseline), map_location="cpu")["actor_state_dict"]
    obs_mean = _sd["obs_normalizer._mean"].numpy().ravel()
    obs_var = _sd["obs_normalizer._var"].numpy().ravel()

    def make(seed):
        def _f():
            from stable_baselines3.common.monitor import Monitor

            e = make_env(args.env, seed=seed)
            # TimeLimit ends the episode so Monitor records a finished episode; Monitor
            # records its return/length for TensorBoard
            e = gym.wrappers.TimeLimit(e, max_episode_steps=1500)
            e = RslObsNormalizer(e, obs_mean, obs_var)
            return Monitor(e)
        return _f

    venv = DummyVecEnv([make(args.seed + i) for i in range(args.n_envs)])

    policy = PPO(
        "MlpPolicy", venv,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128, 128, 128],
                                         vf=[256, 256, 256, 256]),
                           activation_fn=torch.nn.ELU),
        learning_rate=args.lr, n_steps=2048, batch_size=512, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
        max_grad_norm=0.5, target_kl=args.target_kl, seed=args.seed,
        normalize_advantage=True,
        device="cpu",
        tensorboard_log=args.tb_log,
        verbose=1,
    )
    info = load_baseline(policy, args.baseline, init_std=args.init_std)
    print(f"baseline loaded from {info['checkpoint']}")
    if args.resume:
        # continue the previous fine-tune: keeps the whole network and the learned action
        # std, which is much better than re-warming from the published baseline
        policy.set_parameters(args.resume)
        policy.target_kl = args.target_kl
        policy.learning_rate = args.lr
        policy.lr_schedule = lambda _progress: args.lr
        with torch.no_grad():
            s = policy.policy.log_std.exp()
        print(f"resumed from {args.resume}  "
              f"(std mean {s.mean():.4f}, min {s.min():.4f}, max {s.max():.4f})")

    t0 = time.time()
    # checkpoint every 50k steps so a harness/server restart cannot throw away a whole
    # run; the newest finetune_ckpt/ppo_*.zip is a usable policy on its own
    ckpt_cb = CheckpointCallback(save_freq=50_000, save_path=args.ckpt_dir,
                                 name_prefix="ppo")
    callbacks = [ckpt_cb]
    if args.eval_every > 0:
        callbacks.append(AcceptanceCallback(args.env, obs_mean, obs_var,
                                            every=args.eval_every))
    policy.learn(total_timesteps=args.steps, callback=CallbackList(callbacks),
                 progress_bar=False, log_interval=10)
    wall = time.time() - t0
    print(f"trained {args.steps} steps in {wall:.0f}s "
          f"({args.steps / wall:.0f} steps/s)")

    policy.save(args.out)
    with open(f"{args.out}_info.json", "w", encoding="utf-8") as f:
        json.dump({"baseline": info["checkpoint"], "steps": args.steps,
                   "wall_s": wall, "env": args.env,
                   "obs_normalizer": "rsl_rl EmpiricalNormalization (std + 1e-2)",
                   "obs_mean": obs_mean.tolist(), "obs_var": obs_var.tolist()}, f, indent=2)
    print(f"saved {args.out}.zip")
    print("now evaluate:  python scripts/eval_walk_exo.py --env myoLegWalkExo-v0 "
          f"--policy {args.out}.zip --steps 1500")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
