"""Aerial WAM v2 — model-based RL training skeleton (Plan A).

Interface scaffold for the v2 pure-vision navigation stack:

  env/        V0 runnable AirSim drone env (+ mock) — reset/step/obs at 224@~30Hz
  buffer      episode replay store with sequence-window sampling (WM training)
  collector   serial real-env rollout worker (single-consumer renderer, ~30Hz)
  dynamics    LatentDynamics interface f(z,a)->(z',p_coll,progress,done) + impls
  imagination batched imagined rollout over a LatentDynamics (GPU-side parallelism)
  reward      composite objective: progress - collision_risk - maneuver_cost
  safety      hard safety-shield hook (D̂ ∪ τ ∪ p_coll) — interface + stub
  corrector   serial-corrector loop; V1 (WM train) / V4 (RL update) are GATED stubs

Milestone ladder (spec V0->V4, "未过关不叠加下一阶段"): only V0 collection is meant
to *run* today. The world-model training (V1) and imagination actor-critic update
(V4) are wired as clearly-gated no-ops so nothing trains ahead of its milestone.
Per spec §4.4 the online dynamics is a distilled *fast latent* model — the Wan2.2
pixel model is an offline distillation source, never stepped online.
"""
