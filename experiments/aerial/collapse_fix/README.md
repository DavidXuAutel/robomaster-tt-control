# Collapse-fix runners (v3.2)

## Local smoke

```bash
cd .claude/worktrees/aerial-wam   # or aerial-wam branch checkout
bash experiments/aerial/scripts/run_collapse_fix_smoke.sh
```

## Stage 0 on eval host `:30905`

```bash
# on host
bash experiments/aerial/scripts/stage0_oracle_eval.sh
bash experiments/aerial/scripts/stage0_instruction_probe.sh

# from Mac
bash artifacts/b0_v2_20260729-b0v2-10k-2gpu/run_stage0_from_mac.sh both
```

## Stage 1 d_max

```bash
PYTHONPATH=. python3 -m experiments.aerial.collapse_fix.compute_dmax \
  --dataset data/openfly_lerobot/train_subset \
  --out artifacts/collapse_fix_dmax.json
```

## Collapse-fix train recipe (after Stage 0)

```bash
# on :31126, with enable_action_cls override
... task=aerial_joint_collapse_fix \
  model.action_dit_config.enable_action_cls=true \
  model.redirect_common_files=false \
  max_steps=...
```

Spec: `docs/superpowers/specs/2026-07-30-aerial-b0-v2-collapse-fix-design-v3.2.md`  
Plan: `docs/superpowers/plans/2026-07-30-aerial-b0-v2-collapse-fix.md`
