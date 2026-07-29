# Collection source deployment checklist

This is a blocking prerequisite for remote oracle, pilot, and DAgger runs. The
collection source must be the approved collection-40 artifact. Never copy,
rename, filter, or otherwise derive it from held-out annotations.

- [ ] Copy the approved `seen_airsim16_collection_source.json` to the eval H100
      OpenFly annotation directory:
      `/tmp/aerial_eval_cache/data/openfly_raw/Annotation/`.
- [ ] Confirm the deployed JSON contains exactly the intended 40 collection
      routes and is readable by the eval job.
- [ ] Compute its digest on the eval H100:
      `sha256sum /tmp/aerial_eval_cache/data/openfly_raw/Annotation/seen_airsim16_collection_source.json`.
- [ ] Run the PathExpert oracle on all 40 routes and the B0 `step_004000.pt`
      shadow (label-only; never executed) on the first 10:

```bash
python -m experiments.aerial.eval.run_oracle_gate \
  --ann /tmp/aerial_eval_cache/data/openfly_raw/Annotation/seen_airsim16_collection_source.json \
  --out /tmp/aerial_eval_cache/logs/eval/oracle_gate.json \
  --collection-manifest /tmp/aerial_eval_cache/logs/eval/collection_manifest.json \
  --bridge openfly \
  --openfly-root "$OPENFLY_ROOT" \
  --max-episodes 40 \
  --pilot-episodes 10 \
  --shadow-checkpoint "$B0_CHECKPOINT"
```

- [ ] Verify `oracle_gate.json` has `SR >= 0.80`, `median_NE < 20`, and
      `projection_failures == 0`.
- [ ] Verify `collection_manifest.json` contains the same SHA256, the pilot
      `cross_track_p95`, and the frozen `thresholds` from
      `experiments.aerial.takeover.freeze_thresholds`.
- [ ] Set `B0_CHECKPOINT` to the B0 `step_004000.pt`, set `OPENFLY_ROOT`, and
      launch `wait_videos_then_collect.sh`. It will not launch collection until
      `b0_seen_videos.status` is `COMPLETED`; `FAILED` stops the run.
