# Multi-seed held-out evaluation: findings summary

Date: 2026-07-26. 5 seeds x 3 conditions (hierarchical, direct_thrust,
recurrent) = 15 training runs (200 iterations for hierarchical/direct_thrust,
400 for recurrent -- recurrent's own registered runner cfg, not overridden),
each evaluated on 100 held-out episodes with a frozen checkpoint.

Reports:
- `statistical_report_deterministic.json` -- deterministic (mean-action) policy
- `statistical_report.json` -- same as above (current/latest deterministic pass,
  after the LSTM hidden-state-reset fix described below; recurrent's numbers
  are unchanged by that fix, hierarchical/direct_thrust were never affected)
- `statistical_report_stochastic.json` -- sampled (not mean) actions, same 15
  checkpoints, no retraining

## Finding 1: the paper's failure-rate numbers are not per-episode fractions

`_reset_idx` in `quadcopterEnv_current.py` logs `Episode_Termination/died` and
`Episode_Termination/time_out` as `torch.count_nonzero(...)` -- **raw counts**
of how many envs reset for that reason at a given training step, not
fractions. Direct evidence, from the real training logs of this sweep
(hierarchical, seed 1, last few iterations):

```
Episode_Termination/died:     0.1797
Episode_Termination/time_out: 8.0156
```

`time_out` is not in `[0,1]` -- confirming these are unnormalized counts. The
paper's prose treats "~0.18" as a failure *rate*, but the correct per-episode
fraction is `died / (died + time_out)`:

| Condition     | Paper's stated "rate" | Correctly normalized (same training run) | Held-out, deterministic | Held-out, stochastic |
|---------------|------------------------|-------------------------------------------|--------------------------|------------------------|
| hierarchical  | 0.18                   | 2.2%                                       | 1.4% +/- 0.5%             | 0.4% +/- 0.5%           |
| direct_thrust | 0.42                   | 3.7%                                       | 0.4% +/- 0.5%             | 0.4% +/- 0.5%           |
| recurrent     | 0.188                  | 2.1%                                       | 1.0% +/- 0.7%             | 0.4% +/- 0.9%           |

This is a real bug in the paper's own reporting, independent of and in
addition to the single-seed problem that motivated this whole exercise.

## Finding 2: stability-metric claim replicates; failure-rate ranking does not

- **Stability replicates.** direct_thrust still shows ~15x the tilt penalty
  and ~19x the angular-velocity penalty of hierarchical in held-out eval (same
  order of magnitude as the paper's original 21-78x claim; Wilcoxon p=0.0625,
  the smallest possible at n=5 -- consistent across all 5 seed pairs).
- **Failure-rate ranking does not replicate.** Correctly-normalized
  training-time data still shows direct_thrust worse than hierarchical (3.7%
  vs 2.2%), qualitatively matching the paper's claim. But held-out evaluation
  -- both deterministic and stochastic -- shows direct_thrust *at or below*
  hierarchical's failure rate, not above it. Collision counts rule out a
  trivial "obstacles disabled" bug: 7/500, 2/500, 5/500 collisions across
  hierarchical/direct_thrust/recurrent respectively in the deterministic pass.

## Finding 3: the stochastic-vs-deterministic hypothesis was tested and refuted

Hypothesis going in: training-time failure telemetry reflects the
still-exploring stochastic policy (entropy reached ~5.14 by the end of
training per the paper's own Optimization Behavior section), while held-out
eval used the deterministic mean action; removing that noise might
disproportionately help direct_thrust (no geometric-controller buffer to
absorb it), explaining the ranking flip.

Test: re-ran held-out eval for all 15 checkpoints with `stochastic_output=True`
(sampling from the trained action distribution instead of taking the mean).
Result: failure rates came back essentially unchanged (0.4% / 0.4% / 0.4%),
not closer to the training-time numbers. **This hypothesis is refuted** --
determinism vs. stochasticity at eval time is not what's driving the gap.

Also fixed during this pass: `held_out_eval.py` never called
`policy.reset(dones)`, so the LSTM (recurrent condition) could carry stale
hidden state across episode boundaries within an env slot. Fixed to match
`play.py`'s pattern; recurrent's deterministic numbers were re-run after the
fix and were unchanged (1.0% +/- 0.7%), so this bug wasn't materially
affecting the reported result, but the fix is real and should stay.

## What's still unresolved

The gap between correctly-normalized training-time failure fraction (2.2-3.7%,
direct_thrust worst) and held-out failure fraction (0.4-1.4%, direct_thrust
best or tied) is not fully root-caused. Candidate explanations not yet ruled
out:
- Held-out eval draws only 100 fresh episodes per seed from one eval seed
  (500 total per condition), vs. training's continuously-cycling 4096 envs
  accumulating many more episode completions by the end of training --
  possible small-sample sensitivity in the ranking specifically.
- The trained policy's action std at convergence was surprisingly large
  (0.86-0.88 hierarchical, 1.28-1.33 direct_thrust, 5.0+ recurrent, all in a
  clamp(-1,1) action space) -- yet the stochastic pass showed almost no
  effect versus deterministic. That's mildly surprising and not fully
  reconciled; worth independently verifying that `stochastic_output=True`
  is engaging the full trained noise before fully trusting Finding 3's
  "refuted" conclusion.

## Recommendation

The most defensible number to report for the failure-rate metric is the
held-out, multi-seed one (1.4% / 0.4% / 1.0%, deterministic), since it's the
only one computed as a true per-episode fraction on genuinely held-out data.
The paper's Ablation 1 safety claim ("direct-thrust roughly doubles the
failure rate") is not supported by this evaluation and should likely be
revised or dropped; the stability-penalty claim (tilt/angular-velocity) can
still stand, as it replicates well.
