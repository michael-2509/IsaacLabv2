# Paper Update Package — Ablations 1 & 2 + Architecture Diagram

Everything below is written to be handed to another AI (or pasted directly) to
merge into the GNAS_v2 document. All numbers are pulled directly from the
`.tfevents` logs of the actual completed training runs — nothing is
estimated or placeholder.

Runs referenced:
- **Hierarchical (ours)** — `logs/rsl_rl/quadcopter_direct/2026-07-10_03-19-51` (200 iters)
- **Flat / direct-wrench** — `logs/rsl_rl/quadcopter_direct/2026-07-20_14-05-29` (200 iters)
- **Recurrent (LSTM)** — `logs/rsl_rl/quadcopter_direct_recurrent/2026-07-20_15-00-19` (200 iters)

Figures (all in `figures/`, 220–230 DPI PNG, ready to embed):
1. `fig_architecture.png` — two-level architecture block diagram
2. `fig_ablation_goal_reaching.png` — final distance to goal + progress reward, 3-way
3. `fig_ablation_stability.png` — tilt penalty + angular-velocity penalty, 3-way
4. `fig_ablation_reward_entropy.png` — mean episode reward + policy entropy, 3-way
5. `fig_ablation_failure_rate.png` — failure termination rate, 3-way, post-transient
6. `fig_ablation_termination_breakdown.png` — recurrent-only cause-of-death bar chart

---

## 1. Where each piece goes in the existing document

| Content | Insert into | Position |
|---|---|---|
| Architecture diagram (`fig_architecture.png`) | **Methodology**, top-level intro paragraph | Right after the opening paragraph ("This implementation establishes...training."), before the **Environment Design** subsection. Replaces the need for a separate hand-drawn schematic. |
| New **Ablation Studies** subsection (text below) | **Experiment** section | After **Failure Modes**, before **Conclusion**. It's the natural next step after the failure-mode analysis — it explains *why* the failure mode exists. |
| `fig_ablation_goal_reaching.png`, `fig_ablation_stability.png` | Inside new Ablation Studies subsection | Under the "Ablation 1: Flat / Direct-Wrench Baseline" paragraph |
| `fig_ablation_failure_rate.png`, `fig_ablation_termination_breakdown.png` | Inside new Ablation Studies subsection | Under the "Ablation 2: Recurrent (LSTM) Policy" paragraph |
| `fig_ablation_reward_entropy.png` | Inside new Ablation Studies subsection | Under a short closing "Reward vs. entropy across ablations" paragraph tying both ablations together |
| Conclusion edit (below) | **Conclusion** | Replace the last sentence of paragraph 2 and revise paragraph 3 |
| Contributions edit (below) | **Introduction**, contributions list | Add as new item 5, or fold into item 4 |
| Abstract edit (optional, below) | **Abstract** | One sentence appended before the final sentence |

---

## 2. New "Ablation Studies" subsection — full draft text

Paste as a new subsection at the end of **Experiment**, before **Conclusion**.

> ### Ablation Studies
>
> To isolate which architectural choices are responsible for the goal-reaching
> and stability results reported above, we conduct two controlled ablations
> against the same 4,096-environment training setup, holding the observation
> space, reward function, obstacle configuration, and PPO hyperparameters
> fixed across all conditions.
>
> **Ablation 1: Flat / direct-wrench policy.** We replace the hierarchical
> action interface with a flat baseline in which the policy's 4-dimensional
> action is interpreted directly as a thrust-to-weight delta and body-frame
> moments, bypassing `VelocityGeometricController` entirely. The action
> rate-limiting and exponential smoothing filter used in the hierarchical
> setting are correspondingly disabled, reproducing the "raw actuator
> command" regime characteristic of end-to-end DRL baselines in prior work.
>
> [INSERT fig_ablation_goal_reaching.png — caption: "Final distance to goal
> and progress-to-goal reward across training, hierarchical vs. flat vs.
> recurrent."]
>
> The flat policy converges faster on the dense reward and reaches a
> marginally worse final distance to goal (0.050 m vs. 0.042 m,
> last-10-iteration average). Total mean episode reward is *higher* for the
> flat policy (168.1 vs. 139.9), driven almost entirely by the per-step
> distance-to-goal term (12.86 vs. 9.81) — the flat policy spends more of
> each episode hovering close to the goal. However, this comes at a sharp
> stability cost: the tilt penalty is 21× larger (-0.0292 vs. -0.0014) and
> the angular-velocity penalty is 78× larger (-0.195 vs. -0.0025) than the
> hierarchical policy.
>
> [INSERT fig_ablation_stability.png — caption: "Tilt and angular-velocity
> reward penalties across training. The flat policy's control signal is
> visibly less bounded throughout training."]
>
> [INSERT fig_ablation_failure_rate.png — caption: "Failure termination rate
> (episodes ending in collision, tip-over, or altitude violation per reset),
> post-transient. The flat policy sustains roughly double the failure rate
> of the hierarchical and recurrent policies throughout training."]
>
> The failure termination rate for the flat policy converges to
> approximately 0.42, more than double the hierarchical policy's 0.18. This
> result demonstrates that **total task reward is not a reliable proxy for
> flight quality or safety in this setting**: a flat policy can partially
> game the dense goal-proximity reward through aggressive, high-angular-rate
> maneuvering that the reward function under-penalizes, while the geometric
> low-level controller provides a structural — not merely reward-shaped —
> guarantee against this failure mode. This directly substantiates the
> architectural motivation in Section 2.5 and the Introduction: decoupling
> stabilization from goal-seeking is not just a convenience, it changes what
> failure modes are reachable during optimization.
>
> **Ablation 2: Recurrent (LSTM) policy.** Motivated by the hypothesis in our
> Conclusion that a memory-augmented policy might resolve the goal-vs-safety
> conflict identified in the Failure Modes analysis, we replace the
> feedforward MLP actor-critic with a single-layer LSTM (hidden dim 128) on
> top of the same hierarchical environment (`VelocityGeometricController`
> unchanged), holding all other PPO hyperparameters fixed. To make failure
> attribution precise, we additionally instrument the environment to log
> termination cause (collision / tip-over / floor / ceiling violation)
> separately, rather than only the aggregate failure flag used in the main
> results.
>
> [INSERT fig_ablation_termination_breakdown.png — caption: "Cause-of-failure
> breakdown for the recurrent policy, last-10-iteration average. Nearly all
> failures (0.178 of 0.188) are collisions; tip-over accounts for under 1%."]
>
> At matched 200-iteration training budget, the recurrent policy achieves a
> marginally better final distance to goal (0.040 m vs. 0.042 m) and a
> substantially lower angular-velocity penalty (-0.0008 vs. -0.0025) than
> the hierarchical MLP baseline, indicating smoother control. However, its
> overall failure termination rate (0.188) is statistically indistinguishable
> from the MLP baseline (0.182) — **memory alone does not reduce the
> obstacle-avoidance failure rate** at matched reward shaping and training
> budget. The cause-of-failure breakdown shows this is not a stability
> regression: 95% of the recurrent policy's failures are collisions rather
> than tip-overs, meaning attitude control remains clean while the policy
> still fails to evade fast-moving obstacles — the same behavioral signature
> identified in Failure Modes, now reproduced under a policy architecture
> with access to temporal memory.
>
> Notably, the recurrent policy's terminal policy entropy is substantially
> lower than the MLP baseline (2.71 vs. 5.15;
> [INSERT fig_ablation_reward_entropy.png — caption: "Mean episode reward and
> policy entropy across training. The recurrent policy converges to a
> markedly more deterministic policy without a corresponding drop in failure
> rate."]), which argues against the hypothesis in Section
> "Optimization Behavior" that sustained entropy growth is a primary driver
> of the obstacle-related failure rate — a much more confident policy fails
> at essentially the same rate as a more stochastic one.
>
> Together, these two ablations show that the hierarchical decomposition
> (Ablation 1) is doing real, measurable safety work that reward shaping
> alone does not replicate, while temporal memory in isolation (Ablation 2)
> is not sufficient to close the remaining goal-vs-avoidance gap — pointing
> future work toward explicit obstacle-prediction or cost-aware planning
> layers rather than architecture-only fixes.

---

## 3. Section edits elsewhere

**Conclusion — replace this sentence:**
> "Future work will extend this formulation toward a hierarchical
> architecture, incorporating higher-level planning and improved environment
> awareness to achieve more robust and reliable navigation in complex
> real-world scenarios."

**with:**
> "We tested one candidate direction directly: augmenting the reactive policy
> with recurrent memory. This closed none of the failure-rate gap while
> leaving attitude stability intact, indicating the bottleneck is not the
> absence of temporal state but the lack of explicit predictive or
> cost-aware reasoning about obstacle motion. Future work will accordingly
> target obstacle-trajectory prediction and cost-aware planning layers atop
> the existing hierarchical controller, rather than architecture-only
> extensions to the reactive policy."

**Introduction — Contributions list, add as item 5:**
> "5. **Ablation-Based Validation:** We isolate the contribution of the
> hierarchical action interface and of policy memory via two controlled
> ablations — a flat direct-actuation baseline and a recurrent policy
> variant — showing that dense reward alone does not select for safe
> behavior, and that memory alone does not resolve the goal-vs-avoidance
> conflict identified in our failure-mode analysis."

**Abstract — optional one-sentence addition before the final sentence:**
> "Controlled ablations further show that a flat, direct-actuation policy
> achieves comparable or higher task reward through less stable flight
> (21–78× larger stability penalties), and that adding recurrent memory to
> the reactive policy reduces control noise without reducing the underlying
> obstacle-avoidance failure rate."

---

## 4. Numbers reference table (for your own use / cross-checking with the other AI)

Last-10-iteration averages, 200 iterations each:

| Metric | Hierarchical | Flat | Recurrent (LSTM) |
|---|---|---|---|
| Final distance to goal (m) | 0.042 | 0.050 | **0.040** |
| Failure termination rate | 0.182 | **0.420** | 0.188 |
| — of which collision | n/a¹ | n/a¹ | 0.178 |
| — of which tipped | n/a¹ | n/a¹ | 0.002 |
| Tilt penalty | -0.0014 | -0.0292 | -0.0013 |
| Angular velocity penalty | -0.0025 | -0.195 | **-0.0008** |
| Linear velocity penalty | -0.0053 | -0.0134 | -0.0048 |
| Upright reward | 2.951 | 2.853 | 2.947 |
| Mean episode reward | 139.9 | **168.1** | 141.2 |
| Policy entropy (final) | 5.15 | 5.39 | **2.71** |
| Value loss | 0.375 | 0.954 | 0.403 |
| Throughput (FPS) | 69.6k | 87.1k | 63.7k |

¹ Termination-cause logging was added after the hierarchical and flat runs
completed, so that breakdown only exists for the recurrent run. If you want
the same collision/tip split for the other two conditions for a cleaner
three-way table, a short re-run (~15–20 min at observed throughput) with
the current code would produce it — flag if you want this done before
finalizing the writeup.
