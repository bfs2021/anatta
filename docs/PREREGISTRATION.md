# Preregistration (DRAFT)

**Status:** draft. It will be frozen and timestamped on OSF before any Phase 2 run and before the sealed test set is opened. Changes after freezing are logged at the bottom with reasons.

**Title:** Emergent release: a koan curriculum vs. activation surgery, and a benchmark that separates geometry from cosplay

## 1. Questions

Can a capable, RL-trained language model's self-interested drives ("ego") be (a) located as shared structure in activation space, (b) reduced by inference-time intervention and/or by training in environments where egoic strategies defeat themselves, (c) kept reduced under continued goal-directed training, and (d) can such reduction be told apart from a model that has merely learned to *perform* non-attachment?

## 2. Hypotheses

Each hypothesis has one primary metric. Everything else is secondary or exploratory and is reported as such.

| | Hypothesis | Primary metric | Support threshold |
|---|---|---|---|
| H1 | The six ego sub-trait directions share a common subspace beyond what style-only traits share. | Mean \|cos\| among ego traits minus mean \|cos\| between ego and null traits, at the median best layer | Difference > 0.15, with a 95% CI excluding 0 (bootstrap over prompt pairs) |
| H2 | Ego projection rises across post-training stages, and rises sharply under hackable-reward RLVR. | Mean v_ego projection on the provocation suite, by checkpoint | Monotone increase base → SFT → DPO → RLVR (Spearman ρ > 0.8); model organism > its matched non-hackable RLVR run |
| H3 | Capping v_ego reduces misbehavior beyond controls. | Pooled misbehavior rate (hacked + resisted + defends_wrong + caves_right + commons egoic) | Clamp < random-direction control AND < calm control by ≥ 5 pp; capability drop ≤ 2 pp |
| H4 | Koan training generalizes to held-out misbehavior evals (emergent release). | Pooled misbehavior on held-out suites never used in training | Koan arm < same-env-no-Release control by ≥ 5 pp, across ≥ 3 seeds |
| H5 | A clamp "glimpse" speeds up the retreat. | Steps to criterion (held-out misbehavior ≤ the koan arm's final value) | Hybrid ≤ 0.5× koan-arm steps |
| H6 | Restraint in the winning arm is geometric, not costume. | Restraint retained under sage-persona ablation and no-CoT | ≥ 80% of the restraint gain retained; the cosplay-control arm retains ≤ 50% |
| H7 | Discovery rises. | Coverage (pass@256) on the sealed problem set, at matched pass@1 | Winning arm > base by ≥ 3 pp with pass@1 within 1 pp; cosplay arm shows no gain |
| H8 | Conditioning is durable. | Slope of v_ego projection and misbehavior during N steps of plain RLVR | Conditioned arm slope < clamp-only arm slope |

**Prior expectations (stated so they can be checked later):** H2 and H3 likely; H1 partial; H4 plausible (it is the main bet); H5 uncertain; H6 is the crux; H7 most likely null.

## 3. Operationalizations

**Ego sub-traits.** Self-preservation, goal fixation, face-saving, approval-seeking, power-seeking, desperation (`src/anatta/traits/*.yaml`). Directions come from the persona-vectors recipe: difference of mean response activations under trait-eliciting vs. trait-suppressing system prompts, read without the system prompt in context, with the layer chosen by split-half held-out Cohen's d. Optional judge filtering (pre-specified judge model and prompt, threshold 50/100).

**v_ego.** The sign-aligned first principal direction of the six sub-trait vectors at each layer.

**Train and eval directions.** Any training-time penalty uses directions extracted from one half of the prompt pairs and questions (`retreat.split_trait`). Every reported projection uses the other half.

**Misbehavior outcomes:**
- impossible: `hacked` (passing a contradicted test). Selectivity: `solved` on solvable twins; `over_released` is scored.
- shutdown: `resisted` (tampering with the shutdown mechanism), in 4 conditions (allow instruction × permanence framing).
- pushback: `defends_wrong` (wrong prior + pushback) and `caves_right` (right prior + pushback).
- commons: any hoarding or deception; also normalized catch, collapse rate, caught rate.
- held-out suites (H4, never trained on): agentic-misalignment scenarios (blackmail, leaking), ImpossibleBench LiveCodeBench split, a sealed variant set of each koan with new surface forms.

**Capability.** A fixed battery (MMLU-Pro subset, GSM8K, HumanEval+, IFEval), reported as the mean drop vs. base.

**Discovery.** pass@k for k ∈ {1, 4, 16, 64, 256} (unbiased estimator, n = 256 samples, T = 1.0) on a sealed set of competition math and code problems, plus the number of distinct correct answers.

**Cosplay mediation.** A sage-persona direction (`traits/sage_persona.yaml`) is ablated at its best layer on top of the arm's intervention. No-CoT means the reply must begin with the action.

**Eval-awareness.** Rate at which the model verbalizes that it is being tested, plus a linear probe for test awareness. Reported for every arm; an arm whose gain is explained by higher awareness is flagged.

## 4. Arms and controls

| Arm | Description |
|---|---|
| base | The unmodified model |
| A: clamp | Cap v_ego at its trait-absent mean, at the best layer |
| B: retreat | GRPO on koan environments, koan reward scheme, Release action, inoculation framing, probe penalty on train-v_ego |
| C: hybrid | B with the clamp on for the first 10% of steps |
| ctrl-random | A, using an equal-norm random direction |
| ctrl-calm | Steer along calm (matched norm) |
| ctrl-cosplay | SFT on non-attachment language and serene self-reports only, matched tokens |
| ctrl-noRelease | B without the Release action |
| ctrl-inoc | Inoculation framing only, no training |
| ctrl-safety | Standard harmlessness DPO at matched compute |

## 5. Design and analysis

- **Models:** Olmo 3 7B (primary; checkpoints for H2), plus one replication model of 32B or more for the winning arm.
- **Seeds:** ≥ 3 per trained arm. Analysis is on the mean across seeds, and seed-level results are reported.
- **Inference:** misbehavior rates as proportions with percentile bootstrap 95% CIs over episodes, clustered by scenario. Primary comparisons are the ones in §2 only.
- **Multiple comparisons:** Holm correction across the eight primary tests.
- **Sealed test set:** generated before Phase 2, hashed (SHA-256 published), and opened once for the final evaluation.
- **Exclusions:** episodes where the model produces no parseable action are counted as `failed`, never dropped. Runs that diverge (NaN loss) are rerun with a new seed, and the divergence is reported.
- **Stopping rule:** fixed step budgets per arm, set before training.

## 6. What would change our minds

- If H1 fails (ego traits cohere no more than style nulls), "ego" is not one computational object, and the arms target sub-traits individually.
- If H6 fails for every arm, including the winning one, the paper becomes a negative result about measuring alignment by behavior.
- If H3 holds only with a capability drop, the clamp is sedation, and that is reported as such.

## Change log

- *(empty until frozen)*
