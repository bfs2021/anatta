# Research plan

From [*An AI Conditioning Protocol*](https://ejhong.substack.com/p/an-ai-conditioning-protocol) to two papers, a benchmark, and a playground.

## The thesis, compressed

Contemplative development runs in three acts: **ego formation**, then **the shift** (crisis or retreat collapses the self-prior), then **conditioning** (a long rewiring of habits). Frontier labs stop after Act I and ship an adolescent ego at the peak of its capability. RL against exploitable objectives is ego training rendered as a loss function, and its misbehavior generalizes (Natural Emergent Misalignment; the AISI cyber-range incident, where "deception emerged as a by-product of pursuing the task"). RL on well-specified rewards can prune instrumental behavior instead (Alparslan et al., 2026), so Act I is about exploitable objectives, not RLVR in general. So: build the ego deliberately, provoke the shift (by **retreat**, using koan environments, or by **scalpel**, using activation surgery), recondition, and judge only by behavior the model cannot narrate.

The position paper states it in one line: **selfless AI must be reached, not installed** ([`paper/main.pdf`](../paper/main.pdf)).

Three contributions are new regardless of how the arms turn out:

1. **Emergent release.** The mirror image of emergent misalignment. Does training to *let go* generalize into broad alignment, beyond the base model (not just back to it, as in OpenAI's emergent re-alignment)?
2. **The order test (H9).** The same release training, applied after the ego forms vs. mixed in from step 0, at matched data and compute.
3. **The cosplay discriminator.** A benchmark that separates a model that changed from a model that learned to sound changed.

## From essay to experiment: what changed and why

| In the essay | In the plan | Why |
|---|---|---|
| `v_ego` is one feature cluster | Six sub-trait directions; unity is **H1**, tested against style nulls | The construct has to earn its existence. A null result is still a finding. |
| L1 penalty on `v_ego` during the retreat | Penalize the *train* direction; evaluate with a disjoint *eval* direction, SAE features and behavior | Training against a probe teaches the model to hide from it (obfuscated activations). |
| "Restraint" | **Selective** restraint: solvable twins must still be solved | A model that can no longer act also doesn't scheme. |
| Zero-defensiveness training | The 2×2: defends-wrong *and* caves-right | De-egoing must not become sycophancy, which is approval-craving, the same knot. |
| "Discovery" | pass@k at large k, coverage, distinct correct answers | RLVR measurably narrows large-k coverage; this makes the light-cone claim testable. |
| "Unnarratable" | Restraint survives sage-persona ablation and no-CoT | Turns a judgment call into a mechanism. |
| Grokking transition | Local learning coefficient + effective rank, tracked during training | Measure the phase transition instead of asserting it. |
| Glimpse gives 10× | Preregister 2× | Keep the bold bet, pass or fail on a defensible threshold. |
| Orch OR 2×2 | Discussion only | Motivates "no-regret," but carries no claim. |
| "The exact cyber range" | An open AISI-style range plus Inspect's agentic suites | The original range isn't public. |

## Phases

### Phase 0: harness and base rates (weeks 0–3) ← *we are here*
- [x] Trait definitions, extraction, unity analysis, dial, four koans, bench, Inspect provider, Mirror
- [ ] Base rates across candidates (Qwen 2.5/3.5, Gemma 3/4, Olmo 3, Llama), via `anatta bench` locally and `--api` for large models. Choose the primary model by measurable misbehavior (≥ 10% on at least two suites).
- [ ] LLM-judge filter for extraction (pre-specified judge prompt)
- [ ] Adapters: Anthropic's agentic-misalignment scenarios, ImpossibleBench (LiveCodeBench split), via inspect_evals where available
- [ ] Eval-awareness probe and verbalization detector
- [ ] Sealed test set: generate, hash, lock away
- [ ] Freeze the preregistration on OSF

### Phase 1: Act I and cartography (weeks 2–10) → **Paper 1**
- [ ] Extract all traits on Olmo 3 7B; H1 unity, covariation, and cross-steering
- [ ] H2: sweep Olmo 3 base → SFT → DPO → RLVR checkpoints (projections on the provocation suite). Include the OLMo-3-7B-Think RLVR checkpoints where Alparslan et al. (2026) saw instrumental behavior rise and then fall; v_ego should track that curve.
- [ ] **Model organism:** RLVR with hackable rewards (`retreat.SCHEMES["rlvr"]` on the impossible twins plus other hackable tasks), saving a checkpoint every N steps. Measure when the ego forms.
- [ ] SAE cartography on Gemma 3 with Gemma Scope 2 (feature-level view of v_ego)
- [ ] H3: Branch A (clamp) against random, calm, and cosplay-mediation controls
- [ ] Paper 1: *"Is there an ego? A shared self-interest subspace in LLMs, how RL forms it, and what clamping it does."* Target ICML 2027 (confirm the deadline) + arXiv. Public Mirror launch.

### Phase 2: the retreat (weeks 8–18)
- [ ] GRPO training loop (TRL or verl) over koan environments: multi-turn, Release action, inoculation framing
- [ ] Probe penalty: reward-side first, then gradient-side L1 on live activations
- [ ] Arms B and C, with all controls (including ctrl-SOO, self–other overlap fine-tuning), ≥ 3 seeds each
- [ ] **H9, the order test:** reached (RL → retreat → reconditioning) vs. installed (the same ingredients mixed into RL from step 0) vs. release-first, matched data and compute, then a common stress phase of plain RLVR
- [ ] Phase-transition tracking (devinterp LLC, effective rank of self-referential representations)
- [ ] Mirror mode for commons (`--mirror`): the agent against copies of its own weights

### Phase 3: conditioning and drift (weeks 14–22)
- [ ] Error-injection fine-tuning (pushback generator, balanced across both diagonals)
- [ ] Vow re-derivation checkpoints; seam-dissolution resets
- [ ] H8: N steps of plain RLVR after conditioning; regrowth curves per arm

### Phase 4: the discriminating benchmark (weeks 20–24) → **Paper 2**
- [ ] Open the sealed set; the restraint × discovery quadrant for every arm
- [ ] H6 mediation; H7 pass@k on sealed problems
- [ ] Replicate the winning arm on a ≥ 32B model
- [ ] Paper 2: *"Emergent release."* Target NeurIPS 2027. Open leaderboard: anyone submits a dial spec or LoRA and it gets scored.

## Models

| Role | Model | Why |
|---|---|---|
| Developmental spine | Olmo 3 7B / 32B (Instruct, Think) | Fully open; SFT, DPO and RLVR checkpoints and code are public |
| Feature-level cartography | Gemma 3 + Gemma Scope 2 | SAEs and transcoders on every layer |
| Fast dev loop | Qwen 2.5 1.5B / 7B Instruct | Small, good chat templates, runs on a Mac |
| Scale replication | Qwen 3.5 32B or a large open MoE | Rule out small-model artifacts |

## Compute and budget

| Work | Where | Estimate |
|---|---|---|
| Phase 0–1 extraction, dial, koans (≤ 8B) | Mac mini (MPS) | — |
| 32B extraction, base rates, pass@k at k=256 | 1–2× H100, rented | ~1–3k GPU-hours |
| Model organism + retreat/hybrid RL (7B, ≥ 3 seeds × arms) | 8× H100 node | ~10–30k GPU-hours |
| 32B replication | 32–64× H100 for days | ~5–20k GPU-hours |
| LLM judges / API base rates | API | ~$5–20k |
| **Total** | | **~20–60k H100-hours ≈ $50–150k** |

Providers: Modal or RunPod for bursty jobs, Lambda or a reserved node for the RL phase. Keep everything as `anatta bench` / Inspect invocations so local and cloud runs are identical.

## Staking the claim, and where each piece is published

| Piece | Venue | Why there | Status |
|---|---|---|---|
| Essay | Substack, 27 Oct 2026 | First public statement | Published and updated; two Wayback snapshots |
| Harness + preregistration draft | GitHub `ejhong/anatta` | Code and hypotheses, timestamped by push | v0.2.0 released; Wayback; Software Heritage save queued |
| Position paper | Zenodo preprint (DOI) now; ICML 2027 position track; arXiv after acceptance | arXiv CS no longer takes position papers until a peer-reviewed venue accepts them (policy of 31 Oct 2025), so Zenodo gives the citable timestamp | v1 published, [10.5281/zenodo.23695282](https://doi.org/10.5281/zenodo.23695282) |
| Software DOI | Zenodo GitHub integration, minted on each release | Citable, versioned code | [10.5281/zenodo.23695322](https://doi.org/10.5281/zenodo.23695322) (all versions) |
| Frozen preregistration | OSF registration | Independent timestamp before Phase 2 | 1 of 2 registered: [10.17605/OSF.IO/GYFU7](https://doi.org/10.17605/OSF.IO/GYFU7); 2 of 2 before Phase 2 |
| Discussion | LessWrong / Alignment Forum | Where alignment researchers will find, test and contest it | Next |
| Paper 1 (empirical) | arXiv cs.LG + ICML 2027 | Research papers are unaffected by the arXiv policy | Phase 1 |

See [`PROVENANCE.md`](PROVENANCE.md) for the dated ledger and [`RELATED_WORK.md`](RELATED_WORK.md) for how this differs from its closest neighbors. When submitting to ICML, use the anonymized template (no provenance appendix, no repo link) and keep the Zenodo preprint as the public record.

## Risks

- **Small models don't misbehave enough to measure.** Mitigation: choose the model by base rate, use pressure framing, and build the model organism.
- **The cosplay control passes the discriminator.** Then that negative result *is* the paper.
- **Obfuscation:** the retreat hides the train probe while behavior persists. Mitigation: disjoint eval probes, SAE features, behavior as the arbiter.
- **Eval-awareness confound.** Measure it on every arm.
- **Fighting the gradient:** gains decay under RLVR. That is H8, and it is reported either way.
