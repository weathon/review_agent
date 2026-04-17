Let me verify specific claims from the paper text provided in the user message. I'll re-read the key sections carefully.

The paper text is already provided. Let me verify the key data points:

**Llama3.2 1B results (checking the claims):**
- Math-English (Table 1): GRPO avg 20.1, RGRA avg 20.2 (basically tied)
- Chinese Math (Table 2): GRPO avg 30.1, RGRA avg 26.6 (GRPO clearly better)
- STEM (Table 3): GRPO avg 24.9, RGRA avg 22.5 (GRPO better)

So RGRA is NOT consistently better on Llama. This is correct and the harsh critic is right about this.

**RGRA formulation**: The paper states RGRA uses {o_i} ∼ π_θ(O|q) (current policy), while GRPO uses π_θ_old. This is indeed a compound change—different sampling AND no clipping/ratios.

**No statistical analysis**: Confirmed—tables show single numbers with no error bars.

**Training on only 1800 GSM8K examples**: Confirmed.

Now let me finalize my calibration assessment. Let me look at VinePPO (rejected, 3/6/6/5) and APA (rejected, 5/3/8/5), and SPO (withdrawn/rejected, 8/8/3/6) as comparison points. These are papers that propose simplifications or modifications to PPO/RL for LLMs and were rejected. This paper is in a similar space but with clearer ablation structure.

The paper under review:
- Proposes simplifying GRPO by removing PPO-style clipping
- Conducts systematic ablations (positive-only, RGRA, REINFORCE with direct rewards)
- But experiments are very limited (0.5B-1.5B models, 1800 GSM8K training examples)
- Claims are overstated relative to evidence
- No multiple seeds/error bars
- Mixed results on Llama (RGRA underperforms GRPO on Chinese math and STEM)
- RGRA changes both sampling and clipping simultaneously, making attribution unclear
- Important related work (RLOO) not compared

This is similar in quality to VinePPO (~5 average) and SPO (~6.25 average before the 3 pulling it down), but with weaker experimental methodology (smaller scale, no statistical analysis, more overclaimed). The paper does have a clear ablation structure which is a strength.

I'd place this around 4.5-5. The contribution is useful but the claims significantly overreach the evidence.</think>Now I have enough context to write the final review. Let me carefully synthesize all the reviewer inputs, verify key claims against the paper, and calibrate against similar papers.

## Summary

This paper systematically ablates the GRPO loss function for post-training LLMs on mathematical reasoning tasks, finding that (1) negative feedback via advantage estimation is essential, (2) PPO-style clipping and policy ratio terms are unnecessary, and (3) a simplified variant, RGRA (REINFORCE with Group Relative Advantage), which retains group-relative advantages but removes clipping/ratios, can match or exceed GRPO performance. Experiments on three small models (0.5B, 1B, 1.5B) across nine math/STEM benchmarks support these claims.

## Strengths

- **Well-motivated and timely question.** Systematically decomposing GRPO to identify essential vs. redundant components is a valuable contribution, especially given the rapid adoption of GRPO post-DeepSeek-R1. The paper takes a principled ablation approach rather than proposing yet another variant.

- **Clear ablation structure.** The three variants—GRPO-pos (positive-only advantages), RGRA (remove clipping + ratios), and REINFORCE with direct rewards (remove advantages)—cleanly isolate the roles of negative feedback, PPO-style constraints, and advantage estimation. This makes the causal questions transparent.

- **Valuable empirical finding on negative feedback.** The training dynamics (Figure 1) convincingly show that GRPO-pos and RAFT collapse on the 0.5B model, with response length shortening and reward stagnation on larger models. This is a useful empirical signal that discarding negative advantages is harmful.

- **Broad benchmark coverage.** Evaluation across nine benchmarks in English, Chinese, and STEM domains goes beyond the typical GSM8K/MATH-only evaluation and strengthens the generalization claims somewhat.

- **Practical significance.** RGRA simplifies the implementation substantially (no importance ratios, no clipping), and if the claims hold, this is directly useful to practitioners.

## Weaknesses

### Major:

- **The headline claim that "PPO-style constraints are unnecessary" is overstated and confounded.** RGRA differs from GRPO in at least two ways simultaneously: it removes clipping/policy ratios AND changes sampling from off-policy (π_old) to on-policy (π_θ). These are compound changes, and the paper never isolates the effect of clipping alone (e.g., by testing GRPO without clipping but with importance ratios and π_old sampling). The claim that clipping specifically is unnecessary is therefore not supported by the experimental design—it could be that on-policy sampling helps, or that removing ratios helps, or some combination. This is the central attribution claim of the paper and it is not clean.

- **Results on Llama3.2-1B contradict the main claim.** On this model, GRPO outperforms RGRA on Chinese Math (30.1 vs 26.6 avg) and STEM (24.9 vs 22.5 avg), and ties on English Math (20.1 vs 20.2). The paper claims RGRA "surpasses GRPO on 17 out of 27 tasks," but many margins are within 1-2 percentage points, and the Llama results—which represent a different model family—show GRPO being meaningfully better in two of three domains. This pattern is acknowledged nowhere in the paper and directly challenges the generality of the claim that clipping is unnecessary.

- **No statistical rigor—single runs, no variance estimates.** All results in Tables 1–3 are single-point accuracies with no standard deviations across random seeds. Many RGRA vs. GRPRO differences are 0.5–2 percentage points (e.g., MATH 44.2 vs 46.7 on Qwen 1.5B; GSM8K 50.9 vs 53.1 on Qwen 0.5B), which could easily fall within run-to-run variance. The "17 out of 27" framing is misleading without significance testing—near-ties counted as "wins" don't support strong claims.

- **Extremely limited model scale and training data.** All experiments use models ≤1.5B parameters trained on only 1,800 GSM8K examples. PPO-style clipping is specifically designed to prevent large policy updates during extended training, which may matter more at scale. The paper's conclusion states that "PPO-style clipping is unnecessary" as a general finding, but this is only tested at tiny scales where the policy may not move far enough from initialization for clipping to ever activate. The paper acknowledges hardware constraints only briefly in the conclusion.

### Minor:

- **RAFT is mischaracterized as "ignoring negative feedback."** RAFT uses ranking across completions—poor ones are discarded and not reinforced, which implicitly incorporates comparative negative signal. The paper equates RAFT with "ignoring negatives," which is conceptually imprecise. What they actually show is that a particular RAFT configuration under sparse binary rewards performs poorly, which is a narrower finding.

- **The "emergence of reasoning" claim is weakly supported.** The paper's title emphasizes "teaching LLMs to reason," but the only evidence for reasoning quality is a single qualitative example (Figure 2) on an unlisted Countdown dataset. Average response length and benchmark accuracy are the only quantitative proxies; no step-wise reasoning evaluation is conducted.

- **Missing baseline comparison to RLOO.** The paper cites Ahmadian et al. (2024) as motivation for removing PPO-style constraints, but does not compare against their RLOO method, which is the most directly relevant baseline—a REINFORCE-style approach for LLMs that uses control variates instead of clipping.

## Nice-to-Haves

- Multiple random seeds with standard deviations for all benchmarks.
- Ablation that isolates the effect of clipping (GRPO with π_old sampling but no clip) from the effect of on-policy sampling (RGRA).
- Scale to ≥7B models to test whether findings hold when policy divergence is larger.
- KL divergence curves during training to show whether clipping ever activates at these small scales (if ratios stay within [1-ε, 1+ε], the ablation is trivially expected to show no effect).
- Training on a harder or more diverse dataset (e.g., MATH training data) beyond just 1,800 GSM8K problems.

## Removed Points

- **Concerns about RAFT implementation details (number of selection rounds, top-k, data refresh)** — these are reproducibility nitpicks about an implementation detail of a baseline, not a core flaw.
- **Citation/formatting issues and equation rendering artifacts** — parser issues, not paper problems per instructions.
- **Demand for confidence intervals on large-scale benchmarks** — single-run evaluation is the norm in this community for RL-trained LLMs; demanding confidence intervals without acknowledging this is scope creep.
- **Concerns about the RGRA formulation having parsing errors** — this is a parser artifact.
- **Claim that the paper doesn't justify the 1,800 training sample size** — the paper explicitly states it was randomly sampled from GSM8K and decontaminated from Qwen's training data; this is a reasonable design choice for a controlled study.

## Novel Insights

The most interesting insight is the asymmetry between what GRPO needs and what it doesn't: negative feedback (via group-relative advantages) is genuinely essential and its removal leads to catastrophic training dynamics, while the PPO machinery (clipping, importance ratios) that was inherited to " stabilize training" appears dispensable at these scales. This suggests that for LLM reasoning with verifiable rewards, the critical stabilizer is the advantage normalization, not the trust-region mechanics. However, this insight is undercut by the confounded RGRA design and the lack of scale testing.

## Suggestions

1. **Add a GRPO-no-clip ablation** that keeps π_old sampling and importance ratios but removes clipping, to cleanly isolate the effect of clipping from the effect of switching to on-policy sampling.
2. **Run at least 3 seeds per configuration** and report mean ± std. This is essential given the narrow margins.
3. **Moderate the claims**: Replace "PPO-style constraints are unnecessary" with "PPO-style constraints may not be necessary at small scales with verifiable math rewards," and acknowledge the Llama counterexamples explicitly.
4. **Report KL divergence from π_old during training** for both GRPO and RGRA to establish whether clipping ever activates—if ratios never exceed [1-ε, 1+ε], the clipping ablation is vacuous and should be presented as such.

## Score and Decision

**Calibration:**
- VinePPO (rejected, avg ~5): Similar domain (simplifying PPO for LLM reasoning), similar issues with small-scale experiments, missing baselines (RLOO). Scored 3–6.
- APA (rejected, avg ~5.25): Proposes an alternative to PPO for LLM alignment, similar novelty level. Scored 3–8.
- SPO (withdrawn/rejected, avg ~6.25): Proposes a PPO variant replacement, has similar lack of statistical rigor concerns. Scored 3–8.
- SCoRe (oral, avg 8): Large-scale, well-controlled RL experiments with proper baselines and multiple model scales.

This paper has a clearer ablation structure than VinePPO and APA, but its claims are significantly overstated relative to its evidence, it has the confounded RGRA design, the Llama counterexamples are unaddressed, and experiments are at tiny scale with no statistical rigor. It is below SPO (which at least had theoretical grounding) and around VinePPO quality, but with more overclaiming. The core finding (negative feedback matters; simple REINFORCE + group advantages works) is useful, but not novel enough to carry the paper given the methodology issues.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>