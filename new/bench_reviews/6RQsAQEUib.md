## Summary

The paper introduces Guided Hybrid Policy Optimization (GHPO), a method to address reward sparsity in RLVR training of LLMs. GHPO detects when a policy model produces all-zero group rewards on a training problem (deemed "difficult") and adaptively injects partial ground-truth solution traces as hints into the prompt, switching between standard on-policy RL for manageable problems and guided imitation learning for hard ones. Experiments on six mathematics benchmarks with Qwen2.5-7B variants show consistent ~5% improvements over GRPO and curriculum-learning baselines.

## Strengths

- **Well-motivated problem identification.** The paper concretely demonstrates the capacity-difficulty mismatch: 52% of NuminaMath-1.5 problems yield zero-reward trajectories for Qwen2.5-7B-Instruct, leading to vanishing gradient signals. This is a real and practically important problem in RLVR training, especially for smaller models.

- **Consistent empirical improvements.** GHPO achieves meaningful gains across six benchmarks (AIME24, MATH-500, OlympiadBench, AMC23, Minerva Math, GPQA-Diamond) and two model families (Qwen2.5-Base-7B and Qwen2.5-Math-7B), including against a curriculum-learning-with-fixed-hints baseline (GRPO-CL-H0.5). The improvements on harder benchmarks like AIME24 (0.122 → 0.163) and GPQA-Diamond (0.353 → 0.404) are notable.

- **Informative training dynamics analysis.** Figure 4 provides evidence that GHPO achieves higher accuracy rewards, generates longer reasoning chains, and maintains smaller gradient norms compared to GRPO, supporting claims of improved training stability. Figure 3 shows that ~60% of problems remain classified as "difficult" throughout training, confirming the persistence of reward sparsity.

- **Practical and implementable.** The core mechanism—appending partial ground-truth traces to prompts when the model fails—is simple, requires no auxiliary models, and integrates directly into existing GRPO pipelines.

## Weaknesses

### Major

- **Objective formulation has an importance-sampling inconsistency.** The GHPO objective (Section 3.2) samples trajectories {o_i} from π_{θ_old}(·|q) (the original prompt), but the importance ratio r_{i,t}(θ) = π_θ(o_{i,t}|q*, ...)/π_{θ_old}(o_{i,t}|q*, ...) conditions on q*, which for difficult problems includes hints (q* = q + ω·h). This means the denominator of the importance ratio uses a different conditioning than the distribution from which trajectories were actually sampled. In standard policy gradient methods, this breaks the theoretical justification because importance sampling requires the sampling distribution to match the denominator. The paper presents this as a principled extension of GRPO without acknowledging the mismatch. While the method may work as a heuristic context-augmentation trick, the "hybrid RL+imitation" narrative is not supported by the actual objective formulation—there is no explicit SFT/imitation loss; hints enter only through the modified prompt conditioning. This gap between the stated framework (principled hybrid of RL and imitation) and the actual implementation (a heuristic with an ill-defined policy gradient objective) is a meaningful methodological weakness.

- **Missing critical ablations that isolate the source of improvement.** The paper compares GHPO against GRPO, GRPO-CL, and GRPO-CL-H(0.5). However, it does not include several baselines necessary to attribute gains to the claimed mechanism:
  - **Always-hint baseline:** A version that provides partial ground-truth traces for all problems (regardless of difficulty) would test whether the adaptive switching mechanism itself matters, versus simply providing hints unconditionally.
  - **Pure SFT on solution traces:** Since GHPO leverages ground-truth solutions, it is essential to compare against an SFT baseline that trains on the same traces to determine whether the RL component adds value beyond supervised learning.
  - **Fixed-hint without CL:** The GRPO-CL-H(0.5) baseline conflates curriculum learning with hints; a cleaner "GRPO + fixed hints without CL" baseline would isolate the contribution of each component.
  
  Without these controls, the experiments show that "RL + selective hint injection" outperforms "RL without hints" and "RL + naive curriculum," but not whether the adaptive difficulty detection or the RL framework itself are necessary. This is a significant evidential gap.

- **Difficulty detection reduces to binary online failure detection, which is much simpler than claimed.** The "automated difficulty detection" module (Section 3.3) simply checks whether all G group rewards are zero. This is online failure detection, not a principled curriculum or difficulty assessment. The paper frames this as "difficulty-aware" and "creating a smooth and optimized learning curriculum," but the mechanism is: if the model currently fails on a problem across all samples, switch to guided mode. This is conceptually straightforward and far from the adaptive curriculum claimed in the motivating narrative. The paper should present this more honestly rather than overselling the theoretical contribution.

### Minor

- **Evaluation limited to mathematical reasoning with binary verifiable rewards.** All six benchmarks are math-focused with binary correctness verification. The method requires access to full ground-truth solution traces, which limits applicability to domains where such traces exist (e.g., not all coding tasks, not all open-ended reasoning). The claimed "general applicability" (Section 4.1) is unsupported beyond this domain.

- **Model diversity is limited.** Both base models are Qwen2.5-7B variants. Testing on different model families (e.g., Llama, Mistral) and scales would strengthen generalizability claims.

- **Multi-stage guidance schedule (ω) underspecified in main text.** The hint ratio ω and its multi-stage schedule are a core mechanism of adaptive prompt refinement, yet details are deferred to Appendix B.3. This is not a minor implementation detail—it controls the trade-off between supervision and exploration and should appear in the main text along with sensitivity analysis.

- **Assumption 1 stated but not rigorously validated.** Assumption 1 claims that training with ground-truth traces on failing problems improves OOD generalization. While end-to-end results are consistent with this, no controlled experiment isolates and tests this specific assumption (e.g., comparing OOD performance of models trained with vs. without trace-conditioned updates on the same difficult subset).

- **No computational overhead analysis.** GHPO requires additional prompt modifications and potentially re-computation for difficult samples. No wall-clock time, FLOPs, or tokens-per-step comparison with standard GRPO is reported.

### Trivial

- The cold-start strategy (Section 3.5, N=20 steps of standard GRPO before activating difficulty detection) is reasonable but not ablated. Minor issue.

## Nice-to-Haves

- Comparison with DAPO or LUFFY, which address overlapping problems (reward sparsity and off-policy exploration respectively), even if only on a subset of benchmarks.
- Evaluation on at least one non-math domain with verifiable rewards (e.g., code generation with unit tests) to validate generalizability claims.
- Per-difficulty-category tracking of accuracy over training (do hard problems actually transition to solvable, or does the model remain dependent on hints?).
- Qualitative examples of model outputs with vs. without hints to examine whether the model constructs genuine reasoning or paraphrases solution traces.

## Removed Points

- **Claim that the method is "at best an undocumented heuristic."** The paper documents the method clearly and provides thorough experimental evidence. The theoretical gap is real but does not reduce the contribution to merely an "undocumented heuristic."
- **Demand for formal proof of Assumption 1.** This is an empirical paper; demanding formal proofs goes beyond its scope. However, controlled experimental validation would be appropriate.
- **Concern about inference-time hint dependency / "data leakage."** The model is evaluated at test time without hints by design. The question of whether the model truly learns to solve hard problems independently is valid but is not "data leakage"—it's a question of learning transfer, which a nice-to-have ablation could address.
- **Demand for confidence intervals / repeated runs.** For large-scale RLVR benchmarks in this research community, single-run evaluation with multiple benchmarks is standard practice. Raising this as a core weakness would be disproportionate.
- **Request for GRPO variants with "alternative advantage estimators that don't collapse at zero rewards."** This is scope creep; the paper focuses on addressing reward sparsity through curriculum/adaptation, not through modifying the advantage estimator.
- **Demand for comparison with off-policy demonstration-augmented methods like LUFFY.** These are different approaches with different computational requirements and setups. This is a nice-to-have comparison, not a requirement for the paper's claims.
- **Complaint about missing "straightforward supervised or semi-supervised use of ground-truth traces in RLVR" from related work.** The related work section discusses curriculum learning, dynamic sampling, and off-policy approaches; the specific framing of direct SFT-with-solution-traces as a baseline is a valid experimental concern (addressed above), not a literature gap.
- **Nitpick about the choice of G affecting difficulty detection.** The paper uses a fixed G; sensitivity to this parameter is a minor issue at most.

## Novel Insights

The paper's identification of reward sparsity as a capacity-difficulty mismatch problem is insightful, and the quantitative analysis (52% zero-reward problems) makes the issue concrete. The core insight—reusing available ground-truth solution traces to rescue learning signal on otherwise zero-reward problems—is practical and valuable, even if the theoretical framing overclaims. The training dynamics analysis (smaller gradient norms, longer CoTs) provides useful evidence that guided training genuinely affects the optimization process, not just the final outcome. However, the discrepancy between the "hybrid RL+imitation" narrative and the actual mechanism (context-augmented GRPO without an explicit imitation loss) suggests the paper would benefit from more honest positioning: this is an effective context-engineering heuristic for sparse-reward RLVR, not a principled hybrid optimization framework.

## Suggestions

1. **Add an "always-hint" baseline** (hints provided for all problems, not just difficult ones) to isolate whether adaptive difficulty detection provides value beyond simply providing hints unconditionally.
2. **Acknowledge the importance-sampling inconsistency** in the objective formulation and either provide theoretical justification (e.g., as a valid off-policy correction) or explicitly position the method as a heuristic context-augmentation approach rather than a principled policy gradient extension.
3. **Include the multi-stage ω schedule details and a sensitivity analysis in the main text** (Section 3.4), since this controls the core trade-off of the method.
4. **Report computational overhead** (wall-clock time, total tokens generated) for GHPO vs. baseline GRPO.

## Score and Decision

Let me calibrate against similar papers. I'll search for relevant comparison papers.

Looking at the human scores for comparable papers:
- **Auto-CEI** (curriculum/adaptive RL for LLM reasoning): Accept Poster, avg ~7. Strong motivation, good results, limited model variety but clean ablations.
- **WebRL** (curriculum RL for LLMs): Accept Poster, avg ~6.5. Novel curriculum, solid results, but limited baselines and some manual curation.
- **VinePPO** (RL for LLM reasoning): Reject, avg ~5. Missing key baselines, limited novelty.
- **HP3O** (hybrid on/off-policy): Reject, avg ~5.75. Incremental novelty, limited baselines.

GHPO has stronger empirical results than VinePPO and HP3O, but shares some of their weaknesses (incomplete baselines, theoretical imprecision). Compared to Auto-CEI (accepted at ~7), GHPO's empirical improvements are similar in magnitude, but Auto-CEI had cleaner ablations and more transparent methodology. GHPO's objective formulation has a genuine theoretical inconsistency that Auto-CEI doesn't share, and its missing ablations are more damaging because the core mechanism (hints) is so close to SFT.

The empirical effectiveness is real, but the presentation overclaims the theoretical contribution (difficulty-aware curriculum vs. binary failure detection, principled hybrid vs. context-augmentation heuristic), and key ablations are missing. These are significant but not fatal issues. The paper makes a practical contribution that clearly works.

**Score: 5.5** — The paper addresses an important problem and demonstrates solid empirical improvements, but the objective formulation has an unacknowledged theoretical inconsistency, critical ablations are missing (always-hint and SFT baselines), and the "difficulty-aware curriculum" and "hybrid RL+imitation" narratives overstate the simplicity of the actual mechanism (binary failure detection + hint injection). The contribution is real but more incremental and heuristic than presented.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>