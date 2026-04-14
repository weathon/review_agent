=== CALIBRATION EXAMPLE 25 ===

# Final Consolidated Review
## Summary

ProgressCounts proposes a two-stage framework for automated reward engineering: an LLM generates a *progress function* that maps environment states to a small vector of subtask-progress scalars, and those scalars are discretized to produce count-based intrinsic rewards. By reducing the LLM's role from full reward-function authorship (where numerical calibration errors are fatal) to coarse semantic progress identification (where approximate values suffice), the method achieves 0.59 average success rate on the 20-task Bi-DexHands benchmark—outperforming Eureka (0.55) and expert-written dense rewards—using only 4 policy samples versus Eureka's 80.

---

## Strengths

- **Conceptually sharp decomposition of the LLM's role.** The key insight—separate *what matters for progress* (LLM's semantic strength) from *how to turn that into a reward signal* (count-based heuristics that are invariant to scale/weighting errors)—is genuinely novel and cleanly motivated. Most competing methods (Eureka, EUREKA-style RLHF) ask the LLM to also calibrate numerical reward weights, which is where LLMs systematically fail. ProgressCounts architecturally avoids this failure mode.
- **Well-supported ablation revealing necessity of both components.** Table 1 shows that removing either component causes a large aggregate drop: ProgressAsReward drops from 0.59 to 0.45, SimHashCounts drops to 0.34. The joint ablation structure makes it clear that neither LLM-guided progress alone nor generic count-based exploration alone is sufficient, which is the paper's central claim.
- **First method to achieve non-trivial success on TwoCatchUnderarm.** Section 5.2 and Figure 4 show ProgressCounts reaching 0.55 success rate on a task where all baselines (including Eureka) achieve zero success given the same total sample budget (2B environment samples per policy). This is a notable and concrete empirical milestone.
- **Practical 20× reduction in outer-loop sample cost.** Generating 4 progress functions instead of 80 reward functions reduces LLM query cost and RL training cost substantially. Given that each policy sample involves 100M environment steps, this is a practically meaningful efficiency gain.

---

## Weaknesses

### Fatal
None.

### Major

- **Unequal statistical rigor between ProgressCounts and ablated methods (Table 1).** The paper explicitly states that ProgressCounts results are averaged over 5 trials while ProgressAsReward and SimHashCounts are single-trial numbers. Bi-DexHands tasks are well-known to exhibit high variance across seeds. At the task level, multiple comparisons flip sign: DoorCloseOutward (0.90 vs. 1.00), SwingCup (0.97 vs. 0.99), CatchUnderarm (0.76 vs. 0.88), CatchOver2UnderArm (0.90 vs. 0.94)—all in favor of ProgressAsReward on the single-trial numbers. Without equal variance treatment, the conclusion that ProgressCounts reliably outperforms ProgressAsReward cannot be firmly drawn at the task level. The aggregate gap (0.59 vs. 0.45) is suggestive but not robust without matching statistical rigor.

- **Optimism bias from best-of-4 selection without reporting mean.** The experimental protocol (Section 5.1) explicitly selects the best-performing policy from 4 independent training runs. The paper reports only this best-of-4 value as the performance metric. A practitioner deploying ProgressCounts without oracle knowledge of which run succeeded would observe the *mean* policy performance, not the best. Since this selection mechanism is central to the method, failing to also report mean (or median) across policy samples makes the 0.59 figure difficult to interpret as a practical performance estimate. This concern is particularly acute for tasks like TwoCatchUnderarm where variance is likely very high.

- **y_i direction variables introduced but never connected to the algorithm.** Section 4.1.1 states that the progress function "also outputs additional variables [y_1, y_2, …, y_k] that inform our framework whether the progress variables x_i are increasing or decreasing." No subsequent section explains how these y_i variables are used in the binning function D, the discretization procedure, or the reward computation. Readers cannot determine whether these outputs affect the method or are simply unused artifacts of the API design.

### Minor

- **State aliasing in the summation-based bin function.** Step 3 of §4.2.2 defines B(s) = D(P(s)) = Σᵢ x'ᵢ, meaning the bin index is the *sum* of individually discretized subtask progress values. For a two-subtask task with each x'ᵢ ranging over {0,…,m}, the combination (x'₁=3, x'₂=7) maps to the same bin as (x'₁=7, x'₂=3). These represent meaningfully different task states (e.g., advanced on subtask 1 but not 2 vs. the reverse). The paper does not acknowledge this aliasing or justify why summation is preferred over a tuple representation. This could harm exploration efficiency on tasks with truly independent subtask phases.

- **Value range estimation for discretization is underspecified.** Step 1 of the discretization (§4.2.2) says the procedure "estimates relevant value ranges (minᵢ, maxᵢ) for each xᵢ from progress data." It is unclear whether this is done from short offline rollouts, from early training data, or from heuristic inspection. If done online, there is a chicken-and-egg dependency where early bins may be poorly calibrated. The paper delegates to Appendix A.6 but does not describe the procedure in the main text.

- **No analysis of why 7/20 tasks fall below Eureka.** The paper acknowledges in passing that ProgressCounts exceeds Eureka on 13 of 20 tasks, but provides no systematic analysis of what properties—task structure, progress function complexity, observation space type—characterize the 7 tasks where it underperforms. For instance, Switch achieves 0.00 success rate for all methods including ProgressCounts. Understanding when progress functions fail to generate useful bins is critical for scoping the method's applicability.

- **Figure 2 labeling is ambiguous.** The figure alt-text places a red dot at (0 policy samples, 0.61) labeled "Human Dense Reward," yet the main text (line 1936) states ProgressCounts achieves 0.59 and is "13% higher than human-written dense rewards"—implying human dense rewards achieve ~0.46–0.52, not 0.61. The embedded table in the figure shows "Human Dense Reward" at 0.61, which conflicts with the narrative. This appears to be a figure labeling error (the red dot may be intended to represent ProgressCounts at 4 policy samples) and should be corrected.

### Tiny

- No sensitivity analysis on λ_c or the number of bins (both fixed at 1e-3 and 1000 across all 20 tasks). Count-based rewards can be sensitive to these hyperparameters.
- No discussion of count-based reward saturation in highly parallel environments (standard PPO on Bi-DexHands uses many parallel rollout workers, which may rapidly exhaust novelty counts for early bins).

---

## Nice-to-Haves

- **Include RND or ICM as an additional exploration baseline.** SimHash is used as the "generic counts" ablation, but SimHash (2007) is considerably older than modern intrinsic motivation methods (RND: Burda et al., 2018). Including RND would better quantify the contribution of LLM-guided binning over contemporary exploration methods.
- **Report mean policy performance alongside best-of-4.** Even a simple table showing mean vs. best across policy samples would greatly clarify the practical reliability of the method.
- **Analysis of failure tasks.** A case study on Switch (0.00 for all methods) and similar hard failures would help practitioners understand ProgressCounts' boundary conditions.
- **Sensitivity analysis on bin count and λ_c.** Even a small grid over {500, 1000, 2000} bins and {1e-4, 1e-3, 1e-2} for λ_c on a representative task would address reproducibility concerns.
- **More prominent discussion of per-domain engineering cost.** The paper accurately states that the feature library is created once per domain and amortized across tasks (Section 4.1.2), but Table 2's dramatic failure of CatchUnderarm without the library deserves more prominent discussion of what constitutes a "domain" and how much effort library construction requires for genuinely new domains.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **"TwoCatchUnderarm results are inconsistent" (Spark Finder).** The 0.03 in Table 1 is the 100M sample budget result; the 0.55 in Section 5.2 uses 2B samples per policy (different experimental configurations explicitly described in the paper). Not an inconsistency—a misread of the experimental setup.
- **"The 20× claim conflates non-iterative with iterative search and is therefore misleading" (Harsh Critic, presented as major).** The paper is transparent that ProgressCounts uses 4 independent parallel samples while Eureka uses 80 evolutionary samples. The paper explicitly says "without requiring costly feedback-driven evolution." The claim is about *sample count*, not algorithm equivalence, and is accurately stated. Weakening the framing beyond what the paper does would misrepresent the finding.
- **"Unfair comparison with Eureka because the 4 policy samples have no inter-sample learning" (Harsh Critic).** This is a difference in method, not a flaw in reporting. The paper positions this architectural difference as a feature (avoiding costly evolution), not a bug, and empirically validates it. Criticizing a method for not being a different method is scope creep.
- **Request for RND to *replace* SimHash as the primary ablation baseline.** SimHash counts are a reasonable representative of generic hash-based count methods. Requesting RND as the mandatory ablation is not a standard requirement for this type of paper.
- **"Missing related works" criticisms.** Per review guidelines, no missing related works criticisms are included.
- **Demand for theoretical proofs for the count-based convergence or the discretization.** This is an empirical systems paper; theoretical guarantees for count-based exploration in continuous spaces are not a standard requirement in this subfield.

---

## Novel Insights

The most insightful observation from the reviews—confirmed by the paper's ablations—is that count-based exploration is surprisingly robust to imperfect binning in a way that dense reward functions are not. The paper's Discussion (§6) notes this but it deserves stronger emphasis: the reason ProgressCounts can succeed with coarse, LLM-generated progress functions while LLM-generated dense rewards (ProgressAsReward) fail is that a count signal proportional to 1/√c(B(s)) degrades gracefully as B(s) becomes less informative (you get less targeted exploration, but not reward hacking), whereas a misweighted dense reward can systematically direct the agent toward the wrong objective. This asymmetry in robustness is the deeper architectural justification for the paper's design—and it is currently underemphasized relative to its explanatory power.

---

## Suggestions

1. **Re-run ablations with equal trial counts.** Run ProgressAsReward and SimHashCounts with 5 seeds per task, matching ProgressCounts. Report means and standard deviations for all methods. This is essential for validating the task-level comparisons in Table 1.
2. **Report mean and best across policy samples separately.** Add a companion metric to Table 1 showing the mean-of-4-samples performance alongside best-of-4. If the gap between mean and best is large, this should be discussed explicitly.
3. **Clarify or resolve the y_i variable usage.** Either (a) explain exactly where and how y_i variables enter the discretization or reward computation, or (b) remove the mention if they are unused in the current version.
4. **Fix or clarify Figure 2 labeling.** Make explicit what the red dot represents (ProgressCounts or Human Dense Reward) and ensure the legend, embedded table, and main text values are numerically consistent.
5. **Address the aliasing issue in B(s) = Σᵢ x'ᵢ.** Either justify why the sum is empirically sufficient (e.g., subtasks are mostly sequential so the same sum rarely arises from reversed orderings), or replace it with a tuple-based representation and compare empirically.
6. **Add one paragraph to the Discussion explicitly framing the robustness asymmetry** between count-based and dense reward approaches to imperfect LLM outputs—this is the deepest justification for the paper's design and is currently stated too briefly.

---

**Evaluation axes:**
- **Novelty:** High. Redirecting the LLM's role from reward calibration to state abstraction is a genuinely new framing, and the combination of LLM-generated progress bins with count-based rewards is not a trivial extension of prior work.
- **Technical soundness:** Moderate. The core method is sensible, but the summation-based bin aliasing is an unacknowledged issue, and the y_i variables are described without being algorithmically connected.
- **Empirical support:** Moderate-to-good. The aggregate results are compelling and the ablation structure is well-designed, but the unequal trial statistics in Table 1 and the best-of-4 selection bias prevent confident task-level conclusions.
- **Significance:** High. A 20× reduction in the reward engineering outer loop, with matching or better performance, is a practically important contribution to the LLM-for-RL literature.
- **Clarity:** Good overall, with notable gaps around the y_i variables, the range estimation procedure, and Figure 2's labeling.

# Actual Human Scores
Individual reviewer scores: [5.0, 5.0, 8.0, 5.0]
Average score: 5.8
Binary outcome: Reject
