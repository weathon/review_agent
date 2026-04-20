## Summary

This paper introduces DICE, a method for bootstrapping DPO-tuned language models using their own implicit rewards as preference signals for iterative self-alignment. The approach combines length-regularized reward shaping during dataset construction (rather than in the loss function) and experience replay to prevent training collapse. Experiments on two base models (Zephyr-7B and Llama-3-8B) show +8–9% LC win rate improvements on AlpacaEval 2 without any external feedback. The method is computationally efficient and practically valuable for low-resource alignment pipelines.

## Strengths

- **Clean core idea with strong empirical validation**: Using the DPO implicit reward ($r = \beta \log(\pi_\theta/\pi_{\text{ref}})$) as a self-supervision signal for iterative alignment is a conceptually straightforward yet effective approach. Table 1 demonstrates substantial improvements (+8.02% for Zephyr, +9.35% for Llama-3), substantially outperforming both offline DPO baselines and LLM-as-a-Judge methods, confirming that implicit rewards provide a more effective signal than prompting-based judging.

- **Length regularization at dataset construction stage is efficient**: Unlike Park et al. (2024) which regularizes length in the loss function, DICE shapes rewards during dataset construction (Eq. 5) and automatically finds the optimal penalty $\alpha^*$ by minimizing average absolute length difference (Eq. 6). Figure 2 provides compelling visual evidence: vanilla implicit rewards produce a heavily right-skewed length difference distribution (mean 1031), while the regularized approach yields a balanced distribution (mean −21). Table 4 confirms $\alpha^*$ achieves the best LC win rate compared to both $\alpha=0$ and $2\alpha^*$.

- **Experience replay ablation cleanly validates the necessity of mixing strategies**: Figure 3 demonstrates that $\gamma=0.5$ achieves the best LC win rate (20.77%), while pure self-generated data ($\gamma=0.0$) causes catastrophic collapse in iteration 2 (LC drops to 4.5%). This directly supports the claim that implicit rewards are imperfect proxies requiring supplementation with original preference data.

- **Implicit reward outperforms comparable scalar reward models**: Table 5 shows the DPO implicit reward achieves a higher alignment rate (0.698) with GPT-4o preference labels than both an internal reward model trained on the same data (0.624) and an external reward model trained on 555k data (0.656), justifying the use of implicit rewards over training a separate scalar RM for self-alignment.

## Weaknesses

### Fatal
None

### Major

- **Hyperparameter selection on the primary evaluation benchmark**: Section 4.1 (line 170) states: "We hypertuned $\beta \in \{0.01, 0.1\}$ based on the model performance on AlpacaEval 2 for each method and model separately. For our approach, we additionally hypertuned the experience replay ratio $\gamma$ using cross-validation." While the authors note they applied the same tuning procedure to baselines, selecting hyperparameters directly on AlpacaEval 2—the same benchmark used to report final results—introduces a degree of circularity. The LC win rate metric explicitly penalizes verbosity, and DICE's length regularization directly targets this penalty. Tuning $\gamma$ (which controls the balance between on-policy self-generated data and offline data) on the test metric means the reported +8–9% improvements are partially optimized for that specific benchmark. This is a widespread practice in the alignment community and does not invalidate the results, but it does mean the true generalization gap to truly held-out preference benchmarks is unknown.

### Minor

- **No analysis of implicit reward calibration drift across iterations**: The method assumes the DPO implicit reward ($r = \beta \log(\pi_\theta/\pi_{\text{ref}})$) remains a valid, calibrated proxy for human preference as the policy iteratively drifts from its reference state. As iterations proceed, $\pi_{\theta(t-1)}$ and $\pi_{\theta(t-2)}$ diverge, and the log-ratio reward signal becomes increasingly sensitive to distribution shift. The paper provides no analysis of how reward calibration changes over iterations—no reward distribution shift plots, no BT loss against held-out preferences, no KL divergence tracking. The "alignment rate" evaluation in Section 4.4 is a single snapshot on 500 points from the first iteration. While the empirical results demonstrate the approach works for two rounds, the lack of calibration analysis means the method's behavior over more iterations (already acknowledged as a limitation in Section 5) is not well understood.

- **Dataset-level length neutrality does not guarantee pairwise length neutrality**: The length-regularization objective (Eq. 6) minimizes the *expected* length difference across the constructed dataset. However, DPO's gradient updates are computed per-pair. A dataset-wide mean of zero allows long-winning/short-losing pairs to be statistically offset by short-winning/long-losing pairs elsewhere, meaning individual training pairs can still carry strong length-biased signals. The authors show in Figure 2 that the regularized distribution is more centered than the vanilla one, which is helpful, but the mechanism does not mechanically ensure pairwise debiasing. This is a subtle but real gap between the stated goal (debiased preference pairs) and the optimization target (dataset-level mean).

- **Compatibility with DAP variants (IPO, KTO, Hinge) is empirically demonstrated but theoretically under-specified**: Table 3 shows DICE-generated datasets improve IPO, KTO, and Hinge losses. However, these algorithms optimize different implicit reward geometries: KTO expects unpaired magnitude losses while IPO uses margin-based losses. Using DPO-specific log-density ratios to generate preference pairs for structurally distinct loss functions lacks theoretical grounding—there is no discussion of whether reward scaling or different pair-selection criteria should apply per-algorithm. The empirical results are positive, but the theoretical justification for this cross-algorithm compatibility is absent.

### Trivial

- **Overclaim on generalizability**: The paper claims DICE is a "general purpose approach that can improve alignment for any single DPO-tuned base model" (Section 1), but experiments are limited to two models (Zephyr-7B-beta and Llama-3-8B-DPO) that are both heavily pre-tuned and already DPO-adapted. Application to SFT-only base models or models from different families is not tested.

## Nice-to-Haves

- Evaluation on length-agnostic benchmarks (e.g., IFEval, HelpSteer 2, MT-Bench) would help distinguish genuine alignment improvements from verbosity optimization against AlpacaEval 2's LC metric, and also address the length-bias concern more directly.
- Qualitative case studies showing response pairs ranked by implicit reward (both successes and failures) would provide insight into what preference boundaries the method actually learns.
- Tracking implicit reward calibration and KL divergence across iterations would strengthen the claim that the bootstrapping mechanism remains informative over time.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Reference policy notation inconsistency"**: The harsh critic claimed the paper inconsistently uses $\pi_{\theta(t-1)}$ vs. $\pi_{\theta(t-2)}$ as reference. The paper is actually consistent: the DPO loss (Algorithm 1, line 9) uses $\pi_{\theta(t-1)}$ as the reference policy for training, while the implicit reward computation uses $\pi_{\theta(t-1)}$ as the target policy and $\pi_{\theta(t-2)}$ as the *implicit reward reference*. These are two distinct roles (training reference vs. implicit reward computation reference), not an inconsistency.

- **"LLM-as-a-Judge is poorly defined / GPT-4o is not ground truth"**: The paper itself already acknowledges the coarse scoring issue in Section 4.2 (line 208): "We hypothesize this may be caused by the coarse rewards which are not able to provide effective preference signals when responses are of high quality (the prompt template requires LLM judge to provide a discrete score from 0 to 5)." The authors have already addressed this concern.

- **"Method fails to regularize pairwise learning because Eq. 6 is dataset-level"**: While the critic's point about dataset-level vs. pairwise optimization is technically valid (moved to Minor), the claim that this leaves the "claimed debiasing mechanism mechanically incomplete" overstates the issue. The paper demonstrates empirically (Figure 2, Table 4) that the approach effectively debiases the distribution, making this more of a presentational gap than a fundamental flaw.

- **"Cannot independently verify $\beta$ and $\gamma$ tuning details"**: The details are stated to be in Appendix F, which the parser strips. Per the hard rules, references to missing appendices should be removed as they exist in the original submission.

## Novel Insights

The paper's key insight—that the DPO implicit reward, already computed during preference alignment, can be repurposed as a self-supervision signal for iterative bootstrapping without any additional training or external models—effectively collapses the RLHF pipeline back to a purely offline, self-contained procedure. The finding that the implicit reward outperforms both an internal scalar RM trained on the same data and an external RM trained on 10x more data (Table 5) suggests the DPO implicit reward captures model-specific preference geometry that a separately trained reward model cannot easily match. Combined with the clear demonstration that experience replay prevents catastrophic collapse (Figure 3), the paper provides a practical, low-overhead pathway for incremental alignment improvement that may be particularly valuable in settings where external annotation or reward model training is infeasible.

## Suggestions

- Use a held-out validation subset from UltraFeedback (or a separate preference dataset) for hyperparameter tuning rather than AlpacaEval 2, to ensure the primary results reflect genuine generalization.
- Add a table or figure tracking the distribution of implicit reward values, KL divergence from the base model, and pairwise length differences across iterations to provide transparency on the bootstrapping dynamics.
- Clarify the scope of the "general purpose" claim to reflect the tested models, or add experiments on an SFT-only base model to test generalizability beyond pre-tuned DPO checkpoints.

---

## Score and Decision

**Calibration anchors**:
- **SeRA** (uIGnuyDSB9.md): Uses implicit reward margins for self-reviewing iterative alignment; accepted poster with scores **6, 6, 6, 6**. Very similar topic (implicit reward self-alignment), comparable experimental scope.
- **SALMON** (xJbsmB8UMx.md): Self-alignment with instructable reward models; accepted poster with scores **6, 6, 6, 8**. Self-alignment with strong results and some reviewer concerns about missing comparisons.
- **D2PO** (OspqtLVUN5.md): DPO variant with temporal decay for length bias; accepted poster with scores **6, 5, 6, 8**. Similar length-bias mitigation theme with comprehensive ablations.
- **iREPO** (NtAXAvIYuN.md): Iterative self-alignment; **rejected/withdrawn** with scores **3, 3, 5, 3, 3** due to weak experimental design and unfair baselines. DICE is notably stronger than this anchor in experiment quality.
- **Reward-Augmented Data** (mjtCqmujYP.md): Strong results but weak theoretical backing; **rejected** with scores **5, 3, 6, 6, 6**. Comparable result strength, similar concerns about theoretical grounding.

DICE is comparable in quality to the accepted papers in the same space (SeRA: 6s; D2PO: 6/5/6/8; SALMON: 6/6/6/8). It has clean empirical results and a conceptually elegant approach, with weaknesses primarily around test-set hyperparameter tuning and lacking reward calibration analysis—similar issues to those raised against accepted works. It is clearly above rejected papers like iREPO. The main empirical results are strong and the ablations are well-designed.

## Score

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>