# Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
When language models (LMs) are trained via reinforcement learning (RL) to generate natural language “reasoning chains”, their performance improves on a variety of difficult question answering tasks. Today, almost all successful applications of RL for reasoning use binary reward functions that evaluate the correctness of LM outputs. Because such reward functions do not penalize guessing or low-confidence outputs, they often have the unintended side-effect of degrading calibration and increasing the rate at which LMs generate incorrect responses (or “hallucinate”) in other problem domains. This paper describes RLCR (Reinforcement Learning with Calibration Rewards), an approach to training reasoning models that jointly improves accuracy and calibrated confidence estimation. During RLCR, LMs generate both predictions and numerical confidence estimates after reasoning. They are trained to optimize a reward function that augments a binary correctness score with a Brier score—a scoring rule for confidence estimates that incentivizes calibrated prediction. We first prove that this reward function (or any analogous reward function that uses a bounded, proper scoring rule) yields models whose predictions are both accurate and well-calibrated. We next show that across diverse datasets, RLCR substantially improves calibration with no loss in accuracy, on both in-domain and out-of-domain evaluations—outperforming both ordinary RL training and classifiers trained to assign post-hoc confidence scores. While ordinary RL hurts calibration, RLCR improves it. Finally, we demonstrate that verbalized confidence can be leveraged at test time to improve accuracy and calibration via confidence-weighted scaling methods. Our results show that explicitly optimizing for calibration can produce more generally reliable reasoning models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces RLCR (Reinforcement Learning with Calibration Rewards), which augments standard binary-reward RL for reasoning LMs with a Brier score term to encourage calibrated confidence. Unlike conventional RL that promotes overconfidence, RLCR jointly optimizes accuracy and uncertainty estimation. The authors also provide some theoretical results showing that RLCR does not induce the model to generate uncertain and wrong answers. Experiments with Qwen2.5-7B on reasoning benchmarks show that RLCR maintains accuracy while improving calibration.

### Strengths
Strengths

- The paper is well written and easy to follow, with clear organization and motivation.

- Although using proper scoring rules for calibration is a well-established idea, the work’s contribution lies in proposing an RL objective that maintains accuracy while improving calibration in reasoning LMs, which is still an important direction to explore.

- The authors provide comprehensive empirical validation, showing that the proposed method consistently yields better-calibrated reasoning models without degrading accuracy across diverse tasks.

### Weaknesses
Weaknesses and Questions

- I would like to see a deeper analysis of the confidence distribution. For instance, are the confidence values truly diverse across different inputs, or do they mostly cluster around a narrow range regardless of the question type? Understanding this would clarify whether the observed calibration improvements reflect genuine uncertainty modeling or simply uniform confidence outputs near accuracy.

- While the paper theoretically explains why adding NLL instead of the Brier score breaks the correctness incentive, it would be informative to include an empirical comparison showing whether this theoretical difference actually affects performance or calibration in practice by conducting experiments with objectives that include NLL.

- The reported AUROC scores (around 0.5–0.6) seem relatively low; it’s unclear whether such values reflect meaningful discrimination ability. Some qualitative analysis or normalized baselines could help interpret these numbers more convincingly.

- It would be useful to examine how RLCR behaves in larger-scale or instruction-following settings, beyond the QA and math reasoning benchmarks. This would test whether the proposed calibration benefit generalizes to more complex reasoning domains.

- I’m also curious how the model behaves when generating multiple samples for the same question—does it assign noticeably different confidence scores to different (possibly contradictory) answers, or are the confidences largely similar across samples? This would give insight into how well the model captures epistemic uncertainty across reasoning trajectories.

- In Figure 4(a), the standard deviation of confidence across reasoning chains appears to be roughly 0.05–0.1, meaning that on a 0–100 scale, confidence typically fluctuates by 5–10 points, or up to 10–20 points across extremes. Doesn’t this suggest that the model’s confidence is still quite unstable across samples, rather than genuinely consistent? Some further interpretation of this behavior would help clarify the robustness of the claimed self-consistency.

### Questions
See above section

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Reinforcement Learning with Calibration Rewards (RLCR), which improves the reward function of RLVR by combining with a term based on the Brier score over verbalized confidence estimates through reasoning, incentivizing both accuracy and calibration. Empirical evaluations across multiple QA benchmarks in factual, math, and long-form reasoning confirm its effectiveness in achieving better calibration while maintaining or improving accuracy, compared with standard RLVR and RLVR with additional classifier or probe. The paper also demonstrates that RLCR improves the self-consistency of confidence and the verbalized confidence scores of RLCR can be incorporated into test-time scaling to yield improvements in both accuracy and calibration.

### Strengths
* The method is simple and intuitive, and shows potential and effectiveness in improving both task accuracy and calibration.
* The reward design and the choice of score rule are well-motivated and theoretically justified.
* Uncertainty-aware reasoning and RL training for calibration is a timely and relatively under-explored topic.

### Weaknesses
- The paper should make it clearer the contributions and benefits compared with previous RL-based calibration methods cited in related works [1,2,3]. Given the overlap in scope, it would be very useful to include these as baselines potentially stronger than the RLVR variants. Also, given the recent debate on the contamination issue of Qwen-family models in RLVR, it would be helpful to test on other models as well.
- It would be good to isolate / disentangle the role of the Brier term and uncertainty-style CoT reasoning in improving calibration in both in-distribution and out-of-distribution settings. Is the gain mostly coming from the reward formulation (unique contribution of this paper) or simply a better confidence estimate from uncertainty reasoning?
- There is still much room to improve for the OOD settings. Given that SFT warmup is also widely-adopted, the degraded generalization performance of SFT+RLCR is a bit concerning. More insights and analyses on this would be very helpful.

---

[1]  Taming overconfidence in LLMs: Reward calibration in RLHF

[2]. LACIE: Listener-aware finetuning for calibration in large language models.

[3]. Sayself: Teaching llms to express confidence with self-reflective rationales

### Questions
- How sensitive are the results to the reasoning template (e.g. the analysis section, ordering)?
- For SFT warmup in Math, why not use Deepseek-R1 to output confidence scores?
- For GRPO, why remove the KL regularization?
- Are there other working scoring rules suitable for capturing uncertainty, besides Brier score?
- Is it possible to weigh the correctness and the calibration reward terms differently, and explore the tradeoff?
- Does this method show potential when there is no clear ground-truth?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper identifies a key limitation of reinforcement learning with verifiable rewards (RLVR): the binary correctness reward penalizes equally whether the model abstains or produces an incorrect answer. Consequently, models that are initially well-calibrated tend to become overconfident after RL training. The authors propose reinforcement learning with calibration rewards (RLCR), which augments the binary correctness reward with a Brier score term based on proper scoring rules. This modification explicitly incentivizes calibrated confidence estimation in addition to correctness. RLCR substantially improves calibration both in-domain and out-of-domain without sacrificing accuracy, compared to RLVR and other post-hoc confidence baselines. Moreover, combining RLCR with confidence-weighted test-time scaling methods (e.g., weighted majority voting, best-of-N) further enhances accuracy.

### Strengths
- The paper tackles a well-known yet underexplored issue in RL-based reasoning: overconfidence induced by correctness-only rewards. This concern has also been raised in recent works such as *“Why Language Models Hallucinate” (OpenAI, 2025)*, but practical algorithmic remedies have been lacking.

- The proposed modification is extremely intuitive—adding a proper scoring rule (Brier score) term to the RL objective—and is supported by clean theoretical analysis showing that it jointly optimizes accuracy and calibration.

- Furthermore, the analysis of verbalized confidence self-consistency (examining both intra- and inter-solution coherence) meaningfully connects calibration to reasoning coherence.

### Weaknesses
1. **(Minor) Inconsistency in the definition of proper scoring rules.**
    - Equation (4) defines a proper scoring rule as one whose expected value is minimized.
    - However, the examples mix loss and reward conventions:
        - The Brier score (Eq. 6) is defined as a loss (minimized).
        - The logarithmic (Eq. 5) and spherical (Eq. 7) scores are utility functions (maximized).
    - The statement that “all these scores... are maximized” (line 144) contradicts both Eq. (4) and the Brier example. 
    - This confusion between “loss” and “score” formulations creates unnecessary ambiguity and should be clarified for consistency.
        
2. **Lack of ablation on the necessity of “confidence analysis.”**
    - The paper claims that *chain-of-thought reasoning about uncertainty* improves calibration. However, this structured reasoning step (the `<analysis>` tag) adds considerable inference and training cost compared to simply predicting a scalar confidence value.
    - The results suggest that SFT+RLCR achieves the lowest ECE but also suffers from a substantial accuracy drop, attributed to catastrophic forgetting.
    - Therefore, it remains unclear whether the observed calibration improvement stems from enhanced uncertainty reasoning or from task-specific overfitting introduced by SFT.
    - Critically, the paper does not include RLCR results without confidence analysis (i.e., without the `<analysis>` tag), making it difficult to isolate the contribution of verbalized uncertainty reasoning to calibration performance.

### Questions
1. Could you report or comment on RLCR without confidence analysis (i.e., without `<analysis>` reasoning, only outputting numerical confidence)?
2. Could the authors clarify how RLCR differs from or complements traditional Best-of-N sampling (using external reward model or sentence likelihood), and whether its advantages persist when compared under identical computational budgets (number of samples)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
- This paper proposes a reward function that simultaneously incentivizes both answer correctness and probabilistic calibration, moving beyond binary correctness signals traditionally used in RLHF. By introducing a smooth, bounded Brier-score term, the method encourages models to be not only accurate but also appropriately uncertain. The proposed approach, RLCR (Reinforcement Learning with Calibrated Rewards), addresses common issues like reward hacking found with log-likelihood and unbounded scoring rules. The authors demonstrate that RLCR improves both calibration and output quality across reasoning tasks. Extensive experiments show that RLCR outperforms binary reward baselines in terms of calibration error, and answer diversity. Notably, the method avoids overconfident hallucinations and penalizes low-confidence correct answers, promoting reliable confidence estimates.

### Strengths
- The paper tackles a timely and practically-relevant problem supported by a fair amount of experiments. 
- Overall, the paper is clearly written and easy to follow.

### Weaknesses
- **Model coverage** The paper only tests Qwen2.5-7B, and additional models, maybe a non-reasoning model, should be included to demonstrate that RLCR generalizes beyond a single model. I must state that this is not a mere superficial comment rather one of my major concerns.
- **Fair evaluation** Page 27 notes that RLCR produces shorter `<think>` sections than RLVR. It is unclear whether the results were evaluated under equal inference budgets, and this should be explicitly specified to ensure fairness. Further, I wonder whether reasoning length correlates with uncertainty. 
- **Missing hyperparameters** Key training details are absent, including GPU type, learning rate, and GRPO training configuration.
- **Oracle-based labeling concern** For Math datasets, the uncertainty labels are generated using DeepSeek-R1, a far larger model. This could transfer oracle-level knowledge and confound the evaluation.  Please correct me if I understood something wrong.
- **Inconsistent results and missing discussion** While Table 1 reports averaged results, some findings contradict this trend. For example, in Appendix E.3 (Table 3), RLVR achieves an ECE of 0.09 whereas RLCR records 0.30 with nearly identical accuracy. The paper should analyze and discuss such failure cases. Likewise, lines 356–357 state that RLCR maintains accuracy, but it clearly lags behind RLVR.  
- **Formatting confound (analysis tag)** RLCR uses an `<analysis>` tag that allows longer uncertainty reasoning, whereas baselines do not. This formatting difference must be ablated by testing RLCR without the analysis section so that the only change is the reward function. This is crucial because (1) RLCR incurs higher inference costs (see p. 24, 26), and (2) prior work (Yoon et al., 2025) already shows that extended uncertainty reasoning improves calibration. Further, a fair comparison should thus include *RLVR + uncertainty analysis* as baselines. 
- **Missing baselines** The paper omits comparisons with prior works that enhance verbalized uncertainty. Even if those do not jointly optimize accuracy and calibration via RL, they should be included for context. Moreover, instead of training separate RLVR classifiers, a more natural baseline would be SFT with uncertainty labels (e.g., oracle- or self-consistency-based) alongside existing RL-calibration methods. 
- **Failure cases for unbounded scoring rules** Table 1 shows BCE classifiers outperforming Brier ones, yet there is no empirical support for the theoretical claim that unbounded proper scores lead to overfitted low-confidence answers. Experiments using other scoring rules such as logarithmic (Eq. 5) or spherical (Eq. 7) are needed.  
- **Catastrophic forgetting and missing regularization** In Table 1(b), SFT + RLCR performs worse, and the authors blame this on catastrophic forgetting. It is unclear why KL-regularization—a rather straightforward fix—was not applied.  
- **Figure 3 interpretation** Figure 3 indicates (1) some incorrect answers appear with high confidence and (2) majority voting consistently underperforms confidence-weighted voting by a small margin. Clarify: (a) how majority voting handles ties; and (b) include plots of both accuracy and frequency (y-axis) versus confidence bins (x-axis) to visualize calibration.

### Questions
- **Definition of p** Lines 92 and 134 use $p$ without definition; it should be mentioned that $p$ is the true distribution. In line 145, I doubt that the true probability $p(a=1)$ should be $p(a)$.
- **GRPO acronym** Line 202 should spell out GRPO and include a citation.  
- **Confidence scale inconsistency** Appendix B.4 defines confidence in [0, 100] at first, then [0, 1]. I wonder why this is the case.
- **Table 1 labeling** Upper panel uses “OOD” while the lower uses “OOD averaged”; these two should match.  
- **Dataset choice** Why use *hotpotqa-distracted*? Was the original HotpotQA too easy (near-perfect accuracy/calibration)?  
- **ECE binning** What happens if the number of bins increases (e.g., 20)?  
- **AUROC definition** Clarify whether AUROC is computed from verbalized confidence or token-level probabilities.

### Soundness
2

### Presentation
3

### Contribution
2
