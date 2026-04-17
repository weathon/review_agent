# Deep Think with Confidence

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 8

## Abstract
Large Language Models (LLMs) have shown great potential in reasoning tasks through test-time scaling methods like self-consistency with majority voting. However, this approach often leads to diminishing returns in accuracy and high computational overhead. To address these challenges, we introduce \textbf{Deep Think with Confidence (DeepConf)}, a simple yet powerful method that enhances both reasoning efficiency and performance at test time. DeepConf leverages model-internal confidence signals to dynamically filter out low-quality reasoning traces during or after generation. It requires no additional model training or hyperparameter tuning and can be seamlessly integrated into existing serving frameworks. We evaluate DeepConf across a variety of tasks and the latest open-source models, including Qwen3 and GPT-OSS series. Notably, on challenging benchmarks such as AIME 2025, DeepConf@512 achieves up to 99.9\% accuracy and reduces generated tokens by up to 84.7\% compared to full parallel thinking. Our code is available at https://github.com/facebookresearch/deepconf

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Deep Think with Confidence (DeepConf), a test-time method to enhance the reasoning performance and efficiency of Large Language Models (LLMs) by leveraging internal confidence signals from token distributions. DeepConf filters low-quality reasoning traces either offline (after generation) or online (during generation) using localized confidence measures like group confidence, bottom 10% group confidence, lowest group confidence, and tail confidence. It integrates with self-consistency majority voting, achieving weighted voting and early stopping to reduce computational overhead. Evaluations on mathematical benchmarks (e.g., AIME 2025) and models (e.g., GPT-OSS-120B, Qwen3-32B) claim up to 99.9% accuracy with 84.7% token reduction compared to standard majority voting. The method requires no additional training and is plug-and-play.

### Strengths
-  DeepConf demonstrates significant reductions in generated tokens (up to 84.7%) while maintaining or improving accuracy, which is crucial for deploying LLMs in resource-constrained settings. The online mode with adaptive sampling and warmup is a clever way to approximate offline performance in real-time, making it suitable for serving frameworks.
- Experiments cover multiple models (from 8B to 120B parameters) and challenging benchmarks (AIME, HMMT, BRUMO, GPQA). Results are averaged over 64 repetitions, adding statistical robustness. The ablation on confidence thresholds (η=10% vs. 90%) provides insights into trade-offs between precision and diversity in filtering.
-  No hyperparameter tuning or retraining is needed, and the method integrates seamlessly with existing parallel thinking approaches. The anonymous code release supports reproducibility.

### Weaknesses
- The paper assumes that higher confidence correlates with correctness (as shown in Figure 2), but this assumption is problematic. Confidence in large models is not necessarily calibrated, and without calibration analysis (e.g., ECE or reliability plots), confidence values cannot be interpreted as correctness probabilities. Moreover, low-confidence traces may still contain valid, diverse answers that are filtered out, potentially causing mode collapse. A more thorough analysis of calibration quality and failure modes (e.g., overconfident wrong answers) would be necessary to substantiate the claims.
-  While the idea of discarding uncertain reasoning paths during generation makes sense conceptually, the experimental setting (K=512) raises concerns about practicality. The paper does not report the computational overhead or the average inference cost compared to normal decoding. It is unclear how scalable the method is in realistic settings and how strongly the results depend on the choice of K. An ablation or sensitivity analysis on K would be essential to understand the trade-off between accuracy and efficiency.
- The paper mainly evaluates the method on benchmarks with well-defined ground truth answers. To better assess its generalization ability, it would be helpful to include experiments on open-ended tasks where correctness is less strictly defined. Such evaluation could reveal whether the proposed confidence mechanism remains effective under less constrained, real-world generation scenarios.
If the authors can address the above issues, I would be willing to raise my score.

### Questions
In addition to the weaknesses mentioned above, I am particularly concerned about the practical computational cost of DeepConf. The reported results are obtained under the setting of K = 512, which seems computationally heavy. It would be important to clarify the average inference cost and compare it with standard decoding. Moreover, how sensitive is the method to the choice of K? How does the performance change when K is reduced to a more realistic range for deployment?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Self-consistency with majority voting as a test-time scaling (TTS) strategy can effectively improve LLM reasoning accuracy; however, it multiplies computational cost and shows pronounced diminishing returns. To address this, the paper proposes DeepConf, an efficient, training-free TTS method that leverages a model’s internal confidence to dynamically filter low-quality reasoning paths. Across both offline and online settings, DeepConf achieves superior performance while using substantially fewer resources.

### Strengths
1. **Simple and effective with practical value.** Relying only on *Bottom 10% Group Confidence* enables effective adaptive sampling with early stopping and confidence filtering of high-quality reasoning paths. In both offline and online settings, it markedly lowers inference cost while delivering better performance, indicating broad applicability.

2. **Comprehensive and rigorous experiments.** The paper evaluates five mainstream open-source reasoning LLMs from three model families on five challenging benchmarks under two settings. Experimental details are clear and easy to follow. Thorough ablations, across different budgets and hyperparameters--demonstrate the method’s effectiveness, making the evidence methodologically solid.

3. **Clear writing and sound structure.** Centered on confidence and targeting cost-efficient TTS, the paper offers a detailed, comprehensive discussion and comparison of approaches based on internal token distributions and their limitations, and builds a logically coherent, step-by-step case for the proposed method.

### Weaknesses
1. **Limited task generalization.** The evaluation focuses primarily on mathematical reasoning, with little assessment in verifiable settings such as code generation. Adding at least one non-mathematical reasoning benchmark would strengthen claims of generality.

2. **Missing highly relevant strong baselines.** While the paper centers on confidence-based methods and compares unweighted vs. confidence-weighted majority voting, it lacks comparisons with closely related, strong baselines—e.g., ESC [1], DSC [2], and RASC [3]. Including these would improve completeness, better substantiate the method’s effectiveness, and clarify the source of gains.

3. **Missing related work** . The paper omits several classic/recent SOTA parallel-generation approaches under TTS in A.1, such as DVTS [4] and DORA [5].

[1] Escape Sky-high Cost: Early-stopping Self-Consistency for Multi-step Reasoning ICLR2024

[2] Make Every Penny Count: Difficulty-Adaptive Self-Consistency for Cost-Efficient Reasoning NAACL2025

[3] Leveraging Reasoning Paths for Efficient LLM Sampling NAACL2025

[4] Scaling Test-Time Compute with Open Models

[5] Every Rollout Counts: Optimal Resource Allocation for Efficient Test-Time Scaling NeurIPS2025

### Questions
1. **Online, extreme-difficulty cases.** When the model generates a long chain-of-thought (CoT), if certain “reflection” segments exhibit low confidence, could early stopping become overly aggressive—filtering out trajectories that might otherwise converge to the correct answer—and thus hurt accuracy? Should the definition of the group (G_i) adapt to the already-produced output length?

2. **Practical guidance on hyperparameters.** What are the recommended defaults and applicable ranges for (G_i), (\eta), (\tau), and (N_{\text{init}})? Is there a simple heuristic or automatic setting strategy for these?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces Deep Think with Confidence (DeepConf), a test-time method designed to improve the reasoning performance of LLMs. The paper argues that global confidence metrics are flawed as few high confident region can hide critical errors within low confidence segment. It proposes several new, more fine grained metrics such as group confidence and tail confidence, etc to capture local statistics. It then introduces two inference mode, in offline settings confidence filtering and weighted majority voting is applied. In online settings, an offline warmup is required to setup a stopping threshold, and adaptive sampling can be applied based on existing rollout results and partial confidence. The method demonstrates a accuracy improvement compared with majority voting and significant token saving in online settings.

In short, it is an empirical paper that extends the previous global confidence method with finer granularity to improve the test-time performance on token efficiency and accuracy.

### Strengths
1. Clear Motivation and Problem Framing: The paper provides a clear and intuitive justification for moving from global to local confidence metrics.
2. The experimental setup is comprehensive: The authors test across multiple model families and scales, on a suite of reasoning and QA benchmarks. The paper also includes detailed ablations on hyperparameters introduced in the method.
3. The proposed method is practical and simple with multiple variants that can adapte to different downstream need, making it easy and meaningful to integrate into existing frameworks.

### Weaknesses
1. Incremental Novelty: The primary novelty lies in the specific local metrics and the online early-stopping mechanism compared to prior works (e.g., Kang et al., 2025). While effective, the proposed method is an incremental improvement on existing ideas.
2. Overstated “No Hyperparameter Tuning.” The abstract’s “no … hyperparameter tuning” assertion is not faithful to the method as presented. The method introduces several new parameters that require choices: the filtering ratio, the online consensus threshold, the warmup steps and window sizes. For example, the reported ablations (e.g., Table 5 / Fig. 8) show the optimal filtering percentage varies by dataset and model, implying some form of tuning or at least selection is needed for new settings.
3. Warmup Cost Not Fully Accounted. The online mode requires an instance-level warmup that generates N full traces before early stopping is even applicable. For common majority-vote budgets e.g. K=32, setting N=16 is a major overhead that should be counted. The current amortized token-saving curves appear to exclude or underweight this cost.

### Questions
1. As for weakness 2: What default, model-agnostic settings do you recommend for these parameters? What minimal selection protocol (and validation budget) do you propose for new tasks/models? if above questions cannot be addressed, please revise the abstract/claims to acknowledge required hyperparameter choices. 
2. Regarding Weakness 3: Can the authors provide the token efficiency metrics taking offline warmup tokens into account? Can the confidence threshold be reused or shared across questions or within a dataset to reduce per-prompt warmup?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents Deep Think with Confidence (DeepConf), a test-time method that improves LLM reasoning accuracy and efficiency by leveraging internal confidence signals without retraining. The method addresses the inefficiency and diminishing returns of standard self-consistency (majority voting) by dynamically filtering or halting low-confidence reasoning traces. DeepConf replaces a single global confidence value with localized confidence metrics computed over sliding windows of tokens:
- Group confidence: mean of token-level confidence across overlapping windows;
- Tail / Bottom-10% confidence: highlight locally uncertain regions;
- Lowest Group Confidence: the weakest confidence window per trace, used for online early stopping.

In offline mode, DeepConf ranks and filters traces by these scores, applying confidence-weighted voting instead of naive majority voting.
In online mode, it adaptively terminates traces early once local confidence falls below a learned percentile threshold and stops sampling once vote consensus exceeds a preset $\tau$.

Across math-reasoning and QA benchmarks (AIME’24/’25, HMMT’25, BRUMO’25, GPQA-Diamond) and multiple models (DeepSeek-8B, Qwen3-8B/32B, GPT-OSS-20B/120B), the method achieves strong empirical results without any retraining or extra supervision.

### Strengths
1. Practical impact and simplicity: DeepConf can be deployed immediately without retraining or external calibration. It’s a drop-in improvement to any self-consistency pipeline.
2. Local vs. global insight: The authors convincingly show that global average confidence can mask local failures while sliding-window metrics like LGC and Tail Confidence capture these weak spots better.
3. Strong empirical results: Large-scale experiments (5 datasets, 4 models, 64× repeats) demonstrate consistent improvements in both accuracy and efficiency.
4. Comprehensive ablations: The authors systematically explore the effects of confidence thresholding, consensus level, and warm-up size, providing actionable insights for practitioners.
5. Presentation: The presentation, visualizations, and overall narrative are coherent and reader-friendly.

### Weaknesses
1. Confidence calibration: The authors treat percentile thresholds ($\eta = 10 / 90$) as hyperparameters, but the actual confidence scores are not calibrated across models. Reporting AUROC or ECE/Brier scores for trace-level discrimination would provide a more rigorous evaluation of confidence quality.
2. “Confidently wrong” cases: The authors briefly mention (Appendix D) that models sometimes assign high confidence to incorrect reasoning modes, but there’s no robust mitigation strategy. A hybrid ensemble or diversity-aware weighting could improve robustness.
3. Hyperparameters: The choice of top-$k$ = 20 and window = 2048 seems arbitrary. An additional ablation varying these values could offer deeper understanding of sensitivity.

### Questions
1. How does DeepConf compare to semantic entropy and self-certainty in online settings under the same token budget?
2. Can the authors provide quantitative measures (e.g., AUROC, ECE) showing how well their confidence metrics discriminate correct vs. incorrect traces?
3. How often do “confidently wrong” traces dominate the vote, and can local diversity or disagreement be incorporated as a second signal?
4. Can the same percentile thresholds $s$ learned during warm-up be reused across multiple problems from the same domain, or must they be recomputed for every new query?
5. Would varying the top-k (for confidence) or window size meaningfully affect results?

### Soundness
3

### Presentation
4

### Contribution
3
