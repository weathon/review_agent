# Adaptive Margin RLHF via Preference over Preferences

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Margin-based optimization is fundamental to improving generalization and robustness in classification tasks. In the context of reward model learning from preferences within Reinforcement Learning from Human Feedback (RLHF), existing methods typically rely on no margins, fixed margins, or margins that are simplistic functions of preference ratings. However, such formulations often fail to account for the varying strengths of different preferences—i.e., some preferences are associated with larger margins between responses— or they rely on noisy margin information derived from preference ratings. In this work, we argue that modeling the strength of preferences can lead to better generalization and more faithful alignment with human intent. Furthermore, many existing methods that use adaptive margins assume access to accurate preference scores, which can be difficult for humans to provide reliably.
We propose a novel approach that leverages preferences over preferences—that is, annotations indicating which of two preferences reflects a stronger distinction. We use this ordinal signal to infer adaptive margins on a per-datapoint basis. We introduce an extension to Direct Preference Optimization (DPO), named DPO-PoP, that incorporates adaptive margins from preference-over-preference supervision, enabling improved discriminative and generative performance. Empirically, our method outperforms vanilla DPO, DPO with fixed margins, and DPO with ground-truth margins, on the UltraFeedback dataset. 
These results suggest that integrating preference-over-preference information—which requires less precision to be provided accurately—can improve discriminative and generative performance without adding significant complexity. Additionally, we show that there is a tradeoff between discriminative and generative performance: improving test classification accuracy, particularly by correctly labeling weaker preferences at the expense of stronger ones, can lead to a decline in generative quality. To navigate this tradeoff, we propose two sampling strategies to gather preference-over-preference labels: one favoring discriminative performance and one favoring generative performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces a modification to the DPO algorithm, which uses adaptive margins based on "preferences over preferences" - that is, we use the information from the reward model that tells us which preferences are stronger and which are weaker.
The experimental validation shows

### Strengths
The method is straight-forward and well justified. The paper is well-written and understandable. The paper includes a useful guide on selecting the appropriate sampling strategy.

### Weaknesses
The main limitation is the experimental evaluation of the generative ability. At the moment it relies entirely on synthetic evaluation

Furthermore, all results lack any sort of confidence intervals - this is a crucial omission in a scientific publication, as point estimates are not sufficient to estimate the robustness of a given result.

### Questions
I will increase my rating to at least 6 if proper confidence intervals are included.

Beyond that, I would appreciate seeing a more thorough experimental validation, as its current state does not convince me that this is something I should use in my own experiments.


Question: What is the computational overhead of this method, compared to the standard approach?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a adaptive margin variant of DPO method leveraging “preference-over-preference” (PoP) signals—i.e., which of two preference pairs shows a stronger distinction. The paper explores two different ways of constructing the PoP preference datasets, 1) iterative sampling that represents weak pairs with similar frequencies as original preference datasets; it boosts discriminative accuracy but can hurt generation; 2) randomly sampling where strong preferences naturally appear more; it shows both better overall rewardbench performance and better generation. In summary, the authors show that turning that ordinal signal into adaptive, per-example margins inside DPO (DPO-PoP objective) improves both rewardbench performance and real generation quality.

### Strengths
Originality: The paper provides a novel way to get adaptive margins for DPO training by leveraging “preference-over-preference” (PoP) signals. 

Quality: The paper provides detailed and easy to understand theory derivation. The paper also provides quite thorough experiments with necessary baselines. 

Clarity: The figures and tables are otherwise easy to read except for the ones mentioned in weakness.

Significance: The paper provides a solution to incorporate preference margin information into DPO training leveraging the existing preference datasets - no need to create new prompt/responses trios.

### Weaknesses
1. Overall weakness: In the abstract part, the authors indicate that a big part of their motivation is "many existing methods that use adaptive margins assume access to accurate preference scores, which can be difficult for humans to provide reliably." However, the experiments in the whole paper uses Ultrafeedback, a dataset with exact score rating. The authors do claim that they use the ground truth margin as a proxy since using annotator is expensive, but I think it is necessary to do if this is a core advantage that the authors claim to have for their method. Using LLM-as-a-judge to rate the PoP could be a cheaper alternative for example. At least the authors can show what a realistic intended usage case of their method would look like (they intend people to use it when there is no ground truth score rating). 
2. Figure 2: The lines are really close but not exactly overlapping. I would like to be able to compare them but the current presentation makes it hard to read. Perhaps the authors can expand the y axes around that range?
3. Section 4.4: Seeing win rate is nice but it's hard to make sense of what the number means without looking at some generation samples.

### Questions
1. Regarding weakness 1 mentioned. I think it might be quite infeasible to repeat the experiments with human (or LLM) labeling as opposed to what's used in the paper (actual score margin), but I think it's quite necessary to show if the authors want to claim the strength of the method is no need of accurate preference scores. If the authors can provide a convincing claim to justify the design choice or modify the claim I could consider raising the score. 
2. See weakness 2
3. Regarding weakness 3 mentioned: it would be helpful if the authors could provide generation case studies on each representative models. 
4. "Interestingly, even though DPO-margin-gt has access to the actual ground-truth margin values, it performs worse in terms of test classification accuracy." I appreciate the comparison with the ground truth margin baseline. This is an interesting observation, can authors provide any intuition on why this could happen? 
5. What kind of samples are in Chat Hard? Can authors provide any intuition on why both methods drop considerably on this rewardbench subset?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a new method to incorporate a margin into the RLHF objective by introducing a novel data collection strategy "Preference-over-Preference". Instead of standard pairwise comparisons, this method asks labelers to compare two pairs of responses to determine which pair exhibits a more significant difference. The authors demonstrate that this approach, when integrated with DPO, is effective compared to vanilla DPO and DPO with other margin-based baselines. The paper also explores two sampling strategies for the PoP dataset, revealing an important trade-off between the model's discriminative and generative performance.

### Strengths
- The proposed "Preference-over-Preference" data collection and objective are novel..
- The experimental evaluation is comprehensive, analyzing various aspects of model performance.

### Weaknesses
1. Reliability of the Evaluator (UltraRM): The trustworthiness of UltraRM as the primary evaluator is not sufficiently established. UltraRM is trained on GPT-4 annotations (UltraFeedback) and it is not clear how strong this judge is. The paper should include quantitative metrics on the UltraFeedback test set, such as the correlation or agreement rate between UltraRM's predictions and the original GPT-4 annotations. This is crucial for validating UltraRM as a reliable evaluation tool. It might be good to directly use GPT-4 as the judge, since the dataset is annotated by GPT-4.
2. Lack of Optimization Dynamics and KL Analysis: The paper provides no results of model behaviors during the optimization process. Incorporating a margin can sometimes introduce optimization instability, yet no training curves (e.g., win rate or accuracy vs. training steps) are provided. Crucially, the KL divergence from the reference policy during training is not reported. It is difficult to assess the results fairly, as a model's superior accuracy might simply be a consequence of a much higher KL divergence (i.e., straying further from the reference policy). The authors should provide plots (similar to Figure 1 in Tang et al., 2024) showing the evolution of win rate, accuracy, and KL divergence throughout the training process.
3. Baseline Comparisons: The DPO-PoP-random method employs datasets sampled based on the ground-truth margin. This sampling strategy may implicitly provide the PoP models with additional information, as the resampling alters the distribution of margins in the training set. As an ablation, one more baseline DPO should be trained on the same set of preferred pairs derived from the PoP-random dataset (effectively treating 0 as the margin).

[1] Tang et al 24: arXiv:2503.19618

### Questions
1. Please clarify the dataset used for calculating the results in Table 1. Is it the UltraFeedback test set?
2. Appendix E shows that different values of $k$ affect the trade-off between discriminative and generative performance. This appears to be a key hyperparameter. Which value of $k$ was used to generate the main results in Tables 1 and 2? This information is critical and should be included in the main paper. Was the same $k$ used for all experiments reported in these tables?
3. Line 438 states that the method leads to "significant gains" for “when the size of the preference dataset is small” Please clarify what evidence you have to justify this claim.
4. Regarding the label noise analysis in Appendix E:
    - The current method simulates noise by randomly flipping labels with a constant probability. A more naturalistic noise model might be to sample the PoP using a Bradley-Terry (BT) model, where a temperature-like constant could control the noise level. 
    - In Figure 3, could you add the case for label noise level $p=0$ (i.e., no noise)?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes DPO-PoP, an extension of Direct Preference Optimization (DPO) that introduces adaptive margins based on preferences over preferences (PoP): annotations indicating which preference pair expresses a stronger preference contrast. Instead of relying on scalar human or model scores, DPO-PoP infers relative margin strengths through ordinal supervision. The approach integrates this information into the DPO loss using stability enhancements such as margin clipping and Polyak-averaged targets. Experiments on UltraFeedback, RewardBench, and AlpacaEval 2.0 show that DPO-PoP variants outperform standard and fixed-margin DPO baselines. The paper also reports an interesting tradeoff: improving discriminative (classification) performance may reduce generative quality, and vice versa.

### Strengths
1.	Simple yet coherent extension. The proposed method integrates adaptive margins into DPO cleanly, requiring no extra reward model or RL loop.
2.	Comprehensive empirical evaluation. The experiments are extensive, covering multiple model scales, datasets, and sensitivity factors (sampling, scale, label noise).
3.	Practical annotation framing. The use of ordinal comparisons rather than numeric ratings is a plausible direction for more robust human feedback.
4.	Clear empirical takeaway. The study highlights a consistent tradeoff between discriminative and generative alignment, which is valuable for practical RLHF tuning.
5.	Readable and reproducible. Code availability, training details, and hyperparameters are well documented.

### Weaknesses
1.	Limited empirical gains. The improvements over baselines are modest and might not justify the additional supervision and implementation complexity.
2.	Synthetic validation. All PoP data are simulated from existing scalar scores; thus, the central claim that human PoP labels are easier or more reliable is not empirically validated.
3.	Incremental novelty. The technique mainly reuses known margin-scaling principles, framed as "preference over preference." The conceptual jump beyond prior margin-based DPO methods is small.
4.	Lack of theoretical insight. The paper would be more robust if they can explain why PoP improves discriminative calibration or how the discriminative–generative tradeoff arises.
5.	Scalability of data collection. If PoP labels were collected directly, the quadratic comparison cost could make large-scale annotation expensive unless approximate or active-sampling methods are introduced.

### Questions
1.	Have you tested DPO-PoP with actual human PoP annotations rather than synthetic scores?
2.	Can you formalize or visualize the observed discriminative–generative tradeoff? What governs the balance between the two?
3.	How scalable is PoP labeling in real-world RLHF datasets with millions of preferences? Could adaptive sampling mitigate the quadratic cost?
4.	The observed improvements are relatively small—what concrete aspects of model behavior improve qualitatively under DPO-PoP?

### Soundness
3

### Presentation
3

### Contribution
3
