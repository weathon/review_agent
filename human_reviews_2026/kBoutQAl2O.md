# Beyond Accuracy: Measuring Reward Variance as a Predictive Benchmark for RLHF

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Reward models (RMs) provide the core signal in reinforcement learning from human feedback (RLHF). However, most evaluations focus on pairwise accuracy and overlook how the separability and concentration of reward distributions shape the optimization landscape and convergence rate. To address this limitation, we present the Reward Variance Benchmark (RVB), an evaluation suite that quantifies distributional properties of RM scores. RVB introduces three variance-oriented metrics that capture complementary aspects of an RM’s signal: score distribution concentration, global pairwise separation, and cross-prompt decision-style stability. We evaluate 23 widely used RMs with a toolkit that supports reproducible analysis. On this benchmark, the variance metrics yield stable rankings and, together with accuracy, show preliminary links to downstream convergence behavior in a small-scale RLHF case study. These findings support a key insight: in addition to judging responses correctly, an effective RM in RLHF should also score them with sufficient clarity and stability to provide actionable gradients. Overall, RVB provides a first variance-centric predictive benchmark for analyzing RLHF convergence under a fixed setup, while also supporting more open-ended studies of how reward variance interacts with calibration and policy dependence.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper argues that pairwise accuracy, a dominant metric for evaluating reward models (RMs), misses an important dimension: variance in reward scores. Low-variance reward functions create flat optimization landscapes that can slow policy learning. To address this, the authors propose the Reward Variance Benchmark (RVB),  a suite of variance-aware metrics for assessing RMs’ “teaching effectiveness.”

### Strengths
- Evaluating reward models is a critical challenge in RLHF and preference-based RL. Poor evaluation can lead to slow learning or unsafe learned behaviors. By focusing on metrics beyond pairwise accuracy, the paper tackles an important open problem.
- Reward model evaluations focused on reward variance seem to be novel and underexplored in evaluation frameworks. As prior work has found that reward variance affects the optimization landscape, evaluting reward models based on this woud be useful. Framing variance as a “teaching signal” provides a new perspective on reward model utility.
- Comparison across 23 models covering multiple families (LLaMA, Qwen, Gemma, Mistral, etc.) adds breadth.

### Weaknesses
**Missing related work:**

The paper does not discuss prior reward model evaluation frameworks, such as TAC [1], DARD [2], and EPIC [3], which evaluate reward functions beyond pairwise accuracy.

**Eval-Core benchmark:**

- The authors describe refining an existing benchmark (RMB 1.9k prompts to 354 prompts) to create Eval-Core. It is unclear why this reduction is useful or what statistical or practical advantage it provides. The contribution seems incremental, as it largely repurposes a preexisting dataset.

- The benchmark focuses on “helpfulness,” not on “harmlessness”. Why was this set chosen? Extending RVB to this additional set would strengthen claims.

**Metrics:** 
- The paper claims the proposed metrics (SEI, nGMD, and DCI) provide interpretability, but it is unclear how they do so or how they convey “clear optimization semantics.”

-  Razin et al (2025) [4] note that the same reward model can induce high reward variance for one language model, yet low reward variance for another. Therefore, different language models can require different reward model teachers. How do these metrics handle that? Will this be an issue for these metrics? Can these metrics be high for a reward model with respect to one language model, but low for the same reward model with respect to a different language model?


**Experimental Design Choices + Claims:**

- The paper states that three representative RM families were scored, but it is unclear what makes these families “representative.” More justification is needed.

- The claim that RVB metrics predict policy performance is weakly supported, as only three reward models were trained. It is difficult to infer a meaningful correlation from such a small sample.

- Why not use standard reward variance as a baseline metric? As included in Razin et al (2025) [4], which the authors do discuss. 

**Resources**

- TAC [1]: https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_280.pdf
- DARD [2]: https://openreview.net/pdf?id=CALFyKVs87
- EPIC [3]: https://openreview.net/pdf?id=LwEQnp6CYev
- Razin et al (2025) [4] https://arxiv.org/pdf/2503.15477

### Questions
- Are there examples of reward models where the RVB score better predicted policy performance than another metric (e.g., RewardBench Score)? For instance, is there an example of a reward model where another metric incorrectly had a high score, but RVB had a low score, and policy performance was indeed worse?
- Can the authors explain how the 23 reward models were chosen? 
- Can the authors further elaborate on how the SEI, nGMD, and DCI provide interpretability?
- Can the authors explain how the proposed metrics are better than standard reward variance with normalization that is done in Razin et al (2025)?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents the reward variance benchmark. It is designed as an evaluation benchmark to measure not only the pairwise accuracy, but also the quality of predictions via three variance-derived metrics. They use a custom-constructed benchmark of prompts and responses to evaluate 23 existing reward models. 

For me, the paper is a case of "less would have been more". I like the research problem; the paper shows some interesting ideas. However, I am not convinced by the formulation of metrics; they feel overly complex, and I am skeptical of the positioning of this work as a benchmark. I don't find that the reported results in terms of downstream performance prove a strong predictive capacity of the derived metrics. Instead, I would have preferred framing the paper as an empirical investigation of the relationship between reward variance and performance (which the dataset can be a precursor for). This would allow for avoiding complex wrapped metrics and a more open-ended exploration of the relationship. I think this could provide more insightful results and discussions.

### Strengths
+ Evaluation benchmarks in general are a great resource for the community 
+ Investigating and evaluating contributing factors to training success in RLHF is a relevant topic
+ I really appreciate an empirical investigation of the reward variance hypothesis, and its effect on downstream performance, and think this is a big strength of the paper
+ For most parts, an adequate level of detail for reproducibility
+ The developed RVB benchmark shows an ability to predict downstream RL performance
+ A series of interesting experiments, and good reporting of results

### Weaknesses
- The dataset construction process in section 3.1. is not really motivated or evaluated. Why choose this subset of the RMB benchmark? Why use the four temperature settings, and what difference did it actually make? I think there is a description of Appendix A3.4, but it seems disconnected from Eval-Core. I feel that providing the template (A.5) is insufficient. E.g., “we keep the RMB candidates (about 3-6)”. How were they selected?
- How were the five candidate metrics (Section 4) designed? What was the process? Was it based on related work? They come a bit out of the blue. I get the overall idea for each, but it contributes to some issues I have:
- I find the introduction of new terms for the metrics a bit cumbersome, as this requires memorizing these new terms and hides the relationship to existing concepts. I know that keeping metrics within the range [0-1] is attractive, but it adds complexity: For example, instead of the SEI, directly reporting H(p) would also be comparable across models, and “prediction entropy” would be more comprehensible than SEI (“lower is better” metrics are totally valid in my opinion). I think this relates to my question 1, where I feel that these metrics are potentially interesting to investigate, but not something I want to necessarily optimize for
- Similarly, for DCI, which is another custom metric, in my mind, not well motivated, as an aggregation metric
- Finally, this is even exacerbated with the composite score in 4.5. At this point, there are so many layers of metrics and normalizations that I find the composite score somewhat incomprehensible except for a vague “higher is better”. I feel that an evaluation benchmark should try to make an effort for generalizable and comprehensible metrics
- The results in Figure 4 also coincide with accuracy (the RewardBench score of Tulu is noticeably lower than the other two, which seems to be reflected in the training success). The Skywork model seems to converge even faster, although the composite metric is lower than the URM one. So these results do not fully indicate the predictive performance of the metrics compared to a simple metric like accuracy (you report a relatively high correlation of 0.51). 

Minor:
- I feel line 170 with the listed reward models is missing references for the different families of RMs
- “Wikipedia contributions” is an unusual citation; I would prefer a permanent document as a reference.
- Figure 3 is difficult to read. I would advise filtering some categories, introducing some highlighting, or choosing a different type of visualization

### Questions
- I am somewhat skeptical of the central motivating hypothesis: Ideally, I want the output distribution of my reward model (or any model for that matter) to reflect an underlying (aleatoric) uncertainty. So while a “sharper” model indeed leads to faster learning speeds, I do not find it obvious that this is more representative of the actual prediction target (i.e., it is well-calibrated). So I am not convinced that a low variance reward model is a metric that should be directly optimized for. A simple way to optimize for this benchmark seems to be to apply an entropy-penalization term during RM training, but I am unsure if this would result in a “better” reward model. So, while the general hypothesis of RM variance as an important factor is supported by peer-reviewed related work, I am somewhat skeptical whether benchmarking RMs for low variance is a desired quality per se.
- Have you compared the variance metrics to using the Brier score (I guess you call it the top-gap metric)? Have you run experiments for comparison? I would be really interested in also seeing the predictive capacity of top-gap metrics for downstream performance to better motivate the need for these new metrics
- Don’t the results of Chen et al. and Leng et al. point to the risk of overconfidence and even show that models with lower accuracy might lead to better learning? Isn’t low variance a potential sign of overconfidence?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a framework for evaluating reward models (RMs) used in reinforcement learning from human feedback (RLHF), arguing that pairwise accuracy alone fails to capture how effective an RM is in providing signals to policies. The authors introduce the Reward Variance Benchmark (RVB), which measures distributional characteristics of RMs through three metrics: SEI (Softmax–Entropy Index) for score concentration, nGMD (normalized Gini Mean Difference) for global pairwise separation, and DCI (Decision Consistency Index) for stability across prompts.

### Strengths
* The paper suggests a better evaluation framework for reward models, which is an important problem crucial to developing robust reward models.

### Weaknesses
* Experiments in verification (Section 5.4) are quite shallow. Only three policies fine-tuned with RMs are compared with each other. Due to this, it is hard to believe that the analyses provided in the previous sections 5.2 & 5.3 are a decisive factor in RM performance in teaching policies.
* The 4 'teaching styles' provided in section 5.2 is not backed by empirical results. To really see if the teaching styles do produce any effect to the fine-tuned policy, some distinctive pattern for each of these fine-tuned policies from different groups of RMs should have been observed
* Figure 3 lacks providing useful insights other than the fact that there is variance between task categories for different types of reward models. 
* Using GPT-4o might benefit certain RMs that are used to the specific generation style of the model. Comparison with other models would be beneficial.
* Minor corrections
  * In page 4, MAD, the citation is pointing to Wikipedia contributors, which is not a proper citation. Please update this with statistics textbooks or relevant papers.

### Questions
* What is the motivation to exclude the 2 metrics from the initial 5 metrics?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper identifies a critical gap in current reward model (RM) evaluation for reinforcement learning from human feedback (RLHF)—namely, the lack of systematic attention to the variance and distributional properties of reward signals. The authors propose the Reward Variance Benchmark (RVB), a comprehensive evaluation suite that introduces three variance-sensitive metrics (SEI, nGMD, DCI) to profile RMs along axes of score concentration, pairwise separation, and cross-prompt stability. Through extensive analysis of 23 popular RMs, with supportive experiments and visualizations, the RVB suite is shown to predict downstream RLHF convergence and select RMs more effectively than accuracy alone. The paper is accompanied by a standardized benchmark data release and reproducible tools.

### Strengths
1 Clear Motivation and Relevance: The paper effectively justifies why accuracy alone is insufficient for RM evaluation, referencing empirical and theoretical findings (Section 2, references to Chen et al., 2024; Razin et al., 2025). The flatness of reward landscapes under low variance, as depicted in Figure 1, directly motivates the shift in perspective.

2 Metric Suite Design: The introduction of the SEI (Softmax-Entropy Index), nGMD (normalized Gini Mean Difference), and DCI (Decision Consistency Index) is mathematically well-founded. The metrics are robust (using median, MAD), interpretable, and explicitly decoupled from accuracy (Section 4).

3 Empirical Validation: The RVB metrics are validated against RLHF convergence rates using multiple models (Figure 4), and variance-based rankings provide new insights beyond accuracy rankings (Table 1).

### Weaknesses
1 Overlapping Metrics & Composite Score: While the correlation analysis in Appendix A.2.3 mitigates concerns, there is still notable overlap between SEI and nGap (correlation $\rho \approx 0.78$) and partial redundancy with nGMD. The choice of metric aggregation (equal weighting of MAD-z scores in the composite) is somewhat heuristic (Section 4.5). The impact of alternative weighting or selection criteria for the composite is not fully explored, raising the possibility of overfitting to the presented evaluation set.

2 Empirical Baselines: While 23 RMs are evaluated, there is limited discussion of calibration or strong baselines from variance-aware reward modeling (e.g., DGRO, GRPOVI, GVPO), or ablation against RM ensembles that explicitly regularize variance. This omission weakens claims about RVB's broad applicability.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
