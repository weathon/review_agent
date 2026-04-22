# Efficient Reasoning with Balanced Thinking

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 6, 8, 10, 4

## Abstract
Large Reasoning Models (LRMs) have shown remarkable reasoning capabilities, yet they often suffer from overthinking, expending redundant computational steps on simple problems, or underthinking, failing to explore sufficient reasoning paths despite inherent capabilities. These issues lead to inefficiencies and potential inaccuracies, limiting practical deployment in resource-constrained settings. Existing methods to mitigate overthinking, such as suppressing reflective keywords or adjusting reasoning length, may inadvertently induce underthinking, compromising accuracy. Therefore, we propose \textsc{ReBalance}, a training-free framework that achieves efficient reasoning with balanced thinking. \textsc{ReBalance} leverages confidence as a continuous indicator of reasoning dynamics, identifying overthinking through high confidence variance and underthinking via consistent overconfidence. By aggregating hidden states from a small-scale dataset into reasoning mode prototypes, we compute a steering vector to guide LRMs’ reasoning trajectories. A dynamic control function modulates this vector’s strength and direction based on real-time confidence, pruning redundancy during overthinking, and promoting exploration during underthinking. Extensive experiments conducted on four models ranging from 0.5B to 32B, and across nine benchmarks in math reasoning, general question answering, and coding tasks demonstrate that \textsc{ReBalance} effectively reduces output redundancy while improving accuracy, offering a general, training-free, and plug-and-play strategy for efficient and robust LRM deployment. Code and models will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes REBALANCE, a training-free framework that dynamically balances overthinking and underthinking in reasoning models by steering hidden representations based on confidence and variance signals. The method constructs a steering vector from a small calibration set (e.g., 500 samples) and adaptively adjusts the model’s internal states during inference. Experiments across multiple reasoning domains show improved efficiency (i.e., reducing the number of reasoning tokens) and good accuracy without additional parameter updates.

### Strengths
**[S1]** Using confidence to address overthinking and underthinking is interesting. Although the overall framework is complex, the method works surprisingly well in practice, which makes the idea conceptually appealing.

**[S2]** The fact that the approach operates without any parameter updates and can be instantiated with only a few hundred samples (e.g., 500) is impressive. It suggests that the proposed mechanism captures a general behavioral signal of the model rather than relying on large-scale retraining.

**[S3]** The method is applied not only to mathematical reasoning tasks but also to multiple domains, which demonstrates a degree of robustness and general applicability beyond a single task family.

**[S4]** The paper is overall well written and clearly structured, with extensive empirical validation and thoughtful analysis. The presentation is careful, and the experiments are relatively thorough.

### Weaknesses
**[W1]** While Figure 5 presents a reasonable ablation on the number of samples used for mean and variance estimation, the transferability of these statistics remains questionable. The current analysis is limited to sampling from MATH and evaluating within the similar domain. It would be more convincing to test more rigorously (than the current analysis in Figure 5) whether the mean/variance computed from easier datasets (e.g., GSM8K) transfer to harder domains (e.g., AIME), or whether math-derived statistics generalize to different modalities such as code reasoning. Without this, the generalization claim remains somewhat weak.

**[W2]** The method feels overly complex due to the number of hyperparameters involved. Decisions such as which layer to select, how strongly to apply steering, or what window size to use are not clearly motivated. It remains unclear whether there exists a single configuration that works consistently across datasets or domains, or whether each model and domain effectively requires its own tuning. This undermines the claim of universality and adds to the engineering-heavy impression.

**[W3]** The paper is somewhat engineering-driven rather than conceptually driven. Most design choices (confidence thresholds, gating, control surfaces) are empirically fitted, and it is not clear why or when the method should work beyond the tested settings. Thus, the approach lacks principled understanding of why such steering in hidden space improves reasoning. The steering direction is defined by empirical prototypes, but the underlying mechanism or geometry of reasoning states remains speculative. From my experience with steering-based approaches, this line of work is still at a very early exploratory stage. The paper demonstrates that something “can” work, but not really why it works. While the results are decent, the contribution feels somewhat ad-hoc and might not generalize to models or tasks with different internal confidence landscapes.

**[Overall]** I don’t find a strong reason to reject this paper. However, it also doesn’t give strong conceptual insights; it reads as an engineering effort that happens to work well rather than a principled step forward in understanding reasoning control. I would lean toward a weak accept, mostly for its empirical completeness and potential to inspire future, more principled research in steering-based reasoning.

### Questions
I think it is good to add the exact throughput analysis and memory usage analysis for the efficiency claim.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a training-free solution that balances overthinking and underthinking in reasoning models.
Based on the key observation that confidence values and their variances help distinguish these two reasoning modes, the authors introduce a representation steering method that dynamically adjusts the model’s reasoning behavior.
Experimental results demonstrate that the proposed method improves performance while reducing reasoning length.

### Strengths
- The motivation is clear, and the addressed problem is crucial in the reasoning field.
- The paper is well-written and easy to follow.
- The experimental results, from initial observations to ablation studies, strongly support the proposed method.
- The method is simple, easy to implement, and can be plug-and-play across various reasoning models.

### Weaknesses
No major weaknesses are observed, but clarification on the following questions would strengthen the paper and may affect my final score.

### Questions
**[Q1] Extraction of the Steering Vector**

As I understand, the authors extract the steering vector using confidence-defined sets in Eq. (5).
However, directly using the definition in Eq. (3) appears simpler.
Could the authors report the performance gap between these two methods?

**[Q2] Observation Across Models**

What type of model is used in Figure 2 (b)?
I am curious whether confidence values and variances serve as general indicators across different models.

**[Q3] Inference Overheads**
Appendix I provides only a brief latency analysis.
Could the authors include a quantitative throughput evaluation (e.g., tokens per second) comparing their method with the original model?

**[Q4] Number of Samplings**
How many samplings are used per test sample?
Since challenge datasets such as AIME and AMC contain relatively few examples (< 100), multiple samplings may be required for more reliable evaluation.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
This paper addresses the issues of "Overthinking" (redundant computational steps) and "Underthinking" (insufficient reasoning paths) in Large Reasoning Models. The authors propose REBALANCE, a training-free framework that aims to achieve efficient and "balanced thinking." The core idea is to use model confidence and confidence variance as real-time indicators to dynamically steer the model's hidden states. The framework uses a pre-computed "steering vector," derived from reasoning mode prototypes extracted in an offline pass, to guide the model's reasoning trajectory. A dynamic control function modulates this vector's strength and direction, pruning redundancy during overthinking and promoting exploration during underthinking. Experimental results show that the proposed method outperforms existing baselines and achieve better efficiency.

### Strengths
1. Clear Optimization Objective: The paper defines Overthinking and Underthinking based on numerical features of internal states (stepwise confidence and confidence variance) derived from empirical observations. This semantic-agnostic definition, independent of specific keywords, provides a very clear and quantifiable optimization objective.

2. Extensive Experimentation: The paper provides comprehensive experiments and ablation studies. The REBALANCE framework is validated across multiple models of varying scales and tested on diverse domains beyond mathematics, demonstrating strong generalization. Furthermore, thorough ablation studies are conducted for key design choices, offering justifications for the method's components.

3. Demonstrated Effectiveness: The experimental results demonstrate that the proposed approach can successfully balance the trade-off between Overthinking and Underthinking. It outperforms previous baselines in terms of task performance and token efficiency.

### Weaknesses
1. Missing Citation to Key Related Work: The paper focuses significantly on the problem of "Underthinking". However, it fails to cite an important and highly relevant recent study on this specific phenomenon [1]. This omission is a significant gap in the related work section.
Reference:
[1] Wang Y, Liu Q, Xu J, et al. Thoughts are all over the place: On the underthinking of o1-like llms[J]. arXiv preprint arXiv:2501.18585, 2025.

2. Practical Application Burden: Although the method is presented as "training-free," it remains highly dependent on hyperparameter tuning. Furthermore, it requires an additional "seen dataset" and a model-specific calibration step (to extract steering vectors and fit the control function). This introduces an extra implementation and tuning burden for each new backbone model, which may complicate its practical application.

### Questions
1. The intervention method is specifically developed to target the paper's definitions of Overthinking (low-confidence/high-variance) and Underthinking (high-confidence/low-variance). However, could this direct manipulation of internal reasoning states have other unintended side effects on reasoning, beyond just mitigating these two defined issues? For instance, could it negatively impact creativity or the naturalness of the expressions?

2. While the quantitative results are strong, have the authors analyzed the semantic changes in the model's reasoning process post-intervention? For example, did the REBALANCE framework lead to a measurable change in the frequency of keywords associated with reflection or hesitation (e.g., "wait," "alternatively," "let me check")?

### Soundness
4

### Presentation
4

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
This paper addresses a key challenge in LRMs: the trade-off between overthinking and underthinking. The authors propose ReBalance, a training-free and plug-and-play framework that dynamically controls reasoning behavior via internal confidence signals. Experiments on 4 models and 9 reasoning benchmarks show that ReBalance reduces reasoning token length significantly while improving or maintaining accuracy.

### Strengths
1. An important problem that is of strong interest to the LRM community.

2. The paper uses stepwise confidence and confidence variance to detect reasoning modes, with a motivation that is well-explained and intuitively convincing.

3. The method achieves superior results across a wide range of reasoning benchmarks.

### Weaknesses
1. The main text lacks a concise introduction to the baselines, which makes the paper more difficult for readers to follow.

2. Novelty of this paper is limited. Steering vector has been used before. How does ReBalance differ from SEAL? Both of them leverage the steering vector. I want to see an apple-to-apple comparison. 

3. ReBalance is similar in spirit to adaptive halting or early-exit reasoning models, such as TrimR [1] and FlashThink [2]. I noticed that the authors cited these two papers, but neither compared them as baselines nor discussed them in the Related Work section of the appendix or other places.

[1] Weizhe Lin et al. TrimR: Verifier-based training-free thinking compression for efficient test-time scaling. arXiv preprint arXiv:2505.17155, 2025.

[2] Guochao Jiang et al. FlashThink: An early exit method for efficient reasoning. arXiv preprint arXiv:2505.13949, 2025. (edited)

### Questions
see the weakness above

### Soundness
3

### Presentation
2

### Contribution
3
