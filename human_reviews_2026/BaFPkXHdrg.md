# Learning from Reference Answers: Versatile Language Model Alignment without Binary Human Preference Data

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 6, 4

## Abstract
Large language models~(LLMs) are expected to be helpful, harmless, and honest.
In different alignment scenarios, such as safety, confidence, and general preference alignment,
binary preference data collection and reward modeling are resource-intensive but play a central role in transferring human preferences.
In this work, we explore using the similarity between sampled generations and reference answers as a supplementary reward function for alignment.
When unary reference answers are available,
such similarity-based rewards can circumvent the need for binary preference data and explicit reward modeling.
We introduce \textit{RefAlign}, a versatile REINFORCE-style alignment algorithm that does not rely on reward or reference models.
RefAlign utilizes language generation evaluation metrics, such as BERTScore, between sampled generations and reference answers as surrogate rewards.
Beyond general preference optimization, 
RefAlign can be naturally extended to diverse scenarios, including safety and confidence alignment, by combining similarity-based rewards with task-specific objectives.
Across multiple scenarios, RefAlign achieves performance comparable to prior alignment methods while operating without binary preference data or reward models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces an alignment method that is an alternative to the most common way of performing alignment: using binary preference judgments from humans and training reward models on those judgments for use during RL. Instead of using binary judgments, it uses unary signals. Specifically, it uses the similarity (as measured by traditional similarity metrics such as BERTScore and METEOR) between generations and a reference answer as proxies for reward model signals. The authors find performance similar to existing methods using this approach.

### Strengths
1. To my knowledge, the use of existing text similarity metrics as a proxy reward signal is novel. While some methods like KTO (which the authors cite and compare against) don't use preference judgments, this method seems to be more efficient. 
2. The experiments are pretty thorough and clearly presented, with extensive comparisons with baselines. 
3. The comparisons with different similarity metrics, and especially the analysis with respect to factors such as response length, were very interesting and demonstrate the trade-offs of using different metrics.

### Weaknesses
1. My main concern is regarding the scalability of this approach. While it is certainly true that preference annotations are costly to obtain, the same is true for reference answers. High-quality references require much more validation, and that validation itself can be a lot trickier to actually perform. In comparison, relative judgments can be more robust "out-of-the-box." That being said, I think there is certainly an argument for using high-quality references as opposed to binary preferences in certain applications. For instance, I think this may be true for tasks requiring niche domain expertise. I think this work would be more compelling with a more targeted analysis for certain tasks where this is the case, as opposed to the more general helpfulness/harmfulness framework. 
2. There has been a lot of work showing the flaws of automatic metrics (e.g. [1]), especially when references are low-quality. To be clear, the authors do address the trade-offs and particular biases of using certain metrics (such as some metrics causing a bias towards longer responses). However, I think these flaws compound the issue described in the previous weakness. While reward hacking is always a concern even with the usual BT-based setup, I worry that this might be even more prevalent and subtle when using these flawed metrics. Again, with this, I think the paper would be stronger with a more targeted analysis over different domains.
3. While I do think the analysis of outputs is helpful in the paper, I think it could be expanded to analyze, for instance, the diversity of the resulting outputs after using RefAlign. I also think a small human validation/comparison experiment would be helpful. 

[1] Goyal, Tanya, Junyi Jessy Li, and Greg Durrett. "News summarization and evaluation in the era of gpt-3." arXiv preprint arXiv:2209.12356 (2022).

### Questions
1. Why is recall used for BERTScore instead of F1 (or even precision)?
2. For the ~30% of cases where BERTScore does not correspond to the RM, did you notice any patterns?
3. Is there a risk of inducing even more mode collapse when optimizing with respect to one reference? Was this something you noticed when analyzing the outputs after using RefAlign?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes RefAlign, a reinforcement learning (RL)-based alignment method for large language models (LLMs) that leverages similarity (BERTScore) between model-generated responses and high-quality reference answers as a surrogate reward function, eliminating the need for costly binary human preference data. RefAlign simplifies the alignment pipeline and can be effectively adapted for general human preference, safety, and confidence alignment scenarios. Empirical experiments demonstrate that RefAlign achieves comparable performance.

### Strengths
1. This paper proposes an approach that leverages similarity between generated responses and high-quality reference answers as a surrogate reward, thereby eliminating the dependency on binary human preference data.
2. The paper thoroughly validates its approach across various tasks and datasets.

### Weaknesses
Although the method proposed in this paper addresses the issue of reducing dependency on reward models, it introduces several limitations. First, the requirement of high-quality reference answers imposes constraints, as collecting these unary reference responses still demands significant human effort—albeit potentially less than binary labeling. More fundamentally, a key advantage of traditional reinforcement learning from human feedback (RLHF) is its capability to optimize models solely based on prompts without requiring pre-existing high-quality responses. In contrast, the algorithm presented here necessitates high-quality reference responses, making practical application somewhat limited. Lastly, while this method replaces the reward model with a text-similarity model, the actual reduction in computational and annotation costs appears minimal, as one form of labeling and model dependency is merely substituted for another.
When the quality of model-generated responses is actually higher than the reference answers, similarity-based scoring metrics can have counterproductive effects.

### Questions
See Weaknesses.

### Soundness
2

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
- RefAlign aligns language models to a provided reference answer using semantic similarity as the reward in a simple REINFORCE‑style update.
- It removes preference pairs and reward models, aiming for a compact pipeline with competitive results on safety alignment, confidence calibration, and general instruction benchmarks.
- The approach is practical and easy to implement, but performance depends on reference quality and careful length control.

### Strengths
- Originality: Unifies alignment around reference similarity and removes reward models and preference pairs.
- Quality: Strong results in safety and confidence alignment; competitive general alignment; reasonable ablations and comparisons.
- Clarity: Method and objective are easy to follow; training details are practical.
- Significance: Useful for bootstrapping smaller models from strong references, with lower data and engineering cost.

### Weaknesses
- Reference quality dependence: effectiveness hinges on high‑quality, diverse references; noisy or narrow sets can misalign models.
- Length bias and calibration: outputs tend to be longer; needs explicit length penalties or controls and more consistent calibration reporting (e.g., ECE).
- Judge consistency: results may vary by judge; add multi‑judge analysis and open‑source judge checks.
- Baseline breadth: include same‑compute, parallel comparisons to direct optimization baselines (DPO, SimPO, ORPO, KTO) with matched settings.
- Generalization: broaden evidence to more domains and harder tasks beyond the reported benchmarks.

### Questions
- How stable are results across different similarity metrics (e.g., BERTScore, BLEURT, cosine embeddings)?
- Do multiple references per prompt improve robustness, and how are conflicts resolved?
- What is the accuracy–length trade‑off under explicit length penalties or constraints, and how does calibration change?
- How do results differ when references are human‑written or curated versus model‑generated?
- Can you provide same‑compute, parallel comparisons to direct optimization baselines and report training stability?

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
3

### Summary
This paper proposes RefAlign, a REINFORCE-style algorithm for language model alignment. The method uses the similarity between model generations and unary reference answers, measured by metrics like BERTScore, as a surrogate reward function. This approach aims to achieve alignment in various scenarios without requiring binary preference data or a separately trained reward model.

### Strengths
+ The paper is well-written and clearly organized, making the proposed method easy to understand.
+ The experiments and corresponding analysis cover a wide range of tasks. And the baseline methods are thorough.
+ The proposed framework is versatile and can be adapted to different alignment tasks, including safety and confidence alignment, by modifying the reward function.

### Weaknesses
+ The main problem is the issue of unfair comparison in the experimental setup. The authors use external data from Llama-3.3-70B-Instruct to generate reference answers for training RefAlign. It is not clear whether the baseline methods also utilized this external data. My concern is that RefAlign's performance gains may largely benefit from this additional information gain, rather than the algorithmic novelty alone.
+ The core mechanism of RefAlign appears to be a reinforcement learning version of distillation. Using a BERTScore reward to match outputs from a more powerful model is conceptually very close to distillation, which the paper also compares against. As the reward model optimizes for text similarity, the proposed algorithm may be useful to a part of the community, but the contribution remains limited.

### Questions
To my understanding RefAlign is a reinforcement learning algorithm, why not include other RL baselines (GRPO, PPO, DAPO...) ?

### Soundness
2

### Presentation
3

### Contribution
2
