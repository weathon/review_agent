# Teaching LLMs to Abstain via Fine-Grained Semantic Confidence Reward

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Mitigating hallucinations in Large Language Models (LLMs) is critical for their reliable deployment. Existing methods typically fine-tune LLMs to abstain from answering questions beyond their knowledge scope. However, these methods often rely on coarse-grained signals to guide LLMs to abstain, such as overall confidence or uncertainty scores on multiple sampled answers, which may result in an imprecise awareness of the model's own knowledge boundaries. To this end, we propose a novel reinforcement learning framework built on Fine-Grained Semantic Confidence Reward (FSCR), which guides LLMs to abstain via sample-specific confidence. Specifically, our method operates by sampling multiple candidate answers and conducting semantic clustering, then training the LLM to retain answers within high-confidence clusters and discard those within low-confidence ones, thereby promoting accurate post-hoc abstention. Additionally, we propose a comprehensive metric for evaluating the reliability of abstention fine-tuning. Experimental results demonstrate that our method significantly enhances reliability in both in-domain and out-of-distribution benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Authors present a novel method for teaching LLMs to abstain from answering queries when they are uncertain. Their method is based on training an LLM via the following process:

1) for a given query, sample multiple responses
2) cluster these responses
3) assign "high/low confidence" labels to said clusters based on the cluster's size
4) the model is rewarded if its self-estimated high/low confidence aligns with the cluster-based label

Their method shows strong generalization to OOD data, which baselines do not.

Authors additionally introduce a novel reliability metric that combines helpfulness and truthfulness together.

### Strengths
The paper's writing is clear, the method and novel metric are explained effectively.

The metric seems to be effectively justified (Appendix C.1).

The OOD generalization is very compelling. 

Ablations are well-done.

### Weaknesses
I'm not sure that the method is motivated effectively enough (that we want a per-sample abstention boundary not a global one per-query). I understand the concept, but what exactly happens e.g. at test time? The clustering step etc. does not happen, correct? So what are we really gaining when compared to existing methods that use binary labels based on correct/incorrect answers or a single entropy-based reward per question?

Can we not just use this method of clustering at inference time, no training needed?

How exactly is your signal better than competing signals? This feels hand-wavy. What is the exact mechanism? It's very unclear to me.

Are there some confounding variables here w.r.t. training methods? Some competing methods are pure supervised fine-tuning vs. others using RL. 

The distinction between GRPO-SE and FISCORE is very hazy to me. GRPO-SE already does clustering, no? So it's multiple rewards per question (one for each response) vs. 1 aggregated reward per question, is essentially what we're comparing? Is it the clustering specifically that's helping, or is it the fine-grained per-sample rewards/denser/richer reward signal that's helping?

### Questions
Why don't you try just using continuous rewards based on the cluster size proportion vs. the total, vs. a hard boundary?

Doing a causal intervention at inference time would really strengthen the mechanistic explanation of what's going on here. Manipulate the answer and measure confidence change:

1. Generate answer A from model
2. Manually replace A with answer B (from a different cluster during training)
3. Force model to generate confidence for B
4. Does confidence appropriately change based on what cluster B would belong to?

This seems like it would effectively test if the model internalized a representation of the clusters/of its own internal answer distribution frequency.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a novel and well-executed approach to a critical problem in LLM reliability: teaching models to abstain from answering questions beyond their knowledge scope. The proposed method, FISCORE, leverages reinforcement learning with a fine-grained, per-sample confidence reward derived from semantic clustering. It also analyzes the shortcomings of existing metrics and proposes a new metric, F1_rel, to capture the trade-off between helpfulness and truthfulness.

### Strengths
The fine-grained reward moves beyond coarse, aggregated uncertainty metrics, further optimizing the model's abstention behavior. The F1_rel proposed by the authors reflects a pursuit of balance between helpfulness and truthfulness, and their discussion on the effectiveness of F1_rel compared to existing metrics also provides valuable insight. The experimental section is thorough and well-designed, comparing the method to strong baselines and verifying its effectiveness, especially its generalizability. Besides, the paper is very well-written with a clear and logical structure.

### Weaknesses
1. The effectiveness and necessity of the confidence reward are questionable. (a) Effectiveness: The confidence reward essentially guides the model to be more confident and consistent in its output. This might incorrectly encourage hallucinations. For instance, given a complex question the model cannot answer, it might obtain multiple different incorrect answers across samples. The confidence reward could end up rewarding the most frequent of these incorrect outputs, thereby causing the model to be even more stubborn in its hallucinations. This phenomenon of potential reward hacking from such a confidence reward has already been discussed in [1]. (b) Necessity: Judging from the experimental results in Section 4.3, Figure 3(a), the necessity of the confidence reward seems weak. Even when the accuracy reward weight is increased to 4, at which point the confidence reward's weight is comparatively low and the overall reward largely degenerates into one focused only on accuracy, the model still achieves a high F_rel. Does this imply that the performance improvement stems from training with the accuracy reward rather than the confidence reward? Perhaps merely using the accuracy reward while prompting the model to generate a sure/unsure label (without factoring it into the reward) could also yield a model with a decent F_rel.
2. Limitations of the evaluation metric: The F1_rel metric favors a balanced model (where F1_ans and F1_abs are optimized simultaneously and are relatively close), as indicated by the data in Table 1. From the response distribution in Figure 4, one can also see that FISCORE actually generates more incorrect answers (N2+N4), implying poorer reliability. Concurrently, SE-Tuning generates fewer incorrect answers but performs more unnecessary abstentions (N3), which brings higher truthfulness and lower helpfulness, manifesting as a lower overall F1_rel. Thus, it seems the difference in F1_rel between methods depends more on the model's answering rate, that is, its trade-off preference between truthfulness and helpfulness. A model preferring balance is more likely to obtain a high F1_rel. This is inflexible. Many low-error-tolerance scenarios might prefer a model like R-Tuning, which sacrifices helpfulness but makes very few errors. Perhaps a hyperparameter could be introduced to adjust the weights of F1_ans and F1_abs to evaluate performance in different scenarios?
3. Insufficient ablation studies, including: (1) The use of a fixed abstention threshold of G/2: Intuitively, the value of the abstention threshold should be correlated with task difficulty. For a more difficult task, perhaps a lower threshold is required? Has the impact of the threshold's value on training been investigated? (2) Performance and impact of the semantic model: What is the performance of the currently used un-trained DeBERTa? Are there instances of inaccurate clustering? Perhaps some training would achieve better performance?

If my questions are resolved, I will consider raising the score.

References

[1] https://arxiv.org/abs/2505.21444

### Questions
1. Could you show a comparison of the rejection rates for different baselines on each dataset?
2. What is the importance of the format reward? I noticed you set a relatively high weight for the format reward (w_f=2) but provided no explanation. Was this the best value found in your experiments? Does the format reward have a significant impact on training?
3. Suggestion: The three weight values for R_total in equation (9) are important pieces of information. Consider stating them near equation (9) or in the implementation details. It took me some time to find their corresponding values in the appendix.

### Soundness
3

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
This paper proposes a reinforcement learning (RL) framework, FISCoR (Fine-grained Semantic Confidence Reward), designed to mitigate LLM hallucinations by teaching the model to abstain from providing specific, incorrect factual claims within an otherwise correct response. The core innovation lies in replacing traditional coarse-grained reward signals (like overall answer correctness or sampling entropy) with a reward based on explicit, fine-grained semantic confidence tags generated by the model itself (e.g., tagging individual clauses as "sure" or "unsure"). This approach aims to align the model's internal knowledge boundaries with its external generation behavior more precisely than previous methods.

### Strengths
1. The primary strength is moving from sentence- or response-level uncertainty to semantic chunk-level confidence. This allows the model to differentiate between known and unknown information within a single output, promoting more nuanced abstention

2. The method operationalizes the metacognitive ability (self-assessed confidence) into a quantifiable reward signal for RL, offering a direct path to behavioral modification.

### Weaknesses
1. The approach relies on the model simultaneously generating the answer and its confidence. As evidenced by recent work on metacognitive decoupling (e.g., the Answer-Free Confidence Estimation (AFCE) framework[1]), eliciting the answer and confidence simultaneously can introduce a strong cognitive bias, leading to overconfidence. If the confidence signal itself is biased, the resulting RL reward (and thus the trained policy) will be flawed.

2. The "fine-grained" semantic confidence reward is fundamentally a discrete, binary, or limited-scale estimation (e.g., "sure" vs. "unsure"). This is a coarse quantization of the true latent uncertainty (which is better represented by continuous scores or sampling entropy/divergence, akin to ROC-AUC metrics[2] applied to model output probabilities). This discretization may discard valuable information and introduce unnecessary bias.

3. The paper proposes an integrated metric that equally balances helpfulness (answering) and truthfulness/reliability (abstention). This fixed, symmetric trade-off makes the metric difficult to interpret or optimize for real-world scenarios where the cost of error vs. the value of a correct answer is highly asymmetric (e.g., a search engine favors helpfulness, while a medical diagnosis system favors reliability). The metric should allow for scenario-specific weighting.

### Questions
1. Given that simultaneous elicitation of answers and confidence is known to induce overconfidence bias (as argued in prior work on decoupling cognition from metacognition), what experiments were conducted to verify the calibration of the fine-grained confidence tags before using them as the reward signal? Did you consider a decoupled, two-stage prompting method (like AFCE) to generate a less biased confidence reward?

2. The paper critiques coarse-grained signals but utilizes a discrete (quantized) confidence label. What is the performance trade-off between this discrete semantic label and existing continuous, latent uncertainty measures (e.g., sampling consistency, variance, or token log-probabilities) that have been shown to correlate strongly with correctness ?

3. The methodology of generating a binary tag (sure/unsure or similar) alongside the answer has precedents in instruction-tuning for abstention[3]. Please include in the paper a discussion and comparison with related work that uses simple sure/unsure tags for finetuning.

[1]https://arxiv.org/pdf/2506.00582

[2]https://arxiv.org/pdf/2305.14613

[3]https://arxiv.org/pdf/2503.02233

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper employs a reinforcement learning approach to teach the model to abstain. Answers are first clustered based on semantic similarity, and the model is deemed confident enough to respond (receiving a reward of 1) only when a cluster reaches a sufficient size. Experiments demonstrate the effectiveness of this method.

### Strengths
* The motivation is clear and the problem is significant.
* The writing is clear and I can understand this work.

### Weaknesses
* A key baseline is missing: TTRL [1], a reward design based on majority vote. Essentially, the method proposed in this paper, clustering answers semantically and determining confidence based on the number of answers in each cluster, is fundamentally an extension of majority voting. In other words, answers that appear more frequently across repeated samples are assigned higher confidence.
* The exploration of reward combinations is insufficient. Some studies have shown that omitting the format reward may lead to better performance. Have you tried this approach?
* Compared to not considering abstention (i.e., $w_c=0$), how does the task performance (accuracy) change?

[1] https://arxiv.org/pdf/2504.16084

### Questions
* Is it appropriate to measure uncertainty directly using the likelihood of each answer?

### Soundness
2

### Presentation
3

### Contribution
2
