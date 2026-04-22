# The Coverage Principle: How Pre-Training Enables Post-Training

- Avg Score: 7.33
- Decision: Accept (Oral)
- Scores: 8, 6, 8

## Abstract
Language models demonstrate remarkable abilities when pre-trained on large text corpora and fine-tuned for specific tasks, but how and why pre-training shapes the success of the final model remains poorly understood. Notably, although pre-training success is often quantified by cross entropy loss, cross entropy can be poorly predictive of downstream performance. Instead, we provide a theoretical perspective on this relationship through the lens of coverage, which quantifies the probability mass the pre-trained model places on high-quality responses and which is necessary and sufficient for post-training and test-time scaling methods like Best-of-N to succeed. Our main results develop an understanding of the coverage principle, a phenomenon whereby next-token prediction implicitly optimizes toward a model with good coverage. In particular, we uncover a mechanism that explains the power of coverage in predicting downstream performance: coverage generalizes faster than cross entropy, avoiding spurious dependence on problem dependent parameters such as the sequence length. We also study practical algorithmic interventions with provable benefits for improving coverage, including (i) model/checkpoint selection procedures, (ii) gradient normalization schemes, and (iii) test-time decoding strategies.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this paper authors introduce a new quantity called coverage profile which is better aligned with a model's performance on downstream tasks that require Best-of-N sampling and/or RL. They show that autoregressive next-token prediction implicitly optimizes coverage. Coverage has better generalization and convergence than CE. While CE can continuously improve during pre-training, coverage can drop at some point which may reflect the decrease in the BoN downstream performance.
The authors perform experiments with graph reasoning tasks to support their theoretical results. They present a way to estimate coverage profile in practice by defining a Tournament procedure to select checkpoints with the best in-class coverage profile. Moreover, they show that test-time training (TT token-level SGD) improves coverage.

### Strengths
- The theoretical justification of the coverage principle that can explain insufficiency of cross-entropy for accurately predicting downstream BoN performance is an important contribution.
- The paper is well-written and solid. Theoretical results are sound.
- Algorithmic improvements like gradient normalization in SGD can lead to better coverage and therefore better BoN performance.
- The work opens a new interesting direction in connection of the base model and post-training performance by utilizing the notion of coverage profile.

### Weaknesses
- The only concern that I have is that the authors validated their theory on only one task - graph reasoning.

### Questions
- Can the scaling law (eq. 3 and therefore Proposition3.1) be validated empirically (e.g. using the tournament procedure)? 
-  Figure 1, why coverage has non-integer index (e.g. $Cov_{5.7}$), if $N$ is the number of sampling attempts (eq. 1) or does $N$ mean something else here?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
Motivated by the observation that downstream results are not always correlated with cross entropy results, this paper introduces coverage that is better related to downstream performance. The authors show through theoretical results that next-token prediction leads to models with good coverage. From a practitioner's perspective, the authors then demonstrate a decoding intervention to improve coverage, and a way to use coverage for selecting models.

### Strengths
- well motivated and tied to prior work, figure 1 makes this clear
- theory is sound and directly relates to BoN and post-trained models
- also contains practical recommendations (section 6)

### Weaknesses
- lack of experimental evidence. would be nice to expand on the downstream tasks and post-training methods.
- coverage assumes a target distribution, which may limit its applicability for pre-training
- coverage relies on being able to verify samples, which may be difficult for certain (more open-ended) tasks

### Questions
Coverage seems to focus on diversity more than quality - could over indexing on this during pre-training make post-training more difficult?

Would you recommend using this instead of perplexity during pre-training? If so, how would you expect it to behave in a scaling law study?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes the coverage principle: next-token pre-training implicitly drives models toward good coverage over high-quality responses, which better predicts post-training success than cross-entropy alone. The authors provide theoretical justification using autoregressive linear models and some assumptions such as assumption on convexity.
Based on the findings, the paper further presents a test-time training decoding scheme that updates logits while sampling; this improves coverage and can outperform pure MLE-based guarantees.In addition, it also provides a method to select checkpoints with better pass@k than cross-entropy-based selection.
This paper poses new insights on understanding the relationship between pretraining and downstream performance and provide effective ways to utilize these insights in practice.

### Strengths
1. The paper provides a solid theoretical justification on the proposed coverage hypothesis.
2. The insights can be translated to useful applications, including better test-time sampling methods and better checkpoint selection methods.
3. This paper can inspire future works to develop algorithms that can be more effectively used for improving coverage using methods in addition to traditional next-token-prediction loss.

### Weaknesses
1. As a general limitation of the theoretical justification of deep learning, the proofs are shown with linear autoregressive models and assumptions on convexity, therefore the takeaways may have a risk on directly being translated to complex deep neural networks.
2. The hypothesis does not directly indicate a better way than next-token-prediction for pretraining to improve the coverage during training. The proposed distillation-only training is unrealistic in pretriaining.
3. The experiments are conducted on a relatively toy cases, e.g., a graph reasoning task, while real-world tasks are more complicated.

### Questions
1. Would it be possible to demonstrate the checkpoint selection or decoding methods using an LLM and complicated tasks, such as HLE, AIME, and factual QAs like SimpleQA, etc.?
2. Your results often rely on the data distribution \pi_D covering the downstream task. Can you formalize how much misspecification in \pi_D is tolerable while keeping useful coverage guarantees?
3. For the gradient normalization trick, what’s a plug-and-play recipe for transformers and how to implement it?
4. How robust is the proposed TTT sampler to distribution shift, long contexts, and temperature samplig?
5. Results focus on graph reasoning with GPT-2–style models. Can you provide additional evidence on natural-language tasks (e.g., code-gen Pass@N or reasoning benchmarks) to test the coverage–performance link?
6. Can you provide an intuitive example where cross-entropy improves while coverage worsens, and a rule-of-thumb for detecting this in training curves?
7. Any other suggestions for LLM pretraining/mid-training before postraining?

### Soundness
4

### Presentation
4

### Contribution
4
