# NDAD: Negative-Direction Aware Decoding for Large Language Models via Controllable Hallucination Signal Injection

- Decision: Accept (Poster)
- Scores: 4, 6, 2, 6

## Abstract
Large language models (LLMs) have recently achieved impressive progress in knowledge-intensive and reasoning tasks. However, their tendency to produce fabricated or factually inconsistent content remains a fundamental challenge to their practical deployment. To address this issue, we propose Negative-Direction Aware Decoding (NDAD), a novel decoding method that identifies and exploits hallucination signals as repulsive directions in the model’s representation space, thereby improving factual adherence without retraining. Specifically, NDAD elicits hallucination-leaning signals by selectively masking critical attention heads, which exposes unstable hypotheses that the model would otherwise amplify during generation. To regulate the influence of these signals, NDAD employs two complementary weights: a global alignment weight measuring how well the induced signal aligns with the layer’s native activations (thus quantifying its referential utility) and a local weight estimating whether low-probability tokens in the masked distribution are likely to evolve toward the final output. Based on the weights, we derive a latent hallucination distribution that serves as the negative direction. A lightweight gradient-descent step then subtracts mass from hallucination-prone regions of the output distribution, adjusting the final logits while preserving the model’s high-confidence predictions. Extensive experiments across multiple LLMs and diverse benchmark datasets demonstrate that NDAD consistently enhances factual reliability without requiring additional training or external knowledge.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Negative-Direction Aware Decoding (NDAD), a raining-free decoding strategy designed to mitigate hallucinations in LLMs. The core contribution lies in its approach of actively identifying and leveraging "hallucination signals" as a repulsive force during generation. Rather than solely promoting factual content, NDAD isolates hallucination-prone directions by strategically masking influential attention heads. The method then employs a dynamic weighting mechanism, integrating both global consistency and local divergence measures, to controllably steer the model's output away from these identified negative trajectories via a gradient-descent adjustment. Experiments demonstrate that NDAD consistently enhances factual reliability across diverse LLMs and benchmark datasets, offering a lightweight yet potent solution to a critical challenge in model deployment.

### Strengths
- The method is training-free and operates at inference time, making it a practical and computationally accessible approach for improving factuality compared to methods requiring model fine-tuning
- The dual-weighting scheme that considers both global consistency (alignment with layer activations) and local divergence (evolution of low-probability tokens) provides a non-trivial method for modulating the hallucination signal.
- The paper provides ablation experiments that investigate the necessity of its core components, such as the global and local weights and the importance-guided masking strategy, strengthening the justification for its specific design choices.

### Weaknesses
- The performance improvements over the baseline are often marginal, questioning the method's practical significance.
- The method introduces a number of new hyperparameters (number of masked heads, number of layers, top-I tokens, evolution rate α) that likely require careful tuning for different models and tasks, potentially limiting its out-of-the-box utility.
- The paper does not analyze the computational overhead or potential impact on generation latency. Furthermore, evaluation is focused on factuality, with no assessment of whether the method harms other text quality aspects like coherence or fluency.
- All models tested are in the smaller 7B-13B parameter range. The method's viability and computational cost on much larger models (e.g., 70B+) remain unproven.

### Questions
1. What is the computational overhead (e.g., latency increase per token) of NDAD compared to greedy decoding and the baseline?
2. Have you evaluated whether NDAD negatively impacts other text qualities, such as fluency, coherence, or stylistic appropriateness?
3. How do you expect NDAD's performance and computational cost to scale when applied to much larger models (e.g., 70B+ parameters)?

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
3

### Summary
The following paper proposes a novel approach to hallucination reduction in Large Language Models (LLMs). It decerns a direction responsible for the hallucinations by masking out most influential attention heads responsible for the factuality. This direction is then subtracted from the original output logits of the model as a way to reduce the hallucinations in the final predictions. The paper showcases the effectiveness of the proposed method on a number of LLMs and datasets. It shows that proposed approach increases the accuracy on a number of benchmarks compared to existing hallucination reduction methods and studies the impact of the components driving the hallucinations.

### Strengths
+ The paper is well written, the method and the experiments  is clearly explained
+ The authors report results on a number of datasets and models - they show that proposed methods consistently outperform the existing baseline approaches.

### Weaknesses
**Discussion on limitations of the work**

The paper is well written but there are clearly some insights missing on the limitations of the proposed work. 

1) The paper relies heavily on an existing algorithm that identifies heads responsible for factuality but it is unclear if the heads responsible for the factually are consistently the same for all tasks and sample sizes and what's the accuracy in identifying such heads. It is unclear how the number of heads and layers are chosen. It is hard to tell from Figure 4 what number of heads and layers one should choose.

2) It is also unclear how the results are aggregated for single token vs multi-token generation. When we say logits_L is this the logits for the final answer, a single token or aggregation of logits across multiple tokens ?

3) It is also unclear how we set the threshold to select top I tokens ?


Descriptions of the figures and tables. 

The descriptions are quite vague and high level. It might be important to mention in the titles what are we exactly measuring e.g. Overall Accuracy or the Factuality Accuracy. Figure 1 is great but the description is very vague and it requires reading the rest of the paper to understand it. 

It would we also interesting to describe the choice of combining global and local weights. Why do we use the product of W_global * W_local vs additive influence W_global + W_local

### Questions
See weaknesses
Table 1 shows that some of the baseline hallucination mitigation approaches worsen the accuracy. Why is that the case ? Isn't the purpose of those methods to increase the accuracy ?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a training-free method to reduce hallucination named NDAD. The method aim to reduce LLM hallucinations, its core ideas is to steer output distributions away from hallucination-prone directions. First, the method masks critical attention heads (selected based on metrics such as importance scores and entropy) to generate a "hallucination signal," then uses dynamic weighting (global weight for signal-logit alignment and local weight for low-probability token evolution) to control this signal's impact. Finally,  the output logits are modified by gradient descent to penalize the KL divergence from the negative distribution. The paper provides tests across multiple LLMs and different benchmarks, the results show that the method has similar results to baselines such as  DoLa, SLED.

### Strengths
1. The method presents a unique and compelling perspective by "subtracting negatives"—inducing a hallucination signal and then repelling the model from it.

2. The approach demonstrates broad applicability, validated across multiple model families (Llama, Mistral, Qwen) and sizes (7B, 13B, 8B) on a diverse set of tasks (TruthfulQA, GSM8K).

3. Rigorous studies validate the core design choices, including the efficacy of the masking strategy and the necessity of both the global and local weighting components.

### Weaknesses
1. My biggest concern is that the method is not practical at all. The proposed method requires a second, modified forward pass at every decoding step. In the meanwhile, the gradient-descent step is problmetic as well. This will result in inference-time latency and definitely problemetic in many inference settings(may lead to KV-cache related issues). All make the method impractical.

2. The motivation and calculation for the local weight (evolution trajectories, one-hot vectors) lack intuitive, step-by-step clarity

3. The introduction of several new parameters combined with the observed performance curves suggests the method may be highly sensitive to tuning. It can be critical for real-world deployment

4. The choice of using an arbitrary squared transformation for the final weight scores is not sufficiently justified. 

5. Another concern is that even with introduction of overhead and hyper-parameters, the performance gain is very small and even worse compared with existing method

### Questions
see weakness

### Soundness
3

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
3

### Summary
The paper proposes Negative-Direction Aware Decoding (NDAD), a training-free decoding strategy to reduce hallucinations in large language models. NDAD first intentionally disrupts a model’s most fact-relevant attention heads to induce a “hallucination-leaning” prediction, which reveals how the model would start to generate unreliable content. It then identifies the most dangerous parts of that signal — the token directions that are likely to grow into the final answer — and treats them as a negative direction. During actual decoding, NDAD pushes the model’s output distribution away from this negative direction via a small KL-based adjustment to the final logits. Across multiple benchmarks (factual QA, open-ended QA, and chain-of-thought math reasoning) and several popular LLMs, NDAD matches or outperforms prior decoding-time methods like SLED and Activation Decoding, without extra training data, external retrieval, or modifying model weights.

### Strengths
1. The perspective is new: instead of “pulling the answer toward the correct truth,” NDAD “pushes the model away from being wrong.” It first induces a hallucination signal — the direction the model tends to move toward incorrect content after targeted corruption — and then explicitly repels the final output probabilities away from that direction. This reframes the problem from boosting truthfulness to actively repulsing hallucination, which is conceptually distinct.
2. The method introduces a weighted mechanism that combines global consistency and local evolution, instead of bluntly suppressing all probabilities. After combining these two views, it applies normalization and squared weighting to emphasize high-confidence risky directions. This fine-grained control avoids the typical failure mode where aggressive down-weighting breaks fluency and makes the sentence fall apart.
3. The approach shows consistent gains across multiple tasks, and is particularly strong on tasks that require reasoning and multi-step thinking.

### Weaknesses
1. Runtime comparison with baselines. he method requires extra computation at inference (masking heads, computing global/local weights, doing the gradient-style correction), but the paper does not systematically report the runtime or memory overhead versus standard decoding or simpler contrastive methods. 
2. Regarding the reliability and causality of the “critical attention head” selection process: Can the authors provide more fine-grained evidence, for example, whether the heads identified as critical tend to attend to high-confidence knowledge sources (such as factual spans in the context, entity mentions from the question, numerical cues, etc.) rather than merely contributing to fluency or syntax?

### Questions
Please refer to the weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
3
