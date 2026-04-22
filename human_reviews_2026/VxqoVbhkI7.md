# Efficient Forward-Only Data Valuation For LLMs and VLMs

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2

## Abstract
Quantifying the influence of individual training samples is essential for enhancing the transparency and accountability of large language models (LLMs) and vision-language models (VLMs). Existing data valuation methods rely on Hessian information or model retraining, making them computationally prohibitive for billion-parameter models. In this work, we introduce For-Value, a forward-only data valuation framework that enables scalable and efficient influence estimation for both LLMs and VLMs. By leveraging the rich representations of modern foundation models, For-Value computes influence scores using a simple closed-form expression based on a single forward pass, thereby eliminating the need for costly gradient computations. Our theoretical analysis demonstrates that For-Value accurately estimates per-sample influence by capturing alignment in hidden representations and prediction errors between training and valuation samples. Extensive experiments show that For-Value matches or outperforms gradient-based baselines in identifying impactful fine-tuning examples and effectively detecting mislabeled data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a forward-only data valuation framework under the unconstrained feature assumption. It estimates per-sample influence using only a single forward pass, unlike existing influence functions which requires backpropagation or retraining. Experiments on LLMs and VLMs tasks show that the method achieves comparable or superior performance to gradient-based baselines while being orders of magnitude faster.

### Strengths
In influence function, the computation of gradients via backpropagation and the computation of Hessian matrices are major bottlenecks, particularly severe for large-scale LLMs and VLMs with massive parameters and training data. The idea of introducing the unconstrained feature assumption to enable a forward-only influence function is interesting and practically valuable.

Evaluations across diverse models (Qwen2.5, LLaMA) and modalities (text, image–text) confirm the generality and efficacy of the proposed method, though there remain some concerns about the reliability of the reported results as mentioned below.

### Weaknesses
This paper heavily relies on the unconstrained feature assumption, and this dependence should be made more claer. The key idea is not that the authors devised a way to avoid the backward pass, but rather that, under the unconstrained feature assumption, the influence estimation can be computed using only the forward pass. This point should be stated more clearly. For example, it might even be appropriate to add “under unconstrained feature assumption” into the title.

According to Appendix A.2, the paper shows that the influence of training data can be approximated by the inner product of gradients over all parameters (= Hessian-free influence functions). The proposed method further approximates this quantity, under the unconstrained feature assumptions, by using only the inner product of gradients with respect to the final unembedding layer. Therefore, the Hessian-free baseline should, in principle, achieve performance comparable to or even better than the proposed method. However, in the experiments, the proposed method significantly outperforms the Hessian-free, and the paper provides little explanation for this discrepancy, which raises concerns about the reliability of the reported results.

Theorem 1 in this paper and Theorem 4.4 in the following work [1] are fundamentally very similar. In fact, the proof of Theorem 1 (Appendix A.2) follows almost the same line of proof as Section 7.2 (“Proof of Theorem 4.4”) of the referenced paper.

[1] Deng et al., [On the Effect of Negative Gradient in Group Relative Deep Reinforcement Optimization.](https://arxiv.org/abs/2505.18830) AI4Math@ICML25.

### Questions
According to Appendix A.2, the authors claim that, under the neural collapse assumptions, the inner product of gradient w.r.t. all parameters (i.e., Hessian-free influence) can be approximated by that of the final unembedding layer. This point requires a more detailed discussion. To what extent does the inner product of gradients w.r.t. unembedding layer account for the inner product of gradients w.r.t. all parameters? If this proportion is small, the neural collapse assumptions may not actually hold; if it is large, then the authors should explain why, despite this, the Hessian-free baseline performs so poorly compared to their method.

In Appendix A.1, the first equation defining the supervised fine-tuning loss seems to be missing an equality sign.

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
4

### Summary
This paper introduces For-Value, a forward-only method to estimate data importance for LLMs and VLMs using just a single forward pass. It measures alignment between hidden representations and prediction errors, achieving comparable or better results than gradient-based methods with much higher efficiency.

### Strengths
1. The paper is well organized.

2. Experiments cover multiple tasks and model types.

3. The proposed method is highly efficient, and the paper provides a time cost analysis.

### Weaknesses
1. The distinction between For-Value and last-layer gradient–based approaches should be clarified more clearly, as the current presentation leaves the contribution of the method somewhat ambiguous.

2. The paper states that "data valuation can be approximated by the alignment between hidden representations and prediction errors." It would strengthen the work to include experiments comparing the proposed approximation with ground-truth valuations for verification.

3. The font size in some figures is too small and could be increased to improve readability.

### Questions
Please see the Weaknesses part.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces For-Value, a forward-only framework for per-sample data valuation in LLMs/VLMs. The key idea is to estimate how a training example would improve the log-likelihood of a valuation example by aligning two quantities computed from a single forward pass: (i) hidden representation similarity and (ii) prediction-error similarity (one-hot minus softmax). The authors derive a closed-form score, implement it efficiently via a single matrix inner product with a batch-specific sub-vocabulary, and report strong speed/VRAM advantages over gradient/Hessian-based baselines while matching or outperforming them on data selection, mislabeled-data detection, and fine-tuning utility across LLM and VLM settings.

### Strengths
Forward-only & scalable. The method needs only hidden states and softmax outputs; no backprop, Hessians, or multi-checkpoint accumulation. Efficient sub-vocabulary pruning further reduces cost.

Clear theory-to-implementation path. The score has a closed-form derivation and maps directly to an implementable matrix inner product; the intuition (“representation × error alignment”) is easy to grasp.

Broad applicability. Demonstrated on both LLM and VLM benchmarks, with competitive or superior performance for data selection and mislabeled-sample detection.

Strong engineering value. The simplicity and speed make it practical for billion-parameter models and large corpora, enabling workflows (filtering, curriculum, sampling weights) that are otherwise too costly.

### Weaknesses
- **Positioning w.r.t. Hessian-free influence functions (gradient inner products).** The method appears very close to **Hessian-free influence** ideas based on gradient–gradient inner products (e.g., TracIn-like). Under cross-entropy with a linear readout, the per-token log-likelihood gradient w.r.t. the readout is
  $$
  \nabla_W \log \pi_\theta(y_t \mid x)
  \;=\; \big(e_{y_t} - \pi_\theta(\cdot \mid x)\big)\, h_t^\top,
  $$
  and aggregating outer products across tokens gives
  $$
  C \;=\; \sum_t h_t \, \big(e_{y_t} - \pi_t\big)^\top, \qquad
  S(v,i) \;=\; \langle C_v,\, C_i \rangle_F ,
  $$
  which matches the **last-layer gradient dot-product** form (up to sign/constant conventions). Please discuss this connection thoroughly and compare against Hessian-free/gradient-dot-product baselines under matched budgets.

- **Method-level proximity to *Data Shapley in One Training Run*.** Beyond problem framing, the **forward-only expansion** and reliance on forward-available quantities seem methodologically close to *Data Shapley in One Training Run*. Please provide a focused **method-level** comparison (derivation steps, assumptions, computational pipeline), and include direct empirical contrasts where feasible.

- **Missing related work that should be covered and contrasted method-wise.** In particular, (i) *Revisit, Extend, and Enhance Hessian-Free Influence Functions* and (ii) *Layer-Aware Influence for Online Data Valuation Estimation* (which explicitly analyzes **last-layer/representation-level** influence). A detailed method-level discussion of similarities/differences with these works is needed (not just problem-level positioning).

- **Information–efficiency tradeoff needs analysis.** Please analyze, under **equal budget**, how the forward-only proxy trades information content for speed, and when it can match or outperform more informative baselines (last-layer-only gradients; Hessian-free IP/TracIn with checkpoint accumulation). If superior accuracy with less information is claimed, support it with concise analysis or evidence, rather than leaving it as an empirical observation.

### Questions
Same with weakness

### Soundness
3

### Presentation
2

### Contribution
2
