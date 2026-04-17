# PARD: Accelerating LLM Inference with Low‑Cost PARallel Draft Model Adaptation

- Decision: Accept (Poster)
- Scores: 4, 2, 6, 6

## Abstract
The autoregressive nature of large language models (LLMs) fundamentally limits inference speed, as each forward pass generates only a single token and is often bottlenecked by memory bandwidth. Speculative decoding has emerged as a promising solution, adopting a draft-then-verify strategy to accelerate token generation. While the EAGLE series achieves strong acceleration, its requirement of training a separate draft head for each target model introduces substantial adaptation costs. In this work, we propose **PARD (PARallel Draft)**, a novel speculative decoding method featuring target-independence and parallel token prediction. Specifically, PARD enables a single draft model to be applied across an entire family of target models without requiring separate training for each variant, thereby minimizing adaptation costs. Meanwhile, PARD substantially accelerates inference by predicting multiple future tokens within a single forward pass of the draft phase. To further reduce the training adaptation cost of PARD, we propose a COnditional Drop-token (COD) mechanism based on the integrity of prefix key-value states, enabling autoregressive draft models to be adapted into parallel draft models at low-cost.  Our experiments show that the proposed COD method improves draft model training efficiency by 3x compared with traditional masked prediction training. On the vLLM inference framework, PARD achieves up to 3.67x speedup on LLaMA3.1-8B, reaching 264.88 tokens per second, which is 1.15x faster than EAGLE-3. Our code is available at https://github.com/AMD-AGI/PARD.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes PARD, which accelerates autoregressive generation by training a target-independent draft model that generate drafts in parallel. Furthermore, the authors propose the CoD method to accelerate the training of the parallel inference model. Owing to parallel generation and target independence, PARD can quickly transfer across different model within the same series and generate drafts at a low cost, achieving greater acceleration compared to the autoregressive EAGLE.

### Strengths
1. The paper is well-structured, with clear and intuitive diagrams that effectively support the explanation.

2. It explores the concept of parallel draft generation, providing insightful and innovative perspectives on the topic.

### Weaknesses
1. The paper lacks a comparison with other similar draft acceleration methods, such as hierarchical acceleration [1,2] and draft models in diffusion models [3].

2. The experimental setup is not sufficiently detailed, with insufficient explanation of decoding strategies (e.g., chain or tree).

[1] Chen Z, Yang X, Lin J, et al. Cascade speculative drafting for even faster llm inference[J]. Advances in Neural Information Processing Systems, 2024, 37: 86226-86242.

[2] Zhao W, Huang Y, Han X, et al. Ouroboros: Speculative decoding with large model enhanced drafting[J]. arXiv preprint arXiv:2402.13720, 2024.

[3] Christopher J K, Bartoldson B R, Ben-Nun T, et al. Speculative diffusion decoding: Accelerating language generation through diffusion[J]. arXiv preprint arXiv:2408.05636, 2024.

### Questions
1. The experimental results in Table 1 lack reporting on the acceptance length, and the acceleration ratio of EAGLE is lower than that of other benchmark methods. It would be useful to clarify whether Tree validation is employed in this study.

2. In addition to the absence of a comparison with related methods, the related work section overlooks relevant literature on parallel generation techniques, which could offer valuable context for this study.

[1] Kou S, Hu L, He Z, et al. Cllms: Consistency large language models[C]//Forty-first International Conference on Machine Learning. 2024.

[2] Wang Y, Luo X, Wei F, et al. Make some noise: Unlocking language model parallel inference capability through noisy training[J]. arXiv preprint arXiv:2406.17404, 2024.

[3] Gao X, Xie W, Xiang Y, et al. Falcon: Faster and parallel inference of large language models through enhanced semi-autoregressive drafting and custom-designed decoding tree[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2025, 39(22): 23933-23941.

3. As indicated in Table 2, the high acceptance rate of PARD primarily stems from its consistency with existing homologous models, rather than additional training. Consequently, it is essential to conduct comparisons between PARD and other draft model acceleration approaches to better assess its relative performance.

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
4

### Summary
The paper proposes PARD (PARallel Draft), a novel method for speculative decoding (SD) designed to accelerate Large Language Model (LLM) inference while featuring target-independence and parallel token prediction. PARD addresses the high adaptation cost of target-dependent methods like EAGLE by allowing a single, low-cost draft model to be applied across an entire family of target models. The core mechanism involves fine-tuning a small auto-regressive (AR) model into a parallel draft model, utilizing the Mask-Predict approach with special mask tokens ($m_k$) to predict multiple candidate tokens in a single forward pass. This significantly reduces the draft model's latency from $K \times T_D$ to $T_D$.
To reduce the training cost for this parallel adaptation, the authors introduce the Conditional Drop-token (COD) mechanism. COD enables sparse sampling of training data while ensuring the integrity of prefix key-value states, boosting training efficiency by up to $3\times$ compared to traditional masked prediction training. Experiments on the industrial vLLM framework show that PARD achieves a state-of-the-art speedup on LLaMA3.1-8B.

### Strengths
1. This paper proposes PARD, which allows a single same-family draft model to accelerate an entire family of target models via multi-token prediction with look-head tokens.

2. This paper integrates a token-dropping scheme to reduce the draft model’s training cost.

### Weaknesses
- The paper frames advanced SD methods as "target-dependent" and PARD as "target-independent", yet PARD crucially assumes the availability of a same-family small model—often unrealistic for fine-tuned production LLMs. In those settings, self-speculative decoding from the target model itself (e.g., LayerSkip/ACL’24, Self-SD/ACL’24, SWIFT/ICLR’25) seems strictly more applicable. The paper should explicitly discuss this constraint, compare against self-SD scheme, and clarify when PARD is preferable.
   
- Much of the method resembles transferring look-ahead token prediction to a small model and then boosting with fine-tuning. Given that small-model training is a one-off offline cost, the contribution of the token dropping mechanism reads like a patch rather than a core innovation.

- The concept of parallel decoding is not novel. For instance, [1] has proposed a similar idea based on a parallel-decoded drafter and a target model as a verifier for speculative decoding. Additionally, Medusa, Multi-Token Prediction (MTP) are existing parallel decoded speculation approaches. Compared with the existing works, this paper mainly differentiates in its draft model architecture design that leverages the [MASK] tokens as a placeholder to predict future tokens within one model forward pass. However, this design is similar to the approach proposed in [2] [3].

- PARD comes from converting the sequential prediction to a parallel prediction, reducing the time cost of the draft stage from $$K \times T_D$$ to $T_D$. However, the unresolved issue is whether the $T_D^$ of the new draft model will change, and whether it is possible that $K \times T_D$ is less than $T_D^$ when the draft model is too large.

- The paper claims that its draft model is target-independent and can accelerate an entire family of target models. However, for target model families with a large range of parameter quantities, using the same draft model (lacking the adjustment of the draft model to fit the size of the target model) may lead to the problems mentioned in the last weakness. That is, universal draft model may perform slower in parallel decoding than the ideal draft model even if in sequential decoding. The evidence in the paper is insufficient to reveal or fix this issue.

- The Shared Mask Token ID Strategy in ablation study needs concrete details: how mask tokens are inserted across variants, which IDs are shared.

- Presentation issues: 
   * Fig. 4 (especially subfigures (b) and (c) on training-time attention masks) is hard to parse. In addition, for Fig. 4(a), clarify the meaning of green/red. If green denotes permitted attention, the causal mask for query at position t should be upper-triangular over keys/values (e.g., "Q(story)" must attend to its history; the last column should be green). 
   * Equations (7) and (8) have an indexing bug: substituting k=0 (or k=1) yields m_{-1}; define base cases and boundary conditions to prevent invalid indices. 
   * The core idea of COD ("earlier-subtask tokens are more critical; later ones can be dropped") lacks justification—why are later tokens less important? Is this theoretically motivated or empirically observed (e.g., uncertainty, acceptance rates)?

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes PARD (Parallelized Accelerated Reasoning Decoding), a novel speculative decoding method featuring target-independence and parallel token prediction. PARD is highly generalizable: its target-independent design allows a single draft model to accelerate an entire family of target models. Through mask tokens, PARD substantially accelerates inference by predicting multiple future tokens within a single forward pass of the draft phase. And they propose a COnditional Drop-token (COD) strategy for the effient training of PARD. Leveraging the integrity of prefix key-value states, COD enables low-cost adaptation of autoregressive draft models into parallel ones, boosting training efficiency by up to 3× while maintaining accuracy. In the end, the author integrates PARD into the high-performance inference framework vLLM.

### Strengths
1. This paper makes a pivotal conceptual shift by introducing a target-independent speculative decoding framework. Unlike dominant approaches like EAGLE and Medusa that train a draft model tightly coupled to a specific target model, PARD enables a single draft model to accelerate an entire family of target models.
2. The work innovates by integrating parallel token prediction into speculative decoding via masked language modeling.A key novelty is the Conditional Drop-token (COD) mechanism, which leverages the integrity of prefix key-value states to dramatically reduce training overhead. This allows efficient adaptation of autoregressive models into parallel draft models. The use of a shared mask token ID to enable extrapolation beyond the trained prediction length is another clever and inventive design choice that enhances practicality.
3. The integration with the industrial-grade vLLM framework underscores its immediate practical significance, providing a scalable and efficient solution that directly addresses real-world inference bottlenecks.

### Weaknesses
Limited Analysis of the Parallel Drafting Mechanism's Limitations: The paper celebrates the efficiency of parallel drafting but fails to critically analyze its inherent constraints. Generating K tokens in one forward pass necessarily uses less contextual information (relying on masks) than an autoregressive model, which may hurt prediction accuracy for long-range dependencies or complex reasoning steps. This likely explains the performance drop on certain tasks (e.g., PARD vs. EAGLE-3 on SpecBench for LLaMA3.3-70B). The work would be strengthened by a dedicated analysis—for instance, measuring how acceptance rate decays with K on different task types—to delineate the practical boundaries of its parallel approach.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces PARD, a speculative decoding framework that accelerates LLM inference through a target-independent, parallel draft model. It combines mask-based prediction and a novel Conditional Drop-token (COD) training strategy to enhance both training and inference efficiency. PARD achieves up to 3.67× speedup on LLaMA3.1-8B and outperforms the current state-of-the-art EAGLE-3 model in both speed and generalizability.

### Strengths
1. Unlike methods like EAGLE, PARD can accelerate an entire family of target models with a single draft model, significantly reducing adaptation and deployment costs.
2. Moreover, the use of masked tokens allows PARD to predict multiple tokens in a single pass, improving inference speed without sacrificing quality.
3. Extensive benchmarks on HumanEval, GSM8K, and SpecBench across multiple LLMs demonstrate consistent speedups and high acceptance rates compared to EAGLE and vanilla SD.

### Weaknesses
The paper introduces PARD as a parallel and target-independent speculative decoding method. However, the experimental section lacks direct comparisons with other recent parallel speculative decoding methods such as ParallelSpec, BiTA, or PaSS.

### Questions
Could the authors justify the omission of comparing to other recent parallel speculative decoding methods, and if possible, include some comparative results to support the performance and efficiency claims of PARD in the parallel decoding category?

### Soundness
3

### Presentation
3

### Contribution
2
