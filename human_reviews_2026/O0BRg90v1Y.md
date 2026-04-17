# From Compression to Specialization: An Information-Preserving Approach for Dense to Mixture-of-Experts Construction

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4

## Abstract
The high cost of training Mixture-of-Experts (MoE) models from scratch has spurred interest in converting pre-trained dense models into sparse MoE models.
However, existing dense-to-sparse MoE methods are constrained by a fundamental trade-off between initial expert diversity and knowledge inheritance, often requiring extensive post-training to be effective.
We address this by proposing a new expert construction paradigm that repurposes data-driven model compression, and validate that low-rank factorization is uniquely effective at balancing this trade-off.
Based on this insight, we introduce MIDAS, a framework that crafts specialized experts by applying low-rank factorization to a base model, guided by distinct calibration datasets.
Under limited compute budgets, MIDAS significantly outperforms existing dense-to-sparse approaches through a parameter-efficient strategy that trains only its gating network and low-rank adapters.
Crucially, we demonstrate that MIDAS improves model stability by mitigating the severe load imbalance found in prior work, while also producing experts with clear, interpretable specializations that align with established Transformer functional theory.
Overall, MIDAS presents a robust and efficient pathway for MoE construction, addressing the diversity-knowledge trade-off through an information-preserving approach.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces MIDAS, a method that transforms dense LLMs into sparse Mixture-of-Experts models via low-rank decomposition and parameter-efficient fine-tuning. Using Llama-2-7B as the base, each expert is derived from calibration data, followed by 1.3 B-token CPT and 0.4 B-token SFT. The authors claim improved data efficiency (DES) and specialization with minimal training cost.

### Strengths
1. Framing: interprets low-rank compression as a route to expert specialization.
2. Analyses on expert load distribution and calibration sensitivity.
3. Lightweight tuning scheme using LoRA is practical in principle.

### Weaknesses
1. DES metric: Since all MIDAS experiments are conducted using Llama-2 as a backbone, it is inappropriate to claim superiority over Llama-2 in terms of DES.
2. Accuracy degradation ignored: MIDAS (CPT + SFT) consistently underperforms the Llama-2 baseline on several downstream tasks.
3. Lack of compute transparency: The paper fails to report fundamental cost statistics such as FLOPs or GPU hours for training.
4. Outdated setup: All experiments are limited to Llama-2-7B. Stronger modern dense models, such as Llama-3 or Qwen-3, are not tested, leaving it unclear whether the claimed benefits of MIDAS would hold with more capable backbones.
5. Lack of task coverage: The evaluation omits critical domains such as mathematical and coding reasoning (e.g., HumanEval+, LiveCodeBench, MATH-500, BBH).
6. Missing relevant baselines: Contemporary dense-to-sparse conversion methods such as Sparse Upcycling and Drop-Upcycling are not included as baselines under the same computational budget, making it difficult to contextualize MIDAS’s effectiveness.

### Questions
Please clarify the points raised in the Weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an expert-initialization method for converting dense models to MoE models. Specifically, the paper proposes to use different calibration datasets to initialize different experts via low-rank factorization. The specially initialized model is trained to close the gap between the MoE model and its parent dense model.

### Strengths
1. Interesting observation about the sensitivity of data-dependent compression of LLMs on the selection of calibration data, specifically for SVD-based compression

2. The paper is easy to follow

### Weaknesses
1. The main goal of the paper is to convert a dense model into an MoE model. The motivation is that training an MoE model from scratch is challenging. From this perspective, the paper didn't provide any comparison with MoE models trained from scratch.

2. It has already been established in the literature that training MoE is computationally efficient. Therefore, to achieve similar performance, a dense model needs far more training compute. However, the proposed method loses performance significantly compared to its parent dense model, even after training the initialized MoE model.

3. The proposed method can't outperform other dense-to-MoE baselines, despite having a significant load imbalance for the baseline.

4. The proposed expert-initialization method heavily depends on the diversity of calibration data. Therefore, the unavailability of diverse calibration data may undermine the effectiveness of the proposed method. 

5. No formal theoretical justification has been provided for the proposed initialization of the experts.

### Questions
1. What is the Sharing-Inter method? I can't find any citation of Sharing-Inter in the paper.

2. Can the authors provide a clear justification of why one should convert a dense model into MoE, rather than training MoE from scratch?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the challenge covnerting pre-trained dense LLMs into sparse MoE architectures. The authors identify a trade-off between inheriting knowledge from the base models vs diversity of expert modules. They propose an approach that uses low-rank factorization (SVD) with distinct calibration datasets to construct specialized experts, demonstrating that the approach exhibits high sensitivity to calibration data, enabling diversity, while preserving knowledge better in comparison to methods such as structured pruning. Experiments seem to show competitive performance, data efficiency, and improved load balancing.

### Strengths
- the framing of the problem is intuitive, and a preliminary analysis demonstrates the choice for using SVD and low rank decommposition in this manner

- Experimental analysis covers 12 benchmark datasets

- Section 4.5 shows useful analysis of expert specialization (heatmaps)

- Load balancing insights reveal stability issues in prior works, demonstrating further the advantage of the proposed approach

### Weaknesses
-The baseline comparisons are limited. The paper does not compare against a wider range of recent upcycled-MoE baselines such as Sparse Upcycling (Komatsuzaki et al., 2023), Drop-Upcycling (Nakamura et al., 2025), Auxiliary-Loss-Free Load Balancing (Wang et al., 2024).

- All experiments only have 4 experts - no expert number ablation. No ablation studies on key desgin choices (E.g., lora rank)

- The compression ratio is set to 25%, but this is not a well explained choice

- It is claimed that sharing-inter will degrade with continued training due to load imbalance. Can experiments be provided that validate this?

- There is no indication about the proper choice of datasets and how this choice induces specialization equivalent to training MoEs from scratch. What if the test examples do not clearly match calibration datasets?

- It appears that the method does not allow for any overlap between experts (no shared expert). Could this be a downside in some cases?

- There is no clear quantitative comparison of total computation (construction, training, inference) with other MoE upcycling methods. 

- How sensitive is performance to the number and selection of fine-tuning datasets used to form experts? Would including additional baselines such as DeepSeek Balancing or BTX change the conclusions?What is the trade-off between expert diversity and computational cost when scaling to more fine-tuning datasets

### Questions
please see weaknesses above!

### Soundness
3

### Presentation
3

### Contribution
3
