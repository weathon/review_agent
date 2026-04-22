# DCPO: Dynamic Clipping Policy Optimization

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Reinforcement Learning from Verifiable Rewards (RLVR) has emerged as a promising framework for enhancing the reasoning capabilities of large language models. However, existing approaches, such as GRPO,  often suffer from zero gradients. This problem mainly stems from (i) fixed clipping bounds for token-level probability ratios and (ii) the standardization of identical rewards, which can lead to ineffective gradient updates and underutilization of generated responses. In this work, we propose **D**ynamic **C**lipping **P**olicy **O**ptimization (**DCPO**). DCPO (i) introduces a dynamic clipping strategy that adaptively adjusts clipping bounds based on token-specific prior probabilities to enhance token-level exploration, and (ii) employs a smooth advantage standardization technique that standardizes rewards across cumulative training steps to improve the response-level effective utilization of generated responses. DCPO achieved state-of-the-art performance on four benchmarks based on four different models. Specifically, on the AIME-24 benchmark, DCPO reaches an Avg@1 of 46.7 (greedy decoding) and an Avg@32 of 38.8 (32-sample decoding) with the Qwen2.5-Math-7B model, surpassing DAPO (36.7/31.6), GRPO (36.7/32.1) and GSPO (40.0/34.9). On the AIME25 benchmark based on Qwen2.5-14B, DCPO achieves a performance of (23.3/19.0), surpassing GRPO (13.3/10.5), DAPO (20.0/15.3) and GSPO (16.7/9.9). Furthermore, DCPO achieved an average 28% improvement in the nonzero advantage over GRPO in four models, doubled the training efficiency compared to DAPO, and significantly reduced the token clipping ratio by an order of magnitude compared to both GRPO and DAPO, while achieving superior performance. These results demonstrate DCPO's effectiveness in leveraging generated data more efficiently for reinforcement learning in large language models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Dynamic Clipping Policy Optimization (DCPO), a novel reinforcement learning method for large language models that addresses the zero-gradient problem in existing approaches like GRPO. By implementing adaptive token-level clipping bounds and smooth advantage standardization, DCPO enhances exploration and response utilization, achieving good performance across multiple benchmarks—including some improvements over GRPO, DAPO, and GSPO—while boosting training efficiency and reducing token clipping rates.

### Strengths
- This paper is easy to read and easy to follow
- This paper focuses on an important topic in large language model reasoning, aiming at enhancing the reasoning capabilities of existing LLMs via some specially designed approaches
- The idea of dynamic clipping seems to be novel, and the analysis process is interesting

### Weaknesses
- (major) The main experiments are conducted only on Qwen models, raising doubts about its general benefits when using other LLMs as base models. Qwen series are known to suffer from dataset leakage, making it unclear whether the reported performance in tasks like AIME is reliable or trustworthy
- (major) No code, no homepage, and no models are available, making it unclear whether the results can be reproduced. The reviewer deems it important to open-source the code, datasets, and models in LLM research
- (major) The ablation study part is limited. The authors only conduct experiments on Qwen2.5-Math-7B and report Avg@32 on all benchmarks, and the detailed results on each benchmark should appear in the appendix. Furthermore, based on Figure 3, GRPO w/OTM loss achieves quite similar performance as DCPO, indicating that other components used in DCPO are less effective
- (major) DCPO contains numerous components, but many of them are not novel, either simply adapt from some prior works or just make some minor modifications. I hence cannot say that the technical contributions of DCPO are convincing enough to be accepted in this venue. For example, the authors adopt the dual clipping method from (Ye et al, 2020), the OTM loss is a minor modification compared to SLM or TLM loss
- (major) The performance improvement of DCPO is only marginal on numerous tested tasks (e.g., the performance of DCPO is quite close to that of DAPO). The authors claimed that they achieved state-of-the-art results. The reviewer cannot agree with that. 
- (minor) This paper can benefit greatly from including more LLM-related references, especially those that investigate LLM math reasoning or improving DAPO/GRPO. This is a fast-growing research area, and there are numerous papers that do similar things. The authors should include more discussion in the manuscript
- (minor) Figure 1 is hard to interpret, the y-axis has quite confusing scales

### Questions
The performance of DAPO seems to be inferior to its performance in the original paper, have you checked that? Can the performance of DAPO become stronger with more training steps?

### Soundness
2

### Presentation
3

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
This paper proposed dynamic ratio clipping and smoothed reward standardization methods to encourage the exploration of RL with verifiable rewards. The evaluation is conducted on 4 mathematical datasets. An empirical analysis of the clipping mechanism and an ablation study are provided.

### Strengths
- The paper is well-structured. The key information is presented well.

- The idea of dynamically clipping the advantage depending on the policy probability is well-motivated.

- The empirical analysis verifies less clipping and more response utilization under the proposed method.

### Weaknesses
No discussion about the limitation.- What epsilons values (the clipping threshold) are used in the baseline methods? Using larger epsilons in baselines will also lower the TCR, probably reaching the same-level as DCPO (Section 5.2). This raises a concern of what Section 5.2 can reveal.

- The main analysis (Section 5.2 and 5.3) focuses on DAC. However, SAS seems to be the majority source of the performnce improvement (as seen in Section 5.4). The limited performance improvement based on DAC has weakened the support of its analysis.

- There is no discussion about the clear limitations of this work. For example, the evaluated models are all from the Qwen family, and only Avg metrics are evaluated etc.

### Questions
- In Section 3, two issues arising from randomness are mentioned. Are there references about these issues? If not, is there evidence for their importance?

## Minors
Some symbols require definition or specification when they are introduced, even though Appendix A.2 lists the definitions
- Section 2, Equation 1. $f(x)$ requires specification (e.g., "for any function $f(x)$" if this is the case).
- Section 2, Equation 2, $r(x)$ requires definition.
- Section 3, paragraph 1. $R^i_{j=1,...,G}$, where the variable $G$ requires definition.

### Soundness
2

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
5

### Summary
This paper introduces Dynamic Clipping Policy Optimization (DCPO), a novel reinforcement learning method designed to enhance the reasoning capabilities of large language models (LLMs) within the Reinforcement Learning from Verifiable Rewards (RLVR) framework. The authors identify two key issues in existing methods like GRPO: zero gradients due to reward standardization and restricted exploration from fixed clipping bounds. DCPO addresses these by proposing (i) a dynamic-adaptive clipping (DAC) mechanism that adjusts clipping bounds based on token-specific prior probabilities, and (ii) a smooth advantage standardization (SAS) technique that aggregates reward statistics across cumulative training steps. The authors conduct experiments on four mathematical reasoning benchmarks with four model sizes, demonstrating that DCPO outperforms baselines like GRPO, DAPO, and GSPO in terms of performance, data utilization, and training stability.

### Strengths
The paper's primary strength lies in its well-motivated approach to tackling critical limitations in current RLVR methods. The core idea of modifying clipping bounds and advantage calculation to enhance diversity and stabilize training is conceptually sound and significant.

*   **Originality & Significance**: The paper attempts to provide a more principled way to manage the exploration-exploitation trade-off in policy optimization for LLMs. The proposed dynamic clipping based on token probability $q(x)$ is an intuitive way to grant more exploration space to high-entropy (low-probability) tokens, which are often crucial for discovering novel reasoning paths. The cumulative advantage standardization is also a clever technique to mitigate the pervasive zero-gradient problem, thereby improving sample efficiency.
*   **Clarity**: The paper is generally well-written and clearly structured. The motivations are well-explained, and the proposed methods (DAC and SAS) are described with sufficient mathematical formalism. The experimental setup is detailed, and the results are presented with clarity.
*   **Quality**: The empirical evaluation is extensive in terms of model scales and benchmarks. The introduction of new metrics like RUR and the detailed analysis of TCR and entropy provide valuable insights into the training dynamics of different algorithms, which strengthens the paper's empirical contributions.

### Weaknesses
Despite its strengths, the paper suffers from significant weaknesses, primarily related to the lack of comparison with highly relevant prior work and a potential contradiction in its core motivation.

1.  **Insufficient Comparison with State-of-the-Art**: The main weakness of this paper is the omission of comparisons with a large body of directly related work. The core technical contribution is the modification of clipping values based on token probability and entropy. This is a well-explored area, yet the experiments are limited to generic GRPO variants. The claims of superiority are not sufficiently supported without benchmarking against more relevant and potentially stronger baselines.
    *   I strongly recommend the authors compare DCPO against the following categories of algorithms in their main experiments:
        1.  **Entropy-aware methods**: A direct comparison with recent works that explicitly use entropy to guide the RL process is necessary. This includes methods from [1-4]. Some of these works have reported results that appear to be significantly better than what is presented here.
        2.  **Asymmetric Clipping Baselines**: The paper should include an ablation or comparison with simpler variants that asymmetrically modify `clip_high` and `clip_low`, similar to what is done in DAPO but perhaps with different heuristics. The clipping settings for all baselines and the proposed method must be clearly stated.
        3.  **Entropy Regularization**: A comparison with the classical approach of adding an entropy bonus term to the loss function is essential to demonstrate the claimed benefits of dynamic clipping over traditional entropy-promoting techniques.

2.  **Lack of Qualitative and Theoretical Distinction**: The paper does not adequately discuss the similarities and differences between its proposed clipping mechanism and those in prior work, particularly [4], or with entropy regularization.
    *   From a qualitative and experimental perspective, what are the key differences between the dynamic clipping in this paper and the mechanisms proposed in [4]?
    *   How does modifying the clipping bound for high-entropy tokens differ from directly adding an entropy term to the loss? A discussion on the theoretical implications and practical trade-offs would significantly strengthen the paper's contribution. Without this, the novelty and significance of the proposed method are unclear.

3.  **Contradiction with Stated Motivation**: The paper's motivation is to provide more exploration space for high-entropy tokens. However, as discussed in [3], this very approach can be counterproductive. Granting excessive freedom to high-entropy tokens, especially in the early stages of training, can lead to instability and exacerbate the entropy collapse phenomenon, as the policy might quickly learn to avoid these high-variance, uncertain regions. This is fundamentally at odds with the authors' stated goal.
    *   The authors need to address this potential contradiction. How does DCPO avoid the pitfalls described in [3]?
    *   To substantiate their claims, the authors should provide an entropy evolution comparison with the method from [3]. Furthermore, an ablation study showing the entropy curves for DCPO with different initial clipping values would be highly informative to understand its sensitivity and dynamics.

### Questions
1.  Could you please elaborate on the key differences between your dynamic-adaptive clipping and the high-entropy token handling mechanism described in "Beyond the 80/20 rule" [4]? An empirical comparison would be most convincing.
2.  The core idea of DCPO is to encourage exploration of low-probability tokens. However, work like "The entropy mechanism of reinforcement learning..." [3] suggests this can accelerate entropy collapse early in training. How does DCPO's design, particularly the SAS component, mitigate this risk? Could you provide an entropy curve comparison against the method in [3] to demonstrate this?
3.  Can we compare the main experiments with other entropy-related works, such as [1-4] and entropy regularization methods?
4.  In your experiments, what were the exact clipping threshold values used for the baselines (GRPO, DAPO, GSPO)? This information is crucial for a fair comparison, as the performance of these algorithms is highly sensitive to this hyperparameter.


----

**References:**

[1] Proximal policy optimization algorithms

[2] Reasoning with exploration: An entropy perspective

[3] The entropy mechanism of reinforcement learning for reasoning language models


[4] Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning

### Soundness
2

### Presentation
2

### Contribution
2
