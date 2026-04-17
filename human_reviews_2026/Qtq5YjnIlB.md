# Reasoning Self-Evaluation via Trajectory Dynamics Modeling

- Decision: Reject
- Scores: 2, 8, 4

## Abstract
Large Reasoning Models (LRMs) enhance the reasoning ability of LLMs by explicitly generating chain-of-thought (CoT) trajectories before producing answers. Yet, how to enable LRMs to self-evaluate the correctness of their outputs remains largely unexplored, posing challenges for reliability. Existing approaches often rely on external supervision—such as ground-truth labels, auxiliary tools, or additional training—which largely reduces their applicability in real-world scenarios. 
In this work, we show that reasoning trajectories revealing the intermediate dynamics can be leveraged for self-evaluation, allowing models to self-assess the reliability of the reasoning without external supervision. We propose the $D^3$ Reasoning Score, a trajectory-based metric that quantifies output correctness built with three complementary components: Distance ($D_1$), Dynamism ($D_2$), and Difficulty ($D_3$), each of which can be derived purely based on internal signals of the model. Jointly modeling these factors yields a single scalar score aligned with output reliability, providing a fine-grained, white-box measure of reasoning fidelity.  
Extensive experiments across eight open-source LRMs and seven widely used benchmarks—covering both language and multimodal tasks—demonstrate that $D^3$ consistently outperforms prior baselines in AUROC, AUPR, and FPR@95. These results establish $D^3$ as a principled and practical metric for systematic self-evaluation and diagnosis of reasoning trajectories.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes the D³-Reasoning Score, a metric designed to evaluate the reliability of reasoning in Large Reasoning Models (LRMs) by analyzing their internal hidden-state dynamics. D³ combines layer-wise representation changes (D₁, D₂) and predictive uncertainty of reasoning tokens (D₃) to enable white-box self-evaluation of reasoning quality.

### Strengths
Strengths

- This paper is well written and easy to follow.

- The motivation—to enable white-box, self-evaluative reasoning in large reasoning models—is well presented and addresses an important gap in current interpretability and reliability research. The authors convincingly argue for the need to go beyond external evaluation toward internal reasoning analysis.

- The paper provides empirical evidence that the proposed D³-Reasoning Score improves self-evaluation robustness and correlates with reasoning reliability across multiple benchmarks.

### Weaknesses
### Weaknesses

- **Questionable theoretical grounding of D³ components.**  
   The layer-wise metrics (**D₁**, **D₂**) assume that transformer depth corresponds to reasoning progression, which is conceptually inconsistent since layers represent representational hierarchy rather than temporal reasoning steps. As a result, these measures likely capture architectural transformations or activation scaling rather than true reasoning dynamics.

- **Ambiguity in defining the predictive entropy term (D₃).**  
   Best of my understanding, the definition of **D₃** relies on log-probabilities of “ground truth” tokens during the reasoning phase, but no such supervision exists within the `<think>` segment, especially for the test dataset which we cannot access to the ground truth. Also, averaging over the entire trajectory penalizes self-corrective reasoning patterns (e.g., “aha” moments) and conflates transient uncertainty with task difficulty.

- **Lack of normalization and inconsistent scaling across components.**  
   The final score, \( D^3 = D_1 + D_2 - D_3 \), combines quantities with incompatible units—hidden-state L2 norms and log-probabilities—without normalization or weighting. This linear aggregation lacks theoretical justification and likely reflects empirical tuning rather than principled reasoning modeling.

- **Potential over-interpretation of empirical trends.**  
   The observed correlation between D³ and dataset difficulty may result from incidental activation geometry rather than intrinsic reasoning reliability. The metric appears heuristic and architecture-dependent, raising doubts about its generalizability beyond the reported experiments.

- **Unbalanced and partially unreliable presentation of experimental results.**  
   In several result tables, the authors only highlight cases where the proposed model ranks first or second, omitting markings for baselines that perform equally well or even better. Moreover, standard deviations are not reported, despite many results showing very similar scores across methods. This selective and incomplete reporting weakens the reliability and transparency of the experimental claims.

### Questions
See above section

### Soundness
2

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
4

### Summary
This paper introduces the D3 Reasoning Score, a novel, trajectory-based metric designed for Large Reasoning Models (LRMs) to self-evaluate the correctness of their outputs without external supervision. The authors propose that by analyzing the dynamics of the reasoning trajectory, a model can assess the reliability of its own conclusions. The D3 score is composed of three complementary components derived from the model's internal signals: Distance (D1), which captures progress through first-order differences of hidden states; Dynamism (D2), which measures adaptive trajectory dynamics using second-order differences; and Difficulty (D3), which reflects query difficulty via the entropy of the final predictive distribution. The paper presents this as a white-box, fine-grained measure of reasoning fidelity and validates it across a wide range of models and benchmarks.

The paper presents a novel and well-motivated method for self-evaluation of reasoning models, supported by strong and comprehensive experimental results that show significant improvements over existing baselines. While there are some clarity issues regarding the central technical approach, these seem addressable through revisions. The paper's contribution is valuable to the community.

### Strengths
*   **Novelty and Importance:** The paper addresses the important and challenging problem of self-evaluation for large reasoning models, which is crucial for their reliability. The proposed D3 score is a novel approach that leverages internal model dynamics.
*   **Solid Experimental Validation:** The authors conduct extensive experiments across eight open-source LRMs and seven benchmarks, including both language and multimodal tasks. This comprehensive evaluation provides strong evidence for the method's effectiveness.
*   **Significant Results:** The D3 score is shown to consistently outperform existing baselines across multiple standard metrics (AUROC, AUPR, and FPR@95). The results appear robust and significant.
*   **Good Writing and Structure:** The paper is generally well-written and easy to follow.

### Weaknesses
The central weakness of the paper is the potentially misleading use of the term "trajectory dynamics." For instance, the paper claims that "For each query, an LRM produces both a final answer and a CoT reasoning trajectory. By analyzing hidden-state dynamics, the model assigns a self-evaluation score." This suggests an analysis of the reasoning process as it evolves over time. The authors also state they collect hidden representations into a `(K, L, d)` tensor, where `K` is the number of reasoning tokens and `L` is the number of layers. This setup presents two natural choices for an "evolution" parameter: `K` (token-wise evolution) or `L` (layer-wise evolution). The paper's use of "trajectory dynamics" suggests an analysis of the state as it evolves along the token sequence (`K`). However, the proposed method chooses `L` as the evolution parameter and reduces the state space from `(K,d)` to `d` by averaging over the `K` dimension. This choice is not clearly justified and creates confusion about what "dynamics" are actually being modeled.

### Questions
1.  Could you provide a more detailed justification for the mean pooling strategy over the `K` token representations? The claim that this "preserves the core semantics without information loss" is a strong one. Can you elaborate on what "core semantics" are being preserved and why averaging does not lead to a loss of critical information contained in the individual token representations?
2.  Could you please clarify the main motivation for analyzing dynamics along the layer dimension (`L`) after averaging over the token dimension (`K`)? What is the intuition that this approach is more informative for self-evaluation than analyzing the dynamics of the state representations along the token sequence (`K`)?
3.  Did you experiment with using `K` as the primary evolution parameter (i.e., studying the dynamics of the `(L, d)` state as the reasoning trajectory unfolds)? If so, how did those results compare to the layer-wise analysis presented in the paper?
4.  There is a minor inconsistency in terminology, where D2 is referred to as "Dynamism" in the abstract and main text, but "Detour" in Figure 2.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a method for Large Language Models (LLMs) to perform label-free self-evaluation of their generated reasoning paths by analyzing the hidden state trajectory of the reasoning tokens. The authors track the average change in hidden vectors for reasoning tokens at each layer, calculating their 1st and 2nd derivatives (norms) to measure the model's "progressiveness" (termed distance) and "adaptiveness" (termed dynamism). This is combined with an entropy-based difficulty metric derived from the output distribution, which serves as a correction term. The final D3 score = (distance + dynamism − difficulty) is defined to integratively reflect the "progression, adjustment, and uncertainty" of the model's self-generated reasoning trajectory. This score effectively distinguishes correct from incorrect answers without any separate training. Across various benchmarks (GSM8K, MMLU-Pro, MGSM) and LLM scales (1.5B–70B), the D3 score demonstrated higher AUROC and AUPR, as well as lower FPR@95, compared to existing uncertainty-based metrics (e.g., MSP, PPL, CoE). The results validate that the dynamical properties of the reasoning process itself are a valid signal for reliability evaluation.

### Strengths
The paper's core strength is its empirical demonstration that the correctness of reasoning can be determined solely by the dynamics of the LLM's hidden states. In other words, without relying on output probabilities or external labels, the study analyzes the rate of change (1st derivative) and adaptiveness (2nd derivative) of layer-by-layer hidden representations during the reasoning process, effectively its trajectory dynamics. The authors discovered that correct and incorrect reasoning traces exhibit distinct patterns within the internal representation space. This provides a structural basis for the idea that "correct reasoning follows a consistent pattern of progression and adjustment in the hidden space." The study's primary contribution is revealing that these hidden dynamics themselves serve as an intrinsic signal that reflects reasoning correctness.

### Weaknesses
1. **Missing Comparison with Prior Hidden-State-Based, Label-Free Methods**
  
This paper proposes a novel self-evaluation framework that classifies reasoning correctness by analyzing the dynamics of hidden states. However, similar white-box and label-free approaches have already been explored, yet were not included as baselines. In particular, Xie et al. [1] leverage the internal consistency of hidden representations to estimate model confidence, while Bigelow et al. [2] identify forking tokens within reasoning trajectories to trace and correct reasoning errors. Including such prior hidden-state-based and label-free methods as baselines would strengthen the empirical and conceptual positioning of this work.

2. **Ambiguity in the Definition of “Difficulty” and Practical Applicability**
 
The proposed D3 score is defined as “distance + dynamism − difficulty,” but the definition of the difficulty term remains ambiguous. The paper describes it as the average negative log-likelihood of the “correct token,” yet it is unclear what constitutes a correct token during intermediate reasoning stages. It is not specified whether this refers to teacher-forcing–based NLL or a self-generated NLL using the model’s own outputs. 


[1] *Calibrating Reasoning in Language Models with Internal Consistency*, NeurIPS 2024

[2] *Forking Paths in Neural Text Generation*, ICLR 2025

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3
