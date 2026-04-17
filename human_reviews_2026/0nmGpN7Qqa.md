# PiCSAR: Probabilistic Confidence Selection And Ranking for Reasoning Chains

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 6, 4

## Abstract
Best-of-$n$ sampling improves the accuracy of large language models (LLMs) and large reasoning models (LRMs) by generating multiple candidate solutions and selecting the one with the highest reward. The key challenge for reasoning tasks is designing a scoring function that can identify correct reasoning chains without access to ground-truth answers. We propose Probabilistic Confidence Selection And Ranking for Reasoning Chains (PiCSAR): a simple, training-free method that scores each candidate generation using the joint log-likelihood of the reasoning and final answer. This method utilises both the scores of the reasoning path (_reasoning confidence_) and the final answer (_answer confidence_). PiCSAR achieves substantial gains across diverse benchmarks ($+11.7$ on AIME2024, $+9.81$ on AIME2025), outperforming baselines with fewer than at least 2x samples in 20 out of 25 comparisons. Our analysis reveals that correct reasoning chains exhibit significantly higher reasoning and answer confidence, justifying the effectiveness of PiCSAR.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces PiCSAR, a training-free best-of-n reranking method that scores each candidate reasoning chain by the joint log-likelihood of the chain and its final answer, combining “reasoning confidence” log p(r | x) with “answer confidence” log p(y | r,x).  PiCSAR operationalizes this with a simple sampling-and-scoring algorithm that extracts log-probabilities from model generations and selects the highest-scoring pair, and also presents a length-normalized variant (PiCSAR-N).  Across diverse LLMs/LRMs and benchmarks—including AIME 2024/2025 and GPQA-Diamond—PiCSAR consistently outperforms self-consistency baselines while being sample-efficient (often k=6 beating k=16/32).  Analyses show that correct solutions cluster where both reasoning and answer confidence are high, that confidence is predictive within but not comparable across models, and that answer-confidence estimation can be decoupled from generation to reduce compute.  Overall, PiCSAR offers a principled, portable scoring rule that reliably selects higher-quality reasoning chains without extra training.

### Strengths
1. The empirical results are comprehensive: experiments on multiple mainstream datasets and models strongly support the authors’ claims.
2. The paper is well written and easy to follow.

### Weaknesses
1. The core idea appears incremental. As the authors note (Lines 82–84), USC is a “majority consensus pattern,” and PiCSAR selects Best-of-N using “probabilistic confidence”. However, prior work has already leveraged model-intrinsic probabilistic confidence for scoring and selection [1–3]. While PiCSAR uses a different formulation, the paper does not clearly establish—theoretically or empirically—why this formulation is superior to [1–3].
2. The gains are modest. In Table 1, PiCSAR’s improvements over baselines are mostly <1 percentage point across many settings. Even though the authors average across multiple runs, it remains unclear whether the extra answer-extraction step—and its additional compute—offers a favorable trade-off for such small gains.

[1] Kang, et al. Scalable Best-of-N Selection for Large Language Models via Self-Certainty. ArXiv.

[2] Taubenfeld, et al. Confidence Improves Self-Consistency in LLMs. ACL 2025.

[3] Wang, et al. Soft Self-Consistency Improves Language Model Agents. ACL 2024.

### Questions
1. For Gemma-2-9B-Instruct on MATH500 with k=6, shouldn’t the best score be p(True) (46.87) rather than PiCSAR (46.53)?
2. In Figure 3, why do many points have Y=-10 for Q3 and Q4?
3. In Figure 3, how do you operationally define “high” vs. “low” confidence for each point?
4. Lines 411–422: why does a larger scale yield a lower PiCSAR score on Qwen3?
5. How does PiCSAR’s performance vary with the number of samples and with temperature?

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
4

### Summary
This paper proposed a method for incorporating reasoning confidence into the score aggregation process during BoN (Best-of-N) sampling in LLMs. The authors argue that this approach can improve performance at test time scaling with minimal additional computational cost. To support their claim, they first conduct a toy experiment showing a strong positive correlation between accuracy and reasoning confidence. They then demonstrate consistent performance improvements across various LLMs and LRMs on multiple benchmarks. Moreover, the paper provides additional analyses on different scenarios where reasoning confidence is utilized, highlighting the robustness of the proposed method.

### Strengths
The proposed method shows consistent improvements over the conventional BoN sampling approach, as evidenced by Tables 1 and 2. The results are impressive in that the improvements hold across different model sizes, reasoning and non-reasoning models, and a variety of benchmarks. In addition, the proposed scoring function clearly articulates the motivation for aggregating reasoning-related components in reasoning tasks and provides a well-structured decomposition. Given that the method achieves measurable gains with almost no additional computation, it represents a meaningful contribution in the context of growing interest in test-time scaling.

### Weaknesses
Some potential concerns and limitations remain: 
1. $\textbf{Generation length bias}$: The proposed method does not perform length normalization for reasoning sequences. As shown in the appendix, applying length normalization weakens the improvement trend. This aligns with recent observations that longer reasoning chains can often lead to worse performance. However, without normalization, the method might inherently favor shorter reasoning paths, thereby reproducing a bias similar to prior works that benefit from shorter reasoning. This bias seems indirectly observable in Table 3, particularly in the Avg Sentences metric. 
2. $\textbf{Experimental clarity}$: The experimental setup lacks some clarity. While both Tables 1 and 2 present comparisons between the proposed method and baselines, Table 1 includes multiple methods whereas Table 2 only compares against Self-Consistency. Since test-time scaling plays a particularly critical role in reasoning, a more detailed comparison in Table 2 would strengthen the claims. Additionally, the maximum length used during inference is not specified, an important parameter in reasoning tasks that should be explicitly reported.

### Questions
1. Reasoning sentences typically consist of hundreds of tokens, whereas answers are much shorter (tens of tokens), which may lead to scale differences between the decomposed scores. Does such a scale mismatch occur? It would be informative to visualize the distributions of the two score components.
2. What is the correlation between score and generation length? If the correlation is high, how does performance compare to a simple heuristic that ranks responses by length with the answer score?
3. What was the maximum length used during inference in your experiments?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper authors propose PiCSAR (Probabilistic Confidence Selection and Ranking): a training-free, probabilistic scoring method to improve reasoning accuracy in LLMs. Instead of relying on majority voting (like Self-Consistency) or external reward models, PiCSAR uses the model’s own token-level probabilities to estimate reasoning confidence (log p(r | x)) and answer confidence (log p(y | r, x)), to rank and select generations based on joined logprobability log p(r | x) + log p(y | r, x). 
In their experiments, authors show that PiCSAR achieves higher accuracy than Self-Consistency, Self-Certainty, and other baselines with fewer samples.

### Strengths
- Paper is clear and well written
- Authors propose a novel training-free probabilistic joint-likelihood framework to score and select reasoning chains 
- Authors show consistent performance gains across five LLMs and three LRMs on different benchmarks
- The paper compares against six major baselines, including Self-Consistency, Universal Self-Consistency, Self-Certainty, etc, showing systematic improvements
- Proposed method can be beneficial for reaseachers and practitioners, as it can easily be integrated into existing inference frameworks without additional compute for training or fine-tuning

### Weaknesses
1. Generalization: it is unclear whether the approach extends to open-ended reasoning, commonsense, or multi-hop text-based inference tasks where “final answers” (and thus answer confidence) are less well-defined.
2. Ablation: paper briefly mentions length-normalized variant, but don't provide any ablation on the reasoning lengths.

### Questions
Joint likelihood approximation implicitly assumes independence between reasoning correctness and answer correctness given the log-likelihoods. But during generation model conditioned answer tokens on reasoning chains. How do you justify the decomposition?

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
2

### Summary
The paper introduces PiCSAR, a training-free, probabilistic confidence-based scoring method for best-of-n (BoN) reasoning chain selection in large language and reasoning models. PiCSAR scores candidate solutions by decomposing the joint log-likelihood of the reasoning trace and final answer into reasoning confidence and answer confidence components. The paper provides extensive empirical evaluation across several LLM/LRM families and challenging benchmarks, demonstrating that PiCSAR is both sample-efficient and outperforms established baselines  with fewer samples. Analyses highlight the utility and interpretability of the confidence signals, including novel perspectives via information planes and correlation between score dynamics and answer correctness.

### Strengths
1. PiCSAR appears to be straightforward to implement, requiring neither additional training nor an external reward model. Moreover, it can be seamlessly integrated into the reasoning workflow of any LLM or LRM, which highlights its practical utility and potential for broad adoption.

2. The manuscript is clearly written and presents well-structured arguments. The effectiveness of PiCSAR is convincingly demonstrated through extensive experimental validation. Furthermore, the use of an information plane visualization offers an interpretable perspective that enhances the reader’s understanding of the method’s underlying mechanisms.

### Weaknesses
1. The experimental evaluation primarily focuses on benchmarks where reasoning and answer extraction are relatively well-defined (e.g., mathematical reasoning, structured question answering). The generalization capability of PiCSAR to tasks involving open-ended question generation, instruction following, or scenarios with high answer uncertainty has not been examined.

2. The implementation of answer confidence relies on an “explicit instruction prompt” and specific answer extraction rules, which may introduce prompt dependency and parsing fragility.

### Questions
1. The manuscript employs joint log-likelihood as the re-ranking criterion. However, this approach may lead to the loss of a considerable amount of sequence information, which raises the question of whether such loss could influence the selection of the optimal reasoning chain. Could the authors elaborate on how PiCSAR mitigates this potential issue, and in particular, clarify why a straightforward summation of reasoning confidence and answer confidence can effectively distinguish the correct reasoning chain?

2. In addition to mathematical reasoning, has PiCSAR been evaluated on open-domain reasoning tasks? It would be valuable if the authors could further discuss the potential applicability of PiCSAR in other contexts, as well as any inherent limitations that may arise in such scenarios.

### Soundness
3

### Presentation
3

### Contribution
3
