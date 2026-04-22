# Latent Veracity Inference for Identifying Errors in Stepwise Reasoning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Chain-of-Thought (CoT) reasoning has advanced the capabilities and transparency of language models (LMs); however, reasoning chains can contain inaccurate statements that reduce performance and trustworthiness. To address this, we propose to augment each reasoning step in a CoT with a latent veracity (or correctness) variable. To efficiently explore this expanded space, we introduce Veracity Search (VS), a discrete search algorithm over veracity assignments. It performs otherwise intractable inference in the posterior distribution over latent veracity values by leveraging the LM's joint likelihood over veracity and the final answer as a proxy reward. This efficient inference-time verification method facilitates supervised fine-tuning of an Amortized Veracity Inference (AVI) machine by providing pseudo-labels for veracity. AVI generalizes VS, enabling accurate zero-shot veracity inference in novel contexts. Empirical results demonstrate that VS reliably identifies errors in logical (ProntoQA), mathematical (GSM8K), and commonsense (CommonsenseQA) reasoning benchmarks, with AVI achieving comparable zero-shot accuracy. Finally, we demonstrate the utility of latent veracity inference for providing feedback during self-correction and self-improvement.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper focuses on identifying stepwise error in Chain of Thoughts (CoT) of language models. To address this problem, the paper formulates the problem as a latent variable model, and proposes two methods: veracity search (VS) and Amortized Veracity Inference (AVI) to solve the problem.

### Strengths
1. The proposed method provides significant improvement over the stepwise error detection performance. 
2. The inference time cost of the proposed AVI method is relatively low.

### Weaknesses
1. The experiments are done with synthetic errors. I'm curious about how the proposed methods work for "real" errors in model's CoT. Can the AVI method detect and correct real CoT errors thus improve the reasoning ability?

### Questions
1. Can the empirical study in appendix C.6 be extended to other models/datasets? I think the hypothesis here is actually key reason why the proposed methods work: the model actually "knows" the error and assign higher probability to the joint distribution of answer and veracity, if it's closer to ground truth. Providing more empirical evidence on this will strongthen the conclusion.

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
This paper presents an approach to disentangle the content of a chain-of-thought (CoT) from its veracity/correctness. The authors introduce veracity search (VS), which is a metropolis search methodthat samples veracity assignments for reasoning steps using the LM’s joint likelihood as a reward, and amortized veracity inference (AVI), which fine-tunes an LM on pseudo-labels from VS to enable zero-shot veracity prediction. The idea of treating correctness as a latent variable is both intuitive and novel, and the experiments convincingly demonstrate gains over prompting-based verification baselines. The paper is well written and includes extensive experiments across logical, math, and commonsense reasoning.

### Strengths
1. **Conceptual clarity and originality**: The paper disentangles reasoning content and correctness via a latent variable formulation. It provides a neat probabilistic framing of step-wise error identification in chain of thoughts.
2. **veracity search is intuitive**: and effective inference-time algorithm. It  outperforms simple prompting-based verifiers across different benchmarks. 
3. **Comprehensive evaluation.** The paper is well written, and the experiments are well-organized, and includes detailed ablations (e.g., simulated annealing, greedy initialization, scalability with reasoning hops on prontoQA)..

### Weaknesses
1. **Reliance on prompting LMs for veracity scoring:** The entire framework assumes that the LM can reliably evaluate joint likelihoods of veracity assignments, yet prior studies (e.g., Huang et al., 2023; Zhang et al., 2024) show that LMs are often poor self-verifiers, especially on real-world reasoning where correctness is tricky/subtle. Some discussion or empirical evidence of robustness on naturally occurring reasoning errors would strengthen the claims.

2. **Lack of comparison to process reward models (PRMs).** There exist strong baselines such as PRMs (Lightman et al., 2024; Cobbe et al., 2021) that explicitly model step-level correctness and can perform veracity inference without search. This begs the question of why should VS/AVI be preferred over training or using a PRM?
PRMs would likely achieve comparable or better results with far fewer LM forward passes and a clearer training signal.

3. The search process requires up to 200 LM forward passes per example, making it orders of magnitude more expensive than single-pass prompting baselines so the comparison to these is unfair in my opinion. While AVI mitigates this at inference, the paper should discuss the practical feasibility and compute trade-offs, especially when compared to PRMs or trained verifiers.

4. **Limited practical applicability:** Many of the ablations/analysis focus on synthetic or structured boolean reasoning over prontoQA. It remains unclear whether the method can handle open-ended or natural reasoning errors (e.g., in mathematical or real-world proofs). The negation-based correction strategy also seems tied to synthetic logic tasks.

### Questions
See above

### Soundness
3

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
3

### Summary
This paper proposed augmenting each CoT step with a latent correctness variable and a search method, Veracity Search to infers step correctness by maximizing the language model’s joint likelihood of veracity and the final answer

### Strengths
1. strong performance gain
2. label-efficient step verification, the proposed veracity search use LLM's joint likelihood, avoiding expensive step-level supervision.

### Weaknesses
1. Most tests use artificially corrupted chains; evidence on naturally occurring errors is limited and needs broader experiments.
2. The joint-likelihood reward correlates with true veracity but not perfectly (Pearson 0.56–0.74), so misrankings can occur.
3. the computation efficiency can be improved, strong VS performance often use tens to 100 samples, this add-on computation cost compared with single-pass verifiers.

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
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the problem that chain-of-thought (CoT) reasoning in LLMs often contains flawed intermediate steps, which undermines both accuracy and interpretability. The authors propose to augment each reasoning step with a latent veracity variable indicating correctness. They introduce Veracity Search (VS), a discrete search algorithm that leverages the joint likelihood of veracity assignments and final answers to approximate posterior inference. To avoid reliance on ground-truth answers at test time, they further propose Amortized Veracity Inference (AVI), which is trained on pseudo-labels generated by VS and enables zero-shot veracity prediction.

### Strengths
(1) Originality: The methods proposed for improving the CoT are innovative.

(2) Quality: The combination of VS and AVI is well-motivated, and experiments are carefully designed across logical, mathematical, and commonsense reasoning tasks.

(3) Clarity: The paper provides clear definitions, and comprehensive experiments.

(4) Significance: Identifying and correcting reasoning errors is an important challenge for improving the reliability of LMs, and this work provides a promising direction.

### Weaknesses
(1) Many experiments rely on artificially corrupted reasoning chains. It would be valuable to see more extensive evaluation on naturally generated CoTs (this is the real-world use case).

(2) The research on the impact of reasoning time is somewhat lacking.

(3) The AVI is dependent on VS, but VS itself may not able to guarantee accuracy. This influence is not analyzed.

### Questions
(1) How well does the method generalize to naturally occurring errors in CoTs, beyond the controlled corruption schemes?

(2) Could the authors provide more discussion on the computational cost compared to baselines, especially for longer reasoning chains? The influence on reasoning time is not known.

(3) How do you think your proposed method will perform on larger or smaller LLMs? Your experiment only involves LLMs of approximately 4B or 8B.

### Soundness
3

### Presentation
3

### Contribution
3
