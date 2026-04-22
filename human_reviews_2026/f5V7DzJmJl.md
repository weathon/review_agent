# Alignment from Ranking and Rating Information

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4

## Abstract
The class of direct preference optimization (DPO) algorithms has emerged as a
promising approach for solving the alignment problem in foundation models. These
algorithms work with very limited feedback in the form of pairwise preferences
and fine-tune models to align with these preferences without explicitly learning a
reward model. While the form of feedback used by these algorithms makes the
data collection process easy, its ambiguity in terms of the quality of responses has
significant negative implications, including incentivizing policies that favor out-of-
distribution responses, a phenomenon referred to as likelihood displacement. In this
paper, we study how DPO-style algorithms can leverage additional information in
the form of rating gap, which informs the learner how much the preferred response
is better than the rejected one. We present new algorithms that can achieve faster
statistical rates than DPO in presence of accurate rating gap information. Moreover,
we theoretically prove and empirically show that the performance of our algorithms
is robust to inaccuracy in rating gaps. Finally, we demonstrate the solid performance
of our algorithms in comparison to a number of DPO-style algorithms across a
wide range of LLMs and evaluation benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper identifies a key limitation in Direct Preference Optimization (DPO): by using only binary pairwise preferences, DPO ignores the magnitude of the preference. The authors argue this ambiguity—where DPO cannot distinguish between a pair of high-quality responses and a pair with one good and one bad response —can incentivize policies that favor out-of-distribution responses, a phenomenon termed "likelihood displacement".

To address this, the paper proposes to augment the training data with "rating gap" information, which specifies *how much* a preferred response is better than a rejected one. The authors present three new algorithms that leverage this information:
1.  **Rating DPO (RDPO)** 
2.  **Rating IPO (RIPO)** 
3.  **Maximum-Likelihood-based RDPO (ML-RDPO)** 

The paper provides theoretical guarantees, suggesting these methods can achieve faster statistical rates ("acceleration") when the rating gap information is accurate and maintain performance even when this information is noisy, provided the hyperparameters are tuned correctly. Empirically, the authors demonstrate that RDPO and ML-RDPO outperform DPO and other DPO-style algorithms on the AlpacaEval and ArenaHard benchmarks across several base models.

### Strengths
1. **Well-Motivated Problem:** The paper's core premise is intuitive and sound. DPO's reliance on simple binary preferences clearly discards a rich source of information about response quality. The idea of incorporating the *magnitude* of this preference via rating gaps is a logical and compelling extension.
2. **Principled Algorithm Derivation:** The authors provide principled, step-by-step derivations for their proposed algorithms. RDPO and RIPO are derived by modifying the RLHF objective to include rating information, while ML-RDPO is derived from a maximum-likelihood perspective .
3. **Theoretical Analysis:** The paper includes a theoretical analysis that formalizes the concepts of "acceleration"  and "robustness", providing a useful (if idealized) framework for understanding *why* these methods should be beneficial.
4. **Strong Empirical Performance:** The experimental results in Figures 1 and 2 are solid, showing that RDPO and ML-RDPO consistently achieve higher win rates than DPO and other baselines on the chosen benchmarks.

### Weaknesses
1. **Motivation-Experiment Mismatch:** The paper's primary motivation is to solve the "likelihood displacement" problem, which it claims incentivizes OOD responses. However, the experimental evaluation never measures this phenomenon. The experiments are limited to win-rate comparisons on standard benchmarks. While the method improves win rates, there is no evidence provided to substantiate the claim that it actually mitigates likelihood displacement or improves OOD generalization.
2. **Impractical Theoretical Assumptions:** The theoretical guarantees rely on assumptions that are unlikely to hold in practice.
    * Assumption 1  is particularly strong, as it requires the policy class $\Pi$ (i.e., the LLM) to be expressive enough to contain the optimal closed-form policy $\pi_{\beta}^{*}$ for *any* bounded reward function. The paper does not justify why this would be the case for real-world models.
    * ML-RDPO relies on simplifying assumptions for its derivation, such as the conditional independence between the preference label ($z_i$) and the rating gap ($\Delta_r^i$).
3. **Hyperparameter Tuning and Practicality:** The method's practical utility is a major concern. The theoretical analysis itself provides guidance for setting the crucial hyperparameters $\beta$ and $\beta_1$ based on the ratio $Err_{DPO}(N,\delta)/Err_{\pi_{ref}}(\hat{r})$ . This ratio is fundamentally unknowable in a practical setting, as it requires access to the (unknown) true reward error. The robustness experiments in Figure 3  confirm that performance is highly sensitive to finding the *correct* $\beta_1$ (trust in ratings) , but the paper offers no practical heuristic for setting this value, likely requiring expensive, dataset-specific tuning.
4. **Limited Experimental Scope:** The empirical validation is narrow. All main experiments are conducted using only the `ultrafeedback_binarized` dataset and evaluated on only two benchmarks, `AlpacaEval` and `ArenaHard`. This makes it difficult to know if the observed performance gains are broadly generalizable or an artifact of the specific properties of this dataset.

### Questions
See the weaknesses. Most concerns arise from the disconnection between motivation and evaluation, as well as the practical utility.

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
This paper explores how DPO-style algorithms can leverage rating gap information as an additional signal. The authors propose methods including RDPO and ML-RDPO, which achieve superior statistical rates compared to standard DPO when ratings are sufficiently accurate, while demonstrating robustness to rating noise. Comprehensive experiments across various LLMs and benchmarks show consistent performance gains over existing DPO-based approaches.

### Strengths
1. The proposed methods demonstrate consistent improvements across diverse models and benchmarks.
2. RDPO/ML-RDPO provably achieve faster convergence than DPO under the Bradley-Terry model while maintaining robustness to rating noise.

### Weaknesses
1. While the proposed approach shows promise, the technical novelty could be further clarified. The method appears to draw heavily from distill-DPO and DPO.
2. In the 'Experiments to assess robustness' section, how do other baseline methods perform in terms of robustness?
3. Figure 4 indicates that ML-RDPO still heavily relies on rating information. Furthermore, comparisons with other DPO variants are absent, limiting the assessment of the method's unique advantages.

### Questions
1. Why does ML-RDPO consistently underperform RDPO in most scenarios shown in Figure 2?

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
5

### Summary
This paper studies the usage of rating-gap information (the scalar difference in ratings of two responses) along with pairwise ranking data for LLM alignment. 

The authors derive two main algorithms: Rating DPO (RDPO), which incorporates the rating gap into DPO, and Maximum-Likelihood Rating DPO (ML-RDPO), derived from a joint likelihood perspective. 

Theoretical analyses regarding acceleration and robustness to noisy ratings are given, which i quite liked. It adds a theoretical grounding to the work. Experiments are conducted on Zephyr, Mistral, and Llama-3 base models evaluated on AlpacaEval and ArenaHard.

### Strengths
1. Sound Theoretical backing to Algorithmic Claims

The paper is theoretically sound. Several DPO extensions add heuristic regularization terms. Instead, the authors derive ML-RDPO from a joint maximum likelihood perspective. The statistical bounds provided in Theorem 4.1 serve as a theoretical sanity check, confirming that including **accurate** rating information theoretically improves convergence rates compared to ranking-only methods under ideal conditions. (Note the noise in ratings typically sourced from reward models (Which are stochastic) is standard, hence how much this assumption of rating accuracy holds is questionable).

---

2. The idea of using ratings alongside rankings is certainly a good one. 

The idea of using ratings alongside the rankings is good. But the idea is not new. It was first proposed in InfoNCA in neurips 2024 [1] and then extended into the DPO loss function [2,3]. These are references in this area that would be useful to include in their literature review for completeness.


---

### References

[1] Chen, H., et al. (2024). Noise contrastive alignment of language models with explicit rewards. Advances in Neural Information Processing Systems, 37, 117784–117812.

[2] Sun, S., et al. (2025). Reward-aware preference optimization: A unified mathematical framework for model alignment. arXiv preprint arXiv:2502.00203.

[3] Gupta, T., et al. (2025). Multi-Preference Optimization: Generalizing DPO via Set-Level Contrasts. arXiv preprint arXiv:2412.04628.

### Weaknesses
1. No improvement on Old Baselines:

The proposed method fails to improve on SIMPO for Llama, the strongest baseline that they tested on. The method only shows gains on older models (like Zephyr-7B-beta). 

---

2. Theoretical Robustness to noise

Theorem 4.1 is contingent on **knowing the noise level** to set the hyperparameter $\beta_1$ correctly. In practice, this may lead to the brittleness observed in Appendix F.4, where $\beta_1$ must be tuned by orders of magnitude (e.g., 0.1 vs 0.005) across different models. The theoretical guarantee of robustness hence is contingent on finding an appropriate $\beta_1$ suitable to any new model or dataset, is it not?

---

3. Please consider recent baselines.

The paper reports ~29% Win Rates on Llama-3-8B, and shows equal performance with SIMPO [Note Simpo's own paper shows higher WR% and LC-WR % -- see Table 1 of the paper]. But please see the references below which exceed these numbers. For instance, RSPO [1] reports ~35% *Length-Controlled* win rates, while other recent multi-preference and reference-free approaches [2, 3] have reported win rates exceeding 50% on comparable benchmarks. Furthermore Chen et al.,2024 [4] provide a loss which is close to ML-RDPO. Incorporating these variants may better contextualize the method's true competitiveness.


---

### References

[1] Tang, X., et al. (2025). Game-Theoretic Regularized Self-Play Alignment of Large Language Models. arXiv preprint arXiv:2503.00030.

[2] Gupta, T., et al. (2025). REFA: Reference Free Alignment with Fine-Grained Length Control. COLM 2025.

[3] Gupta, T., et al. (2025). AMPO: Active Multi Preference Optimization for Self-play Preference Selection. ICML 2025.

[4] Chen, H., et al. (2024). Noise contrastive alignment of language models with explicit rewards. Advances in Neural Information Processing Systems, 37, 117784-117812.

[5] Wu, Y., et al. (2025). Self-play preference optimization for language model alignment., International Conference on Representation Learning (Vol. 2025, pp. 91558–91582).

### Questions
*   **Degradation on Llama-3.1:** Why do RDPO and ML-RDPO not improve upon the simpler SimPO baseline on Llama-3.1-8B even though there is access to more information. ideally, given the rating information as well as ranking leads to performance improvement. Is it because of lack of sufficient tuning of the hyperparameters 

*   **Gaussian Assumption:** Theorem 4.2 and the derivation of ML-RDPO rely on the assumption that rating gaps are Gaussian distributed. Can you verify this assumption empirically on your datasets (e.g., UltraFeedback)? I'm wondering if the real-world rating distributions may have some heavy tail-ness that might violate this. For example, a histogram of gaps on UF would be a great contribution to the community using this training setup.

---
### Technical Question

- Simpo paper's own numbers are higher than those reported in your paper (as per my understanding in the same setting). 40%LC and 37% WR see Table 1 of their paper. Any reason for this discrepancy?


---

### Suggestion

*   **Baselines:** Please consider extending your work with more recent algorithmic works in this area to make it more empirically competitive.

### Soundness
3

### Presentation
2

### Contribution
2
