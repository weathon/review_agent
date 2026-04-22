# CTC-DRO: Robust Optimization for Reducing Language Disparities in Speech Recognition

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Modern deep learning models often achieve high overall performance, but consistently fail on specific subgroups. Group distributionally robust optimization (group DRO) addresses this problem by minimizing the worst-group loss, but it fails when group losses misrepresent performance differences between groups. This is common in domains like speech, where the widely used connectionist temporal classification (CTC) loss not only scales with input length but also varies with linguistic and acoustic properties, leading to spurious differences between group losses. We present CTC-DRO, which addresses the shortcomings of the group DRO objective by smoothing the group weight update to prevent overemphasis on consistently high-loss groups, while using input length-matched batching to mitigate CTC's scaling issues. We evaluate CTC-DRO on the task of multilingual automatic speech recognition (ASR) across five language sets from the diverse ML-SUPERB 2.0 benchmark. CTC-DRO consistently outperforms group DRO and CTC-based baseline models, reducing the worst-language error by up to 47.1% and the average error by up to 32.9%. CTC-DRO can be applied to ASR with minimal computational costs, and, while motivated by multilingual ASR, offers the potential for reducing group disparities in other domains with similar challenges.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes an algorithm for ASR training on datasets composed of multiple groups such as languages. The contributions are two fold: the authors enforce balanced minibatches with equal total sample length per group (by audio duration), and they introduce a novel update rule that reweights updates according to each group’s loss contribution. An ablation study shows that both components are necessary. In multilingual settings, the method yields substantial improvements on the worst-performing languages and on the overall average compared with the baseline and group DRO. Performance on language identification is more mixed.

### Strengths
The paper has several strengths:

-  Results: consistent improvements on the worst-performing languages and on the overall average versus both the baseline and group DRO.

- Practicality: the method is lightweight, adds minimal complexity, and does not introduce notable computational overhead; useful for real usage.

- The experiments are concise with a proper ablation study showing that both contribution matter, plus complementary scaling experiments on group size.

### Weaknesses
Despite the overall good results, the paper has some weaknesses :

- The results on (language identification) LID are mixed; significance tests would help determine whether the outcomes are due to randomness or indicate genuine advantages of the baseline, group DRO, or CTC-DRO.

- The effect of the procedure on the best-performing language groups is not discussed. Appendix results appear to suggest baseline ≥ CTC-DRO ≥ group DRO. This issue matters in practice and should be analyzed explicitly. I don’t view this as a major flaw (CTC-DRO often outperforms group DRO) but the results are somewhat mixed, and a statistical analysis would help clarify the picture.

- The choice of Equation (10) is not clearly justified and seems motivated by the form of the resulting update. A clearer rationale for this specific form would help readers build intuition and extend the method; otherwise, it may be cleaner to start directly from the group DRO objective (Equation (7)) and propose the extension explicitly using the already existing rationale in the paper.

### Questions
On Equation (14), the justification for "proportionality" is, to me, unsatisfactory. The Lagrange multipliers lambda and lambda_g are intermediate quantities used to reach the solution and will depend on L_g. Without making their dependence on L_g explicit, you can’t claim that q_g is proportional.

If we push the computation further: starting from (14), q_g = L_g / (lambda + lambda_g) − alpha, with the constraints sum_g q_g = 1 and sum_g (lambda_g q_g) = 0 (KKT). For the active set A = { g : q_g > 0 }, complementary slackness gives lambda_g = 0. Hence, for g in A, q_g = L_g / lambda − alpha. Summing over g in A, 1 = sum_{g in A} q_g = (sum_{g in A} L_g) / lambda − |A| * alpha, which implies lambda = (sum_{g in A} L_g) / (1 + |A| * alpha). Therefore, for g in A, q_g = L_g / lambda − alpha = [ L_g * (1 + |A| * alpha) / sum_{g in A} L_g ] − alpha. Thus, the relationship is not simple proportionality in L_g; but in L_g /sum_g' L_g'. This doesn’t necessarily weaken the update choice rationale, but the claim should be stated precisely. I may have made a mistake or missed something and my derivation should be checked.

- The actual chosen per-group batch duration (target total audio length) is never stated. What value did you use, and how sensitive are the results to this choice?

- Have you tried the procedure with different group definitions (e.g., dialect, SNR bins, speaker gender/age, domain, or per-language subsets)? If so, how do the results vary?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes CTC-DRO, a robust optimization method for multilingual ASR under the CTC framework. Standard group DRO fails because CTC losses scale with input length and differ across languages, leading to unstable weight updates. CTC-DRO mitigates this by using length-matched batching so group losses are comparable, and introducing a smoothed maximization objective for group weight updates.

### Strengths
1. The paper addresses the language disparities in the multilingual ASR, which is critical.
2. The proposed modifications are simple, well-motivated, and easily adoptable in existing pipelines.
3. Results show consistent and sometimes substantial improvements over both CTC and Group DRO baselines, especially for worst-case languages.

### Weaknesses
1. Experiments focus on small subsets (5–6 languages per set), with only limited scaling experiments. It is unclear how well the method generalizes to large-scale multilingual ASR (50–100+ languages).
2. The paper does not compare with alternative ASR+LID strategies, such as auxiliary CTC objectives [A] or condition-aware SSL representations [B], which directly integrate language identification and also improve the low-resource language performance.
3. Although an ablation study is included, it is limited to one dataset. More extensive analysis across different training settings would better clarify the contribution of each component.

[A] Chen, W., Yan, B., Shi, J., Peng, Y., Maiti, S., & Watanabe, S. (2023, June). Improving massively multilingual asr with auxiliary ctc objectives. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 1-5). IEEE.

[B] Lu, Y. J., Liu, J., Thebaud, T., Moro-Velazquez, L., Rastrow, A., Dehak, N., & Villalba, J. (2024). CA-SSLR: Condition-Aware Self-Supervised Learning Representation for Generalized Speech Processing. Advances in Neural Information Processing Systems, 37, 50126-50151.

### Questions
1. How does CTC-DRO scale to very large multilingual ASR settings (dozens or hundreds of languages)?
2. Are the reported gains consistent across multiple random seeds?
3. Could this approach generalize to other sequence modeling losses (RNN-T, seq2seq)?
4. Could CTC-DRO be extended to code-switching scenarios, where disparities are often more severe [C]?

[C] Liu, H., Garcia, L. P., Zhang, X., Khong, A. W., & Khudanpur, S. (2024, April). Enhancing code-switching speech recognition with interactive language biases. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 10886-10890). IEEE.

### Soundness
2

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
3

### Summary
This paper focuses on the training algorithm of CTC ASR models. The authors improve upon the vanilla Group DRO algorithm to better align with CTC loss (which yields higher loss values for longer sequences and breaks the group weights assignment in Group DRO). Experiments on multilingual ASR confirm the effectiveness of the proposed algorithm.

### Strengths
The proposed algorithm appears to yield strong gains over baselines. Experiments are solid. The presentation is clear enough.

### Weaknesses
The contribution is a bit incremental over an existing algorithm (Group DRO). Also, I have some questions about the compared baselines in this work, see the Questions section below.

### Questions
1. In table 3, the row "- SMOOTH" shows a huge performance drop, even much worse than baseline. Why is that? Isn't "-SMOOTH" reduced to a standard CTC training with similar total durations across all training batches?
2. For the baseline, I think a more reasonable setting is to have a baseline that is a standard CTC trained with some length normalization.  The main observation/motivation of the proposed algorithm is that CTC places high loss on longer sequences, so what if you simply normalize CTC loss by sequence length?

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
The paper targets the problem of cross-language disparities in multilingual ASR when training with CTC. To solve that, they explore using the classical ML method of Group Distributionally Robust Optimization (Group DRO). First, they found out that the method is not effective when directly applied to the CTC loss, as this loss is not comparable across groups. CTC losses are systematically higher for reasons not aligned with downstream error (e.g., input length and irreducible loss differences), thereby hurting worst-group generalization. To fix this, they propose CTC-DRO, which introduces a smoothed maximization objective and implements length-matched batching to stabilize group weights and correct for group-wise mismatches in loss scale. Through empirical evaluation on the ML-SUPERB 2.0 benchmark across multiple language sets and state-of-the-art models (MMS, XLS-R), CTC-DRO demonstrates consistent improvements over baselines and Group DRO, reducing the worst-language error and narrowing language disparities in ASR and LID performance.

### Strengths
- The authors build a clear problem framing, pointing to a concrete issue in ASR, particularly between CTC loss geometry and group DRO weight updates, with a neat derivation showing why standard group-DRO weights can collapse to one group. The solution is well motivated, enough general, and theoretically principled. 


- The present compelling empirical evidence on a credible benchmark (ML-SUPERB 2.0), with DRO-CTC showing consistent improvements on worst-group CER across five language sets and both XLS-R and MMS encoders. 

- The work presents informative ablations and behavior analysis. The group-weight traces help clarify why it works (smoother, more balanced weights, and fewer collapses).


- The work is clearly documented, making it easy to reproduce. Authors promised to release code upon acceptance.

### Weaknesses
- Robustness to hyperparameter alpha: While the paper discusses theoretically and reports practical ranges for alpha, it does not include a systematic sensitivity analysis. A more systematic hyperparameter sensitivity (including ηq and the batch-duration target) and early/late-training stability plots across sets would increase confidence in robustness and tune-free usability. 
- The work is limited to the scope of assuming languages as the sole group, relying on datasets where groups are well-defined and large enough. The work does not address potential unknown/latent groups, despite being briefly mentioned by the authors in the conclusion.
- Baselines: While the paper compares to Group DRO and standard CTC, it does not benchmark against alternative fairness-promoting or loss-calibrated approaches specifically designed for sequence models, despite mentioning them in the related work. The methods may include invariant objectives, variance-regularized DRO, loss calibration across groups, data reweighting, or even group-awre reinforcement learning. The inclusion or at least discussion of such alternatives for this particular case would strengthen the empirical case.

### Questions
- Would the smoothed update help beyond CTC, with autoregressive losses (token-level NLL) where length effects differ?
- Please consider adding a fixed-budget grid over alpha, showing worst-language CER stability, to help establish robustness of the gains to alpha.

### Soundness
3

### Presentation
3

### Contribution
3
