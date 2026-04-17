# Mixture-of-Experts Can Surpass Dense LLMs Under Strictly Equal Resource

- Decision: Accept (Oral)
- Scores: 4, 4, 4, 8

## Abstract
Mixture-of-Experts (MoE) language models dramatically expand model capacity and achieve remarkable performance without increasing per-token compute. However, can MoEs surpass dense architectures under strictly equal resource constraints — that is, when the total parameter count, training compute, and data budget are identical? This question remains under-explored despite its significant practical value and potential. In this paper, we propose a novel perspective and methodological framework to study this question thoroughly. First, we comprehensively investigate the architecture of MoEs and achieve an optimal model design that maximizes the performance. Based on this, we subsequently find that an MoE model with activation rate in an optimal region is able to outperform its dense counterpart under the same total parameter, training compute and data resource. More importantly, this optimal region remains consistent across different model sizes. Although additional amount of data turns out to be a trade-off for enhanced performance, we show that this can be resolved via reusing data. We validate our findings through extensive experiments, training nearly 200 language models at 2B scale and over 50 at 7B scale, cumulatively processing 50 trillion tokens. All code and models will be released publicly.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors do a big sweep of MoE hyperparameters under equal resource constraints (parameter count, training data, and data budget fixed). They find the optimal shape for their model, and then they show that, surprisingly, they can outperform the dense baseline under these constraints.

### Strengths
- Many trained models, extreme compute budget
- Almost every important hyperparameter checked
- Their model outperforms the dense baseline

### Weaknesses
- The authors find many surprising results that contradict the literature and intuition. One of these is that K=1 outperforms K>1. While I am not aware of other papers with exactly the same fixed budgets, there are papers examining the effect of K under fixed compute budget, e.g [1]. This finding also contradicts my personal experience. Another surprising point is the multi-epoch training efficiency, also mentioned in the Discussion paragraph from L 426-434. The authors should either discuss or experimentally demonstrate an analysis that explains why this might be the case.
- Clarity: see questions below.
- The dataset and the code don't appear to be public, and a lot of implementation details are missing from the paper. For example what is the non-normalized gate? Sigmoid?
- It's unclear how well the dense baseline is tuned compared to the super well-tuned MoE models.

[1] Zoph et al, 2022: ST-MoE: Designing Stable and Transferable Sparse Expert Models

### Questions
- The authors use a lot of single-letter parameterization. Some of these lack a clear conceptual meaning in the paper. However, for lots of them, there is one: for example, alpha is the expansion factor for the MLP, zeta is the aspect ratio of the model, etc. Explaining this would greatly improve the paper's readability.
- What does it mean to "First, decide the MoE to dense layer ratio (Le vs. Ld) to focus on MoE’s internal benefits."? Specifically, what is focusing on internal benefits?
- What is the non-normalized gate?
- What is the exact architecture? What regularization loss is used? What transformer variant? What activation, what norm, what attention, what positional encoding, etc?

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
The paper investigates the performance of MoE with respect to dense models.
First, an ablation studies are conducted to search for the best architecture.
Then, the paper finds that there is an optimal activation rate at 20%

### Strengths
1. Experiments are conducted at a fairly large scale, 7B model.
2. A thorough investigation on the model architecture is performed. 
3. Comparing MoE and dense models by repeating data used to train MoE is new and is a good practice for future work which compare MoE with dense models.

### Weaknesses
1. Some of the architecture search study is not very convincing. For example, it is mentioned that larger K hurts performance and hence K=1 is chosen, but in Table 6 K=1 has worse performance. Also, only K=1 and K=11 are tested with but not more fine-grained values like K=4,8 etc. Hence, it is not obvious K=1 is the best choice.
2. Some of the design choices are quite unconventional, e.g., one dense layer followed by MoE layers and K=1, while popular public models like Qwen, Mixtral use full MoE layers and K larger than 1. This could be due to the fact that the specific greedy architecture search done by the paper leads to such a design choice, and hence the result may not be generalizable well.
3. Many previous MoE studies [2501.12370,2502.05172,2502.03009] provide scaling laws which are concrete guidelines for scaling MoEs. No such studies are performed, which may limit the usefulness of the paper.
4. As a result, this paper sounds like a whitepaper that merely reports experimental results instead of providing findings that are useful to the community and in a more scientific way.

### Questions
1. Could you explain in more detail how you used hyperparameter scaling laws proposed in (Li et al., 2025) (line 208)? Did you re-fit the scaling laws with your data? I could not find the scaling law parameters in the Appendix or elsewhere.
2. Ablation studies in [2409.02060] show that shared experts are not effective. Do you have any idea what leads to this different conclusion?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper shows that MoE can surpass dense Transformers using the same resources for training. A similar result has been shown in one of the recent papers cited in the Related Work section, where the models were matched in total parameters and training compute. In addition to this setup, in Section 6 the submission authors consider an additional modification: to match the number of training tokens used, they train the MoE model for multiple epochs (so the MoE dataset size is still technically larger, but the number of _unique_ tokens seen in training is matched). Furthermore, the authors present a set of ablations on optimally setting the MoE training hyperparameters.

### Strengths
1. Large scale of experiments.
2. Evaluation both based on BPC and on downstream tasks (these two types of evaluation can behave differently for MoE and dense Transformers [2]).
3. Evaluation both in pretraining and SFT.
4. The authors declare releasing all code and models.
5. A considerable set of ablations and studies on setting the MoE hyperparameters.

[2] Jelassi et al., Mixture of Parrots: Experts improve memorization more than reasoning

### Weaknesses
1. Reproducibility: the authors do not disclose the dataset used in the paper, which limits reproducibility.
2. Please also see Questions to the authors, where I placed my technical concerns/questions regarding the methodology.

### Questions
1. Question about optimal sparsity. In the methodology for deriving optimal sparsity, the authors fix the model size and compute budget, and perform a grid across sparsity levels. To keep the compute constant, the authors **modify the dataset size**. Therefore, the observed optimum may just be an effect of seeing **the optimal token to parameter ratio** for the given compute budget. To actually determine the optimal sparsity level for a given FLOPs budget, the paper should contain a two-dimensional sweep over both token-to-param ratios **and** sparsity levels. Are the authors able to provide such experiments or explain/clarify? This is an important technical concern regarding methodology and paper conclusions.
2. Based on the literature, scaling laws coefficients can change depending on the data distribution [3]. The submission authors set lr and batch size based on hyperparameter scaling laws from [4]. While [4] concludes that their results are robust across data distributions, did the authors perform an ablation on a small scale to make sure these optimal hyperparameters are unchanged? If yes, such plot should be placed in the paper. 
3. Question about gate score normalization. Based on Table 5, the authors conclude that normalized/not normalized router logits result in a similar loss. However, In the setups the Top-K used is relatively small (2 or 4). Intuitively, with more granular models, where there are more experts activated (like in Table 6: top-K=33, experts=88), the importance of gate normalization could be more pronounced. Did the authors consider such setup? E.g. comparing the model from the last row of Table 6, but trained with/without gate score normalization.

[3] Brandfonbrener et al., Loss-to-Loss Prediction: Scaling Laws for All Datasets

[4] Li et al., Predictable Scale: Part I, Step Law -- Optimal Hyperparameter Scaling Law in Large Language Model Pretraining

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates whether MoE models can outperform dense models at the same param and compute budget. The authors performed comprehensive experiments to determine the optimal activation rate, and showed MoE model outperforming their dense counterparts in controlled condition.

### Strengths
- This work is done in a very principled way. The authors carefully planned for the steps to determine the design decision, and tested the main question with numerous experiments.
- The result leads to a surprising and potentially impactful conclusion, MoE can outperform dense model at the same parameter scale.

### Weaknesses
- The results on some datatsets, e.g. GSM8k, MMLU, is surprisingly low, when compared to other open sourced models at 7B scale, e.g. LLAMA2. If the quality of fineweb edu is not too bad, have the models have been sufficiently trained? Have the evaluation followed best practices?

### Questions
- Same as weakness.

### Soundness
2

### Presentation
3

### Contribution
3
