# Theoretically Understanding the Hidden Adversarial Price of Low-Rank Adaptation

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Low-rank adaptation (LoRA) has emerged as a prominent parameter-efficient fine-tuning (PEFT) method for large pre-trained models, enabling strong downstream performance with minimal parameter updates. While LoRA is known to outperform head-only fine-tuning in terms of clean accuracy, its impact on adversarial robustness remains largely unexplored. In this work, and to the best of our knowledge, we present the first theoretical analysis of LoRA’s adversarial robustness, comparing it to that of head-only fine-tuning. We formalize the notion of expected adversarial robustness and derive upper bounds demonstrating that, despite its superior clean performance, LoRA can be inherently less robust than head-only tuning due to the additional degrees of freedom introduced by its low-rank components. We further study the influence of LoRA’s initialization scheme and show that simple changes in the initialization distribution of the low-rank matrix can significantly affect robustness. Finally, we support our theoretical findings with extensive experiments on both vision and language benchmarks under standard adversarial attacks. Our results provide a principled understanding of the trade-offs between parameter efficiency, clean performance, and adversarial robustness in commonly used fine-tuning strategies.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work investigates the adversarial robustness of head-only and LoRA fine-tuning methods. A theoretical upper bound on robustness is derived, and empirical attack results on ViT and LLM fine-tuning demonstrate that the head-only method exhibits stronger robustness than LoRA.

### Strengths
1. Adversarial robustness is an interesting topic as fine-tuning may lead to over-fitting due to small datasets in downstream tasks.

### Weaknesses
1. It is not surprising that head-only tuning is more robust than LoRA. When viewed within the full fine-tuning framework, the head-only approach can be seen as applying sparse gradients (i.e., dropping all weight gradients except those of the head), whereas LoRA applies low-rank but still dense gradient updates. Consequently, head-only tuning imposes a much stronger form of regularization than LoRA.

2. The theoretical bound estimates in Proposition 1 and Theorem 1 appear to be quite conservative. Specifically, for Thm.1 it is possible to have $\\|AB\\|=0$ (i.e., LoRA has no effect on the base model) while $\\|A\\|\cdot \\|B\\|$ is arbitrarily large. For example, consider $A=\begin{bmatrix} a & 0 \end{bmatrix}$ and $B=\begin{bmatrix} 0 \\\\ b^\top \end{bmatrix}$ with $a,b\in \mathbb{R}^n$ having large norms. Similar examples can be constructed for $W^Q W^K$ in Prop.1 as well. There exist much tighter Lipschitz bound estimation methods in the literature, such as [1] for the attention layer and [2] for the MLP layer. 

3. The robustness bound estimation also seems disconnected from the empirical findings reported in this work. As I understand it, the theoretical analysis suggests that fine-tuning under bounded parameter norms may enhance robustness. The argument would be more convincing if supported by corresponding empirical evidence. Recent studies [3, 4] have proposed LoRA variants with explicit norm-bounding guarantees that could serve as relevant comparisons.

[1] Havens et al. Fine-grained Local Sensitivity Analysis of Standard Dot-Product Self-Attention. ICML 2024

[2] Wang et al. On the Scalability and Memory Efficiency of Semi-definite Programs for Lipschitz Constant Estimation of Neural Networks: Scaling the Computation for ImageNet. ICLR 2024.

[3] Bini et al. DeLoRA: Decoupling angles and strength in low-rank adaptation. ICLR 2025.

[4] Wang et al. Norm-bounded Low-Rank Adaptation. arXiv 2501.19050.

### Questions
1. For the bound estimation, why take the maximum operator over the heads (i.e. $\mathrm{max}_h$)?

2. The FFN layer is not introduced, i.e., where does $W_{FFN}$ come from?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the relationship between LoRA fine-tuning strategies and adversarial robustness. The authors formalize the notion of expected adversarial robustness and theoretically show that, compared to head-only fine-tuning, LoRA’s gains in clean accuracy come at the cost of increased vulnerability to adversarial attacks. They further analyze how the initialization scheme impacts LoRA’s adversarial robustness.

### Strengths
1. This paper is well written and clearly structured.
2. The derivations of the proposition and theorems are detailed and well-explained, which facilitates understanding.
3. The theoretical findings are supported by extensive experiments on both vision and NLP tasks.

### Weaknesses
1. Several claims lack proper citations. For example, in Section 1 (Introduction), the sentence starting with “A common approach is to freeze ...” (line 17), and the discussion of evasion attacks in Section 4.1 (Adversarial Robustness) require citations.
2. In the supplementary material (Section B, Proof of Theorem 1), the last sentence:“..., head-only finetuning strategy”, appears to be a typo. It should be the LoRA fine-tuning strategy.
3. The term “consistently” is used when explaining experimental results. However, Table 2 shows that on SST-2 and Yelp Polarity datasets, LoRA outperforms Head-Only for both BERT and GPT-2.
4. In Theorem 2 and Lemma 1, the symbol h is used for both layer and head, which may lead to confusion.
5. The hyperparameters used in different adversarial attacks should be listed in the supplementary material for reproducibility.

### Questions
1. If LoRA is further applied to the K and O projection matrices, would it affect the current conclusions? How would C_1^\prime change in this case? A brief derivation or at least an experiment would be helpful.
2. According to Theorem 1, the rank also influences C_1^\prime. Could the authors provide experimental results exploring the impact of different rank values?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the adversarial robustness of low-rank adaptation (LoRA). The authors use theoretical and empirical results to validate that the model after LoRA is more adversarially vulnerable compared to the model after the head-only tuning. Besides, they show that the parameter initialization with a smaller norm and the smaller scale factor can enhance robustness.

### Strengths
- Provide some good insights about the adversarial robustness after LoRA from theoretical results. The robustness is affected by parameter initialization and the scaling factor.
- Provide comprehensive results using CV and NLP datasets to support the claim.

### Weaknesses
- I am sceptical about the fairness of the comparison between LoRA and head-only fine-tuning. These two types of fine-tuning methods utilize different amounts of trainable parameters for fine-tuning. LoRA naturally introduce more parameters compared to head-only fine-tuning. Thus, LoRA naturally has a larger parameter space that can be threatened by the adversary. Therefore, I think the comparison is somewhat meaningless.
- Regarding the experiments of parameter initialization: The authors should show the different norms of parameters w.r.t. the adversarial robustness. Besides, they should show the compatibility of parameter initialization with AutoLoRA and ADV-LoRA.
- Seems that making the scaling factor smaller is equivalent to making the effect of the LoRA module smaller on the performance according to the Eq of LoRA, which is a trivial idea.

### Questions
Please see the Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper investigate an under-explored region: the relationship between fine-tuning strategies and adversarial robustness.
More concretely, it formalizes the notion of expected adversarial robustness uppper bounds in a theoretical way, and analyze the trade-off of PEFT efficiency and adversarial robustness between two main-stream model tuning method, e.g. LoRA and SFT.
The paper further explores different LoRA initialization schemes aimed at improving robustness.
The demonstrating experiments are extensive, ranging from multiple vision and language benchmarks, but there still lack fundamental settings to fully justify the conclusion, e.g. more loose assumption but not the bounded activation, the introduction of adaptive defense, and the relationship between adversarial robustness and other PEFT methods, such as Prompt Tuning, Visual Prompt Tuning, etc.
Empirical evaluations are extensive across both vision and language benchmarks; however, several fundamental settings remain missing to fully justify the conclusions—for example, relaxing the overly strong bounded-activation assumption, introducing adaptive defense baselines, and comparing against other PEFT methods such as Prompt Tuning or Visual Prompt Tuning.

### Strengths
1. Under a theoretical notion of expected adversarial robustness, this work investigate an overlooked but meaningful perspective, which is the relation between different fine tuning method and their adversarial robustness behavior.
2. The derived upper bound provides a useful insight, and the proposed LoRA initialization derived from it empirically improves adversarial robustness.
3. The concrete initialization scheme could be practically valuable for the PEFT community.

### Weaknesses
1. (Major) The assumption of bounded input is reasonable (line 155), but intermediate activations are neither assumed nor proved to be bounded.
   In particular, line 703 (Appendix A) and line 271 rely on this bounded-activation assumption when deriving the single-layer Transformer’s robustness bound and when generalizing to the multi-layer case, respectively.
   However, for deeper layers, if activations are not guaranteed to be bounded, the proof of Proposition 1 no longer holds for intermediate layers.
   It is therefore encouraged to clarify what precise assumption is used and to justify how intermediate activations could be considered bounded (or to add an explicit limitation discussion).

2. Adaptive defense strategies and other PEFT variants (e.g., Prompt Tuning, Visual Prompt Tuning) could be included for a more comprehensive comparison.

### Questions
As above

### Soundness
3

### Presentation
3

### Contribution
3
