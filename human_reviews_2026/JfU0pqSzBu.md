# When Less Is More: Uncovering the Robustness Advantage of Model Pruning

- Avg Score: 4.40
- Decision: Reject
- Scores: 6, 2, 6, 6, 2

## Abstract
The interplay between neural network pruning, a widely adopted approach for model compression, and adversarial robustness has garnered increasing attention. However, most existing work focuses on empirical findings, with limited theoretical grounding. In this paper, we address this gap by providing a theoretical analysis of how pruning influences adversarial robustness. We first show that the pruning strategy and associated parameters play a critical role in determining the robustness of the resulting pruned model. We then examine how these choices affect the optimality of pruning in terms of maintaining performance relative to the original model. Building on these results, we formalize the inherent trade-off between clean accuracy and adversarial robustness introduced by pruning, emphasizing the importance of balancing these competing objectives. Finally, we empirically validate our theoretical insights on different models and datasets, reinforcing our novel understanding of the adversarial implications of pruning. Our findings offer a principled foundation for designing pruning strategies that not only achieve model compression but also enhance robustness without additional constraints or cost, yielding a ``free-lunch'' benefit.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper explores the relationship between model pruning and adversarial robustness by examining how different pruning strategies and parameters influence adversarial behavior. The theoretical analysis reveals that pruning has a positive effect on adversarial risk while negatively affecting natural performance. This dual impact clarifies the trade-off between robustness and accuracy, offering a principled foundation for understanding adversarially robust pruning.

### Strengths
**High readability with clear logical flow**

The paper is logically well-structured and easy to follow. The preliminaries section clearly defines the concepts of adversarial robustness and model pruning. Each step in the theoretical analysis builds coherently on the previous ones, maintaining a strong logical flow that enhances readability and overall comprehension.

**Insightful analysis of adversarial risk and its relationship to natural performance**

The analysis effectively highlights how pruning simultaneously reduces adversarial risk and degrades natural performance. This insight helps explain both the regularization benefits of mild pruning and the challenges of maintaining robustness under aggressive compression.

### Weaknesses
**W1: Adversarial risk does not directly reflect model robustness**

Although Theorems 1 and 2 indicate that a higher pruning probability leads to a smaller adversarial risk, this only measures the distance between predictions on clean and perturbed inputs, $x$ and $\hat{x}$. In practice, a higher pruning ratio often causes a significant drop in natural accuracy. Therefore, despite the reduced adversarial risk, the absolute robustness of the pruned model may still decline. Consequently, it may be inaccurate to directly claim that pruning has a positive impact on adversarial robustness (see lines 288–291).

**W2: Limited discussion on layer-wise pruning ratios**

As shown in [1], effective pruning should account not only for the importance of individual parameters but also for layer-wise pruning ratios to better preserve both adversarial robustness and natural performance. Given that adversarial risk is bounded by natural performance, how can the proposed analysis in Proposition 1 be extended to interpret the variance of pruning ratios across layers?

**W3: Limited evaluation on advanced adversarial perturbations**

The evaluation primarily relies on PGD and FGSM attacks, which do not fully capture the model’s robustness. A more comprehensive evaluation should include stronger and more diverse attacks, such as C&W [4], AutoPGD, and AutoAttack [5].

---
### References

[1] Zhao and Wressnegger, "Holistic Adversarially Robust Pruning," ICLR 2023.  
[2] Sehweg et al., "Hydra: Pruning Adversarially Robust Neural Networks," NeurIPS 2020.  
[3] Ye et al., "Adversarial robustness vs. model compression, or both?" ICCV 2019.  
[4] Carlini et al., "Towards Evaluating the Robustness of Neural Networks," IEEE S&P 2017.  
[5] Groce and Hein, "Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks," ICML 2020.

### Questions
**Q1: How can the proposed magnitude-based pruning analysis be extended to structured pruning?**

The analysis in Section 3 is based on weight pruning. How might these findings generalize to structured pruning? Does the relationship between pruning probability and adversarial risk still hold in that context?

**Q2: Does the theoretical relationship persist after post-pruning fine-tuning?**

Previous works on adversarially robust pruning [1, 2, 3] include adversarial fine-tuning to recover performance. Since fine-tuning updates model weights, does the established relationship between pruning ratio and adversarial risk remain valid after fine-tuning?

**Q3: Can pruning improve adversarial robustness before harming natural performance?**

Empirical studies [1, 3] suggest that mild pruning can enhance robustness through regularization effects. As pruning progresses from minor to aggressive levels, does robustness first improve before degradation in natural accuracy occurs? Is it possible to derive an optimal pruning strategy that maximizes adversarial robustness without compromising natural performance?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates the theoretical and empirical relationship between model pruning and adversarial robustness, aiming to provide a formal theoretical framework that links pruning parameters to adversarial risk.

### Strengths
- The approach is interesting. 
- Studies on adversarial robustness and pruning can be relevant. 
- The used architectures are modern and up-to-date.

### Weaknesses
**Outcomes of the study and relation to prior work:** The paper starts with good premises of providing a formal definition, characterizing a trade-off, and validate insights through experiments. However, in doing so, I believe that the authors end up with some overly generic outcomes which do not add much value to our current understanding of the relationship between robustness and pruning. This is also made more severe by the lack of a critical paragraph in the manuscript which reflects and compares the provided contributions with what is already known from the state-of-the-art. Specifically:
- *Definition of adversarial robustness in the context of pruning:* I do not see the necessity of providing this definition in Equation 3 and claim it as a contribution, since it is a simple definition of certified robustness on a given bound adapted on a pruned model (which is still a model) [ext_ref_1].
- *Outcome from sections 4 and 5:* It is well-known that pruning may or may not enhance adversarial robustness [int_ref_1, ext_ref_2], but that is not being considered in the discussion and does not add much to what is known from previous work. It is also very well-known in adversarial robustness that there exist an inherent trade-off between accuracy and adversarial robustness  [ext_ref_3]. Therefore, in general, the outcome of the theorems cannot be limited to a simple observation which traces already known concepts, but it is supposed to provide something new or at least some new and well-supported opinion highlighted by the theorem itself or further experiments. 
- *Experimental validation:* What one infers from the experiments is that there is an initial improvement given by pruning, which is then reduced as sparsity increases. Once again, prior work has largely discussed the regularization effect of pruning on adversarial robustness [int_ref_1, ext_ref_2].

**Experiments:** 
- *Adversarial robustness evaluation:* In adversarial robustness, it is now well-known that using attacks such as FGSM or PGD can lead to overestimation. Hence, the community has moved a step forward from relying solely on these attacks [ext_ref_4], and advocated for more reliable approaches such as the AutoAttack framework [ext_ref_5]. In addition, these advances have also been put forward in adversarial pruning literature [ext_ref_2], but this is not being considered in the work. I believe that the choice of relying on these attack approaches makes the manuscript largely behind already well-known advances in adversarial robustness literature.
- *Missing details:* It is not clear from the manuscript how are scores computed in the score-based approach used in 6.3.2, as I not find it specified in the setup description. I also fail in understanding whether the pre-trained models are robust against adversarial attacks, and if yes how they have been adversarially trained. In addition, I read in the appendix that the used perturbation bound is $\epsilon=4/255$, but it is common practice to choose the perturbation bound based on the used dataset (e.g., $\epsilon=8/255$ for CIFAR-10 and $4/255$ for ImageNet).

**Other comments and suggestions, not necessarily minor:** 
- Please write a paragraph in the related work section describing how the contributions and outcomes of the theorems differ from known concepts, and additionally consider important missing citations on highly related work such as [ext_ref_2] (which provides a taxonomy and discusses the work and advances in the literature) or other robustness/pruning related papers. I also suggest to follow the adversarial robustness guidelines provided there. 
- Increase legend fontsize in Figure 2, as they are too small and not easily readable. 
- I found abstracts and introduction slightly misaligned. I usually find easier to read papers where what is said to be done in the abstract is then extended and reflected in the introduction. For instance, in the abstract the authors start by saying that "We first show that the pruning strategy and associated parameters play a critical role in determining the robustness of the resulting pruned model", but this looks to me more of a contribution that the authors intend to provide after the theorems and discussion. But clearly, this is more of a subjective preference and for sure not a limitation. 
- The authors mention that users can then use their formulated trade-off to find optimal strategies, but it is not clear to me how one can choose the strategy based on the provided formalization.

[ext_ref_1]: Hein, Matthias, and Maksym Andriushchenko. "Formal guarantees on the robustness of a classifier against adversarial manipulation." Advances in neural information processing systems 30 (2017). 

[int_ref_1]: Artur Jordao and Hélio Pedrini. On the effect of pruning on adversarial robustness. In Proceedings of
the IEEE/CVF International Conference on Computer Vision, pp. 1–11, 2021.

[ext_ref_2]: Piras, Giorgio, et al. "Adversarial pruning: A survey and benchmark of pruning methods for adversarial robustness." Pattern Recognition (2025): 111788.

[ext_ref_3]: Zhang, Hongyang, et al. "Theoretically principled trade-off between robustness and accuracy." International conference on machine learning. PMLR, 2019. 

[ext_ref_4]: Carlini, Nicholas, et al. "On evaluating adversarial robustness." arXiv preprint arXiv:1902.06705 (2019).

[ext_ref_5]: Croce, Francesco, and Matthias Hein. "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks." International conference on machine learning. PMLR, 2020.

### Questions
- What is the added value of formally defining adversarial robustness for pruned models? 
- How does the outcome of Sections 4 and 5 differ from what is already known from the state-of-the-art? Could you explain what new theoretical insight these results provide beyond re-stating known effects, and how they relate to known literature?
- Why did you choose to evaluate robustness only with FGSM or PGD attacks, given that these are known to overestimate robustness?
- How are the scores computed in the score-based pruning method described in Section 6.3.2?
- Were the models pre-trained with adversarial training, or are they standard models pruned? If they were adversarially trained, please clarify which training method and perturbation bounds were used.
- Why is the perturbation bound fixed to $\epsilon=4/255$ for all datasets?
- The paper mentions that the proposed trade-off formulation can help users identify optimal pruning strategies. Could you elaborate on how a practitioner should concretely use this trade-off in practice?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper explores how model pruning influences adversarial robustness. It proposes a formal theoretical framework that relates pruning probabilities to adversarial risk bounds, then validates the predictions experimentally on standard architectures and datasets. The work finds that pruning can improve robustness but introduces a clear trade-off with clean accuracy. While the theoretical part is mathematically sound, some assumptions (e.g., Lipschitz continuity, uniform pruning) may limit applicability.

### Strengths
1. The paper provides the first theoretical understanding of the impact of pruning on adversarial robustness, establishes a unified probabilistic pruning framework, and extends beyond existing empirical studies. The theoretical framework is clearly articulated and builds upon solid mathematical foundations.

2. The study contributes to an underexplored intersection between compression and robustness.

3. The experiment covered various models, datasets, and attacks, verifying the generalization ability of the theory.

4. The paper reveals the implicit regularization effect of pruning, providing a new perspective for robust model design.

### Weaknesses
1. The theoretical analysis assumes uniform pruning probabilities and Lipschitz-continuous layers, which may not generalize to real-world architectures.
2. There is little discussion of computational efficiency or practical trade-offs.
3. Focus is placed solely on magnitude-based and score-based pruning, and the exclusion of other strategies (e.g., structured pruning) could restrict the applicability of the findings.
4. The experimental scope employs only FGSM and PGD attacks; the lack of evaluation against more sophisticated attacks (such as adaptive attacks) means the robustness assessment is not comprehensive.

### Questions
1. Theoretical compactness: Is the derived upper bound compact in practice? Are there any experimental data that can be directly compared with the upper bound?
2. Does the analysis extend to stochastic or dynamic pruning applied during training?
3. What is the intuition behind using Frobenius norms in Theorem 2 rather than spectral norms?
4. How were the pruning rates selected for each architecture, and do they differ by dataset?
5. Is the computational cost (such as fraction calculation) during the pruning process significant?

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
This paper provides a theoretical analysis that formalizes the impact of pruning on the adversarial risk of DNNs (Transformer-based Models and MLPs). Specifically, this paper derives upper bounds for the adversarial risk of pruned models and for the performance deviation (optimality) of the pruned model from the original. These bounds are used to characterize a trade-off between adversarial robustness and clean accuracy induced by pruning. The paper further supports its theoretical claims with extensive experiments on various architectures (MLP, CNN, ViT) and datasets (MNIST, CIFAR10, CIFAR100), offering a principled foundation for designing pruning strategies that not only achieve model compression but also enhance robustness without additional constraints or cost, yielding a “free-lunch” benefit.

### Strengths
1. Theoretical novelty and contribution: This paper and provides a relatively comprehensive theoretical analysis of the adversarial risk of pruned DNNs, which is an important and timely topic. The derived bounds and trade-offs provide valuable insights into how pruning affects model robustness, which can guide future research and practical applications.
2. Stable theoretic foundation. This paper provides a clear mathematical formalization of the adversarial risk of DNNs in Equation (3) and detailed proofs in the appendix, which demonstrates the rigor of the theoretical analysis.
3. Extensive empirical validation: The paper supports its theoretical findings with a wide range of experiments across different architectures and datasets. The empirical results align well with the theoretical predictions, strengthening the validity of the claims made in the paper.

### Weaknesses
1. Lack of direct experimental results to illustrate the gap between the experimentally measured adversarial risk and the theoretically derived bounds. Specifically, in Theorem 1, the author claims that "the adversarial risk of the pruned model g is always smaller than its corresponding original model f". However, the author only proves, if I don't misunderstand, that the adversarial risk *bound* of g is smaller than that of f. It seems that the latter is not strong enough to support the former.
2. Some figures should be improved; they are too blurry to see clearly. For example, it would be better to enlarge the legend in figure 2.
3. See Question.

### Questions
1. Is the definition of adversarial risk of DNNs in Equation (3) widely used or accepted in the AI community? 
2. In figure 2, there always exits a "sweet spot" where pruning achieves best attacked accuracy. However, is it always true? Specifically, given a special situation, as pruning ratio increases, if the harm effect of pruning on clean accuracy is larger than the benefit effect of pruning on robustness, then the "sweet spot" may not exist. 
3. Can you discuss more about the contributions discussed in Abstract and Conclusion. This paper claims that *"offering a principled foundation for designing pruning strategies that not only achieve model compression but also enhance robustness without additional constraints or cost, yielding a “free-lunch” benefit"*. In fact, this paper seems to not provide any convincing pruning strategies (e.g. how to "chosen pruning parameters") without additional cost, but only analyze existing ones.
4. The models used in this paper are mainly vision models (MLP, ViT, CNN) and vision tasks (MNIST, CIFAR10, CIFAR100). Can the findings be generalized to NLP models and tasks, even LLMs, which are more widely used and need pruning more urgently?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper provides the theoretical framework explaining how model pruning affects adversarial robustness. It proves that pruning inherently reduces the adversarial risk upper bound, meaning that pruned models can become more robust to small input perturbations. However, pruning also introduces an optimality gap, as excessive sparsity harms clean accuracy.

### Strengths
+ Good writing and easy to follow.
+ The theoretical derivation is consistent and well-aligned with the empirical results.
+ The idea of bridging pruning to robustness bounds is interesting and potentially impactful.

### Weaknesses
+ Important related work missing. The paper fails to cite a paper that first demonstrated and analyzed the robustness benefits of sparsity [1]. This paper would benefit from a deeper discussion of what new insights are gained beyond [1].
+ Experimental limitations. 
    + The experiments use too few attack methods (only PGD/FGSM). More recent and diverse attacks (AutoAttack, CW, etc.) should be included to make the conclusions generalizable.
    + The evaluation benchmarks and models are relatively small (e.g., MLPs, small CNNs). Results on larger and more modern architectures would make the paper more convincing.
+ I'm just wondering whether activation pruning/sparsity will have similar phenomenon?

[1] Guo, Yiwen, et al. "Sparse dnns with improved adversarial robustness." Advances in neural information processing systems 31 (2018).

### Questions
Please see weaknesses.

### Soundness
1

### Presentation
3

### Contribution
2
