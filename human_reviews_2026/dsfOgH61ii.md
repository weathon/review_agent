# RAHP: Robustness-Aware Head Pruning for Certified Transformer Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 4

## Abstract
Transformers lie at the core of modern AI, yet their susceptibility to adversarial perturbations raises reliability concerns. Empirical defenses often lack guarantees, while certification-based approaches provide them at nontrivial computational cost. We introduce RAHP (Robustness-Aware Head Pruning), a certification-guided pruning framework for Transformers. RAHP scores each attention head with a composite of (i) $\Delta$CLEVER, the predicted increase in a certified-robustness lower bound when masking that head, and (ii) Fisher information, the estimated accuracy cost of removing it. We prune heads that maximize robustness gain per accuracy cost. Across evaluated tasks, RAHP yields compact models with stronger CLEVER lower bounds and minimal change in clean accuracy, and it improves resistance to a wide variety of strong attacks. By leveraging a certified metric to steer structural pruning, RAHP makes certification-oriented robustness more practical and scalable.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper examines how to prune transformer-based models to enhance robustness with little loss of accuracy.
This is realized by ranking each attention head with a combined score based on CLEVER and FI.
The effect of the pruning is demonstrated on several benchmarks.

### Strengths
- The paper is well written and easy to follow.
- The paper stressed the importance of certified accuracy to reliably determine the robustness of the model
- Their presented heuristic is easy to understand, having a focus on the fundamental trade-off between accuracy and robustness.

### Weaknesses
- While the paper motivates the work using certified robustness, the presented heuristic does not live up to this claim,
as the presented heuristic is only based on estimates, and thus no formal guarantees are obtained.
- The CLEVER paper was also published in 2018, and obtaining actual formal guarantees on the robustness of neural networks has drastically improved over the last years [1], and also methods dedicated to transformers were developed [2].
- The presented heuristic is trivial from previous works, and thus does not make a huge contribution.
- Evaluation is limited and does not show the standard deviation or similar. Given how close the values are together, its difficult to judge how much the method actually improves over the other methods.

Minor things:
- Instead of having two weighting parameters $\alpha,\beta$, unify them to a single parameter $\lambda$ and weight the two terms with $\lambda$ and $1-\lambda$, respectively, for easier parameter tuning.
- Placement of Alg. 1 is odd, as it only gets references towards the end of the section.
- The $g(\cdot)$ in (1) is not well defined, I assume this has to do with the classification margin

[1] Brix, et al. "The fifth international verification of neural networks competition (vnn-comp 2024): Summary and results." arXiv preprint arXiv:2412.19985 (2024).
[2] Bonaert, et al. "Fast and precise certification of transformers." Proceedings of the 42nd ACM SIGPLAN international conference on programming language design and implementation. 2021.

### Questions
- Have you evaluated you the heuristic against methods with formal guarantees?
- Why do you want to prune the model in the first place? To improve robustness/reduce inference time/...? How does this compare with just training a smaller model directly? How is the inference time improved if that is the goal?
- Why is $\gamma$ necessary? Shouldn't your heuristic automatically rank important heads in a way that prevents them from being pruned? How are they determined at the moment? 
- As I understand it, you can choose any value how much you want it to be pruned. So an ablation study on that parameter would also be nice (I think you only showed $\rho=0.6$?)
- Alg. 1, l.10,11: how are these terms normalized? Aren't they just scalars?

### Soundness
1

### Presentation
3

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
This paper presents RAHP, a robustness-aware pruning framework that prunes Transformer attention heads based on a trade-off between certified robustness improvement (∆CLEVER) and accuracy cost (Fisher information). It aims to enhance robustness while maintaining clean performance. Empirical studies on GLUE and AdvGLUE show competitive results compared to state-of-the-art robust fine-tuning and pruning approaches.

### Strengths
Strength:
* The integration of ∆CLEVER as a robustness-oriented pruning signal is conceptually appealing and differentiates this work from previous empirical robustness methods.

* The composite scoring mechanism combining Fisher and CLEVER is well justified.

* RAHP achieves competitive robustness–accuracy trade-offs under heavy pruning (up to 60%) while maintaining interpretability in layer-wise patterns.

* The paper provides clear visualizations and ablation studies, illustrating how hyperparameters (α, β) affect robustness and accuracy balance.

### Weaknesses
Weaknesses:

* The baseline coverage is insufficient. Many adversarially robust or certified methods are missing, such as MixADA, TA-VAT, and ProTransformer (robust attention). Without them, it is difficult to judge how much of the robustness gain comes from pruning versus other regularization mechanisms.

* The attack strength is relatively weak—only AdvGLUE is used, which may not fully reflect model robustness under diverse perturbations (e.g., paraphrase-based, gradient-based, or jailbreak-style attacks).

* The comparison with certifiable defense frameworks like SmoothLLM or randomized smoothing methods is lacking.

* The method’s generalization to other modalities (e.g., ViT for vision, GAT for graph data) is only mentioned as future work, but given the simplicity of the approach, even a small demonstration would make the contribution more convincing.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents a new pruning method called RAHP. Specifically, RAHP leverages the CLEVER score and Fisher information to evaluate the robustness and sensitivity of attention heads, guiding the pruning process. Experimental results indicate that the pruned model achieves slight improvements in both accuracy and robustness.

### Strengths
1. The paper is clearly written and easy to follow. The proposed method is straightforward, and the overall pruning pipeline is well-presented and intuitive.

2. The topic is both interesting and practically relevant. As existing studies have shown, current transformer models still struggle with robustness, particularly in efficiency-oriented settings such as pruning. Advancing robust pruning techniques is therefore an important and timely research direction.

[1] Ye, Shaokai, et al. "Adversarial robustness vs. model compression, or both?." Proceedings of the IEEE/CVF international conference on computer vision. 2019.

### Weaknesses
1. The proposed method lacks novelty. The authors directly adopt CLEVER scores and Fisher information, both well-established metrics, to compute importance and perform pruning. The contribution appears largely incremental, applying existing robustness and sensitivity measures to Transformer architectures without introducing fundamentally new techniques or insights.

2. The paper claims certified robustness in the title, yet neither the methodology nor the evaluation reflects true certification. CLEVER only estimates a certified radius and relies on an approximate Lipschitz constant, which cannot provide a theoretical guarantee of robustness. Consequently, the paper does not offer verifiable certified robustness results, making the title misleading.

3. The performance gains are marginal compared to existing work. As shown in Table 1, the average improvements on GLUE and AdvGLUE over ROSE-Ensemble are less than 0.5%, which falls within typical experimental variance and does not convincingly demonstrate the effectiveness of the proposed approach.

### Questions
NA

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes RAHP, a Robustness-Aware Head Pruning method for Transformer models. It leverages a certified robustness metric (CLEVER) and Fisher information to jointly decide which attention heads to prune. By pruning heads that maximize robustness gain per accuracy cost, RAHP achieves compact models with improved certified robustness and minimal accuracy degradation. Experiments on GLUE and AdvGLUE show that RAHP consistently outperforms strong baselines such as ROSE, SMART, and FreeLB, and provides interpretable pruning patterns across layers.

### Strengths
The idea of integrating certified robustness metrics into pruning decisions is well-motivated.

The proposed method is simple, requiring no adversarial retraining and minimal fine-tuning.

The experiments are clear and well-organized, showing that RAHP improves both robustness and model compactness.

The analysis of pruning behavior and the visualization of Fisher/∆CLEVER maps provide useful insight into the model’s structural redundancy.

### Weaknesses
(1) The scope of model backbones is somewhat limited—results are primarily reported for RoBERTa. Evaluating additional transformer variants (e.g., BERT, ALBERT, DistilBERT) would strengthen the generality of the findings.

(2) Other certified defense methods (e.g., SmoothLLM) should be discussed and included for comparison.

(3) The evaluation is restricted to AdvGLUE, which is relatively weak and does not capture the full range of adversarial scenarios. No white-box or gradient-based attacks are considered, which may create a false sense of robustness; this limitation should be explicitly acknowledged and justified.

(4) The CLEVER score provides only an estimated, not guaranteed, lower bound on robustness (see https://arxiv.org/abs/1804.07870). Therefore, the method and evaluation based on this metric might overestimate the robustness. A deeper discussion of its implications is necessary.

(5) A runtime analysis would also enhance the paper by demonstrating the computational efficiency of RAHP. The computational cost of each component should be clearly justified.

### Questions
Please refer to the weakness.

### Soundness
3

### Presentation
3

### Contribution
3
