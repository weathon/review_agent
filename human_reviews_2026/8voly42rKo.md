# Exposing and Defending the Achilles' Heel of Video Mixture-of-Experts

- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
Mixture-of-Experts (MoE) has demonstrated strong performance in video understanding tasks, yet its adversarial robustness remains underexplored. Existing attack methods often treat MoE as a unified architecture, overlooking the independent and collaborative weaknesses of key components such as routers and expert modules. To fill this gap, we propose Temporal Lipschitz-Guided Attacks (TLGA) to thoroughly investigate component-level vulnerabilities in video MoE models. We first design attacks on the router, revealing its independent weaknesses. Building on this, we introduce Joint Temporal Lipschitz-Guided Attacks (J-TLGA), which collaboratively perturb both routers and experts. This joint attack significantly amplifies adversarial effects and exposes the Achilles’ Heel (collaborative weaknesses) of the MoE architecture. Based on these insights, we further propose Joint Temporal Lipschitz Adversarial Training (J-TLAT). J-TLAT performs joint training to further defend against collaborative weaknesses, enhancing component-wise robustness. Our framework is plug-and-play and reduces inference cost by more than 60% compared with dense models. It consistently enhances adversarial robustness across diverse video datasets and model architectures, effectively mitigating both the independent and collaborative weaknesses of MoE.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the adversarial robustness of video Mixture-of-Experts (MoE) architectures, an underexplored area despite their growing success in video understanding tasks. The authors propose Temporal Lipschitz-Guided Attacks (TLGA) to analyze vulnerabilities at the router and expert levels, and further introduce Joint TLGA (J-TLGA), which jointly perturbs routers and experts to expose collaborative weaknesses. To mitigate these vulnerabilities, the paper presents Joint Temporal Lipschitz Adversarial Training (J-TLAT), a hierarchical defense framework enhancing both component-wise and overall robustness. Experiments on UCF-101 and HMDB-51 demonstrate that J-TLGA effectively breaks MoE models while J-TLAT substantially improves adversarial robustness with low computational overhead.

### Strengths
* Addresses an important and underexplored problem of adversarial robustness in video MoE models.

* Provides novel component-level analysis (router and expert) that exposes previously uncharacterized weaknesses.

* Presents strong empirical results across datasets and models with significant robustness gains and reduced inference cost.

### Weaknesses
* Unclear threat model and attacker assumptions

The paper does not explicitly define the threat model used in TLGA and J-TLGA. It is unclear whether the attacks assume a white-box, gray-box, or black-box setting, nor what information the adversary possesses (e.g., access to router parameters, gradients, or architecture details). Since the practicality and interpretability of robustness claims depend heavily on these assumptions, the authors should clearly specify the attacker’s knowledge scope and capabilities. Without this clarification, it is difficult to determine whether the attacks are realistic or merely theoretical.

* Limited diversity of experimental datasets

The evaluation focuses primarily on UCF-101 and HMDB-51, which are relatively small-scale datasets. While these benchmarks are widely used, they may not fully capture the complexity and diversity of modern video understanding tasks. Extending the evaluation to larger and more challenging datasets would significantly strengthen the empirical validity and generalizability of the findings.

* Limited theoretical insight into Lipschitz guidance

The paper briefly introduces temporal Lipschitz coefficients and their role in guiding perturbations, but the theoretical rationale behind why this specific guidance improves attack strength (or defense stability) is not rigorously analyzed. For instance, the connection between the Lipschitz constant and temporal consistency of video features is not deeply discussed. Including theoretical bounds or geometric intuition could clarify how the approach differs from conventional gradient-based or temporal smoothness attacks.

* Defense generalization not fully demonstrated

J-TLAT’s robustness is mainly evaluated against its own attack (J-TLGA). Although it achieves strong performance under this threat, it is unclear whether the defense generalizes to other unseen or non-Lipschitz-based attacks (e.g., FGSM, PGD, or frame-specific perturbations). This makes it difficult to determine whether the method truly enhances general robustness or merely overfits to its own attack pattern. Additional experiments on cross-attack evaluation would greatly improve the paper’s credibility.

### Questions
1. What is the attacker’s access level in TLGA and J-TLGA, does the adversary have full gradient access (white-box) or only query access (black-box)?

2. How do hyperparameters (e.g., Lipschitz weighting, perturbation budgets) influence attack effectiveness and defense performance? Are they transferable between datasets?

3. Can J-TLAT improve robustness against non-Lipschitz or unseen attacks? Some ablation in this direction would be informative.

4. Could the component-level attack/defense framework generalize to other MoE domains, such as multimodal fusion or dynamic routing networks?

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
3

### Summary
This paper investigates adversarial robustness of video Mixture-of-Experts (MoE) models. It identifies that conventional attacks treat MoE as a whole and ignore the distinct vulnerabilities of routers and experts. In this paper, the authors propose Temporal Lipschitz-Guided Attack (TLGA)  as a component-level attack on routers or experts and Joint TLGA (J-TLGA) as a coordinated attack exploiting collaborative weaknesses. Based on these two attacks, they finally propose Joint Temporal Lipschitz Adversarial Training (J-TLAT) as a three-stage defense procedure targeting routers, experts, and overall MoE. Experiments on UCF-101 and HMDB-51 show strong improvements in adversarial robustness.

### Strengths
- The structure of this paper is clear. The paper is well-written and easy to follow.
- The work analyzes component-level vulnerabilities in video MoE architectures. The idea of differentiating between router and expert robustness is insightful and practically relevant.
- The analysis from attack to defense provides a unified view of robustness analysis. The proposed method provides a comprehensive understanding of the studied problem.
- The experiments are comprehensive, covering multiple backbones (3D-ResNet, TSM, Video Swin) and datasets. The reported reduction in Lipschitz constant and GFLOPs is convincing.
- The appendix provides a clean theoretical derivation for why router manipulation leads to cascading failures.

### Weaknesses
- The new attack (TLGA) is basically a standard PGD-style adversarial attack with two small changes: a time-based adjustment of the step size, and an extra term related to the Lipschitz constant. What seems novel is the focus on attacking the router and experts separately in MoE models.
- The loss, e.g., equation (7), is not clear how this actually constrains the Lipschitz constant or enforces smoothness. There’s no clear derivation or estimation of the Lipschitz bound 𝐾.
- The three-stage adversarial training (router → experts → MoE) adds a lot of extra computation. However, the paper only compares inference efficiency, not total training cost. For a fair evaluation, it should compare against other robust training baselines (e.g., AT-M, AAT-M) using the same number of training epochs and attack iterations. Otherwise, the reported robustness gain might simply come from more training rather than the proposed method itself.

### Questions
- What the paper calls “collaborative weakness” may actually come from training imbalance. The router tends to favor a few experts while under-training others, so once those experts are attacked, performance drops sharply. This is a common issue in MoE models and can probably be improved by better load balancing.

- The proposed three-stage defense is quite complicated. Some of the same effects, e.g., reducing router over-reliance, might be achieved with simpler methods, such as balanced routing regularization or stochastic expert dropout, without training three times.

- The table layout is incorrect. Some tables are in the middle of the reference..

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
4

### Summary
This paper is the first to explore adversarial attacks in video mixture-of-experts. Through experiments, the authors highlight component-level vulnerabilities in the video moe structure. Corresponding attacks are designed for both the moe and the router, and corresponding adversarial training is proposed to improve the model's adversarial robustness. Experiments validate the effectiveness of the proposed methods.

### Strengths
1. Exploring the adversarial robustness of the video moe structure is beneficial for the secure deployment of the model.

2. This paper proposes an adversarial training method to improve adversarial robustness.

### Weaknesses
1. My biggest concern is the generalization ability of the experimental results. The number of models is limited, and the results of multiple models cannot be represented in the main body. Most importantly, the clean accuracy in Table 2 is extremely low, lower than the performance of common models on UCF-101, meaning the models were not well trained. Attacking a model that is prone to misclassification is easier and will lead to problematic conclusions.

2. This paper is unclear in its expression. It involves several self-proposed attacks and defenses, with abbreviations that are merely letter abbreviations without any semantic information, making them easily confusing. Even more egregiously, the abbreviation of Temporal Lipschitz Guided Router Attack as TLGA-R is unclear and doesn't seem to follow any specific principle.

3. The performance representation in Table 2 is misleading. Performance under different attacks should also be compared. Based on the experimental results, TLA-M and TLA-E are inferior to PGD, which weakens the contribution of this work.

4. Current research on adversarial examples focuses more on black-box attacks. This paper involves three models; why are there no experimental results for black-box transfer attacks?

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3
