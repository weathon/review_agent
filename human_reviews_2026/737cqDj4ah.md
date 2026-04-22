# Towards Reliable Transferability of Targeted Adversarial Attacks against Model Discrepancy

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Adversarial attacks pose a serious threat to deep neural networks, especially in black-box scenarios where transferability plays a key role. Targeted transfer attacks, where an attacker induces a specific misclassification on an unseen black-box model, remains significantly more challenging than non-targeted attacks. We attribute this gap to model discrepancies between surrogate and target models, including mismatches in feature representations, classifier heads, and Jacobians.
To address these challenges, we define a unified uncertainty set capturing these model discrepancies and propose a principled robust objective over this set. While intractable in full form, this view leads to a tractable relaxation: the Targeted Attack toward Reliable Transferability (TART).
TART integrates three components: (1) expectation over transforms to cover representation and Jacobian variability; (2) latent mixing to model attenuation and clean-feature leakage; and (3) feature matching}to guide perturbations toward semantically robust regions. 
Extensive experiments on ImageNet and CIFAR-10 show that TART consistently outperforms state-of-the-art transfer-based black-box targeted attacks, across both convolutional and transformer architectures. For example, when transferring from ResNet-50 to Swin-S on ImageNet, TART achieves a 42.7\% higher attack success rate than the strongest baseline. Our approach establishes a new benchmark for robust black-box adversarial evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the problem of targeted transferability of adversarial examples across heterogeneous models. It formulates a unified robust optimization framework that jointly considers discrepancies in feature extractors, classifier Jacobians, and decision boundaries. Building on this formulation, the authors propose TART, an attack method that integrates Expectation-over-Transformation (EoT), Latent Mixing, and Feature Alignment to enhance targeted transferability. Under several assumptions, they provide a theoretical analysis showing that TART optimizes a lower bound of the robust margin. Experiments on standard benchmarks demonstrate the competitiveness of TART compared to existing transfer-based attacks.

### Strengths
1. The paper is written in a fluent and accessible manner, making even technically involved concepts easy to follow. The overall organization is logical and reader-friendly.
2. The authors propose a relatively general formulation that explicitly models multiple sources of discrepancy between surrogate and target models (feature, Jacobian, and decision boundary), and then design algorithmic modules tailored to each. This gives a clear sense of completeness and conceptual coherence.
3. The method is supported by a theoretical framework that, under certain assumptions, provides partial guarantees relating the proposed TART objective to a lower bound of the robust targeted margin.

### Weaknesses
Majors:
1. The defense baselines ("Defenses" in Section 6.1) are mostly on early (many from 2017), which are no longer representative of the current state of adversarial robustness research. Including more recent baselines would make the evaluation more convincing.
2. The paper would benefit from a schematic figure illustrating the formalization in Section 3 and the method in Section 4. A visual depiction of the interactions among the EoT, Mixing, and Feature Alignment components would significantly improve clarity.
3. The proof of Lemma 3 contains a mathematical flaw: it incorrectly upper-bounds $\Vert \Delta_h-\Delta_h^t\Vert$ by $|\rho_h-\rho_h^t|$ (Line 690, page 13), whereas the correct bound should be $\rho_h+\rho_h^t$ by the triangle inequality. This leads to an underestimated Lipschitz penalty term and thus affects the tightness of the main theorem’s lower-bound guarantee. As this lemma underpins the core theoretical claim, the issue undermines the rigor of the overall guarantee.


Minors:
1. Some places do not specify which norm ($\ell_2$, $\ell_{\infty}$, etc.) is being used; also, dimensions of key symbols should be clarified when possible.
2. The notation $\rho$ is overloaded, it denotes the uncertainty radius in Section 3 and the Bernoulli parameter in Section 4. Similarly, $\epsilon$ and $\varepsilon$ should be used consistently for different quantities.
3. Line 273 (page 6): the expression "$\delta \in \Delta$" seems incorrect since $\Delta$ is not defined.
4. Line 222 (page 5): the sentence “Using a logit loss $\mathcal{L}$ to avoid saturation." lacks a subject.
5. The appendix contains many redundant expressions like “Eq. equation”; these should be cleaned up.
6. The term Jacobian mismatch should be clarified.
7. There are some incorrect uses of \citet and \citep (e.g., "Defenses" in Section 6.1).
8. In Figure 1, the vertical axis should be labeled TSR instead of ASR.
9. It would be helpful to cite *"Toward Robust Learning via Core Feature-Aware Adversarial Training"*, IEEE TIFS 2025, which is relevant to the topic.

### Questions
See Weaknesses. I am willing to raise the score if the authors address my concerns.

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
5

### Summary
This paper first theorizes the transferability chanlleges into three mismatches: extractor mismatch, classifier mismatch, and decision boundary mismatch. Afterwards, the paper proposes TART, which consists of three key elements, Expectation over Transformation, Latent Mixing, and Feature Alignment to compensate for respective mismatch. Finally, theoretical and experimental validations are presented to demonstrate the effectiveness of TART.

### Strengths
1. The three mismatches, i.e., extractors, classifier, and decision booundary, are generally intuitive to capture the main challenges in transfer-based attacks.
2. The overall writing, though aided by LLM, is clear and easy to follow.

### Weaknesses
This paper suffers from three major weaknesses: **Theory**, **Method** and **Experiments**.

**Theory**: First of all, the three mismatches, although intuitive, are innate definitions of black-box attacks. For example, extractor mismatch (EM) is the primary prerequisite for a 'black-box' attack, i.e., victim models use a different structure than the surrogate ones. The following classifier/decision boundary mismatches (CM, DBM) are all required for similar reasons. Besides, according to the information flow within models, EM, CM and DBM are not necessarily independent, as the former always contributes to the latter. Lastly, the idea of directly resolving these mismatches is paradoxical because if there are no mismatches, it is no longer viable to consider this attack as 'black-box'.

**Method**: Second, the three components of TART, despite the claim that they compensate for the three mismatches, are all well-established methods and widely adopted for boosting transferability. i) EoT, which randomly uses a group of transformations, has been proven effective for better robustness in both earlier [1] and recent works [2]. ii) Latent mixing, which mixes clean features/activations into adversarial ones, as is done in one of the baseline CFM, is also an established method. This paper proposes to use activation in the classifier layers instead of features without clearly explaining the theoretical necessity and demonstrating the experimental superiority. iii) Feature alignment, which essentially pushes the adversarial example further away from the original ones and towards the targeted ones, exactly follows the idea of a triplet structure, i.e., using the original example as a negative, targeted one as a positive. This triplet structure has also been widely adopted in transfer-based attacks [2] and deep metric learning attacks/defenses [3]. In sum, TART falls short regarding the novelty of methodology as it combines several proven effective methods.

**Experiments**: Lastly, the experiments suffer from unexpalined inconsistency. While the overall settings follow CFM, the target models vary significantly from the CFM paper. The only few consistent setttings exhit significant performance gaps. For example, for RN-50 against RN-18 on ImageNet, CFM-RDI reports 88.4% TSR in the original paper but 83.2%, with a 5% performance gap. These inconsistency further undermines the experimental solidarity of the paper.


[1] Lu, D., Wang, Z., Wang, T., Guan, W., Gao, H., & Zheng, F. (2023). Set-level guidance attack: Boosting adversarial transferability of vision-language pre-training models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 102-111).

[2] Gao, S., Jia, X., Ren, X., Tsang, I., & Guo, Q. (2024, September). Boosting transferability in vision-language attacks via diversification along the intersection region of adversarial trajectory. In European Conference on Computer Vision (pp. 442-460). Cham: Springer Nature Switzerland.

[3] Tian, Q., Lin, C., Zhao, Z., Li, Q., & Shen, C. (2024, July). Collapse-aware triplet decoupling for adversarially robust image retrieval. In Proceedings of the 41st International Conference on Machine Learning (pp. 48139-48153).

### Questions
Q1. Could you please clarify the superiority&necessity of using activation mixing instead of features? I notice that the mixing probability used in the paper is also the same as that in CFM (p=0.1), could you share your insight on why using feature and activation in latent mixing yields the same optimal hyperparameters?

Q2. Please clarify the variation of target models as well as the noticeable performance inconsistency of CFM-RDI regarding RN-50 against RN-18 on ImageNet.

### Soundness
2

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
This paper addresses targeted adversarial transfer attacks and proposes TART, which aims to improve transferability from a surrogate to an unknown target model. The method is based on a robust-objective formulation and integrates three components: EoT (Expectation over Transformation), latent mixing, and feature alignment toward a target-class exemplar. Empirical results demonstrate improved targeted transfer success rates, and the paper provides theoretical analysis connecting the surrogate objective to an ideal robust objective.

### Strengths
1 Identifies key challenges in targeted transfer attacks: feature mismatch, classifier sensitivity, and decision-boundary shifts. Formalizes an ideal robust objective and a tractable surrogate, providing a principled framework.

2 The method is intuitive and aligns well with the theoretical framework.

3 Ablations show that each component contributes positively.

### Weaknesses
1 The proposed approach appears incremental, combining several existing methods

2 The conditional lower-bound guarantee is interesting, but relies on standard assumptions. Its practical impact is mostly heuristic justification rather than a strict bound.

### Questions
1 Section 6.3 mentions exemplar choice for feature alignment, but the discussion is brief. Please clarify how sensitive TART’s performance is to different exemplars.

2 Can the authors clarify how often the directional monotonicity assumption (Assumption A4) holds in practice for high-dimensional, non-convex networks, and whether violations of this assumption affect the reliability of the surrogate objective?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces TART (Targeted Attack toward Reliable Transferability), a method designed to improve the reliability of targeted adversarial transfer attacks by modeling three sources of surrogate–target model discrepancy: feature-extractor mismatch, classifier sensitivity mismatch, and decision-boundary misalignment. The authors propose a robust optimization formulation and derive a tractable relaxation combining three techniques: Expectation over Transformation (EoT), Latent Mixing, and Feature Alignment. They claim both theoretical guarantees and strong empirical improvements on ImageNet and CIFAR-10 benchmarks.

While the paper is well written and presents reasonable experimental results, several theoretical and methodological issues raise concerns about the soundness and novelty of the contributions.

### Strengths
1. The problem of targeted transferability is important and underexplored relative to non-targeted transfer.

2. The paper attempts to connect empirical strategies (EoT, feature alignment, etc.) with a theoretical margin-based robustness formulation, which is conceptually appealing.

3. Experiments cover both CNNs and Vision Transformers, with reasonable baselines and extensive tables.

### Weaknesses
1. Questionable theoretical soundness and incomplete assumptions.
The theoretical analysis relies heavily on standing assumptions (A1–A3) that assume the feature extractors and classifiers are Lipschitz continuous with constants. However: 1) The paper does not discuss how these Lipschitz constants are constrained in practice or how large values of those Lipschitz constants would affect the derived margin bounds (Eq. 20 and Eq. 25). Intuitively, if the Lipschitz constants are large (as is common in deep networks), the resulting bound becomes vacuous. 2) The paper also does not empirically verify whether the proposed TART indeed leads to a smaller effective Lipschitz constant or a tighter robust margin, so the claimed “provable lower bound” is largely theoretical and uninformative. This makes the analysis appear formal but not practically grounded.

2. The three core components, EoT, latent mixing, and feature alignment, are all adaptations of existing strategies: 1) EoT has long been used to enhance robustness or transferability by averaging over transformations. 2) Feature alignment is a widely adopted practice in targeted transfer attacks (e.g., CFM, FTM). 3) Latent mixing is a minor variation of existing feature-mixup or interpolation methods. The paper does not provide a convincing theoretical justification for why combining these specific techniques should reduce model discrepancy beyond intuitive reasoning. The claimed “principled relaxation” is not strongly supported by derivation or ablation linking each component to the underlying bound.

3.  The decomposition into D2 (classifier sensitivity mismatch) and D3 (decision-boundary mismatch) appears artificial and overlapping.
Once the classifier $g$ is defined, its decision boundary is uniquely determined by $argmax(g(h(x)))$ for classification tasks. Thus, modeling both D2 and D3 as separate uncertainties introduces redundancy.

4. Experiments are largely limited to classification tasks on outdated architectures (ResNet-50, ViT-Tiny, etc.) with standard ImageNet and CIFAR-10 settings. To convincingly claim “reliable transferability,” the method should be validated on more challenging multimodal or vision-language tasks (e.g., VQA, captioning) or recent robust architectures. Without such evidence, the practical impact remains unclear.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
