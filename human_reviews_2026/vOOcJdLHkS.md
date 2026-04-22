# Label-Free Attribution for Interpretability

- Avg Score: 4.80
- Decision: Reject
- Scores: 6, 2, 2, 8, 6

## Abstract
The importance of attribution algorithms in the AI field lies in enhancing model transparency, diagnosing and improving models, ensuring fairness, and increasing user understanding. Gradient-based attribution methods have become the most critical because of their high computational efficiency, continuity, wide applicability, and flexibility. However, current gradient-based attribution algorithms require the introduction of additional class information to interpret model decisions, which can lead to issues of information ignorance and extra information. Information ignorance can obscure important features relevant to the current model decision, while extra information introduces the incorrect identification of irrelevant features as significant. To address these issues, we propose the Label-Agnostic Attribution for Interpretability (LAAI) algorithm, which analyzes model decisions without the need for specified class information. Additionally, to more rigorously assess the potential of current attribution algorithms, we introduce a variety of new evaluation metrics, combined with the traditional Insertion \& Deletion Scores, to comprehensively assess the performance of our algorithm. To continuously advance research in the field of explainable AI (XAI), our algorithm is open-sourced at https://anonymous.4open.science/r/LFAI-336C

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper identifies two critical biases in gradient-based attribution methods, termed "Information Ignorance" and "Extra Information," which arise from the dependency on a single target class, particularly in low-confidence scenarios. To address this, the authors propose a novel Label-Free Attribution for Interpretability (LFAI) algorithm that generates explanations by maximizing the model's output uncertainty rather than focusing on a specific class logit. Furthermore, the paper introduces a more robust evaluation framework, including a Confusion Feature Algorithm (CFA) for creating unbiased baselines and new KL-divergence-based metrics (KL-INS/DEL). Extensive experiments demonstrate that LFAI significantly outperforms state-of-the-art methods, especially on low-confidence samples.

### Strengths
1. The paper effectively identifies the problems of "Information Ignorance" and "Extra Information." By focusing on low-confidence samples, it highlights an under-explored weakness in existing attribution methods, providing a strong motivation for the proposed work.
2. The experimental validation is thorough. The authors compare LFAI against a wide range of SOTA baselines across multiple standard models and datasets. The superior performance provides strong empirical support for the paper's claims.

### Weaknesses
1. The definitions of "Information Ignorance" and "Extra Information" are primarily illustrated through examples and feel somewhat subjective. It would strengthen the paper if the authors could propose quantitative metrics to measure the extent of these two phenomena in existing attribution methods, moving beyond the conceptual formulas provided.

2. There appears to be a strong coupling between the proposed method and the proposed metric. LFAI is designed to maximize uncertainty (entropy), while the KL-INS/DEL metrics are designed to measure changes in uncertainty. Could the outstanding performance of LFAI on the KL metrics be a result of this "self-serving" evaluation, where the method is optimized for the very quantity the metric evaluates?

3. Equation (3) for entropy appears to be missing a negative sign. The standard definition of information entropy is H(x) = -Σ P(x)log(P(x)). Maximizing entropy is equivalent to minimizing Σ P(x)log(P(x)).

4. There are several minor formatting issues with citations that should be corrected for clarity and consistency. For example:
- Line 40: "(IG)Sundararajan et al. (2017)" should be "(IG) (Sundararajan et al., 2017)".
- Line 43: "(AGI)Pan et al. (2021)" should be "(AGI) (Pan et al., 2021)".
- Line 114: "in (Zhu et al., 2024; 2023)" should be "in Zhu et al. (2024; 2023)".

### Questions
Please address the questions raised in the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper examines how using a target label in attribution can bias explanations. It proposes a class-agnostic attribution that aggregates class evidence without conditioning on a specific label, paired with revised evaluation metrics intended to reduce label-driven bias. Experiments on standard image classifiers report results benchmarked using insertion and deletion metrics.

### Strengths
1. The impact of label choice on attribution is a meaningful topic.

2. The paper is well structured and easy to follow.

### Weaknesses
1. The target of this work is to attribute the effects of different classes. However, the method attributes to the sum over classes, producing class-agnostic maps. This collapses inter-class contrasts and weakens directional interpretability (cannot say “why A class over B class”), which is also problematic for tasks where class-specific reliance matters.

2. The paper claims that label conditioning causes information ignorance. However, softmax modeling already encodes mutual suppression among classes, and many existing attribution works explicitly use class-contrastive objectives [1]. In contrast, the objective of this work is unclear, and it fails to clarify the problems that introducing category information might cause.

3. Empirical evidence relies mainly on pixel-perturbation families (insertion/deletion games). To strengthen claims, include fidelity tests and distribution-robust benchmarks (e.g., ROAR/ROAD or other fidelity tests) to assess whether gains persist beyond pixel masking or after mitigating input distribution shift.

[1] Wang, Yipei, and Xiaoqian Wang. "“Why Not Other Classes?”: Towards Class-Contrastive Back-Propagation Explanations." Advances in Neural Information Processing Systems 35 (2022): 9085-9097.

### Questions
1. What failure modes follow from losing class directionality, and how would you differentiate between classes within your framework?

2. What is the distinct advantage of removing labels altogether compared to class-contrastive attribution objectives? Concretely, what specific problems arise from introducing label information that your method avoids?

3. There are some negative GAP values in Tables 2 & 3. Why do some methods report GAP < 0 (i.e., deletion curves outperform insertion, implying inverted explanations)? What does this mean for an attribution method?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a gradient-based attribution algorithm designed to interpret model decisions without requiring class label information. The motivation arises from the observation that current gradient-based attribution methods depend on predefined class labels, which can cause two biases: information ignorance (overlooking relevant non-target features) and Extra Information (incorrectly emphasizing irrelevant features). The proposed method redefines gradient accumulation to be label-agnostic, using the summed log-probability of all classes rather than a single class output. The paper also introduces new evaluation metrics to assess attribution quality and model uncertainty. Extensive experiments on Inception-v3, ResNet-50, VGG16, and additional models demonstrate that proposed method outperforms existing methods.

### Strengths
1.	The problem of bias in class-guided attributions is real and worth studying.
2.	The paper includes extensive experiments with multiple baselines and models.
3.	The paper is well-written and easy to understand.

### Weaknesses
1.	The distinction between Information Ignorance and Extra Information is presented as a new discovery, but these are fundamental and long-recognized challenges in attribution methods. Existing attribution techniques either fail to identify truly important features or incorrectly highlight irrelevant salient regions.
2.	. The proposed label-free formulation, which aggregates the log probabilities across different classes instead of relying on a specific label, is not truly label-free but rather label-independent. It is therefore recommended that the authors restate or clarify this problem definition.
3.	In Figure 2, I am not convinced that the results produced by LFAI represent the best outcome. Interpretability methods are expected to remain faithful to the model’s internal decision process rather than to align with human-perceived accuracy of attribution.
4.	The formula annotations are insufficient, and many notations lack clear definitions, for example, in Equations (4) and (5).

### Questions
Please refer to the Weaknesses.

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
2

### Summary
This paper porposes an attribution algotihm called Label-Free Attribution for Interpretability (LFAI) that aims to improve the limitations of gradient-based attribution methods. The authors argue that current gradient-based attribution methods can lead to two key limitations: information ignorance and extra information, caused by the methods using class information/labels to help interpret model decisions. LFAI on the other hand analyzes model decisions without introducing class information. The method is primarily applied to image classification tasks, and shows competitive performance compared to other methods in experiments and evaluation metrics.

### Strengths
The paper is well organized and the algorithm they developed (LFAI) is well explained. There are also many experiments that show the reader how LFAI perfromance compares to other methods, and it seems LFAI performs the best making this a strong contribution to the field.

### Weaknesses
I think it could be explained in a bit more detail why the authors believe attribution methods should not rely on labels. If you're explaining model behaviour, models are trained with the labels, so why should the attribution method ignore that? I think this is a slightly more debated topic in the field, so a bit more justification would be good!

With all methods there are some limitations (or at least trade-offs), it would be good to see the authors discuss what they think could be the limitations of LFAI.

### Questions
I think addresses the above weaknesses so:

1. What is the author(s) position on the debate on whether an explanation or attribution method should also use labels to develop the explanation? 
2. What are the limitations of LFAI?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces LFAI (Label-Free Attribution for Interpretability), a new gradient-based attribution method that does not rely on specifying a class label when explaining a model prediction. Instead of asking “why class y?”, LFAI integrates gradients of the sum of log-probabilities over all classes along an adversarial-style path, with the goal of capturing all evidence the model used — including evidence for alternative classes — and avoiding biases introduced by conditioning on a single class. The authors argue that standard attribution methods suffer from two systemic problems:
1. **Information Ignorance**: they ignore features of other plausible classes, so they can’t explain model uncertainty or low-confidence predictions;
2. **Extra Information**: they sometimes assign importance to irrelevant background pixels because the chosen “target class” forces the method to rationalize that class even when the model itself isn’t actually relying on those regions.

The paper also proposes new evaluation metrics (Fair Insertion/Deletion and KL-based variants) that try to remove the bias of using black/zero baselines and instead use a “maximally confusing” baseline image that maximizes predictive entropy, plus KL-based measures of how quickly the model’s uncertainty changes when adding/removing top-ranked pixels. 

On benchmarks using 1000 ImageNet images and standard CNNs (Inception-v3, ResNet-50, VGG16), plus additional experiments (ViT, CIFAR100) in the appendix, the authors report that LFAI beats 11 existing attribution methods across both high-confidence and low-confidence cases, especially in low-confidence regimes where traditional class-conditioned attribution tends to fail.

### Strengths
Main strengths:
- **Label-free attribution that directly targets a convincing gap in gradient-based XAI**, avoids conditioning on a single class, which the authors argue induces bias.
- **Attempt to formalizing two failure modes with set-based definitions**: _Information Ignorance_ = truly relevant pixels not highlighted (missed mass); and _Extra Information_ = irrelevant pixels highlighted (spurious mass).
- The papers even makes a **second, parallel, contribution on evaluation**: _Fair Insertion/Deletion_ (via a confusion-baseline) and _KL-based_ variants to assess distribution-level faithfulness.
- **Method & metric alignment**: the distribution-aware objective (aggregate over all classes) pairs naturally with distribution-aware metrics (KL-INS/DEL), yielding a coherent story for uncertainty and multi-object scenes.
- **Empirical relevance**: results emphasize low-confidence regimes where classic class-conditioned saliency underperforms, with additional analyses referenced in Section 4.4 / Appendix.
- **Reproducibility**: code is (anonymously) released, facilitating verification and uptake.

### Weaknesses
**MAJOR POINTS**
- **Core definitions lack precision / clarity** (which can prevent from full appreciation of the cool work in the paper): The set-based definitions of _Information Ignorance_ and _Extra Information_ are hard to parse as written (quantifiers, what is fixed vs. varying, and what $\Phi$ denotes). For example, authors write “$\exists |\varphi|\ge k$ s.t. $\varphi=(i\mid i\in \Phi \land a_i<\tau)$” and “$\exists |\varphi|\ge k$ s.t. $\varphi=(i\mid i\notin \Phi \land a_i\ge\tau)$” without specifying whether $k$ and $\tau$ are fixed ex-ante, how $\Phi$ is defined/measurable, or whether existence is trivial by tuning $\tau$? These need explicit quantifiers (“for fixed $k,\tau$ …”) and a concrete operationalization of $\Phi$ (e.g., object masks, counterfactual evidence) to avoid vacuity.

- **Equation 5 is under-motivated and its link to II/EI is not made explicit**: Why is the functional form $\frac{\partial}{\partial x_t} \left( \sum_{j} \log P_j(x_t) \right) = \sum_{j} \frac{1}{P_j(x_t)}, \frac{\partial P_j(x_t)}{\partial x_t}$ chosen, as opposed to others? A brief justification/discussion might help convince the reader to accept it. But most importantly, how does it theoretically help reduce II/EI? The paper would greatly benefit from a direct link between eq (5) and the mitigation of II/EI. A final _minor_ point related to the functional form: $\frac{1}{P_j(x_t)}$ can overreact to tiny probabilities; path-integrals may also be path/baseline-dependent and subject to OOD drift if the path leaves the data manifold. Any comments on this? 

**MINOR POINTS**

- **Method–metric alignment risk.**
The CFA baseline and KL-variants seem to be directly aligned with the label-free, distributional objective. That coherence is nice, but it may advantage your method by design, which should be made clear in your paper for honesty purposes. Not sure if it would be productive to report standard Insertion/Deletion (black baseline) results as well?

-  **“Class-independent” wording**: the method is better described as class-agnostic or label-free (it aggregates over all classes) rather than “class-independent,” which could be misread as not using class probabilities at all.

- **CFA / entropy sign**: In the CFA definition, double-check the entropy sign: maximizing uncertainty means maximizing $H(P)=-\sum_j P_j\log P_j$. If Eq. 3 omits the minus, that’s likely a typo?

### Questions
Please respond to the weakness listed above.

### Soundness
3

### Presentation
2

### Contribution
3
