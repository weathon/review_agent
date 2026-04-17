# Transfer is All You Need: Revisiting the Stability–Plasticity Dilemma through Backward and Forward Transfer in PLMs

- Decision: Reject
- Scores: 2, 2, 6

## Abstract
Incremental Learning (IL) has long been an important research area in neural networks. Since IL requires retaining prior knowledge while learning tasks sequentially, many studies have primarily focused on 'Memory Stability' to address catastrophic forgetting, while paying less attention to 'Learning Plasticity'. However, this perspective has recently been challenged. Recent studies have demonstrated that the backbone exhibits sufficiently strong anti-forgetting capabilities, while the classifier (LM Head) is the primary source of forgetting. Moreover, as research on Learning Plasticity has gradually expanded, conflicting findings have emerged regarding the relationship between forgetting and forward transfer. For this issue, we propose a method to evaluate the forgetting and forwarding ability of the backbone itself and compare it with the evaluation in the classifier. To this end, we re-establish the famous metrics BWT (Backward Transfer) and FWT (Forward Transfer) and analyze the correlation between the two. As a result, we find that BWT and FWT are measured completely differently in Classifier, Probing Classifier, and Backbone, and this is the cause of the conflict in previous studies. More specifically, we observed that the considerable capability of the backbone is not effectively transferred to the classifier (LM Head). To address this, we propose 'Just LM-Head Tuning (JLT)', a simple yet highly effective approach that leverages the backbone trained through the IL process to transfer the LM Head. JLT is compatible with all existing IL methods and achieves state-of-the-art (SOTA) performance while allowing the backbone to remain unfrozen and continue acquiring knowledge. This effectiveness has been demonstrated not only on older discriminative backbones such as BERT, but also on very recent generative backbones such as LLaMA3.2 and Qwen3 across eight representative benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper applies the evaluation metrics of Backward Transfer (BWT) and Forward Transfer (FWT) directly to the backbone, avoiding the drawbacks of traditional computation methods. The experiments verify the differences between the two evaluation approaches and propose Just LM-Head Tuning (JLT), which performs lightweight training only on the classifier to adapt to incremental tasks. Although the experiments demonstrate the effectiveness of the JLT method, they do not address the core issue of the paper — the stability–plasticity dilemma.

### Strengths
1. The paper verifies that forgetting typically occurs in the classifier rather than in the backbone.
2. The paper proposes evaluating BWT and FWT on the backbone to avoid the bias introduced by the classifier.
3. The paper is generally clearly written.

### Weaknesses
1. Although the newly defined FWT and BWT metrics partially avoid the drawbacks of traditional computation methods, they also introduce new instabilities (as discussed in the Question section).
2. The new evaluation approach significantly increases the experimental cost and duration, making rapid validation difficult and limiting its applicability to other related research areas.
3. Regarding the stability–plasticity dilemma raised in the Introduction, the JLT method itself does not resolve this issue; instead, it proposes a more efficient way to utilize the backbone, rather than addressing the inherent forgetting problem that may occur in the backbone or the head during incremental learning.
4. The paper’s underlying assumption is that “the backbone is more stable, and most forgetting occurs in the classifier.” However, the subsequent experiments mainly reinforce this existing conclusion. As discussed in related works [1] and [2], both have addressed similar points, and the proposed evaluation method and JLT approach seem to follow naturally from this pre-existing assumption.

[1] Probing Representation Forgetting in Supervised and Unsupervised Continual Learning, CVPR 2022.

[2] Learn or Recall? Revisiting Incremental Learning with Pre-trained Language Models, ACL 2024.

### Questions
About FWT:

(1) When evaluating FWT, does the use of a temporary classifier introduce new bias? Training a classifier for each new task $i$ means that the FWT score will heavily depend on the training quality of this temporary classifier. If the classifier is well or poorly trained, it may overestimate or underestimate the actual transferability of the backbone.

(2) Since the classifier used for evaluation is temporary and unrelated to the final classifier used in the incremental model, could this make the evaluation inaccurate with respect to the final transfer performance we truly care about? Although traditional methods using an external classifier are detached from the intermediate process, the classifier is at least fixed and well-optimized. In contrast, the proposed method is process-dependent and introduces new variables related to training dynamics and hyperparameter sensitivity.

About BWT:

(1) Using clustering as a proxy task to evaluate performance on supervised learning tasks raises the question: can good clustering performance truly be considered equivalent to good classification performance?

(2) The paper experiments with multiple clustering algorithms, but the results appear to vary. Does this indicate that the newly defined BWT is not a stable metric and is sensitive to the choice of clustering algorithm, thereby reducing its comparability?

(3) Clustering algorithms inherently lose some of the fine-grained information that is crucial in tasks, a limitation that the paper does not seem to discuss.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors conduct an in-depth evaluation of the stability–plasticity dilemma in Pretrained Language Models (PLMs). They argue that most of the existing literature focuses on accuracy, while little attention has been paid to metrics such as backward and forward transfer. As noted by the authors, this is partly due to some “controversial” aspects related to the classifier and its handling. In this regard, they evaluate three different strategies for assessing these metrics and apply them to various models. According to their results, forgetting issues appear to be more related to the classifier than to the backbone, which, in turn, seems to exhibit an anti-forgetting effect, an observation consistent with previous studies. Finally, the authors propose a strategy called Just LM-Head, which involves training the classifier only after the incremental learning phase. This approach appears to achieve state-of-the-art results on several language tasks.

### Strengths
The intention of this paper is noteworthy, as it attempts to move beyond standard evaluation protocols to better understand which components of the model are responsible for forgetting issues. 
The evaluation is, in principle, extensive, covering several incremental learning approaches, paradigms, and tasks. Moreover, it focuses on recent language models, offering a renewed and fresh perspective on a longstanding problem.

### Weaknesses
**Justification of Results**
- Just LM-Head Tuning (JLT): the authors propose to train a new classification head after the incremental learning phase. According to Table 2, this leads to remarkably high gains, with improvements of around +40–45% in accuracy. However, it is unclear how such a result is possible. During the incremental learning phase, the model should have already learned the tasks; therefore, it is not evident how retraining a new classifier after this phase could solve all the forgetting-related issues. I find no empirical, theoretical, or intuitive justification to understand or trust this result. Moreover, the preceding sections do not adequately introduce the rationale behind this approach.

**Clarity**
- Just LM-Head Tuning (JLT): In the abstract and introduction, it is difficult to clearly get the difference between the proposed approach and existing methods that freeze the backbone and train only the final classification layer. I suggest revising these sections to better highlight the distinction and possibly reconsidering the name of the method, as the current one may be somewhat confusing.
- Section 4.2: This section is unclear. The authors mention that there is a separate classifier that does not participate in the incremental learning process, yet it is later used for evaluation. It is not clear how the weights of this classifier are obtained if it does not take part in the incremental learning phase. Consequently, since the procedure for computing this separate classifier is not well explained, it becomes difficult to interpret the results derived from this analysis.
- Section 5.2: In the bullet list, the first two points appear to be in contradiction. Have I misunderstood them, or is there an inconsistency that should be clarified?

**Methodology**
- Section 4.3.1: the authors propose an approach based on clustering to compute the classification rule. But, the clustering step could introduce an additional failure point in this process. Hence, why do the authors do not rely on the simplest KNN classifier? This is much more standard in literature.

**Organization and Presentation**
- The organization of the paper is weak. Although the authors report many tests and results, it is often unclear which experiments are truly significant, reliable, or worth emphasizing. As a result, the presentation feels flat, and the reader struggles to identify the key takeaways. The most important findings should be highlighted more effectively—possibly through clearer structuring, dedicated summary boxes, or explicit remarks within the text.

**Overclaims**
-Lines 051–052: “existing IL methods have focused only on overcoming catastrophic forgetting, and mainly evaluate the average accuracy for each task.” This statement is not accurate. Numerous studies have already addressed these specific aspects, making the authors’ claim reductive with respect to the existing literature. For instance, in [1], the authors explicitly investigate the preparation for future tasks; similarly, many other works consider continual learning from both perspective, i.e., mitigating forgetting of past tasks and facilitating learning of future ones.

- Personal comment: I personally find the idea that there could be an absolute answer to the question “Is forgetting mainly caused by the classifier or by the backbone?” quite questionable. In my view, this strongly depends on the experimental setting. Based on my experience, the more the downstream tasks diverge from the pre-trained model’s domain, the more the backbone needs to adapt — and consequently, the more backbone-related forgetting issues become relevant. Therefore, I believe this is a highly complex question that is difficult to address with full rigor.

[1] Boschini, M., Bonicelli, L., Buzzega, P., Porrello, A., & Calderara, S. (2022). Class-incremental continual learning into the extended der-verse. IEEE transactions on pattern analysis and machine intelligence, 45(5), 5497-5512.

### Questions
See section above.

### Soundness
2

### Presentation
2

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
This paper investigates incremental learning in natural language processing. First, it conduct extensive studies to diagnose the insights of FWT and BWT on classifier, probing classifier and backbone. Beyond such empirical analysis, it develops a simple and effective method namely Just LM-Head Tuning (JLT), which can be applied to all IL methods. Thanks to JLT, existing IL methods achieve significant improvements on stability and plasticity.

### Strengths
The motivation on studing FWT and BWT is interesting and important for incremental learning.

This work conducts many empirical experiments to shed light on the differences between classifier, probing classifier and backbone.

The proposed JLT method demonstrates consistent and promising gains over existing baselines.

### Weaknesses
Apart from empirical analysis, this paper lacks theoritical evidence, to explain the  findings.

The motivation behind JLT is not totally clear. It is unknown about why LM can help overcome the problem in FWT and BWT.

The algorithm of JLT needs more explaination. In addition, how can it be integrated with other baselines. It is needed to provide a specific example.

Furthermore, it is helpful for understanding JLT given a new figure.

### Questions
In Figure 2, it claims there are 8 methods. However, how to tell the results from these methods?

In Figure 3 and 4, it lacks explaination on the results achieved by different methods.

This work conducts experiments with NLP benchmarks. It is interesting to clarify whether it can be applicable to computer vision benchmarks as well.

### Soundness
3

### Presentation
3

### Contribution
2
