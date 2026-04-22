# Unlearning during Training: Domain-Specific Gradient Ascent for Domain Generalization

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6, 2, 4

## Abstract
Deep neural networks often exhibit degraded performance under domain shifts due to reliance on domain-specific features. Existing domain generalization (DG) methods attempt to mitigate this during training but lack mechanisms to adaptively correct domain-specific reliance once it emerges. We propose Identify and Unlearn (IU), a model-agnostic module that continually mitigates such reliance post-epoch. We introduce an unlearning score to identify training samples that disproportionately increase model complexity while contributing little to generalization, and an Inter-Domain Variance (IDV) metric to reliably identify domain-specific channels. To suppress the adverse influence of identified samples, IU employs a Domain-Specific Gradient-Ascent (DSGA) procedure that selectively removes domain-specific features while preserving domain-invariant features. Extensive experiments across seven benchmarks and fifteen DG baselines show that IU consistently improves out-of-distribution generalization, achieving average accuracy gains of up to 3.0\%.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents "Identify and Unlearn" (IU), a model-agnostic module that addresses domain generalization by continually mitigating reliance on domain-specific features during neural network training. IU combines an unlearning score (to identify training samples that disproportionately increase model complexity while failing to improve generalization), an Inter-Domain Variance (IDV) metric (to identify domain-specific channels), and a domain-specific gradient ascent procedure to selectively remove these harmful features while preserving domain-invariant content. The method is thoroughly evaluated on seven benchmarks and multiple baselines, demonstrating consistent improvements in out-of-distribution performance.

### Strengths
1.	The proposed inter-domain variance metric is a novel identification of domain-specific channels.
2.	The proposed UI module is model-agnostic, which could achieve performance improvements over multiple datasets and methods.

### Weaknesses
1.	The motivation is unclear. The authors claim that prior approaches (e.g., Aggregated Variance, AV) are highly sensitive to domain imbalance, whereas the proposed inter-domain variance is robust. However, this issue is neither revealed nor proved experimentally, as the datasets used do not appear to exhibit large domain imbalance. Besides, many DG methods already suppress domain-related features or channels during training, which seems to be in conflict with the existing methods that operate “without any mechanism to correct such reliance once it emerges.” Finally, the claim that AV ignores cross-domain variability also lacks direct evidence.
2.	The details of Fig. 1 should be provided. The experimental description of Fig. 1 is vague. The paper does not specify how the fur-edge channel and background-sensitive channel are distinguished. It is also unclear how the analyzed features are extracted. 
3.	The contributions should be emphasized. The core contribution is the adaptive post-epoch removal of domain-specific features. Since this requires access to training data and adds computational overhead, what advantages does the post-epoch strategy offer over in-training suppression? How does the proposed procedure relate to prior domain-specific feature/channel suppression methods? Can the introduced IU module be applied on top of, or integrated into, those methods?
4.	Missing conceptual explanation. The key unlearning set selection relies on the measure of model complexity and a generalization score. Although citations and formulas are provided, the paper does not explain why these items capture “the impact of each training sample on both model parameters and validation performance.” 
5.	Weak theory–method alignment. Theorem 1 crucially depends on Assumption 1, which decomposes features and parameters into domain-specific and invariant components, followed by separate analyses. The link between this decomposition and the proposed algorithmic steps is weak, and the result appears applicable to any domain-specific feature-suppression method. Please clarify which parts of the theorem the algorithm explicitly exploits or adjust the theoretical development to align more tightly with the implementation.

### Questions
See in the weaknesses.

### Soundness
3

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
This paper introduces Identify and Unlearn, a novel, model-agnostic module designed to improve domain generalization. The core idea is to perform a post-epoch unlearning step to correct the model's reliance on domain-specific features as they emerge during training. The authors demonstrate through extensive experiments on numerous benchmarks and fifteen baseline methods that integrating the IU module consistently improves out-of-distribution accuracy.

### Strengths
- The central idea of treating DG as an iterative process of learning and adaptive unlearning is novel and compelling.
- The proposed Inter-Domain Variance (IDV) metric is a principled and intuitive method for localizing domain-specific representations within the model. The ablation study provides clear evidence of its superiority over the more naive Aggregated Variance (AV) heuristic.
- The paper's claims are supported by a remarkably thorough set of experiments and analysis.

### Weaknesses
- The primary practical weakness is the substantial computational cost, as honestly reported by the authors. The reliance on per-sample influence function estimation makes the training time per epoch an order of magnitude higher than standard training. While feasible for research, this presents a major barrier to practical adoption. Could the authors comment on potential strategies to mitigate this cost, such as performing the IU step less frequently (e.g., every 5 epochs)?
- The core claim of the paper rests on the idea of surgically unlearning from the correct channels. While IDV is well-motivated, the analysis lacks a crucial control experiment: what would happen if DSGA were applied to the domain-invariant channels for the identified unlearning set?
- The unlearning step relies on a first-order approximation of retraining via influence functions. When applied repeatedly after every epoch, it is unclear how this approximation error might accumulate. Could this repeated, approximate intervention push the model into an unstable or unforeseen state over the course of training, even if it appears beneficial in the short term? A discussion on the long-term stability of this iterative unlearning process would be valuable.

### Questions
See Weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Identify and Unlearn (IU), a post-epoch correction module for Domain Generalization (DG). While most DG approaches aim to prevent the learning of domain-specific features, this paper argues that once such biases emerge, there is no mechanism to correct them during training. IU fills this gap by periodically identifying and unlearning domain-specific dependencies.
 
The method consists of three parts: (1) Unlearning Set Selection (USS) uses influence functions to find samples that increase model complexity but contribute little to generalization, (2) Inter-Domain Variance (IDV) measures the variance of per-domain activations to detect domain-specific channels, and (3) Domain-Specific Gradient Ascent (DSGA) reverses gradients of those channels to reduce their influence.
IU is model-agnostic and improves performance by roughly ~3% across seven DG benchmarks and fifteen baselines (ERM, IRM, MixStyle, etc.).

### Strengths
The paper clearly articulates an important limitation in DG research: existing methods lack any corrective mechanism once domain-specific bias has been learned. The post-epoch unlearning framework is conceptually elegant and well-motivated.
The combination of USS and IDV forms a coherent pipeline—sample-level influence and channel-level analysis complement each other effectively.
The experimental section is comprehensive, spanning multiple DG settings with consistent improvements, and the writing is clear and well-organized.

### Weaknesses
The computational overhead of IU remains a concern. The authors mention that the inverse-Hessian approximation uses LiSSA, which is efficient, but the main cost comes from recalculating unlearning scores for every sample after each epoch. It would be useful to know whether they have considered lowering the computation frequency (e.g., every K epochs) or adopting stochastic sampling for approximate score estimation.
The description of IDV is intuitive, but more discussion is needed on potential limitations. For instance, how stable is IDV when the number of domains is small or when within-domain noise dominates between-domain variance? Have the authors explored normalization or any other strategies to stabilize this measure?
The paper’s claim that DSGA performs “unlearning” could benefit from conceptual clarification. Gradient ascent here functions as a form of targeted regularization, not data or privacy unlearning. A short note distinguishing these definitions would prevent confusion.

### Questions
​1.​Have you tested whether computing unlearning scores less frequently (e.g., every few epochs) preserves performance while reducing runtime?
​2.​How sensitive is the IDV measure to the number of domains or imbalanced domain sizes?
​3.​Have you considered adding a normalization step or a bootstrapped confidence interval for IDV to make it more robust?

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
4

### Summary
This paper introduces Identify and Unlearn (IU), a model-agnostic module for domain generalization (DG). The core idea is to move beyond purely preventative DG strategies and introduce a corrective, post-epoch unlearning mechanism. The IU module operates in three steps: 1) It uses an unlearning score, derived from influence functions, to identify a set of training samples that increase model complexity without contributing significantly to generalization (Unlearning Set Selection). 2) It introduces a metric, Inter-Domain Variance (IDV), to identify feature channels that are domain-specific by measuring the cross-domain variance of their within-domain activation variances. 3) It performs Domain-Specific Gradient Ascent (DSGA) on the parameters of the identified domain-specific channels for the samples in the unlearning set, selectively removing domain-specific knowledge while preserving domain-invariant features. The experiments demonstrate that IU consistently improves the performance of different DG baselines.

### Strengths
1. The idea of post-epoch unlearning for DG is a fresh perspective that addresses a limitation in the field, moving beyond static regularization techniques.
2. The experimental evaluation includes a diverse set of baselines across seven different benchmarks. This thoroughness provides strong evidence for the method's model-agnostic nature and general effectiveness.
3. The motivation, methods, and results are easy to follow. The ablation studies are clear in justifying the design choices of the proposed IU module.

### Weaknesses
1. The primary weakness of the proposed method is its extreme computational overhead. As reported in Table 11, the unlearning set selection step, which relies on influence function estimation, takes approximately 354 minutes (~6 hours) per epoch on the DomainNet dataset. This makes the method less practical for large-scale tasks. The lack of a baseline ERM training time in Table 11 also makes it difficult to fully grasp the relative overhead.
2. The paper presents IDV as a principled, domain-aware metric for identifying domain-specific channels. However, the ablation study in Table 2 (ERM_DSCS) shows that applying gradient ascent to channels selected by IDV across the entire training set leads to a catastrophic drop in performance. This suggests that IDV is not a robust metric for channel selection on its own; its effectiveness appears to be entirely dependent on its application to the very small, targeted unlearning set.
3. Table 1 presents results for both IU and IUE (IU with Exponential Moving Average). EMA is an orthogonal technique that is not part of the core contribution. Including IUE results in the main comparison table somewhat obfuscates the direct impact of the IU module and can give an inflated impression of the gains.
4. While framed as a strength, performing the complex IU operation after every single epoch may not be optimal. The influence of samples and the set of domain-specific channels might not change dramatically from one epoch to the next. A more efficient approach could be to run the IU module intermittently (e.g., every 5 epochs), but this is not explored.

### Questions
See above.

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
5

### Summary
This paper proposes a model-agnostic post-epoch unlearning module (IU) that effectively improves domain generalization performance. While the idea is interesting and empirically validated, the methodological details (especially in identifying domain-specific channels and the theoretical analysis) remain insufficiently clear.

### Strengths
1. The motivation for applying unlearning in domain generalization is logical and interesting.

2. The experimental results based on improvements over existing methods are sufficient, including comparisons with recent approaches and evaluations on comprehensive datasets.

### Weaknesses
1. In the first part of the Introduction, the authors’ perspective that models should learn domain-invariant features while unlearning domain-specific features aligns with most prior DG studies. However, the details of how texture-sensitive channels and background-sensitive channels are identified remain unclear. It is not specified which layer of the model these channels belong to, nor how they are distinguished.

2. In the motivation part of the Introduction (Figure 1), the phenomenon that texture-sensitive channels and background-sensitive channels correspond to domain-invariant and domain-specific features, respectively, is easy to understand. However, in the Method section, the authors do not explain how texture and background are located, but directly define low IDV values as domain-invariant features and high IDV values as domain-specific features. If, as the authors state, texture and background correspond to low and high IDV values, respectively, does that mean IDV values are used to determine texture and background? Is IDV the necessary condition for this distinction? This remains unclear.

3. The calculation details of AV and IDV are not sufficiently compared, making it difficult to understand the improvements achieved by IDV.

4. The comparison methods are somewhat incomplete. Table 1 only shows the performance improvement when the proposed method is combined with existing methods. However, it lacks a comparison of how existing methods alone improve the same baselines (e.g., DomainDrop [1]).

5. The theoretical proof in Section 3.4 (“THEORETICAL ANALYSIS”) seems weakly related to the proposed DSGA method. It is unclear which encoders correspond to the two feature types $f_{spc}$ and $f_{inv}$. In the Method section 3.3, only one model 
$\theta$ is introduced, and there are no separate encoders $\theta_{spc}$ and $\theta_{inv}$. Therefore, the theoretical section contributes little to supporting the main method.

>[1] Guo J, Qi L, Shi Y. DomainDrop: Suppressing domain-sensitive channels for domain generalization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023: 19114–19124.

### Questions
Please refer to the weaknesses listed above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 6

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper attempts to mitigate the influence of domain-specific futures for domain generalization. The authors proposed an unlearning strategy during training, named Identify and Unlearn (IU). IU firstly selects hard samples which contribute little to the generalization performance, then utilizes Inter-Domain Variance (IDV) to identify domain-specific channels. Subsequently, IU uses Domain-Specific Gradient-Ascent (DSGA) to mitigate the adverse effect of domain-specific features. Experiments demonstrates that IU could enhance the generalization performance across benchmarks.

### Strengths
[+] The paper is easy to follow.

[+] The detail of the method and experiment is well described.

[+] The related work is detailed.

### Weaknesses
Major weakness:

[-] The authors propose Inter-Domain Variance (IDV), which measures the variance of a channel’s within-domain variances across source domains. Since these within-domain variances are numerical values, what is the conceptual significance of taking their variance? How does this metric effectively identify domain-specific channels?

For example, consider a channel whose within-domain variances are [4, 4, 4] and within-domain means are [1, 5, 9] across three source domains. Would this channel be considered domain-invariant under the proposed IDV criterion? Please clarify the intuition behind this measure.

[-] The idea of suppressing domain-specific features via channel manipulation is not novel; methods such as DomainDrop [1] and DMDA [2] also employ this strategy. However, these methods apply channel modulation globally, whereas the proposed IU method suffers from performance degradation when applied globally, as shown in Table 2.

What is the underlying reason or mechanism that causes global channel manipulation to degrade performance in the proposed framework, in contrast to its success in prior works? Please explain this discrepancy.

[-] The paper aims to preserve domain-invariant information while suppressing domain-specific features. To visually demonstrate the effectiveness of this objective, it would be helpful to include t-SNE visualizations that reflect the domain gap, ideally in comparison with state-of-the-art methods. 

[1]. DomainDrop: Suppressing Domain-Sensitive Channels for Domain Generalization. ICCV, 2023.

[2]. Rethinking Domain Generalization: Discriminability and Generalizability. TCSVT, 2024.

Minor weakness:

[-] The writing quality should be improved. For instance, punctuation marks are frequently missing after mathematical formulas, and the reference notation in Table 1 should be formalized to enhance readability.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
