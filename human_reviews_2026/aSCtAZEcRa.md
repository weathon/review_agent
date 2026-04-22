# SCAD: Super-Class-Aware Debiasing for Long-Tailed Semi-Supervised Learning

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
In long-tailed semi-supervised learning (LTSSL), pseudo-labeling often creates a vicious cycle of bias amplification. Recent methods attempt to mitigate this issue via logit adjustment (LA). However, LA-based debiasing remains inherently hierarchy-agnostic and fails to account for semantic relationships between classes. We reveal a critical yet overlooked problem of \textit{intra-super-class imbalance}, where semantically similar classes within a super-class are both highly confusable and locally imbalanced. This combination reinforces early mistakes, causing minority-class representations to be suppressed by their majority neighbors. To break this cycle, we propose Super-Class-Aware Debiasing (SCAD), a framework that performs dynamic, super-class-aware logit adjustment. SCAD leverages latent semantic structure to concentrate its corrective power on the most confusable groups, thereby resolving local imbalances. Extensive experiments demonstrate that SCAD achieves state-of-the-art performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes SCAD (Super-Class-Aware Distillation) for class-incremental learning (CIL). 
Instead of treating each class independently, SCAD groups classes into super-classes based on semantic similarity in the embedding space and performs hierarchical knowledge distillation at both class and super-class levels. 
Specifically, the method first clusters features into super-classes and then aligns both intra- and inter-super-class logits between the old and new models to retain global structure information.
Experiments on CIFAR-100, ImageNet-100, and ImageNet-1K show improvements over standard distillation-based baselines such as LUCIR, PODNet, and PASS.

### Strengths
- The motivation is clear. Existing distillation methods only constrain class-level logits, ignoring the semantic hierarchy among classes, which can lead to loss of global structure.

- The proposed method introduces a simple yet interpretable hierarchical extension, super-class-level distillation, which fits naturally into existing CIL frameworks.

- The experimental setup is standard and fair, covering multiple datasets and comparing with widely used baselines under the same backbone.

- The paper is generally well written and easy to follow.

### Weaknesses
- Core contribution is an incremental extension of LA rather than a new learning principle. SCAD’s main novelty is to contextualize LA with a super-class posterior and precomputed $\Delta_k$. While the construction is sensible, the paper provides no theoretical guarantees (e.g., consistency/optimality of the dynamic term) and largely positions SCAD as a practical tweak to LA.

- Reliance on pseudo-label–driven frequency estimates is under-specified. $\Delta_k$ depends on relative dominance scores $\beta_{k,c}$ derived from counts/estimated frequencies $n_{k,c}$ within each super-class. In LTSSL, such estimates inevitably hinge on pseudo-labels. This paper does not analyze error propagation or provide robustness checks (e.g., how noisy pseudo-labels affect $\Delta_k$ and whether bias could be re-amplified).

- Design choices inside $\Delta_k$ lack empirical justification. For classes outside a super-class $C_k$, SCAD applies a fixed maximum penalty (the strongest suppression level). The paper does not compare against alternatives (e.g., soft penalties proportional to inter-group similarity) or show that this choice cannot harm cross–super-class confusions.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In the paper, the SCAD scheme is proposed to tackle the problem of long-tailed semi-supervised learning (LTSSL). In particular, SCAD assumes access to a super-class inferring mechanism to augment logit-adjustment (LA) by incorporating sample-specific logits correction.The proposed method is conceptually simple, introduces a negligible computational overhead while it can be used in combination with other LTSSL frameworks. Finally, the experiments conducted  across multiple datasets and configurations demonstrate improvements over other SoA methods.

### Strengths
- The paper is generally well-written and easy to follow.
- The method is well-motivated and conceptually simple (in a positive way) making it easy to use along with other LTSSL frameworks.
- The conducted experiments convincingly demonstrate the benefits of the proposed method.

### Weaknesses
- Few parts in the paper make somewhat strong claims that need to be softened and better grounded. (see Questions)
- Despite the improvements, the method utilizes previously explored concepts in the literature known to independently improve performance (i.e., augmenting the concept of LA by super-class awareness). Although it is possible that combining separately explored concepts can provide original insights in the community, one might not find this particular instance to offer much such insights. 
- The main argument to this is that after reading the paper, one might develop only a marginally better understanding of the LTSSL. Namely that utilizing more fine-grained (i.e., sample-aware) logit adjustment improves performance compared to the regular logit adjustment.

### Questions
Typo (T), Suggestion (S) and Question (Q)

Major Points:

S1. The caption in Fig 1 suggests that SCAD “resolves” the critical misclassification. However the Fig 1 d. shows similar misclassification trends but less pronounced under SCAD. In this regard, the terms “mitigate” or “alleviate” are possibly better suited here. The same applies for the “resolve” used in L85.

S2. L106-107: “it discovers hierarchies from class names alone”. This claim sounds a bit deceiving as both class names and a pretrained text encoder a utilized to infer the super-classes. Although assuming access to pretrained text encoders is not unreasonable, this has to be more clearly communicated in this context. 
Same argument for L181-182, where the assumption of a pretrained text encoder is omitted.

Q3. Apparently SCAD requires a class hierarchy to work. In the absence of ground-truth hierarchy it is reasonable to seek after a hierarchy approximation mechanism. In the case of the present manuscript, one such mechanism based on a text encoder is used. Is it possible to motivate this particular choice? Why not use a visual encoder instead? Or a combination of these two? 
Depending on the nature of the data and the annotating mechanism, one would expect the relationship among classes to change leading to different class hierarchies. For example, in a dataset where humans are often sitting on bikes, the “human” and “bike” class might cluster in the same super-class. A text encoder might overlook this relationship.

Q4. Do the terms L231: super-class-aware, L237: sample-agnostic and L245: context-specific all refer to the same concept? Please clarify in the text.

Q5: L254: “assigns a high penalty (close to 1)”. Based on the definition of $\beta_{k,c}$ one would expect in that case the penalty to be exactly 1 i.e., not close to 1. Was the max operator used in the definition of $\beta_{k,c}$ in L252 meant to be a summation? In that case one could understand the “close to 1”.

S6. Table 1. The selection of the reported algorithms’s performance appears a bit unintuitive.
- It would be interesting to see the Supervised w/ LA + Ours and/or Supervised w/ Ours in Table 1. This would provide a clearer picture between the interplay of LA and SCAD.
- L277: FixMatch is used as the baseline which is complemented with different other algorithms. However SCAD is only used in combination w/ DASO. How come FixMatch + Ours is not reported? How come only Fixmatch w/ DASO + Ours is reported and not for example w/DARP + Ours?
- Similar arguments apply for L280 and L284.

It would be good to see a clearer motivation on the table construction, which would help readers to better position SCAD. For example, is SCAD competitive as a standalone method or is it a plug and play modification that can be used with other competitive methods?

S7. Table 2. L296: Similar arguments as in S6 apply. Also how come w/ DASO + Ours was not included here? (like it was in Table 1.)

S8. Table 3 L333: Similar arguments as in S6 and S7 apply. For example why FixMatch w/ Ours was used in this case? In contrast to Table 1. where FixMatch + w/ DASO + Ours?
It would be good to see a more unified and intuitive scheme of result presentation in Table 1, 2 and 3 that will facilitate readers to better position SCAD in relation to the SoA.

S9. Apparently, any benefit from using SCAD originates from (i) the additional super-class supervision and (ii) the sample-aware delta correction. In this regard, the ablation provided in Table 6 is very useful. However, there is the opportunity here to provide a clearer picture of the interplay between SCAD and LA by also including:
- FixMatch + LA
- FixMatch + LA + SCAD
- FixMatch + SCAD

Apparently SCAD involves super-class learning by design. However, could it not be possible to remove the effect of super-class supervision by updating the feature extractor for regular classification while updating the super-class classifier on frozen features?
By doing so one would effectively isolate each of the (i) and (ii) and better capture their interplay with plain LA.

Q10. Table 7. suggests that some text encoders give rise to super-classes assignments that (marginally) outperform the ground-truth under SCAD. Are there any further reflections on this? Could it be purely random variation? or does it reflect something deeper?

Q11. It would have been good with a brief discussion on how SCAD affects calibration in the main paper. Section A.6. provides some empirical results on SCAD improving the calibration in tail classes but only in relation to FixMatch/LA. Does this finding generalize when using SCAD in combination with other SoA methods that have been reported earlier?
It is important to reflect upon this aspect as the results suggest that the optimal performance is achieved when using SCAD in combination with other SoA methods and one might wonder whether the increased performance comes at the expense of increased calibration error.

Minor Points:

S12. Fig 1 c. and d. Please specify which of the horizontal or the vertical axes are the ground-truth and the predicted class.

S13. Fig 2. might confuse readers as the logits in the super-class classifier do not show up but only the softmax operator. On the other hand the logits do show up in the fine class classifier but not the softmax operator. Unifying the visualizations would help to avoid confusion.

S14. Fig 2 caption L125: The characterization “powerful” see,s redundant. The targeted local adjustment already suffices to indicate a potential improvement over the (base) uniform adjustment. Same applies for “powerful” in L199.

S15. L143: define d. i.e., the dimensionality of the input data?

S16. L144: the c is used to refer to an arbitrary class, while the C is used to refer to the number of classes. This choice can confuse the readers in particular in the presence of typos. For example, L150: is it $l^c$ or $l^C$? From the context (and Fig 2) one might assume $l^C$ but cannot be certain. Also, is classifier $g_c$ anyway connected to the arbitrary class c? From the context one has to assume no. Recommendation is to fix the typos and update the notation accordingly.

S17. L147: define p. i.e., the dimensionality of the feature representation?

S18. related to a S13. L150: Consider including $l^s$ in Fig 2.

T19. L219: Should the source be L_{\text{super}}?

S20. L223: “ground-truth super-class” might be confusing as this is inferred from the mapping M which can be an approximation of the class hierarchy. Please reformulate to better communicate that M can either be ground truth or approximation leading to “ground-truth super-class” or “approximated super-class”

T21. L246: a blank space is missing after “vector”.

T22. Eq 7 L250: was $\Delta_{k,c}$ meant to be $\Delta_{k}$?

S23. L701: The acronym ECE (expected calibration error) was not previously defined.

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
This paper studies the long-tailed semi-supervised learning (LTSSL) problem, and proposes a new perspective of intra-super-class imbalance. The proposed idea is movitivated by the fail attempt of previous works that leverage logit adjustment for rectifying pseudo-label bias. The authors uncover that the previous works neglect the semantic relationships between classes, thus hindering effective LTSSL. To overcome this limitation, the authors proposes a new framework called super-class-aware debiasing (SCAD), which consists of a super-class-aware logit adjustment. Specically, it leverages pre-trained text encoders such as CLIP to generate super-class labels, and then jointly train a main classifier and a super-clalss classifier. During inference time, it applies logit adjustment according to the prediction results of super-classes. Experimental results demonstrate that the proposed method surpasses most of the previous methods.

### Strengths
- The proposed idea is well-motivated. The authors first conduct proper empirical studies to show the existence of intra-super-class imbalance, making the studied problem is meaningful.
- Considering super-class imbalance for rectifying pseudo-labels is reasonable, which is not proposed by previous works.
- The experimental results demonstrate the effectiveness of the proposed method.

### Weaknesses
- Although the illustration of intra-super-class imbalance is provided, it is only conducted on CIFAR10-LT. Such results can not demonstrate the universal existence of intra-super-class imbalance on long-tailed datasets.
- The relationship between super-class imbalance and semi-supervised learning is a bit weak. It seems that the proposed super-class-aware logit adjustment can be directly applied to long-tailed classification. Why must use this method for long-tailed semi-supervised learning?
- The authors claim that the proposed method lead to negligible computational overhead. However, leveraing pretrained models such as CLIP does lead to additional costs. Such cost should not be neglected.
- More details should be included regarding how to generate super-class labels using pre-trained models.
- The performance of SCAD should be highly related to the capability of the pre-trained models, which may affect the quality of super-class labels. Have the authors justified the influence of different pre-trained models?

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper clearly identifies the limitation of LA being “hierarchy-agnostic” and connects it to real-world intra–super-class confusion; the motivation is clear. It proposes SCAD, a dynamic logit correction framework weighted by the super-class posterior, aiming to introduce semantic context during pseudo-label generation. The approach is simple from an engineering perspective, easy to integrate into existing LTSSL pipelines, and empirically broad. However, in terms of novelty, automatic super-class discovery (text embeddings + clustering) and hierarchical auxiliary tasks are direct applications of existing ideas, and the core dynamic adjustment can be viewed as LA plus a heuristic penalty weighted by the super-class posterior, lacking substantial algorithmic or theoretical innovation and rigorous analysis.

### Strengths
1. The problem is well scoped, emphasizing LA’s failure under local confusion within super-classes; the motivation is intuitive.
2. The framework is simple and plug-and-play, requiring minimal changes to existing LTSSL methods and is deployment-friendly.
3. The experiments are wide-ranging, compatible with multiple SOTA pipelines, and show performance gains across datasets and settings.

### Weaknesses
- The biggest issue is limited novelty. It is largely an incremental work without standout contributions and, in my view, does not meet the acceptance bar for ICLR. Super-class discovery and hierarchical auxiliary learning are common ideas in class-imbalanced learning; the core dynamic component is a heuristic penalty formed by LA plus super-class posterior weighting, lacking novel algorithmic design or theoretical support.
- Since other models are used to partition classes, providing additional information, the performance improvements are expected. The ablation shows a large portion of gains come from adding the super-class task itself; SCAD’s incremental contribution is small on some datasets.
- The method is also quite constrained because it relies on other models for grouping classes. The datasets used in the paper consist of high-frequency, common categories, where using language models for grouping may work well. If the training set contains fine-grained or rare categories, the results may be significantly affected.
- The definitions and estimation of Δk and nk,c are unclear. The source of frequencies (labeled/pseudo-labeled/mixed), update strategy, stability, and how the scale is balanced relative to log π are not systematically explained.
- Sensitivity analysis for K and clustering configuration is insufficient: K=⌈C/4⌉ lacks broader validation; clustering metric, linkage criterion, and text preprocessing details are not fully disclosed, limiting reproducibility.
- Compared to LA’s Fisher consistency, SCAD’s dynamic component lacks discussion on consistency or optimality of the correction.
- There are regressions in some settings (e.g., with DASO in certain scenarios), without adequate cause analysis.
- The code is not publicly available; it is recommended to release reproducible code during the rebuttal period.

### Questions
Refer to weakness.

### Soundness
2

### Presentation
3

### Contribution
2
