# On the Alignment Between Supervised and Self-Supervised Contrastive Learning

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 8

## Abstract
Self-supervised contrastive learning (CL) has achieved remarkable empirical success, often producing representations that rival supervised pre-training on downstream tasks. Recent theory explains this by showing that the CL loss closely approximates a supervised surrogate, Negatives-Only Supervised Contrastive Learning (NSCL), as the number of classes grows. Yet this loss-level similarity leaves an open question: {\em Do CL and NSCL also remain aligned at the representation level throughout training, not just in their objectives?}

We address this by analyzing the representation alignment of CL and NSCL models trained under shared randomness (same initialization, batches, and augmentations). First, we show that their induced representations remain similar: specifically, we prove that the similarity matrices of CL and NSCL stay close under realistic conditions. Our bounds provide high-probability guarantees on alignment metrics such as centered kernel alignment (CKA) and representational similarity analysis (RSA), and they clarify how alignment improves with more classes, higher temperatures, and its dependence on batch size. In contrast, we demonstrate that parameter-space coupling is inherently unstable: divergence between CL and NSCL weights can grow exponentially with training time.

Finally, we validate these predictions empirically, showing that CL–NSCL alignment strengthens with scale and temperature, and that NSCL tracks CL more closely than other supervised objectives. This positions NSCL as a principled bridge between self-supervised and supervised learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies whether self-supervised contrastive learning (CL) and a supervised counterpart, Negatives-Only Supervised Contrastive Learning (NSCL), produce similar representations during the training process. The authors theoretically prove that CL and NSCL maintain close alignment in their representation spaces under realistic conditions. They provide probabilistic bounds for alignment metrics such as centered kernel alignment (CKA) and representational similarity analysis (RSA), showing that representation similarity improves with larger numbers of classes, higher temperature, and certain batch size conditions. Empirical results confirm the theory: CL and NSCL representations become increasingly aligned as model scale and temperature grow, with NSCL tracking CL more closely than other supervised objectives.

### Strengths
1.	The theoretical proofs and analysis are thorough and well-founded, providing a clear explanation of the differences between contrastive learning and negative-only supervised learning.
2.	There is a strong alignment between the theoretical and empirical results.
3.	The paper is well-written, with a clear and accessible logic flow that is easy to follow

### Weaknesses
1.	My primary concern is the lack of clear practical insights derived from the theoretical analysis in this paper. For instance, [1] identifies that downstream classification performance depends on labeling errors and the connectivity of the augmentation graph, while [2] and its follow-up works analyze the role of negative samples in contrastive learning. However, in this paper, what practical benefits can we gain from the theoretical relationship between CL and NSCL, especially considering that we have no access to labeled data in self-supervised learning? In other words, what real-world applications or improvements in contrastive learning can be derived from the theoretical framework presented here?
2.	I'm uncertain whether the theoretical analysis offers additional advantages over existing work. Specifically, [2] also characterizes the training process of contrastive learning and establishes guarantees between contrastive loss (even when it is not optimal) and supervised downstream loss. The authors note that prior works often rely on restrictive assumptions, but several follow-up works have relaxed these assumptions, including the conditional independence assumption. It would be helpful to further discuss how the theoretical analysis in this paper improves to the existing body of work.
3.	Another weakness is the lack of connection between the terms in the theoretical bounds and the design choices in contrastive learning. For example, while the paper shows there is still a gap between CL and NSCL, it is unclear which specific terms in the bounds contribute to this gap. Additionally, how might we modify these terms to close the performance gap between the two methods?
4.	The empirical results show that the performance of CL and NSCL is still significantly below that of supervised contrastive learning, particularly on datasets like ImageNet. This raises the question of whether focusing only on supervised information in negative pairs limits the potential of the analysis. For instance, [3] directly compares the gap between contrastive learning and supervised contrastive learning. Besides, it is impractical to only use NSCL if we have access to supervised information.

[1] HaoChen J Z, Wei C, Gaidon A, et al. Provable guarantees for self-supervised deep learning with spectral contrastive loss[J]. Advances in neural information processing systems, 2021, 34: 5000-5011.

[2] Saunshi N, Plevrakis O, Arora S, et al. A theoretical analysis of contrastive unsupervised representation learning[C]//International conference on machine learning. PMLR, 2019: 5628-5637.

[3] Cui J, Huang W, Wang Y, et al. Rethinking weak supervision in helping contrastive learning[C]//International Conference on Machine Learning. PMLR, 2023: 6448-6467.

### Questions
See Weaknesses.

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
4

### Summary
The paper asks whether the objective-level affinity between contrastive learning (CL) and negatives-only supervised contrastive learning (NSCL) holds to the representation trajectories. Specifically, the authors bound the discrepancy between the cosine similarity matrices induced by CL and NSCL under shared randomness, and then derive bounds for CKA and RSA. Experiments with ResNet-50 support the predicted dependencies on the number of classes $C$, batch size $B$, and temperature $\tau$. In particular, CL aligns much more closely with NSCL than with SCL or cross-entropy.

### Strengths
1. A technically sound similarity-space analysis of representations. This is an incremental contribution over recent connections between CL and NSCL [2].
1. For large $B$ and $C$, the main high-probability result is simplified to
    $$
        \|\Sigma^{\mathrm{CL}}-\Sigma^{\mathrm{NSCL}}\|_F =  \tilde O\left(\frac{1}{\sqrt{B}}\left(\frac{1}{C} + \frac{1}{\sqrt{B}}\right)\right),
    $$
    ignoring the explicit dependence on $\tau$ and $T$. This in turn gives bounds on representation similarity for CL and NSCL. The empirical trends in $C$, $B$, and $\tau$ match the theory, and the CL-NSCL pair consistently shows stronger alignment than CL-SCL or CL-CE.

### Weaknesses
1. Appendix D explains why the proposed "similarity-descent" mirrors small-step gradient descent on parameters. Additional remark of that argument in the main text would help follow the logic from SGD updates to similarity dynamics.
1. The theory assumes class-balanced sampling. It would be useful to discuss how class imbalance alters the bound, for example by replacing $1/C$ with empirical class priors and indicating the resulting rate.
1. Prior work reports a non-trivial CKA between SimCLR and supervised ResNet-50 trained independently [1]. It would be great to have additional discussions on the extension of the current theoretical results to a broader class of supervised learning algorithms.
1. A comment on the tightness of the upper bound for $\|\Sigma^{\mathrm{CL}} - \Sigma^{\mathrm{NSCL}}\|_F$ would be helpful.

**Minor Comments**

1. Line 308: replace "GPUs" with "GPU".
1.  "ImageNet-1K" (line 84) vs "ImageNet-1k" (Figure 3 lower right)
1. The per-anchor CL and NSCL loss definitions would be improved for clarity.
1. Move the ``Additional notation for high-probability factors'' so it appears immediately before Theorem 1.

[1] Grigg, T. G., Busbridge, D., Ramapuram, J., \& Webb, R. (2021). Do self-supervised and supervised methods learn similar visual representations? arXiv:2110.00528.

[2] Luthra, A., Yang, T., \& Galanti, T. (2025). Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning. arXiv:2506.04411.

### Questions
1. How does the analysis extend to class-imbalanced sampling? 
1. Is the bound for $\|\Sigma^{\mathrm{CL}}-\Sigma^{\mathrm{NSCL}}\|_F$ tight?

### Soundness
4

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
2

### Summary
This work provides an in-depth theoretical analysis of the alignment between Self-supervised Contrastive Learning (CL) and Negative-only Supervised Contrastive Learning (NSCL). In particular, a bound is derived coupling the CL and NSCL similarity dynamics, demonstrating the CL-NSCL behavioral alignment. Several experiments are also conducted to empirically validate the findings.

### Strengths
Leveraging similarity-space dynamics, this paper bridges CL and NSCL behavior in a theoretically rigorous manner. Empirical verifications are also done thoroughly, with near real-world scale datasets like tiny-ImageNet. Empirical results support theoretical findings.

### Weaknesses
As the authors stated in the conclusion/limitation section, the proposed bounds can be loose under large-scale settings due to the exponential factors on cumulative step size.

More importantly, please see my question below regarding the discrepancy between similarity-based dynamics and parameter-based dynamics, which I think is a very important step in the derivation of the proposed bound. It perhaps deserves a more detailed explanation.

### Questions
In line 1054 (Appendix D), the authors stated that "the difference between the two trajectories stays small when the step sizes are not too aggressive." However, Eq. (9) contains an exponential term on cumulative learning rate—does this really guarantee that the trajectories stay close even if the step sizes are not aggressive? Even so, how is this effect accounted for in Thm. 1, while practical algorithms perform parameter-GD instead of similarity-GD?

Intuitively, the similarity-GD and parameter-GD differ significantly as shown in Eq. (5) ($P_t := J_t J_t^\top$). The update delta of each similarity entry, when trained via parameter-GD, can vary depending on the network gradient evaluated at different input points. I find it somewhat counter-intuitive that the results on similarity-GD can be applied to algorithms utilizing parameter-GD. Therefore, It would be glad to hear the authors' responses on this matter (or please correct me if I misunderstood anything).

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
While contrastive learning and negatives-only supervised contrastive learning have similar objectives, this does not necessarily imply that they will learn similar representations through training. Using CKA and RSA, the authors analyze this alignment's emergence in various conditions to provide a better understanding of the similarities between such methods.
This is complemented by clear theoretical results that provide intuition on each parameter's influence on this alignment.

### Strengths
- Beyond training objective similarity, the authors provide an analysis of representation similarity through training, providing better insights in similarities between CL and NSCL.

- On top of the thorough experiments, the authors prove clear theoretical results that help guide empirical analysis.

- The detailed explanation of the bound obtained in Theorem 1 lines 246-262 is greatly appreciated to provide more practical intuitions in the result.

- The overall work strengthens our understanding of similarities and differences between CL and related supervised learning algorithms.

- Experiments are done at a good practical scale (up to ImageNet using a resnet 50) which helps yield more practically relevant insights.

### Weaknesses
1) My main issue is that it is unclear whether alignment is beneficial in practice. Having CL be more aligned with a **fixed** high performing NSCL model is most likely good in practice. However in the scenarios where alignment increases, it is possible that performance suffers as well. It would be interesting to have more data points regarding alignment vs performance potential tradeoffs.

2) While alignment is studied in the general case, it seems that it can get worse in practical settings.
Line 415-146 “Alignment increases with higher values of $\tau$”. Usually for CL temperatures used tend to be low, with performance decreasing as it is increases, suggesting that increasing alignment through temperature comes at a practical cost.

3) Line 079: $C$ is discussed but not introduced yet (I assume it is the number of classes from line 125). It would be good to have its meaning be clear when first discussed.

### Questions
1) In Figure 2 there seems to be a dynamics change around 10-100 epochs when looking at SCL and CE, any intuition as to why ?

### Soundness
3

### Presentation
3

### Contribution
3
