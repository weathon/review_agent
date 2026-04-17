# A Tight  Error Bound for Deep Learning via Distribution and Loss Complexity

- Decision: Reject
- Scores: 6, 2, 6, 2

## Abstract
Generalization is a central requirement for machine learning models in real-world applications, yet theoretically verifying it for trained models - especially deep neural networks (NNs) - remains highly challenging. Existing generalization bounds are often vacuous for modern NN architectures. In this paper, we propose model-dependent bounds that connect a model’s behavior in data space with its generalization ability. Our bounds explicitly capture both the complexity of the data distribution and the loss function, and they highlight the role of alignment between data geometry and the loss landscape. These properties enable our bounds to obtain significantly tighter estimates of test error than prior approaches. Extensive experiments on ImageNet classification and segmentation models show that our tractable bound consistently provides the tightest (nonvacuous) test-error estimates across a wide range of large-scale NNs. Moreover, we find that some parts of our bounds can effectly track the dynamic of test error, offering new insights into how to understand and improve performance in deep learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the long-standing problem of non-vacuous generalization bounds for modern deep nets. It proposes a model-dependent bound that explicitly captures both data-distribution complexity and loss-landscape structure via a partition of the input space, emphasizing how alignment between data geometry and the loss landscape drives generalization. A key technical contribution is a new concentration inequality for multinomial variables that removes the usual $\sqrt{\ln K}$ dependence, yielding tighter rates and enabling a sharper bound. Building on this, the authors derive a computable bound—estimable from training data under i.i.d. and bounded-loss assumptions—that preserves the theory’s structure and avoids the looseness of prior results (notably improving over Than & Phan, 2025). Empirically, across 37 pretrained models (32 ImageNet classifiers, 5 COCO segmenters), the bound delivers the tightest non-vacuous error estimates among compared methods (e.g., ResNet-18: ~57.9% → ~54.7% vs. ~30.3% true error). Moreover, a macro-level term in the bound correlates well with test performance and can track generalization dynamics—even when validation error is misleading—suggesting value as a diagnostic tool for overfitting and model selection. Overall, the paper presents a well-motivated, technically solid advance that meaningfully tightens practical generalization guarantees for deep learning.

### Strengths
- Data/Loss-Aware Bounds, Sharper Concentration, and a Computable Guarantee: The paper introduces a model-dependent generalization bound that captures micro-level (local regions) and macro-level (across the data manifold) behavior via a data-space partition, revealing a previously uncharacterized interaction between data geometry and the loss landscape. A key advance is a new concentration inequality for multinomial variables that removes the usual $\sqrt{\ln K}$ dependence, yielding tighter rates and addressing a weakness in recent SOTA bounds (e.g., Than & Phan, 2025). The theory is presented as two complementary results: a general bound and an exactly computable bound that preserves the structure while requiring only i.i.d. sampling and a bounded loss. Empirically, the computable bound consistently outperforms prior SOTA, reflecting richer loss-landscape information and stronger distribution awareness.

 - Extensive Empirical Validation and Diagnostic Utility: Tested on 37 large-scale models (32 ImageNet classifiers, 5 COCO segmenters), the bound delivers consistently tighter non-vacuous estimates than prior methods across the board. A macro-level term from the theory also tracks test error during training, sometimes outperforming validation error as an indicator—suggesting practical use for model selection and overfitting detection beyond its theoretical guarantees.

### Weaknesses
- Complexity of the Bound and Partition Dependence: A potential weakness is the added complexity introduced by the bound’s dependence on a chosen data space partition (Γ). While the partition-based approach is key to capturing data geometry, it also means the bound comes with several terms (sums over partition regions, local losses $F(S_i,h)$, etc.) that may be hard to interpret intuitively. The paper does not fully discuss how to choose an optimal or canonical partition in practice – the tightness of the bound could depend on this choice. If $\Gamma$ is chosen poorly (e.g. misaligned with the data’s true structure), the bound might become looser. Thus, more guidance on selecting or learning a good partition (and the sensitivity to the partition size $K$) would strengthen the practicality of the approach. As it stands, applying the bound requires an additional design decision (how to partition the input space) that practitioners must make without a clear prescription from the paper.

 - Residual Gap to True Error: Despite the improvements, the bounds can still be somewhat loose in absolute terms for certain models. The authors report non-vacuous and tighter bounds than before, but there remains a noticeable gap between the bound values and the actual test errors in many cases. For example, as noted above, the ResNet-18 classifier with ~30% test error has a bound around ~54.7%, and other large networks (e.g. ResNet-152 V2) with ~17.8% test error have bounds in the low 30% range. While these are the tightest known results to date, they are still far from truly tight estimates of generalization performance. The paper could discuss the sources of this remaining gap, for example, whether it is due to unavoidable worst-case terms (e.g., union bounds or concentration slack), suboptimal partitioning, or other conservative aspects of the theory. A deeper analysis of what limits the bound’s tightness would help in understanding how much further this line of work can go.

 - Scope of Empirical Evaluation: The experiments, albeit extensive for vision models, are focused on image classification and segmentation tasks. It is not entirely clear how well the proposed bounds would extend to other domains such as natural language processing or other data modalities. The underlying theory is general, but certain elements (like how to partition text or sequence spaces, or handling very high-dimensional input differently) might pose new challenges. Similarly, all reported models are pre-trained and fixed; an interesting scenario is using the bound during training to monitor generalization. While the authors hint at tracking test error dynamics, they do not explicitly demonstrate using the bound online for early stopping or model selection.

 - Assumptions: The theoretical results assume an i.i.d. sample and a bounded loss function. These assumptions are standard for classification error (0-1 loss is bounded in $[0,1]$) and were likely satisfied in experiments (where error percentage is the metric). However, for settings using unbounded loss functions, it’s unclear if the bound is directly applicable or if one must switch to a bounded proxy. An explanation on the theoretical generalization bound results with the example of cross-entropy loss or mean-squared loss which are unbounded would give readers a better understanding.

### Questions
1. Could the authors elaborate on how the partition $\Gamma(Z)$ should be chosen in practice? The theory emphasizes that a well-aligned partition (reflecting data geometry and loss) is crucial for a tight bound. Did the experiments use a specific strategy (e.g., clustering data, using class labels, or random partitions) to define $\Gamma$? How sensitive is the bound’s tightness to the choice of partition and the number of regions $K$? Any guidance on selecting or tuning the partition for a new dataset would be very helpful for practitioners.

2. What do the authors see as the main factors contributing to the gap between the bound and the true test error in the empirical results? For instance, is the remaining slack mostly due to worst-case concentration terms (e.g., the $\ln(1/\delta)$ terms or the use of union bounds) or due to suboptimal alignment/partitioning in practice? Insight into whether further tightening is possible (perhaps by refining the bound or using data-dependent $\Gamma$ choices) would clarify the significance of the remaining gap and potential future improvements.

3. The theoretical development assumes a bounded loss function $0 \le \ell(h,z) \le C$. In the experiments, it appears the 0-1 classification error (and analogous segmentation error) was used to satisfy this. How would the framework handle common unbounded losses like cross-entropy or mean-squared error? Would one need to enforce a bound (e.g., clip losses or use a surrogate) to apply the theorem, or is there an extension of the concentration inequality that could accommodate sub-Gaussian or heavy-tailed losses? Clarification on this point would delineate the practical scope of the bounds.

4. The paper demonstrates the bound on vision tasks. Do the authors anticipate any challenges in applying it to other domains such as NLP or speech, where the input dimension and structure differ significantly? Also, given the bound’s ability to track generalization, have the authors considered using it for model selection or early stopping in lieu of a validation set? For example, could one compute the bound during training (on the training data) to decide when a model is generalizing well or beginning to overfit? Exploring such an application would highlight the bound’s utility in practice and we are curious if the authors have preliminary thoughts or results in this direction.

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
This paper proposes new genereralization bounds for modern machine learning algorithms. Inspired by previous works, the proposed bounds on a partition of the data space and new concentration inequalities for multinomial distributions. The multinomial distribution naturally appears in the proofs by looking where the data points of the training set falls inside the partition of the data space. One of the bounds is tractable and computable based on the dataset and an already trained model. This allows the authors to conduct an extensive empirical study, demonstrating that the new bound is tighter than the one recently proposed by Than & Phan (2025) and that some components of the bounds are able to track the behaviour of the test error, hence, showing new iterations between data space and loss landscape.

### Strengths
- The authors propose an approach to generalization that explicitly takes into account the properties of the data distribution, which is an interesting perspective, compared to the rest of the literature.
- The proposed bound is fully computable based on a trained model and the training dataset
- In the experiments, the bound seems to capture the order of magnitude of the actual test error and to track his behavior in a relevant way.

### Weaknesses
*Main weaknesses:*
- Theorem 1 is hard to read because it involves a lot of quantities whose definition depends on each other. I believe the bound should be written in such a way that its actual rate of convergence (with respect to $n$) is explicit. The same remark applies to thm 3.2.
- In equation (1), when ones computes the third term on the right0hand side with the smallest $\delta_2$ possible, it appears to be of order $u^2 / n^3$, which in the worst case is of order $n$ (please correct me if I am wrong). Of course it also depends on the partition, but it seems that the bound can sometimes not go to $0$ when $n \to \infty$. I believe this should require some additional comments.
 - There seems to be a mistake in the proof of theorem B.5. At equation 59, in order to incorporate all the previous inequalities for $i \in T$, a union bound should be applied. Therefore, to my understanding, eq. (59) only holds with probability at least $1 - K \delta$. It seems to me that this where a factor $\sqrt{\log (K)}$ is gained compared to previous works. As this is seen as one of the main contribution and if I am correct, this problem should be fixed.
 - Proof of theorem A.1: To obtain equation (27), it is usedthat $E[\ell(h, z_{ij}) | h,n] = E_{z'}[{\ell(h, z') | z' \in Z_i}]$, where $z'$ is independent of $h$. I think this might be wrong, as conditioning with respect to the predictor is not the same as integrating over an independent data point (again, please tell me if I am mistaken).

*Other issues:*
 - The notion of partition and $K$ are mentioned in the introduction before being properly defined. 
 - Most of the related works section seems to repeat what was written in the intro and cite the same papers. I think these two sections could be merged.
- The term $mac_h$ does not read very well, maybe use $\mathrm{mac}_h$. Same remark with $gap$.

### Questions
- Beside gaining a factor $\sqrt{\log (K)}$, can you elaborate more on the main differences with Than and Phan (2025)?
- Line 139, what do you mean by measuring? Do you mean measurable?
- Can you elaborate a bit more on the rate of convergence that we obtain with Theorem 3.1
- In table 2, can you comment on why the uncertainty almost always have the same value and always the exact same value of the bound of Than and Phan (2025)?

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
2

### Summary
This paper introduces a new generalization error bound for deep neural networks and aims to address the common issue that often occurs where existing bounds are vacuous for modern architectures. The proposed bound is model-dependent and also relies on the complexity of both the data distribution and the loss function (via the loss landscape). The key technical contribution is a new concentration inequality for multinomial random variables that eliminates a $\sqrt{\ln K}$ dependency. The authors also verify their boudn through experiments on large-scale models such as ImageNet classifiers and COCO segmenters. They also argue that a "macro-level" component of their bound can be used to track the dynamics of test error during training, which is a useful metric and even more reliable than the validation error

### Strengths
The main strength of this paper is the theory, both the theoretical insights and its application to practical generalization bounds.
 - the new error bound incorporates information about both the data distribution and the loss surface. It's nice that their error bound is able to capture both 'micro-level' (ie model's performance on individual samples) and 'macro-level' (ie models average performance on regions in data space). That aspect is new to me and quite compelling. I do not believe this characterization has been explored in prior works. 
 - The authors prove a new concentration inequality for mulinomial random variables. This inequality is sharp and a key step in proving their ultimate generalization bound. The tightness of their bound is important since many generalization bounds have been shown to be vacuous and thus not useful in practice.

### Weaknesses
The paper is interesting and well-written but I didn't follow all of the mathematics line by line. There aren't so much weaknesses as points of concern or questions that came up for me. 
Firstly, the authors mention these weaknesses explicitly 
 - they mention in the conclusion that the boudn can be vacuous when dealing with small sample sizes. Namely, the third term can grow very large and dominate the RHS. 
 - The bound also requires a bounded loss function. In the case when the loss can yield infinite values the bound doens't provide meaningful information

Also, it seems that the sum on the RHS of the error bound is quite sensitive (and brittle?) to the choice of $K$. For example, given 200 partitions, the set of non-empty partitions could be large and even if the model is accurate, it is possible that almost every partition could contain one or a few misclassified samples. This would make the local error rate non-zero and summing up all these non-zero error rates could result in a large value making the bound not informative.

### Questions
- It seems that this bound and its tightness depends quite heavily on how the data space $Z$ is partitioned in the $K$ subsets. How sensitive are the experiment results to this choice of partition? and the number $K$ subsets? For example for ImageNet and COCO, $Z$ is partitioned using 200 centroid. For imagenet, the images lie in 224x224x3 dimensional space. Is there any effort to ensure that two 'similar' images are close in the Euclidean sense as well? 
 - How sensitive are these partitions to the initializations. I assume different seeds could lead to quite different partitions and thus skew the error bound results. In Table 1 and Table 2, there are errors reported with the final column "Error bound". Is this wrt to this random initialization of the centroids?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper builds on the work by Than and Phan https://arxiv.org/abs/2503.07325 and derives a slightly tigher PAC-Bayes bound than they do.   The computable bound is tightened by removing a \sqrt{\ln K} penalty. It tunes constants and K-dependence; it doesn’t change the overall paradigm.  This is a constant level improvement— not something that improves the rate. It polishes the weakest step of Than and Phan  (the computable macro term), rather than proposing a new regime where a faster decay rate is achieved.  Overall, this is a marginal improvement. 

Thus, this approach suffers from some of the same weaknesses as the paper on which it is based.  The major drawback is that, just like compression bound by Lofti et al, https://arxiv.org/abs/2211.13609, the bound is mostly collapsed to training error. And it's much worse than the compression bound in terms of tightness. That is the price you pay for a theory that bounds the actual network, rather than a compressed version. But it is still not that strong. Therefore, a marginal improvement to this paradigm is not a significant advance.

### Strengths
Their bound is slightly better than the previous one.

### Weaknesses
The novelty is weak due to the fact that the main work is in https://arxiv.org/abs/2503.07325. -- This approach builds on the same paradigm, and shows how to further tighten one of the terms.  But the paradigm it is building on is solid, but not groundbreaking.

### Questions
none

### Soundness
3

### Presentation
3

### Contribution
2
