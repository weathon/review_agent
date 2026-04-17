# From Sorting Algorithms to Scalable Kernels: Bayesian Optimization in High-Dimensional Permutation Spaces

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 8

## Abstract
Bayesian Optimization (BO) is a powerful tool for black-box optimization, but its application to high-dimensional permutation spaces is severely limited by the challenge of defining scalable representations. The current state-of-the-art BO approach for permutation spaces relies on an exhaustive $\Omega(n^2)$ pairwise comparison, inducing a dense representation that is impractical for large-scale permutations. To break this barrier, we introduce a novel framework for generating efficient permutation representations via kernel functions derived from sorting algorithms. Within this framework, the Mallows kernel can be viewed as a special instance derived from enumeration sort. Further, we introduce the \textbf{Merge Kernel} , which leverages the divide-and-conquer structure of merge sort to produce a compact, $\Theta(n\log n)$ to achieve the lowest possible complexity with no information loss and effectively capture permutation structure. Our central thesis is that the Merge Kernel performs competitively with the Mallows kernel in low-dimensional settings, but significantly outperforms it in both optimization performance and computational efficiency as the dimension $n$ grows. Extensive evaluations on various permutation optimization benchmarks confirm our hypothesis, demonstrating that the Merge Kernel provides a scalable and more effective solution for Bayesian optimization in high-dimensional permutation spaces, thereby unlocking the potential for tackling previously intractable problems such as large-scale feature ordering and combinatorial neural architecture search.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the problem of black-box optimization over permutation spaces. Such problems play a role, for instance, in compiler optimization or chip design, where one searches for a permutation that optimizes some quantity of interest, which is expensive to evaluate. Bayesian optimization (BO) is a popular technique for black-box optimization, but has received little attention in the context of permutation spaces. A popular exception is [1], which uses the Mallows (BOPS-H) and Kendall kernels (BOPS-T) discussed in [2] to perform Bayesian optimization over permutation spaces. BOPS-H performs considerably better and is defined in terms of the number of discordant pairs, resulting in $\frac{n^2-n}{2}$ possible candidates.

In this paper, the authors propose a different approach. Given some permutation, they employ merge sort to transform the permutation into the identity permutation and construct a feature vector by appending a 1 if, during a merge operation, an element is chosen from the left vector, and a -1 otherwise. This way, the number of possible candidates is reduced to $\mathcal{O}(n\log n)$ (worst-case complexity of merge sort). Since this representation does not capture all invariances the Mallows kernel exhibits, the authors apply three tricks to augment the feature vector with additional features that are designed to capture such invariances, such as right-, cyclic-rotation, and cyclic shift invariances.

The authors compare their method to BOPS-H and show that it achieves better simple regret and regret AUC values. They further conduct an ablation study to study the effect of the additional 'invariance'-features, showing that the 'shift-histogram module' leads to the highest performance degradation upon removal, indicating that it is the most expressive added feature.

### Strengths
- The paper addresses a relevant but little-discussed problem setting.
- The paper is mostly clear and well-written.
- The paper clearly defines the boundary between related work and its own contribution.
- The proposed approach outperforms the state-of-the-art in BO over permutation spaces.

### Weaknesses
- While the approach is smart, it is hard to get an intuition for the feature vectors proposed in this paper. A '-1' or '1' in the $\Phi_{\textrm{Mal}}$ vector has an easy-to-grasp interpretation. In contrast, the $\Phi_{\textrm{Mer}}$ feature vector is a concatenation of features originating from different levels in the merge-sort operation.
- The addition of the 'tricks' makes this work seem more heuristic than BOPS-H.
- While the proposed approach outperforms BOPS-H, it only does so by a slight margin. The differences between BOPS-H and COMBO, for instance, are considerably larger than the difference of 'Merge' to BOPS-H, which, judging from the figures and tables, could be a coincidence. The paper is not very critical about its empirical performance.
- The explanation of the additional features is quite brief. Some examples would make it easier to understand their design.

### Questions
- Clearly, the $\Phi_{\textrm{Mer}}$ feature vector does not capture invariances (as acknowledged by the authors), motivating the need for additional features. Does $\Phi_{\textrm{Mal}}$  have similar shortcomings? How do they relate to
?
- In the conclusion, what do you mean when saying that gradient methods could exploit the lower-dimensional feature space?

Other comments:

- For the ablation study, it would be interesting to see the performance without any tricks.
- It shouldn't be hard to construct an ARD version of Eq. 4, potentially sharing one length scale for $\Phi_{\textrm{Mer}}$ and using separate length scales for $\Phi_{\textrm{Spi}}$, etc. Studying their length scales would make it easier to quantify the importance of each feature group and might even improve performance.
- Eqs 1 and 3 are only equal when re-scaling $\ell$.

### Soundness
3

### Presentation
2

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
The authors propose a new embedding of permutations to parameterize an RBF kernel, and compare bayesian optimization against kernels using simpler / slower permutation embeddings.

### Strengths
The method is very well-motivated and performs well on the high-dimensional setting, and stays competitive on the smaller dimensional tasks.  I’m also happy the authors acknowledged the lack of right-invariance of the method, which doesn’t invalidate the embedding but at first blush looks a little surprising.  Their discussion of sacrificing this property for the sake of a fast method is convincing.

### Weaknesses
I’m somewhat surprised the authors didn’t benchmark with other notions of permutation distance, for example they mention Spearman’s footrule on page 5, and there are other notions like the Cayley distance.  The results would I think be a bit stronger if it was demonstrated that a new notion of distance between permutations is strictly necessary to get better performance in bayesian optimization.

The lack of right invariance is still concerning, mainly as it makes it very difficult to interpret what the difference between merge embedding is actually calculating.

### Questions
Would it be possible to consider a simple randomized baseline version of the Mallow kernel?  A small concern I have with the method is that, because it’s not super clear what distance the Merge kernel captures, it may be introducing some extra element of stochasticity that is helpful for exploration in BO.  A comparison to a version of the Mallow kernel where one randomly selects O(n \log n) features out of the total O(n^2) features and only uses these for the permutation embedding would be a useful benchmark to show the proposed method is doing something more substantial, and that one couldn’t naively get the reduced complexity.

### Soundness
3

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
4

### Summary
The paper shows how sorting algorithms can direct the design of GP kernels for permutation spaces, and that in particular merge sort produces a MergeKernel that can be used for compact modeling in high-dimensional permutation spaces.

The baseline method (the Mallows kernel) has a higher-dimensional representation and performs worse for BO.

### Strengths
The paper introduces what is to my knowledge a novel connection between sorting algorithms and permutation kernels. This connection is very interesting, original, and useful.

The fact that the primary existing permutation falls out as a special case in this framework is very interesting and provides strong validity for thinking about permutation kernels this way.

The paper is clear and well written.

### Weaknesses
The empirical evaluation is not completely satisfying.
- Of five problems, only 2 shows significant differences between the methods. The conclusion would be that most of the time it doesn't matter what kernel you use? I recommend trying to expand the set of problems further to include more with clear differences in performance. The paper hypothesizes that the MergeKernel does better in high-dimensional spaces, but I did not find that very convincing based on a signal of just 2 of 5 problems having different performance. Please add a couple more high-dimensional problems to confirm the hypothesis that MergeKernel is important and better in high-dimensional settings.

* Comparison is only with the Mallows kernel. The paper tries to justify this by saying that it is focused specifically on sorting-based methods, and so only needs to compare sorting-based methods. But I don't think that is correct. The end of the introduction states "Our results demonstrate that the Merge kernel provides a practical and efficient tool for permutation optimization, significantly enhancing BO’s applicability to diverse AI scenarios." So, the paper is claiming that the method "significantly enhances" BO for permutation optimization. Generally, not just compared to only other sorting-based methods. The paper does not provide evidence for that claim. The paper mentions TurBO - it has an implementation available and is fast to run and may work well in the high-dimensional spaces where Mallows kernel performed poorly. Running that on the benchmark problems would significantly strengthen the results.

* One other example of a kernel derived from a sorting algorithm would really emphasize the generality of the result.

* The paper describes that the merge kernel is not invariant under right multiplication and that as a result it "sacrifices a certain degree of performance." I don't have good intuition for what type of issues that will cause downstream in the BO, I think it would be helpful to give some examples in the paper of how lack of right invariance can cause modeling difficulties.

### Questions
*  What do the effects of loss of right invariance look like in practice?

### Soundness
3

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
4

### Summary
This paper addresses the scalability issue of Bayesian Optimization (BO) over high-dimensional permutation spaces by proposing a novel sorting algorithm-based kernel design framework. This framework sophisticatedly generates fixed-length feature vectors for permutations via the internal binary comparison results from any sorting algorithms, which theoretically interprets the traditional Mallows kernel as its special case with an $O(n^2)$ feature dimension. Based on this framework, the authors further propose a novel Merging Sorting-based kernel (i.e., Merge kernel), with only $O(n \log n)$ feature dimensions. Incorporating the proposed kernel, permutations can be assigned theoretically shortest features with limited information loss, which achieves the lower bound from the perspective of information theory. Experiments show that the proposed kernel can achieve competitive performance than the traditional Mallows kernel on both low-dimensional and high-dimensional problems, yet with better convergence speed and final regrets, providing potential solutions for large-scale permutation optimization problems such as feature sorting or neural architecture searching.

### Strengths
1. This paper addresses a critical challenge for Bayesian Optimization over permutations, showing potential in efficient permutation encoding and searching.

2. The idea is novel and interesting. The authors are the first to incorporate traditional sorting algorithms into permutation kernel design, forming a powerful and general framework. It provides a new paradigm for kernel functions over permutation spaces.

3. The motivation is clear and easy to follow. The proposed general framework interprets the traditional Mallows kernel as a special case, which reveals the intrinsic limitation of prior techniques and well motivates the proposed kernel design.

4. The proposed kernel is efficient and can generate the shortest permutation encodings, achieving the theoretical information lower bound. This would pave the way for future exploration of complex, large-scale permutation-related combinatorial optimizations.

### Weaknesses
1. While the authors propose a kernel based on Merging Sort and achieve competitive performance, the generality of the selection of sorting algorithms can be discussed in more detail, especially for some stochastic sorting algorithms that contain non-fixed times of binary comparisons. In other words, since the authors make connections between the traditional sorting algorithms and the permutation kernel, readers may be interested in how different properties of sorting algorithms affect the performance.

2. I recommend that the authors conduct a further scaling experiment, in which the explicit relationship between the permutation scale and running time can be studied. This exploration would provide valuable suggestions for practitioners to check whether the proposed framework is suitable for their real-world problems.

3. The quality (in latex) of tables can be significantly improved, e.g., Tab. 2.

### Questions
1. Can we design a technique to randomly select and record the binary comparison? Can the proposed Merging kernel be further improved by incorporating some stochastic binary comparisons? Or can the authors give a further analysis on which pairs of comparisons are critical for a given permutation optimization problem? Based on this analysis, can we add some extra important yet limited number of binary comparisons (e.g., top-K important pairs) to supplement the information loss of permutations due to the loss of right-invariance?

2. See Weakness.

### Soundness
3

### Presentation
3

### Contribution
3
