# Angle K-Means

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 8, 2, 4, 6

## Abstract
We propose an accelerated exact $k$-means algorithm, Angle $k$-means.
As its name suggests, the algorithm mainly leverages angular relationships between data points and cluster centers 
to reduce computational overhead. Although grounded in straightforward geometric principles, 
it delivers substantial performance improvements in empirical evaluations.
In contrast to existing acceleration techniques, our model introduces no new hyperparameters,
preserving full compatibility with standard $k$-means.
Theoretical analysis shows that Angle $k$-means maintains linear time complexity 
with respect to both sample size and dimensionality, 
while empirical evaluations on diverse real-world datasets demonstrate 
significant speedup over state-of-the-art algorithms such as ball $k$-means and Exp-ns.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper provides a new way to make exact k-means faster. Consider a point, its current center, and an alternative center. The basic idea is that if the angle between the point and the alternative is large enough, and the alternative is far away from the current center, then it cannot be the a better center for the point.

### Strengths
This paper presents a new technique for making k-means faster. The idea is explained carefully and the experimental results are favorable. 

The general idea is to calculate a score for each pair of centers, and then to use these scores to recognize for each point that many alternative centers are more distant than the point's current center. This idea (lines 175 to 177) dates back to Elkan (2003) and is illustrated in Figure 1(a) of this submission. Figure 1(c) shows that the new method can eliminate additional centers.

The new idea is a significant increment over previous research.

### Weaknesses
Section 2 should not just list and summarize previous methods. It should say carefully how the methods compare in time and space complexity, and in empirical speed.

There are numerous small imperfections in the English of the submission, such as a comma splice at the end of Equation 5.

What is the point of Figure 1(b)? Why not just have 1(c)? Only 1(c) is used in the actual algorithm.

Besides Lloyd's, why were Annulus, Exp-ns, and Ball hosen for comparison and not other previous algorithms? In particular, including Elkan's method would be interesting since Figure 1(c) seems to show that Angle k-means is strictly better.

Tables 3, 4, 5, 6 would be more clear if they showed other methods as a percentage of Lloyd's as a baseline.

### Questions
Does Figure 1(c) show that the new method is a guaranteed improvement over Elkan's method (in number of distance computations, not necessarily in clock time)? Which other methods have a similar guarantee?

Tables 3, 4, 5, 6 seem to show that speedups never exceed 10x, even with thousands of centers. What is the fundamental reason for this lack of great speedup?

In high dimensions, any two random points are almost certainly almost orthogonal, so their angle is pi/2. Is this fact relevant to the algorithm of this paper? What does it imply for how often the condition will be true in line 19 of the algorithm?

### Soundness
4

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
This paper presents a simple improvement to Elkan’s triangle inequality pruning method, while avoiding the need to store O(nk) distances, using the law of cosines.They give a theoretical motivation and provide experiments, comparing the average number of distance computations per iteration and the average time per iteration to other hyper-parameter free exact k-means algorithms.

### Strengths
The running times of the experiments look very promising.

### Weaknesses
Looking at the pseudo code of Algorithm 1, the loop on line 15 only stops early when the triangle inequality bound is reached (the current center is at least twice the distance from the existing one), regardless of the angle criteria. Thus the p in the running time does not refer to the average number of candidate centers per sample according to the definition of B, but rather to the average number of candidate centers under the triangle inequality bound. This gives a theoretical running time no better than what the standard triangle inequality method would give. It is not obvious how to remedy this since the angle criteria is not monotonic over centers and so cannot be used to stop early on its own. Dealing with this issue would be a significant result.
The presentation of the paper severely obfuscates the underlying idea. Lemma 1, Theorem 2, and the case distinction on page 4 appear to be completely unnecessary since equation (10) can be derived directly from the law of cosines and  (10) is the only statement required to derive the pruning rule given in the caption for Figure 1 c) that is used in Algorithm 1. Deriving (10) from the law of cosines would make the argument far simpler and easier to follow.

### Questions
Please clarify the discrepancy between the running time of the pseudo code and the definition of p. 

Please provide an explanation for why Lemma 1, Theorem 2 and the case distinction are present.

In the abstract, it is stated that the method maintains linear time complexity wrt dimensionality, but the running times given in Section 3 are independent of dimension. Why?

What does “Setup” refer to in Table 1?

Are lines 11-13 of Algorithm 1 necessary?

Do any of the parameterised methods have sensible default parameters? It seems strange to ignore them in the experiments.

Can you explain the discrepancy between the difference in the average number of distance computations and the average running times (Table 3 vs Table 4). On some datasets (eg. Isolet, k=30), your method uses slightly fewer distance calculations, but runs over 10x faster.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Angle k-means, an exact and parameter-free acceleration of standard k-means that exploits geometric angle relations between cluster centers to skip redundant distance calculations. By precomputing center distances and angles, it preserves identical results to Lloyd’s k-means while greatly improving efficiency. Compared with Ball k-means and Exponion k-means, it achieves faster clustering without accuracy loss. Theoretical analysis shows linear complexity in data size, and experiments on 14 datasets report up to 90% fewer distance computations.

### Strengths
The algorithm relies on clear geometric intuition (angle constraints) without complex modifications.

Achieves up to 80–90% reduction in distance computations on real datasets.

Well-suited for low- and mid-dimensional large-scale datasets in industrial use.

### Weaknesses
The idea is largely derived from existing geometric pruning strategies, with only minor reformulation via angular constraints.

The approach strongly resembles recent TILB k-means methods that also exploit triangle-inequality-based geometric filtering, yet the paper provides no comparison or discussion.

The algorithm only applies to Euclidean metrics and cannot extend to cosine or Mahalanobis distances.

Angle-based pruning loses effectiveness in high-dimensional spaces.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes an angle-based acceleration approach for the K-means clustering algorithm, referred to as Angle-means. The main idea is to use angular bounds to prune unlikely candidate centers in the assignment step, thereby reducing the number of distance computations. The authors derive a theoretical bound (Theorem 2) based on geometric analysis and compare Angle-means with several existing acceleration variants, including Annulus, Exp-ns, and Ball K-means. Experimental results show that Angle-means achieves comparable clustering accuracy but faster runtime than baseline methods on multiple datasets.

### Strengths
1. Solid geometric reasoning behind the angular pruning criterion. 

2. Simple algorithmic design that is easy to implement and integrate with existing K-means frameworks. 

3. Empirical results demonstrate consistent, though moderate, runtime gains compared to several baseline acceleration methods. 

4. Theoretical guarantees (Lemma 1, Theorem 2) are sound and clearly derived.

### Weaknesses
1. The proposed algorithm is an acceleration method for exact K-means, but lacks comparison with approximate K-means algorithms (e.g., uniform sampling, coreset-based methods) in terms of runtime, clustering cost, and accuracy (ACC). Such comparisons would help clarify the trade-off between precision and efficiency.

2. While the geometric intuition is sound, the theoretical analysis remains relatively elementary. A more rigorous convergence analysis or worst-case guarantee (beyond per-iteration complexity) would significantly strengthen the contribution. In particular, it would be useful to discuss whether the parameter p in the time complexity can be expressed or bounded in expectation as a function of 𝑘.

3. When the number of clusters 𝑘 is very large, the proposed acceleration scheme seems to offer limited benefit, as the overhead of managing angular relationships may offset the gain from pruning.

### Questions
1. The paper evaluates five algorithms: Lloyd, Annulus, Exp-ns, Ball K-means, and Angle-means. The latter four are intended as accelerated versions of Lloyd. According to Table 3, all of them reduce the number of distance computations compared to Lloyd. However, in Table 4 (runtime), only Angle-means runs faster, while Annulus, Exp-ns, and Ball K-means are actually slower. Could the authors clarify why this happens? Is it mainly due to additional preprocessing or bookkeeping costs?

### Soundness
3

### Presentation
3

### Contribution
3
