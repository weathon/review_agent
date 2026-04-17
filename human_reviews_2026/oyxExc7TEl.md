# An Efficient SE(p)-Invariant Transport Metric Driven by Polar Transport Discrepancy-based Representation

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 8, 4

## Abstract
We introduce SEINT, a novel Special Euclidean group-Invariant (SE(\emph{p})) metric for comparing probability distributions on $p$-dimensional measured Banach spaces. Existing SE(\emph{p})-invariant alignment methods often face high computational costs or lack metric guarantees. To overcome these limitations, we develop a polar transport discrepancy combined with distance convolution to extract SE(\emph{p})-invariant representations. These representations are then used to compute the alignment between two distributions via optimal transport. 
Theoretically, we prove that SEINT is a well-defined metric on the space of isometry classes of normed vector spaces. Beyond its inherent SE(\emph{p})-invariance, SEINT also supports cross-space distribution comparison.
Computationally, SEINT aligns two samples of size $n$ with a complexity of just $\mathcal{O}(n\log n)$ to $\mathcal{O}(n^2)$. Extensive experiments validate its advantages: As a robust metric, it outperforms or matches existing SE(\emph{p})-invariant methods in classification and cross-space tasks under isometries. As a regularizer, it greatly enhances molecular generation performance across both pre-training and fine-tuning tasks, achieving state-of-the-art (SOTA) results on key benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a method to compare structured objects, e.g., graphs, shapes, etc. This is a hard task that is not solvable with standard optimal transport (OT) tools, but that requires tools such as the Gromov-Wasserstein (GW) distance (Memoli, 2019). Notably, the GW distance and its variants are computationally challenging. While more efficient methods exist, they often have severe shortcomings. As a remedy, the authors propose a method that relies on comparing two structure objects via pairwise distance matrices. In particular, they sample from a target reference 1D distribution to then map each distance matrix to 1D distribution. Then, they can compute the efficient 1d Wasserstein distance between the resulting 1D projected distance distributions. From that point of view, it can be considered as an interesting adaptation of sliced GW. 

I think that this is a really interesting idea. While slicing is an important method in the field of OT, the so-called 'sliced GW' (SGW, Vayer et al., 2019)  method has shortcomings. While there are other very recent 'sliced GW' approaches out there that have been missed by the authors (Piening et al., 2025), the authors certainly showed the advantages over the SGW.

However, my main criticism is a lack of experimental validation. The experiments are very synthetic. As this is often the case with GW, this is not my main concern, however. More importantly, the authors failed to quantitatively compare their method against suitable alternatives. They mentioned GW (Memoli, 2011), entropic GW (Peyre et al., 2016), and low-rank GW (Scetbon, 2022), but did not really benchmark against them. The horse sequence is a nice visualization, but does not replace benchmarks on shape/graph classification.

Therefore, I feel as if this is a really nice idea, but the paper lacks substance beyond the introduction of the new metric. In particular, I would have expected either more theoretical contributions or a solid benchmarking to vote for acceptance.

### Strengths
- A good and interesting idea that not everyone would have come up with.
- A nice supplementary section on the OT background
- A interesting discussion of possible extensions in the supplement
- The code in the supplementary is accessible and mostly complete. That is very helpful!
- A nice introduction and categorization of existing methods.
- A well-motivated methodological section.
- Well-done visualizations, especially Fig. 1 & 2
- Proof section is well-structure. While the main theoretical contribution (Metric property) seems rather simple, the ‘identity of indiscernibles’ is not straightforward.

### Weaknesses
*Title*
- I have my doubts about the title. Despite background knowledge, the lengthy title was more confusing than helpful for my initial understanding. I would recommend dropping ‘DRIVEN BY POLAR TRANSPORT DISCREPANCY-BASED REPRESENTATION’ since ‘polar transport’ is not even well-known in the OT community, the word ‘transport’ appears in the title twice, and it would make the title more understandable overall. This is only a minor remark and question of style, however.

*Introduction*:

The introduction is well-written and relies on helpful categorization of existing methods. However, the authors did leave out a lot of modern methods.

- While the broad classification visualized in Figure 1 gives a nice overview and I would mostly agree with it. However, I feel as if the authors missed key references:
- - ‘Extrinsic Strategy’: While I acknowledge the authors’ efforts in citing key references from ‘90s, more recent research in the same direction has been overlooked, see papers on ‘Wasserstein Procrustes’ methods (Grave et al., 2019; Alvarez-Melis et al, 2019) and extensions beyond Euclidean spaces (Alaya et al, 2022; Beier et al., 2025). While these methods indeed rely on costly iterative, non-convex optimization iterations, another line of research (Beckmann et al., 2025) enables *efficient* comparison of Euclidean data via the ‘Normalized Radon CDT’.
 - - ‘Intrinsic Strategy’: Again, Memoli’s Gromov-Wasserstein (GW) paper from 2011 is a very important paper for comparing geometries. Nevertheless, a lot of effort has gone into efficient adaptations of this approach. While the paper mentions GW upper bounds (Li et al., 2023b; Peyré et al., 2016; Scetbon et al., 2022), GW lower bounds (Sato et al., 2020; Piening et al., 2025) have not been considered in the current submission, but they seem to enable rather efficient comparison of intrinsic geometries, while avoiding the non-convex optimization schemes of SGW/RISGW/GW/EGW/low-rank GW.

- Table 1: While this Table gives a nice overview, I have a few comments (from less important to more important): 
- - 1.) The complexity of EMD solvers depends on the problem. I think that for $n \neq m$ the complexity is not directly cubic, but in $\mathcal{O}(n^3 log n)$.  
- - 2.) The complexity of some metrics (especially sliced ones) is also dimension-dependent in a certain sense, whereas this is not the case for all of them. As an example, this should be the case of RISGW vs. GW. 
- - 3.) A key disadvantage of RISGW, GW, and other methods is that they rely on non-convex solvers. 
- - Conclusion: While the table is helpful for the reader, I feel as if further information should be given about convexity/convergence guarantees and dimensional dependence.

- Please note the ‘Sliced Gromov-Wasserstein’ paper has been authored by ‘Titouan Vayer’ and should probably be cited as ‘Vayer et al.’. To my knowledge, the incorrect citation as ‘Titouan et al’ is a persistent error on Google Scholar.

- I would like to point out that the aforementioned sliced GW variant (Piening et al, 2025) employs similar ideas based on projecting pairwise distance matrices to 1D distributions and computing 1D Wasserstein distances. While there are certainly differences, I think that they should be (briefly) discussed, maybe in the background section.

*Experiments*:
The experiments are doing a good job of illustrating key properties. However, there is most certainly a large comparative gap.
- The authors only compared against ‘standard’ OT metrics	 (which are not supposed to be isometry-invariant) and variations of sliced GW (SGW, Vayer et al, 2019). However: 1.) SGW is not isometry-invariant and 2.) the (RI)SGW algorithms has some theoretical problems (see the correction note on arXiv and (Beinert et al., 2022).
I feel as if there should be a benchmarking against another ‘extrinsic’ metric, e.g., ‘Wasserstein Procrustes’ (Grave et al., 2019), against GW solvers (Memoli; Peyre; Scetbon; Sato; Kerdoncuff; Chowdhury …), and against another isometry-invariant slicing-based metric (Piening et al., 2025; Beckmann et al., 2025).
- The presented experiments are highly synthetic. 1.) The horse-gallop experiment is a nice visualization, but not really suitable for benchmarking.  I would also recommend citing existing variations of this experiment, e.g., (Vayer et al., 2019). Moreover, the high-dimensional experiment is rather ‘easy’ as it only relies on normal distributions.
- The experiment on graph generative models is interesting, but again, it is only a very simple ‘benchmarking’ (SEINT vs L2 regularization)

### Questions
- Why are some entries in Table 1 red and others black?
- Why is (E)GW accuracy missing from Table 2? Almost all other reported distances are not isometry-invariant, and the one exception (RISGW) relies on solving a highly non-convex optimization problem.
- Section D.4 ‘Metric Consistency’: I do not know what the term ‘metric consistency’ is supposed to mean here. I feel as if the reported experiments illustrate the *continuity* of the distance. While it is good to see that the empirical estimator displays continuity, I would be more interested in a proof of the continuity. Did you look into it, or did I miss it in the paper?
- Sliced Wasserstein (SW) suffers from a curse of dependence. Can you comment on the dimensional dependence?
- Often the quality of Monte Carlo methods, e.g., SW, scales with $\frac{1}{\sqrt(\text{Projection Number}$ (Nadjahi et al., 2020). This is a simple result of Hölder’s inequality. Could you get similar results for your estimates?
- Can you incorporate graphs describing geodesic distances similar to experiments in (Memoli, 2011), (Beier et al., 2022), or (Piening et al, 2025)?
- Can you extend your setting to the setting of 'Fused Gromov--Wasserstein' (Vayer et al, 2020)?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a computationally efficient approach for a representation based metric  for the problem of comparing probability distributions that lie in different Banach spaces invariant to orthogonal transformations. The method involves looking at the optimal transport between one-dimensional measures over norms and each distributions  norm distribution. The contribution to the transport cost for each point in the distribution provides a non-negative value, and this is distance convolved to provide a representation that is isometric consistent and dimension independent. Solving an optimal transport between the two measures using this representation provides a way to align the two distributions.

### Strengths
The paper addresses a challenging problem that can be applied to point cloud registration and outperforms Gromov-Wasserstein distance. It is clearly written, original, and results are strong.

### Weaknesses
There are a few minor points to clarify.

### Questions
In the algorithm the choice of generating reference distributions as uniform (line 290) is not discussed. Corollary 2 does not mention how the supports $\{z_j\}_{j=1}^k$ are chosen. Is uniform the simplest, why not quantiles of of the empirical norms? 

Line 52 "it may sacrifice the metric properties of the distance"  but in Table 1 it is stated that it isn't a metric. Thus 'may' is not correct. 

Line 147 should state $\lVert X \rVert_X$ and $\mu_Z$ 

Line 157, the notation with norm and $\cdot$ breaking across the line is hard to parse. 

Line 278, it was not previously assumed that the space is $\mathbb{R}^{n\times p}$. While the special Euclidean invariant assumes this, at other times it would seem that the theory is applicable to complex-valued vector space (or possibly other field) which is not necessary in the discrete setting. This should be clarified. 

Minor extra space around the superscript for the footnote 1.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new variant of an Optimal Transport (OT) metric that is Special Euclidean group-Invariant (SE(p)). The main idea of the authors is to proceed in two steps. The first one is to align the original raw samples with a certain reference measure, where the cost function is defined as a difference between a given vector and a coordinate of the vector from the reference measure in 1D. The second step is to convolve the obtained embedding with the raw cost matrix and to use the obtained embedding to define a cost function for the resulting 1D OT problem. This allows the metric to be SE(p), yet it is to be computed efficiently in practice. The algorithmic implementation then takes the sup over the reference measures to calculate the final value of the metric. The experiments, both qualitative and quantitative, showcase the desirable properties of the proposed metric.

### Strengths
1.	The proposed metric seems to be novel
2.	Proposed metric has metric properties and is cheap to compute
3.	Experimental evaluations provide a multi-faceted view on the proposed metric.

### Weaknesses
1.	The introduction of PTD and SEINT is very hard to parse in the main paper. In this sense, I found the way SEINT was presented in the Appendix very helpful. Indeed, on page 22, we see that SEINT is a GW variation where the second transposed coupling matrix is replaced with a product of the two convoluted embeddings depending on it. The text makes the introduction of this metric overly complicated, at least for me. Once we see it through this lens, a natural question is why SEINT outperforms GW? GW is theoretically a SE(p) metric too. It would be great to see some intuitions for why it fails in shape matching tasks, in addition to the favorable computational complexity of SEINT.
2.	To the best of my understanding, the metric is defined as an inf sup problem, yet it is solved algorithmically as a sup inf problem. This can be seen from Algorithm 1 (we solve inf inside the loop and then take max over the solutions). Shouldn’t we use minimax (von Neumann or Sion theorem) somewhere to justify doing this? In general, we know that sup inf <= inf sup, so we cannot use the majorization-minimization argument either, unless f is convex/concave in the first and second argument, which is unlikely given how the problem is defined.
3.	The authors omit some very important GW variations in their literature survey. One such omitted reference that comes to my mind is the Quantized GW (Chowdhury et al.), which is extremely computationally efficient and was proposed, among other applications, for shape matching. Another one is the Sampled Gromov-Wasserstein paper (Kerdoncuff et al.). Its complexity is quadratic in N, and it approximates GW.

### Questions
1.	What is the intuition of defining PTD? This is a crucial step in the paper, yet it is not explained very well. Why do we use the cost function defined as the difference between the norm of a vector in X and a real value of the reference measure?
2.	Why are other more efficient GW variants not considered in the evaluations? I can move my score to 6 if a reasonable justification/additional experiments are provided here. 
3.	I would like to understand better the algorithmic implementation in the sense of my sup inf remark above. Can authors elaborate more on this? 
4.	I had to go to the appendix (page 22) to see how the reference measure \mu_Z is defined. They seem to be uniform measures over the hypercube. Is there any motivation for this?

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
This paper introduces SEINT (SE(p)-Invariant Transport), a novel metric for comparing structured data that maintains invariance under Special Euclidean transformations. The authors develop two unsupervised representation techniques—Polar Transport Discrepancy (PTD) and Distance-convoluted PTD (DcPTD)—that generate 1D SE(p)-invariant features through optimal transport. They rigorously prove SEINT satisfies metric properties (identity, symmetry, triangle inequality) on isometry classes. Experiments demonstrate SEINT achieves 100% accuracy in transformation recognition while maintaining computational efficiency, and improves molecule generation validity by 3-4% over EDM baselines. The approach effectively balances theoretical soundness with practical utility across multiple domains.

### Strengths
**Theoretical Innovation and Rigor**: The paper makes a significant theoretical contribution by establishing SEINT as a proper metric with formal proofs (Section 2.3, Theorem 1). Unlike prior work that "fail to simultaneously optimize for computational efficiency, rigorous metric properties, and general applicability," SEINT achieves this crucial balance. The authors cleverly bridge optimal transport theory with SE(p)-invariance requirements, providing Corollary 2 that establishes metric properties for discrete distributions under realistic assumptions. This theoretical foundation elevates the work beyond heuristic approaches common in the field.

**Practical Impact and Validation**: The paper demonstrates compelling real-world utility across diverse domains. In molecule generation (Table 5), SEINT regularization improves validity from 93.4% to 96.7% while maintaining atom stability. For point cloud classification (Table 6), it consistently enhances accuracy across multiple datasets (e.g., +0.61% on Scan-Objonly). Notably, SEINT achieves perfect (100%) transformation recognition accuracy while being computationally efficient—outperforming alternatives like SW and SGW that "fall below 60% accuracy." This practical validation across multiple challenging tasks underscores the method's significance.

### Weaknesses
**Limited Analysis of the Validity-Uniqueness Trade-off**: While Table 5 shows SEINT improves validity but decreases uniqueness (98.2% → 92.0-93.2%), the authors only briefly mention this trade-off as "acceptable." A more thorough investigation would strengthen the work—specifically, quantifying how much uniqueness degradation is attributable to the SEINT regularization versus other factors, and whether certain molecular properties correlate with this trade-off. Including visual examples of molecules where validity improved at the cost of uniqueness would make this analysis more concrete.

**Insufficient Comparison to Recent SOTA Methods**: The evaluation compares against GW, EGW, RISGW, SW, and SGW, but misses several relevant recent approaches like UniGEM (Feng et al., 2024) mentioned in the introduction. Given the rapid evolution in this field, the authors should benchmark against at least 2-3 additional recent methods (e.g., from ICLR/NeurIPS 2024) to better position SEINT's novelty and performance advantages.



**Ambiguity in Reference Measure Selection**: While Section 2.3 discusses reference measure selection, the practical implications of different choices aren't fully explored. The paper states "the support points for µZ are drawn from [0,M]^k" but doesn't analyze how sensitive results are to k (number of reference points) or distribution type. Figure 5 shows some effects, but a more systematic ablation study would strengthen the method's practical guidance.

While the weaknesses I identified (limited analysis of validity-uniqueness trade-off, insufficient comparison to very recent SOTA methods, and reference measure sensitivity) should be addressed fairly easy none fundamentally undermine the contribution. The authors' argument that the validity-uniqueness trade-off is acceptable (since validity is more critical for chemical plausibility) appears reasonable.

### Questions
1. In Corollary 2, you assume "values of the norm ||x_i|| are distinct"—how would SEINT behave when this assumption is violated in practice (e.g., symmetric molecules with identical atom distances)? Could you provide a mathematical bound on the error when norms aren't perfectly distinct?

2. The complexity analysis mentions "n log(n)" scaling, but how does SEINT perform in high-dimensional settings (p > 100)? Have you tested scalability to non-geometric data where p is large but structure exists?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes SEINT as a Special Euclidean group-invariant distance between probability measures supported on normed vector spaces. The method builds 1D, rotation/translation-invariant features in two steps: (i) Polar Transport Discrepancy (PTD), which couples the norm distribution of data with a 1D reference measure; (ii) a Distance-convoluted PTD (DcPTD) that computes the convolution of pairwise distances against PTD to encode intrinsic geometry. The SEINT distance is then defined as the optimal transport (OT) distance between DcPTD representations, with a min-max structure that adversarially chooses a “least favorable” reference measure. Theoretically, SEINT is a proved to be a metric on isometry classes, which respects translational and rotational invariances. A discrete variant of SEINT provides metric guarantees under distinct-norm and support-size assumptions. Algorithmically, given sample size $n$, the proposed implementation with a single reference measure achieves $O(n^2)$ computational complexity in general and $O(n \log n)$ when the ground distance admits additional structure (leveraging structure specific matrix decompsoition technqiues for accelerated computation). Empirically, the paper reports perfect rotation/translation robustness on a benchmark 3D point cloud classification task, cross-space (2D - 3D) comparisons, favorable perfomance scaling with dimension, and improved 3D molecule generation based on SEINT regularization.

### Strengths
The paper proposes a rotation invariant metric capable of comparing distributions across different ambient dimensions, with theoretical guarantees that establish the metric properties and rotation invariance of SEINT.

### Weaknesses
1. Although the paper claims both rotatioanal and translational invariances as its target invariance choices, the definition of SEINT as well as the applications of SEINT in experiment rely on explicit centering of the data points in the source and target domains, so that both source and target has their respective origins as the center of their corresponding distributions. The definition of PTD in Equation 2 implicitly assumes this, sincethe norm is centered around 0, and not around any arbitrary constant. Therefore, it seems that the claimed translation invariance of SEINT follows as a consequence if this origin-based centering approach, and not as a derived property that holds in greater generality. The paper should either clarify whether the definition of SEINT still works without origin-based centering, or restrict the claims to only rotational invariance.
2. The discussion at the beginning of Section 2.3 (Lines 222-241) is dense and not well-written. Given the importance of the role of the reference measure, it is worth improving the presentation of this section of the text and being clear about the existence and selection fo the reference measure.

### Questions
Please see the Weakness section, particularly the point about the translation invariance of SEINT.

### Soundness
2

### Presentation
2

### Contribution
2
