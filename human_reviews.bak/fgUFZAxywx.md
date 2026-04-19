# Linear Spherical Sliced Optimal Transport: A Fast Metric for Comparing Spherical Data

- Decision: Accept (Spotlight)
- Scores: 6, 10, 6, 8

## Abstract
Efficient comparison of spherical probability distributions becomes important in fields such as computer vision, geosciences, and medicine. Sliced optimal transport distances, such as spherical and stereographic spherical sliced Wasserstein distances, have recently been developed to address this need. These methods reduce the computational burden of optimal transport by slicing hyperspheres into one-dimensional projections, i.e., lines or circles. Concurrently, linear optimal transport has been proposed to embed distributions into $L^2$ spaces, where the $L^2$ distance approximates the optimal transport distance, thereby simplifying comparisons across multiple distributions. In this work, we introduce the Linear Spherical Sliced Optimal Transport (LSSOT) framework, which utilizes slicing to embed spherical distributions into $L^2$ spaces while preserving their intrinsic geometry, offering a computationally efficient metric for spherical probability measures. We establish the metricity of LSSOT and demonstrate its superior computational efficiency in applications such as cortical surface registration, 3D point cloud interpolation via gradient flow, and shape embedding. Our results demonstrate the significant computational benefits and high accuracy of LSSOT in these applications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
A fast metric is proposed for spherical data, based on sliced optimal transport.

### Strengths
The math appears interesting and valid. Spherical data are of interest in various contexts.

### Weaknesses
While the metric appears compariatively fast in comparisons of pairwise distance calculation, it does not address the order of complexity of performing many distance calculations given a dataset of N points, which is often the limiting factor in practical contexts. For example here nearest neighbor search would be O(N), whereas other methods particularly indexing in the context of brain data similary using the same ADNI data offer O(logN) performance.

### Questions
To discuss:
-  Invariance of this metric to nuisances such as rotation. 
-  Is this a metric or pseudo-metric, if a pseudo-metric (as in proposition D1?) why not state this in the title?
-  Discuss invertible spherical harmonics, filter banks, which can be used to form a metric and are used on spherical brain data [1].
-  the order of computational complexity of this metric in a practical spherical data retrieval scenario. Apparently it would require O(N) search time in N the number of data, other fast methods (non spherical) used on the same ADNI brain data require O(log N) [2].

Please demonstrate the utility by identifying the incorrectly labelled ADNI subjects[1], these will immediately stand out as unusually close according to a valid metric as they are the same subjects incorrectly labeled as different[2].

The paper states “the Dice loss to impose biological validity”, whereas Dice and other tissue overlap measures specifically do not enforce biological validity as they permit arbitrary biologically implausible mappings[3].

[1] Yeo, B. T. T., Ou, W., & Golland, P. (2008). On the construction of invertible filter banks on the 2-sphere. IEEE Transactions on Image Processing, 17(3), 283-300.

[2] Chauvin, Laurent, et al. "Neuroimage signature from salient keypoints is highly specific to individuals and shared by close relatives." NeuroImage 204 (2020): 116208.

[3] Rohlfing, Torsten. "Image similarity and tissue overlaps as surrogates for image registration accuracy: widely used but unreliable." Medical Imaging, IEEE Transactions on 31.2 (2012): 153-163.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper combines ideas from sliced optimal transport theory and linear optimal transport and introduces the idea of linear spherical sliced optimal transport. Specifically, they extend a recently proposed work (Martin et al. 2023) that introduced the linear circular optimal transport for embedding  circular distributions into a L^2 space. In this paper, the authors propose a new embedding method for spherical probability distributions. 

The authors present experimental results for the method of linear spherical sliced optimal transport (LSSOT) in applications such as point cloud registration and cortical surface matching.

### Strengths
This paper presents an elegant construction of the spherical sliced embedding in a linear sense. 

The main strength of the paper is the clever extension of the sliced optimal transport and linear circular optimal transport to the case of spherical distributions. This is novel.

There exist other methods for computing distances between spherical distributions, for e.g. spherical sliced-Wasserstein, Stereographic sliced-Wasserstein which also allows to efficiently compute distances. Other works have achieved further computational efficiency due to the linear distance between the circular transport embeddings. In this paper, the authors first obtain an embedding of the spherical sliced measures P(S^{d-1}) and then define a distance between the measures as the linear circular optimal transport distance between the pushforward maps of the spherical measures. While this paper borrows and combines ideas from two earlier works, the novelty of this approach is that in this instance it is achieved for embeddings of spherical distributions. 

The LSSOT distance is a pseudo-metric on P(S^{d−1}) as proven in an earlier work (spherical sliced Wasserstein), since they use the same notion of the distance. In this work the authors show that it is a well-defined metric for probability measures. Also see weakness. 

The results of Theorems D.2 and D.3 and the corollary B.7 are important as they ensure that the pushforward maps can be written in terms of the Radpm transform operator R. 

The paper is written in great detail and provides appropriate background for the concepts that are need to understand it. This is a strength. 

The experimental results are somewhat comprehensive  and the authors do a good job of comparing their method with other spherical slicing methods (Wasserstein etc.). 

The authors also provide a time complexity and runtime analysis and compare them to those with other related methods.

### Weaknesses
This is a minor weakness, but as mentioned in the strengths above, the LSSOT is a metric only for probability measures. Thus the data has to be first converted to distributions, which is both a lossy transformation (which the authors acknowledge). 

By definition, the projection is not invertible. That is one cannot recover the original signal/shape. Again, this drawback exists for all the other methods, including those that the authors build upon. 

Because of the above 2 points, when the authors present the interpolation results, they resort to an auto-encoder that projects the original data to the nearest point in the spherical latent space. Then to reconstruct the intermediate points along the interpolation path between the two signal, they need a decoder. This is similar to the backprojection approaches, which are not ideal. But again this is not a major weakness, but currently, this is the most convenient approach. 

In addition to the above loss of information, the projection step (slicing) will also introduce further discrepancy. However, this is a drawback for all the slicing based methods (including spherical) that are mentioned in the paper. In this paper, the authors use the same idea as Bonet et al. 2023 for performing the projection. 

While the authors do provide algorithms for computing flows using Riemannian gradient descent, this is technically not a geodesic, but an approximation. 

The notation for P^C (Equation 6) and P^U (Equation 7) gets confusing. The first one is a geodesic distance, while the second one is a projection.

### Questions
This question pertains to the original circular optimal transport construction, and the authors do not have to answer it here, as they do something different in this paper. However, out of interest, why is the COT distance defined in terms of c(x, y) uses min{|x−y|,1 −|x−y|}? Can some other distance be used here? For e.g. min(arccos(<x, y>)). Is that the main difference between this earlier work and  the spherical sliced Wasserstein (SSW) approach? 


The computational efficiency of this approach also naturally follows from the idea of slicing (projections on the great circle). Further efficiency is achieved by replacing the circular optimal transport distance by the linear version. However, can the authors provide slightly  more details on why the complexity of spherical sliced Wasserstein better than sliced Wasserstein? The projections on spheres should be more costly than lines. 

Can the authors comment on why LSSOT has better accuracy than SSW? Should SSW in fact perform as well as or better than LSSOT? 
For SSW, the COT distance measures the discrepancy between the push-forward maps of the spherical measures, while LSSOT uses the LCOT distance. So there is yet one more embedding ( transport a reference measure to a target measure) performed. Does this result in loss of accuracy? Is this dataset dependent? Can the authors provide more detailed analysis or any ablation studies to explain this result? 

Can the authors expand more upon the injectivity of R? This is related to the results in theorems D.2 and D.3 and is inspired by the theorem in Bonet et al. 2024.  For e.g. the map R: L^1(S^{d−1}) → L^1(V_2(R^d) ×S^1) is not injective everywhere, although it is true for absolutely continuous measures on the sphere. What are the implications when the operator R is applied in practice? This could be discussed more for the benefit of the reader.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper introduces Linear Spherical Sliced Optimal Transport (LSSOT), a framework for embedding spherical probability distributions into L2 space while maintaining spherical geometry. Extending previous work on circular distributions, LSSOT offers a computationally efficient metric for spherical data comparison, proving to be both a valid metric and faster than baseline approaches. The effectiveness of LSSOT is demonstrated through applications in cortical surface registration, 3D point cloud analysis, and shape embedding.

### Strengths
1.	According to time complexity and runtime analysis and the results in Table 1 in this paper, the proposed method achieves significant computational efficiency while preserving the spherical geometry, making it suitable for large-scale applications.

2.	The authors prove that the proposed LSSOT satisfies metric properties, strengthening the approach's theoretical validity.

3.	This work performed extensive experimental validation. The effectiveness and efficiency of LSSOT are evaluated across multiple applications, including cortical surface registration, 3D point cloud analysis, and shape embedding.

4.	The cortical surface registration results indicate that LSSOT achieves high Dice scores and minimizes area/edge distortions in the deformation field. This may provide biologically meaningful insights into neuroimaging applications.

### Weaknesses
1.	From a technical point of view, the extension of the proposed method to the existing LOCT is limited. Still, it should be noted that it is useful in spherical signal analysis applications.

2.	LSSOT is compared to several baseline metrics, but the paper lacks a more detailed statistical analysis to quantify performance improvements (e.g., significance testing on Dice scores and area distortions). A statistical evaluation would help substantiate LSSOT’s claimed advantages over other metrics.

3.	Although LSSOT demonstrates effective interpolation performance, there is little information on its computational complexity compared to SSW and Spherical OT in point cloud analysis. A complexity analysis of interpolation tasks would better illustrate LSSOT's efficiency benefits.

4.	Hyperparameter settings, such as λ, the number of slices L, and reference size M in LSSOT, are briefly mentioned but not systematically analyzed. A discussion on the sensitivity of these parameters and their impact on performance would be beneficial, particularly for users looking to optimize LSSOT for specific applications.

### Questions
The suggestions are mentioned in Weaknesses(2,3,4).

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
3

### Summary
The authors propose the linear spherical sliced optimal transport framework (LSSOT) as an efficient method for comparing probability distributions on spherical manifolds. The authors build on several valuable past works that aim to linearize or speed-up the optimal transport problem by restricting to specific domains. In their case, the spherical domain has extensive applications in computer vision and especially in neuroscience for problems of cortical surface registration. They demonstrate results on several applications including point cloud interpolation and cortical surface registration.

### Strengths
The paper has several strengths including the clarity of the writing, problem formulation, and valuable contribution to problem of spherical data analysis. I appreciated the authors placing their work in the context of past work in sliced optimal transport and linear optimal transport. They elegantly combine aspects from both to develop the spherical OT framework. There is theoretical justification for the use of the LSSOT as a distance metric, and an efficient implementation for speed. The authors also demonstrate the utility of their method on several experiments, notably cortical surface registration, where working in a spherical domain is essential.

### Weaknesses
I have several questions about the experimental evaluation of this work, specifically as it pertains to cortical surface registration. I appreciate the use of multiple datasets and outlining the problem of surface-to-atlas registration. The authors use extensive evaluation metrics. However, I have some concerns with the experimental setup and results.

The main purpose of these experiments was to quantify the improvement in performance and speed in using their metric over conventional ones. The authors state that they also use a Dice loss and regularization term. Did the authors perform new hyperparameter tuning on these losses when switching the image similarity metric? Failure to do so may bias results.

Why was only the sulcal depth used as a registration feature? Curvature is typically used and also offers a lot of information for registration.

The results on registration are puzzling. The authors' method performs considerably worse on SWD than the MSE loss for the NKI (Left hemisphere) and in MAE for NKI (Right hemisphere). Can the authors comment why? 

In general, registration is an ill-posed problem, and one can modify the hyperparameters appropriately to either obtain very close matchings or well behaved fields. Can the authors discuss this and describe how this tradeoff was achieved. 

How do these results compare to classical freesurfer registration? Which although is slow, represents a SOA in performance.

### Questions
see weaknesses section above.

### Soundness
3

### Presentation
3

### Contribution
3
