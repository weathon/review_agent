# Deep Synchronisation-based Clustering

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Identifying patterns in high-dimensional and complex data, such as images, requires techniques that extract meaningful features. Deep clustering combines the representation power of neural networks with classical clustering and has shown strong performance on such data. However, most approaches build on k-Means, inheriting its assumptions about cluster shapes, requiring the number of clusters in advance, and lacking an intuitive stopping criterion. We propose DeepSynC, the first synchronisation-based deep clustering algorithm that overcomes these limitations. It begins by identifying core points in the embedded space and assigning them to clusters. A novel cluster loss then synchronises similarly embedded objects, enabling the gradual assignment of further points. This combination of synchronisation-based loss and assignment strategy allows greater flexibility in cluster shape and introduces an automatic stopping condition for training.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes DeepSync, a synchronization-based deep clustering algorithm. The method begins by identifying core points in the embedded space and progressively assigns the remaining data points to clusters. The synchronization-based loss and assignment strategy enable greater flexibility in capturing non-convex cluster shapes and provide an automatic stopping condition for training.

### Strengths
The idea of performing synchronization-based deep clustering is interesting and distinguishes this work from the more prevalent centroid-based deep clustering approaches. The connection with the Kuramoto model, where each sample is treated as a phase oscillator, is also conceptually appealing. Moreover, the proposed method demonstrates promising performance on certain tasks.

### Weaknesses
1. Some parts of the paper lack clarity. For instance, the definition of the neighborhood U in the method section is not clearly provided. The experimental details in Section 4.1 are also insufficient. It is unclear which baseline methods are used for comparison, while some are mentioned in the related work section, this should be explicitly stated in the experimental section. Additionally, the statement “ARI values are computed on labeled points only” needs further explanation.

2. Some results appear suspicious. For example, in Table 2, the AE+MS method yields values of 00+00 across several datasets. This requires careful clarification. Furthermore, while the tables report standard errors, the paper does not describe how many runs were performed or how the standard errors were computed.

3. Although the algorithm removes the need to specify the number of clusters K, it introduces several additional hyperparameters such as T, K_neigh, T_conf, and T_stop. These add complexity to the model and may require extensive tuning.

4.  The method mentions using the SHiP clustering framework to determine K, but it is unclear how this is implemented. Is K determined before training or dynamically adjusted during training? The latter could be computationally inefficient. Moreover, relying on the elbow method, which usually depends on visual inspection of curves, for determining K during training may not be friendly to an end-to-end approach.

5. The method employs SynC for initialization. The paper should justify this choice and analyze the sensitivity of the model to initialization by comparing it with alternative strategies (e.g., random initialization).

6.  While the paper attempts to relate the synchronization loss to the Kuramoto model, this connection does not appear to be very strong based on the current exposition. The authors are also encouraged to discuss the relationship between their method and density-based clustering approaches.

### Questions
Please refer to the questions raised in the Weaknesses section.

### Soundness
2

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
5

### Summary
The paper presents an autoencoder-based deep clustering approach that exploits a loss function aiming to adapt the latent space 
so that points belonging to the same latent cluster get closer to each other. The method starts from a initially trained autoencoder and the number of clusters is determined in the initial latent space and is kept fixed during training. Only core points are initially clustered, while other points are gradually attracted towards the clusters. It is possible that at the end some points to be unassigned.

### Strengths
S1. A loss function is proposed aiming to gradually form compact clusters in the latent space.
S2. The idea of starting from core cluster points and gradually assign the other points to clusters is well-motivated.

### Weaknesses
W1. Application of the method involves several critical decisions that are addressed by setting user defined thresholds. The number of hyperparameters is quite large.
W2. A major issue concerns the specification of the number of clusters is handled in the initialization phase. During training the number of clusters is kept fixed.   
W3. The fact that method ends with unassigned points causes difficulty in empirical comparison since competitors assign the ambiguous points to clusters.
W4. There are issues to be clarified and concerns about the empirical comparison that are presented in the questions section.

### Questions
Q1. The rationale behind the proposed loss function can be sufficiently supported without resorting to the described physical metaphor (that could possibly be omitted). For example it is not clear how eq. 2 can be obtained from eq. 1.
Q2. It is not clear whether the proposed mechanism achieves not only cluster compactness but also can force the clusters to move away from each other (cluster separation).
Q3. The deep learning method does not modify the number of clusters which is set fixed at initialization. Based on this important observation, I strongly suggest that the paper presentation and evaluation should focus on the deep learning part of the method considering that number of clusters is given in advanced (for example is set to the ground truth number).
Q4. In the present form of the method, a critical decision is related to the algorithm used to estimate the number of clusters in the initial latent space. Why not using alternative methods (for example from ClustPy library)? Does the employed method (SHiP) contain hyperparameters?  
Q5. I cannot find how the number of clusters is set in the competitors DEC/IDEC, DCN etc that require the number of clusters to be specified in advance.
Q6. A pseudocode describing the method and the hyperparameters involved should be included.
Q7. The existence of unassigned points makes comparative evaluation difficult. Either all points should assigned to clusters (not to singleton ones) or soft version of the competitors should be considered so that ambiguous points will not be clustered.
Q9. I don't think that the idea of DeepSync+ sufficiently addresses the concern of the previous question.
Q10. The method includes too many hyperparameters (and, additionally, the method used for the initial clustering should be selected). It is well-known that in density-based methods such hyperparameters (eg. number of neighbors) critically affect the obtained solution.

### Soundness
3

### Presentation
2

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
This paper proposes a deep learning framework for multi-view synchronization based on permutation-equivariant embeddings. It consists of a pairwise synchronizer for estimating relative transformations and a global synchronizer to recover consistent global alignment. The method supports various transformation groups and is evaluated on both synthetic and real datasets, showing improved accuracy and robustness over classical and neural baselines.

### Strengths
1. The use of permutation-equivariant networks is appropriate for multi-view settings and aligns well with the unordered nature of input views.
2. The method supports various transformation groups and demonstrates applicability to both synthetic and real-world datasets.
3. Experiments are well-structured, with consistent gains over baselines and meaningful ablation studies that support design choices.

### Weaknesses
1. The evaluation focuses on small-scale, flattened datasets (e.g., MNIST, FMNIST, HTRU), using a simple autoencoder without convolution or attention backbones. This limits clarity on how well the method generalizes to high-resolution, large-scale data such as CIFAR-10, ImageNet Dogs.
2. While the method estimates the number of clusters automatically, the estimation is not the core contribution and can deviate from ground truth (e.g., overestimation on MNIST/FMNIST). It would be beneficial to compare against other cluster-number-agnostic approaches such as Robust Continuous Clustering or community-based methods like Louvain clustering.
3. The synchronization loss computes all pairwise affinities within each mini-batch, resulting in $\mathcal{O}(|B|^2)$ time and memory cost. While batch sizes are moderate in the current setup, the paper does not analyze computational efficiency at larger scales, which may hinder scalability in practice.

### Questions
Please refer to weaknesses.

### Soundness
2

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
The article introduces DeepSynC, the first deep clustering algorithm based on synchronization principles derived from the Kuramoto model. The method addresses key limitations of k-means-based deep clustering approaches by: (1) not requiring the number of clusters k in advance, (2) allowing for non-spherical cluster shapes, (3) introducing a gradual assignment strategy that first labels high-confidence core points, and (4) providing an automatic stopping criterion. The synchronization loss encourages nearby points with similar labels to align their embedded positions while preventing alignment between points with different labels.

### Strengths
- The paper is clear about its motivation with sufficient significance and quality.
- The adaptation of the Kuramoto synchronization model to deep clustering is innovative. The connection between phase oscillator synchronization and clustering provides an elegant theoretical framework that hasn't been explored in deep learning before.
- The gradual assignment approach starting from local core points is well-motivated. 
- The ability to automatically determine when to stop training based on label stability (Tconf) or assignment plateaus (Tstop) is a good practical advantage.
- The ability to leave uncertain points unassigned rather than forcing misclassification is valuable for real-world applications.

### Weaknesses
- While the Kuramoto model provides intuition, the paper lacks rigorous analysis of when and why the synchronization loss works. What are the convergence guarantees? Under what conditions might the method fail?
- The O(n²) complexity of the synchronization loss (Equation 3) summing over all pairs in a batch is concerning. How does this scale to large datasets? The kNN-based assignment strategy adds further computational overhead.
- Missing comparisons with recent deep clustering methods beyond k-means variants like simclr based SCAN, NNM or so.
- No analysis of sensitivity to batch size, which seems critical given the pairwise loss.
- While Figure 4 shows robustness across parameter ranges, the interaction between parameters isn't examined. How do T% and kneigh jointly affect core point selection?
- The synchronization loss formulation (Equation 3) needs clearer motivation for the specific weight function wx(y).
- The relationship between the original Kuramoto model (Equation 1) and the clustering loss could be more explicit.

### Questions
- Can you provide any convergence analysis or theoretical guarantees for the synchronization process? What cluster assumptions does this method make?
- What is the total time complexity including the kNN calculations and majority voting? Have you considered approximations for large-scale datasets?
- How sensitive is performance to batch size given the pairwise loss? What happens with very small or very large batches?
- How did you arrive at the specific exponential weight function in wx(y)? Have you experimented with other similarity kernels?
- Given the core point concept, how does this relate to or improve upon DBSCAN-style approaches in the embedded space?
- Can you also compute NMI and ACC besides ARI?
- Does knowing the number of clusters beforehand benefits the method? 
- Does the latent dim (m) is always set to 10 even if k is larger? Is there any relation between m and k? Did you try larger m? how sensitive is m in those cases?

### Soundness
3

### Presentation
3

### Contribution
2
