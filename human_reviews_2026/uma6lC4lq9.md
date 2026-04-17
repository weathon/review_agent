# The Shape of Attraction in UMAP: Exploring the Embedding Forces in Dimensionality Reduction

- Decision: Reject
- Scores: 2, 6, 8, 2

## Abstract
Uniform manifold approximation and projection (UMAP) is among the most popular neighbor embedding methods. The method samples pairs of point indices according to similarities in the high-dimensional space, and applies attractive and repulsive forces to their coordinates in the low-dimensional embedding. In this paper, we analyze the forces to reveal their effects on cluster formations and visualization, and compare UMAP to its contemporaries. Repulsion emphasizes differences, controlling cluster boundaries and inter-cluster distance. Attraction is more subtle, as attractive tension between points can manifest simultaneously as attraction and repulsion in the lower-dimensional mapping. This explains the need for learning rate annealing and motivates the different treatments between attractive and repulsive terms. Moreover, by modifying attraction, we improve the consistency of cluster formation under random initialization. Overall, our analysis makes UMAP and similar embedding methods more interpretable, more robust, and more accurate.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
UMAP is a widely-adopted neighbor embedding method, and understanding how its attractive and repulsive forces define the final embedding is core to improving reliability and usability in downstream tasks. This paper conducts numerical analysis on how these forces evolve under different circumstances and discusses the importance of learning-rate annealing and the asymmetry between attraction and repulsion. The authors provide code for both experiments and visualizations, supporting transparency and reproducibility.

### Strengths
- The writing is fluent, improving the readability of this paper..

- The code release is thorough, including experimental code and plotting scripts. This openness strengthens the paper’s scientific value and facilitates verification.

- The author has provided an extensive list of related works, covering most of the important progresses in this field.

### Weaknesses
- **Uncertain novelty relative to prior research.** 
  - In section 4, the authors assert *a gap* in the literature regarding force dynamics, yet cited works (such as McInnes et al., 2018; Agrawal et al., 2021; Wang et al., 2021; Draganov & Dohn, 2023), as identified by the authors, already examine the evolution of attractive and repulsive updates, hence a distinction must be made to justify the novelty. The authors described their difference as *"Here, we treat the decompositions as independent functions that can take various forms"*, which is quite confusing to me and does not make a good point. In my opinion, the main difference here seems to be the integration of the learning rate into the discussion, so that we can discuss the actual update at each step. The significance of this point is unclear (see below), and it definitely needs more articulation.

  - Section 5.2: It should not be surprising to anyone in the dimensionality reduction field that the cluster formation is dominated by attraction and the repulsion makes the clusters more compact. To list a few examples just from the authors' own citations, similar ideas has been described in Narayan et. al 2020, Bohm et. al 2022, Wang et. al 2025. Besides, description regarding the design of LocalMAP seems to already discussed in Wang et. al 2025 Section 5.

  - Section 6: The authors built a connection between Neg-t-SNE and UMAP, which is a novel point, but the message is also unclear (see below). The authors' other acclaimed contribution, that the attractive and repulsive terms between different DR algorithms are interchangable, has been brought up before, see e.g. Wang et. al, 2021. As long as these terms are following the DR algorithm principles, they can form a good DR algorithm.

- **Lack of a clear overarching message.** While the experimental analysis is detailed, the narrative does not converge on a key takeaway.
  - Section 4 - 5.1: The authors observed that the existence of large updates in UMAP may flip the relative position of certain neighbors. While this observation can be interesting, the reader may wonder how this observation could translate into some real improvements over the dimensionality reduction quality. Comparing Fig 2 (b) and (d), I personally don't feel the authors' proposed change make any difference in the embedding quality, and this observation sort of invalidates the importance of Fig 2 (f-h) comparison. Similar comment could also be made to Fig 1, that for Neg-t-SNE and PaCMAP the default parameters seems to be achieving a good balance between the existing metrics. What **actionable** insight should an algorithm designer or user walk away with? For section 5.1, several conclusions restate design choices already justified in the original UMAP paper—for example, spectral-embedding-based initialization and learning-rate annealing—making the supposed improvements, if any, feel incremental rather than revelatory.

  - Section 6: While the authors built the connection between Neg-t-SNE and UMAP, it's unclear what is the major takeaway one could go from here. Shall we favor methods such as Neg-t-SNE and PaCMAP over the unstable UMAP? Is there statistical evidence to support such a claim? Many questions were left unanswered.


- **Scope feels narrow and presentation should be significantly reorganized**. The analysis focuses in the main text heavily on UMAP and the MNIST dataset. Framing within the broader ecosystem of neighbor-embedding methods could substantially elevate impact to the state required at top venues such as ICLR. It should be noted that, several analyses to other existing works and dataset reside in the appendix, but the main text does not have enough descriptions to these paragraphs. Ultimately, they are not only unlikely to shape reader perception, but also left in an unorganized state as they do not contribute to a coherent message. Integrating those results into the main narrative would considerably improve the overarching message and strengthen the paper’s contribution.

### Questions
My questions are mostly related to my confusion after reading this paper, which I have elaborated in the weaknesses section. Specifically:
- What would you like the reader to walk away after reading this paper? Compared to the default setting of UMAP, shall we increase $a$/$b$ or decrease it, to obtain a more faithful and interpretable result? Shall we increase the far-sightedness in the force gradient? Shall we favor other methods such as Neg-t-SNE/PaCMAP/LocalMAP? Does such choice improve the performance of the algorithm statistically?
- How would you like to separate your work against the existing ones in terms of novelty? Why is the proposed analysis more insightful, if the decomposition method discussed has already been adopted by many?

### Soundness
3

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a theoretical formulation of UMAP’s learning process via a pair of mathematical properties: attraction shape and repulsion shape. With these properties, the authors manage to interpret the general mechanism of how UMAP’s loss function works in two forces under different conditions. The authors further show the practical implications of attraction and repulsion shapes by showing how they can interpret and improve the consistency and cluster formation, and how they explain the connection between Neg-t-SNE and parametric UMAP.

### Strengths
- Provide a careful theoretical analysis with derivations of several key properties.
- Offer insights that help explain behaviors such as initialization effects and cluster formation.
- Indicate practical implications for DR design, e.g., encouraging consistency.

### Weaknesses
- Ad hoc shape tuning. In the experiments of consistency improvements (Sec. 5.1) and cluster formation (Sec 5.2), the attraction/repulsion curve is adjusted ad hoc. While this lowers Procrustes distance, potential trade-offs with broader DR quality (e.g., Trustworthiness, Continuity) are not evaluated.
- Insufficient parameter analysis. The attractive/repulsive forms hinge on parameters a and b, yet the paper shows only scattered results for fixed choices. A systematic sensitivity study—visualizing how a and b affect the shapes and offering selection guidance—would clarify their roles.
- Limited discussion of alternative consistency methods. Related approaches, such as aligned-UMAP, are not adequately discussed. A comparison and a clear statement of the advantages and costs of the shape-guided method would strengthen the case.
- Poor readability and organization. The writing is dense, with many results and observations. Reorganizing the analysis with clearer signposting and bolded subheadings that highlight key takeaways would improve accessibility.

### Questions
(1) The core contribution of this paper is the detailed analysis of UMAP’s gradient update based on the attraction & repulsion shape coefficients. This can be viewed as a stepwise analysis of the local updates. But I wonder whether these shapes can shed light on UMAP's overall convergence property, especially across different settings of a and b? Do the oscillations the authors observe in some conditions cause unstable optimization?
(2) Beyond the cluster formation, the interplay between the attraction and repulsion forces also impacts the preservation of global and local structure. Can the attraction/repulsion shapes help explain some phenomena about this?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper provides an in-depth analysis of the attractive and repulsive forces present in the optimization step of UMAP. The authors first process and simplify the update functions of the point coordinates in the projection space, introducing two functions, $f_a$ and $f_r$ which control contraction and repulsion. Using this tool, they go on and generate the following insights:

- The first function, $f_a$, which is often associated with attraction, can actually act repulsively by flipping points and moving them further apart. $f_a$ and $f_r$ are also plotted for several methods and different learning rates. Lowering the learning rate lowers the distance below which $f_a$ becomes repulsive. Methods where $f_a$ is only attractive by design seem to be more robust to changes of the learning rate.
- The consistency of the projection under random initialization is studied using the Procrustes distance as a measure of projection similarity. It is show that by increasing the attractive force, the projections become more consistent presumably because the force between distant points is large.
- The formation and compactness of clusters is studied. Evidence is provided that the former is caused by attraction while the latter by repulsion.
- Different methods are compared in light of the approach of using $f_a$ and $f_r$. It is shown that Neg-t-SNE are the same in terms of their loss terms.

### Strengths
- Very clean exposition and nice focus on the single topic of the role of attraction/repulsion in dimensionality reduction algorithms. 
- Though a lot the ingredients of the paper already appeared in previous work (e.g. empirical attraction/repulsion function graphs), the way they are used and put together creates a novel view of the topic.
- The insights can help further optimize existing methods and provide a solid understanding on how new methods can setup their projection optimization.
- Follows the nice practice of including plenty of projections and plots for different datasets in the supplementary. This increases credibility and also makes the paper a nice reference.

### Weaknesses
- Only three datasets where used to support the different arguments, two of them quite similar in nature, MNIST and FMNIST. Though it is enough for visualization and demonstration purposes, the evidence feels a bit more qualitative than quantitative. Usually papers in the domain use almost 10 datasets covering different domains. To give an example of how this could have strengthened the evidence, in the study of cluster formation one could have correlated the attraction shape with the Silhouette score among datasets to support the connection of repulsion with cluster formation. One could also consider such ablations on synthetic datasets.

### Questions
- Around line 257 I think there is a typo, you use $\zeta^{-1}$ instead of $\zeta_{-1}$

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper discusses various aspects of the attractive and repulsive forces in UMAP and related visualization methods. It uncovers that for nearby points attraction might even increase their distance, explaining why learning rate annealing is crucial for UMAP. The authors also propose adding a long-range attractive force during the first optimization epochs to achieve a more consistent global layout, even from random initializations. Finally, they explore using different similarity kernels for attraction and repulsion in UMAP and compare UMAP's forces with other visualizations methods.

### Strengths
- S1: The finding about the expansive behavior of the attraction at small scales is interesting, novel and gives relevant pointers for the choice of the learning rate in UMAP. 
- S2: The long-range attraction is an interesting contribution.  
- S3: The extensive appendix extends some of the analysis to many other neighbor embeddings and is a useful resource.  
- S4: Code is submitted, facilitating reproducibility.

### Weaknesses
**Major**  

The paper presents many aspects about forces in neighbor embedding methods. While some of them are interesting, many feel underexplored and some are not novel. My suggestion is to expand the sections on expansive and long-range attraction in favor of the other sections. While this would improve the paper, I do not think that the novelty of the paper is sufficient for ICLR.

*W1: Presentation and exploration of UMAP's expansive attraction*:  
While the expansive nature of UMAP's attraction is interesting, the phenomenon should be better explained and explored. The following would help the explanation in my mind: The expansion is a classical case of an overshooting gradient descent. The update size decreases more slowly than the distance, so that the update eventually overshoots. This happen only for $b<1$.
Figure 1 a takes up a lot of space, but conveys very little information, I would omit it, or at least omit the repulsive part and decrease the size for the attractive part. 
Including some figures of UMAP with non-annealed learning rate (maybe even all four corner cases $\lambda_a, \lambda_r \in \\{0.1, 1\\}$) in the main paper and explaining them in more detail would help. Why can we deduce from this experiment that *only* the attraction necessitates annealing (the setting $\lambda_a = 0.1, \lambda_r=1.0$ still produces a "fuzzy" embedding). Based on these four plots, one could also explain the "failure modes" in Fig 1 h (top left and bottom right corner) qualitatively. It may also be useful to state that the distance $\zeta_{-1}$ around which the attraction would oscillate without annealing is large relative to typical UMAP plot diameters. Finally, it would help to state that by default, UMAP starts with $\lambda = 1$ and anneals this learning rate linearly to $0$ over training. Perhaps it might even be nice to show intermediate training stages, when clusters and their relative positions have already formed, but still appear fuzzy as $\lambda$ is still too large. 

Moreover, the phenomenon should be explored more deeply: One could, for instance, let a single pair of points evolve according to UMAP's attraction from various starting distances and plot how its distance evolves (without learning rate annealing). I assume it would eventually oscillate around 1. Similarly, in a UMAP plot without annealed learning rate one could measure the average distance between the embeddings of k-nearest neighbors and check if it is close to 1. Or compare this mean neighbor distance between the annealed and non-annealed case.

*W2: Discussion and exploration of long-range attraction*  
The proposed fix of adding $||y_i - y_j||^3$ to the attractive loss is quite similar to SNE, which uses attraction $|| y_i - y_j||^2$. It would be good to discuss this relation and perhaps even compare quantitatively. A small point is that in line 299 it is not $f_a^U$ going to zero that is problematic, but what is given in brackets ($|f_a^U| \cdot \zeta$ going to zero). But this is not equivalent, so "i.e." is not the correct choice here. 

In general, this chapter should be made more quantitative. While the authors compute Silhouette scores, Trustworthiness, and Procrustes distances, their comparison would be much eased by putting mean values in a Table. Moreover, the relative positioning of clusters might be also captured well by metrics like Spearman correlation between distances (both relative to the high-dimensional and among different UMAP embeddings). A key questions is, of course, how the long-range attraction compares to informative initializations with PCA or LE. So, a results table should also contain metrics for PCA/LE-initialized embeddings. Are they not simply much more stable than UMAP with long-range attraction starting from random noise? Finally, the authors include experiments on other datasets in the appendix, but a tabular summary of these experiments in the main paper would also strengthen the empirical findings. 

*W3: Cluster formation chapter is vague and less novel*  
Overall, the novelty of this chapter seems limited since, e.g., Agrawal et al. (2021) also mix and match different attractive and repulsive forces and Kobak et al. (2019) (which the paper cites in this section) already discuss heavy-tailed neighbor embeddings, including the effect of $b$ on UMAP embeddings. While the present paper refines their analysis to heavy-tailed attraction being more important, claims such as "to resolve the mystery of cluster formation" are way too strong given the limited novelty. In addition, the paper does not investigate if the additional clusters found with heavy-tailed attraction are meaningful or even undesirable fragmentation. Finally, in line 371 the paper states that changing $a$ affects both repulsion and attraction. But a different $a$ really just rescales the final embedding without changing its appearance, since

$$1 / (1+ad\_{ij}^{2b}) = 1 / (1+(a^{1/(2b)}d\_{ij})^{2b}) = 1 /(1+\tilde{d}\_{ij}^{2b})$$

for the rescaled distance $\tilde{d}_{ij} = a^{1/(2b)}d\_{ij}$.

*W4: Discussion of Neg-t-SNE*  
Relating Neg-t-SNE to UMAP in the context of expansive attraction is good and interesting. I particularly appreciate broadening its explanation for why learning rate annealing is required for UMAP to the attractive forces. However, I would strongly suggest to do so in the chapter on the expansive behavior of UMAP's attraction rather than two chapters later. The other statement in the Neg-t-SNE chapter, Prop. 6.2 on Neg-t-SNE's relation to parametric UMAP, is not novel, but can already be found in appendix E of Damrich et al. (2023).


**Minor**  
*W5: Notation* The notation does not get introduced in many places. E.g. at the start of sec 3: $f, h, d, d_{ij}$ are not defined and in section 5 I do not understand why the notation changes from $f_a$ and $f_r$ to $f_a^U$ and $f_r^U$.

*W6: Related work* I would recommend not to include the spectral methods Laplacian Eigenmaps and Locally Linear Embedding in the middle of the paragraph on neighbor embeddings in the related work. 

*W7: Flowery language* The language is in some place a bit to flamboyant for a research paper. E.g. the first sentence uses both "era" and "deluge", and the discussion section "demystifying".

*W8: Font size in many figures too small*



**Typos**

- line 235: approaches $-\infty$ not $\infty$

- line 484: Wrong citation Damrich & Hamprecht (2021) --> Damrich et al. (2023)

- line 1059 $(_a)$? 

- Eq 86 / 87 Switch in notation between $w_{i \mid j}$ and $p_{i \mid j}$

- Abstract: forces among high-dim data point -->  low-dim data points

- Equality sign missing after $\zeta_{-1}$ in axis-label of figure 1d.

### Questions
- Q1: line 246: "only" UMAP and t-SNE deal with some issue. Only methods among which group?

- Q2: line 252: "only UMAP deals with large [repulsive] values for small distances" Is this a good thing, because non-neighbors get pushed far apart, or a bad thing because it leads to numerical instability and requires gradient clipping?

- Q3: line 294: What is "scale-invariant structure"? Was perhaps "initialization-invariant" structure meant?

- Q4: line 1060: Why does initializing $\lambda$ at $0.5$ satisfies Prop 4.1? It is still the case that $\zeta_{-1} \approx 0.5 > 0$ according to Fig 1d.

### Soundness
2

### Presentation
1

### Contribution
2
