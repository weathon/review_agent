# $\ell_1$ Latent Distance based Continuous-time Graph Representation

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 8, 6

## Abstract
Continuous-time graph representation (CTGR) is a widely-used methodology in machine learning, physics, bioinformatics, and social networks. The sequential survival process in a latent space with the squared $\ell_2$ distance is an important ultra-low-dimensional embedding for CTGR. However, the squared $\ell_2$ distance violates the triangle inequality, which may cause distortion of the relative node positions in the latent space and thus deteriorates in social, contact, and collaboration networks. Reverting to the $\ell_2$ distance is infeasible because the corresponding integral computation is intractable. To solve these problems, we propose a theoretically-sound $\ell_1$ latent distance based continuous-time graph representation ($\ell_1$LD-CTGR). It facilitates a true latent metric space for the sequential survival process. Moreover, the integral of the hazard function is found to be a closed-form piece-wise exponential integral, which well fits the ultra-low-dimensional embedding. To handle the non-differentiable $\ell_1$ norm, we successfully find a descent direction of the hazard function to replace the gradient, enabling mainstream learning architectures to learn the parameters. Extensive experiments using both synthetic and real-world data show the competitive performance of $\ell_1$LD-CTGR.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a latent position model for continuous-time using a hazard function based on $\ell_1$ distance between latent positions motivated by issues regarding the triangle inequality from previous methods using squared $\ell_2$ distance. The model is benchmarked against continuous-time graph representation models and other common graph representation techniques against commonly used synthetic and real dynamic networks with impressive results in incredibly low embedding dimension.

### Strengths
The technical work in this paper is strongly motivated and the argument for using $\ell_1$ distance for social networks, highlighting issues with using the squared $\ell^2$ distance used in GRASSP. While the computations are different in nature to GRASSP, the new approach has the same complexity cost as the previous approach, usurping the technique for incredibly low dimensional embeddings ($D = 2$).

### Weaknesses
The experimental results lack explanation. For instance, it is not explained how comparison continuous-time methods based on instantaneous edge models (CTDNE/HTNE) are applied to the persistent edge datasets. These methods are trying to model a different problem, so it not surprising they underperform. However, $\ell_1$LD-CTGR often outperforms GRASSP which is a like-minded comparison.

More information than what is given in Appendix A.5.1 about in-sample, out-of-sample and across-sample evaluations is needed. I understand the intended aims of the three approaches, but I don't see how it relates to the language of 'simple' and 'hard' sets. More details are needed to make this a repeatable experiment.

### Questions
Some datasets have worse performance when you move from in-sample to out-of-sample to across-sample assessment, for example, synthetic-$\alpha$. In general, I would expect the in-sample to be the easiest assessment as it is looking within training data, but it occasionally underperforms compared to the other assessments. Can the authors provide some intuition as to why this phenomenon occurs?

Does using the $\ell_1$LD-CTGR technique with dimension $D > 2$ lead to any improvement in performance? If not, this would further motivate the argument for using the computational efficient $D = 2$ embedding.

### Soundness
4

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work expands upon the recent GRASSP framework for the modeling of continuous time dynamics using the sequential survival process and piecewise velocity modeling procedure developed in (Celikkanat et al., 2024), Specifically, the paper here makes the following contribution:

1)	Advance the modeling to a true metric embedding model based on the l1 distance as opposed to the squared Euclidean distance used due to computational convenience in (Celikkanat et al., 2024).

2)	Demonstrate that this procedure despite added complexity of inference in practice have same scaling properties with a subgradient procedure proposed well optimizing the L1 distance while preserving other essential properties of GRASSP.

3)	Extensively evaluate the procedure and contrast the framework to more approaches than previously considered finding that the L1 distance based framework has enhanced performance when compared to the previous squared L2 distance embedding framework of GRASSP – pointing towards metric properties enhancing performance.

### Strengths
•	The paper is very well written and clear in its presentation.

•	The contribution is sound and valid and includes a detailed experimental comparison to existing methodologies.

•	The inference procedure is non-trivial and ensures metric properties in latent distance based continuous time link persistent networks.

Originality: The paper expands upon the GRASSP framework to make the embeddings truly metric by use of the L1 norm as opposed to relying on the squared Euclidean distance. The experimentation closely follows the GRASSP framework but is extensive. The contribution is sufficiently original. 

Quality: The developed methodology is non-trivial and the paper well written and the experimentation extensive. It would however improve the paper to compare the L2-norm metric version with the proposed L1-norm metric version. It is unclear what comes from change of metric and what is the result of non-metric squaring the L2-norm. The metric L2-norm would be less computational attractive but could be implemented by use of numeric integration instead of relying on analytical expressions of the integral representations.

Clarity: The paper is clear and well presented.

Significance: The approach appears useful and superior to using the squared L2 distance with results well supporting the utility of the proposed approach.

### Weaknesses
•	The novelty of the paper is somewhat limited but valid expanding the prior work on GRASSP to encompass true metric embeddings as opposed to relying on computational convenient squared L2 distances.

•	It is unclear how difficult in practice using the standard Euclidean distance would be. I.e., the integral rather than being analytically solved could here be solved by numeric integration which would likely not be very heavy in computation and potentially work well in practice. This would strengthen the paper to consider.

### Questions
Experimentally the superiority of the L1 distance metric over squared Euclidean distance is shown. The method compared against have been run in their default setting. It is unclear if the proposed method have been more extensively tuned in hyperparameters in terms of learning rates etc. Only used dimensionality is discussed. Please clarify how learning rate etc. were tuned for the methods.

Could the L2-metric approach using a truly Euclidean metric as opposed to squared Euclidean distance representation be feasibly implemented using numeric integration as opposed to analytical expressions using the squared distance? – and how would such numeric integration based metric approach compare to the presented L1 approach?
This would better position the paper as it is unclear in the current experimentation what the influence of having a metric vs. non-metric embedding implies (this would be comparing squared L2 to conventional L2-norm (metric) - which could have been addressed by accurate numeric integration potentially on smaller toy problems) and what is the effect of L2-norm (metric) vs. L1-norm (metric) presently invoked. These different norms will by themselves substantially influence the results and here the L1-norm may also have benefits over the L2-norm which cannot currently be assessed. This would strengthen the paper to address and help further position the contribution and understanding the merits of the L1-norm.

Given the above I consider this a borderline paper but would be willing to increase my score.

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
3

### Summary
This paper addresses a fundamental theoretical flaw in existing Continuous-Time Graph Representation models, such as GRASSP, which rely on the squared $l_2$ distance in the latent space. The authors correctly point out that $\|\cdot\|_2^2$ violates the triangle inequality, leading to distorted metric properties. They propose $l_1\text{LD-CTGR}$, which replaces the squared $l_2$ distance with the theoretically sound $l_1$ distance ($\|\cdot\|_1$). The core technical contribution is the derivation of the continuous-time integral of the $l_1$ hazard function as a closed-form piece-wise exponential integral and finding a descent direction to handle the non-differentiability of the $l_1$ norm. Experiments show competitive performance and identical computational complexity compared to the $l_2^2$-based baseline.

### Strengths
1.	The motivation is excellent: correcting the use of the squared $l_2$ distance ($\|\cdot\|_2^2$) which violates the triangle inequality (L059), ensuring the latent space is a valid metric space using the $l_1$ distance.

2.	The derivation of the closed-form piece-wise exponential integral for the $l_1$ hazard function (Theorem 1) is a non-trivial technical accomplishment, successfully tackling the intractability concern associated with non-$l_2^2$ distance metrics (L234-236).

3.	The use of descent direction (subgradient computation) to handle the non-differentiable $l_1$ norm (Theorem 3, L309-317) is a sound approach that ensures compatibility with mainstream optimization frameworks like PyTorch (L282, L333-336).

### Weaknesses
1.	While $l_1$ distance is theoretically sound, the paper does not convincingly demonstrate that resolving the metric space flaw translates into significant empirical gains. $l_1\text{LD-CTGR}$ is competitive but rarely dominates its flawed counterpart, GRASSP (e.g., in Table 2, GRASSP often achieves higher scores).

2.	The efficiency gains from the piece-wise exponential integral seem heavily reliant on the assumption of $D=2$ (L249, L342). The paper should explicitly discuss how the complexity scales with increasing dimensions $D$ (where $D$ determines the maximum number of zero points $z_{ij, d}$), and whether the method remains feasible for $D>6$ (L1197).

3.	Theorem 3 correctly identifies that the derived term $-\partial\lambda_{ij}(\mathbf{r}_i)$ is *not* a subgradient when $s=-1$ and $C \ne \emptyset$ (Case 4, L997-1005). Although the authors state it is still a descent direction, the potential impact of using a non-subgradient in the optimization step for the important "disconnected" state ($s=-1$) must be discussed.

4.	The paper argues reverting to $l_2$ distance is intractable due to integral complexity (L065). Given the technical achievement of solving the $l_1$ integral, a more explicit comparison showing *why* the $l_2$ distance integral remains intractable (e.g., complexity of integration beyond $D=2$) would strengthen the necessity of adopting $l_1$.

### Questions
1.	For dimensions $D > 2$, how does the computational complexity of the tensor-parallelized integral (Theorem 2, Eq. 15) scale with $D$? Specifically, what is the complexity when $D=10$ or $D=100$, and is $l_1\text{LD-CTGR}$ still competitive with GRASSP in these higher dimensions?

2.	The core motivation is correcting the metric distortion. Can the authors provide a metric (e.g., a latent space uniformity measure, or the preservation of long-range topological fidelity) that *directly* quantifies the benefit of using a valid metric space ($l_1$ vs. $l_2^2$), thus supporting the strong theoretical claim empirically?

3.	In Case 4 (disconnected state $s=-1$ and $C \ne \emptyset$), the computed direction is not a subgradient. Given that maintaining the disconnected state is essential for modeling complex networks, what are the observed empirical consequences (e.g., convergence speed) of using a descent direction that is not guaranteed to be a subgradient?

4.	Since you successfully solved the non-trivial $l_1$ integral, please elaborate on the specific mathematical properties of the $l_2$ distance integral ($\int \exp(\dots + s\|r_i - r_j\|_2) dt$) that make it intractable, contrasting it with the solvable nature of the $l_2^2$ integral (Gaussian) and your $l_1$ integral (piece-wise exponential).

5.	Proposition 1 confirms that the bounded property of the distance holds for $l_1$ distance as well (L199-204). Beyond verification, how does this property actively guide or constrain the optimization process during training, and is the derived bound $\mathcal{B}(s)$ (Eq. 12) used anywhere in the final loss function or regularization?

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
The paper introduces ℓ1-LD-CTGR, a continuous-time graph representation that uses an ℓ1 latent distance to form a true metric space, overcoming distortions caused by squared ℓ2 distances and enabling tractable hazard function computation. Experiments show that this approach achieves competitive performance on both synthetic and real-world graph datasets.

### Strengths
(1) This paper presents a solid theoretical motivation and analysis for adopting the L1 distance over the traditional L2 distance, offering a complete and self-contained formulation.

(2) The experimental evaluation is comprehensive, covering multiple benchmarks with sufficient baselines and datasets to convincingly demonstrate the effectiveness of the proposed approach. The reported results clearly show that the proposed method outperforms prior baselines.

### Weaknesses
Although the paper is well-structured and self-contained, its overall contribution appears limited. The main novelty lies in replacing the L2 loss function with an L1 formulation and adjusting the corresponding learning algorithm and optimization strategy. Moreover, the work is confined to temporal network learning, which limits its broader impact on the wider research community.

### Questions
No more questions.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies continuous-time graph representation via sequential survival processes and asks how to embed evolving networks in an ultra–low-dimensional latent space without distorting geometry. The main difficulty is that the widely used squared latent distance violates the triangle inequality, which can warp relative node positions. To overcome this, the paper proposes ℓ₁LD-CTGR, which replaces the latent distance with ℓ₁ while preserving tractable learning. Specifically, it (i) proves boundedness of the average ℓ₁ distance across event intervals, (ii) derives a closed-form, piece-wise exponential integral for the hazard, (iii) designs an efficient tensor-parallel algorithm tailored to D=2 embeddings, and (iv) introduces a descent-direction/subgradient scheme that makes the non-differentiable ℓ₁ objective trainable in mainstream autodiff frameworks with the same order of complexity as GRASSP. Theoretical results establish a true latent metric space and closed-form integrals compatible with ultra–low-dimensional modeling. Empirically, across 11 synthetic and real-world datasets and three evaluation regimes (reconstruction, completion, prediction), ℓ₁LD-CTGR attains competitive or superior performance to eight strong baselines, with robust gains on large, challenging networks.

### Strengths
1.The paper clearly shows that the squared ℓ2 distance violates the triangle inequality and can distort latent-space geometry. It also argues that the resulting integrals are less tractable, giving the work clear and well-founded motivation.

2.The paper presents a generally sound theoretical development supported by a careful empirical evaluation.

### Weaknesses
The paper claims “competitive performance” across multiple synthetic and real datasets. However, in several tables, the proposed method either trails strong baselines or offers only marginal gains. The manuscript lacks a systematic analysis (e.g., error breakdowns, dataset-specific failure modes) to reconcile these discrepancies, which weakens the empirical claim.

### Questions
1.In Table 1, DyGFormer outperforms the proposed method on some datasets, and the manuscript does not provide an explanation or discussion of this phenomenon.

2.In Appendix Table 5.4, the proposed method appears slower than existing baselines on several datasets. Does this suggest the claimed efficiency gains only emerge with longer training time, or that further optimization is still needed?

3.In Table A4, the proposed method underperforms several strong baselines on multiple datasets. Please explain the underlying causes and provide analysis to clarify this discrepancy.

### Soundness
3

### Presentation
3

### Contribution
3
