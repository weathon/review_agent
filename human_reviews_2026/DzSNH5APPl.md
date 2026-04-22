# Learning Explicit Single-Cell Dynamics Using ODE Representations

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 6, 6, 2, 2

## Abstract
Modeling the dynamics of cellular differentiation is fundamental to advancing the understanding and treatment of diseases associated with this process, such as cancer. With the rapid growth of single-cell datasets, this has also become a particularly promising and active domain for machine learning.
Current state-of-the-art models, however, rely on computationally expensive optimal transport preprocessing and multi-stage training, while also not discovering explicit gene interactions. 
To address these challenges we propose Cell-Mechanistic Neural Networks (*Cell-MNN*), an encoder-decoder architecture whose latent representation is a *locally linearized ODE* governing the dynamics of cellular evolution from stem to tissue cells. Cell-MNN is fully end-to-end (besides a standard PCA pre-processing) and its ODE representation learns interpretable gene interactions.
Empirically, we show that Cell-MNN achieves competitive performance on single-cell benchmarks, surpasses state-of-the-art baselines in scaling to larger datasets and joint training across multiple datasets, while also learning interpretable gene interactions that we validate against the TRRUST database of gene interactions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose to use a variant of Neural ODEs where the velocity is parameterized as a constant linear function of location at every timestep.  They use this parameterization to fit single-cell RNA data over several timepoints, evaluating with interpolation of a heldout timepoint.

### Strengths
The authors show good results compared to other interpolation methods on several common benchmarks.  The motivation for the method is straightforward and natural, and I appreciate the work in making simpler and more interpretable models competitive with flow-based trajectory learning.

### Weaknesses
Projecting to 5d PCA is a considerable limitation, as the method seems to not scale and the dynamics of single cell data is generally not so easily compressible.  Additionally, the actual training of the model is somewhat unclear (see questions).

I’m also not totally convinced by the gene analysis results, as it probably requires a baseline to convince a reader that these top gene interactions were not simply the ones encoded by PCA, and any model that learns a velocity (say, MFM-OT or TrajectoryNet) could linearize its velocity at some timepoint and find the same gene interactions.  The later claim about enforcing a zero in the spectrum seems a bit flimsy; it’s true that many individual genes will not be changing during the dynamics, but at only 5 PCs one would expect all PCs were relevant to the dynamics.

### Questions
How exactly is the model trained?  The locally linear ODE maps from z_t to z_{t + \Delta_t} but I cannot find an indication of how the authors turn this into an explicit loss.  They make no mention of differentiating through an ODE solver so is there a parameter \Delta_t set in the method and the authors explicitly differentiate through several linearization steps?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors propose Cell-MNN, a new ODE-based method designed to model the single-cell dynamics of cellular differentiation. The method operates in PCA space, and uses a hypernetwork to learn the transition matrix A. The local dynamics are assumed to be linear. The model is trained by optimizing an MMD loss between the predicted and target marginals of gene expression in the latent space. They benchmark their model on single-cell interpolation, amortized training, and gene interaction directionality.

### Strengths
- The method eliminates the need for OT which is typically a computational bottleneck for many methods.
- The method achieves competitive performance on the cell interpolation task, outperforming multiple baselines on three different datasets.
- The authors perform a scaling experiment to benchmark the ability of their model to handle large datasets and show that it is scalable enough for typical contemporary large datasets.
- The ability of the model to learn explicit gene regulatory interactions is useful for interpretability and for learning the explicit gene networks governing differentiation.

### Weaknesses
- The authors perform an unsupervised classification task to predict the direction of regulation for a known TF-gene link from the TRRUST database. The paper does not benchmark the method's ability to predict the links themselves, e.g., by evaluating if the interactions with the highest predicted strength are enriched for known links. Furthermore, analysis of the top source or target genes using gene enrichment analysis could shed light into any relevant pathways.
- The authors mention GRN inference as a complementary line of work but only reference static GRN models. A few works have experimented with learning temporal gene networks for cell differentiation or biological state progression, such as Scenic+, Marlene, Dictys [1-3]. A discussion on how the current work differs from these or a direct comparison is needed to strengthen the contribution.
- The authors project data into 5 dimensions using PCA. This is too limiting for large single-cell datasets with ~20k genes and likely ignores finer interactions between genes. The authors need to present additional work/analysis that uncovers the type of information lost by doing such a projection and the model's ability to learn fine-grained interactions.

[1] https://www.nature.com/articles/s41592-023-01938-4  
[2] https://academic.oup.com/bioinformatics/article/41/Supplement_1/i628/8199402  
[3] https://www.nature.com/articles/s41592-023-01971-3

### Questions
Could the authors comment on the model's capability for extrapolation, that is, predicting a future time point beyond the final observed time in the training set?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes Cell-MNN, a ODE–based framework that learns explicit single-cell dynamics by modeling local linear ODEs in a PCA latent space. The approach avoids explicit OT preprocessing and claims to simultaneously provide interpretable gene regulatory interactions through the learned Jacobian matrices. Experiments on several benchmark single-cell datasets show competitive interpolation accuracy and advantages over OT-based baselines.

### Strengths
The paper is clearly written and tries to tackle an important problem in single-cell dynamic modeling — learning interpretable continuous-time dynamics.

### Weaknesses
1. Latent-space linearization is under-justified and fragile. The method hinges on representing dynamics as locally linear in a PCA latent space, $\dot z \approx A_\theta(z,t)z$, advanced via a matrix exponential. The paper itself acknowledges that evolving too far leaves the validity region and requires re-encoding to refresh the local ODE, but provides no analysis of linearization error growth, step-size control, or robustness to curvature in realistic trajectories. Moreover, the claimed efficiency advantage relies on a very low latent dimension; once d_z grows, computing and applying the operator (incl. eigendecomposition/inversion) becomes the dominant cost (see 4), which does not resolve the core limitations of Neural ODEs for stiff/strongly nonlinear flows— it only replaces numerical integration with piecewise-linear surrogates without guarantees.
2. GRN novelty is overstated; prior work is insufficiently acknowledged. Inferring gene regulatory relations from a learned ODE/vector field via its Jacobian has been proposed and used before (e.g., Dynamo [1]) and applied in multi-timepoint trajectory settings (e.g., TrajectoryNet [2], TIGON [3]). Cell-MNN can be viewed as the special case where the vector field is constrained to be piecewise linear in a low-dimensional PCA subspace. The “Gene Regulatory Network Discovery” related-work discussion omits or underplays these lines, which is misleading about the methodological novelty. 
3. Scalability claims (vs. OT) are overstated, and the mechanism is unclear. The paper attributes scalability improvements to replacing OT preprocessing with MMD-based distribution matching and reports out-of-memory (OOM) errors for standard OT on inflated datasets. However, many Neural ODE–based or flow-matching methods already alleviate memory issues by computing OT or matching losses in mini-batches, thereby avoiding O(n^2) coupling matrices in practice. From this perspective, it is unclear why Cell-MNN “solves” sample-size scalability more effectively than these existing approaches, given that both rely on batched stochastic training. Moreover, substituting OT with balanced MMD does not directly address unbalanced dynamics (cell proliferation/death), where Unbalanced-SB frameworks provide principled formulations. Thus, the claimed scalability advantage seems more an artifact of implementation choices than a fundamental methodological improvement.
4. Cubic scaling in the latent dimension threatens practicality beyond very low $d_z$. The paper states $O(T d_z^2)$ for applying the analytical solution, but also a one-time $O(d_z^3)$ per operating point to form the operator (e.g., eigendecomposition/inversion). For $d_z=50–100$, this rapidly dominates and can exceed the cost of Neural ODE field evaluations, undermining the claimed efficiency. The assertion that 5D PCA is “sufficient” is not substantiated by ablations or biological coverage analyses. And I also believe this assertion is not correct.
5. Wall-clock efficiency is not compelling. Despite a 5D latent space and no OT, the reported runs surprisingly high (one hour) for the stated computational simplicity, and notably slower than many baselines under comparable settings. The paper does not explain where time is spent (e.g., repeated eigendecompositions, kernel MMD computation) or provide profiling.
6. Redundant/ill-motivated invertibility regularization. The additional term $L_{\text{inv}}(\theta)$, encouraging $P_\theta$ invertibility, appears unnecessary: invertible matrices are dense; a randomly initialized square matrix is almost surely invertible. The paper does not justify why this regularizer is needed, nor analyze its numerical side-effects (e.g., unstable gradients via $\det(\cdot)$).
7. Interpolation results of OT baselines are unclear. In Table 1, several OT-based methods underperform the OT-Interpolate baseline, while Cell-MNN surpasses it. Since many methods eventually regress toward OT-derived supervision, it is unclear why Cell-MNN outperforms those indirect OT-fitting approaches. The paper should articulate the mechanism (e.g., bias/variance advantages of distribution-level MMD vs. velocity-level supervision, regularization effects) and provide controlled comparisons 
8. Amortized training procedure lacks essential details. Section 4.2 merges datasets in PCA space for amortized training and feeds a dataset index to the model, but does not explain the details.
9. Gene interaction validation lacks competitive baselines. Validation against TRRUST is useful, but the paper does not compare to other vector-field/Neural-ODE methods that could also extract Jacobian-based interactions under the same preprocessing. Without head-to-head comparisons, it remains unclear whether Cell-MNN offers any real advantage for GRN discovery beyond the convenience of an explicit linear operator.
10. Insufficient ablations/sensitivity analyses. Key design choices—kernel and bandwidth for MMD, discount factor $\gamma$, regularization weights $\lambda_{\text{kin}}$, $\lambda_{\text{inv}}$, latent dimension $d_z$, operator parameterization (e.g., fixing one eigenvalue to 0 vs. not ), and $\Delta t$ sampling—lack systematic ablations. Given that interpretability and stability hinge on these knobs, their impacts should be quantified.

### Questions
Due to these concerns, I lean towards rejecting the paper in its current form. Significant theoretical justification, experimental clarification, and comparison to prior ODE-based methods would be needed for a more favorable evaluation. The authors should: 

1. Clarify justification for latent-space linearization. (See weakness 1)
2. Discuss the relation to prior GRN-from-ODE works. (See weakness 2)
3. Substantiate the scalability argument. (See weakness 3)
4. Provide complexity and latent-dimension analysis.  (See weakness 4)
5. Report detailed runtime and efficiency breakdown.(See weakness 5)
6. Revisit the invertibility regularization. (See weakness 6)
7. Clarify interpolation results and superiority to OT baselines. Could the authors explain why Cell-MNN achieves superior interpolation despite training toward similar objectives?  (See weakness 7)
8. Elaborate on amortized training implementation. (See weakness 8)
9. Enhance GRN validation comparisons (e.g., trajectorynet, TIGON). (See weakness 9) 
10. Include hyperparameter and kernel sensitivity studies. (See weakness 10)

References
1. Xiaojie Qiu, Yan Zhang, Jorge D Martin-Rufino, Chen Weng, Shayan Hosseinzadeh, Dian Yang, Angela N Pogson, Marco Y Hein, Kyung Hoi Joseph Min, Li Wang, et al. Mapping transcriptomic vector fields of single cells. Cell, 185(4):690–711, 2022.
2. Alexander Tong, Jessie Huang, Guy Wolf, David Van Dijk, and Smita Krishnaswamy. Trajectorynet: A dynamic optimal transport network for modeling cellular dynamics. In International conference on machine learning, pages 9526–9536. PMLR, 2020.
3.  Yutong Sha, Yuchi Qiu, Peijie Zhou, and Qing Nie. Reconstructing growth and dynamic trajectories from single-cell transcriptomics data. Nature Machine Intelligence, 6(1):25– 39, 2024.

This review was independently written by the reviewer. An LLM was employed solely for minor phrasing and grammar improvements.

### Soundness
1

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
4

### Summary
This paper introduces a framework for learning a locally linear differential equation in latent space that describes gene expression changes during cell differentiation. The latent space itself is also a linear projection from gene space. A key idea of the paper is to make the dynamics analytically tractable so that gene interactions can be explicitly modeled.

### Strengths
Strengths: 
•	The idea is well-motivated, theoretically sound, and biologically meaningful.
•	The idea is clearly explained. The paper is well-written and I found it easy to read
•	The evaluations presented are overall sound and sensible.

### Weaknesses
Weaknesses:
•	Highly similar concept to VeloVAE, ICML 2022. VeloVAE fits a mixture of linear ODEs and uses an encoder-decoder framework to perform Bayesian inference of cell times and ODE parameters. Another highly similar paper is LatentVelo, which fits a neural ODE end-to-end to learn dynamics in the latent space. Dynamo uses a linear ODE to estimate the Jacobian of gene regulation.
•	The method seems more incremental than revolutionary. There are closely related approaches for the same problem. The move to a locally linear ODE doesn't seem that impactful to me.
•	Baseline chosen for gene regulatory network evaluation is very weak. There are dozens of gene regulatory network inference algorithms that take single-cell RNA-seq data as input. Comparing performance against these would be more informative. Dynamo (Qiu et al. Cell 2022) seems particularly relevant, because a stated goal of the method is to recover gene-gene interactions by estimating the Jacobian.
•	I understand the motivation for interpretability, but it seems that this local linearity would have to come with some loss of predictive power. Approaches like VeloVAE don’t suffer from this limitation while retaining interpretability. I don't understand how this restricted model can outperform less interpretable but more expressive models, apart from scalability concerns.
•	Evaluation in terms of rate parameters seems important. The locally linear ODE can be interpreted in terms of gene expression changes (RNA velocity), and thus it's important to benchmark against the class of methods that aim to estimate these rates directly. Ground truth in the form of metabolic labeling data (see Dynamo paper) is available for an increasing number of datasets.

### Questions
1. How is your approach different from VeloVAE and LatentVelo?
2. Why does your less expressive approach outperform more expressive previous models? This seems surprising, because I would expect the local linearity constraint to make the predictions less accurate.

### Soundness
3

### Presentation
3

### Contribution
2
