# Geometric Moment Alignment for Domain Adaptation via Siegel Embeddings

- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
We address the problem of distribution shift in unsupervised domain adaptation with a moment-matching approach. Existing methods typically align low-order statistical moments of the source and target distributions in an embedding space using ad-hoc similarity measures. We propose a principled alternative that instead leverages the intrinsic geometry of these distributions by adopting a Riemannian distance for this alignment. Our key novelty lies in expressing the first- and second-order moments as a single symmetric positive definite (SPD) matrix through Siegel embeddings. This enables simultaneous adaptation of both moments using the natural geometric distance on the shared manifold of SPD matrices, preserving the  the mean and covariance structure of the source and target distributions and yielding a more faithful metric for cross-domain comparison. We connect the Riemannian manifold distance to the target-domain error bound, and validate the method on image denoising and image classification benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles unsupervised domain adaptation (UDA) under covariate shift, where input distributions differ between source and target domains (but label conditionals remain unchanged). The authors claim that most standard moment-matching methods align low-order moments (means, covariances) between domains using Euclidean distances, ignoring the true geometry of statistical distributions (this is partially true, however the author themselves list many works that "have replaced ad-hoc Euclidean distances with geometry-aware alternatives").

The authors propose a geometrically principled moment alignment approach using Riemannian geometry on the SPD manifold.
Their main idea is to combine first- and second-order moments (mean and covariance) into a single SPD matrix using Siegel embedding and measure domain discrepancy via manifold distances, rather than arbitrary Euclidean metrics.
The method, calle Geometric Moment Alignment (GeoAdapt), comes in two variants: GeoAdapt-AIRD: using Affine-Invariant Riemannian Distance (AIRD) and GeoAdapt-HPD: using a faster approximation, the Hilbert Projective Distance (HPD).

Theorem 1 provides an upper bound on the target-domain generalization error in the case of HPD. 

Experiments are performed on two tasks:
(Sec. 4.1) Unsupervised Task – Image Denoising on MNIST and Fashion-MNIST (clean  $\rightarrow$  noisy domains).
(Sec 4.2) Supervised Task – Image Classification on Office-31 (A, D, W domains) and VisDA-2017 (synthetic $\rightarrow$  real domains).
Results show that  GeoAdapt-AIRD achieves best overall performance outperforming Euclidean and other geometric baselines (e.g. MECA, HoMM).

Some insight provided in the paper:
- Low-dimensional embeddings (32–128) outperform higher ones, as they avoid numerical instability and lie within well-behaved SPD regions.
- Euclidean methods degrade in high dimensions due to rank-deficient covariance estimates.
- The geometric approach explains why classical moment-matching fails under strong curvature or low sample conditions.
- The authors claim that the method is architecture-agnostic and can be integrated into any DA framework using a domain discrepancy term.

### Strengths
The paper presents a sound theoretical approach for matching distributions through moments: Embedding first- and second-order moments as a single SPD matrix via Siegel embeddings is a mathematically elegant and novel idea that unifies mean and covariance adaptation in one structure. The method is effectively applied to Unsupervised Domain Adaptation.

The paper is crystal clear, easy to follow and result presented highlight the superiority of the proposed method wrt other moment-matching methodologies.

The authors provide a theoretical upper bound on the target-domain generalization error (for the Hilbert Projective Distance), extending classical domain adaptation theory.

The paper shows that compact embedding spaces (32–128 dimensions) are sufficient or even superior — improving accuracy and stability while reducing computational cost. This can be seen as a plus in case the task does not require more expressive (i.e. larger) embedding vectors.

I appreciate the authors providing the code.

### Weaknesses
Please see my points below, mainly related to the experimental validation:

1. Few benchmark and domains:

   Only two main image classification benchmarks (Office-31 and VisDA-2017) and one synthetic/toy denoising setup (MNIST/Fashion-MNIST) are tested. Missing: Large-scale or more diverse datasets (e.g., DomainNet, WILDS) and possibly non-visual domains (e.g., text, speech).

2. Limited model comparison:

   Experiments are limited to ResNet-50 and a simple autoencoder. No tests with modern backbones (e.g., ViTs, CLIP, or U-nets w/ diffusion for denoising). This is quite critical given that the authors claim that the method is architecture-agnostic. One would expect a concrete proof of that statement. Besides, more modern pretrained backbones provide higher train-on-source performances which make adaptation less critical.

3. Bounds:

   Train-on-target performance is useful to be reported in the tables as an upper bound.  It would be also interesting to see in practice what would be  the _estimated_ theoretical upper bound which I guess can be easily calculated from the training loss functions. In fact (assuming $\gamma=0$) the upper bound has a term which is the error on the source set (should be easily derived from the training task loss) and a term which is the distance between the source and target distributions. 

4. Modern Baselines:

   The paper mainly compares with older moment-matching baselines (MMD, CMD, CORAL, HoMM, MECA) and lacks comparisons with more recent and powerful UDA methods (e.g. adversarial methods). This makes it difficult to assess state-of-the-art competitiveness - the authors note this themselves, arguing they isolate the “distance effect,” yet this is still a limitation. 

5. Limited Ablation:

   Hyperparameter sensitivity (e.g., $\beta$ for $\mathcal{L}\_{dist}$, mini-batch size is not fully explored. It’s unclear how robust the approach is to these settings or to noisy covariance estimates in small batches. Larger batches should also in principle allow for larger embedding vectors, which could be needed for downstream tasks where more expressiveness of the model could be essential.

6.  Computational Analysis:

    Matrix operations on SPD manifolds (logarithms, inverses, eigen-decompositions) can become computationally heavy even for low-dimensional embeddings and small batches. The paper is not discussing this point.

7. Validation:

   Validation is a long-standing issue in UDA. In principle one should not peek at target metrics for tuning the hyper-parameters, since this would mean validating on the test set. One strategy is to use a toy dataset for validation of the  hyper-parameters (e.g. SVHN $\rightarrow$ MNIST) and then use the same hyper-parameters for the other benchmarks. Alternatively, MECA proposes a criterion based on the estimation of the entropy on the predictions of the target. The paper is not discussing the validation issue in any respect.

### Questions
Please see and discuss the points I raised above. They already include questions and points to be clarified or expanded.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work studies the domain adaptation problem, which aims to learn a generalization model under the observed source domain and target domain. This work considers the covariate shift framework and points out that existing works mainly focus on the low-order statistics. To this end, this work explores the first-order and second-order statistics as distribution parameters and adopts metrics on manifold to measure the distance over high-order domain representation, i.e., SPD matrix consists of mean and covariance. Theoretical results show that the generalization could be bounded by the developed manifold metric-based method.

### Strengths
+ The application of manifold metric for matrix-based domain representations is reasonable.

+ The organization is easy to follow.

### Weaknesses
+ The key idea of representing domains as manifolds or statistics on manifolds is extensively studied by existing works, which are not properly compared in the submission. 

+ The limitations of existing works seem to be over-claimed, since there are many works that already consider the high-order statistics or statistics with stronger power. 

+ The theoretical result is trivial, and not much new insight is provided.

### Questions
**Concerns**

Concern 1. One of the essential ideas is adopting the manifold metric to measure the domain gap/distance over the mean-covariance representation, which essentially shares the same spirit of existing works that consider manifold representation and Riemmanian metric, e.g., statistical manifold [r1], Riemannian manifold [r2], Log-Euclidean metric with better efficiency [r3], Kernel Geodesic [r4], affine-invariant metric [r5]. However, there are only statistical moment-based methods compared, which cannot completely demonstrate the significance of the proposed method.


Concern 2. The limitations of existing works seem to be improper. Since there are many work that considers the high order statistics, which also admit stronger properties on distribution distance, e.g., kernel Wasserstein with mean and covariance w.r.t. RKHS, conditional moments with multi-variable correlation characterization. Moreover, note that the kernel embedding can be taken as moment with infinite order with a proper choice of kernel, e.g., Gaussian kernel, since the corresponding RKHS is an approximation of the space of continuous functions and the moments are taken within such a space.

Concern 3. Though geometric metric is adopted, there are no general guarantees for the distribution discrepancy minimization. The key to connecting the explored metric with statistical distance is the Gaussian prior on distributions. However, such a result seems to be trivial if the Gaussian prior is adopted, as many metrics could also be connected to statistical distance while also endowed with explicit computational formulation, e.g., (kernel) Wasserstein with geometric property, some metrics in $f$-divergence family. 

Concern 4. The generalization error analysis does not provide new insights and seems to be loose. Specifically, the bound is obtained based on existing upper bound where two inequalities are successively applied (which could amplify the error).


**References**

[r1] Baktashmotlagh, Mahsa, et al. "Domain adaptation on the statistical manifold." Proceedings of the IEEE conference on computer vision and pattern recognition. 2014.

[r2] Luo, You-Wei, et al. "Unsupervised domain adaptation via discriminative manifold propagation." IEEE transactions on pattern analysis and machine intelligence 44.3 (2020): 1653-1669.

[r3] Cui, Zhen, et al. "Flowing on Riemannian manifold: Domain adaptation by shifting covariance." IEEE transactions on cybernetics 44.12 (2014): 2264-2273.

[r4] Zhang, Youshan, and Brian D. Davison. "Deep spherical manifold gaussian kernel for unsupervised domain adaptation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.

[r5] Yair, Or, Mirela Ben-Chen, and Ronen Talmon. "Parallel transport on the cone manifold of SPD matrices for domain adaptation." IEEE Transactions on Signal Processing 67.7 (2019): 1797-1811.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this paper, the authors developed a novel moment-matching method for unsupervised domain adaptation (UDA) and evaluated it on image classification and image denoising tasks. The authors do not use state-of-the-art architectures, and I couldn't find a sufficient number of experiments, or detailed data analysis to clearly explain why the method performs better.

### Strengths
The novelty of the method lies in the fact that, instead of matching means and covariances separately, using arbitrary distance metrics,  both can be encoded into a single SPD matrix and can leverage the natural geometry of the SPD manifold (via Siegel embeddings) to compute distances more appropriately.


The method can be plugged in to the other DA methods.

### Weaknesses
The authors could use modern architectures as backbones, such as Vision Transformers (ViT).


Can you please clarify why there is no benchmark on non–moment-matching UDA methods? Is there a specific reason or application context that limits the proposed approach to moment-matching UDA only?


For image denoising, the test are conducted only on two datasets, and for image classification, the improvement achieved over existing methods is not significant.


HoMM and CMD utilize up to third-order moments for UDA. Can you please explain why your proposed method performs better despite using only first and second moments? Additionally, the approach may be limited when applied to datasets that do not follow a Gaussian feature distribution, which could reduce its overall applicability.


For SPD manifolds, there are several possible embedding methods. Can you please clarify the reason for choosing this particular one?


Any justification or experimental evidence for the choice of $\alpha_1$ in Section 3.1 would be great.

### Questions
Please see weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates how to compute similarities and transform statistical moments more effectively for domain adaptation tasks.

The authors leverage differential geometry and map the latent representations of both source and target domains through a diffeomorphic transformation into the SPD (Symmetric Positive Definite) manifold.

This transformation jointly encodes the first and second moments into a single SPD matrix.

By exploiting the Riemannian structure of this manifold, the authors define two geometrically inspired distance measures—Affine-Invariant Riemannian distance and Hilbert Projective distance—to quantify the discrepancy between domains.

### Strengths
The paper is mathematically rigorous and presents a well-grounded theoretical derivation.

The method achieves competitive performance on both supervised (classification) and unsupervised (denoising) tasks, demonstrating strong generality.

### Weaknesses
The baselines used are somewhat outdated — the most recent compared method (HOMM) was published in 2020. It would strengthen the paper to include comparisons with more recent works.

It would be interesting to evaluate the proposed approach using CLIP embeddings or other strong pretrained features on more complex, large-scale, or cross-domain datasets to further test its scalability and robustness.

### Questions
Please see the weakness

### Soundness
3

### Presentation
3

### Contribution
3
