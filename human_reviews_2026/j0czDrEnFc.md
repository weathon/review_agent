# A Bayesian Nonparametric Framework for Private, Fair, and Balanced Tabular Data Synthesis

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 2, 6, 6

## Abstract
A fundamental challenge in data synthesis is protecting the fairness and privacy of
the individual, particularly in data-scarce environments where underrepresented
groups are at risk of further marginalization by reproducing the biases inherent in
the data modeling process. We introduce a privacy- and fairness-aware for a class
of generative models, which fuses the conditional generator within the framework
of Bayesian nonparametric learning (BNPL). This conditional structure imposes
fairness constraints in our generative model by minimizing the mutual information
between generated outcomes and protected attributes. Unlike existing methods
that primarily focus on sensitive binary-valued attributes, our framework extends
seamlessly to non-binary attributes. Moreover, our method provides a systematic
solution to class imbalance, ensuring adequate representation of underrepresented
protected groups. Our proposed approach offers a scalable, privacy-preserving
framework for ethical and equitable data generation, which we demonstrate by
theoretical guarantees and extensive experiments on sensitive empirical examples.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Proposes a Bayesian–nonparametric pipeline for tabular synthetic data that jointly enforces (i) privacy via Dirichlet-process (global) and copula-based (localized) mechanisms, (ii) fairness via minimizing mutual information between outcomes and protected attributes, and (iii) class balancing by conditioning the generator on group labels; instantiated with a conditional VAE+GAN and evaluated on Adult and COMPAS.

### Strengths
1. The authors unified objectives across privacy–fairness–balance. BNPL resampling + MI regularizer + conditional generation is a complete solution to DP + fairness.
2. DP analysis for a Dirichlet mechanism and localized privacy via copula-based marginals are novel in this combination.

### Weaknesses
1. Fairness target & baselines. The method mainly targets statistical parity via MI; could you please justify this choice against equalized odds/opportunity, and have some head-to-head comparisons (or at least rank-correlations) to fairness baselines beyond DECAF/TabFairGAN/FairGAN? 
2. Scalability/“high-dim” claims. DirPMINE is proposed for scalable MI, yet there’s no analysis of variance, sample complexity, or failure modes versus standard MINE. It would be better if authors could discuss more about runtime/memory, or stability across hyperparameters, and add confidence intervals for MI/MMD/utility.
3. Some simple examples/simulation for intuitive insights. It would be clearer if there were any toy experiment showing: (i) how DP resampling alters the empirical distribution, (ii) how localized privacy trades utility vs. protection per column, (iii) how class balancing alone affects SP/MI vs. fairness-regularization alone. The current figures hint at effects but don’t isolate mechanisms.
4. Datasets. The authors provide two datasets. However, I feel these two choices are rather too limited. In recent years, there've been multiple studies discussing the limitations of these two datasets (https://arxiv.org/abs/2108.04884 and https://arxiv.org/abs/2106.05498). It's more convincing if there are more datasets considered or include more discussion about the limitations of these datasets.

### Questions
Please see weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a conditional Bayesian nonparametric (BNP) framework for tabular data synthesis that aims to jointly enforce privacy, fairness, and class balance. Privacy is introduced by resampling from a Dirichlet-process posterior (a finite approximation of DP(a, H)), yielding a distributional randomization that the authors formalize as a Dirichlet mechanism with a DP guarantee on the posterior weights. Fairness is enforced by minimizing dependence between outcomes and protected attributes via a mutual-information regularizer based on a BNP variant of the Donsker–Varadhan lower bound (“DirPMINE”). Class balance is handled by conditioning the generator on protected attributes and adding a KL-to-uniform balancing term for other discrete columns. Experiments on Adult and COMPAS suggest improved fairness/utility trade-offs relative to FairGAN/DECAF while supporting non-binary sensitive attributes.

### Strengths
This paper presents a unified objective that couples utility with MI-based fairness and KL-based balancing, all inside a single conditional generator (VAECGAN), which is practical for tabular data.

### Weaknesses
1. I think there a few prior works that mention that DP-SGD can disproportionately harm minority groups via clipping + noise. But still, I don't fully understand what is the extra challenge when we consider both privacy and fairness at the same time. It would be better if the authors could address the interaction between privacy and fairness in the up front. In practice, enforcing fairness requires accurate group-conditioned statistics which DP then perturbs and budgets, especially hurting small groups; meanwhile fairness constraints can further tighten an already noisy optimization. So it’s unclear whether the proposed coupling makes the joint problem harder or sometimes acts as a helpful regularizer. To me, this paper feels like a combination of two constraints, but it is not clear how these two constraints play together. Is there any lowerbound for canonical private and fair problems like generative modeling of gaussian distributions?

2. (Minor) the fonts in the plots are hard to read.

### Questions
1. Proposition 1 defines adjacency on the probability simplex for posterior weights. How does this mapping correspond to standard record-level neighboring datasets in DP, and what guarantee applies to the atoms/locations vs. just the weights?
2. When combining the global Dirichlet mechanism with local per-attribute privatization, what is the resulting epsilon delta after composition?
3. Is it possible to have a plot to draw the privacy–fairness–utility curves for the trade-off?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes an approach for generating fair and private tabular data by integrating a conditional generator within the framework of Bayesian non-parametric learning. The main objective of the approach is to be able to generate synthetic data that is private, fair and balanced by leveraging a mutual information regularization term which is conditioned on the protected attributes.

### Strengths
-The paper is well-written and the authors have clearly reviewed the challenges associated with the generation of synthetic tabular data. The main contributions are clearly summarized and the outline of the paper is clearly described.

-The proposed approach combines in an innovative way the Dirichlet process together with differentially-private mechanisms such the analytic Gaussian mechanism and randomized response to produce a new differentially-private generative model. One of the strength of the approach is can be combined with a wide range of generator-decoder models. For instance, in the current version of the paper it is implemented through a generative adversarial network combined with variational auto encoder.

-Detailed investigations have been conducted on the Adult and Compas dataset demonstrating that the model is able to generate high quality data that is both privacy-preserving and fair.

### Weaknesses
-While the literature review surveys a wide range of approaches for privacy-preserving, fair or balanced tabular data generation there are no mention of existing works at the intersection of these domains such as for instance :
-David Pujol, Amir Gilad, and Ashwin Machanavajjhala. 2024. PreFair: Privately Generating Justifiably Fair Synthetic Data. In Proceedings of the VLDB Endowment (PVLDB), Vol. 16. https://www.vldb.org/pvldb/vol16/p1573-pujol.pdf
-Sarmin, F. J., Rahman, A. R., Henry, C. J., & Mohammed, N. (2025). Privacy-Preserving Fair Synthetic Tabular Data. arXiv preprint arXiv:2503.02968.
Additionally, how the proposed approach builds on the Dirichlet mechanism from Gohari el al. 2021 should be further clarified. 

-The privacy budget considered are quite high and should be better justified. The privacy analysis of the approach should also integrate an analysis of the success of the privacy attacks such as membership inference to be able to assess empirically the strength of the approach provided. 

-One of the limit of the approach is that for now it seems limited to the statistical parity fairness metric and there is no discussion if it could easily be extended to integrates other group fairness metrics.

-The figures 2 and 3 are not really visible and should be improved.

### Questions
Please see the main points raised in the weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
3
