# PALATE: Peculiar Application of the Law of Total Expectation to Enhance the Evaluation of Deep Generative Models

- Decision: Reject
- Scores: 6, 2, 2, 0

## Abstract
Deep generative models (DGMs) have caused a paradigm shift in the field of machine learning, yielding noteworthy advancements in domains such as image synthesis, natural language processing, and other related areas. However, a comprehensive evaluation of these models that accounts for the trichotomy between fidelity, diversity, and novelty in generated samples remains a formidable challenge. A recently introduced solution that has emerged as a promising approach in this regard is the Feature Likelihood Divergence (FLD), a method that offers a theoretically motivated practical tool, yet also exhibits some computational challenges. In this paper, we propose PALATE, a novel enhancement to the evaluation of DGMs that addresses the limitations of FLD regarding computational efficiency. Our approach is based on a peculiar application of the law of total expectation to random variables representing accessible real data. When combined with the MMD baseline metric and DINOv2 feature extractor, PALATE offers a holistic evaluation framework that matches or surpasses state-of-the-art solutions while providing superior computational efficiency and scalability to large-scale datasets. Through a series of experiments, we demonstrate the effectiveness of the PALATE enhancement, contributing a computationally efficient, holistic evaluation approach that advances the field of DGMs assessment, especially in detecting sample memorization and evaluating generalization capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces an evaluation framework that contrasts training data points with generated samples by leveraging the law of total expectation. Concretely, the authors define a ratio-based score (PALATE) that decomposes an expectation-based baseline metric into contributions conditioned on whether a real sample comes from the train or the held-out set, so that the score rises when generations are closer to the training set (a signal of memorization) than to the test set. Because this ratio alone can have limited dynamic range in some regimes, they further propose a weighted metric (MPALATE) that takes a convex combination of the classical baseline and the new ratio—thereby balancing fidelity/diversity sensitivity (from the baseline) with novelty/memorization sensitivity (from PALATE) and stabilizing behavior across datasets. The framework is instantiated with MMD computed on pretrained feature embeddings (RBF kernel), and evaluated on standard image benchmarks; results are compared against FLD (a holistic but more resource-hungry baseline) and FID (fidelity-oriented). Empirically, the method aims to retain competitive fidelity/diversity tracking while improving memorization detection and offering better computational scalability than FLD.

### Strengths
- **Well-written with a fair experimental setup.** The paper is clear, and the experiments are aligned closely with **FLD** (datasets, budgets, protocol), which makes the comparison credible.  
- **Elegant formulation via the law of total expectation.** The construction cleanly injects a **novelty/memorization** signal into an expectation-based baseline without changing the two-sample structure, so it’s easy to adopt.  
- **Computational efficiency is convincingly demonstrated.** By avoiding KDE and using **V-statistic MMD** with blockwise kernel computations, the method scales to larger sample sizes where FLD becomes cumbersome. Attention to feature maps and implementation details (block sizes, batching) is practical and goes beyond what comparable work typically reports.  
- **Plug-and-play with existing pipelines.** Since the base quantity is an expectation, the approach can wrap standard metrics with minimal engineering.

### Weaknesses
- **Arbitrary design choices with limited sensitivity analysis.** Several decisions feel ad hoc (choice of **MMD**, kernel family/bandwidth, feature extractor, the convex weight **α**, score scaling). 
- **Lack of non-image modalities.** Recent work extends distributional metrics from **images to text**; adding text experiments (even small-scale) using the framework with textual baselines such as **MAUVE** or precision/recall for text models would strengthen the claim of generality [1, 2, 3].  
- **No disentanglement of quality vs. diversity in the chosen base metric.** MMD is a single-number distance and cannot separate fidelity from coverage. However, **precision/recall**  estimated as in [4] or [5] can be written as **f-divergences** [6,7], hence as **expectations of a function**. They should fit your framework as naturally as MMD, with the key advantage of **disentangling quality and diversity**. Exploring **PALATE-Precision** and **PALATE-Recall** (and their weighted variants) could reveal more nuanced behaviors.  


---
[1] Pillutla et al., **MAUVE: Measuring the Gap Between Neural Text and Human Text using Divergence Frontiers**, NeurIPS 34.  

[2] Le Bronnec et al., **Exploring Precision and Recall to assess the quality and diversity of LLMs**, ACL.  

[3] Pimentel et al., **On the usefulness of embeddings, clusters and strings for text generator evaluation**, ICLR 2023.  

[4] Kynkäänniemi et al., **Improved Precision and Recall Metric for Assessing Generative Models**, NeurIPS 2019.

[5] Kim et al., **TopP&R: Robust Support Estimation Approach for Evaluating Fidelity and Diversity in Generative Models**, NeurIPS 2025.

[6] Siry et al., **On the Theoretical Equivalence of Several Trade-Off Curves Assessing Statistical Proximity**, JMLR 2023.  

[7] Verine et al., **Precision-Recall Divergence Optimization for Generative Modeling with GANs and Normalizing Flows**, NeurIPS 37.

### Questions
- Why prioritize **MMD** over alternative expectation-form distances (e.g., energy distance, discriminators)? 
- Since **precision/recall** families can be written as **f-divergences** (hence expectations), could you implement **PALATE-Precision** and **PALATE-Recall**? This would naturally separate fidelity from coverage and might provide clearer guidance to practitioners.  
- Is the computational gain is only over **FLD**, or does it also hold against other expectation-based metrics such as **FID** or **KID**, **PRDC**?

### Soundness
3

### Presentation
4

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
The authors propose PALATE, a new approach to evaluating deep generative models that builds upon the law of total expectation and prior work such as the FLD. A main idea is to leverage a total expectation that is based on a combination of an expectation on the testing data and an expectation on the training data to distinguish models that are merely memorizing.

### Strengths
Overall, I find the clarity and quality aspects to be strengths of this paper. I discuss significance and originality more in the weaknesses section. 

Clarity: The clarity and accessibility of writing is a strength of this paper. The exposition is simple and clear, and the writing is easy to follow to a general ML audience. Overall it was a pleasant read. 
Quality: I comment on the quality of 1. the math/theory portion of the paper 2. the empirical portion of the paper. 
The quality of the mathematical expressions/technical portion is sound to the best of my knowledge. There is minimal theoretical developments in the paper, but this is not a problem since this paper's main contribution is methodological. The experimental section is structured reasonably and appropriate comparisons to competitors are made. 
Significance: The problem of deep generative model evaluation is a significant and timely one. While the topic is significant, whether the paper's contribution is significant is discussed in the next section. 
Originality:  I comment on the originality of the paper's contribution in the next section.

### Weaknesses
On the significance and originality front, I have the following comments and questions. 

1. The core idea of PALATE depends on a split of the data into training and testing portion, and then comparing the relative contributions of each component to determine and balance whether the data is memorizing/overfitted to the training data and whether it faithfully generates points similar to the test data. 

However, this core idea of train test split has been studied and proposed in the machine learning and statistics literature for many decades. In this sense, I find the core idea less original than the paper's description and less significant of a conceptual advance. I also think that there is too little discussion of the relationship with traditional data splitting/cross validation ideas. 

2. the PALATE functional form essentially compares a ratio between an expected "loss" on the test data and an expected loss on the entire data. While this functional form is, to the best of my knowledge, new in the deep generative modeling context, I question whether this is a sufficiently significant/original contribution for a venue like ICLR. 

3. Simplicity if a virture, and the fact that the paper uses relatively elementary/simple mathematical tools is a good thing, so my comment below is not a criticism against the technical depth of the paper per se. However, I find the paper's emphasis on the law of total expectation quite confusing....the law of total expectation is such an elementary/fundamental technique that I find emphasizing that no different from emphasizing other basic things like the laws of arithmetic. I find it difficult to see the significant advancement when such an elementary tool is applied in such a straightforward manner.

### Questions
1. The authors reference Weiss et al for the law of total expectation. However, I don't find such a citation satisfactory. If the authors want to emphasize the law of total expectation, then proper reference to this law is warranted. Candidate references that might suffice include the original work of Kolmogorov and the classical expositions by Feller. 

2. I do not find the application of the law of total expectation "peculiar" in any way. It is just a straightforward mathematical manipulation. Could the authors clarify what they mean by peculiar?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses the challenge of evaluating generative models by enhancing the Feature Likelihood Divergence (FLD) metric. The authors use the law of total expectation (tower rule) to reformulate the underlying evaluation process. They provide numerical results on CIFAR-10 and ImageNet (256×256) datasets generated by various generative models.

### Strengths
- The paper addresses an important problem in evaluating generative models and tries to tackle the computational issues of FLD.

- The paper uses DINOv2 feature embeddings, which have been shown to provide more reliable representations.

### Weaknesses
My major concern is the novelty of the contribution. The authors apply the law of total expectation (as explained in Theorem 1 and proved in the Appendix) to improve FLD. From a theoretical perspective, this improvement is incremental and does not address a substantial challenge. From an experimental perspective, most experiments are conducted using the CIFAR-10 dataset, which may not adequately demonstrate scalability or generalization to larger datasets. Below are more detailed versions of my concerns:

- Theoretical Basis: Using the law of total expectation gives a clear mathematical explanation, but the idea itself is mostly a reframing of existing metrics rather than a truly new approach.

- Feature Embedding: Using DINOv2 as the feature embedding is not original; this was proposed in [1], where the authors extensively studied embeddings and suggested that DINOv2 is a better alternative.

- Figure 4: Although the M$_{PALATE}$ score computation is faster, the score does not converge even with 10,000 samples, while FLD appears to converge with 8,000. This raises the question of what the sample complexity of your proposed score is. Do you have theoretical or experimental results for the convergence of your score?

- Also, the related work section could be improved by including more recent evaluation metrics. For example, Density/Coverage [2], Vendi score [3], FKEA [4], and RKE score [5] for fidelity/diversity evaluation, and FINC [6] for assessing novelty.

---

[1] Stein et. al, "Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models", NeurIPS 2023

[2] Naeem, M., et al., “Reliable Fidelity and Diversity Metrics for Generative Models”, NeurIPS 2020.

[3] D. Friedman and A. B. Dieng, “The Vendi Score: A Diversity Evaluation Metric for Machine Learning.”, TMLR 2023

[4] Ospanov et al., “Towards a scalable reference-free evaluation of generative models”, NeurIPS 2024.

[5] Jalali et al., “An information-theoretic evaluation of generative models in learning multi-modal distributions”, NeurIPS 2023.

[6] Zhang et al., “Unveiling Differences in Generative Models: A Scalable Differential Clustering Approach”, CVPR 2025.

### Questions
- Figure 1: Which PFGM++ samples were used? Are these CIFAR-10 samples?

- Figure 2: Scores were reported for 1, 2, 5, 10 or 100, 200, 500, 1000 classes, which sometimes do not follow the expected trend. Is there a specific reason for this?

- Figure 3: Why are comparisons limited to AuthPCT and C_$T$? Why not include other baselines mentioned in the related work, such as KEN score and Rarity score?

- Figure 4: Scores were reported on CIFAR-10, while computation time is reported on the ImageNet dataset, which is inconsistent. Is there a specific reason for this? (I noticed that part of Figure 7 shows ImageNet, but the inconsistency remains.)

- Figure 5: There is a large gap between 80% and 100% for FLD, suggesting that FLD may be sensitive when all samples are from the training set. Can results be provided for the range [0, 90]% so that the results are more comparable?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
The premise of this paper makes little sense. See attached questions. Unless it is revised to meet a basic scientific standard (precise definitions and sensible justifications behind the modeling choices) I won't be able to assess this manuscript.

### Strengths
Until the premise of this paper is qualified I cannot assess its strenghts.

### Weaknesses
Until the premise of this paper is qualified I cannot assess its weaknesses.

typo: In line 209 the term P(Z | X in M_test) does not parse. Z is a random variable.

### Questions
Line 97: "we have access to two distinct collections of real data: a train dataset and a test dataset, with the former dedicated for training and the latter for evaluation purposes exclusively. These data can be considered independently drawn from a random variable X acting on a given multidimensional Euclidean data space."

Line 190: "The following assumptions underpin our approach: (A1) the samples of training data which are selected for evaluation, are
contained in two non-trivial disjoint parts1 Mtrain and Mtest of the manifold of data M, respectively." 

These two assumptions are inconsistent. If the training and test datasets are sampled from the same dataset, how can they be contained in disjoint parts of whatever manifold you are talking about? What is the DEFINITION of Mtrain and Mtest? 

Line 198: "note that any such Z must implicitly depend on the random variables X and Y —see Equation (5) for an example." No it must not. The random variable Z that is identically equal to zero satisfies E[Z] = 0 but does not "implicitly depend" on any X and Y. There are myriad examples of random variables that have mean zero but do not "implicitly depend" on your X and Y.

### Soundness
1

### Presentation
1

### Contribution
1
