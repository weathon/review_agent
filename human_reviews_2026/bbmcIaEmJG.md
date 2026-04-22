# GDR-learners: Orthogonal Learning of Generative Models for Potential Outcomes

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Various deep generative models have been proposed to estimate potential outcomes distributions from observational data. However, none of them have the favorable theoretical property of general Neyman-orthogonality and, associated with it, quasi-oracle efficiency and double robustness. In this paper, we introduce a general suite of generative Neyman-orthogonal (doubly-robust) learners that estimate the conditional distributions of potential outcomes. Our proposed generative doubly-robust learners (GDR-learners) are flexible and can be instantiated with many state-of-the-art deep generative models. In particular, we develop GDR-learners based on (a) conditional normalizing flows (which we call GDR-CNFs), (b) conditional generative adversarial networks (GDR-CGANs), (c) conditional variational autoencoders (GDR-CVAEs), and (d) conditional diffusion models (GDR-CDMs). Unlike the existing methods, our GDR-learners possess the properties of quasi-oracle efficiency and rate double robustness, and are thus asymptotically optimal. In a series of (semi-)synthetic experiments, we demonstrate that our GDR-learners are very effective and outperform the existing methods in estimating the conditional distributions of potential outcomes.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces GDR-Learners (Gradient Disentangled Regularization Learners), a framework for causal representation learning that enforces gradient-based orthogonality among latent variables. The key idea is to regularize the model such that the gradients of reconstructed inputs with respect to different latent dimensions are orthogonal, promoting disentanglement between the causal mechanisms encoded in each latent. Theoretical analyses suggest that under mild assumptions, this orthogonality constraint helps achieve identifiable and disentangled representations up to monotone transformations. Empirical results on standard disentanglement benchmarks show consistent improvements over state-of-the-art baselines.

### Strengths
1. The paper tackles an important challenge in causal representation learning: achieving identifiable and interpretable factors without strong structural assumptions.
2. The proposed GDR term is simple to compute, model-agnostic, and can be seamlessly added to a wide range of encoder–decoder architectures.
3. Experimental results show that the method improves disentanglement scores across several datasets and remains robust when combined with other regularization terms.

### Weaknesses
1. While the paper provides some intuitive propositions, it does not establish a formal link between gradient orthogonality and causal identifiability. The theoretical results remain qualitative and lack proofs that guarantee uniqueness or sufficiency for disentanglement.
2. The paper mainly compares against unsupervised disentanglement models (β-VAE, FactorVAE, etc.) but omits direct comparison with causal representation learners such as iVAE or CausalVAE, which share similar objectives and stronger theoretical grounding.
3. The evaluation focuses entirely on disentanglement metrics. There is no demonstration that GDR-learners improve causal inference capabilities.
4. The gradient regularization term introduces additional computation and hyperparameters (e.g., λ). The paper does not provide runtime analysis or ablations to assess stability and sensitivity across hyperparameter choices.

### Questions
See Weaknesses Part.

### Soundness
3

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
4

### Summary
This proposes to employ conditional deep generative models for estimating the conditional distribution of a potential outcomes in a causal framework for conditional outcomes.  The key contributes lies in an estimation procedure (with appropriate objective functions) that guarantees  the general Newman-orthogonality constraints.  They also provide some theoretical guarantees by establishing the Quasi-oracle efficiency and double robustness.

### Strengths
+propose a class of deep generative models for modeling the conditional distribution of the potential outcomes that satisfy the Neyman-orthogonality and establish quasi-oracle efficiency and double robustness of the learners 

+ the paper is well-written with very clear presentations (e..g, the graphical presentation in Figure 3 is extremely helpful) 

+a thorough numerical study is included

### Weaknesses
-in my opinion, the key contribution lies in the formulation of the objective functions that ensures the Neyman-orthogonality (as given by equation (7)). This is not a strong weakness per say, but the degree of novelty is somewhat in question 

-given equation (9), one should further discuss the rates of the error based on the various conditional deep generative estimators, utilizing an existing  literature on theory and convergence rates of conditional VAE, conditional discussion models and. so on. 


- 

-

### Questions
Q1: can you provide more intuition and explanation of the bias correction of RA learner behind equation (7)? First glance, it weights the RA and IPTW learners. 

and a quick comment: GDR. should be defined the first time it was introduced ( in the abstract)

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes GDR-learners, a general two-stage framework for estimating the conditional distributions of potential outcomes from observational data using deep generative models. The key idea is to combine estimates of conditional distributions of potential outcomes with a doubly robust, Neyman-orthogonal target loss so that errors in estimation do not affect the second-stage objective to first order. The framework is model-agnostic and can be instantiated with conditional normalizing flows, GANs, VAEs, and diffusion models. The authors claim quasi-oracle efficiency and rate double robustness in theory, and report empirical gains over plug-in, RA, and IPTW alternatives on synthetic and semi-synthetic benchmarks.

### Strengths
1. The proposed method and objective is model-agnostic and applicable to multiple generative models, including CNF, CGAN, CVAE.
2. The proposed framework is able to computer conditional distribution of potential outcomes.
3. Empirical results show improved performance compared with other models.

### Weaknesses
1. Some theoretical conditions (“mild model-dependent convexity and positivity”) are not fully unpacked. The identifiability condtion i - iii are not clear.
2. The proposed method show show very large reported standard deviations in some datasets, such as HC-MNIST. Is this considered as instability. Or some other reason?
3. In additional to the median out-sample w2+std results, it is necessary to report the worst case (e.g., max difference between the estimated distribution and the truth) estimation results. This result will show the performance under failure modes.

### Questions
NA

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
4

### Summary
The authors propose an approach to generate/model potential outcomes using deep generative models such as diffusions, GANs etc in a wya that preseves the neyman orthogonality property, which allows for double robustness and efficient semiparametric type guarantees.

### Strengths
Originality: The ideas in the paper are original and non-trivial in my opinion. 
Significance: Causal ML and Neyman Orthogonality are certainly "hot" topics as of late, and bringing that together with deep generative models is a timely contribution that I find significant enough to meet the bar for ICLR. 
Clarity: The clarity/presentation style of the paper is something that needs some minor improvement. I think the expert will have no trouble understanding the paper, but for a general ML/AI audience this paper will be a rather difficult read. 
Quality: the quality and soundness of the math/theory portions of the paper is high. I find the arguments to be well reasoned and to the best of my knowledge, sound. The authors also performed some experiments which I find sufficient for this paper.

### Weaknesses
Clarity of presentation and accessibility of writing are the biggest areas that this paper can improve on. I highlight some aspects of this below. 

1. GDR, Neyman Orthogonality, Double robustness etc are the core concepts that underlie and motivate this paper. Yet, neyman orthogonality was not formally defined until page 6 of the paper (inside a theorem, and then more formally in the appendix), and double robustness, quasi-oracle efficiency etc are nowhere formally defined in the main text (high level explanations in 2nd page and formal definitions only given in the appendix). this is understandable given space constraints, but I believe some more formal/rigorous explanation of these terms in the beginning of the main text is *absolutely required*. These are not part of the standard training/vocabulary of a typical ICLR or general AI/ML reader, and clarifying these terms will help broaden the reach of the paper to a more general audience. 

2. There are important references (canonical, even) that this paper seems to have missed. Many examples in econometrics come to mind, with the double machine learning paper by Chernozhukov et al and related semiparametric efficiency papers (e.g. Whitney Newey) not mentioned. There are too many references to be listed here. I suggest giving more proper reference to the econometrics literature.

### Questions
1. Typically, one thinks of these DR/Neyman orthogonal approaches are using ML to nonparametrically handle a nuisance with the bigger goal to infer a finite dimensional parameter of interest, in a way that the slow convergence of the nonparametric part doesn't affect the nice properties of the finite dimensional inference on the parameter.  In your case, the parameter of interest appears to be the CDPOs, which themselves are nonparametrically generated. In this case, one can reasonably think that learning the CDPOs themselves might be a more difficult problem than learning the nuisance, which calls into question the value of double robustness/quasi oracle efficiency in this setting. For example ,what if the rate of learning the CDPOs is even slower than estimating the nuisance? 


Minor points: the abbreviation GDR is used in the abstract before definition

### Soundness
4

### Presentation
2

### Contribution
3
