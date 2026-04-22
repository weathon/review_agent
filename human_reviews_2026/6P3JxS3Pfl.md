# Variational Learning of Disentangled Representations

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 2

## Abstract
Disentangled representations allow models to separate factors shared across conditions from those that are condition-specific. This separation is crucial in domains such as biomedicine, where generalization to new treatments, patients, or species requires isolating stable biological signals from context-dependent effects. While several VAE-based extensions aim to achieve this, they often exhibit leakage between latent variables, limiting generalization.
We introduce DisCoVR, a variational framework that explicitly separates invariant and condition-specific factors through: (i) a dual-latent architecture, (ii) parallel reconstructions to keep both representations informative, and (iii) a max–min objective that enforces separation without handcrafted priors.
We show this objective maximizes data likelihood, promotes disentanglement, and admits a unique equilibrium.
Empirically, DisCoVR achieves stronger disentanglement on synthetic data, natural images, and single-cell RNA-seq datasets, establishing it as a principled approach for multi-condition representation learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
DisCoVR is a variational framework that separates invariant and condition-specific factors using a dual-latent architecture, parallel reconstructions, and a max–min objective to improve disentanglement and generalization across conditions.

### Strengths
1. The theoretical part, including statistical derivations and optimization, is very solid.

2. The experimental part is also comprehensive and solid, with per-epoch runtime statistics for multiple baselines and full hyperparameter details; the information is thorough and should enable strong reproducibility.

### Weaknesses
1. lines 216-217 'maximizing this lower bound on I(z;y) also maximizes I(z;y)' do you mean minimize? And even it corrected as minizing, it is still not rigor to say 'minimizing the lower bound imply minimizing the value itself'.

2. line 211 says using 'logistic regression', but in appendix it seems you also use MLP for here.
 
3. In theoretical derivation, you require Qz, Qw to be convex and compact (Proposition 2.2 and standard regularity conditions). And in 193-195, you choose d-dimensional Gaussian wiith diagonal covariance matrices. Could you analyze whether and to what extent this choice meets the theoretical requirements?

4. Typo : line 143 'the the'

### Questions
See Weaknesses.

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
This paper presents a VAE for disentanglement of condition-specific factors and condition-invariant (shared) factors, where condition=label. The key elements of model design include two separate learned latent representations z (condition-invariant) and w (condition-specific) along with separate reconstruction terms from z and w, factorization of q_{z,w | x,y} to q_{z|x} and q_{w|x,y}, coupling of z and w in the prior of w, and adversarial component with the encoder tries to maximize classifier loss g_{y|z} while the classifier tries to minimize it to make representation z independent of y. The method (DisCoVR) is evaluated on several synthetic, natural, and biological dataset for evaluation of disentanglement versus several disentangling VAEs.

### Strengths
- The design of the optimization and DisCoVR is grounded in the theory of variance inference and probabilistic graphical models throughout, providing rigor to the problem formulation.

- The idea of separating condition-invariance and -specific representations is interesting and has novelty. 

- The comparison with related baselines is done both analytically and experimentally, again providing rigor to the formulation of DisCoVR.

- Experimentation considered a variety of datasets ranging from synthetic toy data to biological data.

### Weaknesses
The decision to couple the prior of w and z is not very well justified or explained. It can be understood that doing so will require z to be informative, but the informativeness of z should already be encouraged by the reconstruction loss formulated on x from q(z|x). More importantly, it seems that it would create a conflict with the intended disentangling objective between z and w. The validity of this design should be better clarified theoretically/analytically, and ablated experimentally.

The choice to have the adversarial classifier to work on reconstructed x instead of z also is not well justified. The authors explained that this will “reduce the variance introduced by sampling z from q(z), but the reconstruction of x would also require that sampling before y is applied. Furthermore, the authors stated that the classifier will be optimized separately from the rest of the model: it is not clear if the loss of this classifier actually goes back to optimize the rest of the model (e.g., q(z|x))

Overall, detailed ablation are needed for these key elements of the methods. Especially since some of these seem to be the key differentiators from existing works, e.g., the use of classifier on reconstructed x instead of z seems to be the main difference from CSVAE

Experimental evaluation can be improved in several aspects:
-	Why is disentangling score (I(z;w)) not presented for Table 1
-	Why is standard deviation only reported for some metrics but not others? Are the standard deviation not from multiple runs with random seeds?
-	Experimental results are the two natural image datasets are difficult to interpret. Figure 3 in colored MNIST is very difficult to interpret, to assess what are supposed to be condition-invariant z and what are condition-specific w. Similarly for CelebA (as shown in Fig 3), it is not clear why the domain-invariant representation should add an eyeglass to all images (since the presence of an eyeglass is specific to the label 0).

### Questions
-	Better clarify the prior coupling for w and z which seem to create conflict for the intended independence between these two latent representations & the main goal of the paper.

-	The independence of z and y depends quite significantly on adversarial learning, and hence the classifier. So, the capacity and performance of the classifier is possible to limit it. The choice of having this classifier on reconstructed x instead of z needs to be better elaborated as well as experimentally ablated.

-	Add detailed ablation to show which specific added/modified component in the proposed method leads to the most benefit would be good to have.

-	Better explain the results (in terms of what are expected condition-invariant vs. -specific representations) for the two natural image datasets.

- The method is explained and experiments are conducted considering single condition (label) with multiple classes (the authors do mention multi-condition in section 2 but based on description multi-condition seem to be multiple classes within a label). Does this method generalize easily to more than one label? E.g. maybe by using multiple w for multiple y (labels), or a single w but y as vector? Are there aspects we need to be careful about, leading to it being out of scope and a different analysis/paper altogether?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a dual-latent-code VAE framework for learning disentangled representations using a max–min objective. The authors first describe the model design and derive the theoretical ELBO corresponding to the disentanglement objective. Experiments are conducted on various datasets and compared against baseline methods.

### Strengths
(1) The research direction of this paper is promising. Formulating explicit theoretical formulations and constraints for different parts of the latent space is a reasonable approach to promote learning disentangled representations. 

(2) Overall, the paper is well-organized and easy to follow.

### Weaknesses
(W1) The descriptions of the proposed methods are not clear enough and may contain fatal errors. For example, first, the proof of B.1 for the key Proposition 2.1 does not look correct from (17) to (18). In (17), it appears that the -logp(z,w| x, y) is broken down into -logp(w|x, y) - logp(z|w, x, y), so this −logp(w|x, y) cancels the later +logp(w|x, y) in (17). However, why is $E_{q(w | x, y)}$ dropped when there remains a term logp(z|w, x, y) that still includes w inside the expectation? It would only make sense if z and w are independent given x, y, which clearly contradicts the proposed model as the authors claim in lines 88 and 307 that z and w are not independent given x, y.

Second, while the authors claim that the proposed method aims to reduce reliance on restrictive priors (line 50), why are there still priors needed in Eqns (3) and (4), even though they are class-wise priors? Is there any miscommunication here? Besides, the reasoning from lines 120-123 is not clear. A more detailed analysis and proof are necessary to explain why Eqn (4) is key to constraining z to depend only on x.

(W2) The evaluation of the disentangled representations is weak. First, the authors only select one disentanglement metric, mutual information estimated by MINE. However, MINE estimation is well-known to be unstable [1]. Other popular disentanglement metrics, like MIG and the Factor-VAE one, should also be included for comprehensive evaluations of the disentangled representations. Additionally, the MINE metric is only reported in Table 2, and Table 1 has no disentanglement metrics reported.

Second, the qualitative results are not convincing. Popular traversal plots of the learned representation are missing in this paper. Also, the CelebA results in Figure 4 look confusing. No evidence clearly supports that z learned the label-invariant representations. Semantically, should z learn “no-glass” features instead of “semi-glass” features here? The authors are encouraged to conduct similar experiments on simple datasets like Color-MNIST, and to clearly demonstrate that using z can reconstruct the same digit types with no effects of colors.

[1] Song, Jiaming, and Stefano Ermon. "Understanding the limitations of variational mutual information estimators." ICLR 2020.

### Questions
(Q1) The authors are encouaged to presond to my concerns listed in the Weakness sections. Additioanl explainations and experiment results would be appreciated.

(Q2) Please provide a step-by-step proof of Eqn (4).

### Soundness
2

### Presentation
2

### Contribution
2
