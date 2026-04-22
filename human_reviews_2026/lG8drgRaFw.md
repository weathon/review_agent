# Unpicking Data at the Seams: Understanding Disentanglement in VAEs

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 6, 2, 2

## Abstract
We give a precise, general account of *disentanglement* for smooth generative models.
For a decoder $g:\\mathcal{Z}\\to\\mathcal{X}$ and factorised prior $p(z=\\prod_i p_i(z_i)$, we 
(i) define disentanglement as *factorisation of the pushforward density* $p_\\mu= g_\\#p$ into one–dimensional ``seam'' factors (Def. D1); 
(ii) prove a canonical factorisation of $p_\\mu$; and 
(iii) show that disentanglement is *equivalent* to two decoder conditions (C1-C2). Furthermore, under these conditions, the seam factors are *identifiable* up to permutation and sign.
These results hold for general smooth pushforwards and are independent of VAEs.
Specializing to Gaussian VAEs, we use an *exact* identity to show that diagonal posteriors (and $\\beta$) promote C1--C2 in expectation, thereby explaining when and why VAEs exhibit disentanglement and how $\\beta$ modulates it.
Experiments illustrate this mechanism on Gaussian data, dSprites, and CelebA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper provides a theoretical framework for understanding disentanglement in Variational Autoencoders (VAEs), explaining why and how VAEs learn to separate independent generative factors of data without explicit supervision.
The authors define disentanglement as factorizing the density over a data manifold into independent one-dimensional "seam" factors, where each factor is the push-forward of density over an axis-aligned latent path.
They prove that 2 conditions on the decoder is equivalent to disentanglement.
They also prove the identifiability of the LVAE and Gaussian VAE models.
Experiments on dSprites dataset confirm that diagonal posterior covariances promote diagonalized derivative terms and that diagonality correlates with disentanglement metrics. The paper shows diagonal covariance VAEs achieve better disentanglement scores (MIG and proposed AAS metric) compared to full covariance VAEs, with individual latent coordinates correlating with distinct ground truth factors.

### Strengths
All the claims are proved theoretically providing solid bases for understanding this disentanglement phenomenon.

Correction of a previous paper is useful.

### Weaknesses
Very low experimental support is provided. 
It would have been interesting to have an idea of the required dimension of the latent space to reach the whole disentanglement. Can it be related to the noisy ICA decomposition and the number of independent factors?
What is the effect of on reconstruction when the C1-C2 conditions are satisfied compared to other VAE-like models?
What is the effect on generation as well?
Is this property useful when trying to use VAE with a small dataset?

### Questions
See weaknesses.

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
3

### Summary
This paper provides a theoretical explanation for why Variational Autoencoders (VAEs) often exhibit latent disentanglement—the phenomenon where manipulating a single latent dimension changes only one semantic factor in the generated data (e.g., object position, facial expression). The authors argue that disentanglement in VAEs arises largely due to the common practice of using diagonal Gaussian posterior covariances, not due to explicit disentangling losses. They demonstrate an exact mathematical link between optimal Gaussian posteriors and decoder Jacobians, showing that diagonal posterior assumptions “lock” the decoder into axis-aligned directions in latent space. As a result, the learned data density factorizes into independent 1-D seams corresponding to latent dimensions—producing disentanglement “for free.”

### Strengths
1. Novelty. The paper formalizes this mechanism and proves conditions under which latent factors are identifiable fro VAE, even with a symmetric prior, addressing the long-standing “unidentifiability” concern of nonlinear latent models. It also clarifies how \beta-VAEs enhance this effect by modulating posterior variance and reducing posterior collapse.

2. Theoretical value: Theoretical proof connecting diagonal posterior covariances to axis-aligned latent factorization through VAE Jacobians. It provides explanation of how β-VAE (β > 1) improves disentanglement by controlling posterior variance.

3. Empirical demonstration shows that true generative factors are identifiable in VAEs under stated assumptions.

### Weaknesses
Limited Empirical Scope

1. The experiments primarily use: Synthetic linear-Gaussian settings, dSprites dataset (simple factors, low resolution), Small-scale VAEs (dim=10)

2. While appropriate for theory illustration, they do not adequately test: complex natural image datasets (such as CelebA, Imagenet1K)
Empirical evidence also lacks real-world factor disentanglement settings (e.g., pose, lighting, angle)

3. Claims about generality such as Diffusion or GAN architectures discussed as motivation are not yet experimentally supported, although this is minor.

### Questions
Please see weakness for questions.

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
2

### Summary
This paper provides a precise theoretical explanation for why Variational Autoencoders (VAEs) with diagonal posterior covariances tend to learn disentangled representations. Disentanglement is the phenomenon where varying a single latent coordinate causes a single, semantically meaningful change in the generated sample. 

The authors build their argument on an exact relationship between the optimal Gaussian posterior and the decoder's derivatives. They show that forcing the posterior covariance to be diagonal imposes constraints on the decoder's Jacobian ($J_z$) and Hessian ($H_z$). These constraints, in turn, imply two key properties must hold (in expectation):

1. **(C1)** Right singular vectors Vz of the decoder Jacobian Jz are standard basis vectors for all z ∈ Z, i.e. after relabeling/sign flips of the latent axes, we have Vz = I;
2. **(C2)** The matrix of partial derivatives of singular values ( ∂si∂zj )i,j is diagonal, i.e. ∂si∂zj = 0 for all i = j.

### Strengths
1. Formal Definition of Disentanglement: The paper proposes a precise and geometrically intuitive definition of disentanglement (D1). By defining it as the factorization of the manifold density into independent 1-D "seams", it moves the concept from a vague empirical observation to a testable mathematical property.
2. Precise Theoretical Mechanism: It provides a full theoretical explanation  for a long-observed phenomenon. Instead of relying on approximations , it uses an exact relationship from the Price/Bonnet Theorem to build a rigorous, step-by-step argument connecting diagonal posteriors to the geometric constraints (C1, C2) that provably guarantee disentanglement.

### Weaknesses
1. The proof is limited in Linear situation. These conclusions may not be applied in complex data. The errors caused by non-linearity are not discussed.
2. The conclusions in the paper are vanilla that some papers have explained. For instance, beta controls the regularization strength for index-code mutual information, total correlation, and dimension-wise KL [1]. This can explain the importance of beta and posing low model variance.
3. The impact of this research is unclear. This work has no experiment to prove the benefits derived from these theories. Annealing beta is a common trick in VAEs and disentanglement learning. 


[1] Chen, Ricky TQ, et al. "Isolating sources of disentanglement in variational autoencoders." Advances in neural information processing systems 31 (2018).

### Questions
Can the proposed theory guide the design of novel disentanglement methods? 
What is the novelty of the proposed theory and the comparison to beta-TCVAE?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper aims to develop a theoretical understanding of VAEs and their capacity to perform disentanglement. To do this, it builds upon prior theoretical results to argue that VAEs induce a pushforward distribution of the prior under a decoder which factorizes along the pushforward of each latent component. The authors then argue theoretically that this property yields identifiability. An experimental study is conducted which aims to confirm these theoretical claims empirically.

### Strengths
* I found the theoretical sections to be very well written and logically structured making the paper straightforward to digest and pleasant to read.


* Lemmas 4.1 and 5.1 as well as the theoretical results in Section 6 are potentially interesting for the identifiability and VAE communities.


* The issue regarding the proof in Reizinger et. al, 2022 is an interesting and important observation.


* All figures in the paper are very nicely done and informative.

### Weaknesses
The paper aims to make a contribution to both VAE theory and identifiability theory by analyzing the structure of a VAE decoder's derivative matrices. These are well studied research areas with a long line of theoretical results on understanding VAEs [1, 2, 3, 4, 5], particularly regarding their Jacobian and Hessian structure, as well as on identifiability (see [6] for a review), particularly for functions with orthogonal Jacobians [7,8,9] and diagonal Hessians [10, 11].


This is by no means to say these areas are saturated. However, I believe it is very important for any work in this space to position itself very clearly relative to prior works, in order to understand precisely what is novel. *In my opinion, this paper failed to adequately do this making it challenging to assess its contribution*. I discuss this in detail below.

**Contribution of this work**

My understanding of this paper's main contribution is that it leverages theoretical and empirical results in Kumar et. al, 2020, to directly assume that Property 1 (line 184) holds for VAEs. This property is then used to ultimately prove Lemma 5.1 and to then show that this result implies identifiability (Section 6).

Thus, as I understand, the main aspects of this work that should be understood as novel are Lemma 5.1/Theorem 5.2 for VAEs after assuming Property 1 holds, as well as the identifiability analysis based on these results in Section 6.

**Hessian Diagonality**

Firstly, to prove Lemma 5.1, the authors rely on the assumption of Property 1, which I understand to mean that the decoder's Jacobian is orthogonal and its Hessian diagonal? If this is the assumption, then it is important to note that there have been multiple works in the identifiability community which show identifiability for generators with a diagonal Hessian [10, 11]. Thus, it is unclear to me, at the moment, the extent to which this current identifiability result is novel relative to prior results given this Hessian assumption.

**Role of Beta**

Another stated contribution of the paper is understanding the role of the beta hyperparameter in VAEs for disentanglement as it relates to decoder variance (lines 48-49). As I understand, similar results were shown in Kumar et. al, 2020 [2] as well as Reizinger et. al, 2022 [5] (Appendix A.3). Thus, it is also not clear to me the extent to which this result is novel relative to prior works.

**Experiments**

An experimental study is conducted in Section 7 regarding the relationship between a VAEs derivative structure and disentanglement. My understanding, however, is that very similar experiments have been run in prior works such as [1, 2, 5]. I thus believe it is important for the authors to discuss better how their experiments differ from prior works.

**Disentanglement Definition**

In Section 3, disentanglement is defined to mean that the pushforward distribution of the prior under the decoder factorizes along component-wise pushforward distributions. This disentanglement definition seems tailored to the authors results and does not reflect definitions pursued in prior theoretical works [6] which ground disentanglement more rigorously in terms of identifiability. Thus, I believe more motivation for this definition and how it relates to prior works is needed.

**General framing of contribution**

In general, I believe the paper would benefit from presenting its contribution more precisely in the title, abstract, and introduction. Instead of using the broad phrasing/messaging of understanding disentanglement in VAEs, I believe clearly stating, from the outset, what the papers contribution is relative to prior work on VAEs and identifiability, and how it builds upon these works, would improve the paper.

**Conclusion**

For the reasons discussed above, I found it difficult to assess the contribution of this work relative to prior works. Thus, for the time being, I do not recommend acceptance.


**References**

[1] Variational Autoencoders Pursue PCA Directions (by Accident) (https://arxiv.org/abs/1812.06775)

[2] On Implicit Regularization in β-VAEs (https://arxiv.org/abs/2002.00041)

[3] Diagnosing and Enhancing VAE Models (https://arxiv.org/abs/1903.05789)

[4] Demystifying Inductive Biases for (Beta-)VAE Based Architectures (https://proceedings.mlr.press/v139/zietlow21a.html)

[5] Embrace the Gap: VAEs Perform Independent Mechanism Analysis (https://arxiv.org/abs/2206.02416)

[6] Nonlinear Independent Component Analysis for Principled Disentanglement in Unsupervised Deep Learning (https://arxiv.org/abs/2303.16535)

[7] Independent mechanism analysis, a new concept? (https://arxiv.org/abs/2106.05200)

[8] Function Classes for Identifiable Nonlinear Independent Component Analysis (https://arxiv.org/abs/2208.06406)

[9] Robustness of Nonlinear Representation Learning (https://arxiv.org/abs/2503.15355)

[10] Additive Decoders for Latent Variables Identification and Cartesian-Product Extrapolation (https://arxiv.org/abs/2307.02598)

[11] Interaction Asymmetry: A General Principle for Learning Composable Abstractions (https://arxiv.org/abs/2411.07784)

### Questions
* What do the authors view as their main contribution relative to prior works on VAEs and identifiability? Can the authors comment further on the relationship between their results and the prior identifiability and VAE results discussed above.


* Why are disentanglement and identifiability defined separately in the authors formalism. How do the authors justify this new definition of disentanglement?


* What novelty do the authors believe their experiments offer relative to prior works?

### Soundness
3

### Presentation
2

### Contribution
2
