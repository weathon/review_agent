# Unpicking Data at the Seams: VAEs, Disentanglement and Independent Components

- Decision: Reject
- Scores: 3, 5, 3

## Abstract
Disentanglement, or identifying statistically independent salient factors of the data, is of interest in many aspects of machine learning and statistics, having potential to improve generation of synthetic data with controlled properties, robust classification of features, parsimonious encoding, and greater understanding of the generative process behind the data. Disentanglement arises in various generative paradigms, including Variational Autoencoders (VAEs), GANs and diffusion models, and particular progress has recently been made in understanding the former. That line of research shows that the choice of diagonal posterior covariance matrices in a VAE promotes mutual orthogonality between columns of the decoder's Jacobian. We continue this thread to show how such *linear* independence translates to *statistical* independence, completing the chain in understanding how the VAE objective leads to the identification of independent components of the data, i.e. disentanglement.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper connects orthogonality of the decoder Jacobian in VAEs to disentangling statistically independent latent factors.

### Strengths
The paper is
- generally well-written,
- the figures are high quality,
- the structure is logical and easy to follow.

### Weaknesses
Despite the neat structure and the obvious effort the authors put into this work, **I honestly struggled to discern what the added value of the contribution of the paper is, compared to the literature** (while knowing the field and having worked on related projects). I will try to make the points that confused me clear below. Please let me know if my understanding is incorrect.

## Major issues
- The added value of the contribution is unclear to me. A potential statement for this could be the _wrong_ statement, made in L468: _"Reizinger et al. (2022) relate the VAE objective to independent causal mechanisms (Gresele et al., 2021) which consider non-statistically independent sources that contribute to a mixing function by orthogonal columns of the Jacobian. This clearly relates to the orthogonal Jacobian bias of VAEs, but differs to our approach that identifies statistically independent components/sources"_. Namely, **Reizinger et al. (2022) assume a standard VAE with statistically independent sources.** It seems to me that the authors themselves differentiate their work from Reizinger et al. (2022) only by the presumed (but not true) dependent/independent sources divide. As this is not the case (unless my understanding is incorrect), I cannot see the added value of the paper. This holds for the theorems, and also for the elucidation of the role of $\beta,$ as Reizinger et al. (2022) also relate it to the decoder variance in their Appx. A.3.
- The paper is a bit too heavy on notation in the main text. While I understand that the proofs require precision, communicating (the intuition behind) the results do not. Please rework the main text to include only the necessary notation. Also, please always explain what your notation means (e.g., the caption of Figure 1 is not self-contained)

## Minor points
- L167: expand of what you are trying to do with the two losses
- Eq (9): define $s_i$
- L248: why is the image manifold parallel to the $U$-basis?
- Eq (12): it should include the log absolute determinant of teh Jacobian, and not $W$ as this is about the nonlinear case, right?

### Questions
- What do the two types of bold D's in Eq. (6) denote?
- L156-160: isn't statistical independence of latent factors assumed in VAEs?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper provides a theoretical understanding of disentanglement within Variational Autoencoders when the columns of the jacobian of the decoder are orthogonal (theorem 2). 

Starting from this hypothesis, namely the linear independence of the columns of the decoder Jacobian, the author shows that it is possible to link statistically independent components of the output distribution to independent latent factors of variation.

### Strengths
The authors have clearly defined the problem and the goal of the paper;

The analysis extends existing work, offering theoretical proof of why an architectural constraint such as diagonal posterior covariance induces an effective disentanglement in the VAE latent space.

### Weaknesses
The paper lacks an experimental part. 
Even if the contribution is theoretical, a small experiment on a synthetic dataset corroborates the claims and strengthens the contribution.

Splitting the deductive reasoning between ll 210 to 259 in paragraphs highlighting the key logical steps can also be helpful for the reader.

Additional figures or clarification of the only existing one (Fig 1) can help the reader understand the sequence of logical steps more easily.

### Questions
How sensitive is the disentanglement effect to minor violations of the assumptions (orthogonality of the decoder Jacobian)?

in 059: do you mean encoder or decoder Jacobian?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper aims to make progress on understanding why VAEs are able to learn disentangled representations. To this end, the authors connect orthogonality of the decoder Jacobian to disentanglement by showing that the VAE learns statistically independent sub-manifolds in the observed space. Further, the paper provides additional insights into the VAE objective such as on the role of beta in a beta-VAE.

### Strengths
* The paper addresses an important problem. Namely, developing a principled understanding of how and why disentanglement is possible in deep generative models.

* The paper is well written and the presentation is well structured.

* The theoretical results and insights are presented in an intuitive and accessible way, with nice visual intuition from the figures.

### Weaknesses
My main issue with this paper is that I do not think the contributions presented in this paper offer sufficient novelty relative to prior work to merit acceptance, and, moreover, I do not think the authors adequately compare their contribution to prior work. I discuss these points in detail below with specific examples.

**Identifiability of Latent Factors in VAEs**

The authors build on several prior works which try to explain why VAEs are able to identify the ground-truth latent factors by showing that the VAE objective promotes the decoder to have an orthogonal Jacobian. The authors perform a similar analysis and present a result claiming to be an identifiability result for VAEs.

Understanding the identifiability of VAEs, however, has been analyzed in detail in prior works. Specifically, the work of [1], rigorously showed that the VAE objective with vanishing decoder variance is equivalent to maximum likelihood under an independent Gaussian prior plus a regularization term enforcing that the Jacobian has orthogonal columns. The identifiability of such models with independent latents and orthogonal Jacobians was studied in depth by [2], who proved a certain form of identifiability for such models and showed the theoretical challenges in recovering a complete identifiability result.

The authors do not mention the work of [2] and wrt [1], they state that their analysis differs because [1] assumes statistically dependent factors such that there result is novel relative to this work. This statement is wrong. The work in [1] does not assume statistically dependent factors. To this end, I think the works of [1] and [2] conduct a more rigorous and comprehensive analysis of the VAE objective and its identifiability than the current work, such that I do not feel this works adds sufficient novelty.

Additionally, the analysis of the VAE objective and the identifiability analysis conducted in this work are not as rigorous as these prior works in [1, 2], and are closer to the results in [3, 4]. From this standpoint, however, I am also not sure what the novelty is of the authors SVD based identifiability argument, as similar ideas were presented in [3] as I understand. Perhaps, the authors can clarify this as well.

**Understanding the Role of Beta in Beta-VAEs**

Another stated contribution of this work is the authors' analysis of the role of beta in a beta-VAE (Section 3.3). The authors state that beta can be interpreted "as scaling the variance of the likelihood distribution,". From what I can tell, this result also seems very similar to the result in [1] presented in Appendix A.3 on the role of beta. I am curious if the authors can comment on the novelty of their result relative to this prior work.


**Theorem Statements**

As an additional point, I did not see proofs for Theorems 1 and 3, and am thus curious if I am missing something or if there are proofs for these results.

**Bibliography**

1. Embrace the Gap: VAEs Perform Independent Mechanism Analysis
 (https://arxiv.org/abs/2206.02416)

2. Function Classes for Identifiable Nonlinear Independent Component Analysis
(https://arxiv.org/abs/2208.06406)

3. Variational Autoencoders Pursue PCA Directions (by Accident)
(https://arxiv.org/abs/1812.06775)

4. On Implicit Regularization in β-VAEs (https://arxiv.org/abs/2002.00041)

### Questions
**1.** Can the authors comment on the novelty of their results relative to the prior results discussed above?

**2.** Where are the proofs for Theorems 1 and 3?

### Soundness
2

### Presentation
3

### Contribution
1
