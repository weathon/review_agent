# Skip Connections and Generalization: A PAC-Bayesian Perspective

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Skip connections are a hallmark of modern deep neural networks, yet their effect on generalization remains elusive. We present a PAC-Bayesian analysis showing that skip connections reduce cross-layer correlations in the weight distribution, which directly tightens generalization bounds. Our approach models weight matrices with matrix normal distributions, capturing row- and column-wise dependencies, and reveals that skip connections effectively suppress correlation terms dominating the KL divergence.  This view aligns with the Laplace approximation perspective, where skip connections encourage flatter minima and more dispersed posteriors. Empirical results on multilayer perceptrons and convolutional networks with varying skip configurations support our theory: reductions in cross-layer correlation consistently coincide with improved test accuracy.  Overall, our work provides a new theoretical and empirical explanation of why skip connections enhance generalization, highlighting correlation reduction as a key mechanism behind their success.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors relate inter-layer correlations of posterior weight matrices to the KL term in the PAC-Bayes generalization bound by modeling the posterior weight distribution as a Kronecker-factored matrix normal distribution. These correlations are related to the presence or absence of skip connections in the neural network, as evidenced by some example plots. The authors then validate their approach by showing that it more closely tracks the empirical generalization gap than other complexity measures when the presence of skip connections between layers is varied.

### Strengths
- As far as I am aware, this is the first attempt to relate inter-layer weight correlations to PAC-Bayes generalization bounds. (However, I am not very familiar with the PAC-Bayes literature, and so not confident in this statement.)
- The theory seems sound, though I did not check the derivations carefully.
- For MLPs, experiment shows that the authors' proposed complexity measure tracks the generalization gap better than previous approaches.

### Weaknesses
- The paper is framed as showing a relationship between skip connections and generalization, via posterior weight correlations. However, as far as I can tell, there is no theoretical justification and fairly little empirical justification for the claim that the presence of skip connections relates to weight correlations in a well-understood way.
- The experimental results for CNNs do not favor the approach proposed in this paper.
- Presentation could be improved; in particular Section 4 feels somewhat disorganized and hard to follow.
- The empirics section (5) lacks important details (see Questions).

### Questions
- 235: "A full covariance structure for the posterior captures all information contained in the trained neural network". This is not literally true, right? The posterior is not generally Gaussian, so there may be higher-moment information not captured by just the covariance; you are choosing to model it as Gaussian.
	- I'm not sure how justified it is to model the posterior as Gaussian; neural networks are singular so Bernstein-von Mises does not apply[1]. Is Gaussianity a standard assumption in the PAC-Bayes literature?
- It should be explicitly stated at the start of 4 what $U$, $V$, and $V_{ll}$ are. I'm guessing that they are the parameters coming from fitting a Kronecker-factored matrix normal to the posterior distribution of the stacked weights $W$?
	- In particular, it should be emphasized that $V$ encodes the correlation structure of the posterior of $W$.
- 278:  I don't understand $\text{vol}(W_l)=V_{l,l}\otimes U$; the LHS should be a scalar but the RHS looks like a matrix.
- 284: What is $\mathbb{E}_S[H]$? Expected Hessian of $p(\omega\mid S)$?
- What is the precise definition of the "PBGC" number used in Section 5's experiments? How is it computed? 
- I don't understand how you go from inter-layer weight correlations to a PAC-Bayes bound, since 4.5 and 4.6 are only show monotonicity instead of anything quantitative. Do you fit $V_Q$ and compute its log det? How?
- What are the relevance of Definitions 4.2 and 4.3? I don't see general weight correlation or weight volume used later in the section. The matrix $V_Q$ used later is not the same as either of these, right?
	- In particular, my understanding is that $V_Q$ captures *posterior* covariance between *columns* of $W$, while general weight correlation is a measure of *row* cosine similarity *given a fixed $W$*
- Overall, Section 4 is kind of hard to follow. I recommend:
	- More handholding: explicitly point out that the point of (13) is to provide an expression for the 2nd term in (5), and that (16) is the exp of the part of (13) that depends on $\rho_{l-1,l}$. Thus Prop 4.5 and 4.6 show that one term of the PAC-Bayes bound (3) is decreasing in $\rho_{l-1,l}$.
	- If they are not directly necessary to the main argument, connections to weight correlation and weight volume should be moved to a separate, later, section, possibly in the appendix.
- Is there a known theoretical relationship between skip connections and inter-layer weight correlations?
	- Figs 1 and 2 hint at some empirical relationship between skip connections and correlations, but it's hard to figure out precisely what's going on by just eyeballing a few plots. I would recommend some more careful empirical studies here.
	- E.g., if you use adjacent-layer skips, how close is the resulting correlation matrix to the 1-banded structure assumed in Prop 4.5?
- How were the posterior correlations in Fig 1 estimated? Did you use SGLD? [2]
- Tables 1 and 2 would be much more readable as line plots with generalization gap on the x-axis and complexity measure on the y axis.

[1] Watanabe "Algebraic Geometry and Statistical Learning Theory." 2011.

[2] Welling et al. "Bayesian Learning via Stochastic Gradient Langevin Dynamics". 2011.

Nit:
- 116: "2rd" -> "2nd"
- 162: "Neural Networks" -> "Neural Network" (twice)
- 172: "starts" -> "start"
- 238: "is identical" -> "is unchanged by training" ?
- 245: $W$ -> $M$ ?
- 254 "between layer" -> "between layers"
- 256 "Given weight matrices $W_l$, $W_s$ at the $l$-th and $s$-th layers"
- 265: Is the second sentence in this paragraph just saying the same thing as the first?
- 269: "relates" -> "related"
- 275: "$\ell$" -> "$l$"
- 295: "Let the weights of the neural network be [...]"
- 299: "Thus" -> "Then"

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Residual networks with skip connections have seen much success in practice, but as yet a complete theoretical account for that success is lacking. In this paper, the authors aim to bridge that gap by using the framework of PAC-Bayes. I think the paper is somewhat interesting, and the knowledge gap it aims to bridge is clearly important. However, as I elaborate below, I do not think the manuscript provides a convincing theoretical account of why networks with skip connections might generalize better.

### Strengths
As noted in my summary, the paper aims to study a very significant question, and I am not aware of papers that adopt quite this approach. I must temper this endorsement with the fact that I am not extensively familiar with the literature on PAC-Bayes.

### Weaknesses
There are two issues that stand out to me on first reading: 

- Given that the authors adopt the PAC-Bayes framework, most of the manuscript focuses on studying the KL divergence between the posterior distribution over the trained weights and the initial distribution of weights. All of these investigations are carried out under the assumption that the posterior over weights is nearly Gaussian, with correlations that follow a particular Kronecker-factorized structure. Nowhere in the paper do the authors attempt to test or rigorously justify these assumptions. 

- The closest the paper comes to presenting a justification for the assumptions above is in the authors' empirical studies of the correlation between the generalization gap and weight complexity measures. However, all of the presented rank correlation coefficients (Kendall's $\tau$) are in absolute terms quite small, being on the order of 0.01 at most. 

In combination, these two issues make the authors' rather broad claims in the abstract and introduction ring rather hollow in my ears. The theoretical results regarding simplified forms of the KL divergence also don't strike me as being particularly novel or of particularly broad interest. I am thus left fundamentally unconvinced.

### Questions
**Major questions and comments**

- Can you provide a more direct test of the assumptions regarding the distribution of weights? 

- The paper does not provide an adequate description of how approximate posterior sampling was performed. How do you compute the correlations? 

- The authors do not address previous studies on the role of inter-layer weight correlations, see for instance [Guth et al. (2024)](https://jmlr.org/papers/v25/23-1573.html) and references therein. 

**Minor questions and comments**

- Section 2.2 discusses studies on the relationship between the flatness of minima and generalization, but does not address works since 2019 that have questioned the strength of this link, see e.g. [Andriushchenko et al. (2023)](https://proceedings.mlr.press/v202/andriushchenko23a.html). 

- Lemma 4.1 is an obvious consequence of combining two well-known facts: the closed-form of the KL divergence between two Gaussian vectors, and the vectorized representation of the matrix Gaussian. At present, that is not clear from the presentation; this should be clarified.

- The appendices require editing, as they contain a number of broken links.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies how skip connections affect generalization through a PAC-Bayesian lens. The authors introduce General Weight Correlation (GWC) to capture cross-layer dependencies induced by skip connections, derive how these correlations enter the KL term of PAC-Bayes bounds (via matrix-normal posteriors and a Kronecker factorization), and prove that adjacent-layer correlations enlarge the KL term, while heterogeneous layer-specific correlations can help. They then construct a data-driven complexity measure (PBGC) and evaluate it across all skip-patterns of 5-layer MLPs (Fashion-MNIST) and CNNs (CIFAR-10), finding PBGC best matches empirical generalization trends for MLPs, with a subtler picture in CNNs (and batch-norm reversing some effects).

### Strengths
* Non-trivial theoretical contribution linking architecture to PAC-Bayes. The paper formalizes inter-layer dependencies via GWC, derives closed-form KL for matrix-normal posteriors, and proves monotonic relationships between adjacent-layer correlations and the KL term, thus offering principled insight into why ResNet-style skips may help generalization.
* The adjacent-connection and homogeneous-connection propositions make testable predictions about how skip patterns change the determinant term in the KL (hence the bound), clarifying when longer skips mitigate harmful adjacent correlations.
* Exhaustive skip-pattern studies in 5-layer MLPs show PBGC aligns best (Kendall’s $\tau$) with empirical generalization gaps; the CNN results reveal architecture-specific behavior and an interaction with batch norm worth further study.
* Generally well-written and well-organized

### Weaknesses
* The theory is about generalization complexity (KL term), not FLOPs/latency/params; the paper does not analyze runtime or memory of PBGC estimation beyond a Kronecker approximation, nor does it present wall-clock or scalability studies. This should be clarified and, if claimed, empirically substantiated.
* Dataset/scale limited. Results are on Fashion-MNIST and CIFAR-10; testing on larger datasets (e.g., TinyImageNet/ImageNet subset) would stress-test whether PBGC continues to track generalization under higher capacity/data complexity and richer skip motifs (e.g., bottlenecked ResNets).
* The isotropic prior and shared-variance posterior assumptions aid tractability, but the sensitivity to these choices (e.g., localized priors, non-Kronecker posteriors) is not explored; ablations here would increase robustness.

**Minor**

* “architechture” -> architecture in Sec.3
* “non-paramteric” -> non-parametric
* “Proof of Theorem ??” placeholder not resolved in appendix

### Questions
* How sensitive is PBGC to how the posterior is estimated (Laplace details, damping, data subsampling)? Can you report variance across seeds/runs and a runtime profile for PBGC vs. PFN/PSN?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a PAC-Bayesian framework to theoretically analyze how skip connections affect the generalization of deep neural networks. It introduces the concept of General Weight Correlation (GWC) to quantify inter-layer dependencies and derive how different correlation structures affect the KL divergence and generalization bounds. The paper presents both theoretical derivations and controlled experiments on MLPs and CNNs. The results show that heterogeneous or long skip connections reduce inter-layer correlation and tighten the PAC-Bayes bound, leading to better generalization.

### Strengths
1. The paper provides a conceptually meaningful bridge between the empirical design of skip connections and the theoretical understanding of their generalization by using a PAC-Bayesian framework

2. The paper proposes the concept of General Weight Correlation (GWC), which provides a concrete way to represent inter-layer dependencies and examine their impact on the PAC-Bayesian bound. The formulation is clear and mathematically consistent.

3. The controlled experiments on small MLP and CNN models are sufficient to verify the main theoretical trends, showing that heterogeneous skip connections tend to reduce inter-layer correlation and modestly improve generalization.

### Weaknesses
1. The empirical evaluation is limited to small-scale MLP and CNN models on simple datasets (Fashion-MNIST, CIFAR-10). While these experiments demonstrate the theoretical trend, they do not establish whether the proposed framework holds for more advanced architectures (e.g., ResNet-50, Transformers) where skip connections are most impactful. 

2. The theoretical analysis relies on idealized assumptions, such as modeling the posterior with a matrix normal distribution and isolating skip connections as the sole source of inter-layer dependency, which may not accurately capture the behavior of real networks. The authors could clarify the limitations of these assumptions and discuss how the framework might be generalized or empirically approximated in practice.

### Questions
1. Regarding the posterior modeling assumption, the analysis relies on a matrix normal posterior, would a mixture-based posterior (e.g., mixture of Gaussians) change the observed trends in the KL divergence or generalization bounds?

2. In what ways does modeling structural correlations via GWC provide insights beyond existing approaches like flatness-based or information-theoretic generalization analyses?

### Soundness
3

### Presentation
3

### Contribution
3
