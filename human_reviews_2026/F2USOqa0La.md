# What We Don't C: Manifold Disentanglement for Structured Discovery

- Decision: Reject
- Scores: 6, 4, 2

## Abstract
Accessing information in learned representations is critical for annotation, discovery, and data filtering in disciplines where high-dimensional datasets are common. We introduce What We Don't C, a novel approach based on latent flow matching that disentangles latent subspaces by explicitly removing information included in conditional guidance resulting in meaningful residual representations. This allows factors of variation which have not already been captured in conditioning to become more readily available. We show how guidance in the flow path necessarily represses the information from the guiding, conditioning variables. Our results highlight this approach as a simple yet powerful mechanism for analyzing, controlling, and repurposing latent representations, providing a pathway toward using generative models to explore what we don't capture, consider, or catalog.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces What We Don’t C (WWDC), an approach aimed at uncovering residual factors of variation in latent spaces by intentionally removing information associated with known conditioning signals (e.g., class labels). The method starts from a pretrained VAE, then uses latent flow matching with classifier-free guidance to subtract the influence of a chosen conditioning signal. By reversing that guided flow, the model produces a latent representation where that signal is minimized, making it easier to see what other structure was hidden underneath.

### Strengths
The paper has a clear and compelling motivation: instead of continuing to reinforce information we already understand in a dataset, it focuses on uncovering what remains after known factors are removed. This conceptual reframing is refreshing and feels genuinely useful, especially for exploratory scientific analysis. The authors present the idea in an intuitive way, and the progression of experiments (from synthetic data to real astrophysics imagery) helps build trust in the approach. The qualitative results are effective in showing how the method reveals subtle structure in the data that wasn’t obvious before. Additionally, the fact that WWDC operates on top of pretrained models makes it practical and easy to adopt in real workflows, instead of requiring heavy retraining or specialized architectures.

### Weaknesses
The main limitation is that the evaluation remains largely qualitative, making it difficult to assess how well the method performs relative to established baselines in representation learning or disentanglement research. The paper would benefit from more systematic quantitative comparisons or metrics to support its claims. Some of the theoretical explanations around how information is preserved or removed during the latent flow process are also hard to follow and could use clearer intuition rather than relying primarily on equations. Finally, while the galaxy experiment is visually compelling, the paper does not fully explore the robustness or generality of the method across other complex real-world domains, which leaves questions about how broadly applicable the approach truly is.

### Questions
1. How sensitive is WWDC to the choice of pretrained VAE or latent dimensionality? The method is demonstrated on a particular architecture setup, but it's unclear whether the residual patterns remain stable if the base model or latent size changes. Some ablation or robustness analysis would help clarify whether the effect is consistent or dependent on specific model configurations.

2. Can you quantitative evaluation for the “emergent residual factors” revealed by WWDC? While the visualizations are compelling, it would strengthen the work to include metrics showing how well the extracted residual structure correlates with meaningful latent attributes (e.g., cluster separability, predictive power on downstream labels, or mutual information comparisons).

### Soundness
3

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
2

### Summary
The paper proposes a new method for disentanglement of the latent spaces of supervised (e.g., with class labels) generative models, using flow matching.

### Strengths
- addresses an important problem in an interesting way (including allowing further disentanglement of pretrained models)
- reasonable breadth of experiments, from simple controlled to complex real datasets, including intuitive results in figures 6 and 7
- intuitive results, especially in figures 6 and 7

### Weaknesses
- no reproducibility statement or opensource code, which is especially important for less theoretical contributions like this
- no (argument for the lack of) clear contextualization or comparison against existing disentanglement approaches
- hard-to-follow theory presentation in sections 2 and 3; maybe I just lack the background, but I guess I'm not the only reader who would benefit from gentler, more precise guidance through it
- unpolished writing

### Questions
My main questions, following from the weaknesses above, are:
1. can the authors can precisely explain and formalize the notion of disentanglement being used?
2. having done that, can the authors motivate this notion? Is it just "yet another" disentanglement notion, or is it somehow more fundamentally a better notion that others in the literature?
3. and can the authors relate this to other formalizations of disentanglement, including experimental comparisons where appropriate?

Less important, but still helpful for me to better understand the work:
1. can the authors rephrase the sentence starting at L085? I don't really understand it or even how to parse it.
2. L212: what does "ideal" mean here? Is it not ideal just in practice, or also at a more fundamental level?
3. L234: how do Markov chains come into play here? Is this related to eq. (2)?
4. L238: what does "information is not a sufficient criteria to enable access to that information" mean? 
5. L276: what expected structure "is not obvious" here? Is it obvious in the right panel? I don't understand what I'm supposed to be seeing here.
6. L278: (related to my main questions about disentanglement above,) why is it desirable that "the class information has been entirely remove" here?
7. L317: what's "minimal weighting" mean here? Is this some hyperparameter selection method?
8. L347: what's the "clear pattern across the space" here? I'm not sure I see it.

Typos:
- L029: the YM reference here is formatted incorrectly (also in the bibliography), mixing up last name and fist/middle initials.
- L080: missing space "models(Fuest"
- L087: "to to"
- L138 and beyond: math formatting could be improved, e.g.,
    - $D_{\mathrm{KL}}$
    - $\mathcal{L}_{\mathrm{CFM}}$
    - $p_{\mathrm{cfg}}$
    - ${u_t}^{\mathrm{CFG}}$
- L174: "trained used"
- L207: VAEs -> VAE's
- L305: "features.Figure"
- L313: "a random RGB values"
- L238: "a sufficient criteria" ("criteria" is plural; the singular is "criterion")
- L444: "Uunbarred"
- L466: "we propose to be used to"
- L467: "surveys(e.g."
- L478: "representations of the data that emerge secondary features"
- L485: "in assisting researchers explore what information"

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
In this paper, the author proposes applying flow matching to disentangle the latent space. The main method is designed for variational autoencoders (VAEs), whose latent space approximates a normal distribution, allowing the use of Gaussian conditional optimal transport probability paths for interpolation.

### Strengths
1. The idea of combining flow matching with variational autoencoders (VAEs) is interesting and has potential to inspire further exploration in disentangled representation learning.

2. The paper is well-structured.

### Weaknesses
1.  Since the method is built on top of VAEs and relies on the approximately Gaussian distribution of their latent space, its use is restricted to this specific class of generative models.

2. The paper lacks sufficient supporting evidence. In the experimental section, the author evaluates the method on synthetic 2D Gaussian data, CMNIST, and a real-world dataset. All three datasets are relatively simple, and other existing disentanglement methods are known to perform well on them—particularly on 2D Gaussian and MNIST, which are common benchmarks. More importantly, the paper does not include comparisons between the proposed approach and other established disentanglement methods, which makes it difficult to assess the effectiveness and advantages of the method.

3. If the key claim of the paper lies in achieving meaningful disentanglement, there appears to be no theoretical guarantee or clear intuition explaining why such disentanglement can be discovered using the proposed approach.

### Questions
1. What is the most significant difference between the proposed method and other state-of-the-art disentanglement approaches that makes it stand out? Is there any theoretical justification or empirical evidence supporting this distinction?

2. Can this method be extended or adapted to other types of generative models beyond VAEs?

### Soundness
2

### Presentation
3

### Contribution
2
