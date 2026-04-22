# Sparsity and Superposition in Mixture of Experts

- Avg Score: 2.00
- Decision: Reject
- Scores: 2, 2, 2, 2, 2

## Abstract
Mixture of Experts (MoE) models have become central to scaling large language models, yet their mechanistic differences from dense networks remain poorly understood. Previous work has explored how dense models use $\textit{superposition}$ to represent more features than dimensions, and how superposition is a function of feature sparsity and feature importance. MoE models cannot be explained mechanistically through this same lens. We find that neither feature sparsity nor feature importance causes discontinuous phase changes, and that network sparsity (the ratio of active to total experts) better characterizes MoEs. We develop new metrics for measuring superposition across experts. Our findings demonstrate that models with more network sparsity exhibit greater $\textit{monosemanticity}$. We propose a new definition of expert specialization based on monosemantic feature representation rather than load balancing, showing that experts naturally organize around coherent feature combinations when initialized appropriately. These results suggest that network sparsity in MoEs may enable more interpretable models without sacrificing performance, challenging the common assumption that interpretability and capability are fundamentally at odds.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the superposition in toy MoE models. Most of its techniques are borrowed from a blog post by Elhage et al., which analyzes superposition in a toy dense model. Many toy setups, such as a hidden dimension of merely 1, are insufficient to support a full-length regular research paper.

The authors also observe that the phase change seen in dense models is absent in MoE models, and they study expert specialization and initialization. While the sections on specialization and initialization offer an interesting perspective, they still rely on a hidden dimension of 1. I appreciate the viewpoint here but remain unconvinced by the validity of all the results.

### Strengths
Section 5 presents an interesting perspective. If its findings could be validated in realistic MoE models, rather than the toy setups used, the paper would gain significant value.

### Weaknesses
1. The font in this paper’s template appears unusual and does not conform to the formal requirements of ICLR 2026.

2. This paper offers few novel techniques. Section 5 is interesting, but the rest of the paper merely adapts Elhage et al.’s method—originally applied to a dense toy model—to an MoE toy model, with formulations and definitions borrowed from Elhage et al. The evaluation metrics and analytical methods are also mostly identical.

3. It may be acceptable for Elhage et al. (whose work is merely a blog post) to use toy models with a hidden dimension of m=5. However, this paper uses m=6, or even fewer, m=1. Insights derived from such toy models are insufficient to serve as the core experiments of a regular paper.

4. In modern MoE architectures, SwiGLU is universally adopted as the expert structure, rather than the two-layer ReLU MLP used here. Activation functions and model architectures significantly influence model behavior and training dynamics, further rendering this paper overly "toy-like" in design.

5. On page 4, the conclusion that "The greater the number of experts, the less superposition in the model" is not informative. With more experts, the model is wider, and it naturally does not need to allocate features in a superposed manner. The related experiments are therefore uninformative to me.

6. In Line 220, the claim that the loss gap between the MoE and dense model in Figure 7 is "negligible" is incorrect. Every observed gap (on the order of 0.1) is significant.

7. The captions, labels, and discussions related to Figure 4 are confusing and hard to follow. Additionally, Figure 4A uses a setup with n=2, m=1; I cannot be convinced by experiments on models with a hidden dimension of 1.

### Questions
1. I do not understand why Figure 2b claims that MoE exhibits far less interference with other features than the dense model, as observed in Figure 1b. In both Figures 2a and 2b, a feature interferes with at most one other feature (there is only one blue dot per row or column).


2. To maintain the total parameter count, m=6 is split into three experts with m=2. Is it possible that the inherent separation of a dense MLP into MoE explicitly reduces superposition? In other words, can the reduced superposition in MoE be taken for granted?

3. While I consider Section 5 an interesting perspective, I really can not accept experiments using a hidden dimension of  m=1.

### Soundness
1

### Presentation
1

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
The authors extend the mechanistic analysis of dense models by Elhage et al. to a toy mixture of experts setup. Through a newly introduced attempt to quantify monosemanticity, they show that large expert counts lead to more monosemantic features in their toy setup, and also that MoEs exhibit less of a phase transition as a function of sparsity. Ultimately, the authors advance our understanding of MoEs and tentatively advocate for their use as a more interpretable alternative to dense layers without performance degradation.

### Strengths
## [S1] Strong motivation + important problem

The paper addresses an important and timely issue in mechanistic interpretability. As the authors correctly note, the MoE architecture is quickly becoming the dominant paradigm for pushing capabilities, yet our understanding of how they work is in its infancy. This sets up the authors’ topic to be of high interest to both interpretability and capability researchers alike.

Unfortunately, as I discuss below, I see some issues with extending insights here to MoEs in practice, but the promise of the work’s premise still remains.

### Weaknesses
## [W1] Toy setup formulation needs polish and discussion about transfer to practice

The authors’ proposed toy MoE formulation for each expert as $\text{ReLU}({\mathbf{W}^e}^\top\mathbf{W}^e\mathbf{x}+b^e)$ is quite different to how MoEs are implemented in practice. Even compared to the original Sparse MoE [1], i see two important differences:

1. MoE expert’s input layers’ weights are **not tied to be symmetric**, like the authors’.
2. Each expert’s FFN often includes a second linear transformation after the ReLU [1].

The authors should discuss thoroughly how much the proposed symmetry constraint and omission of final linear transformation hinders our ability to extend insights to the non-toy settings? My concern here is that without explicit justification for why this indeed connects to practice, the authors’ insights might be heavily constrained to their unusual toy model formulation alone. Additionally, many SOTA MoEs in practice now use a shared expert [2,3,4]. The authors should comment on how superposition and/or their analysis is affected under this setup.  

### Modifications needed

Importantly, the technical formulation of the toy MoE setup in Section 3 needs clarity and correction. This is necessary to make the authors’ setup perfectly legible to readers, given its non-standard nature. Whilst each issue alone may appear trivial, the presence of many such errors in presentation leads to the general impression that the paper lacks clarity, and precision--for a paper carefully studying a newly introduced toy setup, it is of paramount importance to clearly and correctly formulate the toy model they are proposing.

Some issues:

- [L111] this equation does not compute. $W_r^\top x$ would be needed (with the transpose) for this to work.
- On [L114], there is a confusing inconsistency between the use of the gating weights with $w_e$ and $w^e$ at once, which also clashes with the notation used for the input layer. The authors should define clearly how the normalization is computed, and I would suggest naming this something different entirely (e.g. $a_e$).
- $W_e$ is not defined, nor are its dimensions (used on [L112]).

## [W2] Missing related work section

The authors do not include a dedicated discussion of related work. Whilst 16 references do appear throughout the paper, a dedicated section is crucial to place the authors’ contributions in context of the prior literature.

As one example of why this is important, one of the authors’ key contributions is a definition of expert specialization for monosemantic features ([L066]). However, the authors do not discuss existing attempts to quantify expert monosemanticity in the literature, and why their proposed analysis offers additional insights; measured through ablations in [5,6]. A detailed discussion of how the proposed analysis relates to both existing works should be made to situate the work in relation to existing attempts.

---

## References

[1]: Shazeer, Noam M. et al. “Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.” ICLR 2017

[2] Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., ... & Piao, Y. (2024). Deepseek-v3 technical report. *arXiv preprint arXiv:2412.19437*.

[3] Team, K., Bai, Y., Bao, Y., Chen, G., Chen, J., Chen, N., ... & Zhang, H. (2025). Kimi k2: Open agentic intelligence. *arXiv preprint arXiv:2507.20534*.

[4] Meta AI. (2025, April 5). *The Llama 4 herd: The beginning of a new era of natively multimodal intelligence*. https://ai.meta.com/blog/llama-4-multimodal-intelligence/

[5]: Park, Jungwoo, et al. "Monet: Mixture of monosemantic experts for transformers." ICLR 2025.

[6]: Oldfield, James, et al. "Multilinear mixture of experts: Scalable expert specialization through factorization." NeurIPS 2024.

### Questions
## [Q1] Mixed definitions of monosemanticity / features

On [L086], the authors state `Monosemantic features are defined as those that are well-aligned with individual neurons`. 

I am a little confused by this definition. In Elhage, monosemanticity is a property of *neurons* (possibly SAE latents), not the high-level concepts; the goal is to establish the independent computational units of meaning. The difference between the two appears to me important.

For example, there may exist multiple neurons that monosemantically correspond to the *same* high-level concept. This is consistent with the Elhage definition of monosemanticity, but not the authors’, when formulated as a property of the concept.

Furthermore, “features” is used on [L086] to refer to human-interpretable concepts, but again on [L110] onwards to denote the $n$ input neurons. Monosemanticity and superposition suggests that this equivalence does not hold.

Might the authors please clarify their use of the terminology here?

## [Q2] Load balancing mixed use

The authors should comment on why load balancing is used for Sect. 4 but not for Sect. 3. At the minute, it is left unexplained; and the extent to which experts are balanced should surely influence the kinds of features it learns.

Specifically, without a load balancing loss for the experiments in Sect. 3, what is preventing the MoE from learning a single expert alone (functionally equivalent to the dense model)? Might the authors please comment on the balance observed in the first section?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper applies the superposition framework of Elhage et al to MoEs, in the context of toy autoencoders. The main claims are: (1) MoEs represent the same number of features as do dense models with the same capacity, but they do so more monosemantically, meaning with less superposition or interference between features. (2) MoEs don't show the same phase transitions that dense models do between monosemantic representation, polysemantic representation and ignoring features, as a function of feature sparsity in the input distribution and imbalance in the feature weights on the loss function.

### Strengths
The approach potentially gives insight into representation and expert specialization in MoEs

### Weaknesses
I had great difficulty figuring out what was done in many parts of the paper. I don’t normally share such detailed notes as I do in the Questions section, but in this case I do to help explain how much work this paper needs.

The main claim that “MoEs represent the same number of features as the dense model, but more monosemantically” (e.g., L220) seems impossible. How can two models match in the number of features they represent and the number of dimensions (“parameters”) they use, but differ in the number of features per dimension (i.e., in monosemanticity)? 

Mathematically, the definitions of feature representation (L141) and features per dimension (L204) are nearly the same. If we stack $W=W^{1:E}$ we get an $Em\times n$ matrix matching the dense model. Setting aside marginal load imbalance (i.e., assuming $p_e=k/E$) the definition of features per dimension is the same for the two models (i.e., stacked and unstacked representations). Moreover the result equals the sum of squared feature strengths: $|W|_F^2 = \sum_i |W_i|^2$ for the dense model and $|W|_F^2 = \sum_i \sum_e |W^e_i|^2$ for the MoE. So if the models really match on summed feature strengths and differ on features per dimension, it seems like this can only be because of the different orders of summation and squaring in how the measures are applied to the two models (and this should be spelled out in the paper), but that would hinge on the seemingly arbitrary choice to define feature strength as $|W_i|$ rather than $|W_i|^2$. Also the fact that strengths in figs 1a and 2a are all near {0,1} suggests that squaring makes little difference.

Putting all this somewhat differently, if superposition happens when a network encodes more features than it has hidden dims then MoE (with the same total number of hidden dims as the dense model) can’t help: monosemanticity will still require dropping some features.

### Questions
L58: What phase change, i.e. what is the macroscopic variable and what are the hyperparameters?

L63: It looks like the hyperparameter for the phase change is network sparsity, but that doesn’t apply to dense models.

L87: Does it have to be a single neuron, or can it be an oblique direction in activation space? It should be the latter because of rotation invariance (for standard FF layers). The important criterion is that features have orthogonal representations.

L102: What claim? The previous sentence is just descriptive. I also don’t understand the second half of the sentence: do you want to quantify representational similarity between experts? (I don’t think that’s the topic of this paper.)

L111: $n$ is the number of input features, not the input features (those are denoted $x$)

L112: $W_r$ is $E\times n$ not $n \times E$. It would also help to state $W^e\in\mathbb{R}^{m\times n}$.

L114: $w_e$, $w^e$

L116: What is the loss? From what comes next I think it’s squared error with weights $I$. Also what is the dataset or generating distribution? What is the optimization/training procedure? These are critical questions for understanding all the experiments in the paper.

L140: $W^e_i$ is column $i$ of $W^e$?

There seems to be an assumption that $|W^e_i| \le 1$. I can see this for $E=1$ because otherwise the reconstruction overshoots. But in MoE the reconstruction is weighted by the gating weights which are $<1$ (see def of $x’$ at L114). For example if a feature is represented by only one expert $e$ then that expert would need to scale up its output by $1/w_e$. 

Does the term ‘dense model’ mean anything more than $E=1$?

How is superposition in figs 1a and 2a defined?

L146: What does “roughly the same number of features” refer to? I see 10 features (0,1,3,5,6,7,8,9,10,12) represented in 1a and 8 features (0,1,2,3,4,5,6,9) in 2a.

In what sense do the models have equal total parameters? They have an equal number of hidden dimensions (6) but the MoE also has gating weights.

Figs 1b 2b are described as measuring interference but they don’t match the definition at L143. Also the claim is less interference in the MoE but I count 4 interfering pairs in fig 1b and 5 in 2b.

L204 (please consider numbering equations): This expression doesn’t work as a count if $|W^e_i| > 1$ (see comment above) because then feature $i$ contributes more than 1 to the count.

L262: I think you mean $x=(x_1,x_2,\dots,rx_n)$. Also it’s poor notation to define the $n$th component of $x$ as $rx_n$.

L263: $x_i\sim U(0,1)$

L263: "$S$ likelihood that $x_i=0$" doesn’t make sense. You want to define a mixture distribution between a uniform and a point mass.

L267: This should be stated formally especially given earlier confusion about param counts. I think you mean $m_{\rm dense} = km_{\rm MoE}$.

Equating active dimensions ($m_{\rm dense} = km_{\rm MoE}$) doesn’t seem like the right comparison (as opposed to $m_{\rm dense} = Em_{\rm MoE}$) because it gives the MoE more capacity than the dense model. The MoE can represent a feature with some expert and choose not to activate that expert when that feature is absent. So it’s not clear whether the differences regarding phase transition in fig 4 are due to dense vs MoE per se or due to differences in model capacity. 

Fig 4: I think the network diagrams for ABC are meant to indicate the values of $n$ and $m$. The implication about $m$ is ambiguous so it would be better just to state the values. Also what are the values of $k$? (Ok we are eventually told $k=1$.)

Each pixel in fig 4 is a completely separate simulation, and each model should be invariant to permutation of the experts, so how can there be systematic differences between experts 1 and 2 in the first two columns?

Why is the importance of only one feature being varied? I suspect the figure is showing values only for that feature but the caption suggests otherwise (subscript $i$ instead of $n$).

L368: Does “feature $x$” indicate $x$ denotes a unit vector $e_i$ (i.e., $x_j = \delta_{ij}$) or does “feature” here refer to any input vector (i.e., arbitrary $x\in\mathbb{R}^n$ or perhaps $x\in[0,1]^n$)?

L371: What measure is used to define volume? Induced Lebesgue measure on the L2 unit sphere?

L373: This claim (“they tend to align experts with particular features”) is not warranted by the tiny sample shown in fig 5. It requires a systematic study, also using more feature dimensions since with $n=2$ it’s nearly impossible to have good load balancing without allocating $x=(1,0)$ and $x=(0,1)$ to different experts.

L409: I don’t think $c_i$ has been defined. Is this a statement about $W_r$ from L111? More importantly, how can the gate matrix be the diagonal (I assume you mean an identity matrix) when it isn’t square ($W_r\in\mathbb{R}^{E\times n}=\mathbb{R}^{5\times20}$)?

Fig 5: what do the colors represent? (Probably separate questions for even and odd columns)

### Soundness
2

### Presentation
1

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
This paper extends the ideas presented in the anthropic blog, with a focus on toy MoE models in the context of superposition. 
The authors examine the superposition of these toy MoE models and observe the absence of a phase transition, which is seen in dense models. 
Additionally, the paper explores expert specialization and initialization, which provides insights to the understanding of MoE behavior.

### Strengths
- The exploration of expert specialization and initialization is interesting to me. 
These topics provide insights into a better understanding of MoE behaviors.

- The authors conduct extensive experiments to support their idea.

### Weaknesses
- Some of the paper is a straightforward extension of the anthropic blog, which adapts the research on dense models to MoE models.
As a result, the contribution feels somewhat limited.

- The authors' findings on toy models are interesting, but they are not entirely convincing to me due to the experimental setups.
Firstly, the experiments are conducted on toy models with a very small hidden dimension (e.g., 6 or even 1).
While interesting, it is hard for me to trust the conclusions draw from such a toy models;
Moreover, it is unclear whether the conclusions drawn from the two layer MLP with ReLU are truly relevant to modern FFNs, such as SwiGLU, which limits their applicability to real-world MoEs.

### Questions
---
Q1: Would it be possible for the authors to conduct the experiments in Figure 5 with different configurations?

The current results, set m=1, do not provide sufficient evidence to convince me.

---

Q2: I find Figure 4 to be somewhat complex and difficult to interpret. 
Could the authors consider revising the caption, improving the labels, and providing additional clarification on the experimental design.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies how mixtures of experts learn some toy models of data, versus how dense MLPs learn them. The paper finds that mixture of experts neurons are much more monosemantic than dense MLP neurons.

### Strengths
* The question of how the mixture of experts architecture interacts with how concepts are represented by neurons (monosemantically vs. polysemantically) is interesting.

* The analyses of this paper look like they are probably quite interesting. I just had a very tough time reading them because the basic definitions and setup are not presented.

### Weaknesses
* Some definitions are missing, which makes the paper unclear in cases and hard to read in others. (See my questions below for examples.) The paper could greatly benefit from a clearer exposition of definitions so that readers can understand what the authors concretely mean by a "feature", or by "monosemanticity" in this context.

* The analyses are all conducted on toy models, without any analysis, e.g. of MoE models trained on real data.

### Questions
* What does importance I = 0.7^i mean? What is the data distribution in the experiments? This is not described until much later on in the lines 260-264, but I am not sure how to parse this definition. Are the features vectors? Why are they scalars in this definition? Why is only the last feature sampled from a different range from the other ones?

* What do the colors of the bars mean in Figure 1(a), Figure 2(a)? Are these D_i scores from 0 to 1?

* The architecture choice in section 3.1 doesn't make sense to me. Why is the ReLU applied outside of everything else when computing x_e' ? Doesn't that mean that the architecture will always output reconstructions with nonnegative entries?

* How is a feature defined as monosemantic or not?

### Soundness
2

### Presentation
1

### Contribution
2
