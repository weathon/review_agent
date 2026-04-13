## Human Reviewer 1

### Summary
This paper aims to train sparse autoencoders (SAEs) to find causal features of language models, which is a problem in the field of mechanistic interpretability. This paper proposes to use the JumpReLU activation, illustrated in Fig. 3(a). The authors provide ways to estimate the gradient of the JumpReLU activation, as well as ways to estimate the gradient of $L_0$ norm regularization. The estimation mainly uses KDE.

### Strengths
1. The estimation of the $L_0$ norm's gradient makes sense and seems non-trivial to me.

### Weaknesses
1. The contribution seems quite incremental to me. The authors propose two things: (i) using the JumpReLU activation, and (ii) a way of estimating the gradient of JumpReLU and $L_0$ regularization. The JumpReLU activation is not invented in this work. And the estimation, though non-trivial, is derived from standard tools.

2. The experimental results show that the proposed method has similar performance to TopK SAE. The only plot where the proposed method seems better than TopK is Fig. 5(a), but this is actually rated by the authors themselves and people from the same group so it a bit questionable. TopK SAE (Gao et al.) was published on ArXiv on Jun 6, so is considered contemporaneous according to ICLR's guidelines. However, it's worth pointing out that the proposed method is not significantly better than TopK SAE.

3. One key ablation study is missing. The authors propose two things: JumpReLU, and using $L_0$ regularization instead of $L_1$. What will the performance be like if only one of them is used? For example, $L_0$ regularization + TopK? I imagine that this could have a very similar performance, which raises a question about the effectiveness of JumpReLU.

### Questions
1. What is the intuition behind JumpReLU? To me, TopK is very intuitive. However, I cannot understand why JumpReLU is reasonable. Is it derived from Gated SAE?

2. In line 86, you mentioned that TopK needs a partial sort while JumpReLU does not. However, what is the downside of this partial sort? Does it make the method slower? If so, how much slower? If it is only a bit slower, then it won't be a big issue.

3. Are there any qualitative experimental results? For example, can you provide some examples of the causal features you found using your method?

**Summary:** I am not an expert in mechanistic interpretability. To me, overall it seems that this paper is incremental and is not significantly better than a previous paper TopK SAE. Although TopK SAE is considered a contemporaneous work by ICLR's standard, this paper heavily depends on that paper so it is questionable. I feel that the contribution is not sufficient for acceptance at ICLR. My current rating is reject with low confidence, and I might change my score after discussing with my fellow reviewers and the AC.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
3

### Confidence
2

---

## Human Reviewer 2

### Summary
This paper tackles the important issue of LLM interpretability and does so by improving Sparse Autoencoders (SAEs), an unsupervised learning technique that decomposes model activations into interpretable features. This paper introduces a new activation function: jumprelu.

Jumprelu removes values below a certain greater than 0 threshold, and the threshold is trained to improve L0 (sparsity) and reconstruction fidelity.

The results are comparable to or better than leading SAE activation functions such as top-k and gated SAEs, but the computational cost is lower than top-k’s.


My **recommendation**: Accept, 8/10 (good paper).



### Why not lower score?

- The results are state of the art, and represent a pareto improvement.
- The approach is well placed in the literature, and well motivated.

### Why not higher score?

- The improvement is incremental, but not beyond comparison with alternatives.
- The application of Jumprelu to SAEs is new, but both tools were already known.

### Strengths
## State of the art results

Jumprelu SAEs are easier to train than top-k SAEs, as they don’t require a sorting operation to get the top-k activations at each pass.  The ***quality*** of the method is such that it attains state of the art results, and it is a ***significant*** improvement over the pareto frontier. It represents an ***original*** application of Jumprelu to LLM activations.

## Direct L0 training

Jumprelu SAEs can be trained directly on the L0 loss, which is an improvement over approximations such as the L1. L0 does not lead to activation shrinkage as is the case with L1. This approach is ***clearly*** justified and explained with the usage of a STE that allows the authors to compute pseudo derivatives.

## Evaluations: automated interpretability 

The results of the automated interpretability experiments are encouraging, and they suggest a big majority of features may be automatically interpretable. This is important as the number of features represented in large language models might scale superlinearly with model size.

## Hyperparameter transfer

The authors introduce a new hyperparameter: epsilon. They use it in the definition of their pseudo derivative and note that it transfers well across model sizes. This can make scaling the method to larger models easier.


## Reproducibility

The results of this paper refer to the Gemma suite of LLMs, which is open source and allows for the reproduction of the author’s results.

### Weaknesses
Most of my concerns are acknowledged in the limitations section of the paper, but the following are not fully addressed.

## Downstream evaluations

Following work on sparse feature circuits (Marks et al., 2024) and using known features the authors could have included evaluations on downstream tasks, such as circuit reconstruction (faithfulness, completeness, etc..) or on the model’s ability to retrieve known features. Downstream evaluations would have strengthened the paper.

## Hyperparameter introduction

Introducing the epsilon hyperparameter without a reliable selection method, while mitigated by the empirical observation that epsilon transfers, still makes for a more complex method. Furthermore, no alternative formulations for the pseudo derivative are proposed/considered. 

## No mitigations for high frequency features

While this is addressed in the limitations and future directions, there is no attempt to mitigate the presence of uninterpretable high frequency features, which represents a drawback of the proposed method. Moreover, there is no explanation for why there are more high frequency features in jumprelu SAEs than in top-k SAEs.

### Questions
Have you tried alternative formulations for the pseudo derivative that don’t require a hyperparameter?


Have you tried resampling (resetting) high frequency features? 
This is done for dead features and can help.

Have you considered downstream task evaluation (faithfulness, completeness, etc..) of known circuits (such as IOI) using mechanistic interpretability (Marks et al., 2024)?

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 3

### Summary
JumpReLu Sparse Autoencoders (SAE) are here introduced. Instead of using ReLU, as in the original formulation of SAE, JumpReLU can result in more sparse but strong activations. This framework is further tuned for performance, with a couple ad-hoc design choices. 1) A loss function consisting of a weighted sum of reconstruction error (L2) and sparsity (L0) is used, which attempts to maximize fidelity with as little activations as possible. 2) In order to train the threshold parameter, authors use STE. This algorithmic trick allows to estimate the gradient of the *expected* loss and thus to train JumpReLU SAE with gradient methods, even when the loss function would not provide useful gradients.

In one experiment, they show matching or slightly better results than benchmark in reconstruction fidelity for three selected layers of Gemma 2.

The authors characterized some intrinsic properties of JumpReLU SAE operations, like the proportion of feature that are very frequently active and interpretability of results. They also performed an editing experiment. In all these cases, results are similar to previous work.

### Strengths
The paper has a clear step-by-step presentation of SAE and of the algorithmic features utilized, moving from general to specific. By introducing general equations, including the loss function and SAE architectures, it provides a solid basis for understanding. Each methodological detail is thoroughly explained, making the thought process reproducible and leaving little open to interpretation. The authors effectively clarify the SAEs used, and discuss the broader implications of results, adding depth and impact to their findings. I also enjoyed the slightly informal writing style, it reads very clear.

The submission is also technically correct, with the exception of some points (see below).

### Weaknesses
My assessment is that unfortunately this paper offers a limited incremental contributions, as it builds on ReLU SAE with the introduction of JumpReLU. It does add two interesting algorithmic tweaks (a custom loss function and STE for gradients) but they are not central. These adjustments fall short of significantly advancing the state of the art within the SAE domain. The authors’ claim of state-of-the-art performance is based on a single experiment with one model, where selected layers appear cherry-picked, with only marginal improvement. Other experiments on encoding explainability fail to surpass benchmarks. 

Changing my opinion on the testing being too selective would imply seeing results across the layer stack (how did you select layer 9-20-31?) and on a second model other than Gemma 2 9B. I believe it's too big of an ask for revision, so I am a bit pessimistic about it.
Additionally, figure legends and labels are frequently inadequate, contrasting with the otherwise high quality of the text.

### Questions
- Figure 1: What do the blue and red colors represent? Please clarify the legend. also please label the axes.
- Why is JumpReLU restricted to a positive threshold?
- Figure 2: This figure is poorly explained. To what does "Delta LM loss" refer here? This term is introduced too early in the paper (Figure 2 is on page 2, while Delta LM loss is introduced on page 6), making the figure difficult to interpret as presented. Please state that axes are log scale.
- The preliminary definition of SAE is incorrect. The encoder should map to a smaller latent space, m<<n . Explicitly stating the activation function in Equation 1 would also be helpful.
- Why were layers 9, 20, and 31 specifically selected?
- Figure 3 please label axes.
- Figure 4: indicate x is log scale, while y is not.
- An example figure of the rating procedure would be helpful, even as an appendix item.
- The results on automated interpretability are well explained but would be clearer if presented in a table, as is standard in this type of publications.
- Typo in Line 185 "be actually be".
- Line 216: refer to the relevant section.
- Page 27-28: Algorithms should be presented as pseudocode, not as actual implementations, as this style is harder to understand in a CS paper. Implementation details can be saved for a code-publishing section.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
3

### Confidence
3

---

## Human Reviewer 4

### Summary
For LM activation decomposition, the paper proposes to use JumpReLU activation and $L^0$ norm instead of $L^1$ norm for sparse autoencoders. The proposed method obtains similar results as the baseline models.

### Strengths
* Good discussion about the limitations and the future works.

### Weaknesses
* The motivation of replacing ReLU by JumReLU activation for SAEs is not lear, the reason is not even explained in the paper.
* The idea of using JumReLU activation for SAEs is already presented in Gated SAEs, if I am understand it well.
* The paper is not well-written and the notations are not rigorous. To name a few, in Equation 1, $\mathbf{f}$ is used both for the function and the vector. For the notations of norm, please use $L^p$ instead of $Lp$.
* While the paper proposes to change the activation function, it proposes also to change the regularization norm type. However, there are no ablation study has been conducted in the experiments, so we have no clue from which part of the contributions comes from the improvement.
* The performance improvement of the method is not evident.

I strongly recommend the authors rework the paper.

### Questions
* Can the authors explain the motivations behind the JumReLU activation?
* Can the authors conduct the ablation study as stated in the weakness?
* Can the author give more context about the problem to solve, for example, why we need the sparse autoenocder to evaluate LM?

### Soundness
1

### Presentation
2

### Contribution
1

### Rating
3

### Confidence
4