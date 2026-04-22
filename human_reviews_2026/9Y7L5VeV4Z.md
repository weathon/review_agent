# Globally 1-Lipschitz Attention Without Sequence-Length Dependence

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 8, 2, 4

## Abstract
Self-attention powers modern deep learning; however, dot-product attention is not globally Lipschitz, which limits stability and robustness. Prior fixes enforce Lipschitz continuity by changing the geometry of attention, which departs from the standard mechanism and still yields bounds that scale poorly with sequence length or spectral norms. We introduce $\mathrm{LipAttn}$, a new attention block that derives coefficients from a convex potential and realizes them through an implicit proximal update, guaranteeing architectural stability. This design ensures the entire block is firmly non-expansive and thus \emph{unconditionally 1-Lipschitz}, independent of sequence length or parameter norms, while remaining structurally close to dot-product attention. Building on monotone operator theory, we establish its contractive properties and develop efficient first-order solvers. 
Experiments on OpenWebText show that $\mathrm{LipAttn}$ achieves meaningful token mixing and retains learning capacity.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a new attention-like layer which is constructed to be 1-Lipschitz in the sense of sequence-to-sequence mappings and w.r.t. the Euclidean (Frobenius) norm on the sequence matrix. They do this by defining a convex potential function such that their operator is an implicit Euler step of the gradient flow on this potential. This implicit Euler step is itself implemented via a proximal operator, i.e., the solution to a convex optimization problem, and as such the mechanistic form of this operator is essentially multiple gradient steps on a convex objective. The authors do experiments with single attention-like layers to demonstrate performance of this approach.

### Strengths
- The theory (Theorem 4) is rigorous and the proof seems correct.
- The paper is well-written and easy to read. The mathematical exposition is clear and a nice, concise introduction to the prerequisite concepts.

### Weaknesses
Unfortunately, there are a multitude of points that need improvement.

- In terms of methodology, it seems that the proposed method of taking multiple gradient steps is very inefficient compared to the standard attention, which seems to only require the computational cost of a single one of these gradient steps. The proposed method uses line search which may be even more costly. To drive the gradient norm (a proxy of the distance to convergence) beyond $10^{-2}$, say, it requires more than 40 iterations --- the forward pass for a single _layer_ of the proposed mechanism is then as costly as a full forward pass for a much, much larger deep network.
- The drop in efficiency does not buy an increase performance, as the PPL on real data is much larger for the proposed method compared to the baseline regular attention (Table 2).
- Table 2 is also not promising; it shows that the performance of a standard attention layer is 7x as good (measured by perplexity) as the proposed method. It then claims that the important metric is not perplexity but perplexity times the Lipschitz constant, which seems very arbitrary, and is also optimized by the zero function (viewed as a sequence-to-sequence operator). This metric seems designed to disqualify regular attention, no matter how good the performance is.
- The setting for Table 2 is also not very realistic; usually one evaluates the performance of whole networks, not individual layers within the network, and it is difficult to interpret the performance of individual layers. In practical architectures, the transformer is followed by a MLP and more generally blocks that may blow up the Lipschitz constant, so it is unclear what benefits this layer will buy you when incorporated into a larger network.
- The justification for caring about Lipschitz constant is the robustness, but this is only justified in the context of classifiers; it is not so clear whether robustness is driven by small Lipschitz constants for sequence-to-sequence models. Some more experiments demonstrating the robustness would be very good to show your point here

Altogether, the paper fails to demonstrate any realized improvement of the new method, and instead it looks worse in every respect than the conventional baselines.

### Questions
See above. My most pressing questions are:

Q1: Are there empirical validations showing that a network architecture using this layer will indeed improve robustness on practical tasks (compared to a regular attention)?

Q2: Can you close the (frankly massive) performance and efficiency gaps compared to regular attention?

### Soundness
4

### Presentation
4

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a new Attention-like block that guarantees 1-Lipschitzness via a convex potential function. First, the paper casts the residual attention update as a continuous dynamical system and introduce a new convex potential function in place of the current attention function. The system is then re-discretized as an Euler update which is realized by the proximal operator. The proximal problem is empirically solved via beam search and Armijo backtracking. By construction the design realizes global Lipschitzness independently of the input sequence length. The paper verifies its findings via experiments and contains promising experiments on small networks.

### Strengths
1. The paper is well-written, clear and mathematically rigorous. The tools used from optimization theory are fundamental and well-explained.
2. The contribution is important and fills an important gap in the literature. Softmax attention is a famously unstable function and alternatives realizing stability often show Lipschitzness bounds depending on the weights or sequence length. This work gives a promising step forward to architectures that don't exhibit either disadvantage.
3. The experiments performed inspire confidence in the result, both in theory and in practice. The method seems to perform as advertised in practice as well.

### Weaknesses
1. The biggest weakness of the paper is that it is unclear if the proposed method scales. Solving the proximal optimization problem, though possible, and evidently achievable, is likely too large of a cost for larger models. Further, the experimental evaluation is limited to one layer neural networks, which further raises question of scalability.
2. A point the paper does not address is the model degradation seen after imposing such strong Lipschitzness conditions on the attention function. The experiments already show there is some tradeoff between stability and expressivity, as the paper recognizes. Indeed, we know many functions require lack of stability to represent (see boolean functions with high sensitivity for example). Is it possible to introduce a hyperparameter in the method that controls the Lipschitz constant, possibly allowing us to reach a "pareto optimal" between perplexity and stability? I would have really liked to see such an experiment. Ultimately, though much better than the $\ell_2$ attention in terms of stability, this method achieves much lower perplexity score, which could be seen as a disadvantage. That being said, the framing of the paper seems to be mostly theoretical and it does present a very promising direction towards reaching this "pareto optimal" point or understanding the inherent tradeoffs. I would recommend a more extensive discussion of this point to go in the next version of the paper.
3. I'm finding the definition of the convex potential a little artificial:
    * Why is $S^{\text{convex}}(x)$ a symmetric matrix? Doesn't this take away the expressivity of capturing the interactions between $x_i$ and $x_j$? Can the authors discuss some intuition behind defining things like this?
    * Is there anything that can be done to avoid the weight tying $W_K = W_Q$? 
    * Are there other functions that would fit this paradigm? Perhaps in combination with other attention-like alternatives? Some ablation study would be nice here.

Overall the paper makes a technically solid and important contribution, with the main weaknesses being scalability and perhaps a slightly wider study of the method's place within the larger AI community.

### Questions
Stated above.

### Soundness
4

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
3

### Summary
This paper proposes AttLip, an implicit self-attention block designed to be globally 1-Lipschitz (firmly non-expansive) without dependence on sequence length or parameter norms, addressing limitations in standard dot-product attention (which isn't Lipschitz) and alternatives like L2-attention (whose bounds degrade with length). It derives attention coefficients from the gradient of a convex potential encoding pairwise token interactions via a quadratic form, then realizes the update as a proximal operator solved via gradient descent with Armijo backtracking. Theoretical analysis, rooted in monotone operator theory, proves contractivity and solver convergence. Experiments focus on a one-layer non-causal next-token prediction on OpenWebText, showing AttLip achieves non-trivial token mixing (perplexity 141) while maintaining an empirical Lipschitz constant of ~1, compared to unbounded standard attention (perplexity 22) and loosely bounded L2-attention (perplexity 28, bound ~2000+).

### Strengths
1) Innovative fusion of convex potential flows and proximal methods to enforce rigorous Lipschitz guarantees by design, avoiding the sequence-length scaling issues that plague prior work like Kim et al. (2021).

2) Strong theoretical contributions: Clear proofs for convexity of the potential, contractivity of the flow, and linear convergence of the solver, with practical bounds on local condition numbers that inform implementation.

3) Empirical Lipschitz estimates via PGD align tightly with theory (constant 1 across N=16 to 2048), and the OpenWebText results demonstrate preserved learning capacity under strict constraints, using a combined PPL × Lip metric to highlight the robustness-expressivity tradeoff.

### Weaknesses
1) The enforced symmetry (via weight tying WQ=WK=W, WV=W^T W) fundamentally alters attention's geometry, making it more akin to distance-based than dot-product, which hurts expressivity which is evidenced by the massive perplexity gap (141 vs. 22) that isn't adequately addressed or mitigated.

2) Experiments are underdeveloped. Only a toy one-layer setup on OpenWebText, no deeper Transformers, no causal masking for real LM tasks, no downstream evaluation (e.g., GLUE for robustness), and no scaling to longer contexts or larger models where length-independence would shine.

3) Even with few solver steps, backtracking adds ops per forward pass, potentially prohibitive in production-scale training; no runtime benchmarks or comparisons to efficient attention variants like FlashAttention.

4) The quadratic potential feels ad-hoc (why this form over others?), no ablations on \eta (fixed at 1), solver tolerance, or ways to relax symmetry without losing convexity. It feels like a proof-of-concept rather than optimized.

5) While bounds are tight, the paper admits prior ones (e.g., L2-attention) are "vacuous" in practice yet still useful; here, the strict 1-Lipschitz comes at high cost to performance, questioning if it's worth it without certified robustness demos.

### Questions
1) Since AttLip's symmetry enforces bidirectional token interactions, how does it handle tasks with inherent asymmetry, like machine translation or summarization, where queries should attend differently to keys?

2) The proximal solver relies on local Lipschitz estimates for backtracking, how sensitive is convergence to inaccuracies in these estimates during training, especially in high-dimensional embeddings?

3) Could the convex potential be generalized to incorporate positional encodings or causal masks while preserving monotonicity, to better suit autoregressive models?

4) In the one-layer experiments, the perplexity is much higher than baselines, does fine-tuning or hybrid integration (e.g., mixing AttLip with standard layers) close this gap without violating guarantees?

5) Can the quadratic convex potential truly capture the inductive biases of dot-product attention, or is AttLip fundamentally learning a different similarity geometry that only mimics attention under low-rank constraints?

6) If AttLip is stacked in a deep Transformer, does the firm non-expansiveness of each block induce a vanishing signal problem stronger than standard attention, and if so, can it be mitigated without sacrificing the Lipschitz guarantee

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work focuses on the non-Lipschitzness of the standard attention block. They propose a non-causal, global 1-Lipschitz attention block, independent of sequence length $N$. They rely on a convex potential function, and the forward pass is computed by solving the proximal problem on this potential function—this guarantees 1-Lipschitzness through well-established proximal operator theory. They provide some experiments comparing the empirical Lipschitzness of a forward pass to another modified Lipschitz attention block in the literature, which theoretically degrades in its Lipschitz constant with increasing sequence length. They also run experiments on OpenWebText data by training their proposed attention block and comparing validation perplexity (PPL) with standard attention and other baselines.

### Strengths
First of all, the work focuses on the very important problem of non-Lipschitzness in standard attention blocks. In theory, having Lipschitz architectural units should imply stability with respect to (w.r.t.) inputs.

I enjoyed reading the paper. I think the work does a good job guiding the reader through the flow of steps: starting with the non-expansiveness of one step of standard GD on a convex, smooth function, and then recasting the architecture as something that implements this step as a forward pass; then moving onto the convex potential function for attention, its non-smoothness; and finally, recasting it as a proximal problem that gives non-expansiveness for free for any convex function. I think it’s a neat idea; although not a novel idea (Bai et al. 2019), it's still pretty neat.

### Weaknesses
One major weakness is that i) the attention block is non-causal, and ii) the key and query matrices are tied ($W_Q = W_K = W$), which limits the expressivity of the model. I think this is fine as a starting point, especially because this is a theoretically-informed work, but I’d like to see a discussion of limitations that highlights these points.

For other major weaknesses, I have the following points/questions:

1. **True Independence from sequence length $N$**: The paper's central claim is a 1-Lipschitz guarantee "independent of sequence length". However, this guarantee only holds for the true proximal operator, which must be approximated by an iterative solver. Appendix (page 18) shows that the local smoothness of the prox problem, scales linearly with $N$ implying slower convergence to an $\epsilon$ tolerance in the forward pass as we blow up $N$. How significant does this become in practice? I think we need more experiments with larger $N$ to fully see this effect and N=2048 is not large enough. 

2. **On Computational Cost (Backward Pass)**: The forward pass requires an iterative solver. How is the backward pass computed? Is it by differentiating through the $K$ unrolled steps of the solver? 

3. **Comparison on compute**: The paper shows convergence with steps for solving the proximal problem in Figure 1. I think a fair comparison to standard attention and $l_2$-attention should be comparing the wall-clock time of all three for both the forward and backward passes. This should also be done for a range of embedding dimensions $d$ and number of heads $N$ to get a true sense of the computational overhead. Also, an empirical Lipschitz estimate for the standard attention block should be included in Table 1 as a baseline.

4. **Role of $\eta$**: As the hyperparameter $\eta$ dictates both the convergence rate of the proximal problem and the expressivity of AttLip (small $\eta$ is less expressive but faster to solve), there should be a natural trade-off here. An experiment ablating this trade-off would be good.

5. **OpenWebText Experiments**: As the expressivity of the model is limited (tied key-query weights), I'm guessing the validation PPL of the proposed block will get worse as we increase the number of layers, compared to the standard attention block. Experiments showing this scaling behavior would be helpful. Also, the paper says it only uses $K_{prox}=1-3$ steps for solving the proximal problem here. What is the empirical Lipschitz constant in this $K=1-3$ setting? The 1-Lipschitz guarantee only holds if the proximal problem is solved to convergence, so this is a critical detail.

My low score is currently due to these limitations, as I think the paper needs a little more work. While my suggestions could point to negative results for the proposed attention block, I believe it is important to highlight these aspects for future work. I would be happy to discuss this further with the authors during the rebuttal.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
