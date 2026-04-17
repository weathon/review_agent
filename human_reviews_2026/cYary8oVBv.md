# Learning without training: The implicit dynamics of in-context learning

- Decision: Reject
- Scores: 8, 8, 2, 2

## Abstract
One of the most striking features of Large Language Models (LLMs) is their ability to learn in-context. Namely at inference time an LLM is able to learn new patterns without any additional weight update when these patterns are presented in the form of examples in the prompt, even if these patterns were not seen during training. The mechanisms through which this can happen are still largely unknown. In this work, we show that the stacking of a self-attention layer with an MLP, allows the transformer block to implicitly modify the weights of the MLP layer according to the context. We argue through theory and experimentation that this simple mechanism may be the reason why LLMs can learn in-context and not only during training.  Specifically, we show how a transformer block implicitly transforms a context into a low-rank weight-update of its MLP layer.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work proves that the impact of context tokens on the output of a contextual block (a generalization of a transformer block) is equivalent to a rank-1 update of the weight matrix of the feedforward component. This result explicitly links in-context learning to a form of gradient descent. Additionally, this work proves that in-context learning converges as context length increases. Both of these theoretical results are demonstrated experimentally for a 1 layer transformer.

### Strengths
The theoretical results are clear, compelling, and are of broad interest to the community.

The experimental results perfectly correspond to the theory.

The equivalent rank-1 weight update enables explicit comparisons between in-context learning and finetuning, which can enable future work.

### Weaknesses
Extending the experimental results to other forms of contextual layers would be of interest, especially as the authors note that different types of contextual layers might be more or less effective in-context learners. Though potentially out of scope, a comparison of the weight updates between e.g., RNN and attention components would be of broad interest. 

Furthermore, any intuitions regarding why finetuning is more efficient in the 5-20 step regime would strengthen the work and help to flesh out that section.

### Questions
See weaknesses section.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper shows that in context learning (ICL) can be reframed as low-rank updates to the MLP weights in a transformer. TThey prove that the efect of any subset of the context Y on the block’s output for a query x can be represented exactly by a rank‑1 update to weight matrix W. A corollary shows how the entire context can be folded into W (Eq. 2), and an extension (Theorem B.2) handles residual connections by adding a bias update equal to the context vector . Building on this identity, the authors derive an implicit learning dynamics across tokens that resembles stochastic gradient descent with a token‑dependent linear loss.
Experiments use a single‑block decoder‑only transformer trained on linear regression ICL tasks. They show that predictions obtained with the prompt are numerically indistinguishable from predictions made without the prompt after applying the rank‑1 weight transfer, that the induced updates converge as more tokens are consumed, and that the implicit updates are highly aligned with explicit finetuning gradients

This work gives an elegant identity that reframes ICL as "learning without training" via rank‑1 weight transfers inside a transformer block. The theoretical statement is clean and broadly framed; the experiments validate the identity in a controlled setting and suggest links to finetuning.

### Strengths
* The topic is interesting and relevant to the community. I find the results really striking and simple. The closeness of the match in figure 2 are very compelling
* Theorem 2.2 provides an exact algebraic identity showing that a contextual layer followed by an MLP is equivalent to a prompt‑free forward pass with a rank‑1 edit to the first MLP layer. The formula is simple and easy to implement.
* This generalizes beyond linear attention, which I think sets it apart in related work that I am familiar with
* I appreciate the discussion on the connections between steering, ICL, and ROME style updates that this work brings together nicely. I think this will be useful for the community

### Weaknesses
* Experiments are confined to linear regression ICL with a small, single‑block model without LayerNorm or MLP residuals (the latter are treated theoretically in App. B but not empirically). I think it would be of great interest to see the empirical results with the residuals at least.
* While the analysis is pretty deep, I wish the authors would be able to say something more about the updates to stacked blocks and how they interact. Is there even a heuristic approximation the authors can do to compare empirically? 
* I think this paper would be much more impactful if it could say something about the generation beyond the first token. Benefits of ICL accrue over rollouts, and I feel like theorem

### Questions
Given an MLP (that follows an attention block) with two weight matrices, W1 and W2, and a relu between them, I'm still a little confused whether this is able to say anything about how W2 changes?

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
4

### Summary
This paper attempts to understand in-context learning by showing that context data can be interpreted as an implicit low-rank update to the MLP weights. 
Specifically, the authors introduce the concept of a contextual block and show that adding context tokens is mathematically equivalent to applying a rank-1 modification to the MLP weight matrix. They further show that processing the context sequentially induces a gradient-descent-like implicit learning dynamics.

### Strengths
The work provides an interesting angle to understanding ICL, which appears to be less restrictive and more general than existing works, e.g., doesn't rely on attention linearization. The writing is clear and easy to follow. Theoretical analysis is supported by empirical verification.

### Weaknesses
### Limited significance
The idea sounds interesting but in practice it is mostly rewriting the MLP function with some basic algebra.  When considering the fact that the MLP does not mix tokens, including the separate context tokens does not seem as meaningful. 

Following the same context data, we could have different questions or instructions. Ideally, the parameter updates associated with context data should be independent to question/instruction tokens afterwards. In the paper's notation, \Delta_x W should be independent of x. In a related work [1], this is achieved by considering attention linearization and the corresponding weight update of each context token is also a rank-1 matrix in the attention "bias". 
However, if my understanding is correct, this is not the case in Thm 2.2 and the results seem less significant.  

The weaknesses of only affecting the first token is one that has been prevalent in the literature.  Before it is solved, this can only provide rough intuition as to how context can be designed rather than concrete improvements.

### Limited scope and practical relevance
The work provides implicit dynamics of ICL within transformers. 
Even though the paper claims to be more generally applicable to different LLM architectures with fewer restrictions, it is only valid for a single transformer block hence the limited scope. 
On the other hand, the theoretical analysis does not seem to highlight what is the meaning behind such dynamics and how they can be taken advantage of moving forward in some practical scenarios. 


[1] Exact Conversion of In-Context Learning to Model Weights in Linearized-Attention Transformers. ICML 2024

### Questions
* Given the weight change of the MLP varies once you have more than a single input  (as the input forms part of the context), is there a tangible direction forward which can possibly remedy this issue?

* What tangible benefits might arise from understanding these implicit dynamics beyond the providing an alternative to understanding ICL?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper claims that the effect of an in-context prompt on a transformer block can be written as an equivalent rank-1 “implicit weight transfer” to the first MLP layer (and a bias tweak in a residual variant). Theorems 2.2 / B.2 state this equivalence; Proposition 3.1 describes an “online gradient” view. Experiments verify near-exact numerical agreement on a toy linear-regression ICL setup.

### Strengths
The paper is easy to follow.

### Weaknesses
The main results reduce to short, fairly obvious linear-algebraic identities; proofs are a few lines and mostly restate the construction. Novelty and significance are limited; empirical evidence is restricted to a tiny, non-representative setting; and there are correctness/notation issues that need fixing, e.g.:
1. Unstated non-degeneracy needed for the update formulas. Both the main theorem and its residual variant divide by $\lVert A(C\setminus Y, x)\rVert^2$ and the corollary by $\lVert A(x)\rVert^2$ without assuming these are nonzero. The proofs also apply the identity $\frac{z^\top}{\lVert z\rVert^2} z = 1$, which requires $z\neq 0$. 
2. You present the update $\Delta W = \dfrac{(W\delta A_x(Y))A(C\setminus Y,x)^\top}{\lVert A(C\setminus Y,x)\rVert^2}$ as if unique. In fact, any $\Delta W + M \text{ with } MA(C\setminus Y,x)=0$ yields the same effect.

### Questions
see weaknesses.

### Soundness
1

### Presentation
2

### Contribution
1
