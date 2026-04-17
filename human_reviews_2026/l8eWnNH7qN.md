# Softmax-Induced Ill-Conditioning in Transformer Models

- Decision: Reject
- Scores: 2, 2, 2, 6

## Abstract
Transformers have become ubiquitous in modern machine learning applications, yet their training remains a challenging task often requiring extensive trial and error. Unlike previous architectures, transformers possess unique attention-based components, which can complicate the training process. The standard optimization algorithm, Gradient Descent, consistently underperforms in this context, underscoring the need for a deeper understanding of these difficulties. To understand this phenomenon, we analyze a simplified yet representative softmax attention model. Our local analysis of the gradient dynamics reveals that the Jacobian of the softmax function itself acts as a preconditioner. We show that when sufficiently many attention coefficients are small across multiple training examples, the Jacobian of the softmax becomes ill-conditioned, severely degrading the local curvature of the loss, which in turn slows the convergence of Gradient Descent. Our experiments confirm these theoretical findings on the critical impact of softmax on the dynamics of Gradient Descent.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper studies why purely gradient descent methods (e.g., SGD) underperform adaptive methods (e.g., Adam). Their theoretical results point towards the Softmax function, esp. when the embedding dimension is underparameterized relative to the data size. For this, they consider a simplified model with a single tunable token that must be trained through attention while keeping other parameters like $W_V$ (represented as a single value vector in the paper) frozen (Eq. 2). They connect attention sparsity to slower convergence, since the softmax function tends to lead to sparser solutions. The authors verify the aforementioned theoretical results experimentally.

### Strengths
* S1: The paper provides convergence rates for gradient descent under different settings (over/underparameterization, attention sparsity). Particularly, results for the underparameterized setting are novel.

* S2: The paper formally links attention sparsity with the conditioning number (Proposition 1, Theorem 2). It explains why SGD can have slower convergence rates.

* S3: The theoretical results are backed by experimental results (Fig. 1, 2) that study the effect of different attention sparsities and over/underparameterization.

### Weaknesses
* W1: The theoretical results neglect potential interaction effects between the value weights and the attention part. Additionally, the focus is only on CLS/last token attention. However, what happens if we have sparsity for other tokens that might be later attended to by the CLS/last token?

* W2: The paper is sometimes hard to follow because symbols are not introduced. Or, for example, Fig. 3 y-labels say attention ratio but the text writes entropy. Having consistent naming would help the clarity of the paper.

* W3: The paper is motivated by saying that gradient descent converges slower than Adam for transformer models (e.g., “Although these works highlight interesting features of transformer training that are different for Adam compared with SGD, their theoretical explanations of the source of these differences remain very limited” l. 45-47). However, there is no theoretical result that shows why and how adaptive methods like Adam would converge faster and its link to the attention sparsity/softmax. Thus, it appears the core questions remained unaddressed.

* W4: The results in Fig. 3 are not that convincing. Particularly, SGD and Adam appear quite comparable in the sparse setting (as their deviations overlap). Particularly, it’s unclear to me where we can see the stagnation that the authors describe in l. 471/472.

* W5: Empirically, it is known that sparse attention can lead to optimization problems (see C1). Thus, the finding is expected. Though, I acknowledge the theoretical derivation of this result. Thus, I only consider this a minor concern.

## Comment

* C1: It is worth adding [1] to the paper’s discussion since they discussed softmax-induced optimization problems, linked it to the attention’s sparsity and the effect on the softmax’s Jacobian, and suggested solutions that modified the softmax function.

* C2: It’s also worth connecting to [2-3] who discussed the role of rank collapse, i.e., high correlation in tokens cause vanishing gradients of queries and keys.

* C3: There seems to be a square error for the $H_i$ in l. 267.

--- 

[1] https://openreview.net/forum?id=HssOwuZiaB

[2] https://arxiv.org/abs/2103.03404 

[3] https://arxiv.org/abs/2206.03126

### Questions
* Q1: How’s R computed and regularized during training (l. 418-419)? Similarly, how does the teacher model work in the MNIST example?

* Q2: Is there a reason why the model with higher R (~= less sparse attention) yields a higher loss in Fig. 1 left?

* Q3: Why is there an initial training loss plateau for the less sparser model in Fig. 2a?

* Q4: Why does next-token prediction (necessarily) induce sparse attention? (l. 468/469)

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper aims to explain the challenges gradient descent (GD) faces when efficiently training transformers. The authors provide a theoretical analysis using simplified transformers based on tunable tokens, while keeping the attention weights fixed. In the overparameterized case, they demonstrate that the PL condition holds, leading to a linear convergence rate. In a local dynamics analysis, they show that when sufficiently many attention coefficients are small, the Jacobian of the softmax becomes ill-conditioned, leading to slow convergence. Experiments on synthetic data, MNIST, and WikiText are conducted to support the theory.

### Strengths
This paper gives a mechanistic explanation of why GD fails in transformers from the perspective of softmax-induced ill-conditioning. The authors also try to show that linear attention eliminates the slowdown.

### Weaknesses
- Assumptions: fixing q,k,v and training only one tunable token is too strict and, to me, far from practice.
- Section 3.1 seems less relevant to the title or main theme, since transformers in the real world do not fall in this regime.
- Writing: the paper is very hard to read, particularly Section 3.2.

### Questions
- What is R in your experiments? How does R depend on your other hyperparameters, including d and \sigma? It seems that you can control R by tuning other hyperparameters, but the relationship is unclear.
- In Figure 3, your SGD and Adam only exhibit a marginal gap, while in other papers like [1, 2], the gap is large. Could you clarify this difference?
- In Figure 2(a), although linear attention makes sparse attention converge faster, it does slow down the small R case. Could you theory explain it?
- How sparse is a language model during training in practice? Are there quantitative results? I suggest you should discuss about this in your main text.

[1] Linear Attention Is (Maybe) All You Need (to Understand Transformer Optimization). Ahn et al., 2024.
[2] Why Transformers Need Adam: A Hessian Perspective. Zhang et al., 2024.

### Soundness
2

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
The authors try to answer why GD underperforms for Transformer training through the analysis on the Jacobian of softmax and its condition number. In the underparameterized regime, they argue that the attention sparsity plays an important role in optimization behaviors.

### Strengths
- The paper provides theoretical and intuitive links between attention sparsity and optimization behaviors.
- It raises an important question about the commonly observed phenomenon that "Why does GD perform poorly on attention models?".

### Weaknesses
- Justification of tunable token $p$ is unclear. The tunable token setting is very different from usual training procedure. It may lead to a different dynamics. Are Fig 2,3 conducted based on the tunable token setting?
- Experimental results (Fig 1,2,3) are not strong enough to validate the theory (equations) and explain cause and effect.
    - Comparing two settings is not enough to confirm the theory. It would be better to test with a wide range of variations (e.g., many $R$'s or $\sigma_p^2$'s).
    - No experiments confirm exact equations (e.g., convergence rate) given in the paper. It would be better to have "theory vs experiment" comparison loss curves.
    - Can you directly control $R$?
    - Can you validate the theory with a larger model? 
    - Can you validate the theory by using different values of temperatures in the softmax function (as the temperature controls the attention sparsity)? 

- Two dots at the end of the sentence (L423)

- In Figure 3, it would be better to explain the different settings, (i) and (ii), of Top and Bottom.

- $C_p$ not shown in Theorem 1

### Questions
see weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper theoretically studies the convergence of gradient descent in a model consisting of a single softmax-attention layer with a single tunable token — a setting that resembles prompt tuning. The goal is to shed light on the question of why attention-based models do not train well with (S)GD. The paper first proves the linear convergence rate of gradient descent in this setting in the overparametrized regime and argues that this setting does not capture well what happens in transformer models. Then it proceeds to studying the dynamics of the gradient descent over parameters in a probability simplex of attention scores around the minimum, concluding with a result that the problem is badly conditioned for sparse attention matrices.

### Strengths
1. The paper studies a question that is important for the Transformer optimization community in a very interesting setting that allows for a closed-form convergence analysis.
2. The idea to study the convergence dynamics in the attention-scores probability simplex seems original.
3. Insights from the paper can have practical implications on how we optimize Transformers. The results suggest that if we have a task that we think won’t need sparse attention matrices, we can successfully fit the model with lightweight SGD instead of adaptive methods. 
4. The preliminary experiments with actual Transformer models seem to confirm the insights from theory.

### Weaknesses
I’m willing to increase my partial and final scores if the authors address the following weaknesses: 
1. A detailed section on the experimental setup is missing. What kind of transformer model was trained on WixiText-2? I can see the shading in the figures so I assume the results are averaged across runs but I don’t know over how many. How were the hyperparameters tuned? It would be great if the authors could describe the experiment details at least in the appendix. 
2. In Figure 3 there is no indication on which row corresponds to which experiments. 
3. The assumptions made (especially Assumption B) are not experimentally verified in real models, nor are there references to literature demonstrating whether such assumptions should hold. I don’t think it is necessarily for them to hold exactly to accept the paper, but I think that the paper is not complete without evaluating assumptions validity.
4. The authors motivate their setup with the prompt tuning, yet there are no experiments in this setting. 
5. The discussion of previous hypotheses on why transformers train better with adaptive methods than gradient descent is not exhaustive. Specifically, the authors do not refer to [1].
6. Proof of Proposition 1 is missing
7. There are some typos and formatting issues in the proofs (that I believe can be easily fixed). For example:
    1. In line 726-727 there should be $diag(z)\mathbb{1} - zz^\top\mathbb{1}$
    2. V_i in line 1050 is a function of z, yet z does not appear in its definition. Then the V_i(z) appears a couple more times in the derivations, although the function does not depend on z, nor z is defined anywhere.
    3. Formatting of the equation in lines 880-885 is broken.

[1] Zhao et al., Deconstructing What Makes a Good Optimizer for Autoregressive Language Models

### Questions
Questions about experimental results:
1. Why should we expect Assumption B to hold in real Transformer models? Can the authors provide experimental evidence for the validity of these assumptions in deep Transformers?
2. Can the authors provide experimental evidence for their claim on the influence of attention sparsity on GD/Adam performance in the prompt tuning setting?
3. How exactly were optimizer hyperparameters tuned to obtain results from Figure 3?

Questions about theory and proofs:
1. How does one partition the tokens into an important and unimportant sets needed for Lemma 3?
2. Can the authors explain more in detail where the eq. 39 comes from?
3. Why are the authors discarding three last terms from the equation in lines 1102-1104 while formulating the equation from line 42? If there is a reason why it is okay to do it here, the authors should explain that reason.

Questions about context/relation to previous work:
1. Can the authors comment more on how their result relates to the Hessian analyses from previous work ([2], [3]) of Transformer and self-attention? Specifically, can the authors draw a direct link between their analysis in the attention simplex and the attention moments from [3]? 
2. Can the authors comment on how the result from the paper relate to the previous work ([1]) that showed that adaptability is needed only for the last layer and layer norms in Transformers?

[1] Zhao et al., Deconstructing What Makes a Good Optimizer for Autoregressive Language Models

[2] Zhang et al., Why Transformers Need Adam: A Hessian Perspective

[3] Ormaniec et al., What Does It Mean to Be a Transformer? Insights from a Theoretical Hessian Analysis

### Soundness
2

### Presentation
2

### Contribution
3
