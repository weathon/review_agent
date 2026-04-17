# In-Context Algorithm Emulation in Fixed-Weight Transformers

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 2

## Abstract
We prove that a minimal Transformer with frozen weights emulates a broad class of algorithms by in-context prompting. 
We formalize two modes of in-context algorithm emulation.
In the *task-specific mode*, for any continuously differentiable function $f: \mathbb{R} \to \mathbb{R}$, we construct a single-head softmax attention layer whose forward pass reproduces functions of the form $f(w^\top x - y)$ to arbitrary precision.
This general template subsumes many popular machine learning algorithms (e.g., gradient descent, linear regression, ridge regression).
In the *prompt-programmable mode*, we prove universality: 
a single fixed-weight two-layer softmax attention module emulates all algorithms from the task-specific class (i.e., each implementable by a single softmax attention) via only prompting.
Our key idea is to construct prompts that encode an algorithm’s parameters into token representations, creating sharp dot-product gaps that force the softmax attention to follow the intended computation.
This construction requires no feed-forward layers and no parameter updates. 
All adaptation happens through the
prompt alone. 
Numerical results corroborate our theory.
These findings forge a direct link between in-context learning and algorithmic emulation, and offer a simple mechanism for large Transformers to serve as prompt-programmable interpreters of algorithms. 
They illuminate how GPT-style foundation models may swap algorithms via prompts alone, and establish a form of algorithmic universality in modern Transformer models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work provides a theoretical framework that **frozen softmax attention modules** within minimal Transformer architectures can emulate a wide class of algorithms purely through **prompt programming**. The authors formalize two emulation modes — *task-specific* and *prompt-programmable*. In the **task-specific mode**, they show that a single-head softmax attention layer with a linear map can approximate functions of the general form $f(w^\top x - y)x$ for any continuous $f:\mathbb{R}\to\mathbb{R}$ and arbitrary precision $\epsilon>0$. This includes key algorithmic operations such as per-sample gradient computation, single-step and multi-step gradient descent, linear regression, and ridge regression. The main theoretical statement, **Theorem 3.1**, asserts that for any bounded input $X\in\mathbb{R}^{(d+1)\times n}$ and coefficient vector $w\in\mathbb{R}^d$, there exists a single-head attention $Attn_s$ and linear map $Linear$ to achieve the desired precision.

From this, they derive **Corollary 3.1.1**, which shows that with $f(t) = \ell'(t)$ for differentiable loss $\ell$, the layer emulates per-sample gradients $\ell'(w^\top x_i - y_i)x_i$, and **Corollary 3.1.2** extends this to one-step gradient descent: for $\eta>0$, there exists an attention map producing the respective approximation of the $w^{+}_{GD}$.

Multi layer extensions show that stacking $(L+1)$ single-head attention layers yields $L$ gradient descent steps with bounded cumulative error. Similarly, **Corollaries 3.1.3** and **3.1.4** construct attention-based emulations for linear and ridge regression respectively.

In the **prompt-programmable mode**, the paper’s main result (**Theorem 4.1**) proves that a two-layer softmax attention module with fixed weights can emulate any single attention head by embedding its weights $(W_K, W_Q, W_V)$ into the prompt. Specifically, defining the prompt as
$$X_p = \begin{bmatrix} X \ W_{\text{in}} \ I_n \end{bmatrix}, \quad W_{\text{in}} = \begin{bmatrix} 0\cdot w & 1\cdot w & \cdots & (n-1)\cdot w \ w & w & \cdots & w \end{bmatrix},$$
where $w = [\text{vec}(W_K);\text{vec}(W_Q);\text{vec}(W_V)]$, they prove
$$|Attn_s \circ Attn_m(X_p) - W_V X \operatorname{Softmax}((W_KX)^\top W_QX)|_\infty \leq \epsilon.$$
Thus, a **two-layer frozen Transformer** is *universal* for all algorithms implementable by single-layer attention. **Theorem 4.2** provides an equivalent formulation, showing that a single-head attention followed by a multi-head layer with linear projections approximates the same target attention mapping, implying that prompt-programmed softmax attention retains permutation equivariance and can simulate any bounded attention mechanism. **Corollary 4.2.1** extends this to a library $\mathcal{A}_0 = {a_1,\ldots,a_k}$ of algorithms, proving that one fixed two-layer module can emulate every $a\in\mathcal{A}_0$ 

Finally, the authors generalize this to arbitrary linear networks, showing that any trainable linear map $f(x)=\Theta x$ can be emulated by encoding $\Theta$ in the prompt — turning static networks into in-context learners that execute dynamically programmable linear mappings.

Empirical studies (Figures 1–2, Table 1) confirm that softmax attention layers approximate $f(w^\top x - y)x$ and even emulate attention heads with mean squared error decreasing roughly as $O(1/H)$ with the number of heads $H$. The findings demonstrate that fixed-weight softmax attention networks can serve as **prompt-programmable algorithmic evaluators**.

### Strengths
1. The extension of previous works to propose a framework adaptable to a broader class of algorithms via the prompt-programmability with the fixed attention module transformer is a strong contribution to the community, especially to the line of research focused on guarantees for prompting based approaches. 

2. I believe the emphasis in lines 361-363 about - “our results are constructive, providing concrete emulation examples in contrast to prior prompting expressivity (Wang et al., 2023; Furuya et al., 2024) or Turing-completeness results ...” - is a strong contribution as well.

### Weaknesses
1. I have a concern about the final bound discussed in lines 1704 to 1727. Specifically, for the value of $\epsilon$ that will be attained by plugging the values of $B_f$ and $B$, the bounds for $\epsilon_1$ and $\epsilon_2$ need to be quite tight. Can the authors discuss this in a bit more detail? I think this is an important characterization. 

2. I have  a serious question about experimental section 5.1 - since the data has been generated synthetically and the parameters ‘W’ as well as the labels ‘y’ are known, I understand that the task should be use the construction in the proof of theorem 3.1 to adjust the input as well as the attention weight matrices appropriately to observe the approximation error. Why was there a need to use an optimizer to perform explicit training? And even if the training has been performed using the optimizer, a direct check is to compute the difference between the learned attention layer weights and the weights generated explicitly in lines 1578-1585 of the proof – although I understand this will likely fail due to issues in optimization and convergence to the other minima. Open to any other suggestions from the authors and looking forward to the clarification in case I mis-understood something here.

### Questions
1. The use of ‘L’ seems overloaded at times. What exactly are ‘L’ and ‘P’ in the proof of theorem D.1 in appendix, initiated at lines 1553-54 of appendix in the definition of $\textbf{X}$ ? Do they follow the same initialization as in lines 2040-2050 on page 38 in the proof of theorem D.6? 

2. I am curious if the authors have thought about the extensions to the cases with positional encoding (some form of absolute/relative positional encodings), instead of the Identify pos encoding used in the proofs, along with the default input $\textbf{X}$. Please note that this question DOES NOT reflect my score, and I am just curious, as the authors have put in much effort to characterize the in-context setup, whether there are any thoughts on the positional encoding case. I understand some forms of positional encoding will break equivariance as well, but irrespective of this, what is your opinion on whether positional encodings will induce any extra complications in the proof sketch if any?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the expressive power of the transformer architecture (especially a single attention layer) in the in-context learning (ICL) setting. It shows that (i) a single attention layer can implement the map $[x; y; w] \mapsto f(x^\top w - y)x$, extending much of the existing work on the ICL expressivity of transformers; and (ii) Transformers can express the functorial map $(X, A) \mapsto A(X)$, where $A$ is an attention layer specified by its weights, i.e., there exists a fixed-weight transformer that, given a prompt containing data X and a "program" A, executes the program and outputs A(X), provided A is represented in terms of attention weights.

### Strengths
The paper is well written, and the intuition is explained in detail. The theoretical results are strong and extend several prior papers on the ICL capabilities of Transformer architectures.

### Weaknesses
(1) The innovation over Hu et al. (2025) should be discussed more comprehensively. For the proof of Theorem 3.1, the ideas appear to follow Hu et al. (2025), albeit with substantial technical development. It would be beneficial to clarify which ideas/techniques are inherited from prior work and what are new here.

(2) The dimension of the linear layer is not stated explicitly in the theorems. Providing an explicit bound on this dimension (e.g., in Theorem 3.1, in terms of regularity conditions on $f$) would help readers understand the expressivity at bounded width and would be useful for establishing generalization bounds.

I will adjust my score provided the issues above are addressed.

### Questions
(1) Section 4.1 shows how transformer can express $(X, A) \mapsto A(X)$ when $A$ is described by attention weights. Could a stronger result show that a transformer can express $(X, A) \mapsto A(X)$ for circuits/Turing machines A subject to structural constraints (e.g., bounded depth)? Does the construction extend to this setting?

(2) Corollary 3.1.3 and 3.1.4 are stated for single-layer attention, but their proof seems to be based on stacking attention attention layers that are implementing GD.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the capability of fixed-weight Transformer architectures to emulate a broad class of algorithms through in-context prompting, addressing a core question in Transformer research: how frozen models execute diverse tasks via context alone. The authors formalize two modes of in-context algorithm emulation:

1. Task-Specific Mode: A single-head softmax attention layer (with a linear map) can universally approximate functions of the form $f(w^\mathrm{T}x-y)$ (for any continuous $f$) to arbitrary precision. This template includes key machine learning algorithms like gradient descent (GD), linear regression, and ridge regression.
2. Prompt-Programmable Mode: A single fixed-weight two-layer softmax attention module achieves universality—emulating all task-specific algorithms via prompt design alone. The key mechanism is encoding algorithm parameters into prompts to create sharp dot-product gaps, forcing softmax attention to follow intended computations without feed-forward layers or parameter updates.
The authors validate their theory numerically: synthetic experiments confirm accurate approximation of continuous functions, with error decreasing as the number of attention heads increases; real-world tests on the Ames Housing dataset show the frozen module matches performance of task-specific baselines.

In conclusion, this paper show a minimalist transformer architecture serve as a general-purpose algorithm emulator in context through prompt design.

### Strengths
- Foundational Theoretical Value: The formalization of two emulation modes and universal approximation results establish a rigorous basis for viewing ICL as "in-context algorithm emulation," addressing open questions about fixed-weight Transformer flexibility.

- Interpretability: Unlike black-box ICL studies, the paper provides a clear mechanism (prompt-encoded parameters + softmax routing) for how frozen models execute algorithms—enabling future work on principled prompt engineering.

- Broad Algorithmic Coverage: The task-specific template unifies diverse algorithms (GD, linear/ridge regression), showing that a single attention layer can capture core ML workflows.

### Weaknesses
- Prompt Scalability: The prompt length grows linearly with the weight dimension of the target algorithm (Section 6, Limitations). This limits practicality for high-dimensional algorithms (e.g., deep neural network training), as prompts could become prohibitively long.

- Lack of Comparison to Prompt-Tuning: The paper focuses on hand-crafted prompts but does not compare to learned prompt-tuning methods (e.g., Lester et al., 2021). It is unclear how hand-crafted prompts perform relative to learned prompts for complex algorithms.

- Limited Algorithm Diversity: The empirical validation focuses on regression and GD—algorithms with clear linear/gradient-based structures. It is unknown if the framework extends to non-linear algorithms (e.g., decision trees, k-means) or sequential tasks (e.g., sorting).

- No Language/Vision Extension: The paper tests only tabular/structured data. It is unclear if the prompt-programmable framework applies to language (e.g., few-shot text classification) or vision (e.g., in-context image segmentation)—domains where ICL is widely used.

### Questions
- Prompt Compression: Given that prompt length scales with weight dimension, have you explored methods to compress prompt-encoded parameters (e.g., low-rank approximation, quantization)? Would compression preserve the sharp dot-product gaps needed for accurate emulation?

- Non-Linear Algorithms: Your framework excels at linear/gradient-based algorithms, but how would it handle non-linear algorithms (e.g., logistic regression with sigmoid loss, or k-means clustering)? Do you need to modify the prompt structure to capture non-linearities, or does the continuous function $f$ in Theorem 3.1 suffice?

- Feed-Forward Layers: You omit feed-forward layers in your constructions. Would adding feed-forward layers improve emulation accuracy for complex algorithms, or does this break the "minimal architecture" claim? Do feed-forward layers introduce interference with prompt-encoded parameters?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
think this paper makes only a marginal contribution over prior work (e.g., [1], [2]). The authors show how a single attention layer can emulate, in context, a broad class of algorithms (with an additional linear layer on the input) — specifically, any function of the form $f(w^Tx-y)x$. As special cases, they derive linear regression, (stochastic) gradient descent, and ridge regression. However, all these results have already been demonstrated in previous works [1], [2], [3], [4] etc.
They further show that attention itself can be emulated in context using two attention layers. One question that arises is why one would want to emulate attention in context when direct access to the attention mechanism is already available. This part mainly illustrates that any attention layer can, in principle, be emulated in context by a fixed-weight transformer by defining appropriate prompts(prompt-specific mode).

[1]: Bai, Yu, et al. "Transformers as statisticians: Provable in-context learning with in-context algorithm selection." Advances in neural information processing systems 36 (2023): 57125-57211.

[2]: Giannou, A., Rajput, S., Sohn, J., Lee, K., Lee, J.D. &amp; Papailiopoulos, D.. (2023). Looped Transformers as Programmable Computers. 

[3]: Von Oswald, Johannes, et al. "Transformers learn in-context by gradient descent." International Conference on Machine Learning. PMLR, 2023.

[4]: Giannou, Angeliki, et al. "How Well Can Transformers Emulate In-context Newton's Method?." arXiv preprint arXiv:2403.03183 (2024).

### Strengths
1. The distinction between task-specific and prompt-programmable emulation is a useful conceptual framing.

2. The coverage of residual update forms $f(w^Tx-y)x$ is broad and subsumes many standard learning procedures.

3. The extension from emulating a single-layer attention mechanism to emulating full (linear) networks is conceptually compelling.

### Weaknesses
1. Most (if not all) of the results presented in this paper have already been proved in previous works, possibly using deeper—but still fixed—architectures. Results on linear regression have appeared in [1], [2], [3], and those on ridge regression in [1], [4] (using softmax attention instead of linear attention introduces only a small approximation error).

In-context algorithm selection was also demonstrated in [1] and [4]—not explicitly, but implicitly through the design of pointer mechanisms that determine which function is executed in context.

Overall, the contribution of this paper is limited.

[1]: Bai, Yu, et al. "Transformers as statisticians: Provable in-context learning with in-context algorithm selection." Advances in neural information processing systems 36 (2023): 57125-57211.

[2]:Von Oswald, Johannes, et al. "Transformers learn in-context by gradient descent." International Conference on Machine Learning. PMLR, 2023.

[3]: Giannou, Angeliki, et al. "How Well Can Transformers Emulate In-context Newton's Method?." arXiv preprint arXiv:2403.03183 (2024).

[4]:  Giannou, A., Rajput, S., Sohn, J., Lee, K., Lee, J.D. &amp; Papailiopoulos, D.. (2023). Looped Transformers as Programmable Computers.

### Questions
1. Could the authors clarify the motivation or potential use cases for emulating attention using attention itself? For example, are there theoretical insights or architectural benefits (e.g., modularity, composability) that justify this construction?

2. Can the authors clarify whether their constructions differ in expressive power or generality from prior results (e.g., in [1], [2])? Are there cases where their framework captures algorithms that earlier works could not? (Even if the layers are less -- it is important to have a comparison of constant vs logarithmic for example). 

3. How does the proposed notion of “prompt-programmable emulation” differ technically from “in-context algorithm selection” in previous work? Is there a formal distinction or new capability demonstrated here?

4. If the authors believe the contribution lies in unifying or simplifying prior results, could they argue this explicitly and explain how this framework might enable new theoretical or practical insights?

### Soundness
3

### Presentation
3

### Contribution
2
