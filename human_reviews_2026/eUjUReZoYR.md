# Learning under Quantization for High-Dimensional Linear Regression

- Decision: Accept (Poster)
- Scores: 8, 2, 6, 4

## Abstract
The use of low-bit quantization has emerged as an indispensable technique for enabling the efficient training of large-scale models. Despite its widespread empirical success, a rigorous theoretical understanding of its impact on learning performance remains notably absent, even in the simplest linear regression setting. We present the first systematic theoretical study of this fundamental question, analyzing finite-step stochastic gradient descent (SGD) for high-dimensional linear regression under a comprehensive range of quantization targets: data, label, parameter, activation, and gradient. Our novel analytical framework establishes precise algorithm-dependent and data-dependent excess risk bounds that characterize how different quantization affects learning: parameter, activation, and gradient quantization amplify noise during training; data quantization distorts the data spectrum and introduces additional approximation error. Crucially, we distinguish the effects of two quantization schemes: we prove that for additive quantization (with constant quantization steps), the noise amplification benefits from a suppression effect scaled by the batch size, while multiplicative quantization (with input-dependent quantization steps) largely preserves the spectral structure, thereby reducing the spectral distortion. Furthermore, under common polynomial-decay data spectra, we quantitatively compare the risks of multiplicative and additive quantization, drawing a parallel to the comparison between FP and integer quantization methods. Our theory provides a powerful lens to characterize how quantization shapes the learning dynamics of optimization algorithms, paving the way to further explore learning theory under practical hardware constraints.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents a systematic theoretical analysis of the generalization performance of quantized Stochastic Gradient Descent (SGD) in the context of high-dimensional linear regression. It studies the effects of quantizing five key components: data, labels, parameters, activations, and output gradients. The main result is a precise, decomposed bound on the excess risk, which clearly separates the approximation error, bias error, variance error, and the unique quantization error. The authors contrast two major quantization error models: additive (related to INT) and multiplicative (related to FP). They use their bounds to provide conditions under which quantized SGD achieves performance comparable to its full-precision counterpart.

### Strengths
The paper tackles an important and novel theoretical problem with significant practical implications. The systematic analysis of five quantization targets is comprehensive. The derived generalization bounds for quantized SGD are the first of their kind to my knowledge. The comparison between additive and multiplicative quantization, particularly the insight into spectral distortion, is a major contribution. The paper is well-written, and the theoretical results are intuitively explained and justified.

### Weaknesses
1. The analysis is restricted to high-dimensional linear regression, leaving the extension to non-linear models (like neural networks) as a critical open question. 
2. More significantly, the paper provides only an upper bound on the excess risk. As noted, this bound, especially for additive quantization (scaling with d and N), appears non-tight and is not compared directly against the numerical simulations, undermining the conclusiveness of the theoretical comparison between additive and multiplicative models. Proving a matching lower bound or providing a tighter analysis for the additive case is necessary to solidify the theoretical claims.

Overall, the paper makes a foundational contribution to the theoretical understanding of quantization's impact on learning performance. The novel bounds and the systematic approach to different quantization targets (data, parameters, etc.) provide invaluable, actionable theoretical guidance for the design of efficient learning systems. While the lack of a tighter lower bound and the restriction to the linear model are limitations, the complexity and significance of the problem tackled, combined with the depth of the current analysis, make this work a major theoretical advance worthy of publication. The authors should be encouraged to pursue a tighter analysis of the additive and multiplicative quantization errors in the revision.

### Questions
1.	Can the author comment on what they think would be the effect of the quantization error in nonlinear models and other algorithm designs, such as multipass SGD and momentum? Do they think that the implication of the result regarding multiplicative quantization will also carry in these settings? 
2.	What does the notation $≲$ mean? 
3.	Lines 355-361: Please indicate where the fourth term appears. It would be helpful to provide a proof sketch that explains all the pieces in the proof. I understand the analysis is similar to Zou et al (2023). However, it would be nice to have a summary here.  
4.	Line 336: Is it clear that these errors remain O(1)? Please provide more intuition on why, rather than just directing to that appendix, as this is central to the analysis. 
5.	Lines 393-397: I am not sure I understand this. The errors are not completely decoupled, as $\epsilon_a$, $\epsilon_o$, depend on $\epsilon_d$, $\epsilon_p$. 
6.	Line 463: It would be nice to provide a clearer explanation and maybe plot the Hessian spectrum as well in both cases, to illustrate the effect.  
Minor: 
1.	Line 293 remove either “plays” or “relies”
2.	Line 394 \epsilon_0 should be \epsilon _o.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper provides a theoretical analysis of Linear Regression with diverse quantization targets: data, labels, parameters, activations, and gradients. The paper analyzes two types of unbiased compression methods: multiplicative and additive quantization, and derives convergence guarantees for SGD under these schemes. The paper also includes a small empirical evaluation on synthetic data.

### Strengths
* The paper brings a theoretical analysis of different quantization targets.
* The definition, assumptions, and notations are easy to follow.

### Weaknesses
* The statement of Theorem 4.1 is difficult to analyze and to derive guidance on what to quantize to maximize performance within a limited resource budget; the paper would benefit from organizing and grouping the results and from plugging in an optimal learning rate.
* The bounds in Theorem 4.1 do not improve with the number of samples $N$, which is atypical for SGD analyses (even in more general convex or non-convex settings). This issue remains even without compression (i.e., when $\varepsilon=0)$.
* The paper uses a simple setup of unbiased compressions (where biased compressions are more practical and common) over a simple linear regression objective for empirical risk minimization (i.e., not guaranteeing generalization).
* Empirical results are limited to a simple synthetic dataset.

### Questions
* It is not clear why quantizing data would outperform quantizing parameters or gradients. Could the authors provide a concrete example or theoretical justification where data quantization provably yields better results than parameter/gradient quantization?
* Regarding the quantization of the model weights in the (Quantized SGD) update step. Wouldn’t it make more sense to quantize both $w_{t-1}$? If only one of them is quantized, it seems we still need to maintain a full-precision copy of the parameters, which weakens the memory motivation.

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
3

### Summary
The paper provides theoretical foundation for understanding how quantization impacts generalization in SGD. The authors bridge the gap between optimization convergence analyses and generalization theory under hardware-limited precision. The paper specifies precise conditions on quantization errors needed for quantized SGD to match the learning performance of full-precision SGD. The authors also show an analytical framework that quantifies how quantization on various components.

### Strengths
1. The authors provide a unified framework that decomposes the excess risk of quantized SGD into interpretable components.
2. The paper is mathematically solid and demonstrates excess risk bounds explicitly with clear decomposition and scaling behavior under both additive and multiplicative quantization.
3. The authors provide interesting insights into precision and generalization trade off, and also provide theoretical link between quantization type, like FP vs. INT, can be beneficial for scaling law and safe precision reduction.

### Weaknesses
1. The experimental section is only on synthetic Gaussian data, it could be more convincing if validated on real world dataset.
2. The analysis relies on idealized assumptions, such as unbiased stochastic quantization, but these may not hold in practical low-precision systems. Discussion or relaxation of these assumptions would strengthen the generality.
3. While motivated by scaling-law literature, the link between derived quantization effects and empirical scaling behaviors remains largely qualitative.

### Questions
1. How do non-uniform or mixed precision quantization methods fit into the framework here?
2. Can it be generalized to multiple epochs and adaptive optimizers, would the excess risk decomposition change qualitatively?
3. Since the case study is on polynomials decaying, I am curious how would the results change for exponentially decaying or heavy tail spectra which also happen in real world embeddings.
4. In Fig 1, how sensitive are the results to batch size or step size, can authors also show some results other than B=1?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper develops a theoretical framework for understanding how low-precision quantization affects the learning performance of stochastic algorithm descent (SGD) for the problem of high-dimensional linear regression. This paper specifically focuses on the impact of quantization on the generalization (population) risk (not just the training loss/sample risk). Quantization is applied to five components of training: Data (both features and labels), model parameters, activations and gradients. Subsequently, the iterate-averaged SGD is analyzed using two quantization schemes:

1. Additive quantization, where the absolute quantization error is constant -- capturing the effect of fixed-point precision formats (e.g., INT8).

2. Multiplicative quantization, where the relative quantization error is constant (in other words, quantization error scales with the magnitude of the quantization input) -- capturing the effect of floating-point precision formats (e.g., FP8).

The key contributions of the paper include:

1. Derivation of explicit bounds on the generalization risk showing contributions from variance and bias (which are prevalent even for full-precision SGD, i.e., no quantization error), approximation error from the quantization of data/labels, and additional quantization error, arising from accumulation of quantization error from the SGD iterates.

2. Study of how the distortion is input data spectrum due to quantizing the features/labels affects the generalization risk. More specifically, the authors propose that multiplication quantization does not lead to significant spectral distortion and matches the sample complexity of full-precision SGD under mild conditions. On the other hand, for additive quantization, spectral distortion increases with dimension, and leads to worse sample complexity. 

3. The work also considers a polynomial eigenvalue decay of the data spectrum, and proposes when additive vs. multiplicative quantization is preferred on the other.

### Strengths
The author positions this work in the context of LLMs and low-precision training, precision-scaling laws, and benign overfitting and overparameterization theory -- which is pretty relevant in today's ML landscape where bigger models and more data are preferred for better performance. The paper tackles a pretty fundamental question: How does quantization affect the generation performance of SGD for linear regression, in contrast to prior works that focus on convergence of optimization algorithms under quantization. The systematic study of quantizing data, label, parameter, activation and gradient is a major contribution as well -- prior works usually isolate one or two of these. Modeling quantization schemes as multiplicative or additive also provides an useful abstraction between this work's theoretical analysis and the data formats generally used in practice (e.g., FP8 and INT8).

### Weaknesses
Firstly, I have a semi-major concern: The paper implicitly assumes quantization is purely detrimental and focuses on bounded degradation relative to full precision. But quantization introduces stochasticity that also plays a role analogous to implicit regularization similar to SGD noise, weight-decay, etc. (Ref: https://arxiv.org/abs/2101.12176). There are also some empirical works in deep-learning where low-precision improves generalization slightly (e.g., https://arxiv.org/abs/2206.12372). The theory for linear regression is exactly the setting where we would want to understand whether/when this can occur. In my opinion, this is a key conceptual gap, which is not apparent from the main body of the paper (pl. correct me if I have missed anything). It would be highly appreciated if the authors could comment on the following:

1. Discuss whether their analysis explicitly take into account this implicit regularization? If yes, where exactly? If not, is it a limitation of the current analysis, or is studying the implicit regularization not relevant?
2. If the authors agree with the potential benign regularization effects of quantization, under what conditions might they emerge?

Secondly, I think the writing of the paper can be significantly improved. Currently, the theorems statements in the main paper have pretty long expressions. If it is possible to state simplified/informal versions of the theorem and/or corollaries wherever relevant to extract the essence of the theorem, I think it would make the paper highly approachable.

Real world linear regression experiments would also be appreciated, where the assumptions on the eigen-decay of the data matrix are justified. I believe this is important because the section on **Implications to integer and FP quantization** is quite relevant to practitioners in determining the choice of precision formats to be used for quantization. However, in practice, INT8 and FP8 perform for or less similar to each other (perhaps because for most recent models, dimension is high?) Sometimes, INT8 is even preferred because INT8 matmuls are faster than FP8 matmuls.

### Questions
I have a few questions/suggestions are would appreciate it if they are addressed:

1. Why is $Q_o$ referred to as the quantizer for *output gradient*? Shouldn't the input to $Q_o$ also include the term $Q_d(\mathbf{X}_t)^\top$, and not just the second factor?

2. Recent techniques for quantized SGD like error feedback (https://arxiv.org/abs/1909.05350) are not considered in the analysis. This somewhat limits the applicability of the results, since error feedback has been around for a few years now and is pretty widely used for optimization with compressed gradients. This should be explicitly highlighted in the limitations section.

### Soundness
2

### Presentation
2

### Contribution
2
