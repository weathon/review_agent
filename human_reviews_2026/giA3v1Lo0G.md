# Learning to Adapt: In-Context Learning Beyond Stationarity

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Transformer models have become foundational across a wide range of scientific and engineering domains due to their strong empirical performance. A key capability underlying their success is in-context learning (ICL): when presented with a short prompt from an unseen task, transformers can perform per-token and next-token predictions without any parameter updates. Recent theoretical efforts have begun to uncover the mechanisms behind this phenomenon, particularly in supervised regression settings. However, these analyses predominantly assume stationary task distributions, which overlook a broad class of real-world scenarios where the target function varies over time. In this work, we bridge this gap by providing a theoretical analysis of ICL under non-stationary regression problems. We study how the gated linear attention (GLA) mechanism adapts to evolving input-output relationships and rigorously characterize its advantages over standard linear attention in this dynamic setting. To model non-stationarity, we adopt a first-order autoregressive process and show that GLA achieves lower training and testing errors by adaptively modulating the influence of past inputs--effectively implementing a learnable recency bias. Our theoretical findings are further supported by empirical results, which validate the benefits of gating mechanisms in non-stationary ICL tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents a theoretical analysis of in-context learning (ICL) for non-stationary regression tasks. By leveraging the linearity of both the gated in-context learner and the underlying task, the authors derive closed-form results for the convergence of weight parameters and provide quantitative expressions for training and testing errors. These errors are explicitly characterized in terms of the forgetting factor $\lambda$, the task dynamics parameter $\gamma$, and the token length $n$. The framework also unifies linear attention and gated linear attention through a single parameterization by $\lambda$, enabling a coherent theoretical comparison.

### Strengths
The paper leverages the linear structure of both the gated ICL model and underlying tasks to obtain the convergence results and explicit error bounds. The dependence of these bounds on $\lambda$, $\gamma$, $n$, provides the meaningful insights for practical model design and parameter selection. Overall, I think this paper makes a valuable contribution, providing the theoretical understanding of how gated linear transformers for non-stationary linear setting.

### Weaknesses
While this paper is theoretically strong and sufficiently novel, it is currently limited to linear models and tasks.
However, I think that these limitations are acceptable for the scope of this paper. Out of interest, I would like to ask the questions below.

### Questions
Do you expect the analysis can be extended to nonlinear settings, such as softmax attention or nonlinear task mappings ? If so, what kind of interpretation would be obtained. Would it be similar interpretation  to the linear case ?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper provides a theoretical analysis of in-context learning (ICL) under non-stationary regression problems, where task distributions evolve over time. It demonstrates that Gated Linear Attention (GLA) outperforms standard linear attention by adaptively modulating past inputs through a learnable recency bias, enabling better handling of time-varying functions. Theoretical results and empirical validation confirm GLA's advantages in dynamic settings.

### Strengths
1. The paper introduces a novel theoretical framework for in-context learning (ICL) under non-stationary regression problems.
2.  The work exhibits certain technical rigor, with precise theoretical derivations.
3. The paper is well-structured, clearly defining the non-stationary ICL problem early and motivating it with real-world examples (e.g., time-series forecasting).

### Weaknesses
1. The theoretical analysis focuses exclusively on linear regression tasks with Gaussian inputs, which is a very narrow setting. Real-world in-context learning scenarios often involve non-linear functions, complex data distributions, and high-dimensional inputs.
2. The paper models non-stationarity using a first-order autoregressive process, which is a highly constrained form of time-varying dynamics. This fails to capture more complex real-world non-stationarity patterns such as abrupt changes, periodic variations, or long-term trends.
3. The paper briefly shows that deeper GLA models perform better but provides no theoretical justification or detailed analysis of how gating mechanisms interact across multiple layers in non-stationary settings.
4. While the paper includes a sentiment classification experiment on SST-2, this is only a single NLP task and doesn't demonstrate the broader applicability of GLA to diverse non-stationary scenarios.
5. While the paper briefly mentions adaptive signal processing methods in Section 3, it doesn't actually compare GLA's performance with established methods like RLS or LMS in the same non-stationary regression setting

### Questions
See the Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the authors study the capability of ICL to learn time-varying function class, thus provides a valuable addition to literature. In particular, the authors provide theoretical results on the convergence of the gradient flows, as well as training and testing errors. Numerical experiments are also provided for validation.

### Strengths
The authors provide rigorous theoretical research on the congruence of the gradient flows for ICL training, and quantify the training and testing errors. The analysis is correct and detailed.

### Weaknesses
1. The function class considered is limited. The analysis is restricted to the case in equation (3), it is desired to demonstrate the implications of the results for more broad classes of functions.
2. The numerical experiments are limited. Given strong assumptions made on initialization for the theoretically results, numerical experiments offer a good opportunity to explore what happens when those assumptions are violated.

### Questions
1. Can you provide comments on the behavior of the gradient flow when the initialization assumptions in both Assumption 1 and the Theorem 1 do not hold?
On Lines 146-147, e_i and \eta_i are not consistent.

### Soundness
3

### Presentation
3

### Contribution
3
