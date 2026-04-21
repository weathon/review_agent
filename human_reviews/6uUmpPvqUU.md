# The Closeness of In-Context Learning and Weight Shifting for Softmax Regression

- Avg Score: 5.00
- Decision: Reject
- Scores: 5, 5, 5

## Abstract
Large language models (LLMs) are known for their exceptional performance in natural language processing, making them highly effective in many human life-related tasks. The attention mechanism in the Transformer architecture is a critical component of LLMs, as it allows the model to selectively focus on specific input parts. The softmax unit, which is a key part of the attention mechanism, normalizes the attention scores. Hence, the performance of LLMs in various NLP tasks depends significantly on the crucial role played by the attention mechanism with the softmax unit. 

In-context learning is one of the celebrated abilities of recent LLMs. Without further parameter updates, Transformers can learn to predict based on few in-context examples. However, the reason why Transformers becomes in-context learners is not well understood. Recently, in-context learning has been studied from a mathematical perspective with simplified linear self-attention without softmax unit. Based on a linear regression formulation $ \min_x \|  Ax  - b \|_2 $,
existing works show linear Transformers' capability of learning linear functions in context. The capability of Transformers with softmax unit approaching full Transformers, however, remains unexplored. 

In this work, we study the in-context learning based on a softmax regression formulation $ \min_{x} \| \langle \exp(Ax), {\bf 1}_n \rangle^{-1} \exp(Ax) - b \|_2 $. We show the upper bounds of the data transformations induced by a single self-attention layer with softmax unit and by gradient-descent on a $ \ell_2 $ regression loss for softmax prediction function. Our theoretical results imply that when training self-attention-only Transformers for fundamental regression tasks, the models learned by gradient-descent and Transformers show great similarity.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors aim at improving our understanding of in-context learning
from a theoretical perspective. Previous work has proved that
a simplified self-attention layer can "in-context learn" the gradient step of a linear
regression. The authors propose to show the same in the case
of a softmax regression, which they propose as an intermediate step between
the linear regression and the actual operation done by self-attention.
The appendix contains empirical results that compare self-attention
to softmax regression and that corroborate the theoretical findings.

### Strengths
1. The problem considered, i.e. a theoretical understanding of
  in-context learning is significant with the rise of LLMs.

2. The abstract is well-written.

3. The review of previous work positions the paper well and makes clear what are the novel contributions.

4. In the appendix there is an empirical verification that a softmax regression model and a single SA layer are similar on one gradient descent step. The empirical approach seems sound as it follows previous work.

### Weaknesses
1. I had quite a bit of trouble reading the paper. I was unable to fill
  quite a few logical steps that I deem significant. I have left questions
  regarding them.

### Questions
My initial rating inclines towards rejection (with low confidence in my assessment): upon reading the paper I am missing a few logical steps that I deem significant. I have left questions regarding these; if you could clarify them it would greatly help me to improve my assessment of the
paper.

**Major Questions**:
  1. Definition 1.3: I miss the motivation for why this would advance
    the understanding of in-context learning for the Transformer.
    It seems that the problem solved by Self-Attention would involve the
    matrix A quadratically in the exponential, while here A appears
    linearly in the exponential. Could you elaborate on why this intermediate step is useful for analyzing
    what would happen in the Transformer?

  1. Upon reading the text a few times, I still do not understand why
    the bounds of **Thm 5.1** and **Thm 5.2**  would imply that the
    transformation induced by the layer would approximate the gradient step, or if I understood **Oswald et. al** correctly, that at least there is 
    a choice of layer parameters that would make it approximate the gradient step.

**Minor Questions**:
* (page 2, first equation): I do not  understand why one needs to introduce a generalized attention formulation if the considered problem is quite simplified.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work studies the in-context learning based on a softmax regression mostly approaching the vanilla self-attention and gives the upper bounds of the data transformations driven by gradient descent for a single self-attention layer. 
Nevertheless, the paper's structure appears to lack the necessary depth to fully elucidate critical findings, such as the significance of their contributions to advancing our understanding of in-context learning beyond existing literature.

### Strengths
This work examined the in-context learning process based on a softmax regression, aiming to  illustrating Transformer’s attention mechanism.

### Weaknesses
1. The structure appears insufficient to fully elucidate the internal mechanisms of in-context learning relying on a single self-attention layer.
2. The significance of the findings, such as the upper bounds of data transformation, is somewhat understated, rendering them supplementary to prior research.
3. Certain mathematical proofs and deductions may benefit from a more concise presentation, potentially relocating them to the appendix, while experimental results could find a more prominent place in the main paper.

### Questions
Could you consider reorganizing the paper to enhance its comprehensiveness and clarity? One suggestion is to separate the model definitions and theorems from the introductory section for better clarity.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper is about clarifying the relationship between in-context learning in LLMs and weight shifting for softmax regression.
The paper tries to understand in-context learning of Transformer models, specifically self-attention, in the perspective of softmax regression.
The optimization of the attention module could be seen as the following softmax regression problem: $\min_{X \in \mathbb{R}^{d\times d}} \lVert D^{-1} \exp ( AXA^\top ) -B \rVert_F $ where $A \in \mathbb{R}^{n \times d}$ is a matrix for document having length $n$ and embedding size $d$, $X$ a weight matrix, and $B$ the target distribution for the probabilities resulting from softmax.
Beyond prior work that simplified the above definition by $\min_x \lVert Ax - b \rVert_2 $ s.t. $A \in \mathbb{R}^{n\times d}, b \in \mathbb{R}^n $, it uses the following more formulation which is proposed in Deng et al. (2023b) and argued to be more close to the definition: $ \min_{x \in \mathbb{R}^d} \lVert \langle \exp(Ax), \mathbf{1}_n \rangle^{-1} \exp(Ax) - b \rVert_2. $

From the above formulation, the loss function following Deng et al. (2023b), which can further be simplified by the shorthand form $ L_{\exp}(x) = 0.5 \lVert c(x) \rVert_2^2 $ where $ c(x):=f(x)-b, f(x)=\alpha(x)^{-1} \exp(Ax), \alpha(x):=\langle \exp(Ax), \mathbf{1}_n \rangle. $

Then, Lipschitz bounds for $ \lVert f(x_{t+1}) - f(x_t) \rVert_2 $ and $ \lVert f(A_{t+1}) - f(A_t) \rVert_2 $ are used to bound
$ \lVert \tilde{b} - b \rVert_2 $ with respect to  $ \lVert f(A_{t+1}) - f(A_t) \rVert_2, $ which reveals the relationship between softmax weight shifting and in-context learning.

### Strengths
- Rich explanation on preliminaries
- Mathematical notations are defined thoroughly

### Weaknesses
- More comparison with the work from Deng et al. (2023b) is needed, which seemingly to be the work most closely related to this work.
- It is hard to distinguish this work's contribution and other prior work's contribution. For example, some important definitions and theorems are already proven in Deng et al. (2023b). I believe this could be made more clear.
- Organization of the content is preferred to be more focused on what to be proven. i.e. "why bounding the single step of $x$ and $A$ relates to clarifying the relationship between in-context learning and softmax weight shift.

Typo:
Lipschtiz → Lipschitz

### Questions
- See above

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
