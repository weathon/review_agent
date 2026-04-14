# Non-linear activation soothes NTK conditioning for wide neural networks: a study in the ReLU case

- Decision: Reject
- Scores: 6, 5, 5, 6

## Abstract
Non-linear activation functions are well known to improve the expressivity of neural networks, which is the main reason of their wide implementation in neural networks. In this work, we showcase a new and interesting property of certain non-linear activations, focusing on the most popular example of its kind - Rectified Linear Unit (ReLU). By comparing the cases with and without this non-linear activation, we show that the ReLU has the following effects: (a) better data separation, i.e., a larger angle separation for similar data in the feature space of model gradient, and (b) better NTK conditioning, i.e., a smaller condition number of neural tangent kernel (NTK). Furthermore, we show that the ReLU network depth (i.e., with more ReLU activation operations) further magnifies these effects. Note that, without the non-linear activation, i.e., in a linear neural network, the data separation and NTK condition number always remain the same as in the case of a linear model, regardless of the network depth. Our results imply that ReLU activation, as well as the depth of ReLU network, helps improve the worst-case convergence rate of GD, which is closely related to the NTK condition number.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper theoretically studies the beneficial effects and interesting properties of ReLU activation function.

### Strengths
The strength of this paper is to show that ReLU activation function has the effects of better data separation and better NTK condition. This paper implies the optimization benefit that ReLU network helps improving worst case convergence rate of gradient descent and faster convergence than shallower one.

### Weaknesses
As mentioned in Conclusion and Discussion, the finite depth case is focused, and not directly extended to the infinite depth case.

### Questions
Out of interest, can the analysis of this paper be applied to only ReLU ? In other words, does this paper use specific properties of ReLU in the proof? For example, can it be little bit generalized to Leaky ReLU (= $ax$ when $x<0$, and $x$ when $x \geq 0$. The case when $a=0$ is special case of ReLU) ?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper investigates the impact of non-linear activation functions, specifically the ReLU in wide neural networks. The authors demonstrate that ReLU activation improves data separation in feature space and enhances the conditioning of the NTK, leading to better theoretical convergence rates of gradient descent optimization algorithms.

### Strengths
The paper provides a thorough theoretical analysis backed by empirical evidence demonstrating that ReLU activation improves both the separation of data in feature space and the conditioning of the NTK.

### Weaknesses
The analysis is specifically focused on networks with ReLU activation and the results primarily demonstrate that ReLU NTK outperforms linear NTK, which may seem somewhat limited in scope.


Typo: Line 209 $\nabla f(x)(z) \to \nabla f(z)$

### Questions
1. Can the findings be generalized to other non-linear activation functions? How might the NTK conditioning change with different functions?

2. What are the implications of these findings on network architecture design? Specifically, how might they influence decisions on depth and width of networks (m is finite)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper compares the deep networks with or without the ReLU activation under the NTK regime. They show that ReLU has two effects: (a) There is a larger angle separation for similar data in the feature space; (b) The NTK conditional better becomes larger.  They also show that the depth of the network will further enhance these effects.

### Strengths
- The paper is well-written, and the claims appear to be sound.
- The experiments are comprehensive and align well with the theoretical results.
- The investigation of the angle between two samples after projection into the feature space is both novel and intriguing.

### Weaknesses
- This paper compares only ReLU networks and linear networks. The results are not surprising, given the established fact that non-linear activations enhance the expressivity of networks.

- The title mentions "Non-Linear Activation Soothes NTK Condition," but the paper focuses solely on ReLU, which is just one type of non-linear activation.

- The NTK regime typically requires the network width to exceed a certain constant. However, the paper assumes that the width approaches infinity. It would be beneficial if the authors could relax this condition.

### Questions
- Could the authors compare additional non-linear activation functions in the experiments?

- Is it feasible to extend the current analysis to GeLU or SiLU?

- Can the condition of infinite width be relaxed to require a sufficiently large width?

- There is a typo in line 195; $G$ should be in $\mathbb{R}^{n \times n}$.
 .

### Soundness
3

### Presentation
3

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
In this work, the authors study the benefits of using ReLU activation for Wide Feedforward Neural Networks under the NTK framework. Contrary to previous works that focused on expressivity, they adopt a novel perspective and show that ReLU activation yields better data separation in the gradient feature space and, hence, better NTK conditioning when compared to Linear Networks. This effect is even exacerbated with deeper networks. They also illustrate their main results with experiments on synthetic and benchmark datasets (MNIST, etc.).

### Strengths
Approaching the study from the perspective of data separability rather than focusing on expressivity proves to be an insightful choice. The insights obtained are interesting and complement the existing results well. Besides, the paper is well-written and accessible to a relatively broad audience. The experiments illustrate well the main findings.

### Weaknesses
The main limitation I observed, which is often anticipated in papers leveraging the NTK framework, is that this initialization differs from those commonly used in practice. While it allows for theoretical insights, the paper would be significantly strengthened if the authors could provide empirical verification to determine if these findings extend to more practical initialization schemes.

A secondary limitation lies in Theorems 4.2 and 4.3, which establish that enhanced data separability in the gradient feature space concretely benefits NTK conditioning. However, these results rest on stronger assumptions, though the experiments partially compensate for this limitation.

### Questions
- In the paragraph beginning on line 132, the authors reference a paper by Arora et al., which suggests that deep linear networks accelerate optimization. This claim appears to contradict the message of Section 2 in the paper. A brief comment could clarify this point and help readers better reconcile these perspectives.

- I would suggest expanding the 'Infinite Width Limit' section (line 177) by adding a couple of sentences to clarify what is meant by taking the infinite limit. Specifically, it would be helpful for the authors to specify the type of convergence they refer to and how they manage successive layers in this context. As stated in the theorems ($m \rightarrow +\infty$), it seems to imply that the widths of different layers go to infinity simultaneously. However, after a high-level check of the proofs, it appears some arguments use induction on the layers, taking the limit successively, one layer at a time. Adding clarification here would improve reader comprehension and strengthen the rigor of the presentation.

### Soundness
3

### Presentation
3

### Contribution
3
