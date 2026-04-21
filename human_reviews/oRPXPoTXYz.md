# Backpropagation-Free Learning through Gradient Aligned Feedbacks

- Avg Score: 3.67
- Decision: Reject
- Scores: 3, 5, 3

## Abstract
Deep neural networks heavily rely on the back-propagation algorithm for optimiza-
tion. Nevertheless, the global sequential transmission of gradients in the backward
pass inhibits its scalability. The Direct Feedback Alignment algorithm has been
proposed as a promising approach for parallel learning of deep neural networks,
relying on fixed random feedback weights to project the error on every layer in
a parallel manner. However, it notoriously fails to train networks that are really
deep and that include compulsory layers like convolutions and transformers. In this
paper, we show that alternatives to back-propagation may greatly benefit from local
and forward approximation of the gradient to better cope with the inherent and
constrained structure of such layers. 

This directional approximation allows us to design a novel algorithm that updates the feedback weights called GrAPE (GRadient
Aligned Projected Error). A first set of experiments are carried out on image classi-
fication tasks with feedforward and convolutional architectures. The results show
important improvement in performance over other backpropagation-free algorithms,
narrowing the gap with backpropagation. More importantly, the method scales
to modern and deep architectures like AlexNet, VGG-16 and Transformer-based
language models where the performance gains are even more notable.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This article presents a novel alternative to backpropagation, that proposes to improve the framework of direct feedback alignment by aligning  the feedback matrices using forward-mode automatic differentiation. More precisely, activity-perturbed forward gradients are computed at the end of every layer, and the feedback matrices are rescaled following their alignment with the forward gradients. They find that the learning rule increases performance compared to direct feedback alignment over several frameworks.

### Strengths
- The proposed approach is novel, and combines two closely related approaches: direct feedback alignment (DFA) and forward gradients, in a natural direction.
- The authors find a performance increase compared to standard DFA.
- The paper is well written and does an extensive overview of the related work.

### Weaknesses
- The proposed method is not motivated enough and is very surprising, in particular since it is the only contribution of the paper. If the goal is indeed to align the feedback matrix, why is this update rule the only one considered? A strange decision is that the matrices are not actually aligned differently after the update step, but only rescaled (since the update step is only $B_l \leftarrow (1-\eta(1-cos(\omega_l)))B_l$. Why not use directly the forward gradient value $g$ to change the alignment of the feedback matrices? 
- Using the cosine value between the forward gradient and the matrix also seems very surprising. If we considered, for instance, the case of a weight-perturbated forward gradient, the direction of $g$ would simply be a random one, and the only important value in it would be its magnitude, which would depend on the directional derivative value computed. Using a cosine similarity on this gradient would not make any sense, since it would collapse the magnitude information, and the similarity would be the same one as with the random vector before the forward gradient computation. This is not exactly the case for an activity-perturbed gradient, but this choice still seems very strange.

I may be completely misunderstanding the update rule, but there seems to be a mistake in it, otherwise, it simply does not make sense as a way to align the matrices. For a feedback matrix not aligned at all with the gradient, with a similarity equal (in expectation) to $0$, the matrix would simply collapse following it. I'm open to discussion with the authors on their idea. 

- No further analysis is provided of the method except for the performances. It seems surprising to not at least measure the alignment of the feedback matrices through time. No further ablations are proposed either. Some that could be considered: w.r.t activity vs weight perturbated forward gradients, other feedback-matrix learning rates, or any variants of the update rule considered.

- No theoretical or empirical analysis of the complexity of the method is considered, which seems necessary to compare it to the others.

- What does Figure 2 bring? It seems just a way to show that forward gradients, when sampled enough time, will converge to the true gradient, which does not seem surprising.

### Questions
- Have the authors considered other learning rules other than this one? Could they offer a more thorough explanation of it?
- Why does DFA only reach 1% test accuracy on CIFAR100? Why are the CNN values for PEPITA equal to NA?

**Minor details**

- The name of the method in Figure 1e (GAEP) is not the same as in the rest of the paper.
- The dependency of $B_l$ on $t$ is unclear depending on the equation.
- Eq 1: a "BP" is missing on one $\delta a$
- line 36: the abbreviation BP was not written before.
- line 204: "as illustrated Figure1"
- line 319: "set the a large"
- line 490 "Transformers Table 3"
- The use of $B_l$ or $B^{(l)}$ varies in Algo 1.
- Some inconsistent number of decimal places in Table 1.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors present Gradient Aligned Projected Error (GrAPE), a backpropagation-free learning algorithm aimed at parallelizing the training of deep neural networks by aligning feedback paths with forward gradients. GrAPE, which modifies the Direct Feedback Alignment (DFA) approach, is evaluated on tasks requiring modern deep architectures such as AlexNet, VGG-16, and Transformer-based models. The results show significant improvements over other backpropagation-free methods, especially in scaling to complex architectures, although it still lags behind backpropagation in generalization performance. Notable is its extension to convolutional layers, which other feedback alignment methods have lacked so far. 
Their modification of DFA includes aligning feedback matrices with forward gradients, which they do with the help of the Zoutendijk theorem. They proved that the method converges and has good empirical results in their experiments. The authors highlight GrAPE’s biological plausibility and its potential for future neuromorphic applications.

### Strengths
i) The authors propose a new update rule for the direct feedback matrices to align their directions with the corresponding gradients. The goal of this update is thus to minimize the orthogonal directions while improving the gradient alignment. This is about the analysis with Feedback Alignment algorithms, which require the angle between the feedback and the true gradient of the forward process to be less than 90 degrees.

ii) Update ensures that the parallel directions with the gradients will be promoted for error projection.

iii) The method successfully addresses the limitations of DFA with respect to convolutional layers.

iv) The experimental evaluation is extensive, including performance on both shallow and deep architectures across datasets (MNIST, CIFAR-10, CIFAR-100, WikiText-103). The results clearly demonstrate that GrAPE outperforms DFA, especially in architectures with complex structures like transformers and VGG-16.

v) The approach is grounded in the Zoutendijk theorem for optimization, which provides a convergence guarantee when feedback directions align with gradient directions. This theoretical underpinning gives the algorithm a sound basis for further exploration.

### Weaknesses
1) The Zoutendijk theorem could have been better explained with and without context to the DFA issue. The readers have to understand the theorem to relate to the claims based on it.

2) The presentation could have been better.

3)  The paper mentions a need for further research on regularization techniques, as the authors have noted.

4) While the algorithm addresses some parallelization issues, the paper does not elaborate on the computational overhead and variance in forward gradient estimations.



.

### Questions
1) How much degradation does it come with for ImageNet? I understand that the process is still in the development stage, this question is not to penalize the paper contributions.

2) How much difference in accuracies are present when the conv layers are implemented with and without GrAPE (with rest of FC layers implemented with GrAPE)?

3) There are methods AugLocal (ICLR 2024) and InfoPro (ICLR 2021) that explain why the feedback alignment methods are failing for deeper networks. In that case, the linear separability of features must be addressed. This property shows how much the earlier layers in the network are neglecting the subsequent layers in adapting to the local loss objective. Could that analysis be done for this method? An interesting understanding might emerge.

4) Could you provide the computational overhead required in forward gradient estimations?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The presented paper proposes an Adaptive Direct Feedback Alignment method which mixes together Forward Gradient calculation and Direct Feedback Alignment. The underlying idea is to augment standard DFA algorithm with forward gradient calculation, in order to derive the cosine angle between the feedback matrix and the true gradient. The authors uses this extra information to make their algorithm adaptive, by updating the feedback matrices before projecting the error. This favors a better alignment with the projected error and the true gradient, yielding much improvements over backpropagation-free alternatives while still lagging behind backpropagation.

### Strengths
1. The paper is generally well-written, albeit with some (very concerning) exceptions. The introduction, background and related work are very understandable, which is not often the case in the dedicated literature.
2. The experiments clearly show results in favor of the proposed method against backpropagation-free alternative. More precisely, authors obtains figure on par or better than standard FA, which itself surpasses standard DFA. Thus, the authors are able to improve the gap with backpropagation while keeping their algorithm parallelizable.
3. The authors acknowledge and characterize a stronger overfitting problem than standard DFA.
4. The variety of architectures studied in the experiments is relevant with respect to the state-of-the-art.

### Weaknesses
**Crucial weaknesses:**
1. There is a simple but major problem with equation (6). The matrices $B_l^t$ does not depend on the input $x$, but the cosine angle does. The cosine angle is defined as
$$
\cos(\omega_l) = \frac{g_{a,l}^{\top} B_l}{||g_{a,l}|| \cdot || B_l ||}
$$
But the gradient $g_{a,l}$ depends on $x$. As a consequence, it is not clear how the algorithm is supposed to behave in a mini-batch setting. Accordingly, the pseudo-code in algorithm 1 is not clear since we iterate over the whole dataset before updating the weights. Either the reviewer does not understand the dimensions of the matrices involved, if there is a mini-batch dimension, either the pseudo-code does not correspond to the actual implementation able to deliver such figures in the experiment section.

2. Albeit empirical performances on various datasets and models are provided, there no ablation on any aspect of the proposed algorithm. The authors claims a good compatibility with Batch-normalization without giving evidence. The authors does not specify how they rescale the feedback matrices, nor if the normalization plays a crucial role. 

3. The authors shows that the cosine similarity of the forward gradient estimate increase with the number of JVP calculation. This is more a debugging experiment than an ablation study as the theory clearly predicts it would happen. It would actually be very interesting to follow the cosine similarity between the projected error and the true gradient on real dataset throughout training, and compare the different methods.

4. To the reviewer, the most crucial aspect of such paper would be to ablate activity perturbation vs weight perturbation. Putting aside the fact that it is not clear how the computations happens for activity perturbation in a batch setting, it intuitively seems more natural to try to align a feedback matrix with the weight gradient rather than the activity gradient.

**Small weakness:**
Even if the paper focuses on improving model performances, it would be nice to have to some runtime comparison between different methods. Even though it would not reflect the parallelization potential given that experiments are performed on single GPU, it would give an idea of the practical overhead induced by forward gradient calculations.

**Minor remarks:**
1. Line 51: typo “the the feedback…”
2. Line 258: typo “feedback path with and the gradient” → is it "with" or "width" ?
3. Fig.1.e → GrAPE instead of GAEP.

### Questions
Please address the crucial weaknesses in details.
If possible, it would be appreciated to have some more information about the small weakness.

### Soundness
3

### Presentation
2

### Contribution
2
