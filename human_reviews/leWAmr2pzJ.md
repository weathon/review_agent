# Understanding the Approximation Gap of Neural Networks

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 5, 3, 3

## Abstract
Neural networks have gained popularity in scientific computing in recent years. However, they often fail to achieve the same level of accuracy as classical methods, even on the simplest problems. As this appears to contradict the universal approximation theorem, we seek to understand neural network approximation from a different perspective: their approximation capability can be explained by the non-compactness of their image sets, which, in turn, influences the existence of a global minimum, especially when the target function is discontinuous. Furthermore, we demonstrate that in the presence of machine precision, the minimum achievable error of neural networks depends on the grid size, even when the theoretical infimum is zero. Finally, we draw on the classification theory and discuss the roles of width and depth in classifying labeled data points, explaining why neural networks also fail to approximate smooth target functions with complex level sets, and increasing the depth alone is not enough to solve it. Numerical experiments are presented in support of our theoretical claims.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the approximation gap of neural networks (NN) in scientific computing. The attempt is theoretical and the authors provided some novel perspectives, by studying the topology of the image set of NN and considering the practical assumption when the representation of floats has finite precision. Additionally, the authors found that increasing depth does not necessarily improve the accuracy in NN approximation. The paper also provides some numerical experiments to validate these claims.

### Strengths
1. The question is interesting: The deep learning (DL) methods have certain limitations compared with traditional numerical methods in solving PDE. Investigating how this limitation arises is an interesting research question and this work is properly motivated. 

2. The findings and perspectives are novel: The authors prove several interesting and novel results, for example, increasing the depth does not necessarily improve the capability of NN on isolable sets. Studying the approximation of NN with the assumption that the float resolution is finite is realistic and novel.

3. Mathematical rigor: The theoretical claims in the paper are mathematically rigorous.

### Weaknesses
1. Nonconstructive result: The implication of the topology on the image set of NN is nonconstructive and hard to grasp. It is hard to understand how this relates to the NN approximation gap. A realistic example can help illustrate this point.

2. Presentation can be improved: It appears that the authors provide several independent causes for the NN approximation gap; and in the numerical experiments, it does not seem all the causes are verified. The authors can draw the connection more explicitly.

3. Limited impact: From the deep learning perspective, applications of NN in scientific computing are not mainstream. For example, the approximation gap is more likely when the PDEs only have discontinuous solutions as the authors claim. However, in applications of vision and natural languages, it is unsure whether such a discontinuity assumption holds. On the other hand, the result that approximating a discontinuous function using continuous NN is hard is not surprising.

Minor:
1. Section 3, paragraph 2: Let us $J(\theta)$ --> Let $J(\theta)$

2. The index for union is $\cup_{K=1}^\infty$ instead of  $\cup_{i=1}^\infty$?

3. A more standard notation for the ReLU function is ReLU instead of Relu.

4. L is used in Lebesgue measurable functions and neural network width.

### Questions
1. Are the results independent? For example, if an NN method fails to solve the PDE while traditional methods are able to handle it, could you attribute why the NN method fails as claimed in the paper?

2. While the result on the NN depth is interesting and novel, is it directly related to the approximation gap of NN in solving PDEs?

3. Suppose the assumptions do not hold in the paper: the PDEs have continuous solutions and the global minimum is not at infinity, will NN methods work for them? 

Overall, more evidence supporting those points can strengthen this paper if understanding the approximation gap is the subject.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Summary. The authors study the approximation gap of neural networks. They derive that a bounded subset of the image set of a neural network does not necessarily have compact closure, especially in the case of discontinuous target functions. The authors also explain that the increasing depth alone is sometimes insufficient to improve the approximation accuracy.

### Strengths
Originality: The related works are adequately cited. One advantage of this paper is to study the approximation property of deep neural networks for some unusual functions and derive some related results. The main results in this paper will certainly help us have a better understanding of the universal approximation property of deep neural networks from a theoretical way. I have checked the technique parts and found that the proofs sound solid. 

Quality: This paper is technically sound.

Clarity: This paper is not very clearly written. I find it is not easy to follow.

Significance: I think the results in this paper are not very significant, as explained below.

### Weaknesses
Some of the results in this paper are not quite interesting. For example, Theorem 3.1 derives a universal approximation theorem of two-layer sigmoid or Relu neural networks for $L^2 (D)$. Although it provides some results for the needed width, it is still not a significant contribution since the universal approximation of neural networks for $L^p$ was already known more than 30 years ago.

### Questions
1. Line 5 in Section 3,  Let us J(θ) denote -->  Let J(θ) denote.
2. Line 13 in Section 3,  $S_K(N))$ -->  $S_K(\mathbb{N})$.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper investigates the theory-practice gap employing neural networks to solve scientific computing problems: In contrast to the universal approximation theorems that ensures neural networks' good approximation ability (given sufficient size), neural networks suffer in situations that involve discontinuous or highly oscillatory solutions.

The paper shows the following:
1. For neural networks, the closure of the set of functions represented by neural networks is not compact (in the $L^2$ space), which implies that there is no global minimum in the loss landscape.

2. In such cases, finite precision in real computers prevents us from approaching the infimum at infinity, incurring some degree of inevitable $L^2$ loss.

3. The paper then turns to classification, studying the role of depth in isolating given data points (similar to the context of shattering). The paper shows that depth alone provides limited benefits in terms of the network's capability to isolate data points: if all shallow networks fail to isolate certain points, a network with an additional layer must also fail.

### Strengths
1. This paper studies the theory-practice gap present in scientific computing (e.g., solving PDEs using neural networks) in a mathematically rigorous way, and provides new insights.

2. The paper reads quite well, although there are some minor typos to be corrected:
- Below Eq (1), $\bigcup_{i=1}^\infty$ -> $\bigcup_{K=1}^\infty$?
- Proof of Theorem 3.1: in the ReLU, $y$ should be corrected to $\sigma(nx+1/2)-\sigma(nx-1/2)-\sigma(-nx+n-1/2)+\sigma(-nx+n+1/2)$.
- Image set $Im(f)$, effective region $\mathcal E(w,b)$ used without proper definition.
- Definition 5.2: shattered by $H$ -> $\mathcal S$?

### Weaknesses
The paper claims to investigate the approximation gap, but I am not really convinced why the main results should be surprising or whether the results are really relevant to approximation.

1. The paper claims that the bounded subset of the image set is compact for classical methods and non-compact for neural networks, and uses this fact to claim that neural networks have limited capability for approximating discontinuous functions. However, can compactness alone guarantee that a scheme is able to approximate discontinuity? I may not be very familiar with the classical approximation schemes mentioned here, but I know the Fourier series suffers from the Gibbs phenomenon, so I am doubtful if classical methods really deal with discontinuity well. Could you comment on this?

2. The intuition for non-compactness of neural network image set is roughly the following: sigmoid $x \mapsto \frac{1}{1+\exp(-\theta x)}$ cannot approximate the step function $1\\\{x \ge 0\\\}$ unless $\theta \to \infty$. But I wonder why this observation should be surprising?

3. I also do not see why we should be surprised about the finite precision result. With finite precision, we expect to see numerical errors anyway. Also, in contrast to the claim of $O(\frac{1}{\sqrt{p}})$ error in the introduction, Eq (2) seems to suggest that the $L^2$ error can be made small by making the grid size $\Delta x$ finer. 

4. I am not 100% convinced of the connection of the classification result to function approximation. Section 5 is about isolability and shattering, which arise in the context of statistical learning theory. Being able to isolate each data point (separation) and assign **arbitrary** labels (gluing) seems to be a much stronger requirement than being able to approximate a given highly-fluctuating function. Hence, drawing conclusions on approximation from these results could potentially be misleading. For example, in the introduction, it is claimed that "This result suggests that the target function whose level sets are highly disconnected are naturally hard to approximate and increasing the depth alone is not sufficient to fit them well." However, this sounds like a contradiction to the well-known result in depth separation "Representation Benefits of Deep Feedforward Networks" by Telgarsky (2015), which shows that deep and narrow networks are much better at representing a highly-oscillating train of triangle waves compared to shallow and wide networks.

5. I am not convinced if the experiments corroborate and solidify the theoretical findings in the paper. Fitting complex functions using neural networks is a highly nonconvex optimization problem, and even if there is a perfect approximating solution, it is possible that optimization algorithms never reach it due to complications arising from nonconvexity. It seems difficult to claim that the poor performance of neural networks can be solely attributed to the limited approximation power.

### Questions
Please see the weaknesses section.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper is motivated by the empirical observation that neural networks are not well-suited to achieve high-performance for tasks in scientific computing and goes forth to propose reasons why this might be the case. The stated reasons and exploration in the paper have to do with the topology of the neural networks and also the discontinuity of sought-after solutions (e.g., in PDEs)

In summary, the paper takes a more realistic approach to the universal approximation theory that has been developed for neural networks, and study the so-called approximation gap that is empirically observed. The authors establish a connection between approximation and classification of level sets and show that to separate differently-labeled input points is handled by the width of the network while assigning same-labeled input points to the same region in space is handledg by the depth. However, if the target function has certain discontinuity properties, then the neural network's depth will not be enough to guarantee good accuracy results.

### Strengths
+well-motivated problem of understanding the approximation gap in practice and with theoretical justification

### Weaknesses
-the paper is written in a very confusing way, in that there is no clear connection between the technical aspect of what is proven, and the realistic interpretation of what the result actually implies for the practice of neural networks or their properties in the real-world. Simply saying that some set is not compact is not good enough.

-I find the weight-precision problem mentioned as a third contribution a bit misleading. Isn't it always a problem that we have finite machine precision?

-overall the paper has a nice collection of interesting results, but none of them is especially impactful. As such, I believe the paper does some incremental progress but doesn't have a "claim-to-fame" result that would be on par with an ICLR paper.

### Questions
Q: I believe for the presentation it would be helpful to start with a very simple example that showcases the difficulty of a neural network to solve a task (e.g. for a PDE). Then it would be nice to showcase how your results are informative in that simple example and actually give interpretation of the results. Currently, the presentation is very complicated.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair
