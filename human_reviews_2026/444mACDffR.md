# NTK with Convex Two-Layer ReLU Networks

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
We theoretically analyze a convex variant of two-layer ReLU neural networks and how it relates to the standard formulation. We show that the formulations are equivalent with respect to their output values for a fixed dataset and also behave similarly during gradient-based optimization as long as the weights on the first layer of standard networks do not change too much, which is a common assumption for their convergence to an arbitrarily good solution.
We further show that for any two-layer ReLU neural network, even considering those of infinite width, there exists a (weighted) network of width $O(n^{d-1})$ with the same output value on all data points. Furthermore, these finite networks have exactly the same eigenvalues $\lambda$ of their neural tangent kernel (NTK) matrix and the same NTK separation margin $\gamma$ as in the infinite width limit. After handling these preliminaries, we get to our main results:
We give a $(1\pm\varepsilon)$ approximation algorithm for the separation margin $\gamma$ which was not known how to evaluate in general and we study two data examples: 1) a circular example for which we strengthen an $\Omega(\gamma^{-2})$ lower bound against previous worst-case width analyses; 2) a hypercube example that can be perfectly classified by the convex network formulation but not by any standard network, distinguishing their expressibility.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors use a formulation of shallow relu networks called ‘convex networks’ which decouple where neurons turn on and off from their actual numeric outputs. The activation region layout is thus fixed at initialization, and only the neuron outputs are trained. The convex network formulation makes it so that data points do not change activation regions during the course of training, so that the optimization problem becomes convex in parameter space. This nice property allows the authors to establish when infinite width networks could be replaced by finite width networks and still maintain the NTK separation margin.

### Strengths
1. The authors are extensively aware of the related literature
2. Use of the convex network formulation is a clever way to make results tractable, while also being interesting in its own right.
3. The authors present an impressive variety of results.

### Weaknesses
1. 43 up to a factor OF two
2. 148 ‘this motivates to’
3. 190 did you mean to say ‘at initialization’ rather than ‘at activation’?
4. 229 improve should be improves
5. 235 should say (omitting parameters other than gamma)
6. The paper does a lot of things, but that might work to its detriment, for example, the contributions section has 8 items in it, but not all of them feel like they’re the central point of the paper.
7. The paper is very dense and symbol heavy with no figures to break it up. Although the main content of the paper is mathematical, the example data problems at least could probably benefit from figures?
8.  Perhaps the paper could dedicate a little more room to motivation. For example, why is it important to have width bounds for fitting training data when the ultimate goal is generalization? Or if the NTK separation margin and minimum eigenvalue are the same for a finite sized network as they are in the infinite case, do we know that the finite sized network would generalize as well as the infinite? Maybe each result in the main body could be accompanied by a bit more of an explanation of why it’s fascinating or related to getting better performance out of neural networks.

### Questions
1. Are the citations you list around line 50 covering both why GD converges to arbitrarily small errors in nonconvex optimization and convergence results for  overparameterized networks? Are those two sentences supposed to be two disjoint sets of citations or are they related? Or was the first sentence just meant to introduce the related works and not be followed by citations?
2. It’s shown that ordinary relu networks can classify a dataset equivalently, but convex networks seem like they’re able to make discontinuous functions since neurons can ‘turn on’ to nonzero values. If the goal of the network is ultimately to generalize, might that matter that convex networks might learn discontinuous functions?

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a new convex approximation for two-layer ReLU networks. The authors prove that their formulation is equivalent to the standard neural networks, thus allowing them to benefit from the common convex optimization analysis of the convergence. Overall, such a formulation is supposed to make the analysis of two-layer ReLU networks more intuitive and bridge some gaps in the existing worst/base case bounds.  Among the main results the authors provide is also a way to estimate the eigenvalue of the NTK in a sound way. Finally, several examples of tasks where the convex formulation is superior to the traditional one are shown.

### Strengths
1.	Novel study on the convex formulation of two-layer ReLU networks
2.	NTK perspective on the established equivalence and the derived proximity of the optimization dynamics. 
3.	Examples of when the convex formulation can solve tasks that cannot be provably solved by the traditional two-layer ReLU network.

### Weaknesses
1.	Very dense paper that is hard to read
2.	Many key results are not compared to prior work
3.	The importance of certain contributions is not clear

### Questions
1.	Why did the authors choose a different name for the object of study? In the manuscript, they mention that they study gated ReLU networks, but call them convex two-layer ReLU networks.
2.	Could the authors provide a clear breakdown of their contributions when compared to prior works? While the authors cite relevant works (to the best of my knowledge), it is not immediately obvious to find a direct comparison to the results presented in them. Ideally, a reader would like to see a clear distinction after each major result, comparing it to existing results and/or explaining it novelty. Otherwise, the reader has to go through all cited papers to actually assess the importance of the authors’ findings.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper theoretically analyzes a convex variant of two-layer ReLU neural networksand establishes its close relationship to the standard, non-convex formulation. The key idea is to "convexify" the network by decoupling the neuron's activation pattern (determined by fixed "orientation" vectors) from its learned weight. This simplifies the optimization landscape, making the training problem convex, while retaining much of the expressive power of standard networks.

### Strengths
This paper theoretically analyzes a convex variant of two-layer ReLU neural networksand establishes its close relationship to the standard, non-convex formulation.

### Weaknesses
In my view, this paper could better isolate and clarify its key contributions to make them easier to grasp. Below are some of my concerns, which may be partially due to my unfamiliarity with the relevant literature.

The authors review certain aspects of convex two-layer neural networks and present some preliminary results. While these findings are somewhat interesting, they are not sufficiently compelling to convince me of the paper's overall strength.

Subsequently, the authors analyze the separation margin and the smallest eigenvalue of the NTK, claiming that when m=|S_{0}|, these quantities equal those of the convex neural network. I am unclear on the purpose or significance of this result.

Finally, they demonstrate that for some datasets, the required network width to approximate the NTK can be improved. If this is the major result, it should be stated explicitly and prominently at the beginning of the paper.

### Questions
same to the weakness.

### Soundness
2

### Presentation
1

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
This work investigates a convex counterpart of a two-layer ReLU network and proves this convex variant shares similar properties with the original ReLU network such as the NTK matrices having the same eigenspectra as well as the same NTK margin. Additionally, the paper constructs a data distribution where any perfect NTK separator with width $O(\frac{1}{\gamma^2})$ must have weights dependent on initializion, improving the lower bound construction from prior work.

### Strengths
The construction of a dataset where a perfect NTK separator with subquadratic (with respect to the reciprocal of the margin) width must have weights dependent on initialization illuminates the difficulty of getting subquadratic width guarantees. 

In addition, showing that GD on the convex variant converges arbitrarily close to the NTK margin is interesting as it provides an algorithm for computing such margins, something that was not known beyond special cases such as parity problems.

### Weaknesses
The convex variant of the ReLU networks seems quite tied to the 2-layer case and it does not seem easy to generalize this formulation to multi-layer ReLU networks, much less architectures with attention.

Furthermore, the convex formulations behave similarly to their original counterparts during training with GD only when the inner layer doesn't change much. In contrast, [1] analyzes GD on two-layer ReLU networks while also allowing for quite a bit of movement $O(\gamma \sqrt{m})$.

[1] Telgarsky, M. (2022). Feature selection with gradient descent on two-layer networks in low-rotation regimes. arXiv preprint arXiv:2208.02789.

### Questions
1. Are there any toy datasets where there exists linear width perfect NTK separator with initialization dependent weights but lack such a separator when the weights are initialization invariant?
 
2. Please provide comparison with [1] (specifically theorem 2.1 and 2.2). My main issue with the paper is the fact that the analysis of GD on the convex variant is only really useful when the inner layer doesn't move much. But [1] handles $O(\gamma \sqrt{m})$ amount of movement while also analyzing GD on the original formulation.

3. It seems the approximation algorithm for computing $\gamma_V$ (i.e. lemma 6.1) requires the knowledge of $\gamma_V$ as one needs to choose $B$ to be sufficiently large enough. Can you remove this circular dependency? I guess a doubling trick should work but that could be expensive if $gamma_V$ is tiny (e.g. if $\gamma_V$ is exponentially small with respect to dimension).


[1] Telgarsky, M. (2022). Feature selection with gradient descent on two-layer networks in low-rotation regimes. arXiv preprint arXiv:2208.02789.

### Soundness
3

### Presentation
3

### Contribution
2
