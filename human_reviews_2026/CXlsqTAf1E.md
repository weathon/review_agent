# On the Convergence Behavior of Preconditioned Gradient Descent Toward the Rich Learning Regime

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 2, 6

## Abstract
Spectral bias, the tendency of neural networks to learn low frequencies first, can be both a blessing and a curse. While it enhances the generalization capabilities by suppressing high-frequency noise, it can be a limitation in scientific tasks that require capturing fine-scale structures. The delayed generalization phenomenon known as grokking is another barrier to rapid training of neural networks. Grokking has been hypothesized to arise as learning transitions from the NTK to the feature-rich regime. This paper explores the impact of preconditioned gradient descent (PGD), such as Gauss-Newton, on spectral bias and grokking phenomena. We demonstrate through theoretical and empirical results how PGD can mitigate issues associated with spectral bias. Additionally, building on the rich learning grokking hypothesis, we study how PGD can be used to reduce delays associated with grokking. Our conjecture is that PGD, without the impediment of spectral bias, enables uniform exploration of the parameter space in the NTK regime. Our experimental results confirm this prediction, providing strong evidence that grokking represents a transitional behavior between the lazy regime characterized by the NTK and the rich regime. These findings deepen our understanding of the interplay between optimization dynamics, spectral bias, and the phases of neural network learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper analyzes how preconditioned gradient methods (PGD) shape spectral bias and grokking dynamics. In the neural tangent kernel (NTK) regime, the authors theoretically show that Levenberg–Marquardt (LM) reduces spectral bias by re-scaling eigenmodes, and that as LM’s damping vanishes, it approaches Gauss–Newton (GN), which further equalizes per-mode convergence rates. Empirically, they evaluate on equation fitting and PDE solving with PINNs and find that PGD accelerates early error decay and compresses the train–test generalization delay associated with grokking, which aligns with the previous analysis. On MNIST classification, PGD speeds early loss reduction and shortens the grokking delay. However, it can the generalization can be weaker. The author suggests to use first-order methods after the higher-order methods, which solves the problem.

### Strengths
The theory analysis looks good to me, showing how LM narrows conditioning and how GN limit equalizes per-mode convergence, linking preconditioning to reduced spectral bias.

Experiments are conducted to examine and support the theory analysis.

The structure of the paper is well-structured with clear notations and helpful visualizations that make the paper easy to follow.

### Weaknesses
The novelty is limited. While the theory is well constructed, the contribution is mainly a clarified perspective on known conditioning effects of LM and the GN limit in the NTK regime.


The conducted experiments are small, and it is unclear whether these findings transfer to real and large-scale settings.


There are some questionable experiment design choices, see questions.

### Questions
1. Do the observed effects still hold true on mid-scale or large-scale datasets or models? It would be better if such experiments could be added to the paper for completion.
2. On MNIST, the authors use MSE loss. Can you explain why you use that instead of cross-entropy loss, since the authors also acknowledge the issue of using MSE loss as stated in the supplementary material? Will the conclusions and findings remain the same with cross-entropy loss?
3. Is there any way to show “all modes converging at a uniform rate” in the MNIST task?
4. The paper claims to first use first-order methods after the higher-order methods. However, how can we know when the training leaves the NTK regime?
5. How did you choose the learning rate for LM/GN, and how does it interact with the damping schedule? How sensitive is the method to the learning rate, since the authors say GN can be unstable at fixed LR?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper explores how Preconditioned Gradient Descent (PGD) affects two major phenomena in neural network training: Spectral Bias and "Grokking."
Neural networks (NNs) tend to learn low-frequency information first (spectral bias). While this can aid generalization, it also slows down learning for tasks that require high-frequency details. Concurrently, the "Grokking" phenomenon (where test accuracy suddenly improves long after the model has overfit the training data) hinders rapid training.
Grokking is a delayed behavior arising from the learning dynamics transitioning from the "lazy" NTK (Neural Tangent Kernel) regime to the "rich" feature-learning regime, and this delay is closely linked to spectral bias.
 PGD overcomes spectral bias and accelerates grokking. The authors propose a hybrid training strategy: first use PGD to rapidly exhaust the Lazy regime, then switch to a first-order method (like AdamW). Experiments demonstrate that this method leverages both PGD's fast convergence and the first-order method's strong generalization capability in the Rich regime, ultimately achieving the best of both worlds.

### Strengths
The paper clearly links three complex concepts: spectral bias, grokking, and the NTK/rich learning regimes. It is a potential problem for optimizer study.
This "higher-order first, then first-order" strategy is counter-intuitive to traditional fine-tuning but is well-supported by experiments (Fig 7), offering significant practical guidance.

### Weaknesses
Computational Cost and Scalability remain issues. The computational cost of PGD is extremely high because of it need inverse related to the Jacobian, it is a problem.
Regarding the rich regime, the paper primarily focuses on reaching this state faster, but it doesn't deeply analyze the complex, non-linear feature-learning dynamics that actually occur within it.

### Questions
Are the new parameters difficult to tune or not?
A experiment on Transformer will be very strong to show your contribution

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
The paper aims to understand the impact of preconditioned gradient descent on grokking.  They show that preconditioned methods converge uniformly along all spectral modes; they claim that preconditioned methods have diminished performance in the 'rich regime'; they argue that this supports recent theories that explain grokking using spectral bias rather than weight norm or adaptivity; and they argue hat preconditioned methods tend to remain in the lazy regime, which causes worse generalization.

### Strengths
This paper studies an interesting question (the intersection between grokking, rich/lazy training dynamics, and preconditioning).

### Weaknesses
* The paper's first main claim, on lines 62-63, that Gauss-Newton preconditioning ameliorates the spectral bias in optimization, is almost immediate from prior works.
* The paper's second main claim, on lines 64-66, that "higher order gradient descent methods allow for faster exploration of the NTK subspace, thereby allowing training to enter rich regime faster," does not seem to me to be a well-defined statement.  What does it mean to "explore the NTK subspace faster"?  It's not like GD and PGD take the same path, with PGD moving faster on the first part of the path -- GD and PGD take different paths.  So I don't know what it means to "explore the NTK subspace faster."  How is this different from just saying that PGD trains faster?
* The paper's third main claim, on lines 67-71, is that higher-order methods generalize worse because they stay close to the lazy regime.  I don't know understand why this would be the case, or what evidence is provided for this proposition.  Moreover, it seems to contradict the second main claim (that higher-order methods "explore the NTK subspace faster").
* The paper frames itself as building upon two papers, Kumar et al '24 and Zhou et al '24, aimed at understanding grokking.  Yet, upon reading these two papers, I feel that they are giving quite separate explanations for grokking (and the latter paper doesn't really seem to be studying grokking, since they seem to rely on a mismatch between the train and test set). Thus, I'm not sure how a paper can built on both of these prior works simultaneously, if they are disagreeing with one another.

Further, see the questions below.

### Questions
* The paper asserts, based on Figure 3, that Gauss-Newton / Levenberg Marquardt preconditioning are helpful only in the lazy/NTK regime, and that their effectiveness goes away in the rich/feature-learning regime.  I am not convinced by the current evidence.  The loss curves do show a slowdown at a particular point, but how do I know that this is when we leave the NTK regime?  And, is it actually true that the preconditioning is not helping at all past this point?  (If we stopped preconditioning, would we still train as fast?).  This is a pretty sweeping claim that is being made based on a single experimental setting.

* In Figure 5, it looks like the LM runs across different $\alpha$'s do not grok, and get similar test accuracy to one another, whereas in Figure 6 it looks like they do grok, and some get bad test accuracies.  Is my understanding correct?  If so, do you know why there is a discrepancy?

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
5

### Summary
The paper connects between grokking, spectral bias, and higher-order GD optimization (*preconditioned* gradient decent). Throughout the paper, the authors argue that grokking can be accelerated/alleviated using higher-order gradient descent algorithms because these methods allow for faster exploration of the NTK subspace and therefore enter into so-called *rich-regime* faster. The authors have justified their claims by showing that well-preconditioned gradient descents such as Gauss-Newton or Levenberg-Marquardt that mitigate the spectral bias also reduce grokking artefacts through experiments.

### Strengths
- The paper is generally well written and easy to read.
- Claims are supported by rich experimental results. The authors have provided various types of optimization algorithms and tasks to demonstrate their arguments. Furthermore, all theoretical results are linked to the experimental justification. All experimental details are provided in the appendices.
- The discussion is honest, with statement of limitations.

### Weaknesses
Some issues prevent me from giving a higher score; I will raise my score if the issues are resolved.

- The theoretical part shows that specific preconditioned gradient descents suffer less from spectral bias *within the NTK domain*, and the linked toy experiments prove that empirically, which is well addressed. However, linking this with the main claim: “the preconditioned gradient descent accelerates grokking since it reduces spectral bias within the NTK domain” only involves empirical experiments as in Figure 5. I believe additional theoretical bridge between this gap will strengthen the claim.
- The *preconditioned gradient descent* is rather a general category, and there should be certain criteria that determine “good precondition” (likewise for “good high-order GD”), which seems to be not explicitly stated and discussed throughout the paper. Rather, exemplar cases such as Adam, GN and LM are individually discussed. I believe providing unifying lens will further clarify the argument.
- Figures 1, 3, and 7 are quite messy. Perhaps scaling up some plot components including labels and plot areas will help visualize better.

### Questions
- The main idea of tuning optimizer to compress the grokking interval is also addressed in previous works such as [Grokfast](https://arxiv.org/abs/2405.20233), which seems to be tackling on the momentum and gradient filtering. How is the idea of the authors differ from previous works?

### Soundness
2

### Presentation
3

### Contribution
3
