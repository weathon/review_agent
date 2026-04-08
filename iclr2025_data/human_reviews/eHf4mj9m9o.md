## Human Reviewer 1

### Summary
The paper is focused on the important question of the role of saddles in learning dynamics. In particular, they classify saddles into two types: Type 1 and Type 2. Type-II saddles are attractive under SGD dynamics due to the so-called vanishing gradient noise. The theory and numerics are provided to support the claims.

### Strengths
The paper is focused on the interesting question of studying the attractivity of symmetry-induced saddles. Even though such saddles might have a negative eigenvalue in the Hessian, they suffer from degeneracy due to the symmetries. 

The one-dimensional model is illustrative and explains the escape mechanism from the saddle. 

Figures are done with care.

### Weaknesses
I think the paper needs a significant revision. I cannot read the mathematical notation as it is so I cannot evaluate the validity of the results. For example, three places where the classification of Type I vs Type II saddles are done -- all three unclear to me. 

1. Line 36: what does it mean "GD noise persists at the saddle"
2. Line 43: Type-I where the **noises of the gradient vanishes in the escape directions**, and Type-II saddles where the **gradient noise vanish in the directions of escape**. 

I read this sentence maybe 5 times, but the two definitions are identical up to a permutation of words.

* Can you provide a precise mathematical definition of "GD noise persists at the saddle" ?
* Can you clarify the distinction between "escape directions" and "directions of escape", or use consistent terminology if they mean the same thing ?
* Can you define these concepts explicitly when they are first introduced?


3. Line 102-103: What is $\hat{H}(x_t)$ here?

Figure 3 seems to capture the core result of the paper and hence could be put on the second page.  Can you provide a clear definition of this concept of $L_2$ stability when it is first introduced, as this seems to be an important concept in the paper ?

Why does SGD escape the Type-II saddle in Figure 1 (even though in a longer time scale). Aren't Type-II saddles are attractive under SGD by definition? Can you explicitly address this apparent contradiction and clarify the conditions under which Type-II saddles are escaped versus when they are attractive?

Finally, the attractivity of saddles and even local max is studied in Chen et al 2023 which is a very important but missing citation in this submission. They also study the symmetry-induced saddles and low-rank matrices for linear networks. 

Chen, Feng, et al. "Stochastic collapse: How gradient noise attracts sgd dynamics towards simpler subnetworks." Advances in Neural Information Processing Systems 36 (2024).

How does this paper compare to Chen et al 2023?

### Questions
See above.

### Soundness
2

### Presentation
1

### Contribution
2

### Rating
3

### Confidence
3

---

## Human Reviewer 2

### Summary
The paper proposes a classification of saddles into two types and then focuses on the second type of saddles, which are characterized by the fact that the variance of the gradient goes to zero as one approaches the saddle. The authors study the stability properties of such saddles and show that they can actually be attractive for large learning rates, as determined by a Lyapunov exponent that can be computed empirically.

The paper then studies a few simple neural network models and observe that SGD can in certain indeed converge to such type II saddle. Interestingly, these saddles tend to have low-rank weights (because they arise as symmetry induced saddles, right?).

### Strengths
I really appreciate the choice to focus on type II saddles, and agree that it is common for such saddles to play an important role. I believe that many previous works on SGD have tried to avoid such saddles (for example by approximating SGD with noisy/Langevin GD or some variant thereof, thus guaranteeing that all saddles are type I and therefore easily avoidable), while type II saddles appear harder to study,  their peculiarity should not be avoided, so I appreciate this paper for going this less common direction.

The experiments are on simple models but the analysis is quite thorough with phase diagrams and comparison with the computed Lyapunov exponents.

### Weaknesses
The theoretical results describing conditions for stability of the saddles are in terms of the Lyapunov exponents, which I am not particularly familiar with and seem hard to work with theoretically. To be honest going into the paper I was expecting that it would be possible to characterize stability/attractivity in terms of statistics of the random gradient/Hessian at the saddle (expectations or variances) like in the solvable 1D dynamics that they present. Do the authors believe that no such general condition could be obtained? And is the section on insufficiency of norm stability trying to explain why such a condition is not possible in general?

### Questions
Overall, while I think that the authors did a good job explaining the intutions between the observed behavior, I find some of the definitions a bit hard to parse, and some parts unclear:
- Is the type I / II classification a strict classification in the sense that every saddle is either type I or type II?
- Why do you define Lp-norm stability to then discard it saying that it is not useful for type II saddles two sections later? I do find Theorem 2 interesting, but maybe it would make sense to define Lp-norm stability in section 4.3, rather than introducing it earlier which suggests that it is a tool that you are going to use throughout the paper. And maybe explain that Lp-norm stability is a common and useful tool to do this in general but you show that it fails for type II saddles, or something like that.
- I find the definition of Lyapunov exponent a bit unclear, when you take the max over the initialization $\theta_0$ can you take any possible initial $\theta_0$, and when you later say that the Lyapunov exponent is independent of initialization, do you mean that it is independent of $\theta_0$?

There are two papers that are quite relevant to the symmetry induced saddles and SGD and how it can lead to a low-rank/sparsity bias (https://arxiv.org/pdf/2306.04251v3 and https://arxiv.org/abs/2305.16038). Both rely on the fact that the noise of the gradient vanish along directions that would allow the network to increase the rank, thus making these symmetry induced region attractive, which is similar to the phenomenology around type II saddles. In the second paper this effect is further amplified by the presence of weight decay (as you briefly note in the solvable 1D dynamics).

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 3

### Summary
In this paper, the authors focus on the dynamics of stochastic gradient descent (SGD) around the saddle points. They propose a fine-grained characterization of saddle points: Type-I and Type-II, where gradient noise vanishes in Type-II saddles. The paper focuses on Type-II saddles which are intuitively harder than Type-I. By analyzing the linearized dynamics in simple settings, the authors show that there are different dynamics depending on stepsize and data. Numerical experiments are provided to support the claims.

### Strengths
-	Understanding the dynamics of SGD near saddle points is an interesting and important research problem
-	The classification of saddle points into Type-I and Type-II seems to be new and interesting.
-	Some of the theoretical insights from simple linearized dynamics can transfer to more complex settings also seem to be interesting.

### Weaknesses
Many places are not very clear to me as a reader. See questions section below.

### Questions
-	For eq (1), I don’t understand how to get it. Also, the definition of $\|\|\Delta\theta\|\|$ is not very clear. It is the distance from saddle point to which point?
-	In eq (3) and (4), the Hassian matrix $\hat{H}(x_t)$ is at which point? I’m assuming it is at saddle point $\theta^*$, but it does not seem to mention this in the paper.
-	For eq (4), I believe it’s worth mentioning that this is an approximate or linearized dynamic instead of the actual dynamic. The current way of writing is not clear about this.
-	In eq (5), are $\theta_t$ and $\theta_*$ scalars or vectors? The description of 1d dynamics makes me think they are scalars, but in Theorem 1 it seems they are vectors.
-	In eq (8), what are $i$ and $d$ here?
-	In Figure 2, which point is the $\Lambda$ computed at?
-	I don’t see how to get the equation in line 288. Is it assuming $\theta_*=0$? Where does the $\|\|\theta_0\|\|^2$ in the denominator come from?
-	For the experiments in section 5.1, I wonder in general how to make the predictions using $L_2$ or probability stability. This seems not discussed in the paper. As a related question, if there are multiple attractive/stable points, how should one predict which point the algorithm should converge to? For example in the description in line 317, I believe both B and C are probability stable, so why SGD converge to C instead of B?
-	I would not agree with the interpretation before Theorem 4 that all neural networks are initialized close to a Type-II saddle. The initialization is not close to the 0 point for example in the sense of Frobenius norm.
-	In Figure 6 and 7, I wonder why use rank and sparsity as the metric to measure whether SGD escapes a saddle point. It is not clear to me why these results show that ‘’Type-II saddle are indeed a major obstacle in the initial phase of training’’ (line 472).

Typo:

-	In line 43, the description of Type-I and Type-II saddle points are the same.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 4

### Summary
The authors introduce a new classification of saddle points, Type 1 and Type 2, based on the variance exhibited by the SGD process near each type of saddle point. Using this distinction, they demonstrate that Type 2 saddle points can slow down or even prevent the convergence of SGD. This leads to a better understanding of SGD’s convergence behavior in neural network optimization.

### Strengths
1. The distinction in type 1 and type 2 saddles is new.

2. The authors not only provide theoretical evidence that type two saddles cause problems for SGD but also provide numerical examples to underline the results.

### Weaknesses
While the general idea behind the classification in these two types of saddle points is intuitive and interesting the following points weaken the contribution of the paper:

1. The notation used is often not formally introduced, details are missing. Even the under lying problem and SGD scheme is not introduced.

2. Given 1, it is hard to verify the correctness of the derived equations and theorems. See questions below.

### Questions
**Q1:** Sec 3. line 96:  loss $\mathcal l$ not introduced. I have a hard time to understand anything if I do not know the underlying problem and the probability space and SGD scheme is also not defined.
Could you provide: 
- A formal definition of the loss function $\mathcal{l}$
- A clear description of the probability space and problem setting
- A precise definition of the SGD update rule used

**Q2:** Eq. (1): Could you clarify the dimension of $\theta $? From the context, it seems to be a vector in $ \mathbb{R}^d $, but it’s unclear how a $ d $-dimensional vector could be $ \mathcal{O} $ of a real value, and whether this holds for any $ t $. I’m not familiar with this notation and am uncertain about its meaning here. Does this equation apply to any saddle point, and how is it linked to an expected gradient norm of zero? Additionally, when you refer to “$\lVert \Delta \theta \rVert$ as the Euclidean distance from the saddle,” do you mean $\lVert \theta_t - \theta \rVert $, or is it some quantity independent of $t$?


**Q3:** Following Q1 and Q2, I’m having some difficulty understanding the derivation of the different saddle types. Could you provide a more rigorous derivation of the saddle types. Are there other types not covered by this classification? Could you explain why only these two types are considered, if that is indeed the case.

**Q4:** Could you clarify what $P$ denotes the projection of— is it $\theta$ or $H$? Based on Definition 1, it seems likely that $\theta$ is intended, but this is just my assumption. 

**Q5:** How does Def. 1 imply eq. (3)? And does property (3) ($\theta$ is a saddle), not already imply property (2) (that $\theta$ is not a local minimum)? Compare to your definition of a saddle in line 96.

**Q6:** In line 113, you refer to $ H(\theta)$, but in Eq. (4) and throughout the rest of the paper, you use $ H(x) $ instead. Could you clarify the difference between these notations and what has changed?


**Q7:** Can you specify which result in Ziyin (2023) leads to eq. (4)? 

**Q8:** Proof of Thm. 1: you start by assuming eq (4), but why do you not need the projection $P$? And I thought you needed symmetry-induced saddles for this equation to hold.. are all type two saddle symmetry induced?



**Remark**

I made a genuine effort to understand the content of this paper, as I find the topic highly interesting overall. Assuming that Eq. (4) holds (without projection) around the so-called Type 2 saddles, the proofs appear solid, and the experimental results are indeed promising. I would be glad to raise my score if the questions above are addressed satisfactorily.

### Soundness
2

### Presentation
1

### Contribution
3

### Rating
3

### Confidence
2