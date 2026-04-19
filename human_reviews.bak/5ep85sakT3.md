# Contextual Bandits with Online Neural Regression

- Decision: Accept (poster)
- Scores: 8, 6, 5, 5, 6

## Abstract
Recent works have shown a reduction from contextual bandits to online regression under a realizability assumption (Foster and Rakhlin, 2020; Foster and Krishnamurthy, 2021). In this work, we investigate the use of neural networks for such online regression and associated Neural Contextual Bandits (NeuCBs). Using existing results for wide networks, one can readily show a  ${\mathcal{O}}(\sqrt{T})$ regret for online regression with square loss, which via the reduction implies a ${\mathcal{O}}(\sqrt{K} T^{3/4})$ regret for NeuCBs. Departing from this standard approach, we first show a $\mathcal{O}(\log T)$ regret for online regression with almost convex losses that satisfy QG (Quadratic Growth) condition, a generalization of the PL (Polyak-\L ojasiewicz) condition, and that have a unique minima. Although not directly applicable to wide networks since they do not have unique minima, we show that adding a suitable small random perturbation to the network predictions surprisingly makes the loss satisfy QG with unique minima. Based on such a perturbed prediction, we show a ${\mathcal{O}}(\log T)$ regret for online regression with both squared loss and KL loss, and subsequently convert these respectively to $\tilde{\mathcal{O}}(\sqrt{KT})$ and $\tilde{\mathcal{O}}(\sqrt{KL^*} + K)$ regret for NeuCB, where $L^*$ is the loss of the best policy. Separately, we also show that existing regret bounds for NeuCBs are $\Omega(T)$ or assume i.i.d. contexts, unlike this work. Finally, our experimental results on various datasets demonstrate that our algorithms, especially the one based on KL loss, persistently outperform existing algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies neural contextual bandits (CB) with the reduction to online regression under the realizability assumption. 
Using existing results for wide neural networks, it immediately gets a regret of $O(T^{3/4})$ for neuCB. 

When the loss function of the online regression satisfies $\epsilon$-almost convexity, Quadratic Growth condition and uniqueness of minima, this paper provides a $O(\log T) + \epsilon T$ regret for the online regression.

Then, since the above result does not directly apply to wide NN due to the violation of uniqueness of minima, this paper proposes a novel trick of adding a small perturbation to the NN prediction. This seemingly simple trick makes the loss satisfy the Quadratic Growth condition with a unique minimum. Consequently, an $O(\log T)$ regret for the online regression with KL loss and squared loss is established. 

Furthermore, this paper provides a lower bound of $\Omega(T)$ for NeuralUCB and NeuralTS with an oblivious adversary.

Finally, Experimental results show that the proposed algorithms outperform baselines on 6 classification datasets.

### Strengths
**Significance**: this paper is a quite extensive theoretical work on NeuralCB. The theoretical results as well as the techniques are of independent interest. 

**Quality**: the quality of the paper is good. Definitions are introduced without ambiguity; the theoretical analysis seems solid to me; experimental details are given. 

**Originality**: this paper contains some novel idea. The perturbation trick is a very interesting trick for bypassing the restriction on the uniqueness of minima. 

**Clarity**: this paper is very well written. The theoretical proof is divided into steps. A sketch of proof like this make it easier for the readers to follow and understand.

### Weaknesses
I did not detect any major technical flaw or major weakness in this paper.

### Questions
1. Is it possible to extend the current results to a broader choice of loss functions? It would be great if the authors can elaborate on this.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work provides a solid investigation on neural regression in contextual bandits. One interesting point is to use perturbation to construct a surrogate network output with unique minimizer. In that analysis, the work manage to get a approximate loss to satisfy strong-convexity-like condition (Quadratic Growth condition) to get analysis through. The article is full of useful theoretical insights and enjoyable to read.

### Strengths
The work provides solid investigation on improving online neural regression for better online decision making. The paper also is well-written, easy to read for audience with relative expertise.

### Weaknesses
Maybe not so readable for people without theoretical background.

### Questions
1. One particular point that draws my attention is to adopt KL loss in the online gradient descent step. Could you make comments on when should we use NeuFastCB and when should we use NeuSquareCB in practice? In particular, how the dimension or datatype of the real dataset impact this decision? 

2. Could you provide your comments on why NeuFastCB and NeuSquareCB outperform NeuralTS?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the contextual bandits problem. The paper proposed NeuCB, an algorithm based on applying SquareCB to neural networks with some assumptions. The paper proved a novel online regression guarantee for neural networks satisfying QG condition and some other assumptions. Based on this, the paper converted this into a $\sqrt T$-type bandit regret bound for NeuCB. Furthermore, the paper presented experiments on various real-world classification datasets to show its advantage over existing neural contextual bandits algorithms.

### Strengths
1. The online regression bound is novel and could further benefit the community.
2. The paper presents experiments to corroborate their theoretical findings.

### Weaknesses
1. The paper seems not to mention about whether the reward function $h$ in Assumption 1 is realizable by a neural network. Is it the case?
2. Similar to most previous NTK bandit paper, the paper made the assumption on both gaussian initialization and lower-bounded eigenvalue of the NTK. This limits the scope of this and all the NTK bandits papers.
3. The paper is written in a nested way so that I have to chase after the links between theorems, lemmas, and assumptions to figure out the true assumption set of a theorem: for each theorem, I need to consider its assumption and the assumptions in all lemmas/theorems it used.
4. The bounds in various theorems used $\widetilde{\mathcal O}$ without saying what's exactly being omitted by it. I suggest the author complete this part of information.

### Questions
1. In regret bounds Theorems 4.1 and 4.2, what's the dependency of the regret bound w.r.t. width $m$? If there's dependency, could you please write it out? If neither the Theorem nor the algorithm used $m$, then we could omit it.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper provides $O(\log T )+\epsilon T$ regret for online regression when the loss function satisfies $\epsilon$-almost convexity, QG condition, and has unique minima. The authors show that even though QG with unique minima may not be realistic, adding a suitably small random perturbation to the predictions, makes the losses satisfy QG with unique minima. Using a reduction from contextual bandits to online regression oracles, the authors use the results to show regret bounds for contextual bandits.

### Strengths
- The paper shows nearly-constant regret for online regression under certain assumptions.
- Some of the assumptions are valid for wide range of networks or can be enforced by perturbing the loss function.
- The regret bounds for contextual bandits are tight in some cases.

### Weaknesses
- The results require strong assumptions (2.a, 3, 5).
- The failure probability is large (grows with T) and cannot be controlled (see for example Thm 3.2).
- The regret bounds for contextual bandits grow with the number of arms rather than a complexity measure of the underlying class of functions. A typical scenario for CBs is when the number of arms is very large or even infinite. This does not recover even the simple result of contextual linear bandits where the regret grows as $\tilde{O}(\sqrt{dT\log K})$.

### Questions
Please see weaknesses.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the contextual bandit problem with the use of a neural network for online regression. The authors build upon recent progress in our understanding of neural networks, providing an online regression algorithm that for the KL and squared loss acheives O(log T) regret a significant improvement over previous O(sqrt(T)) results. Using this, they employ the reduction of contextual bandits to online regression presented in the SquareCB algorithm of Foster/Rakhlin.

### Strengths
- I found the paper to be easy to read, well-presented and interesting.
- The results are interesting and seem like a potentially important contribution. 
- It was harder for me to gauge the novelty of the approach used to prove the given results.

### Weaknesses
- The main result of this paper is given in section 3, where the authors show that a noise perturbed version of the neural network can be used to obtain a O(log T) regret bound. From what I understood, compared to the past results of Chen which just used Assumption 2a, this result exploited 2b and 2c to gain an improvement. This is akin to the fast rates that one gets for gradient descent run on strongly convex functions vs convex functions. Though the result is natural and intuitive based on Chen, I failed to understand what technical contributions the authors made to achieve the result in Theorem 3.1. Is this a straightforward application of existing results, or did it require a new perspective? Is the noise perturbation the main novelty?  Further discussion would have helped couch this work in the existing literature.

- I was a bit confused about why Section 4 was presented in the context of KL-loss. Presumably it applies to squared loss as well given the results of the previous sections?

- I have some concerns about the experiments. Building on the previous comment, it seems that part of the novelty of this paper is the use of a perturbed neural network. In the experiments you compare your algorithm using KL loss to NeuSquareCB and show an improvement. However, it feels like the natural comparison would be NeuFastCB against a variant using the KL loss where you don’t perturb. Similarly for the square loss. This “ablation”-esque study would highlight the important of the algorithmic contributions you have proposed.

Minor Typos

- theta_t in equation 4 should be theta

### Questions
See above

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
