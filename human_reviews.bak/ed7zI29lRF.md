# Learning Neural Networks with Distribution Shift: Efficiently Certifiable Guarantees

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 8

## Abstract
We give the first provably efficient algorithms for learning neural networks with respect to distribution shift. We work in the Testable Learning with Distribution Shift  framework (TDS learning) of Klivans et al. (2024), where the learner receives labeled examples from a training distribution and unlabeled examples from a test distribution and must either output a hypothesis with low test error or reject if distribution shift is detected.  No assumptions are made on the test distribution. 

All prior work in TDS learning focuses on classification, while here we must handle the setting of nonconvex regression. Our results apply to real-valued networks with arbitrary Lipschitz activations and work whenever the training distribution has strictly sub-exponential tails. For training distributions that are bounded and hypercontractive, we give a fully polynomial-time algorithm for TDS learning one hidden-layer networks with sigmoid activations. We achieve this by importing classical kernel methods into the TDS framework using data-dependent feature maps and a type of kernel matrix that couples samples from both train and test distributions.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work considers the Testable Learning with Distribution Shift (TDS) framework.

In TDS, the training and test time distributions can be different, and the learner receives labeled training examples as well as unlabeled test examples. The learner can either declare success—-in which case it must satisfy PAC-like guarantees—or it can declare failure if it detects too large a distribution shift, but this detection must be reliable.

While all previous works studied TDS in the classification setting, this work is concerned with TDS for (non-convex) regression. The authors present a general-purpose polynomial time TDS learning algorithm with theoretical performance guarantees.

The results imply TDS learnability for neural networks, under mild assumptions on the network activations, as well as a hypercontractivity assumption on the training distribution. This setting allows for the use of kernel methods and thereby essentially reduces the learning task to convex agnostic learning over polynomials.

### Strengths
- Extending the TDS framework to regression is an important contribution. As such, this paper is valuable for the broader statistical learning theory community.
- The tools employed for the proofs are technically non-trivial and could be valuable in their own right for future research in TDS. For example, the coupling of the training and test distribution is achieved through the use of data-dependent feature maps imported from the kernel framework. Additionally, the full analysis combines tools from polynomial approximation theory, hypercontractive concentration, and spectral concentration.
- The overall writing style is excellent, and the literature review seems to be thorough (though I am admittedly not familiar with the TDS line of work).

### Weaknesses
- Poor scaling of computation complexity:
While the runtime, as claimed, is polynomial in most of the parameters, it is exponential in $1/\epsilon$ for ReLU neurons and Lipschitz nets. For deep nets, the scaling is doubly exponential in the depth, and in most cases exponential in the width of the first hidden layer.
Even for the parameters that do scale polynomially, the exponents are very large and most likely quite far from being optimal. For example, for general function classes, even if one naively assumes $A,B,C,l$ all to be constant, the runtime of Algorithm 1 scales at least as $\Omega(1/\epsilon^{16})$ from constructing the matrices $\hat{\Phi}, \hat{\Phi’}$ for the spectral tester, which requires $\sim m\cdot N$ kernel evaluations.

- Very large sample complexity:
Algorithm 1 requires $m+N$ i.i.d. samples from both $D$ and $D’$, which for Lipschitz nets is doubly exponential in the depth, and exponential in $1/\epsilon$ and the width of the first hidden layer. Again, even if $A,B,C,l$ are all assumed to be constant, the sample complexity scales at least as $\Omega(1/\epsilon^{12})$ since the sample complexity is dominated by the size of the verification samples.

In the light of the above, the results have more of a proof-of-concept flavour, rather than providing a practically applicable algorithm. I suspect that Algorithm 1 cannot be ran in practise even for very small dimensions and parameter scales.

I want to make clear that I do not consider the above major weaknesses—theoretical results for complex learning settings are often difficult to obtain, and finding efficient, general purpose algorithms can take many years of research, or may not necessarily be possible in the first place, even with the best available mathematical tools.

I think that the paper would benefit from a short discussion regarding the limitations, especially regarding:
- Sample complexity (see above). Sample complexity is currently not discussed at all. I understand that this work is mostly concerned with runtime, but sample complexity should still not be completely left out of the picture in my opinion. Perhaps you could add a column in Table 1 that contains the sample complexities.
- I would suggest to mention at least once explicitly that, generally, rather large exponents are obtained for both computational and sample complexity. This limitation should be obvious to the reader even from a cursory read. It takes some effort to puzzle together the scaling exponents from Algorithm 1 and Table 2 due to the multitude of parameters involved. I suggest adding a short discussion of these limitations at the end of Section 1.1, or in a new Conclusion/Limitations section at the end of the paper.
- This is optional, but perhaps you could also highlight which steps in the analysis you believe to be particularly loose (or sharp), and what you believe the general limitations of the kernel-based approach to be. Such a discussion would make it easier for the reader to gauge the current state of this line of work, and what the major technical roadblocks are.

Note that in this iteration of ICLR, 10 pages are available, so space should not be an issue.

### Questions
- Which steps in the analysis do you believe to be particularly loose (or sharp), and what do you believe are the general limitations of a kernel-based analysis? Perhaps you could provide me with 1-2 key steps you believe to be most ripe for improvement in future work.
- In Def. 1.1, the completeness condition seems rather lax at first glance. I suspect that items 1 and 2 together imply that the algorithm must accept also in cases where $D_x$ is very close to, but not necessarily equal to, $D’_x$. Is this correct?
- In domain adaption, why is it standard to consider the training error part of the performance criterion? Intuitively, one might only care about the test error. Even if one does care about training error, why not consider a weighted combination of training and test error?

### Soundness
4

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
3

### Summary
This paper presents two TDS learning algorithms for regression problems. The first algorithm integrates kernel methods into the TDS framework, achieving fully polynomial runtime for bounded and hypercontractive training distributions. For unbounded distributions (strictly subexponential distribution), the second algorithm utilizes the moment-matching testing and the uniform approximation approaches. The results apply to classes of Lipschitz neural networks.

### Strengths
The paper is well-written, and the results are clearly presented and demonstrated. It introduces a novel TDS learning algorithm that leverages kernel methods for regression problems, which are different from the classification setting that has been discussed in the literature.

### Weaknesses
There is a lack of experimental results supporting the validity of Theorem 3.6 and Theorem 4.2.

### Questions
Is it feasible to conduct experiments and create plots to validate the theorems?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper considers the Testable Learning with Distribution Shift (TDS) framework for learning neural networks under bounded or subgaussian inputs. In the TDS framework, the learner recieves m samples $(x_i, f^*(x_i) + \text{guassian noise})$ from some known covariate distribution $D_x$, and then $m$ unlableled test samples from an **unknown and arbitrary** covariate distribution $D'_x$. If $D'_x = D_x$, then the learner must output labels for the test samples, and is evaluated on the $\ell_2$ loss. If $D'_x$ is not $D_x$, then the learner can abstain. 

The authors main technical result (Theorem 3.6) shows that if
   1. The covariates and the labels are bounded and
   2. the function class $\mathcal{F}$ is well approximated by some kernel K and 
   3. The distribution $D_x$ is *hypercontractive* with respect to K, 

then learning in the TDS framework is possible to $\epsilon^2$ mean squared error with polynomial in in the input dimension $d$ and $1/\epsilon$. A similar result (under stronger assumptions) is shown for unbounded covariates.

This is then applied to TDS for neural networks with bounded data, yielding an algorithm which runs in time polynomial in $d$ for sigmoid nets, 1-Lipshitz nets, and single ReLU units. Similar results are obtained for unbouned covariates, but the dependence is now exponential int the number of hidden neurons. 

The result is proved using standard kernel regression techniques, with some adaptation for the TDS setting: (1) the algorithm rejects if the data is unbounded and (2) The algorithm uses a spectral testor to if the empirical covariance of the kernel matrix is significantly different from the samples from $D'_x$. 

The authors prove the applications for neural networks by showing how too approximating $f^*$ (with a uniform bound on the domain) by a function in the the RKHS given by a low-dimensional polynomial kernel.

### Strengths
The TDS setting is an exciting area, and this paper extends to the TDS toolbox --- which had previously only been studies in classification settings or linear regression settings --- to the kernel regression setting, for both bounded and unbounded distributions. For unbounded distribution, this requires a novel analysis to control the approximation error on the tails of the distribution. In the bounded distribution setting, the main insight to get the results for NNs seems to be using uniform function approximations by polynomials (which is possible for neural nets with low degree polynomials!), as opposed to "sandwitching" approximations which were priorly used for TDS.

### Weaknesses
The application to neural networks does not seem to contribute much to our understanding of NNs, because it seems the main novelty is proving the specific assumptions relating to hypercontractivity, which I do not know how useful they are outside of this TDS setting. Further, the bounded distribution assumption is quite strong.

The paper could be more clear in many place:
- Pointing out where the technical novelty and main challenges are in the algorithm/analysis 
  - For example, for bounded distributions, the Algorithm seems quite standard aside from the spectral test for the TDS part, which seems like a straightforward idea. Could the authors clarify what are the main technical challenges overcome here?
 - There is nowhere in the main body of the paper explaining what new approximation theory ideas are needed for the NN applications. Are there any potential uses for the new approximation techniques outside of this paper?
- The authors should explain at the beginning of "our techniques" that they intend to approximate NNs by a low-dimensional polynomial kernel, as per other cited works. This was confusing at first to understand why the text was talking about kernels without the connection to NNs made explicit. 
- Where in Algorithm 1 is the part about finding a concise reference feature map?

### Questions
See weaknesses above.

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies distribution shift for nonconvex regression using the recently introduced Testable Learning with Distribution Shift framework (TDS) (Klivans et al.’24). The setup is as follows. A TDS  learner observe iid labeled training data from train distribution $D$, along with *unlabeled* iid test data from some other test distribution $D’$. Then, the learner can decide to abstain from predicting on the test data. If it chooses to not abstain, it must produce a hypothesis $h \in \mathcal{F}$ generalize on the test data, where generalization is defined with respect to some function class $\mathcal{F}$. 

The paper designs efficient TDS algorithms under a bounded label assumption (although this is slightly relaxed in the appendix) and sufficiently nice *training* marginals (bounded and hypercontractive or strictly subexponential), when the function class is a Lipschitz neural network. Notably, the paper makes no assumption about the test marginal. The efficient algorithms are based on kernel methods.

### Strengths
1. The paper comes up with a generic TDS framework using kernel methods, and then verifying the assumptions for a number of concrete examples. Previous algorithms in these testable settings relied on moment-based algorithms and various notions of polynomial approximation. 
2. The paper discusses the numerous assumptions on the marginals / label distributions quite extensively. There is also a matching lower bound in the appendix that shows that some assumptions on the test labels are needed. 
3. The algorithm itself is quite interesting, as it certifies a valid comparison between the training and test marginals via the spectral tester; this latter tester only works under the hypercontractivity assumption. 
4. The results in Section 4 also handle the unbounded setting, by using a moment-based testable learner using polynomial regression instead, building on the previous TDS results of Klivans et al.’24. 
5. The paper is well written, and gives substantial intuitions about how the proof works.

### Weaknesses
1. Although I appreciated that the paper does not make assumptions on the test marginal, it feels like this point is a little misleading, since the whole point of the tester is to certify the niceness of the test marginal. I can definitely agree that it is better to have a provable verifier that ensures your learner works on distribution than not, though.

### Questions
1. Can the authors comment a bit on when it would be realistic to have access to unlabeled test examples? I know that it is a standard setup for domain adaptation, but it would be good to have some discussion on it
2. The statement of Lemma A.17 seems unfinished.  
3. It seems that one needs to prove uniform convergence for the function classes used for the TDS algorithm to work. Are there any situations where we might expect to be able to surpass this barrier, as uniform convergence is quite a strong assumption 
4. Line 330, 333, should $\overline{S}_{\mathrm{ref}}$ have no overline?
5. If I have multiple test time distributions, it seems that a naive extension of the current algorithm would need to retrain my TDS learner for each test distribution? Might there be a way to design an algorithm which can be amortized over multiple different test time distributions?

### Soundness
4

### Presentation
4

### Contribution
4
