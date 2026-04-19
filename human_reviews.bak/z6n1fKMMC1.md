# An Efficient Tester-Learner for Halfspaces

- Decision: Accept (poster)
- Scores: 6, 5, 8, 8

## Abstract
We give the first efficient algorithm for learning halfspaces in the testable learning model recently defined by Rubinfeld and Vasilyan [2022]. In this model, a learner certifies that the accuracy of its output hypothesis is near optimal whenever the training set passes an associated test, and training sets drawn from some target distribution must pass the test. This model is more challenging than distribution-specific agnostic or Massart noise models where the learner is allowed to fail arbitrarily if the distributional assumption does not hold. We consider the setting where the target distribution is the standard Gaussian in $d$ dimensions and the label noise is either Massart or adversarial (agnostic). For Massart noise, our tester-learner runs in polynomial time and outputs a hypothesis with (information-theoretically optimal) error $\mathrm{opt}+\epsilon$ (and extends to any fixed strongly log-concave target distribution). For adversarial noise, our tester-learner obtains error $O(\mathrm{opt})+\epsilon$ in polynomial time. Prior work on testable learning ignores the labels in the training set and checks that the empirical moments of the covariates are close to the moments of the base distribution. Here we develop new tests of independent interest that make critical use of the labels and combine them with the moment-matching approach of Gollakota et al. [2022]. This enables us to implement a testable variant of the algorithm of Diakonikolas et al. [2020a, 2020b] for learning noisy halfspaces using nonconvex SGD.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors give the first computationally efficient tester-learner for learning halfspaces where the target distribution is the d-variate Gaussian and the label noise is Massart or adversarial. The tester-learner framework was recently proposed as a generation to the distribution-specific learning setting where the algorithm needs to accept a dataset whenever it comes from a target distribution and needs to achieve the agnostic learning guarantee (error = $opt + \varepsilon$) whenever it accepts. Previous work only gave a sample-optimal tester learner for the same problem which was not computationally efficient. The main technical novelty in this work is tester that looks at the labels as opposed to the label-oblivious testers previously designed. For the adversarial noise setting, the authors achieve the suboptimal risk $O(opt) + \varepsilon$.

The authors build on the non-convex optimization approach of [DKTZ] which uses the a smoothed version of the ramp loss as a surrogate to the zero one loss. Although this is a non-convex function, it was shown that the stationary points are good solution which can be recovered by projected SGD as the first step under the Gaussianity assumption. In the testing-learning framework, we need to additionally check the following assumption: the probability masses of certain regions are proportional to their geometric measures. The tester checks local properties of the distribution in regions described by the stationary points using moment matching techniques. Naively, such a check could only guarantee the empirical mass is additively close to the true mass. However, using a refined moment test conditioned on a band based on the stationary vector (similar to the existing localization-based refinement techniques of Awasti et al. 2017) they could get the stronger multiplicative guarantee. This allows them to argue that if the test passes, the stationary points will indeed be close to the true weight vector in angular distance. This in turn means the returned vectors are good solutions using properties of Gaussian. The later step results in a larger error for the adversarial noise setting as opposed to the Massart noise setting.

### Strengths
The proposed work is an interesting combination of several technical ingredients that have been developed in learning theory for learning halfspaces and testing distributions such as non-convex optimization, fooling functions of halfspaces, and moment-matching tests. Moreover, they achieve the desired polynomial runtime for halfspaces in the newly proposed testing-learning framework.

### Weaknesses
The presentation could have been better. The paper has several forward references, that too from the main body to the appendix, which makes it slightly hard to follow.

### Questions
- Are the constants involved in the complexity very big? Given that ICLR accepts experiments and the dataset is easy to synthesize, how hard is it to implement and test the claimed efficient algorithm? This may be a general question targeted to even some of the prior works as well.
- I believe the results easily extend to non-homogeneous halfspaces where there is a constant offset term?
- I believe only the tester T3 uses the labels to check the fooling and T1 and T2 does not in Algo 1? Small typo: the Run T2 step has $\sigma$ missing in $B'_W(\sigma)$.

### Soundness
4 excellent

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed a polynomial-time algorithm for learning halfspaces on testable fixed well-behaved distributions under Massart and adversarial noise. Unlike its prior works, it takes the labels into account and checks local properties of the distribution by testing the moments of the conditional distributions around the stationary points.

### Strengths
The framework of testable testing was recently proposed and has drawn great attention in the research community. This paper proposes a polynomial time algorithm for learning halfspaces under noisy settings, while the distributional assumptions are replaced by a tester on a fixed distribution. The paper is well-written. The technical parts look sound.

### Weaknesses
As mentioned in the paper, its subsequent work, “Tester-learners for halfspaces: Universal algorithms” has shown a more general tester-learner with stronger guarantees. This largely weakened the merit of publishing the work.

### Questions
Can you justify the unique value of this paper given the subsequent work has shown strictly stronger guarantees?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper worked on the problem of learning Gaussian Halfspace (with extension to more general stongly logconcave distributions) with Massart noise and agnostic noise, under the Tester-Learner models. The authors provided the first tester-learner algorithm with polynomial iteration and sample complexity, that achieves $\mathrm{OPT} + \epsilon$ error for the Massart noise and $O(\mathrm{OPT}) + \epsilon$ error for the agnostic noise (under Gaussian marginal). The technical contriutions of this paper are mainly the following: the authors devised more efficient testers using information of labels and exploiting the local geometric structure of the distribution (the condition probability on a band $P[v\cdot x \in [\alpha, \beta] | w\cdot x\in[-\sigma,\sigma]]$); they also showed that for some carefully designed loss function $\mathcal{L}_\sigma$, its stationary points $w$ are also vectors that are close (in angle) to the optimal solution $w^*$, under some distributions that are efficiently testable.

### Strengths
The paper contributes rigorously to the field of robustly learning halfspaces, and has some very interesting results. 
1. Based on the results from DKTZ20a and GKK23, the authors devised new algorithms that are more efficient comparing to prior works. This includes a new loss function that work better for the specific task, and some new structrual results linking the gradient norm of this loss to the angle between the parameter $w$ and the optimal halfspace $w^*$.
2. The authors used some local property of the distribution that enables them to get desired result using testers that achieves only constant error rather than $\epsilon$ error.
3. These results can further extend from Gaussian distribution to strongly logconcave distribution, and get simialr results (at least for Massart noise).
4. The authors finally get the first polynomial tester-learner algorithm for learning Gaussian halfspaces under massart and agnostic noise.
5. The paper is clear and contains useful explanation on the intuiation of the algorithm.

### Weaknesses
I think there is no obvious weakness in general.

### Questions
1. I am confused why the algorithm needs two $T_3$ testers with different accuracies, $\sigma/6$ and $\sigma/2$?
2. I am not very familiar with tester-learner models. Are there lower bounds on learning gaussian halfspaces under massart/agnostic noise for tester-learner algorithms? Are tester-learner algorithms SQ algorithms?
3. In algorithm 1, what exactly is the function class $\mathcal{F}_{w'}$? How to choose the weights that are orthogonal to $w'$?

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper is a further exploration of the testable learning framework proposed by Rubinfeld and Vasilyan. The main feature of this framework is that it requires learning algorithms that learn near-optimal predictors whenever the input training sample passes a test (soundness), and also training samples pass the test whenever the distributional assumptions are met (completeness). 

The paper gives polynomial-time testable learning algorithms for halfspaces when the marginal distribution is isotropic log-concave and:
(a) under Massart noise, guarantee error at most OPT + \epsilon. [Theorem 4.1]
(b). under adversarial noise, guarantee error at most O(OPT) + \epsilon. [Theorem 5.1]

### Strengths
This is a good paper that significantly extends what is known to be achievable in the testable-learning framework. To establish these results, the paper contributes new testing procedures that go beyond the limitations of prior work (Gollakota, Klivans, Kothari, 2023). Additionally, the techniques of this paper have also led to more general results in testable-learning (Gollakota, Klivans, Stavropoulos, Vasilyan, 2023). 

The paper is well-written and easy to read. The authors do a great job discussing prior work, and how the paper fits with related literature.

### Weaknesses
The results may be a little limited in retrospect. In particular, the paper (Gollakota, Klivans, Stavropoulos, Vasilyan, NeurIPS 2023) already has more general results, including the results of this paper. If I understood correctly (based on page 2, subsequent work paragraph), there is some non-overlap in the techniques used in both papers, and so this paper may still be beneficial to the community.

### Questions
It would be great if the authors could discuss further the contributions of this paper in light of subsequent work (Gollakota, Klivans, Stavropoulos, Vasilyan, NeurIPS 2023). In particular, can the authors make a case for why the contributions in this paper are beneficial/useful given that more general results have already been published.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good
