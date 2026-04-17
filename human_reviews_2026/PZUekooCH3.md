# RPWithPrior: Label Differential Privacy in Regression

- Decision: Reject
- Scores: 4, 2, 6, 4, 4

## Abstract
With the wide application of machine learning techniques in practice, privacy preservation has gained increasing attention. Protecting user privacy with minimal accuracy loss is a fundamental task in the data analysis and mining community. In this paper, we focus on regression tasks under $\epsilon$-label differential privacy guarantees. Some existing methods for regression with $\epsilon$-label differential privacy, such as the RR-On-Bins mechanism and its variant, discretized the output space into finite bins and then applied randomized response (RR) algorithms. To efficiently determine these finite bins, the authors rounded the original responses down to integer values. However, such operations does not align well with real-world scenarios. To overcome these limitations, we model both original and randomized responses as {\it continuous} random variables, avoiding discretization entirely. Our novel approach estimates an optimal interval for randomized responses and introduces new algorithms designed for scenarios where a prior is either known or unknown. Additionally, we prove that our algorithm, RPWithPrior, guarantees $\epsilon$-label differential privacy. Numerical results demonstrate that our approach gets better performance compared with the Gaussian, Laplace, Staircase, and RRonBins, Unbiased mechanisms on the Communities and Crime, Criteo Sponsored Search Conversion Log, California Housing datasets and some simulated datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the label differential privacy problem in regression tasks and proposes a label encryption mechanism named *RPWithPrior*. The core innovation lies in abandoning the traditional discretization approach and adopting a modeling method based on continuous random variables. Experimental results on real-world datasets demonstrate that the method achieves excellent performance.

### Strengths
1. The approach discards the mainstream discretization method and avoids discretization errors by directly modeling continuous random variables. 
2. By estimating the optimal interval $[A_1, A_2]$ and constructing a randomized interval, the mechanism can protect the neighborhood relationships of common response values, while allocating probability in the extended interval to meet privacy requirements.
3. Section 4 considers scenarios with an unknown prior distribution, using a histogram perturbed by the Laplace mechanism to approximate $f_Y(\cdot)$, thereby enhancing the method's practicality.
4. Experiments on multiple real-world datasets show that this method consistently achieves top-tier performance.

### Weaknesses
1. The current theoretical analysis is insufficient to demonstrate the method's superiority fully. 
2. Only experiments on public real-world datasets are presented, simulated data experiments (e.g., generating linear and non-linear datasets) to more intuitively demonstrate the method's effectiveness are not included.

### Questions
1. Theoretically, why is it specifically a novel label DP mechanism for regression tasks?  It is recommended to include a more in-depth analysis of theoretical properties.
2. The authors claim the method has lower time complexity. However, the computational complexity analysis in Section G.5 is inadequate. It is recommended to include a table comparing the **theoretical time and space complexity** of various methods.
3. The paper does not detail the tuning strategy for hyperparameters (e.g., $\delta，\sigma$ ). It is recommended to clearly explain the tuning methodology and associated computational cost. As the authors note, a smaller $\sigma$ allows the histogram to more accurately approximate the distribution from the previous stage, but this comes at the cost of increased computational complexity.
4. Regarding the "unbiasedness" mentioned in the conclusion, $\mathbb{E}[\mathcal{M}(y)] = y$. Is this property self-evident? The claim of unbiasedness, $\mathbb{E}[\mathcal{M}(y)] = y$, requires further clarification or proof. Is this property rigorously upheld by the proposed algorithm?
5. Can the optimal interval method (extending from a **one-dimensional interval** to a **multi-dimensional ball**) be more readily generalized to multi-dimensional settings? Could this lead to further theoretical extensions?

minor issue:

6. line 56-59 leaves a large area of blank.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces an extension of RPWithPrior to continuous responses, achieving label differential privacy (label DP) in regression tasks. Unlike existing methods that discretize the continuous output/label space into bins, this work models both the original and randomized labels as continuous random variables. The core idea is to define an optimal interval [A1, A2] based on a prior distribution of the labels; the mechanism then decides whether the response should lie within this interval or not. The authors provide theoretical proofs for the $\epsilon$-label DP guarantee and demonstrate through experiments on three datasets that their method outperforms baseline mechanisms (Gaussian, Laplace, Staircase, and RRonBins) in terms of test Mean Squared Error (MSE).

### Strengths
1. This paper proposes a new label-DP mechanism for continuous random variables.

2. The theoretical results are straightforward and seem correct, even though I have not checked the proofs.

3. The experiments are clean, and the proposed mechanism improves empirical performance according to the experimental results on the Criteo Sponsored Search Conversion Log dataset.

### Weaknesses
1. The contributions of this paper are significantly over-claimed. In fact, the main contribution is proposing a new discretization of a continuous random variable and then applying Randomized Response (RR) to the discretized response. However, the authors claim that "this work presents the first introduction of continuous random variables," despite similar discretization methods already existing in prior research, such as Badanidiyuru et al., 2023. The abstract and introduction repeat the same high-level contributions multiple times, which is redundant and frustrating.

2. The introduction is poorly written and contains limited information. Furthermore, some mathematical definitions should be moved to a preliminary section.

3. The main contribution is poorly written, making it difficult to understand. Many mathematical notations are not explained.

a) Please explain the motivation for formalizing the problem as (3.3).

b) The objective function $F(A_1,A_2)$ is not introduced when it first appears in Section 3.2. The authors should also explain why it takes the form given in Lemma 3.5.

c) The term "optimal interval" is not clearly defined.

4.The conclusion is not self-contained. The mathematical writing is dense and often inconsistent, creating unnecessary barriers. For example, the authors claim to focus on continuous responses, yet in Lemma 3.5, they assume the pdf of $y$ is a step function, implying a discrete distribution. This inconsistency may undermine the paper's main contribution.

5.  The choice of the $\delta$ parameter for the intervals seems very important, but the authors do not provide details on how to select it. Moreover, in the DP literature, $\delta$ typically refers to the parameter in $(\epsilon,\delta)$-DP, so it would be better to use a different Greek letter.

6. The method for estimating an unknown prior relies on a simple histogram of the Laplace-noised data. This approach can be inaccurate is sensitive to the histogram's bin width ($\sigma$). While Remark 4.1 acknowledges this trade-off, the paper does not explore more sophisticated density estimation techniques that could improve performance.

7. The theoretical analysis relies on one-dimensional calculus, and as noted, discretizing a continuous response is not novel. Therefore, the theoretical contribution is insufficient. My publication recommendation mainly relies on whether the empirical contribution is strong enough.

8.  The empirical contribution is too limited, which is one of the main reasons for my rating.

a) First, the experiments provide limited comparisons to state-of-the-art methods. The authors cite Badanidiyuru et al., 2023 but do not include it in their experimental comparisons.

b) Second, the experiments are conducted on only one simple real-world dataset. Compared to existing works like Badanidiyuru et al., 2023, the experimental investigation is inadequate. 

The authors should compare their method to Badanidiyuru et al., 2023 on more datasets.

### Questions
See the weaknesses part

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies label differentially private regression, which assumes public feature and private labels. Prior work uses randomized response on discretized labels(RR-on-Bins), and then train on noisy labels. This paper proposes to sample the noisy labels from a piecewise function, where the interval are choosing by optimizing an objective function using prior distribution. This label randomizer concentrates its mass on a neighborhood on the y, and the spread some probability mass over an optimal interval computed from prior. My understanding is the main benefit compared to prior work is to reduce the discretization-induced errors. The theoretical results is mostly on privacy proof. And the utility is demonstrated by experiments on real datasets.

### Strengths
1. Prior label-DP regression mechanisms typically discretize the continuous label and apply randomized response over a finite set of bins, which introduces quantization error; using a continuous non-additive randomizer directly on R removes that source of loss.
2. The paper reports consistent gains against additive baselines (Laplace/Gaussian/Staircase) and discrete non-additive baselines (RR-on-Bins and variants) at comparable epsilons, suggesting the continuous construction can be practically superior.

### Weaknesses
1. It’s not obvious why this is fundamentally better than RR-on-Bins: the method still depends on a prior summarized by a histogram and on selecting an interval. If the interval search effectively scans over endpoints induced by k histogram bins (potentially k^2 pairs) there’s a time–accuracy trade-off tied to k that isn’t fully analyzed. Clarifying how interval selection avoids a hidden discretization dependence in the beginning would help audience to understand this better
2. Prior work gives unbiased label randomizers and studies bias–variance trade-offs. this paper acknowledges it does not enforce unbiasedness

### Questions
1. I am confused by the claim that this is first non-additive mechanism. Prior label-DP regression already used non-additive RR-on-Bins. Is the novelty specifically “continuous non-additive”?

2. Beyond the qualitative argument “no quantization,” can you give a formal or empirical decomposition (e.g., bias from binning + stochastic error) showing when continuous non-additive strictly dominates additive noise-adding mechanisms and RR-on-Bins?

3. For highly skewed/long-tailed label distributions (e.g., ad-click or revenue labels), do you have distributional assumptions under which your mechanism yields a provable MSE (or regret) advantage over RR-on-Bins?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new mechanism RPWithPrior, for achieving $\epsilon-$label
differential privacy (LabelDP) in regression tasks. Unlike previous approaches such as
RR-on-Bins or its unbiased variants, which discretize the continuous label space before
applying randomized response (RR), the proposed method models both the original
and randomized responses as continuous random variables and determines an optimal
perturbation interval $[A_1, A_2]$ based on either known or estimated prior distributions.
The paper provides theoretical proofs that the proposed mechanism satisfies $\epsilon-$LabelDP
and validates its effectiveness on three regression datasets, showing lower mean squared
error (MSE) compared to Gaussian, Laplace, Staircase, and RR-on-Bins mechanisms.

### Strengths
1. The idea of formulating LabelDP regression in a continuous space rather than
discretized bins is novel.

2. Theoretical results (Propositions 3.1, Lemmas 3.3–3.5) are clearly stated, showing
that the proposed density function satisfies $\epsilon-$LabelDP.

3. The method extends naturally to the case where the prior distribution is unknown,
providing a practical histogram-based alternative.

4. Experiments across multiple datasets (Communities & Crime, Criteo, California
Housing) demonstrate consistent improvements in MSE over existing mechanisms.

### Weaknesses
1. The prior $f_Y (y)$ is central to the algorithm (Section 3.2) yet poorly defined. It
is unclear whether this represents: (a) an empirical label density estimated from
private data, (b) a fixed external prior, or (c) a Bayesian prior distribution. In
the case of the histogram-based extension (Section 4), privacy and accuracy both
depend critically on the quality of prior estimation. However, the paper provides
neither a sensitivity analysis with respect to estimation error nor a discussion of
how bias in the histogram affects privacy utility.

2. Theoretical derivations reproduce similar privacy constraints to Ghazi et al. (2022)
and Badanidiyuru et al. (2023). The proposed mechanism essentially replaces
discrete sampling with piecewise uniform densities, which still yield the same order
of privacy–utility tradeoff. There is no formal comparison or proof of improved
efficiency or optimality.

3. Algorithm 1 states an $O\left(k^2\right)$ computational complexity (Sec. 3.3), where $k$ denotes the number of candidate interval boundaries derived from the prior partition. However, $k$ is not clearly defined and may scale with the number of histogram bins or data points, which could make the practical cost much higher than implied. Moreover, Section 5 reports only accuracy metrics and omits any runtime or memory comparison with RR-on-Bins or Laplace mechanisms, so the claimed "efficiency" remains qualitative rather than quantitatively supported.

4. The experiments evaluate the method only under the uniform privacy-noise model implied by Eq. (3.4). The authors do not test how the algorithm performs when the true label or perturbation distributions deviate from uniformity (e.g., Gaussian or heavy-tailed noise). Moreover, the parameter $\delta$, which defines the perturbation range, is fixed but never analyzed or justified. Consequently, the robustness of the proposed mechanism to distributional or hyperparameter misspecification remains unclear.

### Questions
1. How sensitive is performance to the histogram bin width $\sigma$ in Algorithm 4? Would
too coarse or too fine bins degrade utility or violate privacy due to poor prior
approximation?

2. How is the total privacy budget composed between Laplace prerandomization and
RPWithPrior sampling? Is it strictly additive $(\epsilon_1 + \epsilon_2)$ or governed by a tighter
composition theorem?

3. Can the method extend to multivariate regression or non-scalar labels, and if so,
how would the conditional density generalize?

4. Have you considered comparing to Badanidiyuru et al. (2023) empirically to assess
unbiasedness and efficiency directly?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies the problem of optimizing randomized response mechanisms over a continuous domain $\mathcal{Y}$ under both known and unknown prior settings, with applications to regression under label-level differential privacy.

In the known prior setting, the paper assumes a prior distribution over $\mathcal{Y}$ whose density function $f_Y(y)$ is piecewise constant. 
Based on this assumption, it formulates an optimization problem for designing a “good” differentially private perturbation mechanism and proposes an algorithm to find the optimal solution. 
The algorithm runs in time quadratic in the number of pieces of the piecewise constant utility function.

In the unknown prior setting, the paper proposes to first estimate the prior (as a piecewise constant density function). However, the privacy and utility guarantees in this section are not clearly stated or analyzed.

### Strengths
The proposed optimization problem admits an optimal solution, and the paper presents a polynomial-time algorithm to compute it.

### Weaknesses
The paper is not well written.  

Moreover, the privacy and utility guarantees of some of the proposed algorithms are unclear and cannot be  verified from the current presentation.

### Questions
In Section 4, the paper proposes using $\mathcal{M}_{\epsilon, \text{Lap}}(y)$ to protect $y \in \mathcal{Y}$.
However, this part is not clearly explained:

1. What is the size or structure of $\mathcal{Y}$?  
2. How large can $|y - y'|$ be, where $y'$ is obtained by replacing $y$ with its counterpart in a neighboring dataset?
   Specifically, can $y'$ be any arbitrary data point in $\mathcal{Y}$?
   If $\mathcal{Y} = \mathbb{R}$ and $y'$ can take any value in $\mathbb{R}$, then the Laplace mechanism would require infinite variance, resulting in meaningless utility guarantees.

**Additional comments:**

3. In line 253, the reference to Equation (E.1) appears to be a typo—it should likely refer to Equation (3.6).  
   
4. It is unclear how the parameter $\delta$ is chosen in Lemma 3.3. 
   
5. It would strengthen the paper to include a clearer motivation for optimizing the objective function in Section 3.1.  
   The paper provides details on *how* to perform the optimization, but offers little explanation of *why* this formulation is meaningful. 
   For example, what is the underlying rationale for optimizing $F(A_1, A_2)$, and how does it connect to the overall utility guarantee of the mechanism?

### Soundness
2

### Presentation
2

### Contribution
2
