# Uncertainty Herding: One Active Learning Method for All Label Budgets

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 8, 6

## Abstract
Most active learning research has focused on methods which perform well when many labels are available, but can be dramatically worse than random selection when label budgets are small.
Other methods have focused on the low-budget regime, but do poorly as label budgets increase.
As the line between "low" and "high" budgets varies by problem,
this is a serious issue in practice.
We propose *uncertainty coverage*,
an objective which generalizes a variety of low- and high-budget objectives,
as well as natural, hyperparameter-light methods to smoothly interpolate between low- and high-budget regimes.
We call greedy optimization of the estimate Uncertainty Herding;
this simple method is computationally fast,
and we prove that it nearly optimizes the distribution-level coverage.
In experimental validation across a variety of active learning tasks,
our proposal matches or beats state-of-the-art performance in essentially all cases;
it is the only method of which we are aware that reliably works well in both low- and high-budget settings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper interpolates between the low-budget and the high-budget regime active learning (AL) by an objective called uncertainty coverage. The proposed algorithm Uncertainty Herding is a combination of the low-budget AL algorithm MaxHerding (Bae et al. 2024) and the classical uncertainty sampling in a way that the objective in MaxHerding is re-weighted by an uncertainty function. There are three improvements other than the objective: the temperature scaling (to weaken the effects of uncertainty functions when low-budget), the radius adaptation (to narrow down the coverage when high-budget), and the entropy regularization. The authors conduct experiments showing the superiority of the algorithm.

### Strengths
1. The proposed algorithm is a natural interpolation between the low-budget MaxHerding and the high-budget uncertainty sampling. The authors provide a smart way (temperature scaling and radius adaptation) to decide each algorithm's proportion and also prove the equivalence in an asymptotic sense.
2. The algorithm has reached a performance surpassing the existing benchmarks, especially the other attempts to interpolate between the low-budget and the high-budget regimes.
3. The algorithm's computational burden is acceptable.

### Weaknesses
1. (Major) The implemented algorithm in the main experiments consists of three improvements compared to the naive Uncertainty Herding: the temperature scaling, the radius adaptation, and the entropy regularization. From my point of view, this paper's main contribution is giving a natural (and probably SOTA) interpolation between the low-budget and the high-budget AL regimes. In that sense, the first two terms seem very natural, yet the third term is unnatural and not analyzed in the paper.  On the one hand, the third term is important (and sometimes even critical) since the third term does boost the performance of the proposed algorithm. For example, in Appendix C, without entropy-regularization, the performance of Uncertainty Herding does not exceed uncertainty sampling in the high-budget regime. Intuitively, without that regularization, it would be weird that the algorithm (as an interpolation of the MaxHerding and the margin-based uncertainty sampling) outperforms both the MaxHerding and the margin-based uncertainty sampling in EVERY regime. However, the ablation study only considers adding entropy-regularization to the random sampling (passive learning). The authors make no attempts to explain or identify the effects of the entropy-regularization, which makes the logic incomplete.
2. (Moderate) The theory part does not contain much contribution and novelty.
3. (Moderate) There seem to be no descriptions of how the temperature scaling changes the uncertainty function. For example, the $f_{\tau}$ function in Definition 5 is used without definition.
4. (Minor) Figures in the main text are too small.

### Questions
1. The entropy-regularization: Is it just an engineering trick or a vital part of active learning? Would you mind conducting some further ablation studies, for example, combining entropy-regularization with other active learning algorithms (the MaxHerding and the margin-based uncertainty sampling)?
2. The framework: This paper unifies other hybrid methods in the uncertainty coverage objective by choosing different uncertainty functions and kernels. Does this framework provide any (theoretical or empirical) insights into selecting the uncertainty functions or the kernels in practice?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes an active learning method suitable for both low- and high-budget regimes. It defines an "uncertainty coverage" metric, estimated through its empirical counterpart. An uncertainty herding method is then introduced, based on the greedy optimization of this estimated uncertainty coverage objective. Numerical experiments are conducted within a transfer learning application.

### Strengths
- This paper is largely well-written, with extensive discussions on the existing literature.
- The numerical experiments are well-executed.

### Weaknesses
- The proposed method seems to rely more on a combination of heuristics and lacks guiding principles. Some analysis is also presented only in the asymptotic regime. In particular, Proposition 4 requires $\sigma\rightarrow 0$.
- A more intriguing question is that the paper does not provide a detailed recipe for choosing the design or detecting whether the budget is high or low.
- In terms of writing, although this paper builds on a line of literature, the exposition is far from self-contained; some notation and basic definitions are absent from Section 3.1.
- Section 3.4 is poorly organized. Instead of simply citing lengthy theorems, it would provide more insights by including a coherent analysis and direct comparisons.
- Figures 3 and 4: The plots use the same markers for different methods.

### Questions
See weakness.

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
2

### Summary
The authors propose a method to adapt to different budget levels in active learning. They theoretically prove that the proposed method achieves near optimal distribution level coverage. Empirically, they show that their method outperforms state of the art.

### Strengths
The theoretical results provide insights into algorithm design, especially how to make proposed method robust across different budget levels.

### Weaknesses
in figure 5, there is a sudden increase in difference in accuracy from 7.2k to 8k budget. This increase applies to all methods and goes against the general trend observed in figure 5, 6, and 7. Could the authors provide some insight on why that may happen? A similar hikes happens in figure 6(a). 

I suspect this is due to the small amount of random seeds used. The error bars in figure 6(a) are quite high, and these hikes might be caused by an outlier point. Could the authors provide results for at least a subset of experiments with 10 or more random seeds and report the median? That may better illustrate the trend.

### Questions
can the authors make the legend in figure 1(a) larger and more readable?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a novel active learning method called Uncertainty Herding (UHerding), which addresses challenges respect to label budget in active learning regimes. Traditional active learning methods often consider either low- or high-budget contexts but not both. This method bridges this gap by incorporating an uncertainty coverage objective, which generalizes both low- and high-budget objectives. This adaptive method achieves state-of-the-art results across various benchmarks, providing a reliable approach for tasks requiring both representation-based and uncertainty-based selection strategies.

### Strengths
UHerding successfully addresses both low- and high-budget scenarios in a unified approach, eliminating the need to switch frameworks and tackling the practical challenge of unclear budget boundaries. This method is also practical, as it doesn’t demand high training costs. This innovative approach is a strong contribution.

The paper provides rigorous theoretical analysis into the estimation quality and parameter adaptation of UHerding. Experiments span multiple datasets and scenarios, showing UHerding’s good performance compared to existing methods across various budget levels.

### Weaknesses
The method relies on having suitable pre-trained feature extractors and an accurate approximation of the data distribution, which might not always be feasible in real-world situations. 

While it shows strong performance in supervised tasks, its effectiveness in transfer learning could benefit from additional validation against approaches tailored to specific domains.

### Questions
1. How sensitive is UHerding to changes in the adaptation parameters, particularly when applied to diverse datasets or models? Would a suboptimal parameter choice significantly affect performance?

2. How well does UHerding perform in low-data or imbalanced data scenarios in practice, where pre-trained features may not fully capture distributional or distinguishable differences?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies pool-based active learning by developing a unified approach that performs well in both low-budget and high-budget regimes. While uncertainty-based approaches typically work better with higher budgets and representation-based methods with lower budgets, the paper proposes an algorithm that adaptively interpolates between the two. This is achieved through greedily selecting data points based on a notion called uncertainty coverage, which incorporates uncertainty into a representation-based objective. Empirically, the paper shows that the algorithm achieves or exceeds state-of-the-art performance across budget regimes.

### Strengths
- The problem of navigating active learning strategies across budget regimes seems interesting and relevant to the field, as highlighted by recent work [Hacohen and Weinshall, 2023].

- Balancing uncertainty and representativeness has been explored before in active learning, as discussed by the authors. The notion of uncertainty coverage introduced in this paper is built upon that of generalized coverage studied in [Bae et al., 2024] -- it is defined by weighting generalized coverage with the model's uncertainty. This idea is simple, intuitive, and seems versatile; the paper also briefly discusses connections to existing hybrid methods.

- The proposed approach seamlessly interpolates between representation-based and uncertainty-based approaches, without relying on a fixed budget threshold to switch behaviors (cf., [Hacohen and Weinshall, 2023]). The practical algorithm is also supported by some theoretical evidence such as approximation guarantees.

- The empirical validation of the algorithm demonstrates the interpolation, and the performance on various tasks/datasets seems promising across budget regimes.

### Weaknesses
- The theoretical results provide some insights but are primarily limited to a generalization guarantee for the empirical estimator of UCoverage and an approximation guarantee of the greedy algorithm. It would be interesting to see a deeper analysis that directly addresses the core goals in active learning, i.e., reducing error rates with as few labels as possible. For example, how does UCoverage perform as an objective in active learning in terms of *excess risk* and *label complexity* (potentially assuming oracle computation)?

- There does not seem to be an empirical comparison with SelectAL and TCM.

- The paper is generally well-written and easy to follow. However, there are a few ambiguities that may lead to confusion. For example, the term "budget" is never clearly defined, except in Algorithm 1, where $B$ is referred to as the budget. If this is the case, the discussion on parameter adaptation in Section 3.2 and parts of the experimental setup may be confusing, as the algorithm's behavior seems to depend on the current size of the labeled data rather than directly on $B$. Even if $B$ is small, $\mathcal{L}$ still grows over time and can be large later on. This raises the question of whether the budget refers to the total number of labeled data over $T$ steps, which was not clarified. I will also outline additional questions in the following section.

### Questions
- There is a discussion about discrete and continuous budget regimes in the comparison with SelectAL. In particular, it is mentioned that this paper cover continuous budget regimes. What do these regimes mean?
- There isn't a formal definition of the kernel function $k$ outside Proposition 4. Do you need to assume that $k$ is in a particular form and has a "spread" parameter, $\sigma$? Would the approach accommodate, e.g., polynomial kernel and sigmoid kernels? This is in reference to Definition 1, the discussion after Theorem 2, Algorithm 2, and the discussion before Proposition 4.
- In equation (1), it would be helpful to clarify what $N$ is.
- In Algorithm 1, what is ECE in line 2? Should $U(x_n)$ in line 5 be $U(x_n; f_{\tau^*})$?
- Can you elaborate on the computational complexity of Algorithm 1?
- The notation in Proposition 3 can be confusing -- what does $U(x;f) \rightarrow c$ exactly mean? Shouldn't $U(x;f)$ tend not to behave like constants as the model $f$ improves?
- Regarding Section 3.4, would you argue that the framework presented in this paper generalizes some existing methods?
- In Figure 3 (high-budget), why does the relative accuracy of UHerding drop as more labels are available?

### Soundness
3

### Presentation
3

### Contribution
3
