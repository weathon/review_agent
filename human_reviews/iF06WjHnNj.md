# Near-optimal Active Regression of Single-Index Models

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 5, 5

## Abstract
The active regression problem of the single-index model is to solve $\min_x \lVert f(Ax)-b\rVert_p$, where $A$ is fully accessible and $b$ can only be accessed via entry queries, with the goal of minimizing the number of queries to the entries of $b$.
When $f$ is Lipschitz, previous results only obtain constant-factor approximations. This work presents the first algorithm that provides a $(1+\varepsilon)$-approximation solution by querying $\tilde{O}(d^{\frac{p}{2}\vee 1}/\varepsilon^{p\vee 2})$ entries of $b$. This query complexity is also shown to be optimal up to logarithmic factors for $p\in [1,2]$ and the $\varepsilon$-dependence of $1/\varepsilon^p$ is shown to be optimal for $p>2$.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies the problem of learning single-index models in the active learning setting, where given a dataset $(A,b)$ and an activation function $f$, the goal is to output some $\hat{x}$ by querying the entries of $b$ such that $||f(A\hat{x})-b||_p$ is close to $opt$, the error of the best $x \in R^d$. Previous works that study this problem give several upper bounds for the query complexity for different $p$, but can only approximate $opt$ within a constant factor of $C$. In this work, the authors develop active learning algorithms for this problem with different $p$. For $p \in [1,2]$, the query complexity of the algorithms is $\tilde{O}(d/\epsilon^2)$, which matches the lower bound provided by this paper. For $p>2$, the query complexity is $\tilde{O}(d^{p/2}/\epsilon^p)$. In this regime, the paper also gives a lower bound of $\tilde{\Omega}(d/\epsilon^p)$.

### Strengths
Learning single-index models and active learning are both central problems in learning theory. Compared to active learning for classification problems, active learning for regression problems is much less understood. This paper gives a nearly optimal query complexity for the corresponding $l_p$ regression problem for a wide range of $p$. The algorithm builds on a similar sample and learn frameworks used by previous works but uses several solid and new ideas to analyze and improve the error guarantee and the query complexity. The results and new techniques would be of interest to ML and theory communities.

### Weaknesses
I think one main weakness of the paper is its presentation. The paper itself is very technical and involves deep mathematical tools such as  Lewis weights that are frequently used for other problems such as subspace embedding. As a person who does not work on these problems and is not familiar with these tools, I found sometimes it hard to go over the paper as the related background is not clearly provided in the main body. Furthermore, I feel that to fully understand the techniques, the paper requires the reader to be familiar with the line of the previous work. For example, in line 219-line 226
>we obtain the following concentration bound .... which has been the central tool in the work on subspace embeddings... In short, when the number of queries is m...

These statements are confusing for a reader who is not familiar with the technique and related literature. It is hard to get an intuition of how to relate the query complexity with the previous error analysis.

2. Another weakness I found is the computational complexity of the problem. I think to implement the algorithm in the paper, one has to solve a sub-optimization problem, which I believe has a super-polynomial time complexity with respect to the error parameter $\epsilon$. In fact, there have been works that show that agnostic learning single index model is usually computationally hard. (see my questions below) Given these hardness results, I am not sure whether it is still motivated to pursue $(1+\epsilon)$-approximation, such a strong guarantee. I think adding some numerical simulations for the algorithm would be helpful to show the algorithm is useful and it is still reasonable goal to achieve $(1+\epsilon)$-approximation for some reasonable choice of $\epsilon$.

### Questions
1. I would like to suggest the authors improve the presentation and move some related background from the appendix to the main body of the paper. 

2. There have been papers such as [DKR23] that show that even if the data is sampled from a Gaussian and the activation function $f$ is a ReLU, it is computationally hard to achieve $opt+\epsilon$. Can you help explain the relation between these hardness results and the results in this paper? Given the hardness result, why it is still motivated to design algorithms that learns a $(1+\epsilon)$-approximate solution?




>Diakonikolas, Ilias, Daniel Kane, and Lisheng Ren. "Near-optimal cryptographic hardness of agnostically learning halfspaces and relu regression under gaussian marginals." International Conference on Machine Learning. PMLR, 2023.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper considers the problem of active regression where the objective is to solve the problems of the $\min \|f(Ax) - b\|_p^p \|$ where $f$ is an unknown Lipschitz function, $A$ is a known matrix. The learner aims to solve the minimization problem by querying only a fraction of elements of $b$. The problem has been well studied for $f(x) = x$. For the case of general $f$, existing results only obtain a constant factor approximation. In this work, the authors obtain a $(1 + \varepsilon)$ approximate solution with general $f$ with a new algorithm and analysis.

### Strengths
I think this is an interesting paper overall. While the problem appears to be simple, there is quite of bit of machinery involved in establishing the results and designing the algorithm. I believe there are several novel aspects in the algorithm design and analysis that might be useful in general for future studies in this direction.

### Weaknesses
I don't see any glaring weakness in the paper.

### Questions
Can the authors elaborate why the techniques in Yasuda (2024) do not carry forward to the case of $p > 2$. As far as I understand, the hard instance needs to be updated where $1/\varepsilon$ is replaced by $d/\varepsilon$ and the analysis should work out. I am trying to understand what does not work out with this choice.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
In summary, this paper studies the single-index (label efficient) active regression problem of $\min_{x} \|f(Ax)-b\|_p$, where $b$ is sampled as efficiently as possible. It contributes a $(1+\epsilon)$-approximation, replacing the constant-factor approximations in the literature, with near optimal query complexities.

### Strengths
**Originality:**
It presents novel findings from theoretical and algorithmic perspectives.

**Quality:**
The approach is well motivated, including being well-placed in the literature.

**Clarity:**
The general approach is observable.

**Significance:**
Regarding potential value and impact, the paper has the goal of better addressing a known application/problem, namely single-index active regression, with new theoretical findings of improved query complexities.

### Weaknesses
The submission is not sufficiently clear, with many occasions of ambiguous wording. 
It seems technically correct but hard to gauge at various times. 
The experimental analysis is missing, hence not rigorous or reproducible. 

The paper does not sufficiently support the claims. A key issue is that the analysis accompanying the theoretical results of query complexities includes many approximations, the significance of which are mostly omitted without necessary explanations, resulting in a somewhat lacking scientific rigor.

The paper could bring new knowledge of sufficient value to the community. However, it needs to be revised to better present its soundness.

### Questions
**Questions:**
- Line 148: why is this approximate?
- Line 163: what is the relation between $\overline{x}$ and $x^*$?
- Line 172: what is this approximate inequality defined as?
- Line 197: is this inequality exact?
- Line 215: again, what is this inexact inequality?
- Line 220: how do you conclude $\hat{x} \in T$?
- Line 299: how do you apply (13) repeatedly, AM-GM inequality? Don't the regularization terms also accumulate?
- Line 397: is the second line an inequality?
- Line 423: how is the last inequality obtained?

**Major Suggestions:**
- There are too many approximative parts in the analysis, which hardens the verifiability. The writing should be improved to ease digestion.
- Line 68: improvement from [2023b] to [2024] is unclearly worded.
- Line 191: error terms need to be expanded and explained more clearly.
- Line 278: need more intuition into (10) and (13).
- Line 461: this multi-dimentional extension could be better explained, feels a bit rushed.

**Minor Suggestions:**
- Line 16: V1, V2? better write these (and similar ones) as $\max, \min$ initially to ease reading.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper presents and algorithm to provide an epsilon approximate solution to the active regression problem. They show that their number of query points is minimax optimal.

### Strengths
S1 - The paper is generally well written.

S2 - The technical claims in the paper are generally well explained.

S3 - The idea is interesting.

### Weaknesses
W1 - Literature review is a bit lacking.

W2 - No numerical experiments.

W3 - The presentation can be improved by first introducing the algorithm. In the current version, the algorithm is somewhat built up during the upper bound proofs.

### Questions
Q1 - Following the impossibility result of Gajjar et al., how does your result compare?

Q2 - Why are the probabilities different in Theorem 1,2,3.

Q3 - In addition to the constant probability ones, do you have a high probability results, where the number of queries is a function of that probability?

### Soundness
3

### Presentation
2

### Contribution
2
