# Unsupervised combinatorial optimization under complex conditions: Principled objectives and incremental greedy derandomization

- Avg Score: 5.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 6, 3

## Abstract
Combinatorial optimization (CO) has significant theoretical and practical implications.
CO problems are naturally discrete, making typical machine-learning techniques based on differentiable optimization inapplicable.
Karalias & Loukas (2020) adapted the probabilistic method, an important tool in combinatorics, to incorporate CO problems into differentiable optimization. 
Their work ignited the research on unsupervised learning for CO, composed of two main components: probabilistic objectives and derandomization.
Several desirable properties of probabilistic objectives have been proposed, but without principled schemes to satisfy them.
Also, the derandomization process is still underexplored.
Motivated by the limitations, we propose our method UCom2, consisting of two schemes:
(1) a *principled* probabilistic objective construction scheme that provably satisfies the good properties, and
(2) a *fast* and *effective* derandomization scheme with a quality guarantee.
We apply UCom2 to various *complex conditions* (e.g., cardinality constraints, non-binary decisions) and problems involving them, highlighting that UCom2 is *general* and *practical*.
We further show the empirical superiority of UCom2 w.r.t. both optimization quality and speed.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper considered the problem of unsupervised combinatorial optimization under complex conditions. The proposed UCOM2 consists of a principled probabilistic objective construction scheme and a derandomization scheme. The authors provided some theoretical results for the proposed scheme and applied them to various complex conditions, such as cardinality constraints and non-binary decisions. The authors also performed some experiments and showed that UCOM2 is general and practical.

### Strengths
Undoubtedly, this paper generalized some results of the references Karalias \& Loukas (2020) and Wang et al. (2022). To the reviewer's best understanding, the core contributions are two folds: (1) the author claimed the expectation $\tilde{f}:[0,1]^n\rightarrow \mathbb{R}$ of an optimization objective $f:\\{0,1\\}^n\rightarrow \mathbb{R}$, defined by $\mathbb{E}_{X\sim p}f(X)$, is differentiable and entry-wise concave with respect to $p$; (2) the authors conducted incremental greedy derandomization. Overall, this paper is well-written (except mathematical expressions are ugly organized, suggest using the align environment in Latex) and mathematically solid.

### Weaknesses
Unfortunately, some proofs of the core theorems are in questions.

(1) Page 16, Proof of Theorem 1: Please explain the fourth and fifth equality, i.e., first and second equalities as follows.
\begin{align}
&\sum\_X\prod\_{v\in V\_X}p\_v\prod\_{u\in[n]\setminus V\_X}(1-p\_u)g(X)\\\\
=&\sum\_X\prod\_{v\in V\_X,v\neq i}p\_v\prod\_{u\in[n]\setminus V\_X,v\neq i}(1-p\_u)(p\_ig({\rm{der}}(i,1;p))+(1-p\_i)g({\rm{der}}(i,0;p)))\\\\
=&p\_i\tilde{g}({\rm{der}}(i,1;p))+(1-p\_i)\tilde{g}({\rm{der}}(i,0;p))
\end{align}
The reviewer suspected that the first equality is wrong and the authors in fact showed $\tilde{g}$ is linear with respect to $p$.

(2) Page 17, Proof of Lemma 4: Please explain the following equalities,
\begin{align}
\tilde{f}\_{OS}(p^\prime;i,h)&=p^\prime\_{v\_1}d\_1+(1-p^\prime\_{v\_1})p^\prime\_{v\_2}d\_2+\cdots+(\prod\_{j=1}^{n-1}(1-p^\prime\_{v\_j}))p^\prime\_{v\_n}d\_n\\\\
&=\sum\_{j<i}\prod\_{k=1}^{j-1}(1-p\_{v\_k})p\_{v\_j}d\_j+0+\sum\_{j^\prime>i}\prod\_{1\leq k^\prime\leq j^\prime-1,k^\prime\neq i}(1-p\_{v\_k^\prime})p\_{v\_j^\prime}d\_{j^\prime}\\\\
&=\tilde{f}\_{OS}(p;i,h)-q\_jd\_j+\frac{p\_{v\_j}}{1-p\_{v\_j}}\sum\_{j^\prime>j}q\_{j^\prime}d\_{j^\prime}.
\end{align}
The reviewer suspected that the second equality is wrong.

(3) Page 17, Proof of Theorem 3: Please explain
\begin{align}
\sum\_{X\in d^n}(\prod\_{v\in [n]\setminus \\{i\\}}p\_{vX\_v})p\_{iX\_i}g(X)=\sum\_{r\in d}\tilde{f}({\rm{der}}(i,r;p))\leq \tilde{f}(p).
\end{align}

### Questions
Thanks the authors for providing the implemented codes. It would be great if the authors could provide readme.txt or demo codes that the reviewer can reproduce the experiment results. As for the experiments, the reviewer has one question, how to choose the parameter $\beta$? Have the authors done some ablation studies on the choices of $\beta$?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The unsupervised probabilistic method for combinatorial optimization is a hot topic in recent machine learning community. The paper proposes UCom2 as a unified framework with principled probabilistic objective construction scheme that provably satisfies the good properties, and a fast and effective de-randomization scheme with a quality guarantee. Under this framework, the paper conduct intensive experiments on combinatorial optimizations with hard constraints and obtains the state-of-the-art performance among unsupervised probabilistic methods.

### Strengths
* The paper formally gives principled criteria for objective functions and de-randomization, which completes the framework of unsupervised probabilistic methods for combinatorial optimization. Also, the paper proves the framework is simple but guaranteed to be effective.
* Then paper conduct intensive experiments to empirically demonstrate UCom2 is general and practical. The experiment settings are detailedly provided and the comparison with baselines are properly discussed.

### Weaknesses
* Though provides a unified view, the proposed framework is basically doing the same thing as previous methods. Leading the novelty is limited. 
* The ad-hoc incremental difference is designed for each problem. Also, only evaluating the difference is a commonly used method to reduce computation. To me, it is more like an engineering effort rather than a machine learning method.
* Minor: the paper looks a bit crowded, making reading a bit tired.

### Questions
* I am wondering whether the author tried to compare the performance of UCom2 with supervised learning. I am interested in their gap.
* Since Ucom2 designed a principle de-randomization, I am curious whether it is possible to make the objective function being de-randomization arrogant?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors are motivated by the research in unsupervised combinatorial optimization and propose to extend prior work in this topic to more complex optimization problems. In particular, they seek to develop a principled approach to constructing probabilistic objectives and effective derandomization scheme with guarantee on solution quality.

### Strengths
The ideas around unsupervised combinatorial optimization are interesting and authors place their contribution well by discussing prior results and adequately motivating their work. The theoretical quality guarantee from derandomization scheme seems to be a reasonable contribution.

### Weaknesses
Since the paper builds on the specific work Karalias & Loukas (2020) and Wang et al. (2022), I was not able to get a solid understanding of conceptual contributions of the paper. Theoretical results - as claimed by the authors - are follow fairly standard arguments. They are based on standard (basic) optimization analysis. While the tightness of these results to the original approach proposed by  Karalias & Loukas might be worthwhile, we do not get a sufficient understanding of the generality of these results. I would have appreciated seeing a more pointed discussion on why rounding/derandomization schemes in classical combinatorial optimization are not helpful here? Without such a consideration, the contribution might myopically advance the idea of pushing differentiable optimization into combinatorial optimization, but might miss on building on rich set of existing results in combinatorial optimization.

### Questions
- how does the result on goodness of greedy derandomization compare with similar ideas in combinatorial optimization? 
- what do we mean by problems with "complex conditions"? 
- what are the features of the problems studied in the experiments section that enable a good solution guarantee after derandomization (vs not)?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
