# Geometry-Aware Approaches for Balancing Performance and Theoretical Guarantees in Linear Bandits

- Decision: Accept (Poster)
- Scores: 8, 8, 8, 6, 5

## Abstract
This paper is motivated by recent research in the $d$-dimensional stochastic linear bandit literature, which has revealed an unsettling discrepancy: algorithms like Thompson sampling and Greedy demonstrate promising empirical performance, yet this contrasts with their pessimistic theoretical regret bounds. The challenge arises from the fact that while these algorithms may perform poorly in certain problem instances, they generally excel in typical instances. To address this, we propose a new data-driven technique that tracks the geometric properties of the uncertainty ellipsoid around the main problem parameter. This methodology enables us to formulate a data-driven frequentist regret bound, which incorporates the geometric information, for a broad class of base algorithms, including Greedy, OFUL, and Thompson sampling. This result allows us to identify and ``course-correct" problem instances in which the base algorithms perform poorly. The course-corrected algorithms achieve the minimax optimal regret of order $\tilde{\mathcal{O}}(d\sqrt{T})$ for a $T$-period decision-making scenario, effectively maintaining the desirable attributes of the base algorithms, including their empirical efficacy. We present simulation results to validate our findings using synthetic and real data.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper studies algorithms for stochastic linear bandits. Specifically, it has been established that frequentist algorithms may not match the optimal minimax rate in O(d\sqrt{T}) but in practice, “hard” problems are rare and algorithms behave better than what theory predicts. This paper proposes a data-dependent switching strategy that adaptively switches between different “types” of linear bandit algorithms depending on the problem at stake. The switch is made thanks to a threshold that must be tuned. 

Beyond this concrete algorithmic contribution, the paper revisit the analysis of linear bandit algorithms and sheds a new light on the problem with a very clear discussion that would be highly valuable to this literature. I recommend accepting this submission to ICLR.

UPDATE: I have changed my score to 8 to also take into account the few remarks raised by the other reviewers and a couple of minor second thoughts I had. However, I still believe this is a clear paper with an original proposal, which I recommend to accept.

### Strengths
* Many significant contributions. Concretely, the paper 1/proposes a new way to analyse bandit algorithms that admits all SOTA methods as special cases, 2/ proposes a novel algorithm with an “oracle” regret bound (Th 2) that depends on a problem-dependent quantity, and 3/ shows how to estimate and exploit this quantity for a fully adaptive algorithm, 4/ tests this algorithm on simulated but important examples and on real datasets. 
* Well written. The paper is very clear and I only have minor comment further below. 
* Impactful. I believe such analysis may have an impact on RL algorithms in Linear MDPs.

### Weaknesses
I am honestly not sure what to point out. I was a bit confused by the discussion of \cX \equiv \cS^{d-1} because the notation of the sphere is not defined and I had missed the discussion  on considering only algorithms for which \tau_t=0. But I am not sure what to recommend to make it clearer.

### Questions
I don't have specific questions that would be useful for me to assess this work further.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper considers the classic stochastic linear bandit problem. Motivated by that the LinTS-freq and Greedy algorithm show great empirical performances but does have minimax optimal regret guarantee, this paper wants to bridge the gap between theoretical guarantees and empirical performance. The paper first proposes a general POFUL algorithm framework which can include LinTS-freq, OFUL, Greedy as special cases and provide a regret bound. Inspired by the analysis, the paper then proposes a Course-Correction version for LinTS-freq and Greedy (TS-MR and Greedy-MR). By adaptively balance the action selection of the original algorithm and OFUL, the paper shows that the proposed algorithm achieves the minimax optimal regret while preserving the original empirical advantage. The better performances of the proposed algorithms are verified through both synthetic and real-world data sets.

### Strengths
1.	The paper proposes a general algorithmic framework POFUL that includes classic OFUL, TS, and Greedy as special cases, and provides frequentist regret analysis for this general framework.
2.	The paper introduces the concept of regret proxy, which connects the variance of the optimal arm/selected arms with the regret. Inspired by this, the paper proposes Course-Correction for standard LinTS and Greedy algorithm. The variants combine the empirical advantage of the original algorithms and the theoretical advantage of achieving minimax optimality of OFUL algorithm.  
3.	Extensive experiments on both synthetic and real-world datasets are conducted to compare the empirical performances of the classic OFUL/TS/Greedy and the proposed TS-MS and Greedy-MS, which demonstrates consistent advantage over baselines.

### Weaknesses
1.	In Line 102, the paper says that “derive a general, instance-dependent frequentist regret bound”. Usually ‘instance-dependent’ in bandit literature means that the regret depends on the sub-optimality gap. Here there may be some confusions. 
2.	Generally speaking, the proposed TS-MR and Greedy-MR can be regarded as a combination of traditional TS/Greedy with OFUL. The algorithm adaptively determines which action to follow based on the value of $\hat{\mu}$. Such combination preserves the minimax optimality. However, since the algorithm still needs to compute the OFUL solution, the computational advantage may be limited. Can authors discuss in the experiments what percentage of rounds need to be replaced with OFUL?
3.	Some writing suggestions:
a)	In Section 4.1, the definition of $\tilde{x}_t$ and $x_t^*$ should be moved before the Proposition 4. 
b)	In Appendix C, the $\leq$ in Line 872 should be $\geq$, the $\in$ in Line 882 should be $\notin$.

### Questions
Please see the last part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper addresses a significant challenge in stochastic linear bandits, where empirically effective algorithms like Thompson Sampling and Greedy algorithms demonstrate unsatisfying theoretical regret bounds. The author closes this gap by proposing a data-driven approach where problematic instances of LinTS and Greedy are identified, and "course-correction" is applied to those instances. The regret incurred by the new algorithm matches the minimax optimal regret of order $O(d\sqrt{T})$.

### Strengths
1. The paper is well written. It follows a logical structure, progressing smoothly from one section to another. 
2. Diagrams, such as those illustrating the POFUL algorithm and confidence ellipsoid geometry, help clarify complex ideas, making abstract concepts more tangible. Examples that walk through different algorithms further enhance understanding.
3. The question considered is important. The proposed framework, POFUL, for linear bandit algorithms is novel, supporting OFUL, LinTS, TS-Freq, and Greedy as special cases.
4. The proposed course-correction idea is novel. It adapts LinTS and Greedy algorithms, making them maintain the desirable empirical performance while achieving minimax-optimal frequentist regret.

### Weaknesses
1. The method involves setting hyper-parameter $\mu$ (also inflation and optimism parameters) for the course-corrected algorithm. How it is chosen in the experiments is not discussed. More insights into how these parameters influence the outcome or suggestions for selecting optimal values would strengthen the practical utility of the method.
2. Empirically, TS-MR appears to perform similar to the better one of LinTS and OFUL in most cases when the balancing parameter is carefully chosen. While this result aligns with intuition, the paper lacks a discussion of the specific situations in which TS-MR could significantly outperform both LinTS and OFUL.
3. The proposed method involves calculating $\hat{\mu}_t$ each round, which can be can be computationally demanding. This could limit the scalability of the proposed method, especially in applications where the dimension $d$ or the number of actions $T$ is large. It may benefit from a discussion of strategies to mitigate computational costs in such scenarios.

### Questions
1. Line 310: "On the right-hand side of equation 4...one cannot apply Proposition 2..." Could you explain why? Proposition 2 applies to an "arbitrary" sequence, so it seems general enough to be applied here. 
2. Line 266: why $\tilde{x}_t=x^*_t$ leads to $\hat{\alpha}_t$ no less than 1? If $\hat{\alpha}_t$ is no less than 1, shouldn't the inequality in line 367 go the other way?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the $d$-dimensional stochastic linear bandit setting. It builds on several previous works, mostly on
- the analysis from Abbasi-Yadkori et al. (2011) where they proposed an algorithm "Optimism in the Face of Uncertainty  Linear Bandit Algorithm" (OFUL) for which they derived a frequentist regret bound in $\tilde{O}(d\sqrt{T})$;
- the work from Agrawal \& Goyal (2013) and Abeille et al. (2017) that studied the frequentist variant of LinTS, referred to as TS-Freq, and showed that its frequenist regret is upper bounded in $\tilde{O}(d^{3/2}\sqrt{T})$. The extra factor $d^{1/2}$ has latter confirmed to be necessary by Hamidi \& Bayati (2020);
- the works from Russo \& Van Roy (2014) and Dong \& Van Roy (2018), who studied the Bayesian regret of the Thompson Sampling algorithm, referred to as LinTS and proved that it is upper bounded in $\tilde{O}(d\sqrt{T})$.

Based on the last points regarding LinTS, the authors conclude that the algorithm performs well in most problem instances but suffers in unfavorable, unlikely settings. Their idea is then to propose a modified "course-corrected" version of LinTS and Greedy algorithms, that can switch to selecting OFUL actions under a criteria based on a geometric analysis the $d$-dimensional confidence ellipsoid. The resulting algorithms are called respectively "Linear
Thompson Sampling with Maximum Regret (Proxy)" (TS-MR) and Greedy-MR. Both algorithms retain the theoretical guarantees of OFUL algorithm while presenting some computational and performance advantages from the LinTS and Greedy method.

### Strengths
The main strength of the paper is to propose a modification of the LinTS and Greedy algorithms, two well studied algorithms for solving $d$-dimensional stochastic linear bandit problems. The modification is based on the work from Abbasi-Yadkori et al. (2011) and their algorithm "Optimism in the Face of Uncertainty  Linear Bandit Algorithm" (OFUL) which enjoys frequentist regret bound in $\tilde{O}(d\sqrt{T})$. The authors resulting algorithms "Linear
Thompson Sampling with Maximum Regret (Proxy)" (TS-MR) and Greedy-MR, can both switch from their respective methods to OFUL algorithm under some criteria. The authors derive frequentist regret bound in $\tilde{O}(d\sqrt{T})$ for both algorithms and illustrate their performance under some numerical experiments.

### Weaknesses
Although the main ideas of the paper are interesting, there are a few points that could be improved, such as listed below.

- The main weakness concerns the reproducibility of the experiments. The authors provide few explanations regarding the experiment setup in the paper and no information regarding the algorithm's parameters and implementation. They did not provide their code, and the details they give in the Appendix are not sufficient to reproduce the same experiments. Therefore, it was impossible to assess the soundness of the simulation results. 
- A second weakness concerns the experimental comparison. The authors should have included experiments assessing the influence of the algorithm parameter $\mu$ on the performance. They authors mention that they "aim to compare TS-MR, Greedy-MR, and key baseline algorithms, via simulation" but the key baseline algorithm they compared their method with are the very same algorithm their method is based upon. It would have been for the authors to include a comparison with a different algorithm such as UCB. 
- Another point of improvement concerns the figures in the paper. First, the font size in all figures in the paper is too small to read on a printed version of the paper. Second, some figures lack explanations such as Figure 3 that seems to show confidence interval for the regret of Greedy algorithm and LinTS algorithm without any explanation.

### Questions
Here is a list of suggestions for the authors. 
- A first suggestion is to include a proof of Corollary 1, if possible, in the main part of the paper. 
- A second suggestion is to include the pseudocode of the proposed methods TS-MR and Greedy-MR in the paper's main text.
- Another suggestion concerns the experiments. The authors should provide enough material and explanations so that it is possible to replicate and verify the results of the simulations. We suggest the authors to share the code they used to generate the simulations, as well as to include more details about the experiments in the Appendix: 
  - the pseudocode and parameters used for the TS-Freq algorithm
  - the pseudocode and parameters used for the OFUL algorithm
  - the pseudocode and parameters used for the Greedy algorithm
  - the values of the parameters ($\delta$, $\iota_t$, $\lambda$) for the TS-MR algorithm as well as explanations regarding the distribution $\mathcal{D}^{SA}(\delta')$.
    The authors should also include references and explanations regarding "the standard approach in the literature that converts classification tasks to contextual bandit problems". Note that there is a typo on line 500 as there is no Example 2 in Section 6.
    Ideally, the authors should also perform experiences to illustrate the tuning of the parameter $\mu$ and its influence on the performance of the algorithm and include comparisons to algorithms that are not part of the proposed method, such as UCB. 
- The authors should also improve the readability of the Figures in the main body of the paper as the font size is too small. They should also provide explanations in Figures 3 (b) and (C) regarding what seems to be confidence intervals.  
- There are some notations throughout the paper that are either not defined or used before they were defined: 
  - the notations $\lambda_1,\ldots,\lambda_d$ and $\lambda_2^{-\frac{1}{2}}(V_t)$ used in Theorem 2 but are not defined. There is also some confusion with both the notation $\lambda$ being used as a regularization parameter. We note that both the notation $\lambda_d$ and $\lambda_d(V_t)$ are present in the paper. It is not clear if it refers to the same quantity.
  - there seems to be a typo on line 423 in Theorem 2, as the indexes $k$ and $k+1$ are used but do not refer to anything. The authors should bring clarification.
  - the function $x^\star_t(\theta)$ is used on line 351 but only defined on line 383. 
  - the notation $\equiv$ is used but not defined. 
  - the notation $\mathcal{S}_{d-1}$ is used but not defined. It is unclear whether it refers to the $d$-dimensional unit sphere or a sphere of radius $S$ centered on the origin.
  - the notation $\beta^{RLS}_t(\delta)$ hides the dependency on the parameter $\lambda$.
  - it is not explained how $\hat{\mu}_t$ relates to $\hat{\alpha}_t$ on line 364.
- Other suggestions include: 
   - avoiding abbreviations such as "we'll" on line 368 and prefer the full expression
   - citing reference on line 370 to explain the "large regret proxies and computational inefficiencies seen in OFUL when determining the action".
   - it is unclear from Example 2 how to set the variables $\iota^{TS}$ such that POFUL matches LinTS. 
   - Assumption 3 could be renamed to "Subgaussian reward noise."
   - rephrase line 390, "As is illustrated on Figure 2." as this phrase lacks a subject and is therefore not a proper sentence. 
   - correct the typo on line 1048: "We ser $\mu$".

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper studied the classical linear bandit problem. It proposed the POFUL algorithm which can be specified to be LinTS, LinUCB and greedy algorithms with different parameters respectively. It analyzed its regret and evaluated its performance with numerical experiments. This paper also considered the continuous action space and the instances with proxy.


======

After rebuttal: I prefer to keep the score.

### Strengths
1. This paper provided a more general algorithm for regret minimization in linear bandits and another aspect to understand the efficient algorithms.
1. The numerical experiments are well designed as different types of bandit examples are considered.

### Weaknesses
1. The author(s) may consider to place the comparison among Theorem 1 and results from existing papers in the main paper. The comparison is not clear at first glance. 
1. Sections 6 and 7 indicate that a preset threshold $\mu$ can clearly reduce the regret and Remark 3 discussed the choice of $\mu$. However, I wonder under which types of real-life scenarios we can know $\mu$ and what is the choice of $\mu$ in simulations.

### Questions
Please refer to the *Questions* section.

### Soundness
3

### Presentation
3

### Contribution
3
