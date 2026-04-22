# Efficient Multi-Step Reinforcement Learning with Expectation-Maximization Bootstrapping

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
Multi-step reinforcement learning (RL) improves agent performance by propagating temporal information across long time lags between actions and consequences through bootstrapping. The key challenge is how to aggregate information from different bootstrapping steps to enable fast learning while maintaining stability. Many existing multi-step RL methods (e.g., Retrace($\lambda$)) primarily focus on the bias–variance tradeoffs but do not explicitly select bootstrapping steps to balance salience and stability (S\&S). We first analyze S\&S in multi-step RL, and introduce a novel corresponding metric to quantify different bootstrapping steps. Viewing bootstrapping steps as the latent variables, our Expectation-Maximization (EM) Bootstrapping (EMB) formulates multi-step RL as an EM procedure, alternating between the E-step: estimating expectations under predefined posterior weights to measure the S\&S of bootstrapping steps, and the M-step: using these estimated expectations to guide the selection of bootstrapping steps. This yields a new return-based Bellman operator EMB($\lambda$). We theoretically establish its convergence and optimality properties. Empirical results on the Atari Learning Environment demonstrate that EMB($\lambda$) significantly outperforms existing multi-step RL methods in both learning efficiency and final performance, matching the performance of Retrace($\lambda$) with approximately $50\\%$ fewer samples on the Atari-10 suite.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Expectation-Maximization Bootstrapping (EMB), a novel framework for multi-step reinforcement learning that interprets the selection of bootstrapping lengths as latent variables. 
By introducing a salience–stability measure b(z_n), the method adaptively weighs different n-step returns to balance bias and variance. 
Empirical evaluations on Atari benchmarks demonstrate promising performance compared to conventional multi-step methods such as Retrace(λ) and Q(λ). 
The paper also provides some theoretical analysis to support the convergence and optimality of the proposed scheme.

### Strengths
1. The paper provides intuitive and novel view of bootstrapping.

2. The proposed metric effectively combines the magnitude and consistency of multi-step targets, providing a practical way to balance return quality and stability.

3. The proposed method achieves consistent improvements over baselines on Atari-10 and Atari-57 benchmarks, showing better sample efficiency and robustness.

4. The inclusion of a convergence proposition and connection to return-based Bellman operators adds theoretical credibility.

### Weaknesses
1. The definition of b is rather heuristic. More theoretical or empirical discussion would help clarify its role and properties.
For instance, analyzing its effect on variance reduction, learning stability, or numerical behavior under different reward scales would strengthen the paper’s justification.

2. The proposed method is EM-inspired rather than a true EM algorithm.
The E-step is approximated by a heuristic salience–stability score instead of a posterior distribution derived from previous parameters, and the M-step simply re-weights the TD loss without maximizing any explicit likelihood.
However, the paper spends a large portion discussing EM, which may distract readers. 
In its current form, the EM framework does not seem to deepen understanding of the actual algorithm or contribute to its improvement.

3. The paper does not specify how b is computed in practice.
For example, what is the exact definition of the normalization terms in Equation (4)?
How is \delta_n calculated (L2 norm per sample or batch average)?
These details are essential for reproducibility.

### Questions
1. From the paper, it seems that Figure 1 uses a direct visualization of Eq. (4) rather than the full EMB method. 
Could the authors clarify whether this figure illustrates the final adaptive weighting mechanism in EMB or just the salience–stability heuristic itself?

2. The ablation study only explores up to 32-step returns. Since the results suggest that longer horizons yield better performance, why not test larger n (e.g., 64 or 128) to observe potential saturation or instability effects?

### Soundness
3

### Presentation
3

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
This paper proposes EMB($\lambda$), a method for adaptively selecting the horizon length for n-step return in multi-step RL. The paper aims to theoretically and empirically justify the proposed method, building on previous algorithms and a novel metric for measuring the saliency and stability of a specific multi-step return target.

### Strengths
The paper proposes a method for determining parameters for multi-step RL, which is found empirically to improve performance over alternative multi-step RL baselines on the Atari Learning Environment. 

The proposed method draws its roots and justification from known and well-used algorithms and methods, thus basing itself well in the existing body of work.

### Weaknesses
The introduction is badly written and hard to follow and understand. It is too long, and brings up a multitude of topics and previous work, with barely any unifying line of thought. Much of it is also repeated in the related work section. 

Also, it describes papers from 2018-19 as “recent”, which in the fast-moving field of RL is hardly a fair description. What work has been done on the subject of multi-step RL in the past few years? Some recent papers have explored adaptively selecting the mutli-step horizon parameter; it seems like they are missing from the related work in this paper. 

The rest of the paper is also unclear and confusing, including many missing definitions for the various bits of notation used. In particular:
- The preliminaries section does not explain what eligibility traces are how they are obtained, or why they help with balancing bias and variance. In particular, $c_j$ is not defined (and is used again later in section 4.3).
- Section 4.1: $N$ is undefined (it is also undefined in the preliminaries, but there it seems like it is the length of some horizon, which does not fully make sense in eq. 4). 
- In section 4.3, $\lambda$ is undefined. Is it the standard $\lambda$ used in $TD(\lambda)$ style algorithms? If it isn’t, the lack of definition makes it unclear whether the assumption at the base of the proof for Proposition 4.1 holds. 

Regarding experimental results: it is unclear (without variance/standard deviation being presented) whether the gains on baselines are statistically significant.

### Questions
1. In line 248, “As shown in Figure 1, we have empirically demonstrated that Equation 4 can efficiently advance existing multi-step RL methods… “: nothing in section 4.1 describes the experiment conducted to produce Figure 1, nor does it describe a full method beyond the definition of a metric. Therefore, how does Equation 4 in itself advance existing multi-step RL methods?

2. Following up on the previous question, in sec. 4.2 the authors describe the $b(Z_n)$ metric as “obtaining optimality” - section 4.1 and Figure 1 do not prove this; at most, they suggest that reward may be correlated with the metric. How do the authors justify this choice of an estimator?

### Soundness
2

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
4

### Summary
The paper proposes a new off-policy online reinforcement learning algorithm called Expectation-Maximization Bootstrapping (EMB). The paper addresses the bias-variance trade-off when learning with multi-step TD targets. EMB learns a metric to evaluate the salience and stability of each bootstrapping step, and uses the measure to weight the TD targets. The derived operator remains a contraction, ensuring the convergence of value estimation under a fixed policy. Empirically, EMB outperforms baseline methods across a wide range of environments.

The key contributions are: (1) introducing a metric for quantifying the balance between salience and stability in equation 4; (2) proving EMB’s convergence and optimality in Proposition 4.1; and (3) demonstrating EMB’s advantage regarding both the efficiency and final policy under a fixed learning budget, compared to baseline methods, across multiple atari games (Figure 2).

### Strengths
1. The method is novel, integrating the expectation–maximization framework into reinforcement learning to dynamically balance salience and stability in multi-step learning. In existing literature, n-step and $\lambda$-return methods rely on a fixed n or $\lambda$ to trade off salience and stability, which limits their adaptability. EMB lifts this limitation by introducing dynamic weighting over multi-step temporal-difference targets, adaptively emphasizing those that are more salient.

2. The method is well supported by theoretical results. Section 4.3 demonstrates that the derived operator is a contraction, which guarantees convergence and optimality of value estimation under a fixed policy.

3. The empirical evaluation is extensive, covering over 60 Atari environments (10 on Atari-10 and the rest on Atari-57), providing convincing evidence of EMB’s learning efficiency.

4. This work performs an ablation study on key parameters ($\lambda$ and $N$), empirically indicating how these choices affect learning efficiency.

### Weaknesses
1. There is remaining ambiguity regarding the optimal choice of N. Figure 3(d) shows that larger N improves performance. Equation (4) indicates that the boundaries of normalization of TD targets and TD residuals depend on N, which suggests that a smaller N may lead to an inaccurate rating due to an overly small range for selecting the minimum and maximum values. This raises the question of whether EMB’s performance gain is mainly due to approaching Monte Carlo-style learning. A more detailed discussion comparing EMB to MC learning would strengthen the paper. Adding an MC-style algorithm into the experimental baseline could also be beneficial.

2. While the evaluation spans many environments, results are averaged over only 3 seeds, which is not sufficient to compute reliable confidence intervals.

3. The learning curves are cut off before showing the sign of convergence, for example, Phoenix in Figure 2, Alien, Asterix, Berzerk, Bowling, and RoadRunner in Figure 4. I would not say it is a significant weakness because baselines showed lower learning efficiency in these environments, however, it may still be worth checking longer runs on these environments to see how EMB converges to gain information on its stability.

### Questions
In the learning curves, does the shaded area represent standard deviation, standard error, or minimum/maximum performance across all runs?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Expectation–Maximization Bootstrapping (EMB), a framework for efficient multi-step off-policy reinforcement learning.
The central idea is to treat the bootstrapping horizon $Z_n$ as a latent variable and optimize it using an Expectation–Maximization (EM) procedure: in the E-step, a salience-and-stability weighting function $b(Z_n)$ is computed to balance bias and variance, and in the M-step, this weighting redefines the multi-step target used for temporal-difference updates.

### Strengths
1. Treating the bootstrapping horizon as a latent variable optimized through EM is conceptually elegant and provides a probabilistic interpretation of multi-step bootstrapping.
The introduction of $b(Z_n)$ as a salience–stability weighting is intuitive and aligns with the bias–variance tradeoff in temporal-difference learning.

2. The empirical study covers both Atari-10 and Atari-57 benchmarks with meaningful metrics (IQM, median, and human-normalized scores).
EMB demonstrates consistent sample-efficiency gains and robust performance on many games.

3. The narrative connecting bias–variance control, multi-step targets, and EM optimization is cohesive and well-motivated.
Figures 2–3 are clean and effectively illustrate the intended performance improvements.

### Weaknesses
1. All formal lemmas (Lemmas 4.1 and 4.2, Proposition 4.1) are restatements of the contraction and optimality theorems from Munos et al. (2016). EMB merely substitutes its weighting $b(Z_n)$ into the Retrace operator and claims inheritance of those properties.
2. The paper never lists the assumptions under which the Munos-style contraction proof holds—finite state/action spaces, bounded rewards, bounded importance ratios, stationary policies, and tabular expectation updates.
Consequently, the claimed convergence of $T_{\text{EMB}}$ applies only to the tabular setting, not to the deep-RL experiments.
The omission risks confusing readers about the theoretical scope.
3. In the Preliminaries, the symbol $\Delta(\cdot)$ appears in the probability expression for latent variables but is never defined.
It likely denotes a Dirac delta distribution or deterministic event; this should be explicitly clarified.
4. From Figure 2, EMB performs worse than Retrace on KungFuMaster and NameThisGame, two dense-reward, short-horizon environments. These underperformances suggest that $b(Z_n)$ may overemphasize long-horizon updates or that the method is sensitive to global hyperparameters. The paper would benefit from discussing this behavior or offer per-game analysis.

### Questions
1. Could the authors explicitly restate the assumptions under which the convergence of $T_{\text{EMB}}$ holds? Is the contraction proof valid only in the tabular setting?

2. Have the authors tested per-game tuning of $\lambda$ or the step limit $N$? Could such tuning mitigate the performance drops on KungFuMaster and NameThisGame?

3. How sensitive is EMB to the computation or normalization of $b(Z_n)$ across games? Does it add variance during learning?

### Soundness
2

### Presentation
3

### Contribution
3
