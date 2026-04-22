# Strategy-driven Central Limit Theorem for Sequential Test

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
A/B testing is a critical tool for evaluating the effectiveness of strategies, but its conclusions are typically limited to the Average Treatment Effect (ATE). However, a more fundamental question arises when deciding whether to implement personalized interventions: whether Heterogeneous Treatment Effects (HTE) exist. This paper addresses the challenge of testing for the existence of HTE. While current methods based on the t-test are effective, the core pursuit of statistical inference is to enhance test power to more sensitively detect subtle heterogeneous effects. To this end, this paper proposes a novel sequential testing framework based on Strategy Limit Theory, specifically designed to more effectively identify these hard-to-detect, subtle differences. The main contributions are as follows: (i) We integrate HTE existence testing into a strategic decision-making process and construct a new test statistic based on Strategy Limit Theory, weighted by parameter $\lambda$ to control Type I error. By maximizing the divergence between the distributions under the null and alternative hypotheses, we enhance the test’s power. (ii) We extend this approach to online experimental settings and introduce a Bi-Optimal Strategy (BOS). This strategy not only improves statistical power but also significantly enhances the cumulative reward of the experiment. (iii) We develop a complete sequential testing procedure. By combining the alpha-spending function with the Bootstrap method, we determine dynamic stopping boundaries to accommodate the complex joint distribution of our statistic. (iv) We validate the effectiveness and superiority of our proposed method through extensive simulation experiments and empirical analysis on Tenrec, a real-world dataset from Tencent's recommendation system.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a new testing method to address the problem of the global existence of Heterogeneous Treatment Effects in A/B test setting. The method, based on Strategy Limit Theory, commits a higher power performance than widely-used t-tests , as well as controls Type-1 error. This paper also gives the limit distribution of the statistic with optimal strategy under $H_0$ and $H_1$. A sequential testing framework is also developed in this paper, allowing for early stopping while controlling FWER using alpha-spending and bootstrap method.

### Strengths
This paper applies Strategic Limit Theory to the challenging problem of detecting the global existence of Heterogeneous Treatment Effects (HTE), which is a more complicated situation than ATE discussed in Wang et. al (2025). The paper also proposes a novel sequential testing framework grounded in Strategic Limit Theory, which utilizes an alpha-spending function and a Bootstrap method for boundary estimation. Powerful simulation and real data experiment also demonstrate the superior performance of the methods.

Reference:
[1]. Wang, J., Wen, Q., Zhang, Y., Yan, X., & Shi, C. (2025). A Two-armed Bandit Framework for A/B Testing. arXiv preprint arXiv:2507.18118.

### Weaknesses
1. The proposed test statistic $S_{J,\lambda}(\theta^*)$ = $\sup_{x \in \mathcal{X}} S_{J,\lambda}(x, \theta^*)$ is considered over the whole covariate space. However, in the proof of Theorem 3.2, the lemmas and the proof mainly concentrate on establishing the pointwise convergence in distribution for a fixed covariate value $x$ instead of the supremum. There may be a theoretical gap. Without this rigorous justification for the supremum, the paper's novel theoretical contribution beyond Wang et al.(2025) appears to be insufficient.
2. The bootstrap method is a good idea to estimate the stopping bound. However, a significant practical concern is that the iterative resampling and recalculation are at least $B \times K$ steps, resulting in a substantial computation burden. Also, Section 4 lacks the desired theoretical guarantee for the bootstrap method and testing.
3. The test is ordering-sensitive. In other words, the result of the test depends on the ordering of the samples, known as the 'p-value lottery'.
4. The experiment may lack sufficient complexity to demonstrate the superiority of the method proposed in this paper. You can find more suggestions in the 'Questions' part.

### Questions
1. In terms of weaknesses 1, please clarify how the proof addresses the convergence of the supremum. Does it rely on implicit assumptions or existing theorems (e.g., from empirical process theory)? If not, explicit arguments for the supremum statistic's convergence are needed
2. Can the authors provide stronger theoretical justification (or citations) for the Bootstrap's validity in this specific sequential setting, including considering path dependence, nuisance estimates accuracy, and FWER? It would be better if the complexity of bootstrap can be discussed and more empirical quantification about it can be provided.
3. In terms of the 'p-value lottery', I am looking forward to a theoretical solution or mitigation. Otherwise, the authors may provide empirical evidence that the effect is negligible in the scenarios.
4. The experiment matches the theory nicely but can be broadened. For example, one can show the performance under heteroskedastic noise since it is highly relevant to HTE. One can also conduct a sensitivity analysis for the weighting parameter $\lambda$.
5. The formulation of the test is based on an important approximation $Q_0(x, a) \approx \varphi(x)^\top \beta_a^*$. I wonder if other (nonparametric) estimation might affect the theoretical results and practical implementation.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a sequential testing framework for detecting heterogeneous treatment effects using a decision-theoretic perspective grounded in Strategy Limit Theory. The authors reinterpret hypothesis testing as a strategic decision process, analogous to a multi-armed bandit, and design test statistics that amplify distributional divergence between the null and alternative hypotheses. Two strategies are proposed: Optimal Strategy and Bi-Optimal Strategy.

### Strengths
1. Reframing hypothesis testing as a strategic process is elegant for me. The use of Strategy Limit Theory to derive a bi-normal limiting distribution is theoretically interesting.
2. The paper includes formal lemmas, theorems, and asymptotic analysis, giving mathematical depth. At the same time, the paper remains to be readable and relatively easy to follow.
3. The paper is well structured.

### Weaknesses
1. Motivation/Practical Implication: The current problem tells the experimenter “there is some heterogeneity,” but it does not tell them where (which regions of $x$), how large, or in which direction (treatment better or worse). There may be only a small region of $x$ that makes the HTE exists, but that region may be very small or even with zero measurement. The paper can be much useful if the authors can get rid of the sup in Eq. (2).
2. Choice of basis / feature map. It is okay to assume that there exists a basis. However, with the existence of the basis, we implicitly assume some low-dimenstionality or smoothness in the problem. Before the experiment begins, how should we decide the basis without any prior information? If there is only a very small group making the HTE exist, the landscape may not be very smooth. Maybe some practical guideline will be more helpful.
3.  The paper should make the intuition sharper for why the strategy-driven statistic is preferable to a high-quality estimator of 
$\beta_0,\beta_1$ (e.g., orthogonalized / debiased / ML-based CATE testing).
4. The data are inherently adaptive—the chosen arm at stage $j$ depends on past statistics / rewards. Do you need any conditions to guarantee that ehe bootstrap is valid for adaptively collected data? In bandit literature, there have been many works to describe the issue generated by the adaptively collected data.
5. Minor comments related to BOS. The multi-objective perspective is very nice. There are some recent works talking about the trade-off between the two objectives you discussed. You may want to consider the tradeoff as well and cite some of the works. Here are several works that I know:

[1] Simchi-Levi, D., & Wang, C. (2025). Multi-armed Bandit Experimental Design: Online Decision-Making and Adaptive Inference. Management Science, 71(6), 4828-4846.
[2] Wei, W., Ma, X., & Wang, J. (2023). Fair adaptive experiments. Advances in Neural Information Processing Systems, 36, 19157-19169.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper reframes HTE testing as a \emph{strategy-driven sequential decision} problem: at each interim, the analyst chooses between two update “arms” to \emph{shape the test statistic’s limit distribution}, enlarging separation between \(H_0\) and \(H_1\). Using an “Optimal Strategy” (OS) and a \(\lambda\)-weighted statistic, the limit under \(H_1\) becomes (skewed) bi-normal—verified by simulations—thus increasing power; an \(\alpha\)-spending + bootstrap procedure supplies stopping boundaries. A “Bi-Optimal Strategy” (BOS) is proposed for online settings to retain high power while improving \emph{cumulative reward}. Simulations show higher power than a sequential \(t\)-test; a Tenrec case study compares OS vs \(t\)-test (BOS is not evaluated online).


The strategic-limit claims are supported by distributional diagnostics and power curves, and the simulation setup is broadly reasonable. However, the central new angle—\emph{jointly considering power and cumulative reward}—lacks \emph{formal guarantees}: there are no regret bounds or a quantified power–reward trade-off; BOS’s “higher cumulative reward” rests on simulations only, and BOS is not validated in a real online/interactive setting. Type-I control under \emph{adaptive} strategy selection and guidance for \(\lambda\) also need stronger theoretical or sensitivity support.

The work is clear, well structured, and readable; figures effectively convey how OS/BOS increase dispersion relative to normality and thereby power. The OS intuition (“flip the sign to keep \(H_0\) normal and make \(H_1\) more dispersed”) is explicitly stated. It's helpful to clarify what is known in literature, for example, Lemma 3.1.

Casting HTE testing as \emph{strategy-shaped limit theory} is interesting, and BOS’s dual objective (power + reward) is potentially impactful. Yet the paper currently lacks the \emph{core theory} to substantiate that claim (no regret guarantees, no formal Pareto trade-off, even in stylized settings), and BOS is not evaluated in a genuinely online or counterfactually valid replay setup. Several positive findings feel \emph{as expected} given the engineered dispersion.

### Strengths
•	Clear framing and narrative: Strategy-driven shaping of the limit distribution is intuitive and well explained; visuals support the story.
•	Interpretable asymptotics: OS/BOS induce (skewed) bi-normal limits under \(H_1\), giving a concrete rationale for power gains (thicker tails/more dispersion).
•	Practical pipeline: \(\alpha\)-spending + bootstrap yields an implementable sequential procedure.
•	Empirical signals: Simulations and the Tenrec case suggest higher power and earlier stopping than a sequential \(t\)-test (for OS).

### Weaknesses
•	No theory for the reward–power trade-off; no regret bounds.  BOS’s value proposition is dual-objective optimality, but there are no (non-)asymptotic regret bounds nor a formal trade-off characterization—even in linear or other simplified regimes. Add regret upper/lower bounds or a constrained-optimality theorem (maximize reward under type-I control, or vice versa).
•	Evidence chain incomplete for BOS. BOS, while understood as the main contribution in this work, is not evaluated in a real online setting. Consider offline-policy-evaluation (IPS/DR-style replay) or a small-scale online study.
•	Engineered, not principled optimality.  OS/BOS are designed to yield bi-modal/skewed limits, but the paper does not show optimality under a principled criterion (e.g., maximizing \(H_0/H_1\) separation under error-budget constraints). A variational or min–max statement would strengthen the contribution.
•	Calibration under adaptivity.  Type-I control with \(\alpha\)-spending + bootstrap under adaptive arm choice needs proofs or broader sensitivity studies; clarify \(\lambda\) selection and whether data-dependent tuning requires sample-splitting/cross-fitting.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a set of results on high-power adaptive testing for the presence of HTE (heterogeneous treatment effects).  Firstly, they develop a new test statistic that is based on a certain carefully designed two-arm bandit decision process; using insights from prior work (Strategy Limit Theory), it is shown that the proposed strategy has power exceeding that of the corresponding t-test. Secondly, an extension algorithm is presented in which this test power desideratum is combined with the reward maximization desideratum for the experiment (called BOS). Further, the above algorithms are turned into a sequential testing procedure, by means of bootstrapping the stopping boundaries. The HTE testing methods are then validated against the t-test benchmark and show a significant increase in power across many regimes.

### Strengths
All of the results of the paper are novel and nontrivial to the extent of my knowledge, both on the theoretical and on the empirical front. The main result, the Optimal Strategy, leverages a carefully constructed two-armed bandit process that makes a pass and leverages the empirical history so far to split the observations into two groups so as to shape the test statistic distribution for maximum separation under H0 and H1.

The nuances involved in sequentializing this testing strategy are also not fully straightforward, and involve carefully bootstrapped dynamically decided cutoff times. The bi-optimal strategy modification, which mixes in some greed along with the HTE test for the purposes of reward maximization, also requires care.

Finally, the empirical section showcases the strong performance of the proposed testing method. The evaluation is reasonably thoroughly designed for a paper for which the theoretical contribution takes center stage (which is indeed necessary in this case to give the readers a sense of the power gap to the t-test); it gives a good glimpse of both the power curves in the synthetic settings, and of the critical value evolution on the real-world dataset.

### Weaknesses
While the results of the manuscript are interesting and difficult, my issue is with some presentational/writing aspects of this manuscript. In particular, certain aspects may not be fully accessible to a wider readership without further clarifications. 

First of all, the results in this paper are to a large extent based on ideas that the authors refer to as Strategy Limit Theory. I read the reference indicated as the source for these ideas, and I believe a self-contained subsection explaining this theory and the main results and techniques is called for, for the readers’ benefit. This is due to the reasons that, (1) I would not call this reference standard in the field such that it would be known to most of the readership; (2) there appears to be a lot of nuance as to when, and why, the conclusions of that theory apply (i.e. these are not “universal CLTs” to my understanding): while the bimodal convergence and other takeaways of it are scattered throughout the present manuscript, it is not clear how much of the generality of those results in the TAB setting is being used for the present HTE testing purposes.

Secondly, the power results are formulated relative to the t-test, and while additional formulas are provided in the theorem statements for the purposes of being explicit (which I think is good), no clear qualitative or quantitative explanation was provided about when the gap is tight vs. wide. (In particular, the empirical plots suggest the widening and shrinking of the gap happens in a systematic fashion).

Third, I am a bit confused about the naming of certain concepts in this paper. First, both the testing strategies, OS and BOS, are called “optimal”. I’d like to clarify what is meant by that, as from the main results we mainly know that e.g. they perform at least as good as the corresponding t-test. Also, the paper itself is named somewhat confusingly generically: The sequential test for what (should have some mention of the HTE)? And the CLT as the main subject of the title is also somewhat confusing; the tests themselves are the main contribution.

### Questions
Please see above in the weaknesses section; these questions are predominantly of a presentational kind.

### Soundness
2

### Presentation
2

### Contribution
3
