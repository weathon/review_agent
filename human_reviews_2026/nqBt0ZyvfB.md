# The Sign Estimator: LLM Alignment in the Face of Choice Heterogeneity

- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Traditional LLM alignment methods are vulnerable to heterogeneity in human preferences. Fitting a naïve probabilistic model to pairwise comparison data (say over prompt-completion pairs) yields an inconsistent estimate of the population-average utility---a canonical measure of social welfare. We propose a new method, dubbed the sign estimator, that provides a simple, provably consistent, and efficient estimator by replacing cross-entropy with binary classification loss in the aggregation step. This simple modification recovers consistent ordinal alignment under mild assumptions, without requiring explicit modeling of user heterogeneity, and achieves the first polynomial finite-sample error bounds in this setting. Using standard benchmark experiments and  a new empirical methodology to assess the impact of heterogeneity, we find that the sign estimator substantially reduces preference distortion compared to standard RLHF. Specifically, it cuts disagreement with true population preferences from 12\% to 8\%, and reduces angular estimation error by nearly 35\%. Our empirical set-up leverages digital twins---LLMs calibrated to real-world US panelists---to simulate realistic population-level heterogeneity and obtain a ground truth alignment target for evaluating different estimators.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies reward-model learning for RLHF under heterogeneous human preferences. It shows that the current existing reward model via cross-entropy/MLE is mis-specified. For linear models, the existing RLHF estimator recovers a variance-weighted average of individual utilities, which overweights uncertain users and underweights confident ones rather than the population mean utility direction. The authors propose the Sign estimator, which replaces cross-entropy with a 0–1 sign-agreement objective on pairwise comparisons. Under a mild symmetry assumption on heterogeneity, the Sign estimator recovers the ordinal preferences of the population mean and the direction of the utility vector in the linear case. They further prove a finite-sample convergence rate of $\tilde{\mathcal{O}}(n^{-1/3})$ for the angular error in the linear setting under mild conditions. Empirically, it beats a simple EM-based panel method while keeping the standard pipeline.

### Strengths
1. The mis-specifiation of the existing RLHF estimator is an interesting finding of this paper. Proposition 1 shows that MLE under heterogeneity amplifies the influence of the user's responses with high variances and ignores the users who are decisive. This is a valuable conceptual contribution. 
2. The Sign estimator is simple and consistent, with rigorous theoretical support in section 4. it guarantees ordinal consistency under mild symmetry and achieves a provable $\tilde{\mathcal{O}}(n^{-1/3})$ convergence rate.
3. The empirical results in section 5 are compelling compared to the RLHF estimator. On a controlled heterogeneous-preference benchmark, the Sign estimator consistently outperforms both the standard RLHF baseline and an EM mixture model, demonstrating its practical advantage without complicating the training pipeline.

### Weaknesses
1. My main concern is that the sample complexity analysis of the sign estimator misses the realistic regime of the dimensionality versus sample size. In practice, $d$ could be much larger than $n$, yet here in Theorem 2, it is assumed that the $n\ge n_0$ so that the angular error is small with high probability. It seems the regime of small $n$ is missing. Also, $K$ and $C_{0}$ in Theorem 2 depend on $d$ as well, yet the paper mainly treats $d$ as a constant. Specifically, the result $\tilde{\mathcal{O}}(n^{-1/3})$ ignores $d$ as well. The results can be strengthened if the sample complexity for small $n$ is also presented, which might be more insightful for practical scenarios.
2. In the presentation aspect, there are some terms or phrases lack of explanation. For example, in line 150, I am not sure what "dense" means. 

---
Minors:
1. equation in line 262-263 should be for $u'$ instead of $\hat{u}^{\text{Sign}}$.
2. there is an extra "reduce" in line 384-385.

### Questions
1. It would be better if there is a slightly more detailed review of the existing results in the heterogeneous preference settings for the sample complexity analysis. The rate $\Theta(n^{-O(1/d)})$ is obtained under what conditions? Are they the same as the settings considered in this paper? 
2. When interpret the results of Theorem 2, can we also get a rate treating $d$ as a variable as well? It seems to me that the rate is roughly $O(d^{4/3} n^{-1/3})$ if ignoring $C_0$. 
3. The angular results in Section 5 show that the sign estimator has smaller error. Yet, both estimators have pretty large angular errors (all greater than 40 degree even n is ~$10^5$). Could you explain why is this? Could this result be regarded a rigorous support for the competiveness of the proposed estimator?

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
This paper aims to address the preference heterogeneity problem in LLM alignment. Traditional RLHF method treats pairwise data as a single average utility function and fit models with cross entropy. This paper finds the distorted utility. By replacing cross-entropy with binary classification loss, this paper is able to reach polynomial error bounds. Besides, simulations of LLM alignment using digital twins shows improvement over baselines.

### Strengths
- The analysis of estimators' bias look interesting and original.
- The empirical evaluation result looks good.

### Weaknesses
- The writing needs to be significantly improved. 

    - Improper references. For example,  Harsanyi's theorem is reference in the second footnote but not in the first foot note. Similarly, the BTL model is not reference in the second paragraph of the first page (or between 129, 130).
    - $\mathcal{X}$ is said to represent "all (prompt, completion) pairs" in line 119 whereas line 139 samples are drawn from $\mathcal{X}^2$ which implies the pair can have different prompt which is not the common case. 
    - For example, line 125 mentions that $\{ \xi_{x,\beta} \}$ is drawn i.i.d. from same distribution. Does it mean we can simply drop the subscript?
    - The $\mu$ is used to denote a distribution in line 137 while $\hat{\mu}$ is used as an estimate of utility function in line 161.
    - Line 161 use RLHF as the superscript but it does not have the RL part.
    - Line 272, dangling ).
    - Too many "striking" in the paper.
    - Too many "some" in the paper. Eg. "some set" in line 37, "some distribution" in line 137.
    - ...

- The rates need to compared with related works such as (zhu et al., 2023).

While I find the topic to be interesting, the current paper is clearly not polished enough for readers and certainly not ready for publication.

### Questions
See weakness.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
- The paper shows that standard reward modeling (using cross-entropy loss) introduces bias by "re-weighting" user preferences. It "amplifies the influence of uncertain users" while "diminishing that of confident ones," effectively ignoring users who have the strongest preferences.
- The authors introduce the "sign estimator," a new method that is a simple drop-in replacement for the cross-entropy loss function , instead using a binary classification (0-1) loss to maximize the agreement between the sign of the estimated utility and the observed preference choice.
- The sign estimator is proven to have some nice theoretical properties, e.g. having "ordinally consistent" preferences (i.e., the correct ranking of choices) under a mild assumption of symmetric preference heterogeneity.
- Some experiments are run to empirically verify that it works better than standard reward modeling.

### Strengths
The paper's strengths are on the theoretical end:
- The proposed "sign estimator" is a "drop-in replacement for cross-entropy loss", making it practical to deploy as it "maintain[s] the implementation simplicity of existing LLM alignment pipelines.
- The sign estimator provides a simple, provably consistent, and efficient estimator that "recovers consistent ordinal alignment under mild assumptions".
- The estimator has strong finite-sample guarantees: It achieves the first "polynomial finite-sample error bounds in this setting".

### Weaknesses
The paper has several weaknesses on the empirical end:
- The motivation seems to be heterogeneity of human preferences, yet the authors do not use real human preferences, despite there being real human preference datasets that can be used to align LLMs (e.g., SHP or StackExchange). This is important because real human preferences have noise and bias that may render the proposed estimator no better than the standard one. Simulated preferences are simply not enough.
- The chosen evaluation metrics are bizarre: the estimators are evaluated on how well they recover simulated preferences and on an embedding-based metric. Why not actually put the reward model in a loop with an LLM and post-train it? Then you run many of the standard evals (e.g., Alpacaeval2 for instruction-following). As it stands, there is no real evidence here that the proposed estimator will be practically useful for aligning models with preferences.

My recommendation would be to either make this a pure theory paper and submit it AISTATS (or a similar venue) or evince the practical benefit of the sign estimator by showing that it works on real human preferences on LLMs of non-trivial size.

### Questions
See the weaknesses section above.

### Soundness
2

### Presentation
3

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
This paper addresses the problem of learning an aggregate reward model for LLM alignment from a population of users with heterogeneous preferences. The authors first demonstrate that the standard approach in RLHF, which fits a single probabilistic model to pairwise preference data using cross-entropy loss, yields a biased and inconsistent estimate of the population-average utility. They show that the standard estimator systematically down-weights users with strong, confident preferences. To remedy this, the authors propose a novel method, the sign estimator, which replaces the cross-entropy loss with a binary classification (0-1) loss. This simple modification shifts the learning objective from predicting choice probabilities to merely predicting the correct preference direction (the sign of the utility difference). The contributions come from both theoretical and empirical. Theoretically, the authors prove that under a mild symmetry assumption on the distribution of user preferences, the sign estimator is a provably consistent estimator for the population-average ordinal preferences. They establish the first polynomial-rate finite-sample error bounds for this problem, showing a better convergence rate O(n^{1/3}), a significant improvement over existing rates for general mixture models. Empirically, using a realistic simulation with "digital twins" derived from real user data, the sign estimator is shown to substantially reduce estimation error compared to the standard RLHF baseline, cutting angular error by nearly 35% and disagreement rates with the true population preferences from 12% to 8%. The proposed method also outperforms more complex panel data heuristics that explicitly model heterogeneity, all while maintaining the simplicity of a drop-in replacement within existing alignment pipelines.

### Strengths
- [S1] This paper is well-written.

- [S2] RLHF has recently become an active and important research field. Improving the preference modeling can be beneficial for LLM alignment.

- [S3] The theoretical results and simulated experiments support the proposal well.

### Weaknesses
- [W1] It is not validated if (1) the sign estimator works in the text preference data in LLMs and (2) can be stable in LLMs finetuning, and (3) the learned preference models benefit RLHF training in LLMs. Also, (4) the title may overstate the contribution ("LLM Alignment" should not be there).

- [W2] Cross-entropy training is actually employed to train not only preference classifiers but also scalar reward models (through Bradley-Terry models), and in practice, the reward models play a more important role in RLHF. In sign estimator, how can we recover reward models?

- [W3] The practical implementation for LLMs is unclear from the paper. Actually some theoretical analysis are based on the assumption of linear model (Assumption 2), which is very different from LLMs.

- [W4] Can't we use  "accuracy" of preference labels as one of the evaluation metrics? It is a bit unclear if better "Angle" and "Disentangle rate" lead to better alignment performance in LLMs.

- [W5] could you explain how EM clustering cluster the preference data more? If my understanding is collect, this paper assumes the preference labels are heterogeneous. So I'm not sure, in that case, how the data is clustered.

- [W6] It is not clear if the real preference data used in LLMs satisfies the assumption of  heterogeneity.

### Questions
Please see **Weaknesses** sections above.

### Soundness
3

### Presentation
3

### Contribution
2
