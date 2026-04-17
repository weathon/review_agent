# e-HC: Adaptive Sequential Higher Criticism Test for Sparse Mixtures

- Decision: Reject
- Scores: 4, 2, 2, 8

## Abstract
We propose e-HC, an adaptive sequential test for detecting sparse and weak signals in a stream of p-values. Unlike existing approaches that rely on asymptotic approximations or require knowledge of alternative parameters, e-HC constructs exact test-martingales using moment-generating function compensators, ensuring anytime-valid Type I error control through Ville's inequality. The method adapts to unknown sparsity and signal strength by maintaining exponential weights across multiple detection thresholds, effectively learning the optimal threshold online. We establish non-asymptotic power guarantees for sparse Gaussian mixtures alternative and derive the expected stopping time scaling for weak signal regimes. The same martingale machinery naturally yields anytime-valid confidence sequences for the proportion of significant p-values. Simulations demonstrate that e-HC maintains robust performance under model misspecification, substantially outperforming sequential likelihood ratio tests when the true alternative differs from assumptions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies a sequential version of the well-studied sparse mixture detection problem where the signals are rare and weak. In order to adapt to the unknown sparsity level and unknown signal strength, a sequential Higher Criticism-type statistic is developed. At the core of their proposal is a martingale construction, and the authors show anytime-valid Type I error control through Ville's inequality. Additionally,  their framework yields the straightforward construction of confidence sequences for various quantities of interest.

### Strengths
The paper is clearly written and the authors do a great job providing good intuition. The problem they study is obviously fundamental, and the sequential aspect is a fresh twist on a canonical topic. Methodologically, the martingale construction and the proposal of using exponential weights to adapt to the thresholds is new to the sparse mixture detection literature.

### Weaknesses
The primary focus of much of the sparse mixture detection literature (in the classical, non-sequential setting) is establishing sharp information-theoretic detection boundaries. In the classical literature, the Neyman-Pearson lemma asserts that the likelihood ratio test is optimal and much work goes into finding the sharp detection boundaries (including sharp constants) delineating the regions in which the null and alternative hypotheses separate or merge asymptotically. Donoho and Jin's proposal of Higher Criticism (which they attribute to Tukey) is notable not only because it adapts to the unknown sparsity and signal level, but also because it provably achieves the sharp detection boundary. This optimality guarantee is a very strong reason to advocate for its use. 

The current paper does not derive a detection boundary nor offer any optimality guarantees of any kind for the Higher Criticism-type procedure they propose. Of course, the sequential setting is quite different and thus likely requires careful thinking in formulating an appropriate notion of a detection boundary and optimality. Offering a coherent formulation would itself constitute a contribution in my view, yet it is absent from the current paper. Due to this, I get the feeling that the "Higher Criticism" aspect of the procedure is not actually that important for the major thrusts of the paper. The anytime validity results seems to be the main point, and it appears only the martingale aspect is needed for these.

### Questions
__(1)__ In the usual, non-sequential setting, the fact that Higher Criticism adapts to the sparsity and signal levels to achieve the optimal detection boundary in the Gaussian sparse mixture detection problem is a very compelling reason to use it. Is there a natural formulation of a detection boundary/optimality in the sequential setting, and can the authors show (or at least discuss) the optimality of e-HC? Even a focused discussion on just the Gaussian setting would greatly improve the paper.

__(2)__ In the paper, the thresholds $0 < u_1 < … < u_m < 1$ are just taken as given and the authors do not discuss at all how to select $u_1,…,u_m$ or even how to select $m$. Can the authors provide some guidance? Clearly choices made here will have important consequences for the test’s power. In the extreme setting $m = 1$ seems a bad choice, so it appears there is much room to do some optimization here. Is there some principle to which the statistician should adhere?

__(3)__ To follow up on the previous question, in the classical definition of Higher Criticism in the non-sequential setting, one takes supremum over all possible thresholds - the statistician does not need to select a grid a priori. In fact, this is very important for Higher Criticism to be adapt to the sparsity/signal and achieve optimality. What are the difficulties of incorporating this strategy into the author’s proposal? Can the authors comment on how much they believe they lose by specifying a grid in advance?

__(4)__ In the classical sparse mixture detection literature, the standard parametrization for the sparsity is $\varepsilon = t^{-\beta}$ (in the non-sequential setting) for $\beta \in (0, 1)$. The dense case is $\beta \in (0, 1/2)$, in which case the usual $\chi^2$-statistic is optimal. The interesting regime is $\beta \in (1/2, 1)$, and it is here where Donoho and Jin propose Higher Criticism. Can the authors comment on whether a similar demarcation between dense/sparse regimes can be made (perhaps with some other parametrization)? It seems roughly that the “weak” corresponds to “dense” and “strong” corresponds to “strong”.

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an adaptive sequential test, e-HC, for detecting sparse and weak signals in a stream of independent p-values. The authors construct exact test-martingales using moment-generating function compensators, ensuring anytime-valid Type I error control through Ville's inequality. Additionally, the e-HC adapts to unknown sparsity and signal strength, maintaining robust performance even under model misspecification.

### Strengths
The e-HC algorithm proposed in this paper constructs adaptive martingales based on independent p-value sequences, achieving anytime-valid Type-I error control with theoretical guarantees, even when the signal strength and sparsity are unknown. The authors further analyze its statistical power under the alternative hypothesis modeled by a Gaussian mixture. Empirical results demonstrate that the proposed e-HC method exhibits robustness under model misspecification.

### Weaknesses
1. This paper lacks insight and has an outdated motivation.  The paper’s core idea—replacing asymptotic Higher Criticism by an exact martingale version—is mostly an algebraic adaptation, not a new principle.  The problem of sparse-signal detection via HC is a classic statistical problem from early 2000s (Donoho & Jin, 2004). Recasting it in an “online sequential” setup does not by itself constitute a compelling motivation in 2025, especially for ICLR. 
2.  The main construction (test martingale via exact MGF compensator + exponential weights) follows similarly from known results in the e-process literature (Ramdas et al., 2021; Waudby-Smith & Ramdas, 2024), and the authors didn't mention this or refer to related works. The “adaptive threshold aggregation” is a straightforward application of Hedge, and the resulting theorems (nonasymptotic Type I control, unified lower bound) read more like a re-derivation of standard facts than a new conceptual advance.  
3. There is no attempt to connect the method to practical applications. All experiments are synthetic Gaussian mixtures with simulated p-values.

### Questions
Please refer to the 'Weaknesses' section. Additionally: 
1. I have some reservations regarding the title and the name of e-HC. The derivation of the term e-HC isn't fully explained in the main text. Given the occasional references to 'e-values' (Line 100) and 'e-processes' (Line 440), might this terminology be analogous to other 'e-' prefixed methods such as 'e-BH'? Some clarification would be helpful. 
2. The implementation of e-HC needs the partition $\{u_j\}_{j=1}^m$(as well as the number $m$), the predictable rule for $\lambda_t$, and the weight rate $\gamma$. Could the authors clarify: (i) What principles should guide the selection of these parameters? (ii) How might these parameter choices influence the method's statistical power? 
3. I would appreciate some additional clarification regarding the Remarks on Line 312 to better understand their significance. 
4. A more thorough introduction to the SLRT method, particularly in the experimental section, would help better contextualize the comparative results. 
5. There seems to be a disconnect between Figure 2 (which lacks SLRT results) and the analysis with SLRT mentioned in Line 353. Clarifying this apparent discrepancy would be helpful for readers. 
6. The results in Table 1 suggest that e-HC may incur longer delays compared to SLRT. Could the authors provide some discussions or insights into this phenomenon?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces e-HC, an adaptive, sequential test designed to detect sparse and weak signals within a continuous stream of p-values. The test aims to distinguish the global null hypothesis $H_0$ (where all p-values are uniformly distributed, $p_t \sim \text{Uniform}(0, 1)$) from a sparse mixture alternative $H_1$ (where a small, unknown fraction of p-values $\epsilon_t$ are drawn from a signal distribution $F_1$ that has more mass near zero).

The HC method that this paper uses is tailored for the "rare-and-weak" signal regime. The core contribution is the construction of an exact, non-asymptotic test-martingale, $M_t$, by merging the adaptive thresholding concept of Higher Criticism with modern e-value-family martingale inference.

### Strengths
- It introduces the idea of HC into the modern, rigorous framework of test-martingales and e-processes. The use of exact moment-generating function (MGF) compensators to build an exact (non-asymptotic) sequential test seems to be a novel technical contribution.

- A critical point in HC is the argumentation method over the pre-threshold statistics representing the sparsity of signals. Instead of the max statistic, the authors propose to combine the multiple-threshold statistics using the hedge algorithm. It makes the argumentation data-adaptive.

- Theoretically, the authors provide an exact martingale property under the null (Theorem 1), which is a much stronger guarantee than typical asymptotic results. This is followed by a unified, non-asymptotic power bound under the alternative (Theorem 2) and a formal analysis of the stopping time in the target weak-signal regime (Theorem 3). The appendix details the proofs, showcasing a high level of technical contributions.

### Weaknesses
- The organization of the methodology is poor, which makes it hard to follow while reading. 
    1. **Lack of Clear Motivation for the Core Martingale Construction.** A significant weakness in the paper's clarity lies in the core technical derivation in Section 4.1. The paper introduces the standardized statistic $Z_t(u_j)$. However, it then immediately reformulates this statistic's increment, $\Delta Z_t(u_j)$, into a sum of a "predictable part" $A_{t-1}(u_j)$ and a "stochastic part" $B_t(u_j)$. The final test martingale (the "wealth process") is then built using only the $B_t(u_j)$ term. If $Z_t(u_j)$ is not the object of interest, why not introduce $B$ directly? Why is its increment the necessary starting point, and why is the $A_{t-1}(u_j)$ component subsequently "deleted" from the construction?
The paper would be substantially clearer if it added some sentences to Section 4.1 to motivate this decomposition. It should explicitly state why this step is necessary. Explaining that $A_{t-1}$ is a predictable drift that must be removed to satisfy the martingale condition would improve the paper's accessibility.
    2. **The regret $R_T$ is undefined.** The hedge algorithm that determines the weights of the HC combination appears to be a critical component of the methodology. However, it is only briefly mentioned at the end of Section 4. In Theorem 2 of Section 5, readers can not even find the definition of the regret $R_T$, which plays a critical role in the lower bound.
    3. The definition of the stopping time $\tau$ apears in the very end of Section 5. The methodology part only introduces the construction of the martingale, which is incomplete in methodology. And it also makes the motivation of the construction of the martingale sequences really unclear.

- **The numerical study is questionable**. Only the SLRT method is compared. And the results seem to be located in Table 1 only. However, Table 1 only reports the performance of the e-HC method, why the red-colored numbers represent the SLRT method? The information in Table 1 is totally misleading. And the authors should also describe why the proposed e-HC method is superior in this numerical setting.

### Questions
Besides the questions in the weakness part. There are several questions I raise upon the reading of the paper.

- **The $\lambda$ Parameter**: The core MGF compensator depends on a parameter $\lambda_t$, which is defined as "predictable". However, the theoretical analysis (Theorem 2) and all experiments appear to use a fixed, constant $\lambda$. This is a significant missed opportunity. The framework allows for a data-driven, adaptive $\lambda_t$, but the paper provides no guidance on how to choose it, nor does it explore the performance gains of an optimized $\lambda_t$ versus the fixed $\lambda=0.2$ used in the experiments. The test's power is likely very sensitive to this choice, and a "bad" $\lambda$ could cripple performance.

- **Grid Size and Regret ($m$)**: The Hedge algorithm's regret, $R_T$, scales with $\sqrt{\log m \log T}$. This is the "price" of adapting over $m$ thresholds. The paper does not discuss this trade-off. What is the practical effect of choosing a coarse grid ($m=20$) versus a very fine grid ($m=2000$)? A fine grid is more likely to contain an "optimal" threshold but will pay a higher regret cost, potentially slowing detection. The paper's choice of $m=200$ is arbitrary, and a sensitivity analysis is needed.

- **Hedge Learning Rate ($\gamma$)**: Similarly, the weight rate $\gamma$ (the Hedge algorithm's learning rate) is set to 0.05 without justification. This parameter's tuning is critical to the algorithm's ability to "catch up" to the best threshold, and the paper should provide either a theoretical or empirical basis for its selection.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper proposes an adaptive sequential testing framework for detecting sparse and weak signals from a stream of p-values. It builds on the classical Higher Criticism test but ensures anytime-valid inference by constructing exact test martingales using moment generating function compensators to control Type I error via Ville’s inequality. 
More specifically, the setting is as follows: a stream of independent p-values $p_1,p_2,\dots $ arrives over time, and the goal is to decide whether all of them come from the null distribution $\text{Uniform}(0,1)$ or if a small, unknown fraction comes from an alternative distribution with more mass near zero (indicating a weak signal. The task is to detect the presence of such sparse, weak signals as quickly as possible, while maintaining Type I error control at any time.


The proposed method, e-HC, constructs an adaptive sequential test by combining ideas from higher criticism, online learning, and martingale-based inference. For a grid of thresholds $u_1, \dots, u_m$, it tracks the cumulative proportion of p-values below each threshold, forming standardized statistics similar to Higher Criticism. For each threshold, it builds an exact test martingale by compensating for randomness with its moment-generating function under the null, ensuring that the expected growth of the process is 1 when no signal is present. These per-threshold martingales are then aggregated using exponential weights (via the Hedge algorithm), allowing the method to adapt online to the most informative threshold without knowing the sparsity or strength of the signal in advance. The resulting “wealth process” $M_t$​ increases multiplicatively over time; when it exceeds $1/\alpha$, the null hypothesis is rejected. This guarantees anytime-valid Type I control, while the adaptive weighting provides signal detection across different regimes of sparsity and signal strength.

The experiments show that e-HC maintains exact Type I error control and achieves strong detection power even for weak or misspecified signals. Its martingale process grows rapidly under the alternative but stays stable under the null, confirming theoretical guarantees.

### Strengths
The paper introduces a novel method that unifies higher criticism, martingale-based inference, and online learning into a single adaptive framework, achieving exact anytime-valid error control. Conceptually, the approach is elegant and applicable when both signal strength and sparsity are unknown, while also having rigorous theoretical guarantees.

### Weaknesses
On the negative side, the results depend on strong assumptions, such as independence of p-values and correctly specified null distributions, which may not hold in practical applications.
Also, the analysis and experiments focus mainly on sparse Gaussian mixtures, so its behavior in other models is unclear.

### Questions
-Can the method and theoretical guarantees extend to other models beyond Gaussian mixtures?
-Can you comment on how crucial the independence assumption is on the results? Can the algorithm tolerate some limited dependence?

### Soundness
3

### Presentation
3

### Contribution
3
