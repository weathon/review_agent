# $\alpha$-DPO: Robust Preference Alignment for Diffusion Models via $\alpha$ Divergence

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Diffusion models have demonstrated remarkable success in high-fidelity image generation, yet aligning them with human preferences remains challenging. Direct Preference Optimization (DPO) offers a promising framework, but its effectiveness is critically hindered by noisy data arising from mislabeled preference pairs and individual preference pairs. We theoretically show that existing DPO objectives are equivalent to minimizing the Forward Kullback–Leibler (KL) divergence, whose mass-covering nature makes it intrinsically sensitive to such noise. To address this limitation, we propose $\alpha$-DPO, which reformulates preference alignment through the lens of $\alpha$-divergence. This formulation promotes mode-seeking behavior and bounds the influence of outliers, thereby enhancing robustness. Furthermore, we introduce a dynamic scheduling mechanism that adaptively adjusts $\alpha$ according to the observed preference distribution, providing data-aware noise tolerance during training. Extensive experiments on synthetic and real-world datasets validate that $\alpha$-DPO consistently outperforms existing baselines, achieving superior robustness and preference alignment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposed the modified DPO algorithm with $\alpha$-divergence for diffusion model alignment based on the learning dynamics analysis on the plain DPO's implicit reward. With simple modification, the authors test their method, $\alpha$-DPO, on varying axes of datasets, models, and metrics, consistently outperforming previous methods.

### Strengths
1. The empirical scope of the paper is extensive, with different models (SD 1.5 and SDXL), a lot of baselines (Diffusion-DPO, SPIN, KTO, …), and metrics/datasets.
2. The evaluation results demonstrate that $\alpha$-divergence is superior in nearly every case mentioned in the paper, shown through the direct win rate and average evaluation scores.
3. Although it is not a novel approach, he motivation for using $\alpha$-divergence is straightforward (see weakness for the details).

### Weaknesses
The main weakness of the paper lies in the **novelty and completeness of the divergence analysis on DPO and $\alpha$-divergence as the solution, which serves as the conceptual foundation of the work**. The paper’s discussion of DPO through the lens of divergence minimization and the proposed adoption of $\alpha$-divergence as a solution overlooks several important prior works that have already analyzed these aspects in detail:

1. Theoretical analyses of DPO as divergence minimization (Section 3.1) [2, 4].

2. Adoption of $\alpha$-divergence in DPO-style formulations (Section 3.2) [1–3].

A few previous studies, highly relevant to this paper’s core idea, have already investigated how different divergences articulate the implicit reward under the DPO framework, including the explicit use of $\alpha$-divergence [1–3]. While [1] and [2] directly address $\alpha$-divergence and its policy-based variants, [3] generalizes DPO to a broader family of $f$-divergences that naturally subsume $\alpha$-divergence. None of these works is cited or discussed in the current manuscript.

Given that this paper also emphasizes the “nice properties” derived from adjusting $\alpha$ within the $\alpha$-divergence family, these prior studies are non-concurrent and directly relevant for establishing the paper’s originality and theoretical positioning. Even considering the domain difference (language model and diffusion model), these references are essential for clarifying what is newly contributed here versus what is conceptually inherited.

(Additional potential weaknesses are further elaborated in the Questions section.)

&nbsp;

**References**

[1] Wu et al., 2024, “α-DPO: Adaptive Reward Margin is What Direct Preference Optimization Needs.” (Preprint)

[2] Gupta et al., 2025, “AlphaPO: Reward Shape Matters for LLM Alignment.” (ICML 2025)

[3] Wang et al., 2024, “Beyond Reverse KL: Generalizing Direct Preference Optimization with Diverse Divergence Constraints.” (ICLR 2024)

[4] Shan et al., 2025, “Forward KL Regularized Preference Optimization for Aligning Diffusion Policies.” (AAAI 2025)

### Questions
1. Is it still within the range of “noise robustness” when the downstream evaluation results remain unchanged across 10–40% label flipping?
   * In Tables 2 and 5–8, the evaluation scores exhibit little to no variation as the label-flipping ratio increases. Intuitively, if the flipping ratio rises to 40% (nearly half of the 1M Pick-a-Pic V2 dataset), one would expect a gradual decline in human alignment metrics. For instance, PickScore, which originates directly from the Pick-a-Pic V2 authors, should display a noticeable drop. Moreover, if the proposed $\alpha$-divergence formulation demonstrated the least degradation under increasing noise, that would provide strong empirical evidence that $\alpha$-divergence genuinely mitigates label noise. It is therefore questionable whether maintaining nearly constant evaluation scores across 0–40% flipping can be interpreted as true robustness, since the underlying preference distribution has substantially changed by flipping the labels while the reported outcomes remain unaffected.

2. Typos in Tables 5-8?
   * The explanations and captions in Appendix A.4.2 state that the results correspond to SDXL, whereas the tables themselves label the backbone as SD 1.5. Could these be typographical errors, or do the results indeed correspond to SD 1.5?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the sensitivity of diffusion model to noisy labels in T2I. The paper first shows that the DPO objectives are equivalent to minimizing the Forward KL divergence, thus explaining the sensitivity of DPO to noise. The paper then proposed an alpha-DPO method aimed to alleviate the above problem. The alpha in the proposed method can be adjusted in a data-aware way, giving the training process more control over noise tolerance. The method was tested on both synthetic and real world datasets.

### Strengths
The main strength is to use alpha-DPO to bypass the over-sensitivity in conventional DPO, as revealed by the FKL analysis. Figure 3 illustrates the difference between the FKL and alpha-divergence in covering noisy distribution, which may account for a small amount of data but cause big penalty in regions with small masses. Addressing this point is a strength of the paper. Results are good.

### Weaknesses
A major weakness is that to address the over-sensitivity to noise in conventional DPO, the proposed alpha-divergence incurs hyperparameter such as the mu on line 304 to tune, this increases the complexity of the method. 
The essence of the alpha-divergence is a weighted control of the KL measurement, which is good but novelty is not particularly high. 
It is not clear why Monte Carlo approximation, starting at line 257, would be a good approximation to p^r and p^g(theta)? Can authors give some reasoning behind this statement.
What is the meaning of Figure 4? Can the authors elaborate?
From line 301 to 303, how is f(x^w, x^l, c) maximized?
For the ablation study of Table 4, it seems there was no meaningful difference in the results for different mu, alpha, beta, and the starting point of dynamic alpha scheduling. Does it mean there was little impact on how to choose these parameters. 
Writing can be better as some abbreviations were defined more than once.
While line 250 refers to equation 2, there is no equation 2.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
2

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
This paper introduces α-DPO, a noise-robust variant of Direct Preference Optimization (DPO) for aligning diffusion models with human preferences. The authors identify the sensitivity of standard Diffusion-DPO to label-flipping noise and propose replacing the Forward KL divergence with α-divergence, along with a dynamic α-scheduling mechanism. The method is evaluated on both synthetic and real-world datasets and shows consistent improvements over existing baselines.

### Strengths
1. This work is to systematically address noise robustness in preference alignment for diffusion models, with a solid theoretical grounding in α-divergence.
2. The connection between DPO and FKL divergence is clearly established, and the motivation for using α-divergence is well-justified.
3. Extensive experiments on SD1.5 and SDXL with varying noise levels demonstrate the effectiveness of α-DPO. The method consistently outperforms strong baselines across multiple metrics.

### Weaknesses
1. The paper lacks a thorough exploration of α values beyond the dynamic scheduling setup. A fixed-α baseline with a sweep could better contextualize the contribution of the scheduling mechanism.
2. The human evaluation is based on only 20 participants and 40 prompts, which may not be statistically robust.
3. There is a lack of comparison with many baselines.  like [1]SPO(SD1.5 SDXL), [2]SmPO(SD1.5 SDXL),  [3] MaPO(SD1.5 SDXL). These baselines (checkpoints) are open-sourced.
 [1]Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization. CVPR2025
 [2] Smoothed Preference Optimization via ReNoise Inversion for Aligning Diffusion Models with Varied Human Preferences. ICML 2025
 [3] Margin-aware Preference Optimization for Aligning Diffusion Models without Reference

### Questions
1. Have you experimented with fixed α values across a wider range? 
2. How does dynamic scheduling compare to the best fixed α?

### Soundness
3

### Presentation
3

### Contribution
3
