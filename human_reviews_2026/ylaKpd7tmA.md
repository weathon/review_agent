# Neural Optimal Transport Meets Multivariate Conformal Prediction

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 6, 4, 6

## Abstract
We propose a framework for conditional vector quantile regression (CVQR) that combines neural optimal transport with amortized optimization, and apply it to multivariate conformal prediction. Classical quantile regression does not extend naturally to multivariate responses, while existing approaches often ignore the geometry of joint distributions. Our method parameterizes the conditional vector quantile function as the gradient of a convex potential implemented by an input-convex neural network, ensuring monotonicity and uniform ranks. To reduce the cost of solving high-dimensional variational problems, we introduce amortized optimization of the dual potentials, yielding efficient training and faster inference.  

  We then exploit the induced multivariate ranks for conformal prediction, constructing distribution-free predictive regions with finite-sample validity. Unlike coordinatewise methods, our approach adapts to the geometry of the conditional distribution, producing tighter and more informative regions. Experiments on benchmark datasets show improved coverage–efficiency trade-offs compared to baselines, highlighting the benefits of integrating neural optimal transport with conformal prediction.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposed a parametric approach to estimate conditional vector quantiles, where the push-forward is parameterized through the gradient of a (partially) convex neural network. An amortized strategy is adopted to reduce computation (by following the unconditional case) and an application to split conformal prediction is discussed. Experiments on small-scale synthetic dataset and regression benchmarks confirm the effectiveness of the proposed methods.

### Strengths
- a neural approach to estimate vector quantiles 

- a more versatile approach for conformal prediction

### Weaknesses
- limited novelty: the majority of the contributions are a straightforward extension of prior works, with no significantly new ingredients

- possible gaps in Theorem 3: the proof on Line 1244-1246 needs more explanation. The proof seems to rely on the assumption that HPD superlevel sets are "optimal"? In Remark 3, how is the displayed equation in Theorem 3 satisfied? Here the left-hand side does not depend on y while the right-hand side does. Is this assumption too strong? 

- writing can be improved. A lot of details are deferred to the appendix, including the description of the proposed methods (Algorithms 1, 2 and 3). At times, it is hard to appreciate what are the significant modifications on the authors' side.

### Questions
My main concerns are on the significance and correctness: 

- most results appear to be rather straightforward adaptions of prior works (Sections 4 and 5). This is not to say there isn't anything new, but the writing makes it feel like these are rather incremental changes. 

- I was not able to verify the proof of Theorem 3 and I think its assumptions may be too strong. In particular, for the elliptical example the authors need to verify the displayed equation in Theorem 3. 

I was intrigued by the authors claim that "Neural OT as we propose here have never been scaled" (Line 321) and "never from a unique joint sampling using the framework of Carlier et al. as proposed in our work" (Line 340). A quick google search revealed the following work "Conditional Generative Quantile Networks via Optimal Transport" (https://openreview.net/forum?id=BBxeo2Vuvbq), which seemed to be based on Carlier et al. and already proposed a neural approach for conditional vector quantile estimation. This latter work did not discuss conformal prediction but perhaps is still very relevant. 

Other comments: 

Line 148: "In the multivariate setting, monotonicity requires the map to be the gradient of a convex function." How is monotonicity defined in the multivariate setting? Why is it needed and why does it require the map to be the gradient of a convex function? 

Figure 3, right plot: the y-axis (log V)/d can be as low as -0.7 (even -5 for the second plot). For moderately large d this means the volume of the conformal set is exceedingly small. Is this expected? 

Line 467: the resulting response dimensions are 16, 4, 2, and 2. The potential based OT approach is known to be difficult to scale (due to the evaluation of the conjugate potential). The authors' experiments seem to confirm this observation. How high the dimension can be for the proposed approach to remain effective?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper extends univariate conformal quantile regression (CQR) to the multivariate setting through the use of transport-based quantiles. To achieve this, the authors propose implementing optimal transport (OT) maps within vector quantile regression via Neural OT. The approach relies on input convex neural networks (ICNNs) to enforce the monotonicity of conditional quantiles or ranks. In addition, amortized optimization is employed to fasten the repeated computations inherent to conditional optimal transport.

### Strengths
This paper effectively integrates recent advances in computational optimal transport into the study of transport-based quantiles. It leverages Partially ICNNs (PICNNs) from Neural OT, amortized optimization for efficient computation of dual conjugates, and entropic regularization. These methodological choices are particularly relevant for enabling the use of transport-based quantiles in machine learning applications, where computational efficiency is crucial. The application of Neural OT to conformal quantile regression is both well-motivated and timely. I also appreciate the level of detail provided in the appendices regarding the different Neural OT implementations and the comprehensive presentation of results across all evaluation metrics.

### Weaknesses
My main concern is that the introduction may give a somewhat misleading impression of the existing literature. The incorporation of Neural OT into transport-based quantiles is not entirely new (see, for example, Continuous VQR, Vedula et al., 2023). Since this is claimed to be the first main contribution, a more explicit discussion of these related works would be valuable, in particular, to clarify the computational differences between the approach presented in this paper and prior methods. Although these papers are properly cited in the Appendix, the comparison should appear more prominently in the main text to help readers understand the methodological positioning.

The connection between transport-based quantiles and conformal prediction is also not entirely new, and the framing of the second contribution could be clarified. As currently written, it appears quite close to the contributions of recent works such as Thurin et al. (2025) and Klein et al. (2025). My understanding is that the approach presented in this paper generalizes conformal quantile regression by learning the conditional distribution Y∣X through transport-based quantiles, in the spirit of Feldman et al. (2023). In contrast, the perspective adopted in Thurin et al. (2025) and Klein et al. (2025) is to combine multivariate scores with multivariate quantiles, which represents a different line of research. This distinction is not clearly reflected in the introduction, although it does emerge in the main text and seems to motivate your re-ranking variants.

In addition, the experimental section would benefit from including numerical comparisons with related approaches, such as Feldman et al. (2023), to better position the method empirically.

### Questions
**PART I - Framing contributions in the literature** 

I.1 *l.49 “While well studied in the univariate case, multivariate extensions are less developed and often reduce to coordinate methods that ignore the geometry of joint distributions (Dheur et al.)”*: The paper by Dheur et al. actually discusses several possible ways to extend conformal prediction (CP) to multivariate regression. It therefore does not seem to be the most appropriate reference to support the claim that existing methods “reduce to coordinate approaches.” You might instead cite works that genuinely adopt coordinate-wise strategies.
In addition, the paragraph currently gives the impression that transport-based quantiles are the only viable direction for multivariate CP, which is not the case, as you yourself note later (lines 343–349) and in the extended related-work section. The introduction should be revised to explicitly mention the existence of alternative multivariate CP frameworks.

I.2 *l.815:”Despite these advances, none of the above techniques exploits the full geometric structure of multivariate quantiles”*: It would help to clarify what is meant by “the full geometric structure.” One could argue that transport-based quantiles belong to the L-CP family of conformity scores introduced by Dheur et al. (2025). Indeed, your method also maps data to a latent space (albeit of the same dimension), similarly to Fang et al. (2025) or Feldman et al. (2023). These approaches appear complementary rather than mutually exclusive, so a more nuanced comparison would strengthen the framing.
For example, the monotonicity of the transport map in your model preserves the ordering of multivariate samples, this property distinguishes your method from the above works. It would be interesting to explain in which settings this specific feature provides a tangible advantage.

I.3 *On Neural OT for multivariate quantiles*: The work of Vedula et al. (2023) on Continuous VQR appears closely related, as it also implements PICNNs for Neural OT. However, the similarities and distinctions between your approach and theirs are not sufficiently discussed. We recommend adding a short paragraph explicitly positioning your method with respect to prior studies that apply Neural OT to transport-based quantiles, possibly including targeted numerical comparisons if computational differences are significant.
You may also wish to cite recent references such as [3], which learns multivariate quantiles with PICNNs for time series, and [2], which uses amortized optimization for repeated OT computations. Clarifying these links would considerably improve how your contribution is situated within the broader literature.

**Part II - Numerical comparisons**

II1. Providing a comparison between different implementation variants is a valuable practice that can guide future practitioners. However, I have several questions regarding the results presented in Section 7.1.
In the tables, it is unclear what formatting (bold or underlined) denotes — could you clarify this in the captions or text? More importantly, what are the main takeaways from these experiments? You compare several Neural OT–based implementations of conformal quantile regression, but it is not evident which one should be preferred in practice.
In Table 1, for instance, are EC-NQR and CPF the best-performing methods overall? Is there a meaningful difference between learning the quantile map and the rank map, or do these approaches yield comparable results? Also, why is the AC-NQR variant used in the experiments for conformal prediction? A short Discussion paragraph after the results section would greatly help interpret these findings and highlight the practical implications of the comparisons.

II2. Furthermore, some of the experimental comparisons seem to involve methods that are not directly comparable. For example, in Figure 4, you compare your CQR variants based on signed residuals (PBS and RPBS) with your main proposals (PB and RPB). Yet these two groups of methods do not rely on the same base model: PB and RPB employ a neural network, whereas other works use a random forest. It is therefore not surprising that PB and RPB yield lower volumes, as neural networks generally achieve higher predictive accuracy. A fairer comparison would restrict attention to PBS and RPBS versus ELL and OTCP(+), which share a similar setup.

II3. In line with this point, it would also be natural to compare your method with Feldman et al. (2023), since their approach uses a neural network–based quantile regression as the base model — arguably a closer alternative to your main proposals (PB and RPB). Finally, since your re-ranking approach falls within the L-CP framework introduced by Dheur et al. (2025), a comparison with some of the methods considered in that reference would also be informative.

**PART III - Clarification**

III1. The denomination “Conditional vector quantile regression” used in the abstract seems inappropriate. Conditional quantiles refer to the quantiles of a distribution conditional on covariates, while quantile regression designates the learning of such conditional quantiles from data. This latter notion corresponds to the focus of your work, together with a new algorithmic framework to achieve it. Therefore, the term vector quantile regression would be a more accurate and conventional choice.

III2. Theorem 1 (iii): you write a probability at the end of your sentence that I do not understand:  what is the meaning of $\leq$ between two vectors? What is F_U(u), when F_U is defined as a probability measure? I assume that the point (iii) should be a sentence similar to the point (i): there exists a mapping that pushes the conditional distribution Y/X=x to the reference distribution F_U. Also, Assumption (2) is not customary: [1] defines the rank function in a semi-discrete setting, a fact that might appear in a remark. 

[1] Ghosal, Promit, and Bodhisattva Sen. "Multivariate ranks and quantiles using optimal transport: Consistency, rates and nonparametric testing." The Annals of Statistics 50.2 (2022): 1012-1037

[2] Amos, Brandon, et al. "Meta optimal transport." Proceedings of the 40th International Conference on Machine Learning. 2023

[3] Kan, Kelvin, et al. "Multivariate quantile function forecaster." International Conference on Artificial Intelligence and Statistics. PMLR, 2022.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper combines neural optimal transport with conditional vector quantile regression to improve multivariate prediction and uncertainty estimation. The authors use convex neural networks to model transport maps, suggest employing amortized optimization to improve efficiency, and apply entropic regularization for stability. They then use the resulting quantile maps to build conformal prediction sets that represent uncertainty in multiple dimensions. While the approach brings together several existing ideas, it is implemented carefully and evaluated on both synthetic and real datasets.

### Strengths
1.	This work introduces a new, novel method (NQR) that connects neural optimal transport with VQR.
2.	Optimality guarantee (Theorem 3).
3.	The writing is generally clear and well-structured.
4.	The experimental results show that the proposed approach yields better performance compared to the baselines.

### Weaknesses
1.	The paper’s novelty is moderate rather and not groundbreaking. Most of its core components, such as PICNN/ICNN (Makkuva et al., 2020; Amos et al., 2017), amortized optimization (Amos, 2023), entropic regularization (Genevay et al., 2016), and recent reranking conformal methods (Thurin et al., 2025; Klein et al., 2025), have already been explored in earlier work. The main contribution lies in how these ideas are combined and adapted for the conditional setting. 
2.	The assumptions of Theorem 3, which is the main theoretical result of this paper, are hard to verify in practice.
3.	The conformal prediction experiments should include comparisons with VQR (Carlier et al., 2016, 2017, 2020), STDR (Feldman et al. 2023) and PCP (Wang et al. 2023).

### Questions
1.	How sensitive are the results to violations of the radial Jacobian assumption in Theorem 3?
2.	How does the proposed approach compare against modern conditional flow or diffusion models for conditional sampling and prediction sets?
3.	Can the proposed approach handle large output dimensions, e.g., $d_y > 20$?
4.	This paper introduces multiple methods: pb, rpb and hdp. Is there a suggestion for which of them to use in practice?

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
The general methodology is to rely on mutlivariate quantiles of Hallin et al. to define a multvariate score, converted into univariate score, from which conformal prediction tools can be applied.

The novelty is the nature of the optimal transport solver used to compute the center outward map. Instead of relying on entropic-regularized solvers (Sinkhorn) like Klein et al, authors rely on the gradient of a partial input convex neural network (PICNN). The training leverages the amortization trick of Amos et al to accelerate the fitting of the dual potential. 

The method is benchmarked on a few multivariate regression datasets, like the ones introduced by Rosenberg et al. The average Sliced-2 wasserstein-2 distance is computed to benchmark the method, and other metrics like coverage or log-volume of prediction sets are compared against competitors works or baselines.

### Strengths
### Motivation

The paper extend the work of Thurin et al; Klein et al using PICNN, a parameterized approach.

###  Exposition

The paper makes a good job at introducing the necessary tools (center outward quantiles, neural dual solvers, conformal prediction).

### PICNN training

ICNN (and PICNN) training can be challenging. Various variants are tested. The results of Figure 3 and 4, show strong results: this is my main motivation for leaning toward acceptance. Fig 1. confirms that this PICNN can fit complex shapes. Since the work is done in low dimensions (the Y remaining much smaller than the X space) PICNN don't suffer as much from the curse of dimensionality.

### Weaknesses
### Novelty on the multivariate conformal prediction part 

This is similar to the approach of two concurrent works (Thurin et al, Klein et al) that this paper acknowledges: using multivariate quantiles of Hallin et al, fit the (empirical) center outward map (or an approximation of thereof), and converting these generalized (multivariate) quantiles into 1D score .  

The main novelty is reliance on PICNN/ICNN instead of cubic linear solvers (Thurin et al) or entropic solvers (Klein et al), using other tricks documented in W2 Monge map learning.  

### Clarity

I had some issues other than navigate the experimental part, that lacked clarity at times. e.g Table 1 is not directly related to conformal prediction, it's simply measuring the distance between $T_{\sharp}P$ and $Q$ (ie it's a sanity check). Also some issues with Fig 2 (see below).

Similarly, it's not clear to me which message Fig 2. is trying to convey.

### Questions
### Fig. 2

Why is the Wasserstein distance reported as negative?

### Pareto front

You write:

> The re-ranking step of RPB and RPBS allows to achieve a slightly
sharper conditional coverage, but the increase in prediction sets volume make it a questionable trade-
off.

I think this observation mandates a Pareto front, to better position the relative strengths of each method.

### Soundness
3

### Presentation
2

### Contribution
2
