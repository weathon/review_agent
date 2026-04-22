# Dynamic Priors in Bayesian Optimization for Hyperparameter Optimization

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Hyperparameter optimization (HPO), for example, based on Bayesian optimization (BO), supports users in designing models well-suited for a given dataset. HPO has proven its effectiveness on several applications, ranging from classical machine learning for tabular data to deep neural networks for computer vision and transformers for natural language processing. However, HPO still sometimes lacks acceptance by machine learning experts due to its black-box nature and limited user control. Addressing this, first approaches have been proposed to initialize BO methods with expert knowledge. However, these approaches do not allow for online steering *during* the optimization process. In this paper, we introduce a novel method that enables repeated interventions to steer BO via user input, specifying expert knowledge and user preferences at runtime of the HPO process in the form of prior distributions. To this end, we generalize an existing method, \pibo, preserving theoretical guarantees. We also introduce a misleading prior detection scheme, which allows protection against harmful user inputs. In our experimental evaluation, we demonstrate that our method can effectively incorporate multiple priors, leveraging informative priors, whereas misleading priors are reliably rejected or overcome. Thereby, we achieve competitiveness to unperturbed BO.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a Bayesian optimization approach that enables the incorporation of human-defined priors into the decision-making process. In contrast to previous work that only supported a single prior, this method allows for the use of multiple priors at different time steps.
To mitigate the risk of getting trapped in a local mode due to misspecified priors, the paper introduces a simple heuristic for rejecting priors when necessary.

### Strengths
- The paper is generally well written and easy to follow.

- The empirical evaluation appears thorough, comparing different priors across various benchmarks. The chosen baselines are reasonable, though the evaluation could potentially be enriched by including additional methods from the literature, for example Seng et al.

- The paper includes an ablation study on the sensitivity of the hyperparameter tau, which seems to play a central role in the proposed method.

### Weaknesses
Significance: While the motivation to incorporate user-provided priors into the optimization process is clear and conceptually appealing, I have some concerns regarding its practical applicability. In many real-world scenarios, I would argue that users might find it more intuitive to provide such priors only at the beginning of the optimization rather than continuously throughout the process. Especially in an AutoML context, where the goal is typically to maximize automation.

Moreover, the current procedure for constructing priors appears somewhat artificial and may not fully capture how practitioners would naturally define priors in practice. Finally, the proposed prior safeguard introduces an additional hyperparameter, τ, which, based on Figure 6, seems to have a substantial influence on performance and may need to be tuned according to the strength of the prior. This raises questions about the stability of the method in practical settings and whether it might introduce additional complexity in real-world applications.

Novelty: The main contribution, compared to the approach by Hvarfner et al., appears to be the weighting of the acquisition function based on the product of multiple priors instead of a single one. However, it seems that the method by Hvarfner et al. could, in principle, also replace the prior during the optimization process. Therefore, it remains somewhat unclear how much additional benefit is gained from using a product of multiple priors compared to relying on a single prior.

### Questions
- How are priors selected during the optimization process? For instance, in Figure 4, for expert priors, do you sample a new cluster at each decision point (vertical line)?

- Section 4.2, line 542: Could you clarify how a Normal distribution is placed on a prior?

- line 266: Do you mean f(\hat{\lambda}) - f(\lambda_{\star})

- Why do you use LCB for Equation 3 but EI as the acquisition function?



## Typos:

- line 327: missing \cite command

### Soundness
3

### Presentation
3

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
The paper considers the problem of informing hyperparameter optimization (HPO) via Bayesian Optimization through a user-specified “priors” / priority functions, which are used to reweigh the BO acquisition function.  Expanding on previous work which introduced this idea, the proposed approach uses multiple priority functions introduced at different time points, which are aggregated multiplicatively.  An additional filtering step is used to reject potentially priority functions with outsized impact.  A theoretical analysis aims to provide convergence properties (Thm 1), robustness to misleading priors (Thm 2) and utility of informative priors (Thm 3). An experimental study evaluates the proposed method on heuristically defined, data-driven priors (obtained using BO also).

### Strengths
- Putting the human in the loop with HPO is an interesting (but not new) problem.
- The experimental study is carried out on realistic HPO tuning tasks (albeit I have concerns about the choice of priors as below).

### Weaknesses
- The theoretical analyses provided are only asymptotic in nature; this is especially misleading for Theorem 2, where for finite-time, clearly misspecified priors will slow down performance.  The analysis essentially relies on the fact that asymptotically the effect of the prior vanishes / becomes trivial.
- The theoretical analysis seems to have some issues, and the exposition contains some vague / unjustified statements (see questions below)
- While the objective functions in the experiment come from realistic AutoML tasks, the choice of priors seems rather simplistic (Section 6.1)
- In contrast to prior work on human-in-the-loop HPO (cf Xu et al ‘24), the paper does not actually carry out any experiments with humans in the loop.

### Questions
- It seems rather odd that for a Bayesian method, the “prior” is not encoded as a Bayesian prior (e.g., through choice of the mean / kernel function).  Why was this not considered?
- There seems to be an implicit assumption that acquisition function needs to be positive (such as expected improvement / probability of improvement). This is generally not the case, e.g. for UCB/LCB.  Is any normalization etc. assumed?
- The approach also seems to require that the design space is continuous.  What about discrete design spaces? The filtering rule (3) would not apply?
- How should the parameters of the normal distribution in (3) to be picked? What about the threshold \tau?
- Thm 1: what are assumptions on acquisition function? The approach is stated in generality, but the proof of Theorem 1 seems to implicitly rely on using UCB?
- There seems to be an issue in the analysis:  Page 16 states that sublinear regret (i.e.., 1/T\sum_{t=1}^T (f(\lambda*)-f(\lambda_t)) \to 0 for t\to \infty) implies vanishing instantaneous regret (i.e., f(\lambda_t)\to f(\lambda*)) – this is not generally true!  Are there any additional implicit assumptions?
- Some statements are quite vague.  E.g., A.1: “established independently of the UCB acquisition function” very vague; it doesn’t hold e.g., for trivial acquisition functions that only pick a single action. Again, are there any implicit assumptions?
- The statement “asymptotic convergence rate improves” in A.3 also not clear. Can you please elaborate?

### Soundness
1

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
The paper proposed DynaBO, a Bayesian Optimization algorithm that can integrate prior information into the standard BO framework to improve the solutions. DynaBO generalizes piBO, a state-of-the-art prior-based BO algorithm, from a single to multiple priors provided by users over time. The key idea is to take the product of the user-specified priors and multiply by the acquisition function values, generating a dynamically-adapted acquisition function, which are maximized for the next data points. The authors further propose to reduce the impact of older priors by scaling them with the inverse of the time step factor. The proposed method is supported by a theoretical analysis of the convergence rate, robustness to misleading priors and convergence acceleration with informative priors. Empirically, DynaBO is compared with vanilla BO and piBO on many benchmark problems, under different prior settings (expert, advanced, local and adversarial).

### Strengths
- The paper is clearly written.
- The idea is novel, extending a previous work to incorporate dynamic priors.
- There are many analyses to support the work, including theoretical analysis and experimental results. Regarding the experiments, there are many different benchmark settings, including different prior settings (from informative to misleading priors), different prior supplying times (fixed and random times) and different surrogate models (Gaussian Process and Random Forest).

### Weaknesses
1.	The choice of square exponent in the formula in line 206 is not clearly explained. The authors mention it is to avoid the slow fading of old priors, but it still does not explain why square (rather than other functions such as cubic, etc.) should be used. There should be another ablation study on this choice, because I think the fading speed of older priors is an important factor to consider.
2.  It is not clear why some baselines are not compared in the experiments, such as PriorBand (Mallik et al. 2023) or BOPrO (Souza et al., 2021a).

### Questions
1.	Would there be any issue if more priors are supplied? By definition, the equation in line 206 computes the product of the priors, so when there are many (but still finite) priors, the product would become nearly 0 regardless of the original acquisition values, because each prior is in (0, 1]. According to Appendix B.2, some clippings are introduced to keep the priors at 1e-12, however, given a large number of priors, there will be many priors with very small values, which I think will still reduce the final product to be very small, and affect the original acquisition function values. Can the authors elaborate more on this? Also, have the authors run experiments with more priors in increased budget scenarios (Fig. 5), such as with 20-50 priors?
2.	Regarding the prior rejection threshold tau, in line 264, the authors explained about the choices of tau for EI and LCB acquisition functions. How about other acquisition function choices? Also, will the ablation study in Sec. 6.4 still be correct if EI and LCB are not used?

### Soundness
3

### Presentation
3

### Contribution
3
