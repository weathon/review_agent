# Input-Adaptive Bayesian Model Averaging

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 2, 6

## Abstract
This paper addresses prediction problems with multiple candidate models, where the goal is to combine their outputs. This task is especially challenging in heterogeneous settings, where different models may be better suited to different inputs. 
We propose Input-Adaptive Bayesian Model Averaging (IABMA), a Bayesian method that assigns model weights conditional on the input.
IABMA employs an input-adaptive prior, and yields a posterior distribution that adapts to each prediction, which we estimate via amortized variational inference.
We derive formal guarantees for its performance relative to any single predictor selected per input, and evaluate IABMA across regression and classification tasks, studying data from personalized cancer treatment, credit-card fraud detection, and UCI datasets. IABMA consistently delivers more accurate and better-calibrated predictions than both non-adaptive baselines and existing adaptive methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the challenge/difficulty of combining multiple predictive models, which is especially difficult in heterogeneous settings where different models may be optimal for different inputs. Standard model averaging uses a single set of global weights, which performs poorly in these scenarios. To solve this, the authors propose Input-Adaptive Bayesian Model Averaging (i.e. the IABMA method), a Bayesian method that calculates specific weights for each model conditional on the input data $x$.

The IABMA method models the choice of the best model as a random selector function g that depends on the input x. It uses an input-adaptive prior and then calculates a posterior distribution over which the model is most plausible given the training data ($D$) and the specific input $x$. This posterior distribution, estimated using amortized variational inference, directly gives the optimal, input-specific weights for combining the models. The authors derive formal guarantees for the performance of IABMA and then evaluate it on regression and classification tasks, including popular UCI datasets such as personalized cancer treatment and fraud detection. The results demonstrate that IABMA consistently delivers competitive performance to existing non-adaptive  and adaptive methods.

### Strengths
One of the main strengths of the paper is that it is clearly written and structured. The authors lay out their argument in a straightforward manner, making the text and the core concepts of the proposed IABMA method relatively easy to follow.

The paper addresses an interesting and relevant problem in model averaging, particularly for heterogeneous data where input-specific models are needed. The proposal itself is presented in a "clean" and simple probabilistic formulation. The paragraphs are well written, the theory is well motivated and the flow of the paper is clear. The intuitive example in section 3.1 as well as the actual taking care of the problem (e.g. how to fit the variational distributions or how to optimize the KL divergence) are well formulated and explained.

Additionally, the authors provide motivation for their framework by connecting it to practical examples like personalized medicine and fraud detection. They also include some theoretical development to support their method, which helps to frame the potential benefits of the approach.

### Weaknesses
While the core idea of the paper is clean and simple, this simplicity in my opinion necessitates a much stronger and elaborate set of experiments to fully justify its contribution. The initial motivation and theory are strong, but the paper ultimately falls short due to a weak and insufficiently comprehensive evaluation. This gap between the promising setup and the empirical evidence was a bit disappointing.

In my opinion, the existing experiments are not presented in a convincing manner. For instance, Figures 2 and 3 take a significant amount of space to present results that could be summarized more efficiently in a small table, especially since the paper has some space to spare. More importantly, it is not clear from the results that the authors have obtained if the performance differences between the proposed method and the baselines is statistically significant (it is not readable from the plots). The comparison to other adaptive competitors, such as Mixture of Experts for example, feels underdeveloped. In many cases, the results do not show a lcear advantage for IABMA over others, and the authors missed an opportunity to properly articulate why one would choose their method over existing, well-established alternatives.

Furthermore, the scope of the analysis is quite limited. The authors only consider a small set of candidate models (e.g., four regressors), which leaves several important questions unanswered. A more thorough evaluation could have investigated:
1.  How the method's performance scales with a larger number of candidate models.
2.  How the method performs when several predictors yield similar (or maybe redundant) performance
3.  The magnitude of the performance gain in a setting specifically designed to be highly heterogeneous, where input-adaptive averaging would be expected to clearly outperform input-agnostic methods.

Finally, the paper could be strengthened by doing some rewriting of its analyses and its context. The qualitative analysis in Appendix B.1, for example, is quite interesting and a portion of it should have been included in the main text to provide better insight and overview of the performance. Given the community's shift toward larger datasets, the paper would also benefit from at least one or two analyses on a larger-scale benchmark to demonstrate its relevance and scalability. Also, there is no discussion of computational cost; it is unclear if fitting the amortized variational posterior is more or less costly than maximizing the likelihood in methods like MoE, or how the methods compare in terms of speed.

### Questions
Please see weaknesses section.

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
4

### Summary
The authors study the interesting and important question of how to adaptively combine multiple prediction models, where here “adaptively” refers to a weighted mixture of the models with weights that depend on the test-point input covariates. The goal is both to improve overall performance relative to an individual model, as well as to improve “personalization,” meaning that the best model is used for a given input. The main claimed contribution is to present a *Bayesian* approach to adaptive model averaging--to contrast their contribution from prior work, the authors claim that “Previous adaptive approaches (see Section 1.1) addressed the task of specifying the adaptive weights $\alpha_j(x)$ from a *frequentist* point of view[...]” (emphasis added). The authors claim a theoretical guarantee that compares the proposed approach to the performance of the individual models, and they present experiments on simulated data, as well as on cancer drug-response prediction and credit-card fraud detection.

### Strengths
As mentioned, the authors study an important problem of how to best personalized combined models, that is, by taking an input-adaptive approach to model averaging. The approach seems reasonable, well-motivated, and empirical results appear okay. The theoretical analysis seems okay too, although I didn’t yet check it carefully due to the main weakness (see next section), for which is the main factor influencing my evaluation.

### Weaknesses
**Originality/significance:** In my view, the main weakness of the paper is that it does not make clear if/how the proposed methods differ from or improve on prior approaches to input-adaptive model averaging, including Bayesian approaches. In particular, there are at least two large categories of methods that I think at least the related work, and probably also the experiments section, should compare against more thoroughly: (1) mixture-of-experts models and (2) dependent Bayesian mixture models (eg, dependent Dirichlet mixture models or even input-dependent Gaussian mixture models). 

For example, in the related work, the authors claim “Few methods assign input-dependent weights” before only providing one citation on mixture-of-experts, despite this area having a very large literature on data-dependent model averaging. It’s also not clear to me why MoE is not considered Bayesian. Eg, the authors could consult the following review papers:
- Masoudnia, S., & Ebrahimpour, R. (2014). Mixture of experts: a literature survey. Artificial Intelligence Review, 42(2), 275-293.
- Mu, S., & Lin, S. (2025). A comprehensive survey of mixture-of-experts: Algorithms, theory, and applications. arXiv preprint arXiv:2503.07137.
- Yuksel, S. E., Wilson, J. N., & Gader, P. D. (2012). Twenty years of mixture of experts. IEEE transactions on neural networks and learning systems, 23(8), 1177-1193.

Regarding dependent Bayesian mixture models, the authors could consult the following reviews, in particular with an eye to methods that are “covariate-dependent” mixture models:
- Barcella, W., De Iorio, M., & Baio, G. (2017). A comparative review of variable selection techniques for covariate dependent Dirichlet process mixture models. Canadian Journal of Statistics, 45(3), 254-273.
- Quintana, F. A., Müller, P., Jara, A., & MacEachern, S. N. (2022). The dependent Dirichlet process and related models. Statistical Science, 37(1), 24-41.

### Questions
Could the authors please clarify how the proposed methods relate to existing literature on mixture of experts and/or input/covariate-dependent Bayesian mixture models? This seems necessary to properly understanding the paper’s contribution, and I didn’t find this sufficiently discussed in the paper.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a framework for combining multiple predictive models by assigning input-specific weights in a Bayesian manner. Unlike classical Bayesian Model Averaging (BMA), which uses global model weights, IABMA introduces an input-dependent prior over model-selection functions, leading to input-adaptive posterior weights. The posterior is approximated using amortized variational inference, yielding instance-specific model weightings. The authors derive a finite-sample likelihood guarantee showing that the proposed predictor performs competitively with the best per-input model selector.

### Strengths
Conceptual coherence: The probabilistic formulation that derives adaptive weights from a Bayesian posterior is principled and internally consistent.

Empirical breadth: The experiments span both regression and classification, synthetic and real datasets, with comparisons to multiple baselines.

### Weaknesses
1.	Limited novelty: The proposed approach is largely a Bayesian reinterpretation of existing adaptive ensemble methods such as Mixture of Experts and Bayesian Hierarchical Stacking . The key innovation, introducing an input-dependent prior, is conceptually modest and primarily repackages known ideas in new notation.

2.	Superficial theoretical development: The likelihood guarantee is a straightforward adaptation of Jensen’s inequality, offering minimal insight into the behavior of the amortized inference procedure or its generalization properties. No convergence analysis, uncertainty quantification, or theoretical justification for the variational approximation is provided.

3.	Lack of methodological clarity: The construction of the input-adaptive prior is heuristic, based on an “energy” integral that lacks intuitive interpretation and appears computationally impractical for high-dimensional continuous outcomes. The method relies on ad-hoc Monte Carlo approximations, raising concerns about scalability and stability.

4.	Amortized inference design is under-specified: The paper treats the amortized posterior network as a black box, without ablation or sensitivity analysis on architecture, optimization, or overfitting. It is unclear whether performance gains come from the variational parameterization or the adaptive prior itself.

5.	Empirical evidence is weakly convincing: Improvements over baselines are small and inconsistent across datasets; Comparisons do not include recent or stronger baselines in adaptive ensembling (e.g., deep mixture-of-experts architectures); Some tasks (e.g., PRISM, fraud detection) lack details on train/test splits, data leakage control, and statistical significance of reported differences.

6.	Overstated claims: The paper claims “formal guarantees” and “Bayes-optimal adaptive weights,” but these rely on unverified approximations and assumptions. The results fall short of demonstrating real-world robustness or interpretability advantages.

7.	Expository issues: While the paper is long and dense, it lacks intuition; the heavy notation and abstract measure-theoretic framing (e.g., pushforward arguments) obscure rather than clarify the contribution.

### Questions
How does IABMA differ in substance from Mixture of Experts with a Bayesian treatment of gating? What new insights or properties does the input-dependent prior confer?

How sensitive is the model to the design of the energy-based prior and the range of integration for continuous outcomes?

What guarantees (if any) can be provided for the variational approximation quality or convergence of amortized inference?

Could the same adaptive weighting effect be achieved more simply with a discriminatively trained gating network, without the Bayesian formalism?

What is the computational complexity of evaluating Eq. (9) and optimizing the ELBO in large-scale settings?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
The paper studies a Bayesian way to combine different models.

### Strengths
Cannot assess

### Weaknesses
cannot assess

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
3
