# Mitigating Forgetting in Low Rank Adaptation

- Decision: Reject
- Scores: 8, 6, 4, 2

## Abstract
Parameter-efficient fine-tuning methods, such as Low-Rank Adaptation (LoRA), enable fast specialization of large pre-trained models to different downstream applications. However, this process often leads to catastrophic forgetting of the model's prior domain knowledge. We address this issue with LaLoRA, a weight-space regularization technique that applies a Laplace approximation to Low-Rank Adaptation. Our approach estimates the model’s confidence in each parameter and constrains updates in high-curvature directions, preserving prior knowledge while enabling efficient target-domain learning. By applying the Laplace approximation only to the LoRA weights, the method remains lightweight. We evaluate LaLoRA by fine-tuning a Llama model for mathematical reasoning and demonstrate an improved learning-forgetting trade-off, which can be directly controlled via the method's regularization strength. We further explore different loss landscape curvature approximations for estimating parameter confidence, analyze the effect of the data used for the Laplace approximation, and study robustness across hyperparameters.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper targets catastrophic forgetting when fine-tuning LLMs with LoRA. It proposes LaLoRA, a curvature-aware regularizer built via a Laplace approximation computed over the LoRA weights only, then used to penalize updates along high-curvature (source-critical) directions during downstream training. Empirically, LaLoRA improves the stability–plasticity trade-off over LoRA and other baselines.

### Strengths
- Methodology is principled yet practical: local quadratic (Laplace) approximation over LoRA parameters gives per-weight uncertainty and a simple quadratic regularizer; computing it only for adapters keeps cost manageable and fits existing LoRA codepath. 
- Thorough experiments & analyses: the effectiveness and robustness of the proposed method is demonstrated in various different experiment settings. Moreover, the parameter updating behavior and sensitivity to data coverage are both analyzed.

### Weaknesses
- Resource reporting gap: I couldn’t find a quantitative comparison of memory/storage and runtime overhead against LoRA/MIGU/MiLoRA under the stability-improving settings shown; adding this would strengthen the practical case.  
- Data access assumption: effectiveness depends on having (proxy) source data to estimate curvature; guidance on minimal Ns and robustness is only partially discussed.

### Questions
- If one has no access to the pre-training data, is there any possible way of estimating $\Sigma$?
- Can the Laplace regularizer be adapted to other PEFT methods beyond original LoRA, such as DoRA, Pissa, MiLoRA? 
- The proposed algorithm is a two-stage one. Will it be benifit if we dynamically adapt $\Sigma$ during fine-tuning?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces LaLoRA, a new weight-space regularization technique designed to mitigate catastrophic forgetting in parameter-efficient fine-tuning scenarios, particularly with Low-Rank Adaptation (LoRA) layers in large language models. By leveraging a Laplace approximation, the method estimates parameter uncertainty and imposes a curvature-aware penalty that restricts changes to "critical" parameters within the LoRA adapters during downstream task fine-tuning. The method is evaluated on math reasoning transfer with Llama models, demonstrates improved learning-forgetting trade-offs over several recent baselines, and is analyzed for sensitivity to curvature approximation, source data, and hyperparameter settings.

### Strengths
1. **Clear Motivation**: The application of the Laplace approximation specifically to LoRA adapters is well-justified, namely finding those less important weights in pretraining. Moreover, it is demonstrated to be efficient, and avoids the prohibitive computational cost of full-parameter curvature modeling. The two-stage algorithm is clearly described and mathematically well-formulated, e.g., in Equations (1)-(5), with careful distinction between diagonal and structured Kronecker-Factored (K-FAC) curvature variants.

2. **Improved Pareto Trade-off**: The method effectively controls the stability-plasticity (learning-forgetting) trade-off via the regularization strength $\lambda$. Both main claims as improved source domain retention and competitive target domain learning are supported by compelling empirical evidence across Figures 2a, 2b, and Table 1. Moreover, the authors clearly demonstrate the advantages of the proposed methods with proper visualization.

3.  **Insightful Mechanism Analysis**: The paper investigates the actual update patterns induced by the regularizer (Figure 3a), showing that high-curvature ("important") weights are indeed less modified than more flexible ones. This is an excellent validation of the intuition behind the method.

### Weaknesses
1. **Unclear Theoretical Guarantees and Some Ambiguous Symbolism**: The Laplace-regularized loss in Equation (5) and associated regularizer expression could be made clearer, with more rigorous notation for how $\overline{\Sigma}$ is estimated, especially in multi-dataset settings. Although the practical motivation for restricting the Laplace approximation to LoRA weights is strong, a more explicit analysis of the cases where this is justified (i.e., under what assumptions low-rank space alone suffices to capture key catastrophic forgetting axes in full-parameter space) would strengthen the theoretical contribution.

2. **Limited Empirical Setting**: All experiments are conducted on a single base model (Llama-3.2-3B) and on mathematical reasoning as the target with commonsense datasets as proxies for source domains. While this setup is well-motivated (per Biderman et al., 2024; Shuttleworth et al., 2025), the conclusions are less generalizable to other domains and model families. Figure 11 does show per-dataset breakdown but broader task diversity would make the results more convincing.

3. **Unreported Implementation and Efficiency Details**: Several important implementation details remain unreported. For example, it is unclear whether adding the proposed regularization introduces additional GPU memory consumption or computational overhead. Furthermore, the paper does not discuss how efficiently the regularization terms can be computed in practice.

### Questions
1. Could the authors more rigorously quantify or theoretically bound how the choice of proxy dataset for the Laplace approximation affects retention? Is there an evaluable metric for proxy dataset representativeness? What are the signs of proxy failure?

2. Since the proposed regularization method is pretrained task-dependent, it often requires selecting a surrogate dataset when the original pretraining data are unavailable. This raises concerns about potential **dataset dependency**. Specifically, would the results vary significantly depending on the choice of surrogate dataset? Could different datasets lead to different degrees of memorization or forgetting? Furthermore, if certain samples are not represented in the surrogate dataset, is it possible for the model to still forget relevant information?

3. K-FAC vs. Diagonal: Why do richer curvature approximations (e.g., block tridiagonal K-FAC) not outperform diagonal ones? Is this due to rank limitations, poor sample efficiency, or structural redundancy in LoRA updates? Could the authors provide a concrete case (or synthetic experiment) demonstrating when more curvature pays off—or never does?

4. While excluding less important parameters may improve efficiency, it simultaneously restricts the flexibility of the adaptation process. The reviewer is uncertain whether, in very low-rank settings (e.g., r=1), this selective updating strategy could lead to degraded performance. Can authors test both very low and high rank cases and draw useful insights? The reviewer doubts the performance if r=1, since we have limited parameters to update, and narrawing down further may (significantly) hurt the performance.

5. Follow-up on Weakness 3 – Computational and Memory Overhead: As noted in Weakness 3, the paper does not clearly report the additional computational time and memory cost introduced by the proposed method. Moreover, for larger models, it remains uncertain whether computing the regularization terms would lead to substantial increases in training time or resource consumption. 

6. The authors mention using a diagonal Fisher Information Matrix (FIM) to compute $\Sigma^{-1}$. Could this simplification lead to a performance drop, particularly when dealing with high-dimensional gradients where off-diagonal correlations may be important? Furthermore, how efficient is this approximation in practice compared to using a full or low-rank FIM?

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
The paper proposes LaLoRA, a Laplace-regularized variant of LoRA designed to mitigate catastrophic forgetting during fine-tuning. It applies a Laplace approximation on LoRA parameters to estimate curvature and penalizes updates along high-curvature directions. Experiments on LLaMA-3B fine-tuning for GSM-8K show improved retention of pre-training knowledge with a minor loss in target performance.

### Strengths
- The idea is conceptually sound, combining Laplace-based uncertainty estimation with LoRA. 
- The paper draws a clear connection to EWC-style continual learning while adapting it to PEFT.

### Weaknesses
- The method assumes the availability of source or surrogate data to estimate curvature, which is unrealistic for most LLM fine-tuning scenarios. The proposed "minimal proxy batches" solution only partially addresses this. 
- No analysis of computational efficiency relative to vanilla LoRA is given; specifically, incorporating the cost from Stage I. 
- It is unclear how much source-domain data is required for LaLoRA to perform well, or if the regularization is robust when limited data are available. 
- Equations 11-12 are poorly typeset and confusing - I believe there is a notation error that makes it unclear how the regularization term is applied. 
- Experimental results are modest and high-variance (Table 2; no actual gain in performance with high variance). Only a single model/task was evaluated. 
- The claimed practicality needs some quantitative analysis, as Stage I adds extra computation and requires data that may not exist. 
- Evaluation lacks diversity; no tests on other domains or larger models, so generality remains unclear.

### Questions
Please refer to the weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes LaLoRA, a weight-space regularization technique that applies a Laplace approximation to LoRA. It estimates the model's confidence in each parameter and constrains updates in high-curvature directions, thereby preserving prior knowledge while enabling efficient target-domain learning. Experiments on mathematical reasoning tasks show an improved learning-forgetting trade-off that can be directly controlled via the method's regularization strength.

### Strengths
1. This paper is clearly written and well-motivated. The topic of combatting catastrophic forgetting is important.
2. The paper proposes an efficient approach to calculating the curvature information, specifically via Fisher information.
3. The experiments demonstrate that LaLoRA is effective in combatting forgetting.

### Weaknesses
1. The proposed method requires (a subset of) source data, which is typically unavailable for task-specific fine-tuning.
2. The significance of the proposed approach is questionable. In Figure 2(a), I find that the learning performance saturates around 2 epochs with very little forgetting. Thus, vanilla LoRA with early stopping is sufficient.
3. More baseline methods are needed, especially mentioned Bar, Flat-LoRA, etc.

[1] Implicit Regularization of Sharpness-Aware Minimization for Scale-Invariant Problems
[2] Flat-LoRA: Low-Rank Adaptation over a Flat Loss Landscape

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3
