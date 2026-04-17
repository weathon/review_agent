# Scaling Laws for Diffusion Transformers

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Diffusion transformers (DiT) have already achieved appealing synthesis and scaling properties in content recreation, \emph{e.g.,} image and video generation.

However, scaling laws of DiT are less explored, which usually offer precise predictions regarding optimal model size and data requirements given a specific compute budget.

Therefore, experiments across a broad range of compute budgets, from 1e17 to 6e18 FLOPs are conducted to confirm the existence of scaling laws in DiT \emph{for the first time}. Concretely, the loss of pretraining DiT also follows a power-law relationship with the involved compute.

Based on the scaling law, we can not only determine the optimal model size and required data but also accurately predict the text-to-image generation loss given a model with 1B parameters and a compute budget of 1.5e21 FLOPs.

Additionally, we also demonstrate that the trend of pretraining loss matches the generation performances (e.g., FID), even across various datasets, which complements the mapping from compute to synthesis quality and thus provides a predictable benchmark that assesses model performance and data quality at a reduced cost.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents the first systematic investigation of the scaling laws of Diffusion Transformers (DiT) for text-to-image synthesis. The authors conduct extensive expriements accross compute budgets $C$ from $1e17$ to $6e18$, and predict the performance for a larger 1B-model with compute budgets $1.5e21$.

The key contributions are:
1. **First Establishment of Scaling Laws of DiT**: This paper is the first to investigate and confirm the power-law relationship with the compute budget.
1. **Fit the Power Law Equations**: This papers fits the power-law equations for model/data size: $N_{opt}=0.0009\cdot C^{0.5681}$ and $N_{opt}=186.8535\cdot C^{0.4319}$, as well as for the loss $L=3.3943\cdot C^{-0.0273}$.
1. **Validation on Larger Compute Budget**: The derived laws are validated by extrapolating to a significantly larger compute budget ( $1.5e21$), a 1B parameter model, and demonstrating that its loss and performance metrics match the predictions.
1. **Evalution Metrics Justification**: The authors also verify generative quality metrics like FID also follows the scaling laws ($FID = 2.2566 \times 10^6 \cdot C^{-0.234}$), as well as other evaluation metrics like VLB, exact likelihood, GenEval, and human preference scores. This also justifys that these evaluation metrics are suitable for text-to-image synthesis evaluation.
1. **Generative Robustness**: The paper demonstrates the robustness of these laws by showing they hold for out-of-domain (OOD) data (e.g., COCO) and across different model architectures.
1. **Guidance on Model/Data design**: This paper shows difference model architecture (In-Context *v.s.* Cross-Attention) has different coefficients on the power-law equations, which helps evalutate future model designs using small compute budgets.

### Strengths
1. This paper for the first time provide a systematic investigation and verification of predictive, quantitative scaling laws of Diffusion Transformer (DiT).
1. The expriments conducted to investigate the scaling laws of DiT is reasonable and thorough, which include large-scale validation, evaluation of various metrics and coefficents comparision of different models.
1. The proposal to use scaling exponents as a predictable benchmark is practically usable and provides the community with a powerful and low-cost tool for architectural and data-quality comparisons.

### Weaknesses
The paper is generally good, but there's some mirror "weaknesses" or improvement sugesstion. I am happy to increase the score if they are addressed.
1. **Contradiction in Section 4**: There is a small contradiction between the text and Table 1 when comparing model architectures. The text states, "The Cross-Attention Transformer exhibits a *larger* model exponent". However, Table 1 shows the Cross-Attention model exponent (0.54) is smaller than the Vanilla In-Context model's exponent (0.56). The text's conclusion—that "more resources should be allocated toward scaling the dataset" —is correct and does align with the table (data exponent 0.46 > 0.43).
1. **Some writtings are confusing**:
    1. Line 241, $C$ is the symbol of "Compute", but later it's referenced as compute budget in line 252. First-time reader might not know "Compute" and "Compute budget" are the same.
    1. There are three sub-figures in Figure 1, but they are refered as a whole, e.g., in Line 265, it should refer to the first sub-figure of Figure 1.
    1. The logic from Line 243 to 251 is not clear to me. It says $C=6ND$, a linear relationship between $C$ and $N$ or $D$, but later it ypothesizes $N_{opt} \propto C^{a}$ and $D_{opt} \propto C^{b}$. It's unclear to me why, untill I review Figure 1 that the x-axis is in log-scale and the fitting curve is a strage line. But still the y-axis of Figure 1 does not say they are $N_{opt}$ and $D_{opt}$ so it's still unclear to me how the hypothesis comes from.
    1. Figure 3 shows legend the second and the third sub-fogures of Figure 1 do not. And the second sub-figure of Figure 3's legend is *fitter* curve.

### Questions
1. As shown in Figure 3, as the $C$ increases, FID and Loss decrases, and the fitted curve is strage line (as y-axis is log-scale). However we cannot increase compute budget forever to obtain a minus loss, there should be saturation or even overfit. How the scaling curves show staturation and overfit?
1. For a fixed compute budge $C$, the relationship between it and $N$, $C$ is defined as $C=6ND$, but how about $N_{opt}$ and $D_{opt}$?  Say we have a designed a model, and fit the coefficents of $N_{opt}$ and $D_{opt}$, and we have limited compute budget, how can we balance the between $N_{opt}$ and $D_{opt}$?
1. The second and the third sub-figures of Figure 1, the y-axis should be *optimal* parameter $N_{opt}$ and *optimal* token $D_{opt}$?
1. Any experimental FID result of Vanilla In-Context vs Cross-Attention?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper investigates the scaling laws of text-to-image diffusion transformers (DiTs). It shows that these models follow a scaling behavior similar to that of large language models: the training loss and other performance metrics exhibit a power-law relationship with compute when the model size and token count are optimally balanced. Furthermore, the paper demonstrates that this trend generalizes across out-of-domain datasets and different model architectures.

### Strengths
* The paper is very well written.

* The findings enable practitioners to tune the hyperparameters of DiTs more efficiently.

* The paper demonstrates scaling laws not only with respect to the loss, but also for other useful metrics such as FID and human preference reward.

### Weaknesses
Beyond text-to-image generation, diffusion models have also been applied to tasks such as class-conditioned image generation and text-to-video generation. However, the paper only experiments on text-to-image generation, so it remains unclear whether the same scaling laws extend to these other tasks.

### Questions
Overall, I enjoyed reading this paper. While the techniques are not new and the results are not particularly surprising given the established scaling laws of large language models, confirming that similar laws hold for DiTs is nevertheless a valuable and solid contribution. I have a few questions listed below and hope the authors can address them in the rebuttal.

* I would like to confirm how Figure 2 is derived. Is it obtained using the same procedure as Figure 1? That is, for each metric, (1) train models of varying sizes under different compute budgets, (2) fit a parabola between the metric and model size for each compute budget, (3) identify the optimal model size for each compute budget, and (4) plot the metric versus compute budget using these optimal model sizes?

* Eq. 4: This equation is somewhat confusing. First, N is not defined. Second, the summation index i does not appear in the summand, making it unclear what is being summed over.

* In Section 3.1, "Variational Lower Bound" and "Exact Likelihood" are subpoints under "Likelihood." However, in the current formatting, "Likelihood," "Variational Lower Bound," and "Exact Likelihood" appear at the same heading level, which may mislead readers into thinking they are parallel sections, causing potential confusion.

* Figure 5: How many data points are included in each line?

* Section 4 states that the experiments show cross-attention transformers exhibit a superior scaling trend compared to in-context transformers. However, as discussed in Section 4, recent works suggest that in-context conditioning performs better. Could the authors clarify where this discrepancy might come from?

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
3

### Summary
This paper investigates the scaling laws of diffusion models for text-to-image generation. The authors conduct experiments across a wide range of compute budgets, spanning from 1e17 to 6e18 FLOPs. The results reveal empirical relationships between training loss, evaluation metrics, and compute expenditure. Based on these relationships, the paper claims that the performance of larger diffusion models can be accurately predicted.

### Strengths
- This paper addresses an important problem: establishing scaling laws for text-to-image generation with diffusion models. The findings have the potential to offer valuable insights to the community.
- The authors conduct extensive experiments and dedicate substantial effort to derive and validate the proposed scaling laws.

### Weaknesses
- In Figure 1, the assumption underlying the use of parabolic fitting for the performance curve is not clearly stated. If the curve is assumed to be unimodal, then a ternary search strategy could directly identify the optimal loss without requiring curve fitting.
- In Figure 20 (GenEval results), only the value "10" appears on the y-axis, making it difficult to determine the specific GenEval scores associated with each data point. Providing a complete y-axis scale would significantly improve readability.
- The scaling analyses use log scale for FID (Figure 3) and GenEval (Figure 20), but a linear scale for Human Preference Rewards (Figure 21). It would strengthen the consistency and interpretation of the results to explain why different scales are used and why each metric is expected to exhibit a linear or log-linear relationship with model performance.
- The analysis would be further strengthened by comparing the derived scaling laws to existing text-to-image diffusion models. Including these models in figures or table would provide a more comprehensive empirical grounding and reinforce the conclusions about scaling behavior.

### Questions
Please refer to the weakness session.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper is an empirical study on how pretraining loss and downstream generation quality of DiTs scale with compute, model size and data. It also shows that the pretraining loss follows a power-law relationship with compute. It proposes rules to to pick optimal model / data and to predict downstream metrics from pretraining loss. Additionally authors also identify the relationship between pretraining loss and generation performance.

### Strengths
Paper is very well-written and easy to follow. The method section has been particularly well-described.
1. Authors explore a very timely topic in diffusion based transformer models. 
2. Authors conduct experiments with DiT across various compute budgets and provide empirical proofs.
3. The paper also reports a correlation between pretraining loss and downstream FID, GenEval and human preference  metric which can be potentially beneficial for practitioners.

### Weaknesses
1. This paper fixes  a particular training dataset derived from Laion-5B. 
However, with generative models, we are observing that a careful data curation pipeline can affect the training and model scaling options. Authors can consider also studying how noisy / clean data can affect scaling properties.

2. The power-law relationship between training budget and generation performance provides a sign that the scaling law can predict generation performance. However, it is a bit unknown if this law will continue to hold under different hyper-parameter / diffusion sampling settings including varying classifier guidance strength for example.

### Questions
Same as weeknesses

### Soundness
3

### Presentation
4

### Contribution
3
