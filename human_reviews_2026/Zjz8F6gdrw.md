# How Stable is the Next Token? A Geometric View of LLM Prediction Stability

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
Large Language Models (LLMs) exhibit impressive capabilities yet suffer from sensitivity to slight input context variations, hampering reliability. Conventional metrics like accuracy and perplexity fail to assess local prediction robustness, as normalized output probabilities can obscure the underlying resilience of an LLM's internal state to perturbations. We introduce the Token Constraint Bound ($\delta_{\mathrm{TCB}}$), a novel metric that quantifies the maximum internal state perturbation an LLM can withstand before its dominant next-token prediction significantly changes. Intrinsically linked to output embedding space geometry, $\delta_{\mathrm{TCB}}$ provides insights into the stability of the model's internal predictive commitment. Our experiments show $\delta_{\mathrm{TCB}}$ correlates with effective prompt engineering and uncovers critical prediction instabilities missed by perplexity during in-context learning and text generation. $\delta_{\mathrm{TCB}}$ offers a principled, complementary approach to analyze and potentially improve the contextual stability of LLM predictions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose the Token Constraint Bound ($\delta_{TCB}$), which is a metric that quantifies the amount the hidden states of an LLM can be perturbed without significant change in the output. They argue that $\delta_{TCB}$ measures the stability of an LLM for a specific input using the internal geometry of the model. They show through experiments that optimizing for $\delta_{TCB}$ allows for better prompt engineering and stability.

### Strengths
- The paper is well-motivated. Quantifying the stability of predictions is a useful task for understanding the behavior of LLM predictions.
- The application to prompt engineering is interesting and has high potential for practical applications.

### Weaknesses
While the motivation and use case of this approach are interesting, how this method compares relative to other approaches, both in formulation and in experimental results, is not clear from the paper. To me, positioning this approach in comparison to other robustness metrics is necessary.

1. There is limited literature review on prior methods for evaluating the robustness of neural networks or LLMs. The authors discuss perplexity and confidence metrics, but there are many approaches related to quantifying robustness that are not discussed in the paper or appendix. Many of these approaches also rely on computing the response to perturbations, so it is not clear to me where the method proposed by the authors is positioned relative to these works. For example, [1] seems like a very related method computed for neural networks. 
2. The experiments provide limited baseline comparisons to existing methods. The only method that the authors compare to for prompt optimization is optimizing for perplexity. Even for this experiment, there is only a gain in worst-case accuracy and standard deviations are not reported. 
3. The experiments are only shown for a single model (Llama-3.1-8B). Additional evaluations for other models would strengthen the experimental results.

[1] Sensitivity and Generalization in Neural Networks: an Empirical Study. Novak et al., 2018.

### Questions
- Do the authors test how large the approximation error in Equation 3? To me, the softmax nonlinearity makes it unclear that this approximation will hold consistently unless $W\Delta h$ is very small relative to $Wh$
- Could the authors clarify the context of the experiment in 4.3.1? Without knowing what the "gsm8k_811" question is, it is unclear what the the interventions are doing.

Minor typo: spaces missing in lines 47-48

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces the Token Constraint Bound, a metric that quantifies the maximum internal state perturbation an LLM can withstand before its primary next-token prediction changes significantly. The authors' experiments demonstrate δTCB’s utility in assessing prompt effectiveness and its ability to uncover critical prediction instabilities missed by perplexity. This evidence positions TCB as a valuable complementary tool for analyzing and potentially enhancing the contextual robustness of Large Language Models.

### Strengths
This paper defines a novel metric, TCB, to evaluate the connection between a model's hidden state and its output instability. It provides rigorous theoretical proofs and derivations, which are substantiated by thorough experimental validation and data analysis. The establishment of this new metric is highly significant for the interpretability of large models and for prompt engineering. This work contributes to a research area that is both novel and highly promising.

### Weaknesses
1. The entire mathematical derivation of TCB is built upon the assumption of $L_2$ norm perturbations for the hidden state **h** and the Euclidean distance between output embeddings $w_i$. This is a very strong assumption. Semantic similarity in high-dimensional embedding spaces is not always well-captured by Euclidean distance; two semantically related tokens may not be proximal in terms of $L_2$ distance.  It is more likely that meaningful perturbations occur along specific semantic directions. TCB might be completely insensitive to these structured, non-isotropic perturbations. While I understand that the $L_2$ norm is a natural choice, I would like to see a more detailed explanation and analysis justifying this simplifying assumption.

2. The paper compares TCB with perplexity (PPL), but the comparison against simpler, less computationally expensive uncertainty metrics is insufficient. This includes metrics such as the entropy of the output probability distribution and the difference between the top-1 and top-2 probabilities.

### Questions
1. **Practicality of TCB for Inference:** The computation of TCB requires a summation over the entire vocabulary *V*. For modern large models, the vocabulary size *V*  typically ranges from 32,000 to over 100,000. This implies that for the generation of *each* token, a complex geometric computation involving the entire vocabulary must be performed. This appears to be computationally prohibitive, potentially rendering TCB infeasible for any practical applications requiring low latency. The manuscript currently lacks a detailed analysis of TCB's computational cost, and I would appreciate it if the authors could provide this information.
2. The current experiments are based solely on the Llama-3.1-8B model. I would like to understand the extent to which the effectiveness of TCB generalizes to models of different scales and architectures. I would encourage the authors to supplement their findings with experiments on a smaller model (e.g., in the 3B parameter range) or on models with different training provenances, such as Qwen or the Pythia suite, which is designed for such analyses.

To summarize, my primary concerns lie with **Weakness 1**  and **Question 2**. I would be willing to reconsider my score should the authors provide a satisfactory response to these points.

### Soundness
4

### Presentation
4

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
This paper introduces a novel metric called the Token Constraint Bound ($\delta_{TCB}$) to quantify the local stability and robustness of llms next-token predictions. The authors argue that conventional metrics like perplexity (PPL) fail to capture the true resilience of the model's internal state to minor perturbations. $\delta_{TCB}$ is defined from a \textit{geometric perspective} in the output embedding space as the maximum internal state perturbation an LLM can withstand before its dominant prediction significantly changes. Experiments demonstrate that $\delta_{TCB}$ correlates with effective prompt engineering and uncovers critical prediction instabilities missed by PPL, offering a principled and complementary tool for analyzing and improving LLM prediction reliability.

### Strengths
1. The core contribution, $\delta_{TCB}$, is highly original. It shifts away from traditional probability-based evaluation by proposing a geometry-based metric that quantifies stability as a “safety margin,” providing a novel and interpretable theoretical perspective on LLM reliability.

2. The metric is clearly defined and theoretically supported by its intrinsic link to the geometry of the output embedding space. Experimental results effectively validate its use as a complementary tool that can identify prediction vulnerabilities that perplexity fails to capture.

3. The paper clearly articulates the motivation for the metric, highlighting the limitations of PPL in assessing local robustness. The definition of $\delta_{TCB}$ and its connection to geometric principles are well-explained.

4. $\delta_{TCB}$ holds significant potential for real-world impact. It can serve as a quantitative measure of \textit{contextual effectiveness}, guiding the development of more robust prompt engineering techniques, and is a valuable addition to the field of LLM reliability and risk assessment.

### Weaknesses
1. $\delta_{TCB}$ exclusively measures the stability of the predicted token, without incorporating semantic meaning or factual correctness. This is a critical limitation as the metric cannot distinguish between a "stably correct" and a "stably incorrect" prediction. Its interpretability must always be tied to traditional accuracy metrics.

2. $\delta_{TCB}$ is a geometric distance metric, but the model's prediction behavior is governed by the Softmax function. Changes in the Temperature parameter can drastically alter the prediction probability distribution. The authors must clarify whether $\delta_{TCB}$ is independent of Softmax temperature or detail how temperature variations affect the correlation between the metric value and empirically observed robustness.

3. The exact calculation of $\delta_{TCB}$ involves finding the maximum perturbation radius by identifying the closest "competitor" token in a high-dimensional space. For LLMs with massive vocabularies, the time complexity and computational overhead may be substantial. The authors should discuss the efficiency bottlenecks in practical applications and explore or propose more efficient approximation algorithms to enhance its practical utility.

### Questions
1. $\delta_{TCB}$ is primarily a local metric focusing on the next token's stability. How do the authors propose to generalize or extend this local measurement to assess the overall stability of an entire sentence, paragraph, or long-form generated result? For multi-step reasoning tasks, how can this local metric accurately reflect global robustness?
2. $\delta_{TCB}$ is currently an evaluation metric. Is it possible to integrate $\delta_{TCB}$ into the model's \textit{training objective} or fine-tuning process? For instance, could a loss function be designed to explicitly \textit{maximize $\delta_{TCB}$} to train a model that is inherently more locally robust?
3. $\delta_{TCB}$ is grounded in the geometry of the \textit{Output Embedding Space}. What is the relationship between $\delta_{TCB}$ and the stability of other critical internal components, such as \textit{attention weights} or the outputs of intermediate hidden layers? Can an analysis of $\delta_{TCB}$ across layer depths help pinpoint the most fragile or robust architectural components within the model?

### Soundness
3

### Presentation
2

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
This work introduces a measure called Token Constraint Bound that measures how much changes in the internal representation an LLM can withstand before the token prediction changes. The goal is to measure this in additional to evaluate metrics to measure robustness citing examples that the models’ output change a lot depending on the prompt formatting, context etc.  This work derives TCB from the Jacobian of the softmax output with respect to the hidden state and show that it is intimately tied to the geometry of the output embedding space. They also give theoretical insights into their metric and provide experiments which corroborate that this metric aligns with effective prompt engineering and in-context examples and identifies brittle predictions missed by perplexity.

### Strengths
- The connection of TCB to the geometry of the output embedding space is quite interesting. 
- Their method can be used to select prompts and in-context examples which have higher accuracy as well as robustness. This approach seems novel to me and would be of interest to the community.

### Weaknesses
- This work considers that the output probabilities should not change by epsilon. I am not sure why we need a fixed epsilon because if the model is already very confident, then this epsilon can be large. Hence, instead of being a constant, this should depend on how confident the probabilities are.
- Also, it seems like this bound could be very loose in practice.
- The experiments are only limited to one model Llama 8b.
- Overall, I feel the experiments section is weak as many details are not clearly presented (see questions below).

### Questions
- How do you expect the findings to change depending on the model size? Moreover, it seems like the values of TCB would depend on the model family and hence, it would be hard to tie a value with stability across various models.
- The authors mention “The Low-Veﬀ Targeted Dataset (LVD) was created by modifying DPD prompts to generate high-confidence predictions.”. Can the authors please elaborate how these were created? 
- Also, the authors mention ‘We synthetically manipulated W (clustering/dispersing competitor embeddings) while holding h and o (thus local PPL) constant for diverse MMLU prompts.” Can the authors please clarify how they manipulated W?
- The authors use ~300 prompts for computing their results. Can the authors please comment if these were enough and how much was the variation across runs?
- How would their method apply to prompts where the answer is not a couple of words?
- In table 3, the authors measure variance of accuracy. What is the randomness here in these experiments?
- What temperature is used here for the experiments and how would different sampling techniques affect the results?
- In table 3, the authors mention that they select prompts based on accuracy and TCB? What is the exact criterion that they use because later, they mention that prompts with very high TCB can be confidently wrong.

### Soundness
2

### Presentation
2

### Contribution
3
