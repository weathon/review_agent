# Manipulate Large Language Models in Time Series Forecasting by Token Disruption

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Although Large Language Models (LLMs) have demonstrated substantial potential as powerful zero-shot time series forecasters, recent evidence shows that even small adversarial perturbations can significantly degrade their performance under strict black-box settings. However, existing attacks typically rely on repeated queries to the target LLM forecaster, making them easily detectable and anomalous in real-world scenarios. To overcome this limitation, we introduce the Token Disruption Attack (TDA) that generates perturbations by solely querying the local tokenizer rather than directly interfering with the model. We first formulate the attack as a non-convex optimization problem that maximizes the divergence in encodings produced by the target tokenizer, and then design a dynamic programming–based method to solve it efficiently. By injecting subtle perturbations into the raw time series, TDA induces substantial distortions during tokenization, which subsequently propagate through the model and ultimately result in severe forecasting errors. Extensive experiments on ten LLM-based and two non-LLM-based forecasters across six applications demonstrate that minor perturbations can cause large downstream distortions, leading to forecasting errors that increase by nearly 20%.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the vulnerability of LLMs as time series forecasting, where a subtle perturbations in raw time series can induce severe forecasting errors. Since existing attacks typically rely on repeated queries which are easily detectable, the authors introduce the Token Disruption Attack (TDA) that generates perturbations by solely querying the local tokenizer which is publicly available. The authors then formulate TDA as a non-convex optimization problem and design a dynamic programming-based method to solve it efficiently. Extensive experiments on 10 LLM-based forecasters across 6 applications demonstrate that TDA substantially degrades forecasting accuracy, while maintaining high stealth and efficiency.

### Strengths
1. **Originality of research question**: The authors clearly articulate the limitations of existing attacks -- iterative queris to the target forecasting model incurs high computational costs and makes them easily detectable. Building on this, they propose an original and well-defined research question with clear constraint -- manipulate an attack without querying the target model.
2. **Innovative method design**: Despite the strong constraint of operating without querying the target model, the authors leverage the publicly available tokenizer, and reformulates the attack in tokenization process.
3. **Extensive experiments and analysis**: The authors execute experiments on 10 LLM-based forecasters across 6 applications, and provide detailed quantitative and qualitative analyses. For example, the demonstration of imperceptible perturbation (Figure 3) is clear, we can observe that the input signals remain close, yet the outputs diverge significantly.

### Weaknesses
1. Although the overall method design is innovative, the technical description lacks sufficient depth, particularly for the dynamic programming–based perturbation generation presented in Section 4.3. The current form of Algorithm 1 is difficult to follow, and several key computational steps are not clearly explained in the main text. It is recommended that the authors provide additional descriptions or illustrative explanations of the algorithm’s critical steps to improve readability.
2. Some figures are not fully clear or consistent with the accompanying discussion. For example, in Figure 4, the clean sample points appear to be missing in the temporal space visualization. In Figure 5, while I agree that the input distributions are visually similar, I do not observe the rightward shift of input token distributions claimed in line 423. The authors may consider revising these figures or clarifying their visual interpretation to ensure consistency between text and visualization.

### Questions
1. Could the authors elaborate on the intuition behind the DP decomposition and how it guarantees effective perturbation search?
2. In Figure 4, the clean samples seem missing in the temporal space visualization. Could the authors clarify whether this omission was intentional or due to visual overlap?
3. In Figure 5, the text (line 423) mentions a rightward shift in the input token distribution under TDA, but this shift is not visually apparent. Could the authors explain how this shift is quantified, or revise the figure to better illustrate the described phenomenon?

### Soundness
3

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
This paper studies query-free attacks on LLM-based time series forecasters by perturbing inputs so that, after tokenization, the token encodings diverge substantially from the clean ones. The authors formalize a non-convex objective over token differences (considering temporal patterns, decimals, delimiters) and propose a dynamic-programming (DP)–based solver to select adversarial tokens under a budget constraint. Experiments across multiple LLM forecasters and datasets show 7–41% error increases with small perturbation budgets, while non-LLM forecasters are reported as largely unaffected.

### Strengths
[1] Introduces a novel attack surface by manipulating tokenization without querying the model.
[2] Dynamic programming heuristic is a creative method for non-convex search under budget constraints.
[3] Covers a diverse set of datasets and models, showing broad effectiveness.
[4] The paper provides insightful qualitative analyses, including embedding visualization, token alignment, and uncertainty distribution.

### Weaknesses
[1] In Algorithm 1, how are the candidate tokens N restricted in practice? This paper lacks clear interpretation and justification.
[2] Although the paper targets a novel attack surface, comparisons to adversarial methods on time series would contextualize the novelty and strength of the approach.
[3] Some replicated definition, define X_t \in R^{d \times T} at line 161 but X_t \in R^{T}  in Algorithm. This cause confusion.
[4] Writing errors, e.g Table header spells “Metrcis” instead of “Metrics”;
[5] There are duplicated “REFERENCES” section . A full References block appears (Page 10), then another References block starts again in Page 13 at Line 654.
[6] “Equation. 2” has an extra period (Line 192); use “Eq. (2)” or “Equation (2)”.

### Questions
[1] Since TDA operates through a local tokenizer, how transferable is the attack across different tokenization schemes
[2] The dynamic programming solver is claimed to be efficient, but the vocabulary size in LLM tokenizers is typically large. How is the candidate subset selected?
[3] All experiments use a 96→48 forecasting setup. How does the attack perform with longer horizons or variable-length input sequences?
[4] To what extent do formatting choices (e.g., token separators, decimal formatting, prompt templates) influence attack success? Could prompt engineering mitigate the attack impact?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a **Token Disruption Attack (TDA)**, a novel *query-free* adversarial method targeting large language models (LLMs) used for time series forecasting. TDA perturbs the **tokenization process** itself by introducing small modifications to the raw time series before tokenization. The method formulates this as a non-convex optimization problem that maximizes the divergence between tokenized representations of clean and perturbed sequences, solved heuristically via a **dynamic programming–based algorithm**.

### Strengths
(1) TDA is a query-free attack. This design improves stealthiness and computational efficiency, which are relevant considerations in real-world forecasting scenarios.

(2) The experimental section evaluates the proposed method on multiple LLM backbones (e.g., GPT-3.5/4, Llama-2, Mistral, Gemini, Claude) and diverse datasets (e.g., ETTh, Traffic, Weather, Solar), demonstrating that small perturbations can substantially degrade model performance.

(3) As LLMs increasingly appear in time series applications (e.g., TimeGPT, Chronos, Time-LLM), exploring their adversarial robustness is a relevant and timely problem for the ICLR community.

### Weaknesses
While the paper presents an interesting perspective on query-free attacks against LLM-based forecasters, several issues limit its clarity, rigor, and overall contribution.  

**(1) Ambiguous threat model.** The proposed attack assumes access to a “local tokenizer,” yet the paper fails to articulate any realistic setting in which an adversary could both control the local tokenizer and benefit from degraded forecasts. In practice, only the model provider (like OpenAI) or the end user would typically have access to the tokenizer, and neither party has an incentive to deliberately attack the model to obtain inaccurate results. This makes the attack scenario difficult to justify in real-world contexts.  

**(2) Poorly defined figures and metrics.** Section 4.2 introduces the concepts of “first-level” and “second-level” alignment without explicit labeling or quantitative definitions. Moreover, the mathematical exposition surrounding Eq. (4) is incomplete: the specific norm used is not identified, and the distinct roles of $token_i$ (whole tokens) and $token_j$ (fractional tokens) remain unclear. From my perspective, these two types of tokens appear to measure similar differences, so further explanation is needed to clarify how this index distinction meaningfully affects their functional roles.  

**(3) Weak algorithmic explanation.** Section 4.3 presents pseudocode but provides almost no intuitive description of how the dynamic-programming procedure accomplishes its optimization goal, nor why it represents a reasonable relaxation of the non-convex problem in Eq. (3). As a result, the algorithm feels more heuristic than principled.  

**(4) Lack of theoretical or empirical justification.** The paper offers no theoretical guarantee to explain why the proposed procedure could effectively disrupt forecasting outputs. Without theoretical support, the contribution (particularly concerning the role of the dynamic-programming design) remains largely empirical.  

**(5) Experimental concerns.** The experimental results show that TDA performs comparably or even worse than DGA on several datasets (e.g., *ETTh1* and *Traffic*), where DGA outperforms. These mixed outcomes weaken the claimed superiority of TDA and suggest that the proposed method may not provide consistent advantages across datasets or model families.

### Questions
See the weakness

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Token Disruption Attack (TDA), a purely black-box adversarial attack for time-series forecasting. Unlike existing attacks that require repeated interaction with the target system, TDA only needs access to a locally hosted tokenizer, which greatly improves stealthiness. Experiments on multiple LLM-based and non-LLM forecasting models show that TDA induces forecasting errors comparable to those of existing query-based attacks while being more efficient and stealthy.

### Strengths
1. Although the proposed technique is not conceptually novel from a general adversarial attack perspective, the authors identify an underexplored attack surface for LLM-based time-series forecasting.
2. TDA operates in a strictly black-box setting and does not require interaction with the target forecasting model.
3. The proposed attack is efficient, showing faster runtime compared with query-based baselines.

### Weaknesses
1. In Equation (4), the paper does not clearly define how token differences are computed. If the attack directly compares token ID values, larger ID gaps may not correspond to larger semantic differences, which undermines the validity of the attack's objective.
2. Commercial forecasting services may switch or ensemble LLMs with different tokenizers based on the user's tier and requst property. Since the threat model assumes access to the target tokenizer, it is unclear how TDA performs when the true tokenizer is unknown or mixed. A small experiment on tokenizer mismatch or multi-LLM settings would strengthen the paper.
3. It is unclear whether TDA’s optimization could produce overly long perturbed inputs or concentrate changes on specific numbers in the sequence, which would reduce its stealthiness. The paper does not clearly indicate whether the optimization includes any penalty or regularization term to prevent such behavior.
4. In Algorithm 1, the choice of 30% in lines 8 and 10 is not justified. It appears to be an empirical setting rather than a theoretically grounded choice.
5. Since the optimization objective focuses solely on maximizing token distance differences, there is no theoretical guarantee that this objective would lead to larger forecasting errors. In fact, several cases in Table 1 show that TDA only causes limited error increases.
6. The real-world feasibility of the attack remains unclear. The authors should specify concrete scenarios in which an attacker could realistically modify and optimize forecasting inputs.

### Questions
Please check my questions in the Weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
3
