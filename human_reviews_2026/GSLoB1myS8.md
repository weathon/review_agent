# EAT: Expert Account Tracker for Efficient MoE Inference

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Mixture-of-Experts (MoE) models have emerged as a revolutionary method to scale Transformer models. However, traditional MoE architecture still suffers from inefficiency since a large number of experts are unnecessarily activated. Existing approaches for reducing the number of activated experts often overlook the historical performance of each expert. In this paper, we propose EAT, a novel method called $\textbf{Expert Account Tracker (EAT)}$, which utilizes history-awareness metrics and adaptive thresholding to dynamically select the most important experts, thereby reducing the activated expert number while effectively maintaining the model performance. Experiments show that EAT outperforms the existing baseline Top-P method across multiple models and datasets, achieving over 25% an average reduction compared to the vanilla method in the number of activated experts and performing better token generation speed compared to the baseline. Additionally, through ablation studies, we find that excessively reducing the number of activated experts can significantly harm model performance, and the importance of experts varies across layers, with higher-level experts being generally more critical.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes EAT (Expert Account Tracker), a novel dynamic expert activation method for Mixture-of-Experts (MoE) models. Unlike traditional routing strategies that rely solely on current gating probabilities, EAT incorporates history-aware metrics to evaluate experts’ long-term reliability. It computes a comprehensive importance score combining each expert’s historical activation frequency, cumulative weight, and contribution magnitude, and integrates this with an adaptive thresholding mechanism to dynamically select experts during inference. The method aims to reduce redundant activations while maintaining model quality. Extensive experiments on Mixtral-8x7B and Phi-3.5-MoE-instruct across diverse benchmarks demonstrate that EAT achieves around 25% fewer activated experts and faster token generation speed compared to the baseline.

### Strengths
The paper introduces a history-aware expert selection mechanism that tracks experts’ past performance using a composite score (activation frequency, cumulative gate weight, and contribution score). I think this consideration is quite comprehensive.


This effectively addresses a limitation in prior dynamic routing approaches (e.g., Top-P) that rely solely on current routing probabilities

### Weaknesses
The method combines historical information and performs adaptive thresholding. Why is this necessary? Is it solely for improving performance? The paper lacks detailed analysis and motivation for these design choices. 

Comparisons are limited to Top-P; more modern baselines (e.g., Ada-K, Expert Pruning and Skipping, CMoE) are mentioned but not empirically compared.

The paper’s exposition is occasionally dense, especially in the method section; a clearer pseudocode or algorithm box would help reproducibility.

### Questions
I find the proposed method rather complicated. Could this complexity affect the inference efficiency in practice?

How does the paper achieve “better token generation speed”? Theoretically, which step or operation ensures this improvement?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Expert Account Tracker (EAT), a history-aware expert routing strategy for Mixture-of-Experts (MoE) models.
EAT maintains a long-term importance score for each expert, computed as a weighted combination of historical activation statistics and the current router-assigned score.
This history-based routing stabilizes expert utilization and accelerates inference by reducing redundant expert activations.
Experiments on Mixtral-8×7B and Phi-3.5-MoE reportedly show slightly improved accuracy and higher token throughput compared to Top-P routing.

### Strengths
1. The proposed method is simple and compatible with existing architectures.
2. The concept of combining short-term gating with long-term statistics is intuitive and interpretable.

### Weaknesses
1. Insufficient validation of inference acceleration:  Despite claiming acceleration, only tokens/sec are measured. Latency (TTFT, TPOT), throughput, and memory consumption are not reported.
2. Unclear experimental setup: Details about the inference environment, implementation, KV-cache, and batching are missing.
3. Lack of references: Key claims in the Introduction, such as performance degradation from high sparsity, are unsupported by citations.
4. Limited experimental scope: Only outdated MoE models are used; no evaluation on recent architectures or reasoning/coding benchmarks.
5. Results against vanilla baseline: EAT often loses to vanilla routing in generation speed and accuracy.

### Questions
1. What exact inference framework, batch size, and KV-cache configuration were used for measuring tokens/sec?
2. Why were only Mixtral and Phi-3.5 chosen? How does EAT perform on more recent MoE models (e.g., Qwen3, DeepSeek)?
3. Why are well-known math and code reasoning benchmarks such as MATH-500, GSM8K, HumanEval+, and LiveCodeBench missing?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes EAT (Expert Account Tracker) to utilizes history-awareness metrics and adaptive thresholding to dynamically select the most important experts, aiming to reduce the activated expert number.

### Strengths
- Easy to follow. The figures and tables are clear.

- Useful Ablation Study: The analysis in Section 4.3 provides a key insight that experts are not equally important across layers.

### Weaknesses
- Heavy Hyper-parameter Issue. This method introduces a large number of new hyper-parameters.

- Lack of performance on larger MoE LLMs, such as Qwen3-30B-A3B.

- Poor performance. As shown in Table 1, proposed EAT underperform Vanilla strategy. Such as 69.00 vs. 75.17 for Phi model on HellaSwag.

### Questions
- ZERO reference in the Introduction section. More references would help reader to understand better.

- The caption of Table should above the tabular, following the official guidelines.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the problem that the history of expert selection is not considered in the expert routing of MoE models. The authors propose an additional importance metric based on a linear combination of three indicators (activation count, total probability, contribution score). By mixing this value with the expert probability calculated from the current token, they aim to achieve routing based on both the current situation and historical data. Furthermore, to adaptively change the number of activated experts, they introduce an adjustment mechanism based on the shape of the expert probability distribution and the recent perplexity.

According to experiments where the routing strategy of existing MoE models was replaced with the proposed method, it was found that while the performance does not match the original (vanilla) performance, it reduces the number of activated experts while maintaining better performance than the top-P method, which is a similar adaptive expert selection technique. The analysis investigates the relationship between the number of activated experts and performance, revealing that excessively reducing the number of experts significantly degrades performance in terms of perplexity, and that layers closer to the input are more amenable to expert reduction than layers closer to the output.

### Strengths
The proposed method contains no components that require training and can be easily applied to any MoE model with a general architecture.

It is an almost rule-based adaptation method, and its computational cost is smaller than the main computation of the MoE parts, making it practically negligible.

### Weaknesses
For most tasks, there is a non-negligible performance gap compared to the vanilla MoE. Although the overall computational cost of the model is reduced, a trade-off between cost and performance must be considered.

The parameters (indicators) added by the proposed method need to be cached within the inference engine, similar to a KV-cache, which may require method-specific implementation support.

### Questions
Why did you choose these three indicators: activation count, total probability, and contribution score? Were any other indicators considered? Is there any important information that these indicators fail to capture?

Around L.269: There appear to be typos in the $\tau_{\mathrm{PPL}}$ definitions—$\lambda_+$ and $\lambda_-$ are possibly inverted.

Are the indicators reset for each test example? If not, dependencies between test examples could arise, making it impossible to interpret the results as independent. Conversely, what would happen if a "burn-in" process for the indicators was inserted before processing the test example?

### Soundness
3

### Presentation
2

### Contribution
2
