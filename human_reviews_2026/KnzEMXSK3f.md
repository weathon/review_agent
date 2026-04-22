# Regression Language Models for Code

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
We study **code-to-metric regression**: predicting numeric outcomes of code executions, a challenging task due to the open-ended nature of programming languages. While prior methods have resorted to heavy and domain-specific feature engineering, we show that a single unified Regression Language Model (RLM) can simultaneously predict directly from text, (i) the memory footprint of code across multiple high-level languages such as Python and C++, (ii) the latency of Triton GPU kernels, and (iii) the accuracy and speed of trained neural networks represented in ONNX. In particular, a relatively small 300M parameter RLM initialized from T5Gemma, obtains $>$0.9 Spearman-rank on competitive programming submissions from APPS, and a single unified model achieves $>$0.5 average Spearman-rank across 17 separate languages from CodeNet. Furthermore, the RLM can obtain the highest average Kendall-Tau of 0.46 on five classic NAS design spaces previously dominated by graph neural networks, and simultaneously predict architecture latencies on numerous hardware platforms.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes Regression Language Models (RLMs) that treat code-to-metric prediction as text-to-text regression with an encoder–decoder LM (initialized from T5Gemma). The same model predicts (i) peak memory for Python/C/C++ programs (APPS, CodeNet), (ii) Triton kernel latency, and (iii) NAS metrics (accuracy and multi-hardware latency) from ONNX text dumps, using a custom numeric tokenizer (P10) and constrained decoding. On APPS memory the model achieves ρ≈0.93, on CodeNet multiple languages ρ≈0.35–0.75, Triton latency ρ≈0.52, and on NAS Kendall-τ it slightly edges a strong FLAN baseline on average. The paper also shows multi-objective decoding (accuracy→latency chains), and ablations on pretraining, tokenization, head choice (decoder vs. regression head), and context length.

### Strengths
One small-ish (~300M) RLM covers diverse inputs (source code, ONNX graphs) and outputs (accuracy, memory, latency), reducing feature engineering and task-specific heads.

Strong APPS memory (ρ≈0.93), respectable CodeNet across 24 languages, non-trivial Triton latency, and average Kendall-τ on five NAS spaces matching/exceeding FLAN without zero-cost proxies.

### Weaknesses
Hardware coverage is narrow for Triton (single A6000); latency predictors can be highly hardware-sensitive. Broader devices or cross-device generalization would strengthen claims.

CodeNet regime mixes train/test questions (few-shot per problem). Authors acknowledge this limits zero-shot difficulty; results may over-estimate real-world generalization to new problems with unseen inputs.

### Questions
How many samples per input are used at inference, and how are means/medians chosen? Did you evaluate calibration or UQ quality (CRPS/NLL) when using the density view? 

The ONNX graphs can be long; what fraction of graphs truncate at 1k/2k/4k tokens, and how does that affect τ/ρ?

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
This paper introduces Regression Language Models that frame code-to-metric regression as a text-to-text generation task. Instead of relying on domain-specific feature engineering, the authors propose using a unified encoder-decoder model that reads the raw text of code or intermediate representations and autoregressively decodes the numerical metric using a special P10 tokenizer.

### Strengths
1. The paper's core idea, i.e., treating all code regression as a text-generation problem, is original and simplifies a process that traditionally requires heavy, domain-specific feature engineering.
2. A key strength is the demonstration of a unified model successfully handling regression tasks across highly disparate inputs and predicting different metrics.
3. The methods outperform or remain competitive with baselines.

### Weaknesses
1. The paper's central choice to formulate regression as an autoregressive token-generation problem (using P10) instead of using a standard regression head is not convincingly justified. Have you tried using a log transform for the output within the MLP? It's also not clear if the conditional (auto-regressive) modeling provides any real benefit over a simpler multi-head approach that predicts all metrics independently from the encoder embedding.
2. The paper motivates its approach by contrasting it with feature-engineering methods. However, it only compares against deep learning feature engineering. For tasks like memory and latency prediction, the true established baselines are often traditional static analysis tools or logic-based methods from the programming languages and compiler communities. The paper provides no comparison to these methods.
3. The study relies exclusively on T5Gemma. This is not a common base model in modern LLM research (compared to Llama, Qwen, etc.).
- The authors should include results based on more standard base models.
- The paper should also report the performance of the base T5Gemma model without RLM fine-tuning to properly isolate the gains from the proposed training method.
4. Since the task is reformulated as text-to-text generation, the most obvious baselines are missing: the performance of modern, general-purpose LLMs. How well do state-of-the-art models (e.g., GPT, Gemini, DeepSeek) perform on this task under different settings? This comparison is essential to understand if the specialized RLM is necessary or if the capability is already present in general-purpose models.

### Questions
See Weaknesses.

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
2

### Summary
The paper proposes Regression Language Models for code. It introduces encoder-decoder language models based on T5Gemma, fine-tuned to perform numeric regression on code-related tasks. They have discussed predicting memory footprint, latency, or model accuracy from textual code or ONNX representations. The authors claim that a single unified model can regress across diverse programming languages and the created NAS spaces, outperforming specialized graph-based regression models (e.g., GNNs, FLAN).

### Strengths
1. It is a very interesting direction for code applications: predicting performance metrics directly from code.
2. The idea of using a unified single model across code, kernel, and graph inputs is conceptually appealing.
3. The presentation of the paper is organized with supportive tables and illustrative examples.

### Weaknesses
1. Novelty: I can hardly agree that the major concept is new. It uses text-to-text regression and autoregressive regression with trivial modifications.
2. Missing details of the experiments.
a. The multi-task regression setup mixes heterogeneous tasks without normalization or loss balancing.
b. No ablation on negative transfer or cross-domain interference.
The results thus cannot support the central claim that “a unified RLM generalizes across tasks.”
3. Evaluation
a. The >0.9 Spearman on APPS is due to the dataset’s extremely small label variance.
b. The gains in table 4 are statistically insignificant.
4. In general, the authors are using inflated language such as “massively simplifying graph regression” and "general-purpose universal regressor,” with no mechanistic or theoretical support.

### Questions
1. Appendix A.3 shows that fine-tuning reduces performance due to catastrophic forgetting, and A.2 shows that multilingual pretraining hurts zero-shot languages. Do they directly contradict the paper’s claims of generality?
2. How does the model handle tasks with vastly different output scales (e.g., bytes vs milliseconds vs accuracy)?
3.Why compare to FLAN without describing its training details?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces a unified framework called Regression Language Models (RLM) for "code-to-metric" regression. The core idea is to predict various numerical metrics directly from source code and other textual representations of programs, eliminating the need for manual feature engineering.

### Strengths
- Elimination of Feature Engineering: A significant advantage of the RLM is its ability to learn directly from raw text representations of code. This simplifies the prediction pipeline and makes it more adaptable to new programming languages, hardware, or model architectures.
- Strong Empirical Results: The RLM achieves impressive performance, with a Spearman's rank correlation greater than 0.9 on predicting the memory usage of competitive programming submissions. It also performs well on predicting Triton kernel latency and achieves state-of-the-art results in Neural Architecture Search benchmarks, outperforming graph-based methods.[1][2][3][4]
Multi-Objective Prediction: The autoregressive nature of the decoder allows for the conditional prediction of multiple metrics. This is a powerful feature for tasks like hardware co-design and compiler optimization, where trade-offs between different performance aspects need to be considered.
- Effective Use of Pretraining: The work demonstrates the significant benefit of initializing the RLM from a pretrained language model (T5Gemma), which leads to faster convergence and better overall performance.

### Weaknesses
- Limited Exploration of Larger Models: While the paper shows promising results with a 300M and a 600M parameter model, a more extensive analysis of how performance scales with model size would be beneficial. A marjor concern is that: modern SOTA models excel at generalizing from just a handful of examples provided in a prompt, without any weight updates. This is a far more flexible and cost-effective approach than fine-tuning. A frontier model like a hypothetical GPT-5 or Claude-4.5 could potentially become a powerful code-to-metric regressor on the fly, adapted to novel metrics or hardware simply by being shown a few examples. The paper misses the opportunity to investigate whether its proposed task can be solved "in-context," which is a crucial question for practical applicability.

- Dependence on Textual Representation: The model's performance is contingent on the quality and completeness of the textual representation of the code or computation graph. For complex structures, serializing them into a linear text format might lead to a loss of information.

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
2
