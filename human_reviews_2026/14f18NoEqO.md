# KnowProxy: Adapting Large Language Models by Knowledge-guided Proxy

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
Adapting large language models (LLMs) using smaller proxy models has been shown to improve training efficiency, where the LLMs remain frozen while the proxies are tuned on top. However, this approach typically requires access to the output probability distributions of LLMs, which are often inaccessible or unstable. To address this limitation, we propose KnowProxy, a knowledge-guided proxy framework in which the proxy is trained with textual knowledge rather than probability distributions. Specifically, we first elicit textual knowledge and reasoning from frozen LLMs through prompting, and then the proxy model learns to adapt this reasoning to target task distributions. We evaluate KnowProxy on diverse reasoning benchmarks with different fine-tuning scenarios. Comprehensive results show that KnowProxy achieves competitive or even better performance without direct access to probability distributions, thereby providing a scalable and versatile alternative to traditional fine-tuning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents KnowProxy, a framework for adapting Large Language Models (LLMs) to effectively incorporate structured knowledge (e.g., from graphs or databases) without fine-tuning the LLM itself. Instead of directly injecting structured information into the LLM—which can lead to instability and catastrophic forgetting—KnowProxy introduces a proxy network that mediates between the LLM and the structured knowledge source. The proxy learns how to represent knowledge entries in a way that aligns with the LLM’s latent space, enabling efficient and parameter-light integration.

### Strengths
1. The idea of introducing a proxy representation as an intermediary between LLMs and structured knowledge is both novel and well-motivated. It offers a pragmatic alternative to parameter-heavy fine-tuning and complex RAG pipelines.
2. The model design is clearly described, with a modular architecture that separates the proxy learner, retriever, and aggregator components. The training objectives (semantic and factual alignment) are theoretically sound and intuitively justified.
3. Achieves strong results while training only a small proxy network, keeping the main LLM frozen. The framework is compatible with various LLMs (e.g., LLaMA, Falcon, Mistral), highlighting its plug-and-play nature.

### Weaknesses
1. While empirically effective, the paper lacks a formal analysis explaining why the proxy layer aligns so well with the LLM’s latent representations. More theoretical grounding would strengthen the argument.
2. The learned proxy embeddings are treated as black boxes. The paper could discuss whether they encode semantic meaning or merely serve as latent alignment vectors.
3. The training objective combines multiple losses (semantic, factual, regularization) with limited ablation on their individual contributions.

### Questions
See weakness

### Soundness
3

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
This paper introduces KNOWPROXY, a novel framework for efficiently adapting  LLM in scenarios where access to their internal parameters or full probability distributions is unavailable. Diverging from existing proxy-based methods that rely on probability distributions, KNOWPROXY's core idea is to first elicit textual knowledge and reasoning chains from a frozen LLM via prompting. This elicited knowledge is then used to train a smaller proxy model, which learns to map the original input query combined with the LLM-generated knowledge to the final task output. Furthermore, the authors incorporate a dynamic routing mechanism based on confidence scores, which adaptively decides whether to invoke the proxy model at inference time. This balances performance and efficiency.

### Strengths
1. The overall framework of KNOWPROXY is well-designed. The pipeline, from knowledge elicitation and proxy optimization to dynamic routing, is logical and coherent. By combining the confidence scores of both the prediction and the generated knowledge to assess the reliability of the LLM's output, the dynamic routing mechanism effectively strikes a balance between performance and inference cost.
2. The experimental validation in this paper is outstanding. The experiments span various LLMs, multiple sizes of proxy models, and a diverse set of complex reasoning datasets. The paper includes thorough ablation studies that validate the necessity of key components.

### Weaknesses
1. While being the cornerstone of the method, the discussion on the knowledge elicitation process is somewhat brief. The performance of KNOWPROXY heavily depends on the quality of the knowledge elicited from the LLM, which in turn is highly sensitive to prompt design. The authors should provide more discussion in the main text regarding the importance of prompt engineering and its potential impact on knowledge quality. For example, how might different prompt styles affect the final performance?
2. The entire dynamic routing mechanism hinges on the elicited confidence scores from the LLM. Although experiments (Figure 2) show that the proposed aggregated confidence is superior to the baseline, it is a well-known issue that LLMs tend to be overconfident.

### Questions
1. The final confidence score C_final is calculated by multiplying several confidence scores. This implicitly assumes conditional independence between the knowledge pieces and the final prediction. Is this assumption justified? Have you experimented with alternative aggregation methods, such as a weighted sum or a small learned network to aggregate these scores?
2. In the knowledge filtering step, how was the threshold alpha selected? How sensitive is the method's performance to this hyperparameter?
3. Table 5 demonstrates the transferability of the proxy model, which is a very interesting finding. Does this suggest that the proxy model learns a general, LLM-agnostic ability to "reason using textual knowledge"? Do you have any deeper insights into this phenomenon?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposed a novel proxy model tuning method without the requirements of the output distributions from LLMs. Specifically, let LLMs output the knowledge or cues, and the proxy model uses these cues along with the query to get the correct answer. Furthermore, it uses the confidence from LLMs to determine which to use the proxy model. Experiments have demonstrated the effectiveness.

### Strengths
1. The proposed method has good applicability and can be applied to black box models.

2. From the experimental results, it can be seen that LLMs can effectively improve their performance in knowledge reasoning tasks.

3. The paper writing and experimental setup are relatively complete.

### Weaknesses
1. Suggest using newer models, such as Qwen3. Using a more powerful model may result in inconsistent performance in Figure 3, where a larger amount of knowledge can further improve accuracy.

2. Typoes, e.g., Table 4.6 in line 455. Some values should be further claimed, e.g., 38.7% in line 467.

3. Suggest more tasks, such as mathematical coding, to verify the universality of methods.

### Questions
Additional questions:

1. From Table 4, it can be observed that without routing performance, there will be a decrease, which suggests that the proxy model may have answered some questions incorrectly, while the LLM responded correctly. Can the original answer of the LLM model also be input into the proxy model?

### Soundness
3

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
Direct fine-tuning of LLMs is computationally expensive and impossible for proprietary black-box models. Existing proxy methods require access to LLMs’ probability distributions and suffer from unstable distributions, limiting their applicability. This paper proposes KNOWPROXY, a knowledge-guided proxy framework for adapting Large Language Models (LLMs) without relying on their output probability distributions—addressing key limitations of existing proxy-based LLM adaptation methods.

* KNOWPROXY outperforms existing proxy methods across all benchmarks and matches direct LLM fine-tuning on tasks like OpenBookQA and StrategyQA.
* It works for black-box LLMs (e.g., ChatGPT) and quantized models (e.g., 4-bit Llama-2-13B), with performance scaling with SLM capability.
* Ablation studies confirm filtering and knowledge adaptation are critical (removing adaptation causes the largest performance drop), while dynamic routing reduces inference cost without accuracy loss.
* Cross-LLM transferability: Proxies trained on one LLM (e.g., Llama 3.2) maintain performance when paired with another (e.g., ChatGPT).

### Strengths
1.	By replacing LLM probability distributions with textual knowledge, KNOWPROXY enables adaptation for black-box LLMs (only text outputs available) and avoids instability from unreliable distributions—filling a key gap in existing proxy methods.
2.	The dynamic routing mechanism balances accuracy and inference cost by invoking the proxy only when needed, addressing the "always-on proxy" inefficiency of prior work.
3.	Works with diverse LLMs (open-source/black-box/quantized) and SLMs, with cross-LLM proxy transferability (no retraining needed for new LLMs).

### Weaknesses
1.	Performance relies heavily on the quality of prompted knowledge-poorly designed prompts or LLM hallucinations (even after filtering) could degrade proxy training. The paper does not explore how prompt variations (beyond task-specific templates) impact results.
2.	The threshold $/alpha$ for knowledge filtering is predefined but not justified (e.g., no sensitivity analysis on how $/alpha$ affects performance across tasks).
3.	The threshold $/tau$ for routing is also predefined, with no discussion of how to optimize $/tau$ for different tasks/LLMs (e.g., whether τ should be task-specific).

### Questions
The same as the weakness.

### Soundness
3

### Presentation
3

### Contribution
3
