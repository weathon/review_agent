# Attribution-Guided Decoding

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
The capacity of Large Language Models (LLMs) to follow complex instructions and generate factually accurate text is critical for their real-world application. However, standard decoding methods often fail to robustly satisfy these requirements, while existing control techniques frequently degrade general output quality.
In this work, we introduce Attribution-Guided Decoding (AGD), an interpretability-based decoding strategy. 
Instead of directly manipulating model activations, AGD considers a set of high-probability output token candidates and selects the one that exhibits the highest attribution
to a user-defined Region of Interest (ROI). This ROI can be flexibly defined over different parts of the model's input or internal components, allowing AGD to steer generation towards various desirable behaviors.
We demonstrate AGD's efficacy across three challenging domains.
For instruction following, we show that AGD significantly boosts adherence (e.g., improving the overall success rate on Llama 3.1 from 66.0\% to 79.1\%).
For knowledge-intensive tasks, we show that guiding generation towards usage of internal knowledge components or contextual sources can reduce hallucinations and improve factual accuracy in both closed-book and open-book settings.
Furthermore, we propose an adaptive, entropy-based variant of AGD that mitigates quality degradation and reduces computational overhead by applying guidance only when the model is uncertain.
Our work presents a versatile, more interpretable, and effective method for enhancing the reliability of modern LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work propose a method to reduce hallucinations and improve factual accuracy. In order to reduce computational cost, an entropy-based adaptive mechanism is applied. In the experiments, higher factuality is obtained on two benchmarks (IHEval and SysBench).

### Strengths
1. integration of attribution into decoding: introducing attribution-guided decoding (AGD) that dynamically adjusts token-level generation probabilities based on feature attributions. 

2. Experiments span three representative long-form tasks (fact-grounded QA, long summarization, and reasoning). The authors evaluate across multiple model families (Llama-3, Qwen2.5, Gemma2) and both open- and closed-source settings.

3. Interpretability and visualization:
The paper presents clear visualizations (Figures 3–5) of token attributions and their decoding effects. These figures concretely show that AGD improves focus on factually relevant spans and reduces hallucinated continuations

### Weaknesses
1. Limited novelty in algorithmic mechanism: Although the integration of attribution and decoding is original, the underlying technique (re-weighting token probabilities) is relatively straightforward and derivative of prior constrained or guided decoding approaches (e.g., likelihood re-weighting, constrained beam search). Following are some related work on this direction

GeDi: Generative discriminator guided sequence generation. EMNLP 2021
Unlocking Anticipatory Text Generation: A Constrained Approach for Large Language Models Decoding. EMNLP 2024

2. Evaluation lacks human verification.
Improvements in factuality metrics are fully automatic. Human or expert evaluation of factual correctness and coherence would make the results more convincing, especially for interpretability claims.

### Questions
1. Human interpretability:
Have you tested whether human evaluators actually find the attributions helpful for understanding the model’s reasoning? Qualitative examples could strengthen this claim.

2. hyperparameters setting
There are two sensitive hyperparameters: entropy-gated threshold and the minimum probability threshold.  It looks depends on models and tasks. Did you have any related experience?

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
3

### Summary
- **Idea:** At each step, form a small candidate set (top-k with prob floor), compute attribution of each candidate to a user-defined Region of Interest (ROI) (e.g., instruction tokens, context embeddings, selected attention heads), and pick the token with the highest ROI attribution. No logits or activations are modified; it’s selection rather than intervention.  
- **Variants:** Use LRP/AttnLRP (primary) or Input×Gradient for attribution; propose entropy gating to apply guidance only when the model is uncertain to curb quality loss and cost.  
- **Results:** Improves instruction adherence on IHEval/SysBench and boosts factuality in closed-book (via “knowledge heads”) and open-book (via context embeddings or “in-context heads”); entropy gating trades off adherence vs quality and reduces backward passes.

### Strengths
- **Simple, model-agnostic control:** Decoding-time, fine-tuning-free; clear ROI abstraction spanning inputs or internals.  
- **Practical quality/cost knob:** Entropy gating reduces intervention and keeps fluency while retaining most adherence gains.  
- **Coverage:** Evaluated on instruction following (IHEval, SysBench) and factual QA (closed/open-book) across three small/mid models.

### Weaknesses
- **Positioning vs RAG/contrastive baselines is incomplete:** Open-book uses an ROI over context/heads, but no retrieval step; comparisons are to greedy/nucleus/CAD, not to RAG-style pipelines or “ROI-prompt only” ablations. This blurs how much gain comes from attribution versus simply highlighting the ROI.  
- **Compute overhead and scalability:** AGD requires backward passes per candidate token when triggered; the paper reports entropy-gated reductions and average backward-pass counts, but lacks end-to-end wall-clock or token-cost comparisons against strong external baselines. Memory implications are not quantified.  
- **Generalisation to stronger LLMs is unclear:** Results are on Llama-3.1-8B, Qwen-2.5-7B, Gemma-3-4B; it’s unknown whether gains persist or shrink on frontier small models (e.g., Qwen3, DeepSeek-R1-8B).

### Questions
1. **RAG vs AGD:** In open-book, could you compare against a standard RAG baseline (retrieve-then-decode) and a “ROI-prompt only” baseline that emphasises the ROI without attribution? This isolates attribution’s unique contribution.  
2. **Cost & memory:** Please report wall-clock, token cost, and peak memory with and without entropy gating, and versus CAD/RAG on the same hardware. Also provide the average number of backward passes per token under typical thresholds.  
3. **Stronger models:** Do gains hold for newer small models (e.g., Qwen3 or DeepSeek-R1-Qwen3-8B)? Any evidence of diminishing returns as base model adherence improves?  
4. **Head selection robustness:** For “knowledge” or “in-context” heads, how stable are results across identification methods and datasets? Could you provide ablation of head sets and performance sensitivity?

### Soundness
2

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
3

### Summary
This paper presents Attribution-Guided Decoding (AGD), a fine-tuning-free decoding strategy to enhance the reliability of LLMs. AGD is based on the selection of tokens from a set of high-probability candidates. Selection is based on the maximization of the attribution score of a candidate token to a user-defined "Region of Interest" (ROI). This ROI can be flexibly defined over parts of the input prompt or over internal model components. The authors show the effectiveness of AGD in improving instruction following, increasing factual accuracy in closed-book question answering by avoiding hallucinations, and enhancing grounding in open-book (in-context) retrieval tasks. Regarding the computational overhead of calculating attributions at each step, an adaptive variant is also proposed, entropy-gated AGD-e, which applies guidance only when the model output distribution shows high uncertainty, thus preserving the quality of the output while saving cost.

### Strengths
1. AGD repurposes passive attribution methods for active generation. Unlike "interventionist" methods that alter activations, this "selectionist" approach chooses among the model's high-probability outputs, preserving text quality while making generation interpretable through token-rationale relationships.

2. The ROI framework applies broadly to diverse problems. AGD improves instruction adherence (using instruction text as ROI), factual recall (targeting "parametric knowledge heads"), and in-context grounding (using context embeddings as ROI).

3. The authors tested across multiple models (Llama 3.1, Qwen 2.5, Gemma 3) and tasks with strong baselines like CAD and DoLA. The analysis includes hyperparameter ablations and case studies visualizing attribution scores to explain why the method works.

### Weaknesses
1. AGD's "selectionist" design limits it to tokens already proposed by the base model with high probability. If the desired token isn't in the candidate set, AGD cannot generate it, making success dependent on the base model's capabilities. 

2.  The method seems to be computationally expensive, requiring backward passes for each candidate token to compute attribution scores. While entropy-gating helps, this overhead could limit its practice. In the meantime, we need to predefine the threshold for the entropy, which can not always be reliable.

3. While ROI flexibility is a strength, it requires users to define relevant ROIs for each task. This is straightforward for instruction following but non-trivial for tasks like factuality improvement, which requires identifying specialized components like "knowledge heads." 

4. Many baselines are missing such as DoLA.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3
