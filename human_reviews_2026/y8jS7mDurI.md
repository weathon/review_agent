# Proactive Interference Reveals Working Memory Limits in LLMs Beyond Context Length

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Information retrieval in Large Language Models (LLMs) is increasingly recognized as intertwined with generation capabilities rather than mere lookup. While longer contexts are often assumed to improve retrieval, the mechanics of intra-context interference, as instantiated in MRCR test, remain understudied. To address this, we adapt the proactive interference (PI) paradigm from cognitive science, where earlier information disrupts recall of newer updates. In humans, susceptibility to such interference is inversely linked to working memory capacity. We introduce PI-LLM, an evaluation to measure LLM working memory by sequentially streams co-referenced key–value updates, where the same key is sequentially rebound to multiple values, and queries only the final values. Although these final values are clearly positioned just before the query, LLM retrieval accuracy declines log-linearly toward zero as co-referenced interference accumulates; errors arise from retrieving previously overwritten values. Attempts to mitigate interference via prompt engineering (e.g., instructing models to ignore earlier input) yield limited success. These findings reveal a fundamental constraint on LLMs’ ability to disentangle interference and flexibly manipulate binding information, suggesting a working memory bottleneck beyond mere context access.

PI-LLM bridges (i)LLM performance in MRCR tests and (ii) studies of entity binding in LLM mechanistic interpretations. And provides a cognitive-science inspired measurement of LLM working-memory-like capacity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
LLMs are tested with PI-LLM, a setup that streams many co-referenced key–value updates and then asks for the final value. Across diverse models, accuracy falls roughly log-linearly as interference (more updates/keys/longer values) grows—even when total prompt length is held constant—implicating a working-memory-like interference limit rather than context size. Prompting or CoT doesn’t fix it; a simple “reset” hack helps only partially. The paper proposes an Interference Endurance Score (IES) and observes larger dense models resist interference better than smaller/MoE ones.

### Strengths
* Clean, controlled benchmark that isolates proactive interference (not search or sheer length) via co-referenced updates and a fixed-length control.
* Reveals a consistent, interpretable log-linear degradation pattern across many model scales, providing a quantitative handle on interference robustness.
* Demonstrates that current prompting and reasoning strategies are insufficient

### Weaknesses
The main concern is that the setting is extremely limited. The experimental design is clean, but it’s also very artificial. Because the task uses simple key–value updates, it’s hard to know whether the same interference appears in real settings.
* You could test a long article where an entity’s attributes change over time and see if the model retrieves the most recent one.
* A multi-turn chat or agent session, where a small profile field keeps being updated, would also make the setup feel more realistic.
* Showing that models with stronger interference in your test also make more “old value” mistakes in these realistic cases would make the results more convincing.

Most of the interventions the paper tests are just prompt variations, which doesn’t really address the underlying issue. If the problem is interference between parts of the context, there should be at least one experiment that changes how the model or system handles memory.(e.g., KV-cache resets/segmenting, retrieval-augmented state, local attention, active-experts in MoE).

The formatting of the figures could be improved (e.g. in figure 1, the 1*, 2*, 3*, 4* is hard to read).

### Questions
How is the fixed-length control enforced at the token level? The same number of key–value pairs doesn’t guarantee equal token length—please specify the tokenizer, any padding/truncation, and whether the final query length is held constant.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper investigates into a particular factor that impacts LLM performance: limited anti-interference capacity, where the paper shows that LLM performance degrades in a log-linear way as interferences increase. 

In the experiments, the authors show that this increase is disentangled with length increase (Section 3), thus makes it an independent causal factor. The tests have been carried over numerous LLMs and they show similar log-linear decrease trend. Besides, the authors have investigated into ways to mitigate the issue including various prompt changes as well using CoT. The improvement was only minimum and the authors show that the problem persists well.

### Strengths
Understanding the LLM limitation is surely an important topic, particularly if novel insights are brought into the community. The paper identified the anti-inference capacity as an LLM performance limitation; most importantly, the paper has shown that the inference factor is independent of context length factor to impact LLM performance. 

The paper has demonstrated the results over various LLMs to show that it is a general problem for current LLMs; besides, the paper has investigated into different prompt strategy including CoT to mitigate the issue and show that the problem persists.

### Weaknesses
I don't doubt the novelty in this work, nevertheless, I would encourage the authors to include a Related Work to show the connections that the paper has with existing research.

The results demonstrated in the paper, well kind of novel, is not surprising: the LLM performance is not conditioned on the length of the context but also the problem difficulty; for the later case, the community has identified quite early on that the LLM memory can be one of such problem difficulty (in this sense, the current paper is well related to these lines of works). While prompt strategy has been investigated, we also notice that more involved architectures (LLM workflow, or agents) are not investigated in the paper; sure that it is not directly linked with the interference factor that limits the LLM performance, but it is important to show that these problems can/cannot be solved alternatively, for the current paper, it reads like it is not solvable which I believe is not true.

### Questions
None

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
This paper investigates how information retrieval in LLMs is affected by intra-context interference, adapting the proactive interference (PI) paradigm from cognitive science. The authors propose PI-LLM, an evaluation where models must recall the latest value among sequentially updated, co-referenced key–value pairs. Results show a log-linear decline in retrieval accuracy as interference accumulates, with models often recalling outdated values. Prompt-based mitigation fails to resolve the issue, indicating a working memory–like limitation in LLMs. By removing irrelevant context (“haystack”) and directly measuring interference, PI-LLM provides a principled framework to assess and improve LLMs’ information disentanglement and memory control.

### Strengths
1. Clear and well-written: The paper is logically structured and easy to follow, with clear motivation and experimental design.

2. Valuable and insightful finding: The discovery of log-linear interference effects offers deep insights into LLM working memory limits and provides meaningful guidance for future model development.

### Weaknesses
1. No discussion with a very related work (motivation and discovery): https://arxiv.org/abs/2502.05252.  This paper also discusses how to insert noise in the context and find something very similar to log-linear degradation.

2. Do not provide executable insights on how to improve LLM training in the discussed problem.

### Questions
In the weakness.

1. Are models making advancements in the discussed problem? For example, Llama 2, Llama 3, Llama 4/Qwen3/Deepseek —how is the trend? 

2. Does model architecture make a difference? For example, xLSTM and Mamba are compared to transformers.

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
4

### Summary
LLMs are known to struggle when conflicting evidence is present in the context. This has been extensively studied in the past in relation to RAG (retrieval augmented systems). This paper proposes a task to systematically evaluate the same. The context consists of key value pairs, with same key potentially having differing values in the same context. The LLM is supposed to find the value associated with the last occurrence of a key. The work shows that increasing such conflicts leads to a consistent drop in the accuracy.

### Strengths
- THe experiments are thorough. The authors clearly specify the prompts they used to ensure that the model follows the task as desired. 
- The control for various confounders like length. 
- The capacity analysis is nice.

### Weaknesses
- It is a well-documented fact in the RAG literature that intra-context conflicts cause performance degradation (see https://arxiv.org/abs/2507.21544, https://arxiv.org/abs/2504.13079v2). In fact, prior works have shown that when conflicting evidence is present in context, models tend to rely on parametric knowledge biases rather than the retrieved evidence (https://arxiv.org/abs/2402.07867, https://aclanthology.org/2022.emnlp-main.146.pdf).

- Given these studies, it is unclear what the proposed benchmark adds—whether it identifies a genuinely new failure mode or deepens understanding of existing, well-known interference phenomena.

- Relation to real-world scenarios: In realistic settings, conflicting evidence typically co-occurs with confounders such as the confidence of retrieved snippets or the credibility of their sources. These factors often dominate which information an LLM uses in its final response. The proposed benchmark abstracts away such factors, making it difficult to extract actionable insights for practitioners. It largely remains a toy key–value benchmark with limited ecological validity.

- Suggestions: Future versions of the work could incorporate more nuanced and realistic setups—e.g., multi-hop reasoning or information extraction under conflicting evidence—to better connect with real-world retrieval challenges.

- Minor: The plots can be made much more cleaner in the main paper, with only key curves being shown and rest deferred to the appendix. In general, the manuscript needs significant more efforts in better presentation.

### Questions
See the weakness section

### Soundness
2

### Presentation
1

### Contribution
2
