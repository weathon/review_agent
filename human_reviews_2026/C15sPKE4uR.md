# Deconstructing Self-Bias in LLM-generated translation benchmark

- Decision: Reject
- Scores: 10, 6, 2, 4

## Abstract
As large language models (LLMs) begin to saturate existing benchmarks, automated benchmark creation using LLMs (LLM-as-a-benchmark) has emerged as a scalable alternative to slow and costly human curation. While these generated test sets have to potential to cheaply rank models, we demonstrate a critical flaw. LLM-generated benchmarks systematically favor the model that created the benchmark: they exhibit self-bias on low resource languages to English translation tasks. We show three key findings on automatic benchmarking of LLMs for translation: First, this bias originates from two sources: the generated test data (LLM-as-a-testset) and the evaluation method (LLM-as-an-evaluator), with their combination amplifying the effect. Second, self-bias in LLM-as-a-benchmark is heavily influenced by the model’s generation capabilities in the source language. For instance, we observe more pronounced bias in into-English translation, where the model’s generation system is developed, than in out-of-English translation tasks. Third, we observe that low diversity in source text is one attribution to self-bias. Our results suggest that improving the diversity of these generated source texts can mitigate some of the observed self-bias.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The paper studies the self-bias in LLM generated benchmarks. LLM generated benchmarks favor the LLM that generated the benchmark dataset. The paper finds strong self-bias in low resource language to English translation. This self-bias comes from two sources, LLM's generation capability in the source language and using LLM as an evaluator. The authors then demonstrate that this self-bias can be attributed to LLM's limited generation capability in the source language.

### Strengths
1. The experimental design is sound. It cleanly separates out the two components of self-bias in LLM-as-a-benchmark: benchmark generation and evaluation.
2. The authors perform experiments to determine whether LLM self-bias can be explained by the LLM's preference to generate source texts that are easily translatable by the LLM. Their results of source-only vs source+reference generation conclusively show that this hypothesis is true.
3. The authors also show that even for source-only generation the LLM has low diversity, which leads to the self-bias persisting even in this scenario.
4. Finally the authors also conduct experiment for out of English direction showing lower self-bias. Thus the experiments are comprehensive.

### Weaknesses
1. The self bias measurement is based on ranking. This measure might be sensitive to the number of LLMs under consideration (just three in this paper). A larger set of LLMs (including open source multilinguality focused LLMs) may strengthen the results.

### Questions
Why were the four low-resource languages chosen? Why not other languages?

### Soundness
4

### Presentation
4

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
This submission addresses the self-bias issue in the LLM-as-a-benchmark evaluation, where an LLM can overrate itself when using it as the evaluator or towards the benchmarks that it creates. This submission formally defines and decomposes self-bias into two components: bias from testset generation and bias from evaluation. Focusing on low-resource machine translation, this submission conducts experiments with Gemini-2.5-Pro, GPT-4.1, and Claude-Opus-4 across six translation directions. The results show that self-bias exists in both components and is strongest when combined. This submission further analyses the biases under different aspects, presenting a study that covers multiple points toward the self-bias in LLMs.

### Strengths
1. This submission addresses an important issue in LLM evaluation and is generally well written.
2. It provides a formal definition and decomposition of self-bias into testset and evaluator components, which improves the clarity over prior discussions.
3. The experiments cover three major LLMs across six translation directions with multiple controlled conditions.
4. The analysis is interesting and can be insightful for future work.

### Weaknesses
1. All experiments are limited to MT, leaving unclear how similar self-bias mechanisms apply to other generative or reasoning tasks.
2.  Each language direction includes only 200 instances, which limits statistical robustness and the strength of causal claims. Also, there is no significance analysis in the results.
3. The evaluated systems are all closed-source (Gemini, GPT-4.1, Claude), where open LLMs should be considered.

### Questions
Please see above.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper shows that using LLMs to generate training data (even just inputs) or to act as a judge, biases any evaluation that also uses the same LLM as a participating system. The paper is well written but overall this is such an obvious problem that it does not really require such detailed investigation. There are not really any new insights presented.

### Strengths
The experiments are well designed and executed. 

The paper is well written.

The analysis of lack of variety of generated sources is interesting.

### Weaknesses
The overall point that LLMs are biases when used as both data generator/judge and participant is a very obvious problem. Anything a model produces will lie well within its distribution, and therefore it is likely to cover it well. 

The particular setup to use LLMs to generate _source_ sentences for translation is a bit - there is typically not a lack of monolingual sentences to act as source, the problem is the lack of accurate reference translations. Of course, generating reference translations with an LLM and then treating them as gold standard is even more problematic (and it is the reason why at WMT shared tasks the professional translators are prohibited from using any machine translation tool).

### Questions
n/a

### Soundness
3

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
3

### Summary
This paper studies self-bias in LLM-generated translation benchmarks. It first decomposes the concept of LLM-as-a-benchmark into two components: LLM-as-a-testset and LLM-as-an-evaluator. The observed bias arises from the interaction between these two roles and is more pronounced in low-resource-to-English translation tasks. The paper further demonstrates that limited diversity in the source texts contributes to self-bias, while increasing source-text diversity can mitigate it. Despite this limitation, LLM-as-a-benchmark remains a valuable approach for ranking weaker models and for evaluating translations.

### Strengths
1. The paper conceptualizes LLM-as-a-benchmark as consisting of two distinct components, LLM-as-a-testset and LLM-as-an-evaluator, and provides dedicated analyses benefiting from this decomposition.
2. LLM-as-a-benchmark remains a valuable approach for ranking weaker models and for evaluating translations.

### Weaknesses
The testset generation process in the LLM-as-a-benchmark framework is not sufficiently well designed. A well-constructed benchmark should incorporate both diversity control and quality control [1] [2] [3]. The conclusion that “generating more diverse source texts can mitigate self-bias” arises precisely because the LLM-as-a-benchmark approach adopted in the paper lacks diversity control, which leads to unnecessary self-bias. Similarly, the degeneration issue observed in the generated source texts (see Figure 1) results from the absence of quality control. Consequently, the self-bias observed in a simply generated testset is analyzed within an overly narrow and unrealistic context, which undermines the persuasiveness of the argument. Therefore, it is helpful for testset generation to incorporate appropriate diversity control and quality control, so that analyses of self-bias conducted on such datasets can be sufficiently convincing.

References

[1] https://lmsys.org/blog/2024-04-19-arena-hard

[2] Li, Tianle, et al. "From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline." arXiv preprint arXiv:2406.11939 (2024).

[3] Lin, Bill Yuchen, et al. "WildBench: Benchmarking LLMs with Challenging Tasks from Real Users in the Wild." The Thirteenth International Conference on Learning Representations.

### Questions
1. Since self-bias is sensitive to the diversity and quality of the benchmark, it is helpful that the diversity and quality of the benchmark are presented in detail.
2. Moreover, studying self-bias across benchmarks with varying levels of diversity and quality provides more informative insights than studying self-bias on a simply generated benchmark.

### Soundness
2

### Presentation
3

### Contribution
2
