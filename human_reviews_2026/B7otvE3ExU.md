# FusionFactory: Fusing LLM Capabilities with Multi-LLM Log Data

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 4, 2

## Abstract
The rapid advancement of large language models (LLMs) has created a diverse
landscape of models, each excelling at different tasks. This diversity drives re-
searchers to employ multiple LLMs in practice, leaving behind valuable multi-
LLM log data. This naturally leads to the question of whether such logs can be
fully leveraged to fuse LLMs’ complementary capabilities. Although prior work
has explored various strategies for integrating multiple LLMs, we argue that prac-
tical fusion must meet two essential requirements: (1) compatibility with real-
world serving scenarios (e.g., local and API-based serving), and (2) flexibility to
operate at different stages of the LLM pipeline to meet varied user needs (e.g.,
fine-tuning and inference stages). To this end, we introduce LLMFusionBench,
a large-scale benchmark for LLM fusion that spans 14 tasks across five domains,
with responses from 20 open-source LLMs (8B–671B) totaling 103M tokens.
Building on LLMFusionBench, we propose FusionFactory, a systematic
framework with three elaborated levels: (1) query-level fusion via tailored LLM
routers, (2) thought-level fusion leveraging retrieved abstract reasoning tem-
plates, and (3) model-level fusion via distillation from top-ranked responses. Ex-
periments show that FusionFactory consistently outperforms the best individ-
ual LLM across all 14 benchmarks, with the optimal fusion configuration varying
across benchmarks, highlighting the promise of multi-LLM log data as a practical
foundation for fusing diverse LLM capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper addresses the challenge of leveraging the complementary strengths of diverse Large Language Models (LLMs), which are often captured in real-world multi-LLM log data. The paper argues that practical LLM fusion must satisfy two requirements: 1) compatibility with real-world serving scenarios (local and API-based), and 2) flexibility to operate at different stages of the LLM pipeline (fine-tuning and inference). The introduction of LLMFusionBench spans 14 tasks across 6 domains. It includes responses from 20 open-source LLMs (8B-671B) and rich supervision data, including task performance, cost, and LLM Judge scores. Then, a systematic framework fusionFactory is introduced for stage-aware fusion. Experiments show that FusionFactory consistently outperforms the best individual LLM across all 14 benchmarks. Thought-level fusion achieves the best overall performance, while Query-level fusion offers the best balance of performance and efficiency.

### Strengths
1. It introduces LLMFusionBench, a large-scale, diverse, and well-structured benchmark covering 14 tasks across 6 domains, responses from 20 LLMs (8B-671B), and including critical metadata like performance, cost, and LLM Judge scores.
2. FusionFactory is an innovative and systematic framework that comprehensively explores fusion at three distinct stages—Query-level (Early), Thought-level (Mid), and Model-level (Late). This stage-aware design satisfies the requirement for practical flexibility and demonstrates broader applicability than prior work.

### Weaknesses
1. For model-level fusion. The analysis should include more robust distillation or merging methods (e.g., parameter merging or logit-distillation for open-source models) to truly demonstrate the limit of model-level fusion using the logs, rather than just the limit of the chosen SFT strategy.
2. While the results claim Query-level fusion has minimal computational overhead, there is no dedicated, quantitative comparison of the latency or API cost of the three FusionFactory levels when deployed in an inference setting.

### Questions
1. What is loss funciton L in the Eq. 5. 
2. To fully justify the framework's practical claims, please provide a quantitative comparison of inference latency/cost for a sample task across the optimal configurations of the three FusionFactory levels.
3. For the Query-level fusion, please provide an ablation on the input features used by the router (e.g., GraphRouter).

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
The paper introduces LLMFusionBench, a large-scale benchmark comprising responses from 20 open-source LLMs (ranging from 8B to 671B parameters) across 14 tasks in 6 domains, totaling 103M tokens. Built from real multi-LLM log data, it captures both direct and reasoning-augmented (chain-of-thought) responses, along with performance, cost, and LLM-judged quality scores. On top of this benchmark, the authors propose FusionFactory, a systematic framework for fusing LLM capabilities at three distinct levels:

(1) Query-level fusion: uses tailored routers to select the best LLM per query;

(2) Thought-level fusion: retrieves abstract reasoning templates from high-performing past responses to guide new generations;

(3) Model-level fusion: distills top-quality responses into a single base model via supervised fine-tuning.

### Strengths
(1)The work is motivated by the widespread practice of using multiple LLMs in real systems (e.g., API platforms, agentic workflows), which naturally generates valuable multi-LLM log data—making the research question highly relevant and actionable.


(2)Introduction of LLMFusionBench which is a Comprehensive and Publicly Valuable Resource

(3)The framework is designed to work in both local (weights accessible) and API-based (black-box) serving scenarios, addressing a critical gap in prior fusion methods that often rely on internal model states or logits unavailable via APIs.

### Weaknesses
(1)The benchmark is constructed by actively querying 20 open-source LLMs with fixed prompts, rather than using real-world operational logs from actual multi-LLM deployments (e.g., user-facing API platforms). This synthetic setup may not reflect true usage patterns, query distributions, or failure modes seen in practice.

(2)The paper introduces an LLM judge to score “insightfulness,” but this introduces potential circularity: the same type of model used in fusion is also used to evaluate it. Moreover, the judge’s prompt and reliability are not rigorously validated.

(3)While thought-level fusion shows strong gains, the method depends heavily on the quality of the summarizer (LLaMA-3-70B) and the similarity search. There is no ablation on: the impact of summarization errors, the sensitivity to embedding model choice and the performance when retrieved templates are irrelevant or misleading.

(4)Model-level fusion (via SFT) consistently lags behind other methods, yet the analysis stops at “overfitting” and “task heterogeneity.” No experiments probe whether architectural mismatches, training instability, or label noise contribute.

### Questions
(1)Would techniques like curriculum learning, response filtering, or reinforcement learning improve distillation?

(2)Is the base model (LLaMA-3-8B) too small to effectively absorb knowledge from diverse, larger LLMs?

### Soundness
4

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
This paper introduces LLMFusionBench, a large-scale benchmark for LLM fusion that spans 14 tasks across five domains, with responses from 20 open-source LLMs (8B–671B) totaling 103M tokens. Building on LLMFusionBench, we propose FusionFactory, a systematic framework with three elaborated levels: (1) query-level fusion via tailored LLM routers, (2) thought-level fusion leveraging retrieved abstract reasoning templates, and (3) model-level fusion via distillation from top-ranked responses. Experiments benchmark different methods.

### Strengths
1. Query-level fusion, Thought-level fusion, and Model-level fusion for LLMs are important.

2. Benchmark datasets are provided.

3. Experiments show the performance on different fusion levels.

### Weaknesses
1. The three fusion level is related but not very close. Each fusion level already has a few benchmark papers. It might be suitable for industry pipeline as an all-in-one pipeline for fine tuning a model while the research contribution may be limited.

2. Benchmark on each level is relatively simple and lacks in-depth research analysis.

3. In model-level fusion, the fine-tuned model performs worse than the zero-shot model.

### Questions
The primary area should be benchmark track, not "foundation or frontier models, including LLMs"?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents LLMFusionBench, a large-scale benchmark built to study and evaluate how multiple large language models (LLMs) can be combined using multi-LLM log data. While framed under the “FusionFactory” framework, the work does not propose a fundamentally new fusion algorithm; rather, it systematizes and benchmarks existing approaches—routing, reasoning retrieval, and distillation—across three stages: query-level, thought-level, and model-level. LLMFusionBench includes responses from 20 open- and closed-source LLMs across 14 tasks and 6 domains, offering standardized data for analyzing cross-model complementarity. Through empirical comparisons, the paper shows that previously known techniques perform differently across fusion levels, with thought-level fusion achieving the strongest gains.

### Strengths
1. The paper introduces LLMFusionBench, a large-scale benchmark that compiles and standardizes multi-LLM log data—responses from diverse language models across multiple tasks and domains—to facilitate systematic studies of model capability fusion.

### Weaknesses
1.The paper offers limited to no novelty in term of methodology, as it mainly consolidates previously established techniques, such as routing, reasoning retrieval, and distillation, into a benchmark framework, and most of the findings reported from the proposed setup are already well-known in existing literature, e.g. any "fusion method" is better than vanilla LLM.
2. What are the direct strengths of the curated dataset here over previous datasets that also generate data from LLM for distillation purposes?
2. The paper provides no valuable insights or actionable suggestions on how to design or improve fusion methods—the benchmarking results are largely descriptive, without yielding deeper understanding or principles that could guide future research.

### Questions
Please see weaknesses.

### Soundness
1

### Presentation
2

### Contribution
1
