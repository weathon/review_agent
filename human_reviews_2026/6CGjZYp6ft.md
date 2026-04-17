# Reasoning Language Model Inference Serving Unveiled: An Empirical Study

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
The reasoning large language model (RLLM) has been proven competitive in solving complex reasoning tasks such as mathematics, coding, compared to LLM.
However, the serving performance and behavior of RLLM remains \textit{unexplored}, which may undermine the deployment and utilization of RLLM  in real-world scenario. To close this gap, in this paper, we conduct a comprehensive study of RLLM service. We first perform a pilot study on comparing the serving performance between RLLM and traditional LLM and reveal that there are several distinct differences regarding serving behavior: (1) \textit{significant memory usage and fluctuations}; (2) \textit{straggler requests}; (3) \textit{adaptive running time}; (4) \textit{domain preference}. Then we further investigate whether existing inference optimization techniques are valid for RLLM. 
Our main takeaways are that model weight quantization, KV cache quantization, and speculative decoding can improve service system efficiency with small compromise to RLLM accuracy, while prefix caching may degrade inference serving performance for small RLLM in some scenarios. Lastly, we conduct evaluation under real world workload modeled by the Gamma distribution to verify our findings. 
Empirical results for real-world workload evaluation across different datasets are \textit{aligned} with our main findings regarding RLLM serving.
We hope our work can provide the research community and industry with insights to advance RLLM inference serving.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents an in-depth empirical study of inference serving for Reasoning Large Language Models, i.e. language models augmented or fine-tuned for complex multi-step reasoning. The authors introduce a new evaluation framework called ASU and a benchmarking suite named ASU-Perf for systematically measuring RLLM serving performance. Using these tools, the paper compares the serving behavior of RLLMs to that of standard LLMs across multiple model scales and tasks. The study finds distinct differences in how RLLMs perform under inference workloads, notably: (1) substantially higher and more volatile memory usage due to long reasoning chains  (2) the presence of “straggler” requests that take significantly longer than others in batched processing (3) an adaptive running time phenomenon where RLLMs spend more time on harder queries (4) a domain-specific performance gap, with RLLMs markedly outperforming same-size LLMs on math reasoning tasks but only matching their performance on knowledge-intensive queries. In addition to characterizing these behaviors, the paper evaluates common LLM serving optimizations on RLLMs. It reports that techniques like model quantization and speculative decoding can significantly improve throughput and latency for RLLM inference with only minimal accuracy loss. In contrast, methods such as prefix caching and KV-cache quantization do not consistently help RLLMs and can even degrade performance or accuracy for smaller models. Finally, the authors simulate a real-world workload to validate their findings under realistic conditions. The results confirm that RLLMs exhibit distinct serving behavior compared to normal LLMs even with irregular, bursty traffic, and the observed performance characteristics and trade-offs remain consistent. Overall, the paper’s contributions include exposing the unique challenges of serving reasoning-oriented LMs and providing insights and tooling to guide future research and engineering in efficient RLLM deployment.

### Strengths
Novel Problem: The paper tackles how to efficiently serve reasoning-augmented LLMs, which has not been systematically studied before. As RLLMs are increasingly relevant for complex tasks, understanding their inference behavior has significant practical and scientific value. This novelty in focusing on RLLM serving performance makes the contribution unique and valuable.

Introduction of Benchmarking Tools: Beyond experiments, the paper provides concrete tools to the community. The ASU framework unifies how to assess an LLM service from multiple perspectives rather than just single metrics. Along with this, the ASU-Perf benchmark suite is introduced for evaluating RLLM serving performance in a standardized way. These contributions are likely to be useful for researchers and practitioners working on LLM infrastructure, as they offer a way to consistently measure improvements and compare approaches on RLLM workloads.

Clear Identification of Key Findings: The paper does a good job of distilling its empirical observations into a set of clear findings, which have practical implications. For example, it highlights that RLLM inference can cause extreme memory fluctuation due to long chain-of-thought. This insight warns that existing serving systems might need to be adapted to avoid OOM issues when deploying RLLMs. Another useful finding is the “straggler request” problem: when processing a batch of queries, if one query requires an especially long reasoning chain , it will significantly lag behind others and occupy the resources, reducing overall throughput while it finishes. The study’s visualization of long-tail latency distribution and identification of this bottleneck is valuable for anyone designing scheduling or batching algorithms. The authors also notice an “overthinking” phenomenon that beyond a certain token budget, adding more chain-of-thought steps can hurt performance on some datasets. This is an intriguing insight that RLLMs might sometimes generate excessively long reasoning that isn’t beneficial, indicating a need for controlling reasoning length.

### Weaknesses
Certain Findings Are Expected: A few of the observed differences between RLLMs and standard LLMs are intuitive given the nature of chain-of-thought reasoning. For instance, the fact that generating a lengthy reasoning chain consumes more tokens and thus more memory and time is not surprising. RLLMs, by design, use more tokens to arrive at an answer, so higher memory usage and occasional long latencies are to be expected. The paper strengthens these points with data, which is good; however, a skeptical view is that some results confirm known intuitions rather than reveal completely unforeseen phenomena. This could be critiqued for reinforcing obvious points although it does add value by measuring the extent. A stronger theoretical or analytical exploration of why these phenomena occur would further strengthen the contribution.

Lacks guiding solutions: The paper does an excellent job in identifying and describing problems, but is deficient in proposing or evaluating solutions. Regarding the straggler request issue, the authors provide a clear diagnosis but do not explore potential mitigation strategies, such as preemption, gang scheduling, or other advanced scheduling techniques known in the system literature. This limits the constructive contribution of the paper.

The analysis is superficial: the paper reports a surprising finding that certain optimizations reduce the performance of 7B RLLMs. However, the analysis stops here. The paper should, at the very least, propose a plausible hypothesis for such an interesting and unexpected result. Is it because small models have a weaker ability to handle quantization noise in long reasoning chains? Or is it due to some architectural issue? The lack of deeper exploration is a missed opportunity.

### Questions
The paper identifies straggler requests as a key issue in RLLM services. Given this finding, have the authors considered or experimented with any scheduling strategies other than the default schedulers in vLLM/SGLang, such as preemption of long-running requests or adopting an approximate shortest job first strategy, to mitigate this problem?

The author found that certain optimizations can reduce the performance of 7B models. Could you provide a specific hypothesis to explain this phenomenon?

The proposed ASU framework is an excellent conceptual tool. Could you elaborate on the specific metrics selected for the server-side and user-side components in your ASU-Perf suite, and explain why they are most critical for the dedicated evaluation of RLLM?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents an empirical study on the serving performance of reasoning large language models (RLLMs), highlighting how their serving behaviors differ from those of traditional LLMs. It also investigates inference optimization techniques, such as quantization and KV caching, and examines whether these methods provide measurable benefits when serving LLMs.

Their main contributions include:
1. ASU, a framework for assessing RLLM serving performance, along with ASU-Perf, its corresponding benchmark suite.
2. An empirical investigation into the key differences in serving behaviors between RLLMs and traditional LLMs.
3. An empirical study on the effectiveness of serving optimization techniques (e.g., quantization and KV caching) when applied to RLLMs.

### Strengths
1. The proposed benchmark and framework for evaluating RLLM serving are valuable contributions, especially as RLLMs become increasingly prevalent.
2. The empirical study is extensive, offering interesting observations on both serving behaviors and serving optimization techniques for RLLMs. These findings should be useful for researchers and practitioners aiming to improve systems for serving LLMs.
3 The serving performance of reasoning models remains under-explored, and this paper helps fill that gap.

### Weaknesses
This paper primarily presents empirical observations without offering much in-depth analysis. The authors treat the models largely as black boxes, running benchmark experiments and reporting results without providing deeper insights or interpretations.

Some of the evaluations, while interesting, have been explored in prior work, such as comparisons between RLLMs and LLMs that do not focus on the serving aspect. It would strengthen the paper to narrow the scope and focus more clearly on serving-related issues, which would make the key messages and contributions more distinct.

### Questions
1. In the experiments analyzing the impact of different batch sizes, could you clarify why batch size would affect model accuracy at all? Are there any interactions between computations across examples within a batch? While batch size certainly influences serving performance, it should not directly affect accuracy.
2. Could you provide an explanation for why AWQ and L4 could worsen end-to-end latency and throughput?
3. In Figure 5 (first subplot), how does the prefix cache affect accuracy? My understanding is that prefix caching should not influence the model’s output at all.

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
4

### Summary
This paper presents an empirical study on Reasoning Large Language Model (RLLM) inference serving. The authors compare RLLMs (e.g., DeepSeek-R1-Distill-Qwen) with traditional LLMs under various workloads and inference engines (vLLM, SGLang). They identify distinct serving behaviors—such as high memory fluctuation, straggler requests, adaptive running time, and domain preference—and evaluate how existing serving optimizations (quantization, speculative decoding, prefix caching, KV cache quantization) affect RLLM serving efficiency. The paper further validates these findings under real-world workloads modeled by Gamma distribution.

### Strengths
Novel and relevant empirical perspective.
This is one of the first systematic studies on the serving characteristics of reasoning-oriented LLMs, which are becoming increasingly important in practice. The empirical exploration fills a gap between model-level reasoning research and system-level inference efficiency.

Comprehensive experimental coverage.
The paper evaluates multiple model scales (7B–70B), reasoning datasets (GSM8K, MATH500, AIME24, GPQA), and optimization techniques (quantization, speculative decoding, prefix caching). This breadth enhances the generalizability of observations.

Insightful findings for system designers.
The work identifies several counterintuitive behaviors including: (1) significant memory usage
and fluctuations; (2) straggler requests; (3) adaptive running time; (4) domain preference.

### Weaknesses
Lack of explanation for partial observations.
Several empirical findings are not sufficiently explained. For instance, the paper reports that prefix caching provides little or even negative benefit for 7B reasoning models, but no analysis is given on why this happens. Without such interpretation, the results remain descriptive rather than insightful.

Missing comparison with standard LLMs in Section 5.
Section 5 focuses on evaluating the effectiveness of several optimization techniques—such as quantization, speculative decoding, prefix caching, and KV-cache quantization—on reasoning LLMs. However, it does not include any baseline results for LLMs under the same experimental setup. Without this comparison, it is difficult to determine whether the reported behaviors (e.g., prefix caching being ineffective for 7B models, speculative decoding offering limited speedup, or quantization showing inconsistent gains) are unique challenges of reasoning models or common limitations of large language models in general.

Unclear implications for system optimization.
While the paper presents many interesting empirical patterns, it does not clearly explain how these observations could guide future optimization of RLLM inference systems. The paper stops at observation without turning the results into actionable guidance for improving RLLM

### Questions
Same as weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
In this paper, as indicated by the title, the authors empirically investigates how reasoning LLMs (RLLMs) behave under ``inference serving`` and how well standard serving optimizations transfer from vanilla LLMs. The authors first motivate that RLLMs, due to long CoT generation, place qualitatively different demands on serving systems and pricing models (token budgets). They pose the central question: `` Is there any distinct difference in serving behaviors between LLM and RLLM?``

Then the authors introduce ASU, a three-part assessment that considers Accuracy, Service-end, and User-end metrics. They also provide an evaluation suite  and run controlled studies across model sizes (7B–70B), two engines (vLLM and SGLang), and datasets emphasizing math/knowledge tasks (GSM8K, MATH-500, AIME-2024, GPQA). 

The authors have some good findings:

- KV-cache usage:  ``RLLM exhibits significant KV Cache fluctuations and usage.``

- Straggler requests: ``long tail distribution of requests running
time caused by slow requests.``

- Adaptive running time: ``RLLM solves different difficulty level problems with adaptive
running time.``

- Domain preference: `` RLLM excels LLM on math reasoning while on-par on knowledge intensive tasks.``

The paper then investigates common serving optimisations for RLLMs, including 
- Weight quantization 
- KV-cache quantization 
- Prefix caching
- Speculative decoding 

And the authors got further meaningful observations:

- `` MWQ methods exert differing impacts on various metrics of RLLM inference ``

- `` KV Cache quantization can improve running efficiency for sufficient large RLLM. ``

- `` PC can accelerate larger RLLMs (14B and above) without performance degrade. ``

- `` SD improves the running time of RLLMs and deteriorates metrics like TPS. ``

I have to admit that I'm not an expert in LLM serving. It appears to me that this paper makes meaningful contributions.

### Strengths
``S1``: As long-CoT reasoning models become mainstream with the emergence of many prevalent deep reasoning models, their serving behavior is a an important systems problem.

``S2``: Good breadth of experiments across engines (vLLM and SGLang), model scales, and several optimizations (weights, KV, prefix cache, speculation).

``S3``: The summary of observations and findings are clear.

### Weaknesses
``W1``: The study focuses on GSM8K, MATH-500, AIME-2024, GPQA. Given rapid shifts in eval suites and the benchmark nature of this paper, I’d suggest the authors add SuperGPQA and CommonsenseQA.

``W2``: I would have expected to see some elaborated explanations of some observations in the paper. For example, the authors claim that:

- “We find that for sufficiently large RLLMs (14B and above), prefix caching significantly improves runtime speed and serving metrics without compromising performance. However, for 7B models, prefix caching negatively impacts efficiency, leading to increased latency.”

It’s interesting to see some discussions on the behind reason.

``W3``: Some typos issues such as “emeraged” and “perserved”. The authors should more carefully polish their paper.

Again, I have to admit that I’m not very familiar with LLM serving. I’ll look into the comments of other reviewers who are experts in this area for my final rating.

### Questions
``Q1``:  Why do prefix cache and KV-FP8 help large models (14B and above) but harm 7B?

### Soundness
3

### Presentation
2

### Contribution
3
