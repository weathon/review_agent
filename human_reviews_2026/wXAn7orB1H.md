# FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 2, 6, 6

## Abstract
Large language models (LLMs) have been widely deployed with rapidly expanding context windows to support increasingly demanding applications.
However, long contexts pose significant deployment challenges, primarily due to the KV cache whose size grows proportionally with context length.
While KV cache compression methods are proposed to address this issue, KV dropping methods incur considerable accuracy loss, and KV retrieval methods suffer from significant efficiency bottlenecks.
We propose FreeKV, an algorithm-system co-optimization framework to enhance KV retrieval efficiency while preserving accuracy.
On the algorithm side, FreeKV introduces speculative retrieval to shift the KV selection and recall processes out of the critical path, combined with fine-grained correction to ensure accuracy.
On the system side, FreeKV employs hybrid KV layouts across CPU and GPU memory to eliminate fragmented data transfers, and leverages double-buffered streamed recall to further improve efficiency, enabling effective overlap with computation, full latency hiding, and practical speedups from speculative recall.
Experiments demonstrate that FreeKV achieves near-lossless accuracy across various scenarios and models, delivering up to 13$\times$ speedup compared to SOTA KV retrieval methods.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
FreeKV introduces a unified algorithm-system co-optimization framework that dramatically accelerates LLM inference (decoding specifically) by eliminating KV cache retrieval latency. It does this through speculative retrieval, which reuses KV caches from the previous decoding step to fully overlap selection and recall with ongoing computation, and head-wise fine-grained correction, which selectively updates only attention heads whose query patterns change. FreeKV achieves up to 13 times faster inference compared with prior KV retrivial methods and better accuracy then KV dropping methods.

### Strengths
The paper proposes FreeKV, a speculative KV retrieval framework built on the key insight that query vectors in adjacent decoding steps exhibit high similarity. This observation enables efficient reuse of previous KV selections, and represents a novel and meaningful advancement in accelerating LLM inference (if the observation is empirically validated).

The paper presents the approach with clear organization and technical depth, detailing both the algorithmic design (speculative retrieval and head-wise correction) and the system-level implementation (hybrid layouts and streamed recall). The figures and timelines are well-constructed, making the method intuitive to follow.

The experimental evaluation is comprehensive and convincing, covering multiple models and datasets. The results consistently demonstrate that FreeKV achieves a superior trade-off between efficiency and accuracy compared with existing KV retrieval and KV dropping methods, highlighting both practical impact and generality.

### Weaknesses
While the design of FreeKV is primarily motivated by the observed similarity of queries across adjacent decoding steps, the validation of this core assumption remains limited. Section 3.1 provides some preliminary evidence, but the analysis is neither comprehensive across diverse models and prompts nor sufficiently detailed to characterize the factors influencing query similarity. A deeper investigation would strengthen the paper, because of the worry that this method could perform bad on corner case scenarios, which is also interesting to know. For example, examining whether similarity levels correlate with specific tasks, semantic patterns, or model architectures, and how training or fine-tuning methods affect this property. Additionally, it would be valuable to quantify what degree of similarity is necessary to maintain decoding accuracy. Although the appendix includes relevant ablation studies, a concise summary and discussion of these findings in the main text would make the paper’s design justification more convincing.

### Questions
What factors could affect the "query similarity" property the paper assumed?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper addresses the efficiency bottleneck of KV cache retrieval in LLM inference and proposes FreeKV, an algorithm–system co-optimization framework. It introduces speculative retrieval to move KV selection and recall off the critical path and fine-grained correction to maintain accuracy. On the system side, it uses hybrid CPU–GPU layouts and streamed recall to reduce data transfer overhead. Experiments show near-lossless accuracy and up to 13× speedup over prior retrieval methods.

### Strengths
1. The paper tackles an important and practical problem in LLM serving: efficient KV retrieval under long contexts.
2. FreeKV presents well-motivated system–algorithm co-design with comprehensive experiments covering multiple models and tasks, demonstrating impressive empirical speedups and accuracy preservation.

### Weaknesses
1. FreeKV improves runtime efficiency via engineering and overlap techniques. Its algorithmic novelty is incremental over prior work like InfiniGen and ArkVale.
2. The speculative reuse in FreeKV depends on strong query similarity assumptions that may not generalize to all model architectures or reasoning tasks.

### Questions
1. quantify how query similarity impacts recall accuracy or establish conditions where correction is triggered.
2. Include ablation studies isolating algorithmic vs. system-side gains, to show which components (speculative reuse, fine-grained correction, hybrid layout) contribute most to the speedup.
3. Strengthen novelty positioning: highlight what design principles or insights make FreeKV different from prior works like InfiniGen and ArkVale, rather than appearing as an engineering integration.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes FreeKV, an algorithm–system co-optimization framework for efficient KV-cache retrieval in long-context LLM inference. The key idea is to leverage the similarity of query vectors across adjacent decoding steps to enable speculative retrieval — shifting KV selection and recall out of the critical path — combined with fine-grained correction when query deviation occurs. On the system side, FreeKV introduces a hybrid CPU–GPU KV layout (HND on CPU, NHD on GPU) and double-buffered streamed recall to overlap transfers and computation. Experimental results show up to 13× speedup over prior retrieval methods (Arkvale, ShadowKV, InfiniGen) with negligible accuracy loss.

### Strengths
1. Clear motivation and strong setup:
The paper provides a well-motivated problem statement supported by preliminary empirical analysis.
2. Novel speculative retrieval mechanism:
The proposed speculative retrieval with fine-grained correction is a conceptually novel idea that effectively breaks the strict dependency between KV selection and query scoring, enabling computation–I/O overlap. 
3. Comprehensive and convincing experiments:
The paper evaluates FreeKV across diverse models and tasks (e.g., LongBench, and reasoning datasets) against a wide range of baselines. The results show substantial speedups and near-lossless accuracy, demonstrating both practicality and robustness.

### Weaknesses
1. Missing analysis of correction overheads:
While the fine-grained correction mechanism is central to FreeKV’s efficiency–accuracy balance, the paper lacks a quantitative analysis of correction frequency and its impact on latency under different similarity thresholds. Such results would clarify the trade-off between performance and accuracy.
2. Limited study on KV budget sensitivity:
The experiments fix the KV budget B but do not explore how varying B influences accuracy and throughput.

### Questions
1. FreeKV’s speculative retrieval design always reuses the previously recalled KV pages for the next token. Intuitively, one might expect cumulative drift or stale-selection errors over long generations, especially in reasoning tasks. However, the reported performance on long-context reasoning excels significantly. Could the authors elaborate on why this drift does not appear to accumulate in practice?
2. Since the KV recall happens right after the (t-1)th token selection, which is far before the KV is used by t, maybe we can even store the KVCache in disk?

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
The paper introduces FreeKV, a KV-cache retrieval framework that combines algorithmic ideas—speculative retrieval that reuses pages recalled at the previous step, group-consistent page selection with softmax-pooled scores, and fine-grained head-wise correction triggered by low query-similarity—with a systems design that uses hybrid KV layouts (HND on CPU, NHD on GPU) and double-buffered streamed recall to cut fragmented transfers and overlap CPU to GPU recall with compute. The system keeps page summaries and last-step queries on GPU, stores the full KV pool on CPU in a transfer-friendly layout, and converts layout on-the-fly during recall. Across LongBench v2, LongGenBench, and several long-reasoning tasks, FreeKV matches or nearly matches full-KV accuracy while delivering large end-to-end latency reductions versus prior retrieval methods. Reported speedups reach up to 13.7× vs ArkVale and up to 8.4× vs ShadowKV on A100-PCIe settings, with ablations attributing a major share to the hybrid-layout plus streamed-recall pipeline.

### Strengths
* The paper offers a clear algorithm–system co-design in which speculative reuse moves selection and recall off the critical path and head-wise correction restores accuracy only when needed.
* The hybrid HND/NHD layout and double-buffered streamed recall directly target PCIe fragmentation and enable effective overlap, which is a practical systems contribution likely to transfer to real serving stacks.
* The empirical study spans multiple models and benchmarks with detailed settings, and demonstrates near-lossless accuracy relative to full-KV along with large end-to-end speedups, supported by ablations over lengths and components.

### Weaknesses
* LongBench v2 spans 8K to 2M tokens, yet the paper truncates inputs to 64K and caps generation to 16K, which leaves the very-long regime underexplored where offloading dominates; please add at least one ≥128K long-input case and one ≥32K long-generation case to validate scaling and stress the recall pipeline.
* Since accuracy relies on an LLM-as-judge (Qwen-3-32B), consider strengthening the evaluation with a larger or multi-judge setup, report inter-judge agreement to calibrate the scores.

### Questions
* What is the average correction rate (fraction of KV heads corrected per step) across tasks? How often does correction cascade across layers? A per‑layer histogram and the incremental latency/bytes attributable to correction would clarify the practical cost of robustness.
* In which scenarios does speculative reuse mis‑speculate?

### Soundness
3

### Presentation
3

### Contribution
3
