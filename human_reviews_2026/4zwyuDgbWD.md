# Can LLMs Maintain Fundamental Abilities under KV Cache Compression?

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 6, 2

## Abstract
This paper investigates an underexplored challenge in large language models (LLMs): the impact of KV cache compression methods on LLMs' fundamental capabilities. Although existing methods achieve impressive compression ratios on long-context benchmarks, their effects on core model capabilities remain understudied. We present a comprehensive benchmark KVFundaBench to systematically evaluate the effects of KV cache compression across diverse fundamental LLM capabilities, spanning world knowledge, commonsense reasoning, arithmetic reasoning, code generation, safety, and long-context understanding and generation.Our analysis reveals serval key findings: (1) \textit{Task-Dependent Degradation}; (2) \textit{Model-Type Robustness} (3) \textit{Prompt Length Vulnerability}; (4) \textit{Chunk-Level Superiority}; (5) \textit{Prompt-Gain Sensitivity}; (6) \textit{Long-Context Generation Sensitivity}. Based on our analysis of attention patterns and cross-task compression performance, we propose \method{}, a novel compression approach that distinctly handles prefill and decoding phases while maintaining shot-level semantic coherence. Empirical results show that ShotKV achieves $9\%$-$18\%$ performance improvements on long-context generation tasks under aggressive compression ratios.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents KVFundaBench, a comprehensive benchmark designed to systematically evaluate the effects of KV cache compression across a wide range of fundamental LLM capabilities, including world knowledge, commonsense reasoning, arithmetic reasoning, code generation, safety, and long-context understanding and generation. Through empirical analysis, the authors identify several challenges in existing methods, particularly concerning generalization across different prompts and generation lengths. Motivated by these observations, the authors propose ShotKV, a novel compression approach that separately handles the prefill and decoding phases while preserving shot-level semantic coherence. The proposed method demonstrates superior performance over existing baselines, especially in many-shot arithmetic reasoning tasks.

### Strengths
- The authors propose an effective method based on their observations.
- The analysis is conducted from multiple perspectives, providing rich and informative insights.
- The proposed method demonstrates strong performance on some benchmarks, including LG-GSM8K.

### Weaknesses
- Use of subjective and non-standard terminology and statements
  - Several terms, such as “shot-level” and “aggressive compression ratios” in the Abstract, are subjective and not well-defined within the section, reducing the interpretability of the section.
  - Figure 4 is not self-contained. Abbreviations like WK, CSR, and AR are not standard and require readers to refer back to previous sections, making the figure difficult to interpret independently.
  - In Figure 2, the paper describes “universal attention distributions” and “specialized patterns” (e.g., for arithmetic reasoning and safety tasks) without providing clear definitions of these terms. It is also unclear what models and datasets were used for the visualizations, and whether the results are statistically meaningful.

- Need for deeper and more statistically significant analysis
  - While the paper analyzes task-dependent degradation, the explanations could be made more intuitive and direct. For example, how do differences in datasets or task characteristics lead to varying results? Beyond simple task categorization, analyzing differences by underlying capability, such as factual retrieval, multi-hop reasoning, or memorization, could provide a more meaningful interpretation.
- A broader evaluation across multiple models and datasets, along with statistical significance tests for Figures 3-7, would strengthen the findings.

- Section 4 introduces the proposed method but lacks a schematic or conceptual diagram, which makes it difficult to understand the method.

- The paper does not clearly explain how the proposed benchmark differs from or improves upon existing ones, such as LongBench, SCBench, or RULER. A comparative discussion would clarify its novelty and relevance. Also, the analysis would be more convincing if evaluated on newer and more challenging datasets. For instance, SCBench (2025) reveals limitations of prior methods such as SnapKV, showing significantly reduced performance compared to the results shown in this paper.

- Although the paper focuses on analysis, the range of compared compression techniques is not comprehensive. Incorporating more recent and diverse methods, such as KVPress (NVIDIA, 2025) or KVZip (2025), which demonstrate strong performance, would provide a more complete evaluation.

### Questions
The paper mentions “universal attention distributions,” while stating that arithmetic reasoning and safety tasks exhibit “more specialized patterns.” However, the definitions of “universal” and “specialized” in this context are unclear. What specific criteria or metrics are used to distinguish these patterns?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces KVFundaBench, a benchmark designed to evaluate how KV cache compression affects LLMs’ fundamental abilities beyond long-context understanding. While prior studies primarily focused on memory savings in extended contexts, KVFundaBench systematically measures the impact of various compression methods across diverse tasks—including world knowledge, commonsense reasoning, arithmetic reasoning, code generation, and safety. Throughexperiments, the authors claim to uncover key insights such as task-dependent degradation, model-type robustness, and prompt-length vulnerability, revealing that reasoning-heavy and few-shot tasks are especially sensitive to aggressive compression. Motivated by these findings, the paper proposes ShotK*, a shot-aware compression framework that treats prompt examples as coherent semantic units and applies distinct strategies to the prefill and decoding phases.

### Strengths
1. The paper tries to analyze the performance degradation of llms across a diverse set of scenarios, not limited to conventional benchmarks such as LongBench or NIAH.

### Weaknesses
1. The paper claims to analyze the performance degradation of LLMs under long-context scenarios, yet most of the datasets used are short- to medium-length (e.g., GSM8K) — which undermines the strength of the “long-context” claim.
2. Many of the core observations have been presented in previous works such as [1] or 2] & SnapKV shows that (4) chunk-level is a better strategy — so the novelty of the empirical findings is limited.
3. The proposed benchmark is built entirely on existing datasets and thus offers no novel dataset contribution.
4. The number of baselines considered is limited: the paper omits some state-of-the-art and up-to-date approaches (for example [1]) that are closely related to the proposed method, which reduces the strength of the comparative evaluation.
5. The ultimate goal of most KV-cache compression techniques is inference efficiency (faster decoding, lower memory footprint) with minimal performance loss. While the proposed work emphasizes improved performance under compression (which may itself be questioned in competitiveness to SOTA), the work provides no experiments on efficiency metrics (latency, GPU memory, throughput) — thus its practical viability remains unclear.
6. The proposed ShotKV method exhibits limited novelty. Its two core strategies closely resemble prior work: the mechanism in prefilling stage to compute importance scores is analogous to that in SnapKV. Meanwhile, the decoding-phase reuse policy mirrors approaches already established in the existing literature (e.g., [1]). Thus the contribution may be characterized as incremental rather than substantially novel.
7. ShotKV’s design assumption — namely that the “prefill” phase consists of multi-shot examples — limits its generalizability. Numerous realistic tasks (e.g., multi-turn dialogue systems, single-shot prompts, interactive question-answering) do not conform to a few-shot prefilling paradigm. It is therefore unclear how well ShotKV will generalize to these non-few-shot settings, especially compared to other state-of-the-art KV cache compression methods that do not rely on shot-based structures.

[1] Kv cache compression, but what must we give in return? a comprehensive benchmark of long context capable approaches. Yuan et. al., EMNLP 2024.
[2] Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference, Tang et. al., ICML 2024.

### Questions
Please see Weaknesses.

### Soundness
1

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
4

### Summary
This paper addresses the critical challenge of maintaining Large Language Model (LLM) performance when utilizing Key-Value (KV) cache compression techniques. The paper introduces a new evaluation suite, KVFundaBench, specifically designed to test fundamental LLM capabilities after compression. Using this benchmark to analyze existing compression methods, the paper proposes a novel approach, ShotKV, which is claimed to successfully mitigate the primary performance degradation issues.

### Strengths
1. KVFundaBench provides a valuable structured evaluation suite for assessing the impact of compression on fundamental capabilities, which is an important diagnostic tool for the community.
2. The proposed ShotKV method successfully demonstrates superior performance over existing compression baselines on specific, targeted tasks within the benchmark, indicating promising conceptual improvements.

### Weaknesses
1. The experimental results are not comprehensive enough to definitively establish the superiority of ShotKV across the full spectrum of compression tasks.
2. The KVFundaBench benchmark's novelty is somewhat limited, as it appears to be primarily a compilation and reorganization of existing public datasets and tasks.

### Questions
Questions
1. What is the computational overhead of ShotKV compared to other KV cache compression algorithms?

Suggestions
1. Equations (4) through (6) currently describe a solution for finding an optimal set that maximizes a score. However, the corresponding explanation in the text details a greedy algorithm approach. The equations must be modified to accurately and explicitly represent the iterative, local optimization performed by the described greedy algorithm to ensure consistency between the mathematical formulation and the implementation.
2. Figure 6 appears to be referenced and positioned after Figure 7 and Figure 8. It would be better to renumber the figures sequentially.
3. The related work on KV Cache Compression should include KVZip[1], a recent KV cache compression method that outperforms previous methods.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper evaluates KV-cache compression methods across a diverse set of LLM benchmarks under a few-shot chain-of-thought (CoT) prompting setup, which the authors term KVFundaBench. The experiments show that pruning few-shot exemplars causes varying degrees of degradation across word knowledge (WK), commonsense reasoning (CSR), arithmetic reasoning (AR), code generation (CG), and safety tasks. To mitigate this, the authors propose KVShot, a KV-cache compression approach that prunes exemplars at the *shot* level based on attention scores.

### Strengths
The paper evaluates KV cache compression methods in a few-shot chain-of-thought (CoT) setting using a diverse set of benchmarks.

### Weaknesses
- Unclear writing
    - The paper is difficult to follow. It is unclear what the evaluation setting and CoT prompt are, and that the goal is to measure KV cache compression under the few-shot CoT prompting setting until the end of the Section 3. Moreover, the paper does not clearly indicate when compression occurs; whether during the decoding phase, and if so, how frequently it is applied.
- Question about generalizability & effectiveness of the method
    - The proposed ShotKV method does not demonstrate meaningful improvements, as shown in Figure 5. In the WK / CSR case, its performance is comparable to StreamingLLM and significantly worse than PyramidKV. In the AR case, it is not statistically significantly better than ChunkKV. In the CG case, it performs worse than H2O. These results are also omitted in Tables in Section 4.
    - The method appears to be tailored specifically for the few-shot prompting setting and does not generalize well to other scenarios (e.g. inference time reasoning models).
- Question about the contribution of the KVFundaBench
    - The benchmark appears to be a collection of existing datasets evaluated under the few-shot CoT prompting setting for KV cache compression. Framing this as measuring the “degradation of fundamental capability” is misleading; the main contribution seems to be evaluating KV cache compression methods in few-shot CoT contexts, which does not generalize to other scenarios. Furthermore, Section 3.3 and Figure 5 do not specify how many few-shot examples are used for each benchmark.
- Concerns about the effectiveness of ShotKV
    - If the focus is solely on few-shot CoT prompting as a testbed for KV cache compression, the paper should include a simple baseline: reducing the number of shots at random (instead of selecting shots based on attention scores, as ShotKV does). Comparing ShotKV against this random-shot baseline would clarify its actual benefit.
- Missing results on real latency improvements
    - The paper does not report end-to-end latency reductions. It would be important to show how much latency is reduced in the prefill and decoding steps compared to FullKV.

### Questions
See the above weakness section.

### Soundness
2

### Presentation
1

### Contribution
2
