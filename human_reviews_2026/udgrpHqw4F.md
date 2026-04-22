# Hold Onto That Thought: Assessing KV Cache Compression On Reasoning

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Large language models (LLMs) have demonstrated remarkable performance on long-context tasks, but are often bottlenecked by memory constraints. Namely, the KV cache, which is used to significantly speed up attention computations, grows linearly with context length. A suite of compression algorithms has been introduced to alleviate cache growth by evicting unimportant tokens. However, several popular strategies are targeted towards the prefill phase, i.e., processing long prompt context, and their performance is rarely assessed on reasoning tasks requiring long decoding. In particular, short but complex prompts, such as those in benchmarks like GSM8K and MATH500, often benefit from multi-step reasoning and self-reflection, resulting in thinking sequences thousands of tokens long. In this work, we benchmark the performance of several popular compression strategies on long-reasoning tasks. For the non-reasoning Llama-3.1-8B-Instruct, we determine that no singular strategy fits all, and that performance is heavily influenced by dataset type. However, we discover that H2O and our decoding-enabled variant of SnapKV are dominant strategies for reasoning models, indicating the utility of heavy-hitter tracking for reasoning traces. We also find that eviction strategies at low budgets can produce longer reasoning traces, revealing a tradeoff between cache size and inference costs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper evaluates a variety of state-of-the-art KV cache compression algorithms across eight prominent reasoning benchmarks. The study fills a gap in existing literature by shifting focus from prefill-stage, prompt-centric compression to long-decoding, multi-step reasoning scenarios, revealing nuanced tradeoffs between memory savings, computational latency, and answer quality.

### Strengths
1. The study rigorously benchmarks multiple leading KV cache compression strategies using a representative suite of reasoning datasets
2. The analysis highlights that, contrary to some existing assumptions, attention-based heavy-hitter strategies (H2O, SnapKV-D) dominate in the decoding phase for reasoning models—sometimes even outperforming the full cache baseline. This finding has practical consequences for both research and deployment.

### Weaknesses
1. Several directly relevant methods in the recent literature, especially regarding head selection, layer- or depth-wise compression, and low-rank approaches, are omitted from the empirical comparison and insufficiently discussed. 
[1] R-KV: Redundancy-aware KV Cache Compression for Reasoning Models
[2] Layer-Condensed KV Cache for Efficient Inference of Large Language Models
2. The mathematical treatment of token importance measures (accumulated attention, L2norm...) is mostly descriptive and empirical, with little formal analysis of conditions under which these heuristics succeed or fail. Beyond Equation 1 and qualitative descriptions, there is limited exploration of convergence, stability, or guarantees about quality degradation as budget shrinks. For example, there is no explicit loss/objective formalization relating retained tokens’ attention scores to model output fidelity
3. Inconsistent or ambiguous presentation. Several figures lack legends or use color encodings without a keyed explanation (see Figure 4), reducing interpretability. Table 1 is dense and difficult to parse, with cryptic abbreviations (e.g., M, DQ, DL, LN) left undefined in the immediate context, and missing values are not clarified. This impedes reproducibility and transparency.
4. The study benchmarks only on a handful of popular reasoning datasets, many of which have similar format and chain-of-thought prompting artifacts. Thus, the generality of claims about method superiority might be weaker in other settings (e.g., dialogue, scientific QA).
5. the core contributions are predominantly evaluative or minor variants (e.g., SnapKV-Decoding). There is no clearly articulated new compression principle or learned policy.

### Questions
See Weaknesses for details.

### Soundness
2

### Presentation
2

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
The paper examines how different KV cache compression techniques affect the performance of LLMs on reasoning tasks. It highlights that existing compression strategies are often tailored for the prefill phase and may not be suitable for tasks requiring extensive decoding. Through benchmarking, the authors identify H2O and a variant of SnapKV as effective strategies for reasoning models, emphasizing the need for heavy-hitter tracking in reasoning traces. The study also discusses the balance between cache size and inference costs, noting that smaller cache budgets can result in longer reasoning traces.

### Strengths
1. Addresses a critical issue in LLMs related to memory constraints during reasoning tasks.
2. Provides a comprehensive evaluation of multiple KV cache compression strategies across various reasoning benchmarks.
3. Identifies specific strategies (H2O and SnapKV variant) that enhance performance in reasoning tasks.

### Weaknesses
1. The study focuses on a single LLM (Llama-3.1-8B-Instruct), which may limit the generalizability of the findings to other models.
2. Lacks a discussion on the computational overhead introduced by implementing the recommended compression strategies.
3. Does not provide detailed explanations of the underlying mechanisms of the identified effective strategies.

### Questions
1. How do the identified compression strategies perform on LLMs with different architectures or sizes?
2. What are the computational costs associated with implementing H2O and the SnapKV variant in practice?
3. Could the findings be influenced by specific hardware configurations, and how might they generalize across different platforms?
4. Are there any potential drawbacks or limitations to using these compression strategies in real-world applications?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies KV-cache compression strategies for long-chain reasoning in LLMs. The authors benchmark multiple KV eviction techniques—including H2O, StreamingLLM, K-Norm, and an adapted variant of SnapKV (extended to decoding phase, termed SnapKV-D) on multi-step reasoning tasks. They show that long-form reasoning workloads differ substantially from long-context inference: compression during decoding is critical, and conventional prefill-focused KV strategies can degrade or even destabilize reasoning traces. SnapKV-D generally performs best among tested methods, and the authors also surface an interesting phenomenon where aggressive eviction may increase generation length due to reasoning loops.

### Strengths
1. This paper focuses on an increasingly important setting: long reasoning LLMs, where decode-phase KV dominates.
2. Systematic comparison across multiple KV compression methods and budgets.
3. Empirical evidence that heavy-hitter approaches (SnapKV-D, H2O) outperform naive recency or norm methods for reasoning.

### Weaknesses
1. The paper evaluates eviction‐based strategies but does not compare against modern sparse attention approaches designed for long reasoning (e.g., DeepSeek sparse attention, SeerAttention). These are important reference points for understanding where eviction sits in the design space.
2. No evaluation on multi-turn or interactive reasoning. Long decoding commonly occurs in iterative workflows (MathChat-style step reasoning, planning with feedback). Single-turn benchmarks may not fully reflect practical long-chain settings.
3. The work observes several phenomena (e.g., compression can increase output length), yet provides limited guidance for practitioners on how to choose or tune strategies in practice. 
4. Lack of fine-grained analysis on RL-trained reasoning models. RL-distilled reasoners (e.g., DeepSeek-R1-Distill) have been shown to exhibit different attention concentration patterns vs non-reasoning models. The paper observes performance differences but does not probe whether attention sparsity, focusing behavior, or step-critical token retention varies across training paradigms. Understanding this distinction would increase interpretability and generalization of findings.

### Questions
1. Could you elaborate on SnapKV-D?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a systematic study of KV cache compression strategies on reasoning-heavy LLM benchmarks, emphasizing the decoding phase rather than the prefill phase. It provides valuable insights into how different eviction methods—particularly attention-based ones like H2O and a decoding-enabled variant of SnapKV (SnapKV-D)—perform under constrained memory settings. The authors also extend NVIDIA’s kvpress library to support decoding-phase compression.

The paper fills a meaningful empirical gap and is well-executed in its benchmarking methodology. While it is not theoretically novel, it is well-motivated and will likely interest both practitioners and researchers working on efficient inference for reasoning-capable LLMs.

### Strengths
1. Clear empirical gap: Most cache compression work targets long-prompt tasks (e.g., LongBench, RULER), but this paper focuses on long-reasoning scenarios where the generation dominates memory use. This is a genuinely underexplored regime.

2. Comprehensive evaluation: The benchmark suite spans eight reasoning datasets (DROP, ReClor, FOLIO, StrategyQA, CommonSenseQA, OpenBookQA, GSM8K, MATH-500) and multiple models (Llama-3.1-8B-Instruct, DeepSeek-R1-Distill-Llama/Qwen, Nemotron-Nano-8B). This breadth makes the results broadly relevant.

3. Methodological rigor: The experiments cover multiple cache budgets (128–512), token limits (up to 2048), and both latency and accuracy metrics. The discovery that smaller cache budgets can paradoxically lead to longer reasoning traces is particularly insightful.

4. Practical contribution: Extending kvpress to handle decoding-phase compression is a useful engineering effort likely to be adopted by others studying efficient inference.

5. Empirical insight: Attention-based “heavy-hitter” methods (H2O, SnapKV-D) consistently outperform other approaches. In several settings, SnapKV-D even surpasses full-cache accuracy, showing that selective eviction can regularize reasoning behavior.

### Weaknesses
1. Limited novelty: The work is primarily empirical. Extending SnapKV for decoding and modifying kvpress are incremental, though valuable, contributions.

2. Implementation details under-specified: The paper lacks detail on how token importance is updated during decoding for SnapKV-D—e.g., whether attention aggregation happens online or at fixed intervals.

3. Normalization and fairness: Because compression methods produce different output lengths, accuracy comparisons may not be normalized by generation length. This could bias metrics toward “talkative” models.

4. Presentation overload: Table 1 is extremely large and difficult to interpret. Averaging across datasets or grouping results visually (as in Figure 4) would make trends clearer.

5. Experimental gaps: No statistical significance or confidence intervals are reported. Additional ablations (e.g., observation window size, eviction frequency) would better characterize SnapKV-D behavior.

### Questions
See the weakness.

### Soundness
3

### Presentation
3

### Contribution
3
