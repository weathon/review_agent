# Toward Effective Tool-Integrated Reasoning via Self-Evolved Preference Learning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 6

## Abstract
Tool-Integrated Reasoning (TIR) enables large language models (LLMs) to enhance their internal reasoning ability by integrating external tools. However, models with TIR often exhibit suboptimal behaviors, including insufficient tool calls, excessive tool calls, and overthinking after receiving tool call results. How to empower LLMs to perform TIR efficiently and accurately, while stabilizing the reasoning process, remains an open challenge.
In this paper, we first analyze the impact of tool calls on model reasoning from the perspective of information entropy. We find that when tool call results are provided, the information entropy of subsequent reasoning content will show a clear trend of change, and the overall information entropy of the reasoning chain will vary depending on the number of tool calls. Based on these observations, we propose Tool-Light, a framework designed to encourage LLMs to perform TIR efficiently and accurately. Our framework consists of dataset construction and multi-stage fine-tuning. For dataset construction, we use the trained model for continuous self-evolved sampling, integrating two methods: vanilla sampling and entropy-guided sampling. At the same time, during the sampling process, we design strict criteria for selecting positive-negative pairs. For the training process, we introduce a two-stage method, which includes a Supervised Fine-Tuning (SFT), and Self-Evolved Direct Preference Optimization (DPO).
Test results on 10 datasets reveal the effectiveness of Tool-Light, significantly improving the efficiency and accuracy of the model in completing TIR tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies the Tool-Integrated Reasoning for LLMs, and proposes a new framework namely Tool-Light to resolve the problem of excessive tool calls during the reasoning. The key designs of the proposed framework contain dataset construction and multi-stage fine-tuning. Specifically, for dataset construction, Tool-Light employs two sampling strategies and for multi-stage fine-tuning, it introduces a two-stage training method. To support the effectiveness of Tool-Light, extensive empirical studies are conducted.

### Strengths
- The authors provide extensive experiments to demonstrate the effectiveness of **Tool-Light**, particularly through the experiments in **Figure 1**, which clearly illustrate the motivation behind the work.
- The paper is clearly written and well organized, making it easy to follow.

### Weaknesses
- **Figure 1** lacks sufficient explanation. For instance, the meaning of *Token Index* and the specific roles of *Step 1–4* are unclear.
- In **Figure 4(c)**, it is not specified whether the response length includes the tool-calling part. The figure shows that **Tool-Light** produces shorter responses than **Tool-Star**, yet the examples in the appendix (Examples 1 and 2) suggest otherwise.
- In **Line 257**, the computational complexity is claimed to be $O(n\log m)$, but the derivation of this result is not provided.
- The authors adopt two sampling strategies but do not include an ablation study to compare their effects.
- The experiments are conducted only on the 7B model. To further validate the effectiveness of **Tool-Light**, the reviewer encourages testing on both smaller (e.g., 3B) and larger (e.g., 72B) models.
- In **Line 80**, the authors state that *Pre-Aligned DPO Training* can reduce redundant tool calls. However, it is unclear why this is the case, as *Pre-Aligned DPO Training* does not appear to include mechanisms that explicitly control the number of tool calls.
- It is recommended to report the standard deviation across multiple runs to demonstrate the robustness of the experimental results.

### Questions
Answer questions in Weaknesses.

### Soundness
3

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
5

### Summary
The paper studies “tool-intelligence” in LLM agents and targets two failure modes: overthinking (too many tool calls/long chains) and underuse (skipping necessary tools). It first runs a pre-experiment that tracks token-level entropy over reasoning traces and observes patterns around tool-call moments. Building on this, it proposes Tool-Light, a two-stage pipeline: (1) SFT to establish a reasonable tool-using policy, then (2) self-evolved DPO using curated preference pairs to favor accurate, tool-efficient traces. In inference, it adds entropy-guided branching that expands alternatives at high-uncertainty steps while keeping compute controlled. Experiments span math and knowledge-intensive QA with web search and a code interpreter, reporting accuracy alongside two bespoke metrics—Efficiency (tool economy) and Necessity (using tools when needed). Results show competitive or better accuracy with fewer tool calls and shorter outputs compared to strong baselines (e.g., Tool-Star, Search-R1), plus initial ablations on sampling knobs. The paper positions entropy as a practical signal for both training supervision and test-time exploration control.

### Strengths
(1) Actionable diagnostic. Using token-level entropy to profile traces is simple, implementation-light, and yields intuitive visualizations to reason about when/why tools are called.

(2) Modular training recipe. The SFT → self-evolved DPO pipeline is straightforward to adopt on top of popular backbones and existing tool-use frameworks; no exotic infra required.

(3) Competitive results. Shows strong performance relative to recognized baselines while reducing tool calls—evidence that “lighter” tool use need not sacrifice accuracy.

### Weaknesses
**Weaknesses**

(1) Missing details. The pre-experiment omits decoding settings. No temperature, logit scaling, or sampling vs. greedy are reported. These choices change entropy levels and trends. Please specify the exact decoding config used to measure entropy and justify it. You say the entropy study spans “multiple QA datasets,” but the figure does not list them. It’s also unclear whether the same model/datasets are used later to train Tool-Light. Please enumerate the datasets in Figure 1 and clarify any reuse. For the method, entropy decides branching at top-k steps and uses top-i prefix averages. The values of k and i and their stability are not reported.

(2) Missing ablations. Table 2 varies loop count and a few sampling knobs only. Missing: (i) no-entropy (vanilla) baseline, (ii) entropy-only,  (iii) β sensitivity in DPO, (iv) reference-policy choice, and (v) branch-width sensitivity.

(3) “Multi-tool” over-claim. Most experiments use only two tools (web search + code interpreter), and several evaluations appear to be single-tool setups. There’s no test on unseen tool types or cross-tool composition tasks. As a result, the paper does not yet support broad “multi-tool generalization” claims.

**Suggestions**

(1) Control for length. Lower entropy often comes with shorter or more templated outputs. Your own results show Tool-Light reduces sequence length (Fig. 4). The same factor could explain both “lower entropy” and “fewer tool calls.” Please report entropy at fixed token positions or use length-normalized entropy.

(2) Link entropy to correctness. “Low-entropy chains use fewer tool calls” may reflect early commitment, not better answers. Please test whether entropy predicts correctness. Report AUROC for path-level mean or area-under-entropy, controlling for tool-call count.

(3) Stress test with noisy tools. Overthinking and underuse show up when tools are imperfect. Add controlled corruptions: vary retrieval precision/recall, inject code stderr/noise. Show how Tool-Light adapts tool frequency and preserves accuracy vs. Tool-Star/Search-R1. This directly probes the “analysis paralysis” claim.

### Questions
(1) Entropy dips after tool calls.
Could the “drop before the next tool call” be driven by inserting long, deterministic tool outputs (i.e., context length/format effects) rather than better reasoning? What happens if you replace tool results with semantically equivalent, length-matched paraphrases? Does the pattern remain?

(2) Scope to multi-tool.
Do the pre-experiment findings hold in true multi-tool settings, not just single-tool or search-heavy cases? Please show the same entropy analysis when multiple heterogeneous tools are available.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work addresses inefficiencies in Tool-Integrated Reasoning (TIR), where LLMs overuse, underuse, or mismanage tool calls. The authors analyze tool-usage dynamics through information entropy, finding that tool call results significantly influence reasoning entropy patterns. Building on this insight, they propose Tool-Light, a framework for more efficient and accurate TIR. Tool-Light combines self-evolved dataset construction—using vanilla and entropy-guided sampling with strict positive-negative selection—with a two-stage training scheme of Supervised Fine-Tuning and Self-Evolved DPO. Experiments on 10 datasets (e.g., AIME24, MATH, HotpotQA, etc) show that Tool-Light notably enhances both the efficiency and accuracy of tool-integrated reasoning.

### Strengths
1. Tool-Integrated /Agentical reasoning is a important topic with many potential practical applications.
2. The information-entropy based analysing after calling tools is novel and insightful. 
3. The experiment evaluation is comprehensive and the performance boost is obvious.

### Weaknesses
1. Discuss more related work from the Self-Evolved Preference Learning perspective, such as Zeng et al., Evolving LLMs' Self-Refinement Capability via Synergistic Training-Inference Optimization, and Su et al., Trust Region Preference Approximation: A Simple and Stable Reinforcement Learning Algorithm for LLM Reasoning.
2. Clarify whether the information entropy observation generalizes across all tools, and list the specific tools used in the experiments for completeness.

### Questions
Please see the above Weaknesses.

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
This paper presents Tool-Light, a framework designed to enhance the efficiency of tool-integrated reasoning (TIR) in large language models. The core empirical finding demonstrates that invoking external tools causes significant shifts in downstream token-level uncertainty, as measured by entropy. Based on this observation, the authors propose: (i) an entropy-guided sampling procedure for constructing training data, and (ii) a two-stage self-evolved Direct Preference Optimization (DPO) pipeline comprising Pre-Aligned and On-Policy phases. Experimental evaluation across ten benchmarks covering mathematical reasoning and multi-hop question answering shows comparable or superior accuracy while reducing redundant reasoning and improving both efficiency and necessity of tool usage.

### Strengths
1. The paper is well-structured with clear writing that facilitates comprehension of the proposed methodology.
2. The investigation of information entropy changes during tool invocation processes is particularly interesting. The authors effectively leverage these insights to guide their methodological design, providing a principled foundation for their approach.
3. While some components of the proposed data construction and self-evolved DPO framework draw upon established techniques, the overall approach remains intuitive and theoretically sound.
4. The authors conduct thorough experiments demonstrating the effectiveness of their method. The entropy distribution analysis particularly convincingly shows that the approach achieves lower entropy distributions, validating the theoretical motivation.

### Weaknesses
1. The analysis of entropy in tool invocation has been explored in prior work, and the conclusions drawn are not particularly surprising or groundbreaking.
2. The proposed techniques, including entropy-based sampling and evolved DPO, represent relatively incremental advances rather than significant methodological innovations.
3. While the core idea is interesting, the evaluation is restricted to DPO. The work would be significantly strengthened by demonstrating applicability to other preference learning methods such as GRPO or PPO, which would provide stronger evidence for the generalizability and robustness of the approach.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3
