# An Empirical Study of Extremely Low-Bit Quantization for Large Language Models

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Recent research on LLM quantization has predominantly focused on post-training quantization (PTQ). While effective at higher bit-widths, PTQ still suffers from severe performance degradation in extremely low-bit settings (eg 2-bit), limiting its applicability in resource-constrained environments. In contrast, quantization-aware training (QAT) offers a promising solution to recover the accuracy loss introduced by quantization. However, due to its substantial demands on training data and computational resources, QAT remains largely underexplored for LLMs. In this work, we present a comprehensive empirical study of QAT for extremely low-bit quantized LLMs. We investigate critical factors affecting QAT effectiveness, including quantizer design, quantization granularity, initialization strategies, training data selection, and training hyperparameters. Based on these insights, we propose a general QAT recipe and validate it on LLaMA3 models, achieving state-of-the-art performance under extremely low-bit settings. All code and training details will be released to facilitate reproducibility and foster future research on QAT for LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a comprehensive empirical investigation of QAT for extremely low-bit LLMs. Motivated by the limitations of PTQ under 2-bit settings, the authors systematically study key factors influencing QAT performance including quantizer design, granularity, initialization, training data, and hyperparameters. They propose an enhanced quantizer named ELSQ+, that integrates floating-point offsets, learnable step sizes, and stretched elastic levels for improved stability and representational capacity. Building on these insights, the paper introduces a general and cost-effective QAT recipe called NashQuant, achieving remarkable performance on LLaMA3 models (1B–3B) with 2-bit quantization, outperforming PTQ and QAT baselines such as ParetoQ and EfficientQAT.

### Strengths
- The originality of this work is remarkable. This work is among the first to systematically study QAT for LLMs under extremely low-bit settings, addressing a clear gap in prior research by providing some QAT-specific characteristics for LLM quantization. 
- Some empirical insights provided by this paper is impressive, like training data selection based on perplexity distribution. I believe these observations could provide valuable insights for future studies.

### Weaknesses
- While positioned as an empirical study, the scope of exploration is relatively narrow compared to prior works like [[1]](https://arxiv.org/abs/2402.18158). Much of the paper’s content focuses on tuning-level ablations (e.g., hyperparameter selection, quantizer variants) that, while useful, offer limited conceptual novelty or broader methodological insight.
- The proposed method (i.e. NashQuant)'s novelty is limited, as it appears to be an incremental improvement over existing approaches like ParetoQ, primarily combining known design elements (e.g., learnable step size, stretched quantization levels) with empirical tuning. The resulting gains, while consistent, may not convincingly demonstrate a substantial methodological innovation.
- The experiment results are not convincing enough to show the paper's proposed insights & method's general applicability, which might limit the paper's significance for this field:
  - The experiments are restricted to small-scale models (1B–3B), leaving uncertainty about whether the proposed findings and methods generalize to larger LLMs (e.g., 7B–70B), where 2-bit quantization is more practically relevant. The conclusions drawn from these smaller models may therefore not reliably reflect behavior at deployment scales.
  - The evaluation focuses mainly on standard and relatively simple zero-shot benchmarks (e.g. PIQA, HellaSwag), omitting more challenging or representative tasks such as long-context reasoning, multi-turn dialogue, or mathematical problem solving. Since prior studies [[2]](https://arxiv.org/abs/2402.18158) have shown that quantization effects vary significantly across task types, the current experimental coverage may be insufficient to substantiate the claimed generality and robustness of the empirical conclusions.

Overall, I think the contributions and novelty of this paper are relatively insufficient, and could be improved with more comprehensive experiments and methodological designations.

### Questions
Could you further provide additional evaluation results on more advanced tasks (like long-context tasks and reasoning tasks) and larger models (like 7B), to justify your results' general applicability?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents a comprehensive empirical investigation into Quantization-Aware Training (QAT) for Large Language Models (LLMs), specifically focusing on the challenging regime of extremely low-bit quantization (2-bit). The authors argue that while Post-Training Quantization (PTQ) methods struggle to maintain accuracy at such low bit-widths, QAT remains underexplored for LLMs due to its high computational cost. Based on these insights, the authors formulate a general QAT recipe named NashQuant. They validate this recipe on LLaMA3-1B and LLaMA3-3B models, demonstrating that NashQuant achieves new state-of-the-art results in 2-bit quantization, significantly outperforming existing PTQ and QAT baselines in both perplexity and zero-shot accuracy.

### Strengths
1. The primary strength of this paper is its methodical and thorough investigation. Instead of proposing a single, isolated trick, the authors deconstruct the entire QAT pipeline for LLMs and provide data-driven insights for each component. This kind of systematic study is incredibly valuable to the research community, as it helps establish best practices and provides a solid foundation for future work.

2. : The proposed ELSQ+ quantizer is well-designed. Each of its three components (floating-point offset, learnable step size, extra levels) is clearly justified and empirically validated with strong ablation studies (Figure 1). The resulting quantizer demonstrates superior stability and convergence.

3. The analysis of training data based on perplexity (Section 2.4) is a novel and significant contribution. The finding that "middle-PPL" data is optimal for QAT is non-obvious and provides a practical, low-cost strategy for improving quantization performance without needing proprietary, highly-curated datasets.

### Weaknesses
1. The paper is motivated by the need to deploy LLMs on resource-constrained devices, which implies a focus on latency, memory footprint, and energy consumption. However, the evaluation is based entirely on software metrics (perplexity and accuracy). The inclusion of actual inference speedups or memory usage measurements on a target hardware platform (e.g., a mobile CPU/GPU) would have made the practical benefits of the proposed method much more concrete.

2. The experiments are conducted exclusively on the LLaMA3 model family. While the results are strong, the claim of a "general QAT recipe" would be more robust if it were validated on at least one other distinct model architecture (e.g., a model from the Mistral, Phi, or Gemma families) to demonstrate its generalizability.

3. The authors did not scale the quantization results on larger model size, which is more important and suitable for quantization.

### Questions
1. The finding that "middle-PPL" data is optimal for QAT is fascinating. Could you provide more intuition as to why this might be the case? Is it possible that low-PPL data offers too weak a gradient signal for effective weight reconstruction, while high-PPL data is too noisy or "out-of-distribution" for the quantized model to learn from effectively?

2. Figure 3 shows a clear performance benefit when moving from a group size of 128 to 64. Did you experiment the inference efficiency between different group-size?

3. The authors should also include the recent PTQ method for low-bit quantization, such as QuIP# [1]， SliM-LLM[2] ...

[1] QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks
[2] SliM-LLM: Salience-Driven Mixed-Precision Quantization for Large Language Models

### Soundness
3

### Presentation
2

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
The paper presents an empirical study of quantization-aware training (QAT) for 2-bit (extremely low-bit) weight quantization of LLMs. It surveys design choices—quantizer form, granularity, initialization, data selection, and training hyperparameters—and assembles them into a recipe the authors call NashQuant. Key ingredients include an “**ELSQ+**” quantizer (learnable step size with a floating offset and a stretched-elastic mapping), channel-wise quantization, MSE-based initialization, a WSD learning-rate scheduler, and a *middle-perplexity* subset of SlimPajama for training. Experiments (mainly on **LLaMA-3-1B/3B**) indicate lower perplexity and higher zero-shot accuracy than several PTQ and QAT baselines; code is promised for release.

### Strengths
* Clarity of design space. The paper systematically enumerates QAT choices (quantizer, granularity, init, data, schedule) and ties them to an end-to-end recipe. 
* Quantizer detail. ELSQ+ is specified with an explicit mapping (Eq. 5), making the quantizer reproducible; the discussion of gradients for clipping vs step-size is helpful for QAT stability.  
* Practical training guidance. Concrete training settings (80k steps, batch size 128, AdamW, WSD) and a general QAT recipe are documented.  
* Data selection insight. The mid-perplexity subset hypothesis is interesting and grounded in simple model-data alignment diagnostics. 
* Competitive results (at small scales). On LLaMA-3-1B/3B, the recipe tends to outperform reported PTQ/QAT baselines on perplexity and zero-shot accuracy.

### Weaknesses
* Questionable novelty. ELSQ+ mainly combines existing elements (learnable step/offset as in LSQ/LSQ+, stretched-elastic levels as in ParetoQ/SEQ), with limited ablation isolating each component’s marginal effect. Consider head-to-head ablations: LSQ vs LSQ+ vs SEQ vs ELSQ+, all else fixed.  
* Comparability & fairness. Main comparisons fix PTQ at group-size 64 while NashQuant is reported channel-wise; only later does NashQuant† add g64. This risks over-attributing gains to granularity. Please report matched-granularity comparisons for all methods or normalize all to the same granularity.  
* Baseline limitations. EfficientQAT was evaluated with a broken/reduced training stage in the channel-wise case; that weakens conclusions regarding SOTA. A fix or alternate reproduction is needed. 
* Scale and breadth. Despite mentioning 8B, core evidence focuses on 1B/3B and seven EleutherAI zero-shot tasks. Results on larger models, activation/KV-cache quantization, latency/throughput on real hardware, and downstream finetuning tasks are missing—limiting practical significance.  
* Reproducibility is promised, not demonstrated. Code and scripts are “to be released,” which undermines verification at review time.  
* Narrow evaluation target. The work studies *weight-only* 2-bit QAT; many production scenarios require activation or KV-cache quantization and end-to-end speed/footprint metrics.

### Questions
1. Granularity control. Can you report a table with *all* methods at (a) channel-wise and (b) group-size 64, with identical training budgets, to cleanly separate method vs granularity gains? (Your own text emphasizes this distinction.) 
2. ELSQ+ necessity. Provide ablations isolating offset vs stretched-elasticity vs step-size learning, measured at fixed granularity and schedule. How much does each piece contribute? 
3. Baseline integrity. Can you fix the EfficientQAT reproduction and update results? If not, consider removing it from SOTA claims or clearly labeling it as incomplete wherever cited. 
4. Scaling evidence. Do your conclusions hold at 8B (or larger) with the same budget? Please add results or discuss failure modes if training becomes unstable. 
5. Practicality. Report latency/throughput and memory footprint on at least one real inference backend for 2-bit channel-wise vs g64; otherwise, it’s hard to judge deployability.
6. Beyond weights. Any preliminary results for low-bit activations or KV-cache? Even 4-bit activations would broaden impact.
7. Data selection. The mid-PPL hypothesis is interesting—does it generalize across models/datasets? Please add a cross-model validation or negative result.

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
4

### Summary
The paper investigates quantization-aware training (QAT) for large language models (LLMs) under extremely low-bit settings (e.g., 2-bit), where post-training quantization (PTQ) typically fails due to severe accuracy loss. Unlike PTQ, QAT can restore model performance but is rarely explored for LLMs because it requires large computational and data resources. In this study, the authors conduct a comprehensive empirical analysis of QAT, examining key factors such as quantizer design, quantization granularity, initialization and others. Based on these insights, they propose a generalized QAT training recipe that achieves state-of-the-art performance on LLaMA3 models at ultra-low bit precision.

### Strengths
- It investigates multiple influential factors (quantizer design, granularity, initialization, data, and hyperparameters), offering some insights for the community.
- This paper is well-written.

### Weaknesses
- Some of the key observations are rather trivial for a research paper. For instance, the claim that “group-wise quantization consistently outperforms channel-wise quantization” is unsurprising, since group-wise quantization is inherently more fine-grained. Similar findings have already been reported in prior studies such as GPTQ. Likewise, the statement that “advanced initialization provides clear benefits for coarse-grained quantization” is also expected, as coarse-grained quantization quantizes more real-valued weights/activations into a limited precision range, making initialization naturally more influential.

- From my perspective, the paper does not introduce any genuinely new methodology. Instead, it primarily explores existing hyperparameter choices in QAT (e.g., scaling selection) and well-known best practices from PTQ (e.g., fine-grained quantization). While these experiments may provide some empirical insights, they do not rise to the level of methodological novelty or conceptual contribution typically required for a top-tier venue like ICLR. 

- The experimental evaluation focuses exclusively on the LLaMA-3 family of dense models. However, as of 2025, mixture-of-experts (MoE) architectures have become dominant in LLMs. The absence of results on MoE-based models limits the paper’s relevance and generalizability to current research trends. 

Minor Comments: 
- There should be a space before citations, e.g., “information extraction(Xu et al., 2023)” -> “information extraction (Xu et al., 2023)”.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
