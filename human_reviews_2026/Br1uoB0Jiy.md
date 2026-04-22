# TokenSeek: Memory Efficient Fine Tuning via Instance-Aware Token Ditching

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 4

## Abstract
Fine tuning has been regarded as a de facto approach for adapting large language models (LLMs) to downstream tasks, but the high training memory consumption inherited from LLMs makes this process inefficient. Among existing memory efficient approaches, activation-related optimization has proven particularly effective, as activations consistently dominate overall memory consumption. Although prior arts offer various activation optimization strategies, their data-agnostic nature ultimately results in ineffective and unstable fine tuning. In this paper, we propose TokenSeek, a universal plugin solution for various transformer-based models through instance-aware token seeking and ditching, achieving significant fine-tuning memory savings (e.g., requiring only 14.8% of the memory on Llama3.2 1B) with on-par or even better performance. Furthermore, our interpretable token seeking process reveals the underlying reasons for its effectiveness, offering valuable insights for future research on token efficiency. Homepage: runjia.tech/iclr_tokenseek.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes TOKENSEEK, a method for memory-efficient fine-tuning (MEFT) of LLMs that addresses the bottleneck of activation memory. TOKENSEEK introduces an instance-aware paradigm that dynamically identifies and preserves the most informative tokens per input instance while discarding less crucial ones during gradient computation.

### Strengths
>s1: The key insight behind TOKENSEEK is that not all training tokens within LLMs contribute equally to
model fine-tuning.

>s2: The token selection process is based on a transparent, hybrid scoring mechanism leveraging both contextual attention and gradient information. This provides interpretability into why certain tokens are selected and contributes to more stable and effective fine-tuning compared to data-agnostic or random baselines.

>s3: The presentation is clear, and the figures are of high quality.

### Weaknesses
See Questions. I would reconsider my score if these concerns are adequately addressed.

### Questions
> w1: Regarding the computation of gradient information, the description in Lines 230-232 specifies the use of "the activations in the penultimate layer," and Lines 262-263 mention freezing "all layers except the output head and the final decoder block." Was the use of gradients from more layers explored? Furthermore, would expanding the scope of gradient computation in this manner still ensure stable fine-tuning, or could it potentially introduce instability?

> w2: The manuscript introduces scalars α and β to integrate context and gradient information (Lines 236-245) and notes their "distinct but complementary patterns" (Lines 405-407). However, the rationale and process for selecting the optimal values for α and β are not sufficiently discussed. A more detailed analysis is needed to clarify how these critical hyperparameters were determined and how sensitive the method's performance is to their specific values.

>w3: The authors identify a "global anchor" in the attention map (Lines 373-377), evident as a prominent vertical line. Based on the proposed TokenSeek framework, could the authors provide an explanation for why this specific token emerges as a global anchor? Furthermore, given that your method ditches tokens based on a combined score, is this particular "anchor" token ever identified as less informative and subsequently ditched, or is it consistently preserved? Clarifying this would enhance the interpretability of the token selection process.

>w4: The insight that not all training tokens contribute equally to model fine-tuning is well-founded and reasonable. A similar phenomenon has been observed in LLM reasoning research [1,2]. Could the authors analyze the connections and distinctions between your findings and those from the reasoning domain?
>
>[1] Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning.
>
>[2] Demystifying Reasoning Dynamics with Mutual Information: Thinking Tokens are Information Peaks in LLM Reasoning.

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
4

### Summary
This paper proposes TOKENSEEK, a plug-in for memory-efficient fine-tuning that is instance-aware at the token level. The core idea is to (1) seek salient tokens per input by combining context signals (attention-derived scores) and optimization signals (gradients of the loss w.r.t. penultimate-layer activations), then (2) ditch the rest by disabling gradient computation for unselected tokens during backprop (retaining forward activations but avoiding their gradient/optimizer memory).

### Strengths
Combines attention-derived context and gradient saliency to rank tokens per example; empirically more effective and more stable than random selection. Reports 2.8 GiB peak in one setting and ~15% of full-token QLoRA peak on Llama-3.2-1B, while maintaining accuracy; cumulative with PEFT (LoHa/QLoRA). isualizations show complementary early-token bias from attention and late-token focus from gradients; helps explain the chosen subset.

### Weaknesses
Need a controlled knob table for each baseline (checkpointing, offloading, micro-batching, seq length, optimizer sharding) and the resulting peak+average memory to ensure apples-to-apples comparisons. Gradient-based scoring requires a partial backward pass; quantify this overhead per step and analyze action oscillations/instability of the selected set across training. Provide seed variance tables.

### Questions
For LoRA/QLoRA/LoHa/IA3 and full FT, which memory knobs (checkpointing schedule, ZeRO, offload, seq length, grad accumulation, activation precision) were enabled? Please add a per-run configuration table with measured peak+average memory. How often does the selected token subset change across epochs/steps? Any evidence of training instabilities when the subset shifts? Could you learn α/β or the token fraction from validation loss via a small controller?

### Soundness
2

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
TOKENSEEK introduces a universally applicable plugin for memory-efficient fine-tuning of large language models without altering their architecture. It adaptively identifies and retains only the most salient tokens per instance—based on contextual and gradient importance—while discarding less useful ones to reduce activation memory. This instance-aware token ditching achieves significant memory savings with stable or improved performance, offering a generalizable, architecture-agnostic approach to efficient fine-tuning.

### Strengths
Solid problem formulation  as fine-tuning large LLMs is highly memory-intensive, with activations contributing a major share of the cost.
And Innovative approach to integrates gradient information with context scores to capture a more holistic measure of token importance, addressing the limitation that context-based evaluation alone reflects only intra-sequence relevance, not fine-tuning contribution.


Some of the specific strength:
 - Architecture-agnostic design: The proposed plugin can be applied to any pretrained LLM without modifying its core architecture that is very practical.
 - Strong empirical results: Demonstrates substantial memory savings while maintaining or improving model performance across benchmarks. Very solid results in table 1, specifically with QLora.
- Interpretability and analysis: Offers clear insights into token-level importance and provides comprehensive analysis on how token ditching affects fine-tuning. Figure 4 and other analysis plots are insightful.
- Clarity and presentation: This paper is very well-written, logically structured, and easy to follow, with clear motivation and experimental validation.

### Weaknesses
- Scalability concerns: The token regrouping step—where tokens are sorted by importance and selectively included for backpropagation—may pose significant implementation and communication challenges in large-scale distributed fine-tuning setups. Synchronizing token importance scores and managing uneven token partitions across devices could offset some of the claimed memory savings.

- Complexity of integration: Although conceptually modular, integrating the method into existing large-scale MEFT pipelines may require non-trivial modifications to data loading and parallelism strategies.

 - Limited large-scale validation: Experiments are mostly conducted on moderate-sized models (<=3B) and datasets; the method’s stability and efficiency under massive multi-node training scenarios remain unverified.

- Selective update imbalance: Dropping gradients for less salient tokens could bias training if token importance is misestimated or unstable across iterations.

### Questions
- In Eq 3, the notation of i and j is a bit confusing. It might be more clear to express context score for ${t_j}$ token as $\Sigma_{i=1}^{n} A_{ij}$. To clarify that you are summing the attention scores across all rows for the given column j.
- Similarly the notation in eq 4 should indicate that the the gradient score is for  token $t_j$.

- More clarification on the token regrouping process and its practicality in distributed fine-tuning settings would strengthen the paper.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the limitation of current fine-tuning methods that largely overlook the information contained in individual training instances. The authors propose TokenSeek, a universal plug-in framework for Transformer-based models that performs instance-aware token selection and ditching. By selectively fine-tuning on important tokens identified via attention and gradient signals, TokenSeek reduces activation memory consumption while maintaining comparable performance.

### Strengths
- The paper is clearly written and generally easy to follow, with only minor typos.
- The proposed TokenSeek method is conceptually simple and practically implementable.
- Experimental results demonstrate competitive performance on multiple downstream tasks (e.g., QA, reasoning) across different LLM architectures such as LLaMA and Qwen.
- The experiments are comprehensive and include ablations and cross-task evaluations, which strengthen empirical credibility.

### Weaknesses
- Limited Novelty Clarification:
The main contribution—token importance–based selection—extends the previous TokenTune framework rather than introducing a fully new paradigm. While TokenSeek improves token importance estimation compared to random selection, the core ideas of memory reduction and generalizability are largely inherited from TokenTune. The paper should better articulate what is fundamentally novel about TokenSeek beyond methodological refinements.

- Comparison to Low-Rank and Partial-Tuning Methods:
On Line 835, the paper notes that TokenSeek and TokenTune incur ~11–15% more GPU hours than full-token tuning. In contrast, low-rank PEFT methods (e.g., LoRA) typically achieve both lower memory and faster training. Since TokenSeek inherits TokenTune’s extra computational overhead, the paper should clarify whether TokenSeek offers any non-trivial advantages over low-rank or partial-tuning approaches.

- Missing Baselines and Related Work:
Several recent sparsity-based PEFT methods are missing from the related work and experimental comparisons. As TokenSeek belongs to the PEFT family, adding one representative partial-tuning or sparse fine-tuning baseline (e.g., [1–3]) would provide a more complete evaluation and contextualization of the method’s contribution.


- Some minor typos
“as showin in Fig. 1 (a)” → shown; 
“Benifit from” → Benefit from; 
L313 “Unde” → Under; 
“achiving” → achieving; 
“TokenSeek achieve” → achieves; 
“LoRA/QLoRA achieves” → plural → achieve




[1] Scaling Sparse Fine-Tuning to Large Language Models

[2] Sparse Matrix in Large Language Model Fine-tuning

[3] The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks

### Questions
- Code-Domain Generalization:
The experiments cover QA and Open-Platypus datasets. How would TokenSeek perform on code-related datasets (e.g., CodeAlpaca, HumanEval)? Code data tends to have dense information where most tokens are important, which could reduce token sparsity efficiency. Including one experiment on a code dataset would better demonstrate TokenSeek’s generalizability.

- Source of Memory Savings:
Please clarify precisely where the memory savings originate. Since token selection is gradient-based, forward propagation must still process all tokens, and optimizer memory (e.g., Adam states) typically depends on trainable parameters rather than token count. How does TokenSeek achieve the reported activation memory reduction—through selective gradient storage, reduced backward pass, or another mechanism? A detailed explanation would strengthen the technical soundness.



At present, this appears to be a borderline paper (score ~5). The core idea is promising, but its novelty and advantage over prior TokenTune and LoRA-based methods need clearer articulation. If the authors can convincingly address the questions above—particularly regarding memory savings and broader generalization—I would be open to raising my score.

### Soundness
3

### Presentation
3

### Contribution
2
