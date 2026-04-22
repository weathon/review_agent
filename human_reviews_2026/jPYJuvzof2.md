# MemMamba: Rethinking Memory Patterns in State Space Model

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
With the explosive growth of data, long-sequence modeling has become increasingly important in tasks such as natural language processing and bioinformatics. However, existing methods face inherent trade-offs between efficiency and memory. Recurrent neural networks suffer from gradient vanishing and explosion, making them hard to scale. Transformers can model global dependencies but are constrained by quadratic complexity. Recently, selective state-space models such as Mamba have demonstrated high efficiency with $O(n)$ time and $O(1)$ recurrent inference, yet their long-range memory decays exponentially. In this work, we conduct mathematical derivations and information-theoretic analysis to systematically uncover the memory decay mechanism of Mamba, answering a fundamental question: what is the nature of Mamba’s long-range memory and how does it retain information? To quantify key information loss, we further introduce horizontal–vertical memory fidelity metrics that capture degradation both within and across layers. Inspired by how humans distill and retain salient information when reading long documents, we propose MemMamba, a novel architectural framework that integrates state summarization mechanism together with cross-layer and cross-token attention, which alleviates long-range forgetting while preserving linear complexity. MemMamba achieves significant improvements over existing Mamba variants and Transformers on long-sequence benchmarks such as PG19-PPL and Passkey Retrieval, while delivering a 48\% speedup in inference efficiency. Both theoretical analysis and empirical results demonstrate that MemMamba achieves a breakthrough in the complexity–memory trade-off, offering a new paradigm for ultra-long sequence modeling. The code and the pre-trained models will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces a new metric to evaluate the memory forgetting degree as layer and time-step going deeper and validate that their architecture with memory recalling beats baselines in long-context benchmarks.

### Strengths
The work conduct extensive experiments to show their superiority with other long-context Mamba methods.

### Weaknesses
Experiments: model scale is limited to 100+ M. Conclusion at this scale is hard to transfer and very variable.

Writing: paper writing is not clear, especially Method Sec.. and many new terminology is unnecessary (like vertical-horizontal, not see any necessity to replace the vanilla description of layer/timestep).

### Questions
1. can you show the effectiveness of your method in 1B+ model? (pretrain a 1B model or finetune a even larger model)?

### Soundness
3

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
The paper investigates the memory decay problem in state-space models with a focus on the Mamba architecture and presents MemMamba, a memory-augmented variant designed to improve long-range information retention while preserving linear complexity. The authors conduct both mathematical and information-theoretic analyses to explain Mamba’s exponential memory decay and introduce two metrics, Expected Token Memory Fidelity (ETMF) and Expected Cross-Layer Memory Fidelity (ECLMF), which quantify information loss across tokens and layers. MemMamba incorporates a Note Block for dynamic state summarization together with sparse cross-token and cross-layer attention to restore salient information. Experiments on long-sequence benchmarks such as PG19 and Passkey Retrieval demonstrate improved stability and efficiency compared with Mamba and Transformer baselines.

### Strengths
- The topic is timely and relevant given the growing interest in SSM-based long-sequence modeling.
- The paper presents a clear analysis of memory decay and provides intuitive metrics (ETMF / ECLMF) to visualize horizontal and vertical information loss.
- The proposed architecture is well-motivated and empirically improves robustness on long-context benchmarks such as PG19 and Passkey Retrieval.
- Experimental presentation and ablations are thorough, and the writing is generally clear.

### Weaknesses
**1. Methodological novelty is minimal within the hybrid SSM + attention family**

MemMamba combines a selective state-space model with sparse cross-token and cross-layer attention, but this pattern has already been explored in Compressive Transformer, RetNet, LongMamba, and other recent variants. The Note Block functions similarly to prior compression or summarization modules, offering only minor procedural differences rather than a genuinely new architecture.

**2. Theoretical analysis lacks rigor and depth**

The paper’s mathematical treatment mostly restates well-known properties of linear recurrent systems, such as exponential decay under $|A| < 1$. The proposed ETMF and ECLMF metrics are intuitive but not derived from principled information-theoretic foundations, and their empirical correlation with downstream performance remains unclear.

**3. Experimental gains are not sufficiently validated against strong baselines**

While results on PG19 and Passkey Retrieval are promising, comparisons exclude important contemporaries such as Mamba-2, RetNet, and RWKV. Parameter counts and training setups also differ, leaving uncertainty over whether improvements stem from the proposed mechanisms or from implementation choices.

**4. Claimed efficiency improvements are weakly supported**

The reported 48% inference speedup is measured on a single GPU under a limited setup. No analysis is provided for scaling behavior or memory usage under multi-GPU or distributed inference conditions, making the efficiency claim difficult to generalize.

**5. Writing occasionally overstates the contribution**

Phrases like “breakthrough” and “new paradigm” exaggerate the paper’s significance given its incremental contribution. A more balanced presentation would strengthen credibility and highlight the genuine empirical strengths.

### Questions
See the weaknesses.

### Soundness
2

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
This paper proposes MemMamba, a memory-augmented extension of state-space models for long sequence modeling. The authors systematically analyze memory decay patterns in Mamba with the novel horizontal–vertical memory fidelity metrics. They also introduce the state summarization with cross-layer/cross-token attention mechanisms to mitigate information loss over extended contexts. The evaluation are mainly focused on diverse tasks such as language modeling (PG19), synthetic Passkey Retrieval, and cross-document reasoning, demonstrating improvements over serveral baselines.

### Strengths
1. This paper provides a systematic analysis of memory decay in Mamba by mathematical derivations and the memory fidelity metrics (ETMF, ECLMF) in both main text and Figure 4.
2. Paper is clearly written for the most parts, with a good contextualization within the SSM and long-sequence modeling literature.

### Weaknesses
1. Although the paper claims to offer a fundamentally “new paradigm” for ultra-long sequence modeling, the MemMamba approach can be interpreted as a synthesis and adaptation of several established ideas (memory summarization, cross-layer attention, and sparsity in attention), rather than introduction of entirely unprecedented architectures. The degree of originality, while respectable, may be somewhat overstated in the positioning.
2. There are a few areas for improvement in the paper presentation: for example, the clarity of Figure 3 and the font size within the figures could be further adjusted. Also, are lines 456-457 redundant with a previous paragraph? They could be removed.
3. The description of the thresholding mechanism (e.g., for triggering note-taking and cross-attention) in Section 4 lacks a fully articulated rationale for the chosen thresholds ($\tau_1$, $\tau_2$). Moreover, there is no sensitivity analysis or ablation study on their values—an important omission since these could significantly influence empirical outcomes.
4. The paper lacks a comparison of GPU memory usage between the proposed model and baseline models. Also, regarding Section 5.2 'Efficiency,' why are specific results omitted, with only a comparison between MemMamba and Transformer being presented? 
5. Potential missing related work or baseline models that should be compared in the paper:

[1] Wang, Qianning, He Hu, and Yucheng Zhou. 'Memorymamba: Memory-augmented state space model for defect recognition.' arXiv preprint arXiv:2405.03673 (2024).

[2] Gui, Yiyu, et al. "EEGMamba: Bidirectional state space model with mixture of experts for EEG multi-task classification." _arXiv preprint arXiv:2407.20254_ (2024).

### Questions
See weaknesses

### Soundness
3

### Presentation
2

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
The paper analyzes why selective state space models like Mamba forget over distance, formalizes this with a horizontal–vertical memory fidelity framework for token-level and cross-layer information loss, and shows that Mamba’s long-range contributions decay exponentially. It then introduces an architecture that couples lightweight state summarization (“Note Block”) with cross-token and sparsely triggered cross-layer attention to retain salient signals while preserving linear time/space complexity.

### Strengths
1. “Note Block” state summarization + cross-token and sparse cross-layer attention improve long-range recall while keeping O(n) time.

### Weaknesses
1. Passkey Retrieval is a relatively simple task on in-context retrieval, and it is better to try a more difficult mutli-key-value retrival task., such as Phonebook and RULER.
2. Missing technical details and ablations. See Questions.

### Questions
1. The Note Block and MemMamba block rely on importance scores and dual thresholds (τ₁ for “take note,” τ₂ for cross-token attention). Are τ₁/τ₂ learned, scheduled, or fixed? Are these threshoulds sensitive for the model's performance?
2. What exact priority metric is used for Note block? Please compare FIFO vs priority on quality/latency.

### Soundness
2

### Presentation
2

### Contribution
2
