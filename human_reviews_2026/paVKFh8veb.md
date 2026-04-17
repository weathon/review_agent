# MIDUS: Memory-Infused Depth Up-Scaling

- Decision: Reject
- Scores: 6, 2, 4, 2

## Abstract
Scaling large language models (LLMs) demands approaches that increase capacity without incurring excessive parameter growth or inference cost. Depth Up-Scaling (DUS) has emerged as a promising strategy by duplicating layers and applying Continual Pre-training (CPT), but its reliance on feed-forward networks (FFNs) limits efficiency and attainable gains. We introduce Memory-Infused Depth Up-Scaling (MIDUS), which replaces FFNs in duplicated blocks with a head-wise memory (HML) layer. Motivated by observations that attention heads have distinct roles both across and within layers, MIDUS assigns an independent memory bank to each head, enabling head-wise retrieval and injecting information into subsequent layers while preserving head-wise functional structure. This design combines sparse memory access with head-wise representations and incorporates an efficient per-head value factorization module, thereby relaxing the usual efficiency–performance trade-off. Across our CPT experiments, MIDUS exhibits robust performance improvements over strong DUS baselines while maintaining a highly efficient parameter footprint. Our findings establish MIDUS as a compelling and resource-efficient alternative to conventional FFN replication for depth up-scaling by leveraging its head-wise memory design.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces MIDUS (Memory-Infused Depth Up-Scaling), a method for Depth Up-Scaling (DUS) that replaces FFN layers with a "Memory Block". The main contributions include the Head-wise Memory Layer (HML), which assigns memory per attention head, and efficient storage mechanisms (PKM for keys, HIVE for values). The authors report that this method achieves performance comparable to or better than DUS baselines while adding significantly fewer trainable parameters and offering comparable or faster computation times.

### Strengths
1. MIDUS can achieve strong performance for DUS using siginifantly fewer parameters
2. Ingenious design of the memory block. Using two separate $K$ to reduce computational cost and parameter count.

### Weaknesses
1. Limited scope. The paper only explore the interleaving memory blocks. And other DUS policies are not discussed which may also enhance thier efficency.
2. New hyperparameter complexity. MIDUS introduces new and non-trivial design choices such as the memory size, which increases the difficulty of finding the optimal hyperparameters.

### Questions
1. The paper explicitly avoids stacking Memory Blocks. Was this simply out of scope, or did you find any evidence that stacking $M^{HML}$ blocks leads to instability?
2. The decomposition of $K_h$ into two parts for the two-dimensional search is an effective strategy. Has this approach been extended to higher dimensions, such as a three-dimensional search?

typo: line 81-82, "\\$H\\$"

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
4

### Summary
The paper proposes MIDUS (Memory-Infused Depth Up-Scaling), a drop-in alternative to standard depth up-scaling (DUS) that replaces duplicated FFN layers with Memory blocks. Each Memory block combines an attention layer without the output projection (Attn′) and a Head-wise Memory Layer (HML) that performs per-head product-key retrieval with an efficient value factorization (HIVE). The design preserves identity at initialization by zero-initializing memory values and routing the retrieved signal through a residual path, so the expanded model initially matches the base model’s outputs. Experiments on Llama-3.2-1B with 8 inserted blocks demonstrate lower perplexity and improved average zero-shot accuracy compared to strong DUS baselines, while reducing trainable parameters and training-time memory. Figure 1 contrasts DUS vs. MIDUS; Figure 2 (p.5) details the six-step Memory-block pipeline.

### Strengths
1. Clear, modular design (Memory block + HML + HIVE) that can be inserted wherever DUS would add FFNs; identity-preserving init is well motivated.

2. Consistent gains across CPT and SFT with lower parameter/memory cost than DUS, plus ablations and placement analysis.

3. Clarity & completeness: math formalization, stepwise diagram (p.5), and a reproducibility-friendly recipe; an anonymized code link is included.

### Weaknesses
0. Some claims, especially those related to the major motivation, that existing methods rely "on dense feed-forward networks", are not accurate. For example, papers [1][2] are using "mixture of depth" like a sparse module for up-scaling. There is no discussion on the difference between these works, and they were not included as baselines.  

1. All results use a 1B backbone. Claims about general-purpose LLM scaling would be stronger with a 7B-class (or larger) model and at least one instruction-tuned setting beyond Alpaca-GPT-4.

2. Iteration times are reported, but sensitivity to sequence length (e.g., 8k–32k) isn’t analyzed; PKM’s two-stage top-k might have different break-even points. (Tables 2–4 give per-iter stats only)

3. Helpful ablations are included, but further disentangling the impact of (i) Attn′ vs. full MHA, (ii) exact k and n choices, and (iii) HIVE’s parameterization per head would clarify where the gains come from.

4. The benchmark suite is knowledge-centric; evaluating on reasoning-heavy or long-context tasks would test whether HML helps beyond factual retrieval

[1] Raposo, David, et al. "Mixture-of-depths: Dynamically allocating compute in transformer-based language models." arXiv preprint arXiv:2404.02258 (2024).

[2] Tan Z, Dong D, Zhao X, et al. Dlo: Dynamic layer operation for efficient vertical scaling of llms[J]. arXiv preprint arXiv:2407.11030, 2024.

### Questions
1. Can you detail the difference/limitation of the related works I mentioned above, preferably conduct an experiment against them?

2. How does MIDUS-HML scale on 7B–13B backbones? Any obstacles (e.g., memory-bank thrashing) at larger widths? 

3. Can you report latency/flops scaling with sequence length and batch size (train & inference), and compare to DUS? Where’s the crossover vs. dense FFN?

4. How sensitive are results to k, n, and the per-head transform size in HIVE? Any head-importance-aware allocation strategies tried?

5. For CPT, do MIDUS and DUS baselines share identical data order, schedule, and optimizer hyperparameters? If not, please provide tuned-per-method tables and/or a unified ablation.

6.

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
2

### Summary
The paper proposes MIDUS, a novel method for scaling LLMs by increasing depth through memory-based rather than feed-forward expansion. MIDUS replaces these FFNs with Memory Blocks built around a Head-wise Memory Layer (HML), where each attention head maintains an independent memory bank for sparse retrieval. To further improve efficiency, the authors introduce Head-wise Implicit Value Expansion (HIVE), which factorizes per-head value spaces to preserve head alignment without redundant parameter storage. The design allows capacity to be added in a retrieval-based, head-aligned manner, effectively decoupling model quality gains from dense computation. Experiments on continual pre-training (CPT) and supervised fine-tuning (SFT) with Llama-3.2-1B demonstrate that MIDUS-HML consistently surpasses strong DUS baselines in both perplexity and zero-shot accuracy, while using fewer trainable parameters and less GPU memory.

### Strengths
- MIDUS achieves depth expansion through sparse retrieval rather than dense FFN projections, decoupling performance gains from the heavy parameter and activation costs of FFNs.
- MIDUS–HML achieves the low perplexity and high average zero-shot accuracy, particularly excelling in benchmarks such as CSQA, BoolQ, PIQA, and MMLU.
- HML assigns an independent memory bank per attention head, enabling selective retrieval aligned with head specialization, thereby minimizing cross-head interference compared to block-shared memories.

### Weaknesses
- The work lacks formal justification or theoretical analysis of why memory retrieval at head level leads to better generalization or gradient propagation.
- The paper would benefit from visualization or analysis of what the head-wise memories actually learn or retrieve—whether they store task-specific patterns, contextual cues, or token-level semantics.
- Since MIDUS replaces dense FFN expansion with sparse retrieval, it introduces the need to carefully determine memory size and placement, which may affect optimal scaling.

### Questions
- The authors fix the learning rate. Could the authors clarify whether this fixed rate was empirically tuned or simply adopted from earlier works?
- What underlying dynamics cause internal residual connections to weaken retrieval signals?
- What is the per-token inference overhead introduced by HML compared to standard FFNs?

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
3

### Summary
This paper investigates depth upscaling to enhance LLM performance with light continued pretraining or supervised fine-tuning. Since attention heads specialize differently and additional dense FFN layers are computationally heavy, MIDUS assigns an independent memory bank to each head, enabling head-wise retrieval also in FFN. In experiments, the depth-upscaling layers adapt quickly with only light continued pretraining or supervised fine-tuning and deliver better accuracy than baseline models on commonsense reasoning tasks. The method consistently outperforms prior work while using fewer parameters and achieving higher efficiency.

### Strengths
* The paper proposes a memory-based alternative to FFN replication, motivated by the head-independent representations in attention. Thus, it can be more efficient than prior depth-scaling approaches that use dense FFN layers, and it may be easier to learn due to the sparse, head-wise connections.

* The experiments report both accuracy and efficiency. It also appears that the paper compares fairly with prior work and consistently outperforms it.

* The method is easy to understand, and the presentation is clear.

### Weaknesses
* Efficiency. As I understand it, the method adds additional layers. Then, why does it use fewer parameters and less GPU memory compared to the original model? Also, for latency, does the paper measure prefilling time or decoding time?

* Task coverage. The experiments seem to focus on general-purpose commonsense reasoning. Could the authors also report results on harder domains such as code and math? Baselines may already perform well in these specialized areas, whereas depth upscaling trained on 2B web tokens may not transfer as effectively. In short, can depth upscaling still perform well (better performance than the original model) on code, math, or long-context tasks under the 2B web token CPT setup?

* Comparisons. Why is the same zero-shot accuracy repeated in Table 2 as in Table 1? How does depth upscaling with CPT/SFT compare to training the same total number of layers from scratch? How does the method perform—in both efficiency and accuracy—on larger models such as 3B or 7B?


* MIDUS layer design. In MIDUS, there appears to be no hidden-state mixing across heads in the feedforward layer; the whole hidden-state mixing happens only in the initial projections that produce Q/K/V. Do the authors think this could introduce any implicit limitations?

### Questions
Please see above

### Soundness
2

### Presentation
2

### Contribution
2
