# ChainGPT: Dual-Reasoning Model with Recurrent Depth and Multi-Rank State Updates

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Large language models, constrained by the fixed-depth Transformer architecture, struggle to solve complex reasoning tasks in an end-to-end manner. Existing approaches, such as Chain of Thought, improve reasoning depth to some extent but rely heavily on natural language generation, with computational costs increasing rapidly as the length of the generated sequence grows. To address these limitations, we propose ChainGPT, a dual-reasoning model that shifts reasoning into latent computational space. Within each layer, ChainGPT employs multi-substep state updates combined with state-guided sparse attention, enabling deep local computation and efficient long-range modeling without quadratic costs. Across layers, recurrent depth approach iteratively refine latent states, supported by adaptive training and stopping strategies that balance reasoning depth against computational budget. Theoretically, we show that ChainGPT can, in principle, simulate general computation, and empirically it delivers consistent improvements over comparable models, including on reasoning tasks that remain challenging for existing systems. By unifying efficiency and reasoning ability, ChainGPT provides a principled foundation for next-generation language models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes ChainGPT, a dual-reasoning architecture that pushes reasoning from token generation into latent space via (i) RWKV-Product multi-substep state updates inside each layer and (ii) State-Guided Sparse Attention (SGSA) with sliding windows plus periodic anchor. It uses a recurrent-depth core with entropy-based early stopping across layers.

### Strengths
* **Clear architectural idea with theory hooks**: The dual mechanism (multi-substep “diagonal + rank-M” updates + sparse global anchors) is well motivated and analyzed (rank expansion, MQAR solvability, Turing-completeness under idealizations).

* **Efficiency claims backed by complexity and ablations**: SGSA reduces attention cost to $O(T(W+T/G))$ and tracks full-global attention perplexity on PG-19 when using periodic anchors.

* **Empirical improvements at matched scale**: On LM-Eval tasks, ChainGPT-0.5B/1.5B outperform Qwen2.5 models of the same size, respectively. The sub-step and recurrence ablations are thorough.

### Weaknesses
* **Novelty vs prior recurrent/hybrid work is somewhat incremental**: RWKV-Product extends RWKV-7 with LoRA-style multi-substeps. Many components (looped/recurrent depth, window+anchors) resemble existing hybrid archs. Thus, positioning versus models like DeltaProduct, Jamba/Samba, Mamba-2, and HRM could be sharper.

* **Claims on “hard tasks” need stronger rigor**: The ARC-AGI-1 (38.6%), Sudoku-Extreme (54.4%), and Maze-Hard (77.4%) numbers are promising but depend on small (30M) models and bespoke setups; fairness vs large LMs and exact eval pipelines deserve more detail.

* **Theoretical claims hinge on idealized assumptions**: Turing-completeness requires unbounded memory/steps and arbitrary precision; practical implications for finite precision and training stability remain unclear.

* **Paper structure can be polished**: From my reading perspective, the organization of Sections 3 and 4 could be improved to better align with the proposed pipeline and enhance overall clarity. For example, these two sections could be merged into a single, cohesive section. In that structure, the current Section 4.1 could serve as an overarching overview, explaining where the chain-block fits within the entire pipeline and its overall function. Subsequent subsections could then provide a more detailed introduction and analysis of the chain-block itself.

* **Illustrations can also be improved**: The current Figures 1, 2, and 3 each contain limited information, and presenting them separately leads to fragmented understanding. It would be more effective to integrate all three into a single, comprehensive pipeline diagram. Additionally, the current figures appear blurry.

### Questions
* **SGSA complexity & memory**: Can you quantify the runtime and activation memory of SGSA vs dense attention across $T∈[4k,32k]$ and report end-to-end wall-clock with/without anchors, beyond the single-GPU microbenchmarks? Also include KV-cache implications.
* **Efficiency Analysis**: You report a comparison of computation time. Could you provide a more comprehensive efficiency analysis that also covers training GPU-hours and inference latency?
* **Early-stopping robustness**: The entropy-diff rule uses k and threshold τ. How sensitive are quality and compute to these hyperparameters across tasks?
* **Ablation on RWKV-Product substeps and rank**: Can you include an ablation removing either the multi-substep (K) and the rank-M updates in RWKV-Product?

### Soundness
3

### Presentation
2

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
This paper
- Proposes the ChainGPT architecture and introduces RWKV-Product. The architecture proposed can achieve good performance on reasoning tasks compared to Qwen.
- Mathematically proved that the proposed architecture has superior expressivity and is Truing complete under ideal conditions.

### Strengths
I think the paper is sound provided the combination of math proof and supportive experiments.

### Weaknesses
W1: Though you mentioned Geiping's work in intro, I don't find any comparison between existing recurrent models and yours, which is hard for me to judge your contribution to recurrent models.

### Questions
Q1: In Chapter 3, I do not find the definition of "multiple sub-steps reasoning".

Q2: Can you calculate the number of params explicitly so that I can compare to the normal models?

Q3: There is a reference error in line 252.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces ChainGPT, a dual-reasoning architecuture. In one block of ChainGPT, RWKV-Product achieves intra-layer communication, and SGSA achieves efficient long-range attention. The block design is aimed for "internal multi-step reasoning" (RWKV-Product) and "sparse aggregation" (SGSA). Across blocks, ChainGPT uses a recurrent paradigm to interatively refine internal states, with dynamic early stopping using entropy. Theoretical discussion and empirical experiments are conducted.

### Strengths
1. **Clear Positioning**: The introduction provides a clear case for why fixed-depth Transformers and current architectural hybrids suffer from xpressive and computational limitations for deep reasoning.

2. **Architecture Innovation with Theoretical Support**: This paper proposes a grounded achitecture innovation, including (1) intra-layer multi-substep reasoning using RWKV-Product, theoretically proven to expand representational power; (2) inter-layer recurrent refinement, with a theoretically motivated early stopping mechanism based on entropy.

3. **Extensive Details**: Many ablations are given showing the effectiveness of each component, and details provided in the Appendix further improve the relia

### Weaknesses
1. **Concern on Component Integration**: ChainGPT appears to be an ensemble of several independent components, each building upon or modifying existing prior work. This approach could be argued to undermine the central academic contribution by suggesting the performance gains are primarily due to a complex engineering aggregation rather than a singular, fundamental architectural breakthrough.

2. **Missing Discussion about Soft Thoughts**: The paper lacks a critical discussion comparing ChainGPT's approach to existing literature [1-3] that utilizes dense gist tokens (termed "soft thoughts"). These works also achieve reduced computational cost and perform reasoning via implicit states, making a comparison essential to fully delineate and underline the unique significance of ChainGPT's methodology.

3. **SGSA Issue**: I'm a little concerned about the SGSA experiments. In Table 4, it seems that even a fully localized sliding window attention can achieve comparable PPL with long contexts, which raise doubts about the evaluation reliability. In practice, the selection of window size $W$ and anchor interval $G$ for all models are all set to 512 and 64, lacking empirical justification.

4. **Experimental Issues**: Few related methods are compared with ChainGPT. Moreover, Since ChainGPT is framed as a "dual-reasoning" model, it is crucial that it be compared against SOTA CoT reasoning models. For example, Qwen3-1.7B can achieve an accuracy of $>0.8$ on the ARC-Challenge task, which significantly outperforms ChainGPT's $0.3$ accuracy, severely compromises the contribution of the proposed reasoning capabilities.

> [1] Training Large Language Models to Reason in a Continuous Latent Space.
> [2] CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation.
> [3] LightThinker: Thinking Step-by-Step Compression.

### Questions

### Questions
1. **Hyperparameter Sensitivity (Weakness 3 Related)**: The SGSA module relies on window size $W$ and anchor interval $G$. Can you provide guidance on selecting these hyperparameters in practice?

2. **Empirical Comparison with Closest Works (Weakness 4 Related)**: Could the authors provide experimental results or more detailed discussion comparing ChainGPT’s dual-reasoning approach to releated methods? What are the measured gains or trade-offs in similar settings?

### Soundness
2

### Presentation
3

### Contribution
3
