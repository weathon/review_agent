# Change of Thought: Adaptive Test-Time Computation

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 4, 6

## Abstract
Standard Transformers apply a fixed amount of computation to every token, limiting their expressive power, while more powerful iterative approaches often introduce significant architectural complexity and cost. We introduce Fixed-Point Self-Attention (FPSA), a parameter-free, drop-in replacement for self-attention that enables a model to adaptively ``think longer'' by iteratively refining each layer's representations to a fixed point. We train this recurrent process end-to-end using implicit differentiation, ensuring that memory usage during training and inference remains constant and identical to a standard Transformer layer, regardless of the number of refinement steps. Without adding any parameters, FPSA significantly improves strong baselines like BERT-Base and ELECTRA-Base on the GLUE and SQuAD v2.0 benchmarks. We demonstrate similar consistent gains for vision (ViT-B/16) and vision-language models, achieving accuracy improvements of up to 20\%. This performance boost comes at a modest computational cost: a median of 3--6 refinement steps results in a $\approx1.6\times$ GFLOPs and $\approx1.3-1.4\times$ latency overhead compared to an equivalent BERT-Base model. Analysis shows FPSA dynamically allocates compute to challenging inputs and converges to stable fixed points. Furthermore, integrating FPSA into language models improves performance on complex reasoning tasks like GSM8K, BBH, and LogiQA. Ultimately, FPSA bridges the gap between fixed-computation and iterative reasoning, offering a powerful building block that adaptively allocates compute while preserving architectural simplicity.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Fixed-Point Self-Attention (FPSA), a parameter-free and computationally adaptive replacement for the standard self-attention mechanism in Transformers. Instead of applying a fixed number of transformations per token, FPSA iteratively refines the attention outputs until convergence to a fixed point, allowing “adaptive computation” per token or per head. The method uses implicit differentiation to train efficiently, maintaining a constant memory footprint regardless of iteration depth. Empirical results show notable improvements across multiple domains: NLP (GLUE, SQuAD v2.0), vision (ImageNet, image restoration), and multimodal tasks, as well as reasoning benchmarks for large language models (GSM8K, BBH). The approach claims up to +20% relative accuracy improvements with ~1.3–1.6× compute cost.

### Strengths
1. The paper evaluates FPSA across multiple settings — encoder-only Transformers, decoder-only LLMs, and vision/multimodal models — showing consistent gains.

2. The authors provide a clear convergence analysis showing contractivity of attention mappings under spectral normalization and pre-LN. The implicit differentiation approach is rigorously justified.

### Weaknesses
1. While appendices provide algorithm sketches, more implementation-level specifics (e.g., PyTorch pseudocode, convergence thresholds per dataset, training schedules) are needed for full reproducibility.

2. The main baselines (BERT, ELECTRA) are standard, but the paper does not compare against other recent adaptive computation methods (e.g., ACT, MoD, or LayerDrop) on exactly matched compute budgets.

3. The ablations focus on iteration counts and convergence, but lack sensitivity studies for hyperparameters such as spectral norm bound σ, halting threshold ϵ, or gradient clipping T.

4. Results for 7B models are promising but brief; the paper would benefit from discussion of FPSA behavior at larger scales (70B+) or under distributed inference constraints.

5. For long-context tasks, does FPSA’s advantage hold beyond 8k tokens (e.g., 32k)?

### Questions
Please see the weaknesses.

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
The paper proposes Fixed-Point Self-Attention (FPSA)—a parameter-free, drop-in replacement for self-attention that iteratively refines the attention outputs inside each layer until a fixed point, trained end-to-end via implicit differentiation so the memory footprint is constant w.r.t. the number of refinement steps. Empirically, swapping standard attention for FPSA improves strong, size-matched baselines (e.g., BERT-Base, ELECTRA-Base) on GLUE/SQuAD, shows gains for ViT and VL models, and yields small but consistent boosts when integrated into 7B decoder-only LLMs (LLaMA-2/Mistral) on GSM8K/BBH/LogiQA—without adding parameters. Overhead is modest (median ~3–6 inner steps, ~1.6× GFLOPs and ~1.3–1.4× latency vs. BERT-Base).

### Strengths
1. The proposed structure is simple and maintains constant memory. Iterative refinement within the attention sublayer trained via implicit differentiation; avoids storing the inner unroll and heavy checkpointing. Architectural simplicity is preserved. 

2. The proposed structure improves the performance with the same parameters.  It improves size-matched encoder baselines on GLUE/SQuAD and shows benefits for ViT/VL, supporting generality.

3. The writing of the paper is clear. It clearly conveys the main idea of the new structure.

### Weaknesses
1. The main idea of the proposed iterative structure is to scale the computation. In other words, the performance gain comes with the increased training and inference computation. The training cost is not reported in the current work. The inference computation comparison is missing. Almost all the results in the paper show that the proposed structure performs better than vanilla attention with more computation. However, a fair comparison is to constrain the computation budget of both models, which is missing in the current work.

2. The baselines for the proposed algorithm are not sufficient. The ''iterative'' or ''looped'' structures are widely proposed in the LLM community, .e.g, the looped transformer [1]. Why we only iterate the attention calculation instead of the other parts or the whole network? Such a comparison is not presented.

3. The paper considers some non-causal transformers, .e.g., ViT. However, a line of works shows that pure attention iteration suffers from low-rankness [2]. This is exactly the proposed structure does. Whether the proposed structure also suffer from the low-rankness? If not, please explain the reason.

[1] Giannou A, Rajput S, Sohn J, et al. Looped transformers as programmable computers[C]//International Conference on Machine Learning. PMLR, 2023: 11398-11442.

[2] Dong Y, Cordonnier J B, Loukas A. Attention is not all you need: Pure attention loses rank doubly exponentially with depth[C]//International conference on machine learning. PMLR, 2021: 2793-2803.

### Questions
Please see the weakness part.

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
This paper proposes Fixed-Point Self-Attention (FPSA), a new Transformer-based architecture based on fixed-point iteration. This pushes the individual layer to be in fixed point, and this basically works like looping while value state is fixed. The authors extensively experimented across various settings to support FPSA's superiority.

### Strengths
- The proposed approach seems quite reasonable and good, based on fixed point iteration method.
- This paper has comprehensive analysis and results, to show how FPSA actually works and the performance under various downstream tasks.

### Weaknesses
- First, I feel like the overall presentation could be much enhanced. Some useful explanation and experimental results are hidden in Appendix parts (e.g., Fixed-point iteration is not explained in main body properly, some qualitative results as well). It would be good to re-place contents clearly for the reader.
- As the author mentioned, this mechanism seems having high relation to recursive / looped transformer architecture. There is lack of discussion to recent papers. And I'm curious about comparison to them. For example, comparison with Figure 1c or 1d, and ablation results without fixing value state.
- All layers seem like being updated by fixed-point iteration. What will happened if you select certain layers only? Some redundancy can be existed like current architecture, but this pushed too much for all layers.
- With this paradigm, can pruning be more critical? I feel like redundancy gets disappeared for each layer, which means that we can fully leverage the "depth" of models while loosing some chances to prune.
- Selecting some layers would be the solution though, inference latency and throughput seem inefficient. Especially, how could you deal with batched inference? Some tokens should wait for the other incomplete tokens at certain depth always.

### Questions
See above Weakness parts.

### Soundness
3

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
This paper introduces Fixed-Point Self-Attention, a parameter-free, drop-in replacement for the standard self-attention mechanism. The core idea is to iteratively refining the attention alignment matrix within a single layer until it converges to a fixed point. The authors propose to train end-to-end using implicit differentiation, which keeps memory usage constant.

### Strengths
1. The paper offers a more granular approach to adaptive computation compared to prior work that typically repeats entire blocks.

2. The use of implicit differentiation is a major technical strength, making the iterative approach practical by avoiding the memory explosion that would occur with standard backpropagation through time. The compute-matched comparisons (Appendix G) is nice.

### Weaknesses
1. The claim of being parameter-free is a bit misleading. While FPSA adds no new learnable model weights, it introduces several crucial hyperparameters that require tuning: the convergence tolerance $\epsilon$, the maximum number of iterations $K_max$, and the gradient clipping threshold. The learned halting variant (FPSA-LH) further adds a small gating MLP and a ponder cost hyperparameter. The paper lacks a sensitivity analysis for these hyperparameters, which seem critical to the method's performance and efficiency.

### Questions
1. Can authors provide a trade-off between the additional number of layers required in simple transformer layer compare to the fixed-point approach proposed here to match the same loss? Mainly, having more iteration seems that should reduce the number of layers in total. But it has not been really discussed in terms of achieving the same or comparable loss.

2. Could authors elaborate why they decided to choose a static value across iterations? What happens if it also gets updated?

3. In Table 4, it doesn't seem that Self-Transformer is providing any helpful improvement, and the enhancement in SR could also be just due to the increase of FLOPs. Can authors compare the results in this table more fairly?

4. Minor typo: Eq 1, I think it should be concatenation over $h$. In Table 5, I think for Top1 acc, the underline should be for the first row.

5. A clarification question: What are the colors in right plot of Fig 3?

Suggestions:
I would suggest to include the compute-matched results of Appendix G in the main paper. It makes the point of using the approach more clear.

### Soundness
2

### Presentation
2

### Contribution
2
