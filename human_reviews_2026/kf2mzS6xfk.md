# Decoupling The "What" and "Where" With Polar Coordinate Positional Embedding

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
The attention mechanism in a Transformer architecture matches key to query based on both content---the *what*---and position in a sequence---the *where*. We present an analysis indicating that what and where are entangled in the popular rotary position embedding (RoPE). This entanglement can impair performance particularly when decisions require independent matches on these two factors. We propose an improvement to RoPE, which we call *Polar Coordinate Position Embedding* or *PoPE*, that eliminates the what-where confound. PoPE is far superior on a diagnostic task requiring indexing solely by position or by content. On autoregressive sequence modeling in music, genomic, and natural language domains, Transformers using PoPE as the positional encoding scheme outperform baselines using RoPE with respect to evaluation loss (perplexity) and downstream task performance. On language modeling, these gains persist across model scale, from 124M to 774M parameters. Crucially, PoPE shows strong zero-shot length extrapolation capabilities, whereas RoPE's performance degrades significantly on longer sequences at test time without fine tuning or the use of position-interpolation methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce Polar Coordinate Position Embeddings (PoPE), an improvement to RoPE which uses polar coordinates to encode positional information. The primary benefit is that by disentangling content and position in RoPE through the use of angle and magnitude information, there is better mixing of information that can be performed by the model in a context-specific manner. This is demonstrated through superior performance in various sequence modeling tasks and zero-shot length extrapolation.

### Strengths
- The method is motivated mathematically and intuitive to follow.
- The method is simple and not architecture dependent (aside from being used in Transformers), thus I am under the impression it can be integrated with any existing method on this front.
- The experiments are comprehensive and demonstrate the benefits of PoPE in relation to RoPE.

### Weaknesses
- While the method does show improvements relative to RoPE in the context of the models being studied, some standard benchmarks could be included (ex. LongBench, RULER, etc.) to better show some of the long-context scaling behaviour beyond just perplexity. Additionally, results after instruction-tuning the model would be appreciated to better highlight the benefits of the method.
- One small consideration is that given the additional parameters that are stored for RoPE, this could affect cache optimization at inference time. I would expect this to potentially lead to come limitations on this front, and this could also lead to some problems with current cache optimization techniques or inference-time scaling techniques (ex. attention sinks or meta-tokens).

### Questions
See above.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Polar Coordinate Positional Embedding (PoPE), a modification of the widely used Rotary Position Embedding (RoPE) for Transformer architectures. The authors argue that RoPE entangles content (“what”) and position (“where”) information in query–key attention, which may hinder performance on tasks that require independent reasoning about these two aspects.

PoPE reformulates RoPE in polar coordinates, where the magnitudes (content) and phases (position) are explicitly separated. The magnitudes of queries and keys are made positive via a softplus transformation, and the position is represented purely by the phase terms. This eliminates the cross-term between the query/key phases that causes what–where coupling in RoPE.

Empirically, PoPE shows strong improvements across multiple domains: 1. Indirect Indexing diagnostic task: 95% vs 11% accuracy (RoPE) 2. Sequence modeling: lower NLL on music (JSB, MAESTRO) and genomic (HRG) datasets; 3. Language modeling: reduced perplexity on OpenWebText across 124M–774M parameter scales 4. Downstream and extrapolation tasks: better zero-shot accuracy and robust long-context generalization without interpolation.

### Strengths
1. The paper provides a geometric explanation of RoPE’s limitation, mixing content and position, and introduces PoPE as a principled disentanglement.
2. The polar-coordinate derivation clearly isolates the “interaction term” $\phi_k-\phi_q$ and justifies its removal to achieve independence between what/where components.
3. Experiments span diagnostic synthetic tasks, structured symbolic domains (music, genomics), and large-scale language modeling, showing consistent, domain-general gains.
4. The length extrapolation results (up to 10× longer sequences) are compelling and address a long-standing weakness of RoPE-based models.
5. PoPE can be dropped into any Transformer with almost no code change and negligible computational overhead (only one extra multiplication).

### Weaknesses
1. The “disentanglement” argument is mostly qualitative. There is no formal analysis or proof that the removed phase interaction necessarily leads to improved generalization.

2. Softplus is heuristic. While used for ensuring positive magnitudes, its necessity and effect are not theoretically grounded beyond empirical ablation.

3. It would be useful to see comparisons to YaRN, ALiBi, or other strong length-extrapolatable positional encodings, not just RoPE.

4. While conceptually appealing, it remains unclear how the model behaviorally separates content vs. position in downstream reasoning.

### Questions
1. Can you formalize why removing the $\phi_k-\phi_q$ interaction improves expressivity or inductive bias, beyond the empirical results?
2. Did you test other non-negative mappings (e.g., ReLU, square, exponential)? How sensitive are results to the choice of activation?
3. Could this reintroduce a form of coupling or shift-dependent bias that undermines the core disentanglement principle? 
When $s \approx t$, the query and key correspond to nearby positions in the sequence, and one would expect 
$\cos((t - s)\theta_c) \approx 1$ in Eq.(5). 
However, introducing a learnable bias term $\delta_c$ could force 
$\cos((t - s)\theta_c + \delta_c) \neq 1$, 
which appears counterintuitive and may distort locality or positional consistency.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
In this paper, the authors rewrite RoPE positional encoding in polar coordinates and discover that the phase of the cosine function in RoPE is related not only to relative position ("where," i.e., $(s - t)\theta_c$ in Equation (2)), but also to the query/key themselves ("what," i.e., $\phi_{k_{sc}} - \phi_{q_{tc}}$ in Equation (2)). The authors argue that the coupling effect of 'what' and 'where' on the phase of the cosine function may be detrimental to model performance. Therefore, they propose PoPE positional encoding, which removes the 'what' term from the cosine function's phase, so that the phase depends solely on relative positional relationships. In their experimental setup, the authors compare RoPE and PoPE on tasks such as Indirect Indexing, synthetic music sequence modeling, human genome sequence modeling, language modeling and downstream evaluation, as well as length extrapolation during testing.

### Strengths
(1) PoPE shows an improvement over RoPE on the Indirect Indexing synthetic task. This task involves string sequences where, given a source character and an offset relative to that character, the model needs to retrieve the target character obtained by applying the offset to the source. Experimental results demonstrate that, compared to RoPE, PoPE—whose cosine function phase depends only on relative positional relationships—enables more accurate localization for this type of task.  

(2) PoPE demonstrates better length extrapolation ability than RoPE during testing. According to Figure 2 in the paper, when models are trained on sequences of length 1024 and then evaluated on sequences up to 10240 in length, PoPE’s language modeling perplexity loss remains nearly unchanged. In contrast, RoPE shows a noticeable increase in perplexity, highlighting PoPE’s advantage in handling longer sequences against RoPE.

### Weaknesses
The paper lacks citations and discussion of works with very similar formulations. This work shares an almost identical formulation with [1] and complex version of [2], yet it neither cites nor discusses the relationship between them. Additionally, the improvements appear marginal, and there is no efficiency/speed comparison provided.

Citations:
1. https://arxiv.org/abs/2202.08791
2. https://arxiv.org/abs/2307.09270

### Questions
See the weaknesses part.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents an analysis of the popular Rotary Position Embedding (RoPE), arguing that it entangles content ("what") and position ("where") information within the attention mechanism. The authors hypothesize that this entanglement impairs model performance, especially on tasks requiring independent matching of these two factors. To address this, they propose Polar Coordinate Positional Embedding (PoPE), a novel method that decouples content and position by re-framing queries and keys in polar coordinates, where content is mapped to magnitude and position to phase.

### Strengths
The paper offers a clear and insightful diagnosis of a fundamental limitation in RoPE—the entanglement of content and position. The proposed PoPE method is theoretically clean, directly addressing the identified issue with scientific taste. The central claims are backed by strong experimental evidence across diverse domains (synthetic, language, music, genomics) and model scales. In particular, the method's ability to achieve robust zero-shot length extrapolation without special tricks is highly compelling and demonstrates significant value.

### Weaknesses
1. In its current implementation, PoPE doubles the K/V cache memory footprint, which is a serious practical barrier for training and deploying large-scale models.
2. The length extrapolation evaluation lacks a direct comparison to RoPE augmented with established extension techniques like YaRN or PI, making it difficult to assess its relative advantage over existing SOTA solutions.
3. The paper does not deeply investigate the role or the learned patterns of the `δc` bias term, despite showing it is critical for performance.
4. On standard language modeling benchmarks like OpenWebText, the performance improvements from PoPE, while consistent, are relatively modest in absolute terms.
5. The finding that different `δc` initialization strategies benefit in-distribution vs. extrapolation performance suggests the method's optimal configuration may be task-dependent, potentially complicating its use.

### Questions
1. Could you estimate the potential throughput (TFLOP/s) overhead of a memory-optimized PoPE kernel compared to a standard Flash Attention implementation?
2. How does PoPE's performance on the PG-19 length extrapolation task compare against a RoPE baseline that has been enhanced with YaRN?
3. Could you provide a visualization of the learned `δc` values to reveal if any structural patterns emerge across layers or frequencies?
4. Is RoPE's failure on the "Indirect Indexing" task highly sensitive to the choice of the `base` frequency `θ`? Would changing `θ` mitigate this failure?
5. The softplus function is crucial for decoupling. Did you experiment with other non-negativity constraints (e.g., ReLU), and how did they perform? This would help clarify its exact role.
6. The evaluation of long-context capabilities is narrow, resting on perplexity on a single benchmark (PG-19). This assesses modeling fluency but fails to test functional abilities like long-range information retrieval or QA, which are crucial for substantiating the paper's primary contribution.

### Soundness
2

### Presentation
2

### Contribution
2
