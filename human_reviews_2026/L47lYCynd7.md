# PLA: The Optimal Path from Softmax Attention to Linear Models via KV Cache Compression

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Transformers, despite their remarkable sequence modeling capabilities, are fundamentally constrained by the quadratic complexity of Softmax attention and the unbounded growth of the key–value (KV) cache. Replacing Softmax attention with linear variants has emerged as a promising direction, yet existing approaches lack a systematic functional comparison with Softmax attention, clear error analysis, and a theoretically guided roadmap for improvement. 
In this work, we approach the problem from the perspective of KV cache compression and present a theoretically grounded pathway from Softmax attention to linear models.
Our analysis reveals five critical components: redundancy elimination, tokenizer-level quantization and positional information separation, positional information compression, inter-layer similarity, and multi-state decomposition. For each, we provide succinct theoretical justification, derive error bounds, and demonstrate equivalence to existing mechanisms. Building on this pathway, we introduce PLA, a linearized attention model that inherits pretrained weights and achieves state-of-the-art performance. Notably, PLA surpasses strong baselines such as MVA and GSA on multiple benchmarks while requiring only 80\% of the fine-tuning resources. Our findings provide both theoretical clarity and practical guidance for advancing linear attention, highlighting a principled route towards efficient and scalable alternatives to Softmax attention.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper, “PLA: The Optimal Path from Softmax Attention to Linear Models via KV Cache Compression,” aims to establish a theoretically grounded connection between Softmax attention and linear attention. The authors argue that existing linear models (e.g., Performer, GSA, MVA) lack a systematic theoretical bridge to Softmax attention and propose five compression-based theoretical principles that describe how to transform Softmax attention into a linear, fixed-state formulation.

### Strengths
1. Provides a systematic theoretical framework connecting Softmax and linear attention through the lens of cache compression.
2. The five derived principles are interpretable and consistent with known empirical behaviors (e.g., redundancy of KV cache, similarity between residual layers).
3. Offers a unified perspective integrating ideas from fast-weight models, memory compression, and linearized attention.

### Weaknesses
1. Limited conceptual novelty: The proposed “optimal path” mainly restates and integrates ideas already seen in pruning, cache compression, and linear attention. There is no fundamentally new attention mechanism.
2. Lack of quantitative efficiency evidence: Despite the focus on bounded-state inference, the paper does not report runtime, memory, or throughput comparisons.
3. Narrow experimental scope: Only one model scale (7B) and limited dataset diversity; unclear if the conclusions hold across architectures or larger LLMs.

### Questions
1. How is “optimal” defined in the paper title？Does it refer to minimal error, bounded memory, or some efficiency–accuracy tradeoff?
2. Have authors compared actual runtime and memory against RetNet or MVA implementations?
3. Are there cases where the proposed compression harms long-range dependency modeling?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents PLA (Path-optimized Linear Attention), a novel approach to linearizing Transformer attention mechanisms by framing the problem as KV cache compression. The authors identify five theoretical principles that form an optimal pathway from softmax attention to linear models. For each principle, the paper provides theoretical justification, error bounds, and demonstrates equivalence to existing mechanisms. The authors then implement PLA based on these principles, showing it can inherit pretrained weights and achieve state-of-the-art performance on multiple benchmarks while requiring only 80% of the fine-tuning resources compared to alternatives like MVA and GSA.

### Strengths
The primary strength is the paper's unique theoretical perspective, framing linear attention as a KV cache compression problem rather than just a kernel approximation issue.

The five theoretical principles provide a much-needed systematic framework for understanding the relationship between softmax and linear attention. The equivalence proofs showing how existing methods map to these principles are particularly valuable for the field.  Experimental validation is exceptionally thorough, with well-designed ablation studies for each principle that clearly demonstrate their individual contributions. The authors provide extensive benchmarking across multiple domains (text, speech), which strengthens the practical relevance of the findings. The PLA implementation demonstrates clear practical benefits, achieving superior performance with reduced fine-tuning costs - requiring only 8B tokens compared to 10B for MVA while surpassing it on multiple benchmarks.

The connection between tokenizer properties and attention compression is novel and insightful. The paper makes a compelling case for the importance of enhanced reading mechanisms. By demonstrating that reading mechanisms are as important as storage mechanisms, the authors provide a more complete understanding of linear attention's limitations and potential. The 80% fine-tuning resource reduction is a significant practical contribution that could substantially reduce the computational burden of adapting large language models to linear attention.

### Weaknesses
The paper could benefit from more detailed discussion of computational complexity trade-offs beyond the theoretical analysis. While the paper mentions memory and time comparisons in Table 5, a more comprehensive analysis of the computational trade-offs would strengthen the practical contribution. 

The comparison to other linear attention methods could be more systematic, particularly regarding training costs beyond fine-tuning. The discussion of how these principles might apply to different architectures (e.g., ViT, speech models) is limited. Some theoretical sections are quite dense and could benefit from additional intuition to make them more accessible to a broader audience. Additionally, the paper doesn't fully address the potential trade-offs between the number of vocabulary levels and performance. While it shows that two levels work well, a more comprehensive analysis of the performance/efficiency trade-off curve would help practitioners make informed implementation decisions.

### Questions
- The paper demonstrates PLA's effectiveness with Mistral-7B, but how well does the approach transfer to other model architectures (e.g., ViT, speech models, or larger models like Llama-3 70B)? Could you provide more details about architectural dependencies?
- Section 4.2 shows PLA requires 8B tokens for fine-tuning compared to MVA's 10B tokens, but what is the actual training time reduction? - - Given that PLA has higher memory usage (Table 5), is the overall computational cost actually lower?
- The paper focuses on linear attention for language models, but how would PLA perform on vision tasks where positional information is fundamentally different? Are there necessary modifications for non-sequential data?
- Could you elaborate on the "multi-level" aspect of PLA? The paper mentions two-level vocabulary decomposition, but the theoretical section discusses c levels. How does performance scale with more levels, and what are the practical limits?
- On autoregressive tasks, it would be better if you add comparison to more recent (Gated-)DeltaNet(as you have mentioned), Mamba-2, RWKV7(if possible) in table 6 as they are all RNN-like models.
- Id like to see other reviewers' opinions before recommending acceptance.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Presents a theoretically grounded pathway from Softmax attention to linear models through five principles (redundancy elimination; tokenizer-level quantization & positional separation; positional compression; inter-layer similarity; multi-state decomposition) with error analyses and equivalences to existing mechanisms. Introduces PLA linearized attention that inherits pretrained weights and achieves SOTA vs. strong baselines (MVA, GSA) with ~80% of fine-tuning cost.

### Strengths
This paper's strength lies in unifying the transformation from Softmax to linear attention through a mathematically grounded, stepwise derivation—bridging a long-standing conceptual gap between dense and efficient attention mechanisms. The work is both theoretically elegant and empirically validated, with strong experimental support across multiple benchmarks, showing that the proposed path maintains accuracy while achieving measurable efficiency gains. The clarity of exposition, structured proofs, and comprehensive experiments reflect high research maturity.

### Weaknesses
Weakness is the lack of variance analysis and large-scale validation—the paper primarily focuses on mid-sized models, leaving open questions about stability at trillion-parameter scales. Additionally, while the theoretical pathway is rigorous, it could be strengthened with formal generalization analysis and a deeper exploration of its limitations relative to state-space or hybrid models. Overall, PLA is a technically sound, original, and impactful contribution that would be of strong interest to the ICLR community.

### Questions
When inheriting pre-trained weights and applying PLA’s five-step transformation, has there been any observance of training instability, gradient explosion, or sensitivity to layer order? More Quantitative details could further advocate for reproducibility.

Does this scale to very large frontier models ,can you do this to a 70B+ model and still get similar gains and only need ~80% of the tuning tokens versus MVA/GSA?

### Soundness
4

### Presentation
3

### Contribution
3
