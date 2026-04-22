# SimpleFold: Folding Proteins is Simpler than You Think

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4

## Abstract
Protein folding models have achieved groundbreaking results typically via a combination of integrating domain knowledge into the architectural blocks and training pipelines. Nonetheless, given the success of generative models across different but related problems, it is natural to question whether these architectural designs are a necessary condition to build performant models. In this paper, we introduce SimpleFold, the first flow-matching based protein folding model that solely uses general purpose transformer blocks}. Protein folding models typically employ computationally expensive modules involving triangular updates, explicit pair representations or multiple training objectives curated for this specific domain. Instead, SimpleFold employs standard transformer blocks with adaptive layers and is trained via a generative flow-matching objective with an additional structural term. We scale SimpleFold to 3B parameters and train it on approximately 9M distilled protein structures together with experimental PDB data. On standard folding benchmarks, SimpleFold-3B achieves competitive performance compared to state-of-the-art baselines, in addition SimpleFold demonstrates strong performance in ensemble prediction which is typically difficult for models trained via deterministic reconstruction objectives. SimpleFold challenges the reliance on complex domain-specific architectures designs in protein folding, opening up an alternative design space for future progress.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces SimpleFold, a novel protein folding model based on flow-matching and a standard Transformer architecture. The central thesis challenges the necessity of complex, domain-specific architectural components (like MSAs, explicit pair representations, triangle updates, equivariant modules) prevalent in state-of-the-art models such as AlphaFold2. SimpleFold adopts a generative approach, treating folding as analogous to text-to-image generation, mapping amino acid sequences (encoded via a frozen PLM, ESM2-3B ) to all-atom 3D coordinates using only standard Transformer blocks with adaptive layers conditioned on time. The authors scale SimpleFold up to 3 billion parameters and train it on a large dataset combining PDB experimental structures and approximately 9 million distilled structures . On standard benchmarks (CAMEO22, CASP14), SimpleFold-3B achieves performance competitive with SOTA baselines. Notably, it demonstrates strong performance in ensemble prediction tasks, which is often challenging for models trained with deterministic objectives. The work suggests that scale and general-purpose architectures might suffice for learning complex folding patterns, opening alternative design avenues.

### Strengths
- The core argument—that domain-specific inductive biases might be replaceable by scale and general architectures—is highly stimulating and challenges conventional wisdom in protein folding. If validated further, this could significantly simplify model design in structural biology and beyond.
- SimpleFold's reliance solely on standard Diffusion Transformer blocks makes the architecture remarkably simple compared to those with attention bias in AF3. This generality facilitates leveraging advances from the broader Transformer ecosystem and potentially simplifies implementation and optimization (like flash-attn and deepspeed).
- The paper provides a clear demonstration of performance scaling with model size (up to 3B parameters) and training compute on folding benchmarks. This empirical evidence supports the claim that scaling is a viable path for improving performance with this simpler architecture. The scaling with data size is also shown.

### Weaknesses
- While competitive, SimpleFold-3B does not consistently surpass the very best models like AlphaFold2, especially on the challenging CASP14 benchmark. This suggests that current domain-specific designs and/or MSA information still provide an edge, particularly for difficult targets.
- SimpleFold 1.6B and 3B models achieve accuracy comparable to (or on par with) ESMFold. However, their inference speed is noted to be slower. This presents a critical trade-off, and the justification for using these models over ESMFold is unclear given this substantial speed limitation.

### Questions
- How frequently do SimpleFold's predictions contain steric clashes or distorted covalent geometries compared to models like AlphaFold2 (before relaxation) or other generative models? Is a relaxation step necessary/used for evaluation?
- Given the performance gap on CASP14, do the authors believe further scaling (model size, data) can close this gap, or are there fundamental limitations to the purely data-driven approach without explicit biases for certain protein families or topologies?

### Soundness
3

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
4

### Summary
This paper proposes SimpleFold, a flow-matching based protein folding model that relies solely on general-purpose transformer blocks for efficient structure prediction. In contrast to prior methods that employ triangular updates, explicit pair representations, or multi-objective training, SimpleFold utilizes standard transformer blocks enhanced with adaptive layers and is trained with a generative flow-matching objective augmented by a structural loss term. Evaluated on standard folding benchmarks, the SimpleFold-3B model—trained on both distilled structures and PDB data—achieves performance competitive with state-of-the-art baselines. Additionally, the model demonstrates strong capability in ensemble prediction tasks.

### Strengths
•	SimpleFold effectively removes complex components commonly used in previous protein folding models—such as MSA processing, pairwise representations, and triangle modules—leading to significant computational speedup while maintaining competitive prediction accuracy.

•	The model shows promising results in ensemble generation, highlighting its potential for capturing conformational diversity.

•	The paper is clearly structured and easy to follow.

### Weaknesses
•	The scope of SimpleFold remains limited to single-chain protein structure prediction. Given the emergence of efficient and generalizable predictors such as ESMFold and Protenix-Mini [1]—which are also capable of predicting biomolecular complexes—SimpleFold does not demonstrate a clear advantage in either efficiency or accuracy, raising questions about its practical added value.

•	The overall framework constitutes a relatively straightforward integration of existing techniques from computer vision and protein modeling (e.g., flow matching and transformer adaptations), and the core technical innovation appears incremental.

[1] Protenix-Mini: Efficient Structure Predictor via Compact Architecture, Few-Step Diffusion and Switchable pLM

### Questions
•	SimpleFold appears to adopt training structures and strategies similar to prior flow-based folding methods such as AlphaFlow or ESMFlow. Could the authors provide further insight or ablation studies to explain why SimpleFold is more effective at recovering multi-state protein conformations? Is this capability attributable to the architectural design, the training objective, or the data used?

### Soundness
3

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
The paper presents SimpleFold, a flow-matching generative model for protein folding. It uses a standard transformer architecture to generate full-atom structures, conditioned on embeddings from a frozen ESM2-3B protein language model (PLM). This approach notably avoids the explicit pair representations and triangular updates common in models like AlphaFold. The largest 3B parameter model, trained on approximately 9 million distilled and experimental structures, achieves performance that is competitive with, but not superior to, existing state-of-the-art baselines on standard folding and ensemble generation benchmarks.

### Strengths
The primary strength is **architectural simplification**. The work demonstrates that a general-purpose transformer can achieve comparable performance to more complex, bespoke architectures, provided it is leveraged at a large scale and conditioned on a powerful pretrained PLM. The clear empirical validation of scaling laws for both model and data size is a useful, albeit expected, finding.

### Weaknesses
**Limited Novelty**: The approach is highly derivative. It combines a standard transformer with a known flow-matching objective and, most importantly, relies heavily on the powerful ESM2-3B PLM. These attempts have already appeared in previous work and are not surprising; they cannot be called new folding methods.

**Insignificant Performance**: In the context of the rapidly advancing protein design field, the results are not a significant breakthrough. The model achieves "competitive" or "comparable" performance, failing to clearly outperform established baselines like AlphaFold2 or ESMFold. This incremental result does not demonstrate a clear advantage.

### Questions
See **Weaknesses**.

### Soundness
2

### Presentation
3

### Contribution
2
