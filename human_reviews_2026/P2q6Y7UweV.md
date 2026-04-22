# INSTANT: Compressing Gradients and Activations for Resource-Efficient Training

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 6, 2

## Abstract
Deep learning has advanced at an unprecedented pace. This progress has led to a significant increase in its complexity. However, despite extensive research on accelerating inference, training deep models directly within a resource-constrained budget remains a considerable challenge due to its high computational and memory requirements. In this paper, we introduce INSTANT (compressIng gradieNtS and acTivAtions for resource-efficieNt Training), a method designed to address both the computational and the memory bottlenecks when training. INSTANT reduces resource demands during backpropagation by projecting gradients and activations into a low-rank subspace and performing computation within that compressed representation. Experimental results demonstrate that INSTANT achieves a $15\times$ reduction in computational cost and $32\times$ reduction in activation memory with negligible impact on model performance. The code will be made publicly available upon the paper's acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a new distillation technique called grafting, aimed at improving the generalization of autoregressive language models. The grafting strategy integrates sequence trees generated at multiple temperatures into a single distillation target. The paper emphasizes the balance between model compression and generalization, addressing the mode-covering behavior crucial for building more generalizable models. The experimental validation demonstrates the method's effectiveness across several datasets, particularly CIFAR-10 and CIFAR-100.

### Strengths
Originality: The grafting strategy is an original approach for balancing model compression and generalization in language model distillation.

Experimental Validation: The method is validated across several datasets, and the results show some potential for improving model performance while reducing computational and memory costs.

Clarity: The paper is well-organized and clearly explains the experimental setup and methodology.

### Weaknesses
Lack of Comparison with Related Work: While the paper mentions the error accumulation problem in other activation compression methods, it does not provide a sufficient comparison with existing methods like Sakr & Khailany’s ESPACE or Yang et al.’s LBP-WHT in terms of accuracy and performance. Without this comparison, the claims about grafting’s effectiveness remain unclear.

Limited Experimental Setup: The experiments are mainly focused on simple image classification tasks (e.g., CIFAR-10, CIFAR-100). These datasets may not fully demonstrate the method's potential in more complex or real-world scenarios.

Unaddressed Limitations: The paper does not sufficiently explore some of the limitations of the proposed method, including the computational cost of SVD and its scalability to larger models or long sequences.

### Questions
Could you provide a more detailed comparison between grafting and existing methods like Sakr & Khailany’s ESPACE or Yang et al.’s LBP-WHT, particularly in terms of accuracy and computational efficiency?

How do you plan to scale the grafting method for larger models or longer sequences, and what are the potential challenges?

Could you provide further experiments on more complex datasets or real-world tasks to validate the method’s generalizability?

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
4

### Summary
This paper proposes a compression method called INSTANT. INSTANT is a method that projects the activations and gradients to lower ranks, based of analysis that is done at regular intervals during training. The authors show the benefits of compressing both the gradients and activations during training by showing reduction in FLOPs and training time (in some cases).

### Strengths
1. The paper is  tackling both memory and computational bottlenecks.  Activation compression for memory saving has been explored previously, but the idea of also compressing the activation gradient ($g_y$) to reduce backpropagation FLOPs is a useful contribution.

2. The authors provide empirical evidence (Fig. 3) to support their intuition that activation gradients are inherently low-rank, justifying their compression approach.

3. NSTANT's  SVD-based approach is data-driven. This allows it to generalize more effectively to other modalities, as demonstrated by its strong performance on NLP tasks (Table 2) where previous works (eg LBP-WHT) struggles.

4. The authors validates the method's usability by reporting significant wall-clock speedups (2x to 12.5x) on resource-constrained edge CPUs.

### Weaknesses
I believe, currently this paper's primary weaknesses lie in the evaluation of its computational claims and the transparency of its overhead costs.

**Omission of Wall-Clock Training Time on GPUs:**

1. The paper's main results (Tables 1 & 2) were generated on an NVIDIA V100 GPU but only report FLOPs, not wall-clock training time.

a) The authors' justification (Section 4.1) that "FLOPs... [are] unaffected by implementation details" is a weak defense. A reduction in FLOPs does not guarantee a proportional reduction in training time, especially on GPUs. Modern deep learning libraries (like cuDNN) are highly optimized for large, dense matrix multiplications, and the low-rank operations introduced by INSTANT may not be as implementation-efficient, thus creating a bottleneck.

b) The authors themselves admit a discrepancy in Appendix I, stating, "The (12x) time reduction is not comparable to (17x) FLOP reduction," even on a CPU. This gap is likely to be even larger on a GPU, and the lack of this data makes it difficult to assess the practical speedup in a typical training environment.

**Cost and Nature of the "Static" Subspace:**

2. The term "static" subspace (used in the contribution list and abstract) could be misleading. The subspace is not fixed; it is "periodically" recalibrated every $N_t$ steps (e.g., $N_t=50$ or $N_t=200$).

a) The computational cost of this recalibration (Algorithm 1) appears to be non-trivial and is not accounted for in the reported per-step FLOP savings. This calibration requires running multiple batches and performing SVD for every layer being compressed. This overhead could significantly diminish the overall wall-clock time savings.

b) The method's stability relies on an "oversampling" hyperparameter $p$, which is introduced to "reduce information loss when the core bases change" between calibrations (Section 3.2, Fig. 6). This suggests the "static" assumption is fragile and introduces another sensitive hyperparameter that must be tuned, adding to the method's complexity.

### Questions
1. Could the authors please provide the total wall-clock training time (not just backward time) for the main V100 GPU experiments in Tables 1 and 2? This is essential for evaluating the practical speedup of INSTANT in a standard training scenario.

2. How is the computational cost of the periodic calibration step (Algorithm 1) factored into the total FLOPs reported? Could you provide an analysis of this overhead, for instance, as a percentage of total training time or FLOPs?

3. Given that the subspace is recalibrated every $N_t$ steps, would "periodically updated" or "cached" be a more accurate description than "static"?

4. The oversampling parameter $p$ appears critical for performance (Fig. 6, Fig. 8). How sensitive is the method to the choice of $p$ and the calibration frequency $N_t$? Does this introduce a significant hyperparameter tuning burden?

5. Following up on the 17x FLOP vs. 12x time gap noted in Appendix I: What is the authors' hypothesis for the performance gap on a V100, where highly optimized dense matrix multiplication kernels are the standard?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes INSTANT, a training-time compression method that projects activations and gradients into low-rank subspaces and performs backpropagation in those compressed representations. The authors describe (i) an SVD-based calibration to build per-layer projectors, (ii) projection/truncation with an energy threshold + oversampling, and (iii) a low-rank backward algorithm. Experiments show large reductions in activation memory and FLOPs across Transformer and CNN models with small accuracy drops.

### Strengths
* Solid empirical evaluation across modalities (vision & NLP) and architectures (Transformers, CNNs).
* Clear exposition, reproducibility-minded appendices and pseudocode.
* Practical relevance: reduces activation memory and backward FLOPs as demonstrated in their experiments, with sensible ablations (oversampling, calibration frequency, rank choices).
* Sound theoretical guarantee for stable low-rank training. Their analysis shows that SVD-based projections minimize reconstruction error and that gradient approximation error remains bounded through depth, vanishing as the retained energy ε → 1.
* Includes deployment-relevant experiments (edge device / Raspberry Pi).

### Weaknesses
1. **Severe overlap with prior work (CompAct, NAACL 2025, publicly available on arXiv since Oct 2024):**   
Conceptually highly similar and structurally parallel to CompAct [1], differing mainly in projection construction and calibration choices, without mentioning CompAct at all, despite the latter being published and publicly available months before ICLR submission. 
Both papers:    
* Compress activations via low-rank projections during the forward pass.
* Compute gradients in the compressed subspace and decompress for weight updates.
* Aim to reduce memory and optimizer overhead jointly.
* Demonstrate scaling benefits on LLaMA and BERT-like models.
The algorithmic structure of INSTANT (forward compression → compressed backward → decompression for update) matches CompAct’s Algorithms 1–3. The overlap extends to terminology (“projected activations,” “reduced optimizer states”), theoretical justification, and empirical evaluation. 

2. **Novelty and contribution are overstated:**   
While INSTANT adds implementation refinements, such as calibration-based rank selection and changes the choice of projection matrix, these are engineering extensions rather than conceptual advances. The claim of being the “first to jointly compress activations and gradients” is false, due to the overlap with prior work.

3. **Lack of comparison:**   
The experiments do not compare with the most appropriate relevant works like GaLore [2] or any of the myriad of works that followed it (VeLORA [3] ,Grass [4] ,WeLore [5]...). Reported baselines are insufficient to establish novelty or superiority.


References
* [1] CompAct: Compressed Activations for Memory‑Efficient LLM Training – Shamshoum et al., NAACL 2025, arXiv:2410.15352v1. 
* [2] GaLore: Memory‑Efficient LLM Training by Gradient Low‑Rank Projection – Zhao et al., ICML 2024, arXiv:2403.03507.
* [3] VeLORA: Memory Efficient Training using Rank‑1 Sub‑space Activations – Miles et al., NeurIPS 2024 Poster, algorithm for compressing activations into 1-D subspace. 
* [4] Grass: Compute Efficient Low‑Memory LLM Training with Structured Sparse Gradients – Muhamed et al., arXiv:2406.17660.
* [5] WeLore: Weight Low‑Rank Projection for Memory‑Efficient Fine‑Tuning – Jaiswal et al., ICLR 2025, arXiv:2407.11239.

### Questions
* Were the authors aware of CompAct (NAACL 2025) at submission time?
* What specific conceptual or methodological innovations distinguish INSTANT from CompAct beyond the choice of projection matrix?
* Can you provide a direct empirical comparison to CompAct under matched conditions (e.g., same model, rank, dataset)?
* How do the claimed FLOP reductions compare to a random projection baseline?

### Soundness
3

### Presentation
4

### Contribution
1
