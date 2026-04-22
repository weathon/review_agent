# LoFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 8

## Abstract
Large pre-trained models are commonly adapted to downstream tasks using parameter-efficient fine-tuning methods such as Low-Rank Adaptation (LoRA), which injects small trainable low-rank matrices instead of updating all weights. While LoRA dramatically reduces trainable parameters with little overhead, it can still underperform full fine-tuning in accuracy and often converges more slowly. We introduce LoFT, a novel low-rank adaptation method that behaves like full fine-tuning by aligning the optimizer's internal dynamics with those of updating all model weights. LoFT not only learns weight updates in a low-rank subspace (like LoRA) but also properly projects the optimizer's first and second moments (Adam's momentum and variance) into the same subspace, mirroring full-model updates. By aligning the low-rank update itself with the full update, LoFT eliminates the need for tuning extra hyperparameters, e.g., the LoRA scaling factor $\alpha$. Empirically, this approach substantially narrows the performance gap between adapter-based tuning and full fine-tuning and consistently outperforms standard LoRA-style methods, all without increasing inference cost. The code is available at https://github.com/tnurbek/loft.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces LoFT, a new parameter-efficient fine-tuning method designed to make low-rank adaptation behave like full fine-tuning. Existing LoRA-based approaches mainly focus on gradient approximation but ignore the optimizer state misalignment, particularly in the first and second moments of the AdamW optimizer. LoFT explicitly aligns both gradients and optimizer states with full fine-tuning dynamics through six carefully designed components: alternating updates, gradient scaling, optimizer state calibration, second-moment alignment, projected full update reconstruction, and gradient clipping. Theoretical analysis proves that LoFT reduces exactly to AdamW when the rank equals the full dimension. Experiments on multiple large language models and vision transformers demonstrate that LoFT achieves higher accuracy and faster convergence than LoRA and DoRA while maintaining the same inference cost and number of trainable parameters.

### Strengths
1.	The method directly addresses the optimizer state misalignment problem that has been largely overlooked in prior low-rank adaptation research.
2.	The theoretical analysis is rigorous and provides a clear guarantee that LoFT degenerates to AdamW in the full-rank case.
3.	The six-component design is systematic, and is validated through detailed ablation studies.
4.	LoFT consistently outperforms LoRA and DoRA on both natural language and vision benchmarks, showing broad applicability.

### Weaknesses
1.	The additional memory cost, which can reach about twenty-five percent compared to LoRA, is not fully analyzed for its impact on large-scale training.
2.	Experiments are limited to models of eight billion parameters or smaller, leaving scalability to larger models unverified.
3.	The effect of optimizer state projection on stability and convergence speed is discussed conceptually but lacks quantitative analysis.
4.	The paper does not report concrete throughput or training speed measurements compared with LoRA or DoRA.

### Questions
1. What is the quantitative impact of the additional memory requirement on training efficiency and GPU utilization?

2. Can LoFT be extended to other optimizers such as Muon that use different moment estimation mechanisms?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a new method called LoFT for parameter-efficient fine-tuning of large pre-trained models. LoFT extends the Low-Rank Adaptation (LoRA) approach by aligning the internal states of the optimizer (including momentum and second moments) with full fine-tuning, thereby attempting to reduce the accuracy gap typically seen between low-rank and full fine-tuning methods. The authors test their method across multiple language and vision tasks, showing performance improvements compared to previous low-rank adaptation methods, especially at very low ranks. They also discuss trade-offs in terms of memory usage and computational overhead, presenting simpler variants with lower overhead.

### Strengths
- **Substantive technical contribution with theory.** The paper proposes a concrete improvement over standard LoRA-style adaptation and backs it up with clear derivations/analysis. The core ideas are technically motivated (e.g., aligning updates with full fine-tuning dynamics), and the method’s components are explained rather than presented as ad-hoc tricks.
- **Broad empirical validation across domains.** Experiments cover multiple modalities/datasets (e.g., language and vision) and a range of ranks/settings, suggesting the approach is not narrowly tailored to a single task.

### Weaknesses
- **Gap between theory and the strongest claim.** While the derivations are compelling, there remains a gap between the formal analysis and the paper’s strongest claim(s) (e.g., exact equivalence to full fine-tuning/AdamW under certain limits). A precise theorem with assumptions, or a more cautious phrasing, would strengthen the work.
- **LLM evaluation is too basic.** The large-language-model experiments rely on relatively easy, small benchmarks. For a model like Llama-3-8B, a more representative LLM evaluation suite (e.g., code, math/reasoning, or long-context benchmarks) would be more convincing. Multi-seed runs with statistical reporting would further solidify the results.

### Questions
In Table 6, several DoRA results (e.g., r=4 with BoolQ=32.35, PIQA=7.13, Winogrande=0.00) are anomalously low and LoRA sometimes degrades as rank increases (e.g., r=4 worse than r=1 on PIQA/HellaSwag), suggesting hyperparameter/setup issues—could you explain these discrepancies?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes LoFT (Low-rank adaptation that behaves like Full fine-Tuning), a novel parameter-efficient fine-tuning (PEFT) method designed to closely approximate the optimization dynamics of full fine-tuning within a low-rank subspace. Building on the LoRA framework, LoFT introduces several key components: alternating updates, gradient scaling, first- and second-moment state calibration, projected full-model updates, and gradient clipping. Together, these allow LoFT to mimic AdamW’s optimizer behavior while maintaining the computational and inference efficiency of low-rank tuning.

Empirical results across language (LLaMA-7B/2-7B/3-8B) and vision (ViT-Base) models demonstrate that LoFT consistently outperforms existing PEFT methods such as LoRA and DoRA, particularly under extreme low-rank constraints (e.g., rank ≤ 4). Ablation studies confirm that optimizer state calibration is critical to LoFT’s strong performance.

### Strengths
**Strong conceptual motivation**: The paper identifies a previously underexplored source of suboptimality in LoRA — optimizer state misalignment — and provides a well-motivated correction grounded in optimization theory.

**Methodological completeness**: The framework integrates multiple components (gradient projection, alternating updates, moment calibration) into a cohesive, well-defined optimizer (LoFT-AdamW), which provably reduces to full fine-tuning when rank = full.

**Theoretical insight**: The analysis on matrix factorization clearly shows how LoFT recovers full fine-tuning dynamics, with formal smoothness guarantees and equivalence to alternating least squares in the special case.

**Extensive empirical validation**: The experiments span multiple model families and domains, including large LLMs and ViTs, with clear, consistent performance improvements over LoRA and DoRA.

**Careful ablation studies**: The paper convincingly demonstrates the necessity of each component, especially the importance of first-moment calibration for stable convergence.

**Practical relevance**: LoFT eliminates the need to tune the LoRA scaling factor (α), reducing hyperparameter sensitivity and simplifying deployment.

### Weaknesses
**Missing citation and discussion of concurrent work**:
The Alternating Updates component (Building Block 1) reproduces an idea conceptually similar to AltLoRA [1], which independently proposed alternating optimization of low-rank factors to eliminate second-order coupling in LoRA updates.

The absence of a citation or discussion of AltLoRA is a notable omission, especially since the “alternating update” mechanism is presented as a key innovation. This should be acknowledged as concurrent or parallel work, with clarification of LoFT’s additional contributions beyond AltLoRA (notably optimizer-state alignment).

**Complexity and memory overhead**: While the paper discusses the cost of storing previous iterates and cross-terms, the actual scalability to very large models (≥70B parameters) remains untested; empirical results are limited to ≤8B models.

**Presentation clarity**: The main text can be dense, with many mathematical expressions introduced in rapid succession. It would be helpful to include more detailed mathematical explanations or derivations in the appendix to improve readability and reproducibility.

[1] Yu, Xin, et al. "AltLoRA: Towards Better Gradient Approximation in Low-Rank Adaptation with Alternating Projections." arXiv preprint arXiv:2505.12455 (2025).

### Questions
See Weaknesses.

If the authors are willing to carefully clarify the relationship and differences between LoFT and AltLoRA during the rebuttal phase, I would be inclined to raise my score.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a new LoRA method which explicitly tries to mimic full-finetuning dynamics. Crucially, the paper identifies the importance of matching the optimizer state in addition to the updates. This is accomplished via 

1. alternating updates
2. gradient rescaling
3. momentum recalibration
4. second moment recalibration
5. projecting reconstructed AdamW update
6. approximating gradient clipping

These "building blocks" ensure that in the limit of full-rank the full-finetuning dynamics are recovered. In the low-rank regime the experiments demonstrate improved performance over vanilla LoRA.

### Strengths
The authors provide a principled derivation for a LoRA method meant to explicitly mimic full-finetuning updates. The approach recovers the correct dynamics in the full-rank limit. The authors provide extensive experiments showing the promise of the method especially for low-ranks. The method is practically efficient, it requires modest memory and runtime overhead, and is simple to implement.

### Weaknesses
The experiments are only conducted with $r \leq 32$ and models with $\leq 8$B parameters.

Second-moment calibration appears to have low-impact at a high cost, however it is still valuable to derive and test this idea.

It is unclear if alternation is helpful or not.

### Questions
Do the authors have any intuition about when mimicking the full finetuning update is optimal or not?

### Soundness
4

### Presentation
4

### Contribution
3
