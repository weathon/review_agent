# Influence-Preserving Proxies for Gradient-Based Data Selection in LLM FineTuning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Supervised fine-tuning (SFT) relies critically on selecting training data that most benefits model's downstream performance. Gradient-based data selection methods such as TracIn and Influence Functions leverage influence to identify useful samples, but their computational cost scales poorly, making them impractical for multi-billion-parameter large language models (LLMs). A common alternative is to use off-the-shelf smaller models as proxies, but they remain suboptimal since their learning dynamics are unclear, their sizes cannot be flexibly adjusted, and they cannot be further aligned with the target model in terms of gradient-based influence estimation. To address these challenges, we introduce IProX, a two-stage framework that derives influence-preserving proxies directly from the target model. It first applies a low-rank compression stage to preserve influence information of the target model, and then an aligning stage to align both model gradients and logits, thereby constructing proxies that flexibly control computational cost while retaining the target model’s influence. Experimental results across diverse LLM families and evaluation tasks show that IProX consistently outperforms off-the-shelf proxies and baseline methods. On Qwen3-4B, a 1.5B proxy constructed with IProX achieves stronger performance than the larger 1.7B off-the-shelf proxy. Notably, on Llama3.2, IProX achieves better performance than baselines while reducing computational cost by more than half relative to the full 3B model. These results show that IProX provides effective influence-preserving proxies, making gradient-based data selection more scalable for LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes IPROX, a framework for building smaller, influence-preserving proxy models to make gradient-based data selection feasible for large language models. Instead of using fixed off-the-shelf proxies, IPROX derives them directly from the target model through a two-stage process: an Influence-Preserving SVD that retains gradient-relevant components via second-moment reweighting, followed by a gradient-alignment stage that matches the proxy’s gradients and logits to the target model for stability. The authors provide theoretical justification that this compression preserves influence consistency and validate the method empirically across multiple model families (Llama3, Gemma3, Qwen2/3) and benchmarks (MMLU, BBH, TyDiQA). Results show that IPROX consistently outperforms larger off-the-shelf proxies while cutting computation by more than half, establishing a principled and efficient approach for scalable data selection in LLM fine-tuning.

### Strengths
1. Presents a novel, well-motivated framework (IPROX) that bridges model compression and gradient-based influence estimation for LLM data selection.
2. Offers strong theoretical grounding through Proposition 4.1, connecting layer perturbations to bounded influence deviation.
3. Demonstrates consistent empirical improvements across multiple LLM families (Llama3, Gemma3, Qwen2/3) and tasks (MMLU, BBH, TyDiQA) and achieves significant computational efficiency, reducing cost by over 50% while maintaining or improving performance.

### Weaknesses
1. The theory relies on strong smoothness and bounded shift assumptions that may not fully hold for deep transformer landscapes.
2. The alignment stage introduces extra hyperparameters (e.g., λ_KL, τ) that may require dataset-specific tuning.
3. Limited exploration of extreme compression regimes (beyond 70% sparsity), where influence preservation may degrade.
4. Experiments focus solely on text-based fine-tuning, leaving open how the method generalizes to multimodal or retrieval-augmented LLMs.

### Questions
1. The paper primarily evaluates with TracIn and Influence Functions. How would IPROX behave with more recent estimators such as Fisher-based or curvature-aware influence methods? Is IPSVD theoretically compatible with these variants?
2. What is the sensitivity of the extra hyperparameter (e.g., λ_KL, τ)?
3. The experiments cap at 70 % sparsity. What happens under more aggressive compression (e.g., 80–90 %)—does influence preservation collapse gradually or sharply?
4. The reported efficiency is impressive for 3B–7B targets. Could the authors provide projected or preliminary numbers for larger models, such as 70B, to demonstrate scalability under realistic industrial constraints?
5. Two highly relevant recent works should be included: (a) LENSLLM: Unveiling Fine-Tuning Dynamics for LLM Selection and (b) EvoSLD: Automated Neural Scaling Law Discovery With Large Language Models.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces IPROX, a two-stage framework for constructing influence-preserving proxy models to enable scalable gradient-based data selection in the fine-tuning of LLMs. The approach first compresses the target LLM using a low-rank, influence-aware SVD (IPSVD) and then further aligns the proxy with the target via internal gradient alignment and output anchoring losses. Empirical results are presented across several LLM families, demonstrating improved performance and efficiency compared to existing off-the-shelf proxy models and two baseline approaches.

### Strengths
- **Principled Influence-Preserving Compression**: The paper develops a well-justified influence-preserving low-rank SVD technique for model compression, addressing the misalignment between traditional reconstruction objectives and the needs of influence-based data selection.
- **Comprehensive Theoretical and Implementation Details**: The paper carefully connects theoretical derivations to practical considerations, including efficient computation using “skinny” SVDs and probe-based approximations.

### Weaknesses
- **Theoretical Limitations and Empirical Edge Cases**: While the influence-retention bound is elegant, its assumptions (local smoothness, geometric coherence, bounded covariate shift) could be restrictive in practice. Furthermore, the treatment of the embedding and LM head as incompressible is not systematically evaluated in ablation.
- **Performance Gains need more explanation**: The paper acknowledges task-type sensitivity but does not deliver deeper insight into when/why influence preservation matters most versus simple SVD or other alternatives.

### Questions
- Is there empirical evidence on how sensitive the effectiveness of IPROX is to the choice and size of the probe set N? Can an ablation show how influence approximation degrades (or not) as N varies, especially in large model regimes?
- Regarding the incompressible layers (embedding, LM head): Are there empirical ablation studies quantifying their effect on overall model or proxy performance?
- Could the authors clarify the mathematical assumptions required for the influence preservation bounds, and discuss scenarios where these may not hold (e.g., high non-linearity, significant distribution shift)?

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
3

### Summary
This paper addresses the computational inefficiency of gradient-based data selection methods for large language models (LLMs), which are critical for SFT but impractical for multi-billion-parameter models due to high costs. It introduces IPROX, a two-stage framework that constructs influence-preserving proxy models directly from the target LLM: first, an Influence-Preserving SVD stage compresses the target model’s weight matrices to retain gradient-based influence information (unlike standard SVD, which prioritizes reconstruction loss), and second, an alignment stage refines the proxy by matching its gradients to the target model in low-rank space and anchoring output logits via KL divergence.

### Strengths
- **Targeted solution to an important problem**: The paper directly addresses the limitation of gradient-based data selection, which is the  poor scalability with LLM size. The authors focus on proxy design, which is an orthogonal approach to simplifying influence computation itself. This fills a gap in existing work, which either uses suboptimal off-the-shelf proxies or reduces influence calculation cost without preserving alignment to the target model.
-  **Theoretically Grounded Proxy Construction**: IPROX is not heuristic: Proposition 4.1 provides a theoretical bound linking the expected squared error of layer perturbations to influence preservation, and IPSVD is designed to minimize this error via second-moment reweighting of inputs and upstream gradients.
- **Flexibility and Efficiency**: IPROX allows flexible control of proxy size via the rank parameter in IPSVD, enabling trade-offs between computational cost and performance.

### Weaknesses
- **Dependence on Probe Set Quality**: IPROX relies on a small probe set to approximate second-moment matrices for IPSVD. The paper mentions using token sampling to avoid length bias but does not evaluate how probe set size, diversity, or representativeness affects performance.
- **Narrow Task and Data Scope**: While the paper uses three diverse tasks, all training data is drawn from DOLLY (instruction-response pairs). It does not test IPROX on other data types (e.g., code, scientific text) or tasks with larger distributional shifts from DOLLY (e.g., low-resource language QA). This limits conclusions about IPROX’s performance in more heterogeneous fine-tuning scenarios.
- **Lack of Ablation for Second-Moment Reweighting**: While IPSVD’s second-moment reweighting is core to its design, the paper does not ablate this component (e.g., comparing IPSVD to unweighted SVD) to isolate its impact on influence preservation.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
IPROX addresses the computational cost of gradient-based data selection for LLM fine-tuning by constructing smaller proxy models that preserve the target model's influence characteristics. The framework uses two stages: (1) Influence-Preserving SVD (IPSVD) that compresses the target model while retaining influence-relevant components via second-moment reweighting, and (2) alignment that matches gradients in low-rank space and anchors output logits.

### Strengths
1. **Flexible efficiency-performance trade-off**: Enables controllable proxy size through sparsity parameter ρ, allowing practitioners to balance computational cost and selection quality.

2. **Practical scalability with minimal overhead**: Achieves >50% cost reduction on Llama3.2 with proxy construction adding only ~5-7 minutes overhead, using efficient probe-based approximation that avoids forming large  matrices.

### Weaknesses
**Missing comparisons to established data selection methods**: The paper lacks empirical comparison with core-set and generalization-based selection methods like GLISTER[1], Coresets, and gradient-based selection frameworks. These are only briefly mentioned in related work but not evaluated. The absence of comparison to methods like LESS[3] and DsDm[2] weakens the argument for IPROX's advantage over the broader data selection literature beyond just gradient-based influence methods.




[1] GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning
[2] DsDm: Model-Aware Dataset Selection with Datamodels
[3] Less: Selecting influential data for targeted instruction tuning

### Questions
How does IPROX compare to non-gradient baselines like GLISTER, LESS, or coreset methods?

### Soundness
1

### Presentation
2

### Contribution
2
