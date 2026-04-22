# Shadow-FT: Tuning Instruct Model via Training on Paired Base Model

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Large language models (LLMs) consistently benefit from further fine-tuning on various tasks. However, we observe that directly tuning the Instruct (i.e., instruction tuned) models often leads to marginal improvements and even performance degeneration. Notably, paired Base models, the foundation for these Instruct variants, contain highly similar weight values (i.e., less than 2% on average for Llama 3.1 8B). The Base model tends to be a good learner yet a weak backbone without post-training. Therefore, we propose a novel Shadow-FT framework to tune the Instruct models by leveraging the corresponding Base models. The key insight is to fine-tune the Base model, and then directly graft the learned weight updates to the Instruct model. Our proposed Shadow-FT introduces no additional parameters, is easy to implement, and significantly improves performance. We conduct extensive experiments on tuning mainstream LLMs, such as Qwen 3 and Llama 3 series, and evaluate them across 19 benchmarks covering coding, reasoning, and mathematical tasks. Experimental results demonstrate that Shadow-FT consistently outperforms conventional full-parameter and parameter-efficient tuning approaches. Further analyses indicate that Shadow-FT can be applied to multimodal large language models (MLLMs) and combined with direct preference optimization (DPO).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes **Shadow-FT**, a simple scheme to tune **INSTRUCT** models **via** their paired **BASE** models. Instead of updating the INSTRUCT weights directly (which the authors observe can yield small gains or even regressions), Shadow-FT **fine-tunes the BASE model** on the target data and then **grafts the learned weight delta** onto the INSTRUCT model: (W_I^+ !=! W_I + ( \text{Tune}(W_B) - W_B)). The motivation is that paired BASE/INSTRUCT checkpoints are **highly similar** (average gap (\sigma < 0.03) for several families), so deltas learned on BASE should transfer well to INSTRUCT without extra training cost.

### Strengths
- **Clear problem diagnosis:** Directly tuning INSTRUCT often underperforms; the paper quantifies cases where FT/LoRA **regress** compared to the vanilla INSTRUCT. 
- **Simple, precise method:** The BASE→INSTRUCT delta-grafting update is explicit and easy to implement; it **adds no training cost** beyond standard tuning. 
- **Empirical consistency:** Gains replicate across **sizes (1B–32B)** and **settings** (full FT and LoRA), with domain-transfer and LoRA-rank sweeps.

### Weaknesses
- **Baseline coverage:** While conventional FT/LoRA baselines are solid, there is **no direct empirical comparison** to closely related *proxy/transfer* approaches (e.g., Proxy-Tuning, RE-Adapt, task/chat vectors), even though they are discussed; such comparisons are important to position Shadow-FT against the nearest alternatives. 
- **Scope & generalization:** Results emphasize Qwen/Llama families and the BAAI-2k plus a few domain sets; stronger **model and data diversity** (more families, public corpora, non-English, longer-context tasks) would reinforce generality claims.  
- **Similarity assumption auditing:** The claim that BASE/INSTRUCT are “highly similar” ((\sigma<0.03)) underpins the method; more **layer-wise and per-submodule** analyses (e.g., attention/MLP/norms, embeddings, positional encodings) and more **models** beyond the shown examples would strengthen the premise.

### Questions
Can you add head-to-head results vs. **Proxy-Tuning** and **RE-Adapt** on at least one model family and dataset to establish competitive positioning?

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
4

### Summary
The paper introduces Shadow-FT, a framework for fine-tuning instruction-tuned (INSTRUCT) LLM by leveraging their corresponding BASE models. The key idea is to fine-tune the BASE mode and then transfer the learned weight deltas directly to update the INSTRUCT model. The authors claim that this avoids the over-regularization effect introduced by instruction-following priors and yields consistent improvements across multiple model families (Qwen, LLaMA, Gemma, etc.) and domains (math, code, reasoning, multimodal).

### Strengths
1. The paper identifies a useful observation — the BASE and INSTRUCT models share over largely weight similarity (σ < 0.03) — and exploits this to propose a minimal yet elegant transfer mechanism. The “learn-on-BASE, apply-to-INSTRUCT” paradigm is intuitive.

2. Experiments are conducted across several major model families (Qwen, Llama, Gemma, Falcon, Yi) and cover 19 benchmarks (Math-7, Code-3, Knowledge-9). The results are extensive and generally consistent, showing that Shadow-FT often outperforms both full-parameter fine-tuning and LoRA-based PEFT.

### Weaknesses
1. Lack of theoretical grounding. The claim that INSTRUCT fine-tuning “degenerates due to preexisting priors” lacks a formal mechanism or controlled ablation. 
2. Lack of recent related work. Several existing works already transfer knowledge between BASE/INSTRUCT or different model checkpoints, the authors identified these work but did not properly compare to them, such as RE-Adapt [1], Task/Chat Vectors [2, 3] and Proxy-Tuning [4]
3. Missing of some implementation details, such as (a) whether optimizer states (Adam moments) are reused or reinitialized when applying BASE deltas, (2) how learning-rate scaling interacts with the BASE–INSTRUCT mapping. (3) after you use BASE model's gradient to update INSTRUCT model, do you also update the BASE model?

[1] https://arxiv.org/pdf/2405.15007
[2] https://arxiv.org/abs/2212.04089
[3] https://aclanthology.org/2024.acl-long.590.pdf
[4] https://arxiv.org/abs/2401.08565

### Questions
Please see the questions in weaknesses. 

1. After your Shadow-FT, does the INSTRUCT model still keep the capacity on open-end instruction following task? Any impact on the model's helpfulness, harmlessness, fluency and hallucination?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Shadow‑FT, a simple framework for tuning INSTRUCT variants via their paired BASE models. Empirically, directly fine‑tuning INSTRUCT models often leads to marginal gains or even degrade. The authors propose to (1) fine‑tune the BASE model with standard SFT or LoRA and (2) graft the learned weight deltas onto the INSTRUCT model. Across many base models and benchmarks, Shadow‑FT typically matches or exceeds tuning the INSTRUCT directly.

### Strengths
- Simplicity + breadth. The method is simple and straightforward and seems widely applicable according to experiments, with coverage across different benchmarks.
- Paper writing is clear and easy to follow.

### Weaknesses
- Novelty is limited. The idea is closely related to task vectors, chat vectors, RE‑Adapt, and proxy tuning.
- More/stronger theoretical framing beyond weight similarity maybe needed.
- Baselines such as RE‑Adapt could be added.

### Questions
How does $\sigma$ distribute across layers/blocks? Does transferring only specific layers (e.g. MLPs vs attention) retain most gains?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Shadow-FT, a fine-tuning framework that addresses performance degradation when directly tuning instruction-following (INSTRUCT) models. The key finding is that BASE and INSTRUCT model weights are highly similar (σ < 0.03), enabling transfer of weight updates learned on BASE models to INSTRUCT models via W⁺ᵢ = Wᵢ + (W⁺ᵦ - Wᵦ). The authors evaluate 7 LLM families across 19 benchmarks covering math, coding, and reasoning tasks. Results show Shadow-FT generally outperforms conventional fine-tuning, with gains most pronounced on smaller models (Qwen-3-4B: 68.0→69.4) and under LoRA settings. The method introduces no additional training cost and extends to multimodal models and DPO alignment. However, results are inconsistent across settings, with multiple cases showing performance degradation and minimal gains on larger models.

### Strengths
1. **Simple method**: The method requires only weight arithmetic W_I + (W⁺_B - W_B) with zero additional parameters, no extra training cost, and easy implementation. This simplicity aids adoption compared to complex meta-learning or distillation approaches

2. **Mechanistic insight**: Figure 5's analysis showing BASE exhibits 22.6% lower initial loss, 3.25× smaller initial gradient, and more stable gradient decay (58% vs. 11% at step 61) provides the strongest evidence for why BASE is a better learner than INSTRUCT. This mechanistic insight goes beyond empirical observation to suggest fundamental training dynamics differences

3. **Weight similarity systematic documentation**: Figure 2, Figure 6 (Appendix C), and analysis across 9 model families systematically documents σ < 0.05, showing BASE/INSTRUCT weights differ by <5% despite capability gaps. This empirical contribution establishes foundation for future work on BASE/INSTRUCT relationships

4. **Comprehensive multi-setting evaluation**: Evaluation spans 7 LLM families (Qwen 2/2.5/3, Llama 3.1/3.2, Gemma-3, Yi, Falcon), 19 benchmarks (Math-7, Code-3, Knowledge-9 aggregates), multiple model sizes (1B-90B parameters), both tuning strategies (full-parameter, LoRA), and various LoRA ranks (4-512). This breadth provides evidence across diverse conditions

5. **Rank robustness for Shadow-FT**: Figure 3 demonstrates Shadow-FT maintains gains across LoRA ranks 4-512 (scores 30.15-32.03) while vanilla FT degrades with larger ranks (scores 28.47-31.84), suggesting the method doesn't overfit to specific parameterizations and may regularize against LoRA's documented forgetting issues

6. **Extensions demonstrate generalizability**: Applications to MLLMs (Table 4: Gemma-3-27B 60.28→63.80, Llama-3.2-Vision-90B 79.92→80.60) and DPO alignment (Table 3: Shadow-DPO 55.39 vs vanilla DPO 54.62) show the approach isn't limited to standard supervised fine-tuning of text-only models

7. **Pass@k analysis reveals capability preservation**: Appendix D.4 (Figure 7) shows Shadow-FT preserves model exploration capability better than vanilla FT in both thinking and non-thinking modes. Vanilla FT maintains >30% gap in thinking mode across k, while Shadow-FT matches INSTRUCT baseline, suggesting Shadow-FT avoids capability collapse

8. **Domain-specific tuning validation**: Table 2 and Figure 4 demonstrate effectiveness on specialized datasets (Medical-o1-reasoning-SFT, Code-Z1, LIMO, OpenR1-Math), showing the method works beyond general instruction data

### Weaknesses
1. **No direct comparisons**: The paper cites Chat Vector, RE-Adapt, and concurrent work but provides no empirical comparisons. It should evaluate Shadow-FT against these and standard PEFT baselines to clarify both performance and efficiency advantages.

2. **Unexplained inconsistencies**: Shadow-FT sometimes degrades performance substantially (e.g., –3.3 to –18.7 on MATH tasks), contradicting its claimed consistency. The paper offers no analysis or framework for predicting when it helps or harms.

3. **Weak theoretical basis**: The intuition that "BASE learns without instruction interference" is plausible but unproven. Reported weight similarity doesn’t imply transferability, and no formal analysis connects Δ_B to INSTRUCT improvements.

4. **Limited experimental scope**: Experiments use small datasets (2K–10K) and English-only benchmarks, limiting generalizability. Baseline fine-tuning often underperforms, suggesting weak tuning, and scalability to larger datasets remains untested.

5. **Ad-hoc design choices**: Key hyperparameters (α, LoRA rank) and the similarity metric are arbitrary or unjustified. Sensitivity analyses reveal large effects, yet no optimization or rationale is provided.

6. **Results**: Results lack confidence intervals, significance tests, and variance reporting. Many improvements are small and could fall within measurement noise.

7. **Unclear distinctiveness**: Almost any weight interpolation improves results, suggesting Shadow-FT’s benefit may stem from generic weight mixing rather than a novel mechanism.

### Questions
1. Can the authors empirically compare Shadow-FT vs. Chat Vector vs. RE-Adapt vs. Lin et al. on shared benchmarks? Which is better in accuracy/efficiency?
2. What predicts when Shadow-FT helps vs. hurts? Can the authors provide diagnostic criteria based on model size, σ value, and dataset characteristics?
3. Can the authors formalize conditions for successful weight transfer? Why does weight similarity imply update compatibility?
4. Does Shadow-FT work with 100K+ samples? Is the benefit specific to the low-data regime?
5. Which layers (embeddings, attention, FFN) transfer best? Why does LoRA sometimes outperform full?
6. Table 8 shows α ∈ [0.1, 2.0] affects results significantly. Was any optimization performed? Try layer-wise or learned scaling?
7. Does Shadow-FT preserve instruction-following ability? Evaluate on IFEval, MT-Bench, and AlpacaEval.
8. All experiments are in English. How does Shadow-FT handle morphologically rich languages?
9. Is Shadow-FT computationally cheaper than alternatives?
10. Why don't Coverage gains (Table 1) translate to Best-of-N gains (Table 2)? Investigate selection methods and sample diversity.

### Soundness
2

### Presentation
3

### Contribution
2
