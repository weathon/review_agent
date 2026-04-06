=== CALIBRATION EXAMPLE 23 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title is clear and reflects the paper's core inquiry. The abstract succinctly states the problem (SFT's tendency to memorize), the method (selective fine-tuning of Transformer modules), and the main finding (attention-only SFT improves OOD generalization). However, the claim that attention-only SFT achieves "performance comparable to state-of-the-art reinforcement learning (RL) alignment methods" is not substantiated in the abstract and is only weakly supported later in the paper (Table 2). The abstract should more accurately reflect the paper's contribution, which is a comparative analysis of modules within SFT, not a direct SFT-vs-RL benchmark.

### Introduction & Motivation
The introduction effectively motivates the problem by linking the observed overfitting in SFT to the density of parameter updates, contrasting it with the sparse updates from RL (citing Mukherjee et al., 2025). The central question—which Transformer modules drive memorization—is well-posed. However, the listed contributions are incomplete (the text cuts off at "1."). This is a significant clarity issue. The reader must infer the contributions from the rest of the paper. The introduction should be revised to explicitly list 2-3 clear contributions.

### Methodology
The core methodology is selective fine-tuning (FNN-only, Attention-only, Full-model). The description of the SFT objective is standard. Major concerns arise in the experimental design for fair comparison:
1.  **"Parameter not matched" vs. "Parameter matched"**: The rationale for having two settings is sound (isolating the effect of module type from parameter count). However, the implementation of the "parameter-matched" setting is problematic. Equation (3) describes matching the number of parameters in *all* attention layers to the parameters in the *first L layers* of another module. This compares a **global** module (all attention layers) to a **localized** subset (first L layers of FNN or full blocks). This confounds module type with layer locality, a critical confounding variable. The fair comparison should match global module to global module (e.g., all FNN params vs. all Attn params by adjusting layer count for *both*).
2.  **FLOPs Matching**: The justification for matching total training FLOPs is good for computational fairness. However, the paper does not clarify if this matching is done for both the "parameter not matched" and "parameter matched" experiments. Figure 1's caption suggests FLOPs matching, while Figure 2 discusses parameter matching. The relationship between these two control strategies needs to be explicitly stated.
3.  **Reproducibility**: The selective freezing procedure is clear in principle. However, for full reproducibility, the paper should specify which specific parameters within attention layers (query, key, value, output projections, attention biases) and FNNs (up/down/gate projections, biases) were made trainable.

### Experimental Setup
The choice of benchmarks (GeneralPoints, V-IRL) follows prior work (Chu et al., 2025) and is appropriate for controlled OOD evaluation. The model (Llama-3.2-Vision-11B) is a relevant, modern foundation model. The evaluation metrics (success rate, per-step accuracy) are standard for these tasks.
A minor point: For V-IRL, which uses a vision-language model, it is unclear if the visual encoder is frozen or fine-tuned. This should be specified, as it impacts the parameter count and the interpretation of "FNN-only" tuning.

### Results and Discussion
This is the core of the paper, but several issues weaken the evidence.
1.  **Figure Interpretation**: The description of Figures 1 and 2 contains inconsistencies. For instance, the text states that in Figure 1 (GP, OOD), "attention-only fine-tuning maintains success rates above 10%," but the corresponding figure (which is not provided, only described) would need to show this clearly. More critically, the text describing Figure 2 (parameter-matched) says attention-only fine-tuning reaches "close to 100% success rate" on in-distribution, but then states it maintains "10-15%" on OOD. The magnitude of this generalization gap is enormous and warrants deeper discussion. Is a model that achieves 100% ID but only 15% OOD considered a success for "generalization"?
2.  **Statistical Significance & Reporting**: Results are presented as single runs on line plots. There is no mention of standard deviations across random seeds or multiple data splits. For claims about the superiority of one method over another (especially in Table 2), confidence intervals or statistical tests are essential for ICLR.
3.  **Comparison to RL Baselines**: Table 2 is arguably the most impactful claim but is presented with insufficient context. It compares SFT variants to "SFT (FFT) + RL" from Chu et al. The results show attention-only SFT scoring higher. However:
    *   Are these results at the same computational budget? The RL method likely requires significantly more samples/compute.
    *   Are the SFT results using the optimal low learning rate from Section 5.2? This is not stated.
    *   The improvement (e.g., 94.13 vs. 91.8 on V-IRL) is small. Without statistical significance, it's hard to claim "performs on par with or better than."
    *   This comparison distracts from the paper's primary module-analysis contribution. It should either be robustly expanded into a major section or de-emphasized.
4.  **Learning Rate Analysis (Section 5.2)**: This is a valuable addition, showing that low LR can mitigate OOD collapse. However, it is presented for "full fine-tuning" only. The natural and crucial follow-up question is: **Does a low learning rate also help FNN-only fine-tuning?** If a low LR allows full fine-tuning to generalize, is the primary issue with FNNs simply that they are more sensitive to LR? This missing ablation is a significant gap.

### Writing & Clarity
The writing is generally clear. The major clarity issue is the incomplete contributions list in the introduction. Figure/table references are clear, though the actual figures are missing from the text (this is understood to be a parser issue). The narrative flow from motivation to results is logical.

### Limitations & Broader Impact
The paper lacks a dedicated limitations section, which is a notable omission for ICLR. Key limitations that should be acknowledged:
1.  **Task Generality**: Experiments are on two structured, rule-based reasoning tasks. The findings may not extend to broader instruction-following or creative generation tasks where the FNN-as-memory paradigm might be more beneficial.
2.  **Architectural Specificity**: Results are shown for one model family (Llama). It is unknown if this holds for other architectures (e.g., models with MoE, different normalization schemes).
3.  **Confounding in Matched Experiments**: As noted, the parameter-matched design confounds module type with layer depth.
4.  **Lack of Theoretical Explanation**: The paper provides a compelling empirical pattern but only a high-level, post-hoc explanation (FNNs as memory, attention as context processor). A deeper mechanistic analysis (e.g., probing what knowledge is changed in each module) is left for future work.
The broader impact statement is minimal. The positive impact (improved, efficient fine-tuning) is stated. Potential negatives (e.g., if selective tuning makes models easier to maliciously edit or forget safety training) are not discussed.

### Overall Assessment
The paper addresses a timely and important question: understanding and mitigating overfitting in SFT through the lens of Transformer modules. The central empirical finding—that attention-only fine-tuning leads to better OOD generalization than FNN-only or full fine-tuning—is novel, interesting, and potentially impactful for practice. However, the work in its current form has significant methodological and presentation flaws that prevent it from being a strong ICLR accept. The parameter-matching experiment is confounded, the comparison to RL is underdeveloped and potentially misleading, critical ablations (LR effect on FNN-only) are missing, and statistical rigor is absent. The contribution stands as a promising empirical observation, but it requires a substantial revision to solidify the evidence, properly contextualize the claims, and acknowledge limitations before it meets ICLR's standard for acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper investigates how different Transformer modules (attention vs. feedforward networks) contribute to memorization and out-of-distribution (OOD) generalization during supervised fine-tuning (SFT) of large language models. Through controlled experiments on two rule-based reasoning benchmarks (GeneralPoints and V-IRL), the authors find that fine-tuning only attention layers preserves or improves OOD performance, while fine-tuning FNNs or the full model leads to severe generalization collapse due to memorization. The paper further shows that using smaller learning rates can mitigate memorization in full fine-tuning.

### Strengths
1. **Clear, Controlled Experimental Design:** The paper uses two well-defined, rule-based reasoning benchmarks (GeneralPoints and V-IRL) that allow for precise manipulation of in-distribution vs. out-of-distribution evaluation. This setup is ideal for isolating memorization effects from genuine generalization.
2. **Rigorous and Fair Comparisons:** The authors carefully control for total compute budget (FLOPs) and also conduct experiments with a matched number of trainable parameters (via Equation 3). This strengthens the claim that the observed effects are due to the functional roles of the modules, not just differences in parameter count or optimization steps.
3. **Actionable Insights:** The findings are practical and immediately useful. Demonstrating that attention-only SFT can achieve OOD performance comparable to or better than more complex SFT+RL pipelines provides a simple, efficient strategy for practitioners to improve generalization. The analysis of learning rate effects further offers a concrete hyperparameter tuning guideline.
4. **Connects to Prior Literature:** The work is well-motivated by recent findings on the sparsity of RL updates versus the density of SFT updates, and the hypothesized roles of FNNs as key-value memories. It effectively builds upon and tests these ideas.

### Weaknesses
1. **Limited Task and Model Scope:** The conclusions are drawn from only two reasoning tasks (arithmetic and navigation) and a single model family (Llama-3.2-Vision-11B). ICLR typically expects evidence of robustness across a broader range of tasks (e.g., coding, commonsense QA, creative writing) and model architectures to ensure findings are generalizable.
2. **Mechanistic Explanation is Lacking:** While the empirical results are compelling, the paper offers only a high-level, speculative explanation for *why* attention-only tuning generalizes better (e.g., "attention supports context-sensitive adaptation"). A deeper analysis—such as probing what knowledge is retained/changed in FNN vs. attention weights, or analyzing attention pattern shifts—would significantly strengthen the contribution.
3. **Overstatement of RL Comparison:** The abstract and results claim attention-only SFT performs "comparable to state-of-the-art RL alignment methods." However, Table 2 compares against only one baseline (Chu et al. 2025's SFT+RL). A more comprehensive comparison with a wider array of modern RLHF/DPO methods would be necessary to substantiate this strong claim for an ICLR audience.
4. **Practical Applicability Remains Unclear:** The paper does not discuss the potential downsides or limitations of attention-only fine-tuning for real-world, broad-instruction tuning. For example, could it hinder the model's ability to learn new factual knowledge or stylistic patterns that might reside in FNNs? A discussion of these trade-offs is important.

### Novelty & Significance
**Novelty:** The core idea of selective parameter fine-tuning is not new (e.g., LoRA, prefix-tuning). However, the specific focus on *architectural modules* (attention vs. FNN) and their differential impact on *OOD generalization* (as opposed to mere parameter efficiency or in-distribution performance) provides a novel and insightful angle. The link between FNN updates and memorization collapse offers a fresh perspective on SFT's limitations.
**Significance:** The findings are significant for the LLM alignment and fine-tuning community. If robust, they suggest a straightforward and compute-efficient method (attention-only SFT) to mitigate a major known weakness of SFT (overfitting), potentially simplifying the post-training pipeline. The work also advances our understanding of Transformer internals during adaptation.

### Suggestions for Improvement
1. **Broaden the Empirical Scope:** Include experiments on 1-2 additional model families (e.g., a decoder-only model like Gemma and an encoder-decoder model) and 1-2 more diverse task types (e.g., a knowledge-intensive QA dataset or a style transfer task) to convincingly demonstrate the generality of the findings.
2. **Deepen the Analysis:** Add a section analyzing *how* the weight changes differ. For instance, compute and compare the weight change norms/distributions in FNN vs. attention layers for the different fine-tuning strategies. Use probing classifiers or causal mediation analysis to trace how information flow changes, providing a mechanistic explanation for the observed generalization benefits.
3. **Strengthen the RL Baseline Comparison:** Expand Table 2 to include results from standard RLHF (PPO) and popular offline RL (DPO, KTO) baselines on the same benchmarks. This would provide a more rigorous and complete picture of where attention-only SFT stands relative to the alignment methods it claims to match.
4. **Discuss Limitations and Trade-offs Explicitly:** Add a subsection discussing scenarios where attention-only SFT might be suboptimal (e.g., when the task requires memorizing new factual associations) and whether a hybrid or staged approach (e.g., attention-first, then gentle FNN tuning) could be beneficial. This would provide a more nuanced view for practitioners.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Direct comparison to standard parameter-efficient fine-tuning (PEFT) methods like LoRA or prefix tuning.** The paper claims attention-only SFT is a superior strategy, but without comparing to widely adopted PEFT methods that also sparsely update parameters (often attention layers), it's unclear if the benefit is unique to attention-only updates or a general property of sparse updates. This is critical for claiming novelty and practical value.
2. **Ablation across model scales and architectures.** All experiments use a single model (Llama-3.2-Vision-11B). The core claim about Transformer module roles must be validated across different model families (e.g., pure decoder vs. encoder-decoder) and sizes to ensure the findings are not architecture- or scale-specific.
3. **Explicit measurement of memorization (e.g., training data leakage, loss gap).** The paper infers memorization from OOD performance drops. To directly support the claim that FNN updates cause memorization, they must measure memorization explicitly (e.g., via canary insertion, training loss vs. validation loss, or verbatim reproduction of training examples). Without this, the memorization explanation is speculative.
4. **Direct head-to-head comparison with RL fine-tuning under identical selective settings.** The paper claims attention-only SFT matches RL performance, but only cites prior RL results. They must run RL (e.g., PPO/DPO) on the same tasks and model, and also analyze which modules RL updates, to properly attribute the generalization benefit to module selection rather than algorithmic differences.

### Deeper Analysis Needed (top 3-5 only)
1. **Mechanistic analysis of why attention-only updates preserve generalization.** The paper hypothesizes that attention supports context-sensitive adaptation while FNNs store long-term memory, but provides no analysis of weight changes or attention patterns. They should analyze the magnitude and distribution of updates in each module (e.g., gradient norms, weight deviations) and correlate them with OOD performance to substantiate the claimed mechanism.
2. **Interaction analysis between learning rate and module selection.** The effects of learning rate and module choice are presented separately. It is crucial to test whether a low learning rate can rescue full or FNN-only fine-tuning, or if attention-only's advantage is independent of learning rate. This would clarify whether module selection or update magnitude is the primary factor.
3. **Quantification of pre-trained knowledge retention.** The claim that attention-only preserves generalizable pre-trained knowledge requires measuring performance on diverse pre-training benchmarks (e.g., MMLU, HellaSwag) before and after each fine-tuning strategy. Without this, it's unclear if the OOD benefit comes from preserving prior knowledge or another factor.

### Visualizations & Case Studies
1. **Visualization of weight change distributions per module (e.g., histograms of gradient norms).** This would directly show whether attention updates are indeed more sparse or localized compared to FNN updates, supporting the hypothesis that RL-like sparse updates underlie the generalization benefit.
2. **Case studies contrasting attention-only vs. FNN-only predictions on OOD examples.** Concrete examples of model reasoning steps (e.g., chain-of-thought) for both settings would reveal whether attention-only leads to more flexible reasoning strategies and where FNN-only fails due to rigid memorization.

### Obvious Next Steps
1. **Expand experiments to more diverse OOD benchmarks, including natural distribution shifts.** The current rule-based synthetic tasks limit the generality of the findings. Testing on more realistic OOD scenarios (e.g., domain shift in dialogue, style variation) is necessary to claim broad applicability for improving SFT generalization.
2. **Compare to layer-wise fine-tuning baselines (e.g., tuning only top layers).** The paper only compares module types, but a common practice is to tune only later layers. Without comparing to this standard baseline, it's unclear if attention-only is better than simply tuning fewer layers, regardless of module type.
3. **Investigate the effect of dataset size and diversity on module-specific memorization.** The paper uses fixed datasets. Understanding whether the benefits of attention-only tuning hold with varying data size and diversity is essential for practical guidance.

# Final Consolidated Review
## Summary
This paper investigates how different Transformer modules contribute to memorization and out-of-distribution (OOD) generalization during supervised fine-tuning (SFT). Through controlled experiments on two rule-based reasoning benchmarks, it finds that fine-tuning only attention layers preserves or improves OOD performance, while fine-tuning feedforward networks (FNNs) or the full model leads to generalization collapse. It further shows that smaller learning rates can mitigate memorization in full fine-tuning.

## Strengths
- **Clear and controlled experimental design** isolates the effect of module type by matching total training FLOPs and conducting parameter-count-matched experiments, strengthening the claim that the observed differences are due to functional roles rather than capacity or optimization steps.
- **Actionable and novel finding** that attention-only SFT substantially improves OOD generalization compared to FNN-only or full fine-tuning, providing a simple, efficient strategy to mitigate a known weakness of SFT. The additional analysis showing that smaller learning rates act as a regularizer further enhances practical utility.

## Weaknesses
- **Methodological confounding in parameter-matched experiments.** Equation (3) matches the number of parameters in *all* attention layers to those in the *first L layers* of another module (FNN or full blocks). This confounds module type with layer locality, making it difficult to attribute effects solely to module function.
- **Underdeveloped and potentially overstated comparison to RL methods.** Table 2 claims attention-only SFT performs comparably to or better than an SFT+RL baseline, but this comparison lacks crucial context: it does not address differences in computational budget, does not establish statistical significance for the small improvements shown, and uses only a single RL baseline. The strong claim in the abstract is not fully substantiated.
- **Limited exploration of the interaction between learning rate and module choice.** Section 5.2 shows that low learning rates help full fine-tuning generalize better but only analyzes this for the full model. A critical missing ablation is whether a low learning rate can similarly rescue FNN-only fine-tuning, which is necessary to disentangle whether the primary issue is the module being updated or the magnitude of the updates.

## Nice-to-Haves
- Comparison to standard parameter-efficient fine-tuning (PEFT) methods like LoRA would help situate the practical value of attention-only tuning within the existing landscape of sparse update techniques.
- Experiments on a broader set of tasks (beyond two controlled reasoning benchmarks) and model architectures would strengthen the claim of general applicability.
- A deeper mechanistic analysis (e.g., of weight change distributions or attention pattern shifts) could provide stronger evidence for the hypothesized roles of FNNs as memory stores and attention as context processors.

## Novel Insights
The paper provides a novel and empirically grounded insight: during SFT, updating attention layers preserves OOD generalization, while updating FNNs primarily drives memorization and collapse. This finding refines the common understanding that SFT broadly overfits, pinpointing the feedforward networks as the key locus of this problem. It also suggests that the generalization benefits of RL alignment might be partly attributable to its sparser update pattern, which this work shows can be mimicked in a supervised setting through selective module updates.

## Suggestions
- Revise the parameter-matched experiment design to compare global module to global module (e.g., match parameters by adjusting the number of layers updated for both attention and FNN modules across the entire network) to remove the layer-locality confound.
- Either robustly expand the RL comparison (with matched compute, multiple baselines, and significance testing) or temper the claim in the abstract and discussion to focus on the core module-analysis contribution.
- Conduct the missing ablation: apply the low learning rates from Section 5.2 to the FNN-only fine-tuning condition to see if it also improves OOD generalization.
- Fix the incomplete contributions list in the introduction (Section 1).

# Actual Human Scores
Individual reviewer scores: [2.0, 0.0, 4.0, 0.0]
Average score: 1.5
Binary outcome: Reject
