=== CALIBRATION EXAMPLE 29 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title clearly reflects the paper's focus on analyzing SFT training dynamics to mitigate overfitting. The abstract succinctly states the problem (SFT's memorization and poor OOD generalization), the approach (selective fine-tuning of Transformer modules), and the key findings (attention-only SFT improves OOD generalization, matching RL methods). Claims are specific and appear supported by the results described later. However, the claim that attention-only SFT achieves performance "comparable to state-of-the-art reinforcement learning (RL) alignment methods" is a strong comparative statement that must be rigorously validated in the experiments.

### Introduction & Motivation
The introduction effectively motivates the problem: SFT is prone to memorization, while RL generalizes better, and prior work suggests differing parameter update patterns. The research question—which Transformer modules drive memorization—is clearly derived from this motivation. However, the list of contributions is cut off (appearing as "1" alone), which is likely a parser artifact but must be corrected. From the text, the contributions are implicit but should be explicitly enumerated for clarity.

### Related Work
The related work adequately covers OOD generalization in SFT and the roles of Transformer modules (attention vs. FNNs). It positions the paper's novel focus on module-level contributions to memorization and OOD generalization. A minor gap: the discussion could engage more with parameter-efficient fine-tuning (PEFT) literature (e.g., adapters, LoRA) which also selectively updates parameters, to clarify how this work differs (focus on module types rather than parameter efficiency per se). This is not critical but would strengthen the framing.

### Methodology
The core methodology—selective fine-tuning of FNN-only, attention-only, or full-model—is straightforward and reproducible. However, several key details are missing or unclear, which hinders reproducibility and raises questions:
1. **Module definitions**: Exactly which parameters are included in "attention layers" and "FNNs"? For attention, does this include query, key, value, and output projections, or only attention weights? For FNNs, are layer norms included? This must be specified.
2. **Parameter matching**: Equation (3) and the surrounding text describe matching the number of trainable parameters by selecting the first \(L\) layers for FNN-only or full fine-tuning. However, it is ambiguous what "module \(M\)" refers to—presumably, for FNN-only, \(M\) is FNNs, but then \(N(\theta_{0:l}^M)\) would only count FNN parameters in those layers, not attention parameters. The description needs to be precise.
3. **FLOPs matching**: The paper states that total training FLOPs are matched across conditions by adjusting iterations. However, FLOPs depend on forward and backward passes, and different sets of trainable parameters affect backward pass FLOPs. The method for calculating FLOPs should be explained to ensure the comparison is fair.
4. **Learning rate experiments**: The learning rate analysis is presented only for full fine-tuning. To strengthen the claim that module choice is key, it would be informative to see how learning rate interacts with module selection (e.g., does a small LR also help FNN-only?).

### Experiments & Results
The experimental design uses two controlled benchmarks (GP and V-IRL) suitable for probing memorization vs. generalization. However, several issues limit the strength of the evidence:

1. **Comparison to RL methods**: Table 2 claims attention-only SFT outperforms SFT+RL from Chu et al. (2025). This is a notable result, but the comparison is limited to a single RL method on two tasks. To support a claim about "state-of-the-art RL alignment methods," more diverse RL baselines (e.g., DPO, PPO) and tasks are needed. Additionally, the RL method in Chu et al. uses SFT as a warm-start; a fairer comparison might be attention-only SFT vs. attention-only SFT + RL. The paper should temper or broaden this claim.

2. **Statistical robustness**: No measures of variance (e.g., standard deviations across multiple seeds) are reported. Given the instability often seen in fine-tuning, results should be averaged over several runs with error bars or confidence intervals.

3. **Ablations**: The study focuses on FNN vs. attention, but other components (e.g., layer norms, embeddings) are not ablated. While this focus is justified, a brief discussion of why these are excluded would be helpful. Also, combining modules (e.g., attention + some FNN layers) could provide further insight.

4. **Figure interpretation**: The figures are referenced but not included in the text. Assuming they are provided in the submission, the descriptions are clear. However, from the text, Figure 2 shows attention-only outperforming even in-distribution under matched parameters, which seems to contradict the hypothesis that FNNs are better for memorization. This point warrants deeper discussion.

5. **Model choice**: Using Llama-3.2-Vision-11B, a vision-language model, for language-centric tasks may introduce confounding factors. Is the vision encoder used? If not, why choose this model? A pure language model would be more appropriate, or justification is needed.

6. **Learning rate analysis**: The finding that small LRs improve OOD generalization for full fine-tuning is interesting, but it is not integrated with the module selection story. Does a small LR also mitigate the overfitting of FNN-only fine-tuning? This could test whether LR and module selection are independent factors.

### Writing & Clarity
The writing is generally clear, though some sections suffer from parser artifacts (e.g., cut-off contributions, broken formatting in equations). The methodology section needs more precise descriptions as noted. The flow is logical, and the paper is easy to follow overall.

### Limitations & Broader Impact
The paper lacks a dedicated limitations section. Important limitations to acknowledge include:
- The study is limited to two rule-based reasoning tasks; findings may not generalize to more complex, open-ended tasks.
- The model architecture is specific (Llama-3.2-Vision); results might vary with other architectures (e.g., encoder-decoder models).
- Selective fine-tuning might not be optimal if the task requires updating factual knowledge stored in FNNs.
- Broader impact: Improving OOD generalization could lead to more robust and reliable LLMs, but potential negative societal impacts (e.g., if selective fine-tuning inadvertently preserves harmful biases from pretraining) are not discussed.

### Overall Assessment
The paper presents a timely and interesting investigation into the module-level dynamics of SFT memorization. The core finding—that attention-only fine-tuning preserves OOD generalization while FNN-only or full fine-tuning leads to overfitting—is novel and potentially impactful for guiding fine-tuning practices. However, the evidence is currently limited by a narrow set of tasks and baselines, missing statistical robustness measures, and insufficient methodological detail for full reproducibility. The claim about matching RL methods is overstated without broader RL comparisons. If these issues are addressed—particularly by expanding experiments, reporting variance, clarifying methodology, and adding a limitations discussion—the contribution would be strong for ICLR. As it stands, the paper provides valuable insights but requires substantial strengthening to meet ICLR's high standards.

# Neutral Reviewer
## Balanced Review

### Summary
This paper investigates which Transformer modules—attention layers versus feedforward networks (FNNs)—are primarily responsible for memorization and poor out-of-distribution (OOD) generalization during supervised fine-tuning (SFT) of large language models. Through controlled experiments on rule-based reasoning tasks (GeneralPoints and V-IRL), the authors demonstrate that fine-tuning only attention layers preserves or improves OOD generalization, while fine-tuning FNNs or the full model leads to significant memorization and generalization collapse. A secondary finding is that using smaller learning rates in full fine-tuning can also mitigate OOD degradation.

### Strengths
1. **Well-Designed, Controlled Experiments**: The study uses two established, rule-based benchmarks (GeneralPoints and V-IRL) that allow for precise control over in-distribution and out-of-distribution conditions. This is crucial for cleanly isolating memorization from genuine reasoning generalization.
2. **Rigorous Comparison Across Conditions**: The authors thoughtfully address the confounder of parameter count by running experiments under two settings: (a) matched total training FLOPs (with varying numbers of training steps) and (b) matched numbers of trainable parameters (by fine-tuning only a subset of layers). This strengthens the claim that the observed effects are due to the function of the modules, not just their capacity or the optimization budget.
3. **Clear, Actionable Findings**: The core result—that attention-only SFT substantially outperforms FNN-only or full SFT on OOD tasks—is presented clearly with supporting figures. The additional analysis on learning rate provides a complementary, practical lever for controlling memorization.

### Weaknesses
1. **Limited Mechanistic Explanation**: While the paper convincingly shows *what* happens (attention layers generalize better), it offers only a high-level, speculative explanation *why* (e.g., FNNs as "key-value memories" are prone to overwriting). A deeper analysis of the actual changes in attention patterns or FNN activations would significantly strengthen the contribution. The claim that this finding "provides new insights into the mechanisms" is partially supported but not deeply explored.
2. **Narrow Scope of Evaluation**: The conclusions are drawn from only two tasks (arithmetic reasoning and navigation). While they are well-chosen, generalizing the findings to broader domains (e.g., creative writing, code generation, factual QA) remains an open question. The paper would be more impactful with evidence from a more diverse task suite.
3. **Superficial Comparison to RL Methods**: The abstract and results claim attention-only SFT performs "comparable to state-of-the-art RL alignment methods," but this comparison is relegated to a single table (Table 2) without a discussion of the trade-offs (e.g., sample efficiency, stability, reward hacking). A more thorough discussion positioning selective SFT within the broader alignment landscape is needed.

### Novelty & Significance
**Novelty**: The work builds directly on recent findings (e.g., Chu et al. 2025; Mukherjee et al. 2025) about differences between SFT and RL. Its novel contribution is isolating the effect *within* the SFT paradigm to specific Transformer modules. While the idea of selective fine-tuning is not new, applying it to study memorization vs. generalization dynamics in rule-based reasoning tasks provides a fresh perspective.
**Significance**: For the research community, the paper offers a more nuanced view of SFT's failures, moving beyond "SFT overfits" to "SFT overfits primarily when updating FNNs." For practitioners, it suggests a simple, computationally efficient intervention (attention-only SFT or lower learning rates) to improve OOD robustness. If validated more broadly, this could influence standard fine-tuning protocols.

### Suggestions for Improvement
1. **Deepen the Analysis of "Why"**: Conduct a targeted analysis to bolster the mechanistic claim. For example, probe how attention head outputs or FNN activation distributions change for ID vs. OOD inputs under different fine-tuning regimes. Alternatively, analyze the rank/gradient norms of updates to each module type to link sparsity to generalization.
2. **Broaden the Empirical Validation**: Include at least one more task family (e.g., a symbolic reasoning benchmark like GSM8K or a factual knowledge editing task) to test the generality of the conclusion that attention layers are the "generalization" module. This would greatly increase confidence in the external validity of the findings.
3. **Expand the Discussion and Positioning**: Elaborate on the comparison to RL methods. Discuss scenarios where attention-only SFT might be preferable (e.g., data efficiency, simplicity) or insufficient (e.g., tasks requiring explicit reward optimization). Also, explicitly discuss limitations, such as whether this finding holds for encoder-decoder models or models of different scales.
4. **Improve Presentation Clarity**: Some sections, particularly the methodology (Sections 3 & 4), are slightly fragmented due to potential parser issues. Ensure the final manuscript has a smooth flow, with all figures and tables referenced correctly and equations properly formatted. The contribution list in the introduction is incomplete (marked as "1" only).

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Compare against standard parameter-efficient fine-tuning (PEFT) baselines (e.g., LoRA, prefix-tuning).** The claim that "selective SFT" is a promising direction is incomplete without showing it outperforms widely-used, modular PEFT methods that also update sparse parameter sets.
2. **Validate on a broader suite of reasoning and instruction-following benchmarks.** The paper's claims are built on two synthetic tasks (GP, V-IRL). To argue this is a general strategy for improving SFT generalization, results on more diverse, real-world OOD tasks (e.g., MMLU, BBH, GSM8K) are necessary.
3. **Ablation on layer selection within "attention-only" tuning.** The paper fine-tunes all attention layers. An ablation freezing early/late attention blocks is needed to show if the effect is localized or distributed, which is critical for mechanistic understanding.
4. **Direct comparison to RL fine-tuning (e.g., PPO, DPO) on the same tasks and compute budget.** The abstract claims attention-only SFT matches RL methods, but Table 2 cites prior RL results. A controlled, head-to-head experiment is required to support this strong claim.

### Deeper Analysis Needed (top 3-5 only)
1. **Analyze the *type* of knowledge changed or preserved.** To substantiate the hypothesis that FNNs are "long-term memory stores" whose update causes overfitting, analyze weight divergence (e.g., L2 distance from pre-trained weights) per module and correlate with OOD drop.
2. **Quantify memorization directly, not just via OOD performance.** Use metrics like train-data exposure scores or canary insertion tests to directly measure memorization for each fine-tuning strategy, separating it from generalization failure.
3. **Investigate why lower learning rate helps in the full fine-tuning setting.** The learning rate result (Fig 3) is presented but not explained. Analysis of loss landscapes, gradient noise, or effective step sizes per module is needed to connect this hyperparameter to the module-level story.
4. **Check if results hold across different model scales and families.** The conclusion is drawn from a single 11B vision-language model. A scaling law analysis or tests on purely textual models (e.g., LLaMA, Mistral) is needed to ensure findings are not architecture-specific.

### Visualizations & Case Studies
1. **Visualize attention pattern changes before/after fine-tuning for successful (attention-only) vs. failed (FNN-only) OOD cases.** This would show if attention heads adapt to implement generalizable rules versus memorizing surface patterns.
2. **Case studies of model outputs on OOD failures.** Provide qualitative examples showing how FNN-only/full fine-tuning models fail (e.g., regurgitating training templates) versus how attention-only models attempt generalizable reasoning.

### Obvious Next Steps
1. **Scale the experiments to larger models (e.g., 70B parameters).** ICLR expects evidence that findings are not limited to medium-scale models, especially when making broad claims about Transformer modules.
2. **Combine attention-only fine-tuning with a low learning rate.** The two main findings (module selection and LR) are presented separately. The logical next step is to combine them and test for additive benefits, which should have been in the paper.
3. **Discuss practical trade-offs and limitations.** The paper promotes "attention-only" SFT but does not discuss potential downsides, such as slower in-distribution convergence (Fig 2) or whether it harms capabilities outside the fine-tuning domain. A balanced discussion is required.

# Final Consolidated Review
## Summary
This paper investigates which Transformer modules drive memorization during supervised fine-tuning (SFT) of large language models. Through controlled experiments on two rule-based reasoning benchmarks, it demonstrates that fine-tuning only attention layers preserves out-of-distribution (OOD) generalization, whereas fine-tuning feedforward networks (FNNs) or the full model leads to overfitting and generalization collapse. A secondary finding shows that lower learning rates can also mitigate OOD degradation in full fine-tuning.

## Strengths
- **Rigorous and controlled experimental design.** The study uses two established benchmarks (GeneralPoints and V-IRL) that allow precise control over in-distribution and out-of-distribution rules, cleanly isolating memorization from generalization. The authors thoughtfully address confounders by comparing setups with matched total training FLOPs and with matched numbers of trainable parameters, strengthening the causal claim about module function.
- **Clear, actionable core finding.** The result that attention-only SFT substantially outperforms FNN-only or full SFT on OOD tasks is demonstrated consistently across both benchmarks and under both matched-compute and matched-parameter settings. This provides a simple, practical intervention for improving SFT robustness.

## Weaknesses
- **Limited mechanistic explanation.** While the paper convincingly shows *that* attention-only tuning generalizes better, the explanation *why* remains high-level and speculative (e.g., citing prior work that FNNs act as key-value memories). A deeper analysis of how attention patterns or FNN activations change for ID vs. OOD inputs under each regime would significantly strengthen the contribution and justify the claim of providing "new insights into the mechanisms."
- **Narrow empirical scope.** The conclusions are drawn from only two synthetic, rule-based reasoning tasks. Generalization to broader domains (e.g., open-ended generation, factual QA, or code) is not established, limiting confidence in the external validity of the findings for general SFT practice.
- **Incomplete comparison to reinforcement learning (RL).** The abstract claims attention-only SFT performs "comparable to state-of-the-art RL alignment methods," but this is supported by a single comparison to prior RL results on the same two tasks. A more thorough discussion of trade-offs (e.g., sample efficiency, stability) and a direct, controlled comparison with standard RL methods (e.g., PPO, DPO) would be needed to substantiate this strong claim.
- **Lack of statistical robustness measures.** Results are presented without any indication of variance (e.g., standard deviations across multiple random seeds). Given the known instability of fine-tuning, reporting such measures is important for assessing the reliability of the presented trends.

## Nice-to-Haves
- **Experiment with parameter-efficient fine-tuning (PEFT) baselines.** Comparing attention-only SFT against widely-used PEFT methods like LoRA would help position its practical utility.
- **Ablation on layer selection.** Investigating whether the benefits of attention-only tuning are localized to specific layers (e.g., early vs. late) could offer finer-grained mechanistic insight.
- **Direct quantification of memorization.** Supplementing OOD performance with metrics that directly measure memorization (e.g., train-data exposure scores) could more cleanly separate memorization from other generalization failures.
- **Investigate interaction between module selection and learning rate.** Exploring whether a low learning rate can also rescue FNN-only fine-tuning would help unify the two main findings.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Criticism that Figure 2 contradicts the hypothesis.** The figure shows attention-only tuning also excels in-distribution under matched parameters; this demonstrates its efficiency, not a contradiction.
- **Criticism that the model choice (Llama-3.2-Vision) is inappropriate.** The paper uses a standard, publicly available LLM backbone for language tasks; this is a reasonable choice, and the vision capability is not used in a confounding way for the language-only benchmarks studied.
- **Demand for ablations of other components (e.g., layer norms, embeddings).** The paper explicitly scopes its investigation to attention vs. FNN modules; demanding analysis of other components is scope creep.
- **Request for scaling experiments to 70B models.** The use of an 11B model is sufficient for the study's purpose; this demand imposes an arbitrary rigor requirement not standard for an initial empirical investigation.

## Novel Insights
The paper's novel contribution is isolating the differential roles of Transformer modules within the SFT paradigm. Building on prior work that contrasts SFT with RL, it shows that the overfitting characteristic of SFT is primarily driven by updates to feedforward network layers, while updating attention layers preserves—and can even enhance—out-of-distribution generalization. This refines the common understanding of "SFT memorizes" to a more precise, module-level narrative and suggests selective updating as a targeted intervention for improving generalization.

## Suggestions
- **Clarify methodological details in Section 3.** Precisely specify which parameters constitute "attention layers" and "FNNs" (e.g., query/key/value/output projections, MLP weights) to aid reproducibility.
- **Report variance estimates.** Run experiments with multiple random seeds and include standard deviations or confidence intervals in results tables and figures.
- **Add a limitations section.** Acknowledge the restricted task domain, the use of a single model family, and that the findings may not apply to tasks requiring factual knowledge updates stored in FNNs.

# Actual Human Scores
Individual reviewer scores: [2.0, 0.0, 4.0, 0.0]
Average score: 1.5
Binary outcome: Reject
