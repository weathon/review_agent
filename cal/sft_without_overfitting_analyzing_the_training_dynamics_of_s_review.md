=== CALIBRATION EXAMPLE 23 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title "SFT Without Overfitting" is somewhat misleading — the paper does not eliminate overfitting but rather shows that certain selective fine-tuning choices *reduce* it. The abstract is coherent but makes a strong claim: attention-only SFT achieves "performance comparable to state-of-the-art reinforcement learning (RL) alignment methods." Looking at Table 2, attention-only SFT scores 94.13 vs. 91.8 on V-IRL and 19.23 vs. 15.0 on GP against SFT+RL. On GP, the OOD absolute performance is 19.23% — barely above chance-level for a structured reasoning task — and the gap versus RL is 4.23 percentage points from a very low baseline. "Comparable to state-of-the-art RL" significantly overstates what has been demonstrated.

---

### Introduction & Motivation (Section 1)

**Critical structural defect:** The paper explicitly introduces a contribution list with "Our primary contributions are as follows:" but no bullet points follow — the list is entirely absent, jumping immediately to "Together, these findings provide new insights..." This is not a PDF parsing artifact; the numbered-list content is simply missing. At ICLR this is a serious presentation failure.

The motivation draws on Mukherjee et al. (2025) for the claim that RL updates only 5–30% of parameters, but this is an arXiv preprint dated May 2025 (2505.11711). The introduction's reasoning — "RL updates few parameters → preserves generalizable knowledge; SFT updates many → overfits" — is qualitatively plausible but the leap to "therefore attention layers are the culprit" is not clearly motivated. Why attention vs. FNNs specifically? The introduction asserts the FNN-as-long-term-memory hypothesis as established fact, but this is one interpretation of Geva et al. (2021) applied in a very different context.

---

### Related Work (Section 2)

The related work is thin and narrowly scoped. Notably absent is any engagement with the **parameter-efficient fine-tuning (PEFT) literature** — LoRA (Hu et al., 2022), adapters, prefix-tuning — all of which involve selective updating of specific modules (LoRA by default modifies attention projection matrices). This is a major omission: the finding that attention-only tuning generalizes better is directly adjacent to why LoRA practitioners freeze FNNs, yet this connection is never drawn. The paper also does not discuss **early stopping** or **regularization** approaches for reducing SFT memorization, which are natural baselines.

The "Positioning of Our Work" paragraph restates the hypothesis without engaging with contradictory evidence or alternative interpretations.

---

### Methodology (Section 3)

**Incomplete equations:** Section 3.1 introduces Equations 1 and 2, but Equation 2 (the token-level loss) appears displaced and fragmented in the text (Section 3.2 header is inserted between the two equations). More importantly, the paper never specifies the exact optimizer, weight decay, batch size, warmup schedule, or other hyperparameters that could affect memorization independently of module selection. This is a reproducibility concern.

**The "fair comparison" is only partially fair:** In the unmatched setting (Figure 1), attention-only tuning has fewer parameters but runs for more iterations to match FLOPs. However, fewer parameters means less gradient noise and different effective learning dynamics. In the matched setting (Figure 2), the gap narrows considerably (10–15% vs. 6–8% OOD on GP), suggesting that raw parameter count explains some of the benefit. The paper does acknowledge this but claims "the benefit persists" — yet the effect size in the matched setting is quite small and is presented without any statistical testing.

**No statistical significance:** No error bars, standard deviations, or confidence intervals are reported anywhere in the paper. Given that the claimed improvements in the matched setting are on the order of 4–7 percentage points from already-low baselines (6–15%), and given that only one model seed appears to have been run, the robustness of these conclusions cannot be assessed.

---

### Experimental Setup (Section 4)

**Severe lack of breadth:**
- Only **one model** is tested: Llama-3.2-Vision-11B. There is no evidence the findings generalize to other architectures, scales, or model families (e.g., GPT-style models, smaller models, encoder-decoder). The dependence on this single architecture is a fundamental limitation.
- Only **two tasks** are evaluated, both taken directly from Chu et al. (2025) — the very paper being used as the primary baseline. This creates a circularity where the conclusions are tightly coupled to the experimental choices of the baseline work. No new benchmark is introduced, and no standard NLP/reasoning benchmarks (e.g., GSM8K, ARC, MMLU) are included to test generalizability.
- Both tasks test **rule-based symbolic reasoning** — a narrow slice of LLM behavior. Claiming the findings shed light on "SFT training dynamics" broadly is unjustified.

**Unspecified hyperparameters in Table 2:** The table compares SFT variants (Full Fine-Tuning, FNN-only, Attention-only) to SFT+RL, but it is unclear which learning rate was used for each SFT variant. Given that Section 5.2 demonstrates learning rate alone can bridge the gap between SFT and RL (with lr=1e-8 achieving 17.95 on GP vs. SFT+RL's 15.0), the learning rate choice is confounded with the module-selection choice in Table 2. Were all SFT variants in Table 2 run with the same learning rate? If the attention-only row used a lower learning rate than the full fine-tuning row, the comparison is unfair.

---

### Results & Discussion (Section 5)

**Figure 1 vs. Figure 2 discrepancy:** In Figure 1 (FLOP-matched, unequal parameters), the OOD gap is dramatic: attention-only sustains >10–80% while full and FNN tuning collapse near zero. In Figure 2 (parameter-matched, fewer FLOPs for full and FNN), the advantage of attention-only is present but much more modest (10–15% vs. 6–8% on GP). This discrepancy is important: it suggests the dominant effect in Figure 1 may be due to attention-only being undertrained (fewer parameters × more steps ≠ equivalent capacity optimization). The discussion undersells this alternative interpretation.

**Section 5.2 (Learning Rate) partially undermines the central claim:** The paper shows that Full Fine-Tuning with lr=1e-7 or 1e-8 achieves 70–80% on V-IRL OOD and 10–17% on GP OOD — comparable to attention-only tuning. Table 2 shows Full Fine-Tuning achieving 89.05 on V-IRL and 17.95 on GP, which is close to attention-only (94.13 and 19.23). If learning rate alone can achieve similar OOD generalization, then the module-selection story is significantly weaker. The paper frames these as complementary findings, but a reviewer naturally asks: *is attention-only tuning simply a softer update mechanism that mimics small learning rate effects?* This alternative hypothesis is never tested or discussed.

**The absolute performance numbers on GP are concerning:** Even the best method (attention-only SFT) achieves only 19.23% on GP OOD. This is very low for a structured arithmetic task. It raises questions about whether any of the SFT methods are actually learning meaningful reasoning at all, or whether differences in these very low numbers are meaningful.

**No mechanistic analysis:** Section 5 is entirely behavioral (input→output performance). The paper claims FNNs are responsible for memorization but provides no mechanistic evidence — e.g., gradient magnitude analysis, weight change norms per module, probing classifiers, or attention pattern analysis. The hypothesis remains observational correlation, not causation.

---

### Writing & Clarity

The missing contribution list in the introduction (described above) is a significant clarity problem, not just stylistic. Section 3.2 ("Transformer Modules") appears after Section 4.1's header in the text — the section ordering is disrupted, which may reflect parser issues but is disorienting. The paper is also quite short (~5 pages of substantive content), limiting the depth of analysis expected at a top venue like ICLR.

---

### Limitations & Broader Impact

There is **no limitations section**. Given the narrow experimental scope (one model, two tasks from one prior paper, one architecture), the absence of a limitations discussion is a significant weakness. The paper presents conclusions that apply broadly to "SFT training dynamics in LLMs" without acknowledging:
- Results may not hold for encoder-decoder models, smaller models, or non-vision-language models.
- Both tasks are rule-based symbolic reasoning — findings may not transfer to generative tasks, dialogue, or open-ended reasoning.
- No discussion of whether attention-only SFT affects other model capabilities (forgetting of pre-trained knowledge, performance on held-out tasks not tested here).
- The practical implication that "attention-only SFT is computationally efficient" is questionable: matching FLOPs requires more iterations, so wall-clock time may not be reduced.

---

## Overall Assessment

This paper addresses a genuinely interesting and practically relevant question — which Transformer modules contribute to memorization vs. generalization during SFT — and produces an empirically suggestive finding: attention-only fine-tuning tends to generalize better OOD than full or FNN-only tuning. However, the work falls well short of ICLR's acceptance bar in its current form. The experimental scope is extremely narrow (one model, two tasks, both borrowed from a single prior paper), statistical rigor is absent (no error bars, no significance tests, single seed), and a critical alternative explanation — that attention-only tuning simply amounts to a lower effective learning rate — is not addressed. The paper's own Section 5.2 partially undermines the central module-selection story by showing that learning rate alone achieves similar OOD performance. The contribution list in the introduction is literally missing. There is no limitations section, no mechanistic analysis, and no engagement with the closely related PEFT literature. The absolute performance numbers on GP (best: 19.23%) raise questions about whether any method genuinely learns rule-generalizable reasoning. In sum, the core empirical observation is interesting and worth investigating, but the paper needs substantially broader experiments, mechanistic analysis, statistical grounding, and a more carefully scoped set of claims before it merits acceptance at ICLR.

# Neutral Reviewer
## Balanced Review

### Summary
This paper provides an empirical investigation into the training dynamics of Supervised Fine-Tuning (SFT) in Large Language Models, specifically analyzing how updating different Transformer modules (Attention vs. Feedforward Networks) impacts out-of-distribution (OOD) generalization. Through controlled experiments on arithmetic and navigation reasoning tasks, the authors demonstrate that fine-tuning only attention layers preserves OOD reasoning capabilities and mitigates memorization, significantly outperforming full-parameter or Feedforward-only fine-tuning. The results suggest that selective Attention-only SFT can achieve OOD performance comparable to state-of-the-art Reinforcement Learning (RL) alignment methods, offering a computationally efficient alternative for post-training.

### Strengths
1.  **Rigorous Controlled Experimental Setup:** The authors effectively address confounding variables by matching total training FLOPs across different fine-tuning strategies (Section 3.1, Table 1) and conducting a parameter-matched setting (Section 5.1). This ensures that observed differences in OOD performance are attributed to module dynamics rather than simply capacity or compute budget advantages.
2.  **Clear Empirical Evidence of Generalization Collapse:** Figures 1 and 2 provide strong visual data illustrating the "memorization trap" in FNN/full-parameter SFT versus the stability of Attention-only SFT. The divergence in Out-of-Distribution metrics (Figure 1, right columns) clearly supports the central hypothesis that FNN updates correlate with rule-specific memorization.
3.  **Practical Implications for Alignment:** The claim that Attention-only SFT matches or exceeds RL baselines on OOD tasks (Table 2) is highly significant if reproducible. It highlights a potential pathway to reduce computational costs and complexity in model alignment pipelines, which is a high-priority topic for ICLR.

### Weaknesses
1.  **Indirect RL Benchmarking:** The comparison to RL alignment methods relies on baselines reported in Chu et al. (2025) rather than running RL on the exact same training data and model configuration in this work. This makes the claim that Attention-only SFT "performs on par with" RL slightly less robust, as hyperparameters and training efficiency differences in the RL experiments are not accounted for (Table 2).
2.  **Limited Diversity of Experiments:** The evaluation is restricted to a single model architecture (Llama-3.2-Vision-11B) and two rule-based benchmarks (GeneralPoints and V-IRL). While these tasks are controlled, it is unclear if these findings generalize to open-domain reasoning, creative tasks, or different model scales (e.g., small vs. large context windows), potentially limiting the scope of the generalization claim.
3.  **Mechanistic Depth is Limited:** The paper successfully identifies *that* Attention layers aid OOD generalization but offers limited theoretical explanation *why*. While it hypothesizes FNNs act as memory stores (citing Geva et al. 2021), it lacks deeper analysis of attention patterns or gradient dynamics to conclusively prove the mechanism of generalization preservation.

### Novelty & Significance
**Novelty:** The paper demonstrates moderate-to-high novelty. While the concept that FNNs store knowledge is established, the specific empirical framing of *module-level selectivity* as a mechanism to prevent SFT generalization collapse is new. The direct comparison of module updates against RL generalization trends adds a fresh perspective to the current landscape of post-training studies.

**Significance:** The significance is high. If selective fine-tuning can substitute for costly RL stages while maintaining generalization, it could substantially change how LLMs are aligned. The findings challenge the current assumption that full-parameter SFT is the standard route and suggest a simpler, more efficient architecture for robust reasoning.

### Suggestions for Improvement
1.  **Direct RL Comparison:** To strengthen the claim in Table 2, the authors should run at least one RL baseline (e.g., DPO or PPO) on the same model and data setup used for the SFT experiments. This would rule out dataset or hyperparameter mismatches as the cause of performance differences.
2.  **Broader Model and Task Evaluation:** Include experiments on at least one pure-text model (e.g., Llama-3.1-8B) and one additional task with less symbolic structure (e.g., simple commonsense reasoning) to verify if the attention-memorization trade-off holds across modalities and task types.
3.  **Deeper Mechanistic Analysis:** Incorporate analyses such as gradient norms across layers, attention head saliency maps, or probing classifiers to explain *why* Attention updates preserve knowledge better than FNN updates. This would transform the paper from an empirical observation to a mechanistic insight, increasing its appeal to ICLR reviewers.
4.  **Clarify "Generalization" Definition:** Ensure the distinction between "rule-following" and "reasoning" is clearer in the discussion. In OOD settings, the model might still memorize the *logic* but adapt to the *symbols*; clarifying the level of abstraction at which generalization is occurring would strengthen the contribution.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Compare against standard PEFT methods (e.g., LoRA):** The paper proposes selective fine-tuning but omits comparison to LoRA, the standard parameter-efficient method. Without this, it is unclear if attention-only tuning offers unique generalization benefits or simply performs similarly to low-rank adaptation.
2. **Reproduce RL baselines under identical conditions:** Table 2 claims attention-only SFT matches/exceeds RL (Chu et al., 2025), but relies on cited numbers rather than direct comparison. Re-run RL methods (PPO/DPO) on the same hardware and data budget to validate the claim that selective SFT replaces RL.
3. **Learning rate ablation for attention-only tuning:** Section 5.2 shows learning rate critically impacts Full FT generalization, but no sweep is provided for attention-only tuning. Without this, the performance gain may be due to implicit regularization from fewer parameters rather than the module type itself.

### Deeper Analysis Needed (top 3-5 only)
1. **Quantify weight update magnitudes per module:** The core hypothesis is that FNNs store memory while Attention handles context. Provide layer-wise $\|\Delta \theta\|$ statistics to empirically verify that FNN updates correlate with memorization metrics while Attention updates do not.
2. **Representation similarity analysis (CKA/CCA):** Analyze hidden state representations before and after fine-tuning to determine if FNN tuning collapses the feature space for OOD inputs. This is needed to confirm that attention-only tuning actually preserves pre-trained knowledge geometry.
3. **Error analysis on remaining OOD failures:** Attention-only OOD performance on GeneralPoints is still low (~15%), meaning the method fails 85% of the time. Analyze whether failures are due to reasoning errors or rule misinterpretation to assess if the method truly solves generalization or merely delays overfitting.

### Visualizations & Case Studies
1. **Training loss vs. OOD accuracy divergence curves:** Plot training loss alongside OOD accuracy for all methods to visualize the memorization point. If FNN tuning memorizes, training loss should drop sharply while OOD accuracy collapses; this divergence must be shown explicitly.
2. **Attention map comparisons for ID vs. OOD inputs:** Visualize attention patterns for successful vs. failed OOD cases under Full vs. Attention-only tuning. Rigid, overfit attention patterns in Full FT would support the claim that attention flexibility drives generalization.
3. **Qualitative case studies of rule adherence:** Provide specific input/output examples where Full FT hallucinates the training rule but Attention-only correctly applies the new OOD rule. Quantitative metrics alone do not prove the model is reasoning rather than guessing.

### Obvious Next Steps
1. **Scaling laws for selective SFT:** Validate if the attention-only benefit holds across model scales (1B to 70B) and architectures (text-only vs. vision-language). ICLR expects robustness claims to be verified beyond a single 11B vision model.
2. **Hybrid fine-tuning schedules:** Since Full FT fits ID data faster but Attention-only generalizes better, experiment with scheduling (e.g., Attention-first then FNN) to optimize both in-distribution and out-of-distribution performance.
3. **Integration with explicit regularization:** Compare attention-only SFT against standard regularization techniques (e.g., weight decay, dropout) to determine if selective tuning offers unique benefits beyond what standard methods achieve with full parameter updates.

# Final Consolidated Review
## Summary
This paper investigates how different Transformer modules (attention layers vs. feedforward networks) contribute to memorization and out-of-distribution generalization during supervised fine-tuning (SFT). Through controlled experiments on two reasoning benchmarks (GeneralPoints and V-IRL), the authors find that fine-tuning only attention layers preserves OOD generalization while full-parameter or FNN-only tuning leads to memorization collapse. The paper also examines learning rate effects on generalization.

## Strengths
- **Rigorous experimental design with compute-matched and parameter-matched comparisons:** The paper carefully controls for confounding factors by matching training FLOPs across fine-tuning strategies (Table 1) and includes a parameter-matched setting (Figure 2) to isolate the effect of module selection from capacity or compute advantages. This strengthens the claim that observed differences stem from module dynamics rather than training budget.

- **Clear empirical demonstration of memorization dynamics:** Figures 1 and 2 effectively show how FNN/full-parameter tuning causes OOD performance to collapse toward zero as training progresses, while attention-only tuning maintains reasonable OOD performance (e.g., 70-80% on V-IRL vs. near-zero for other methods).

- **Practical relevance for alignment pipelines:** The finding that attention-only SFT achieves OOD performance comparable to RL baselines (Table 2: 94.13 vs 91.8 on V-IRL, 19.23 vs 15.0 on GP) offers a potentially simpler alignment approach, though with caveats noted below.

## Weaknesses
- **Learning rate findings confound the central module-selection story:** Section 5.2 shows that full fine-tuning with low learning rates (1e-7, 1e-8) achieves OOD performance comparable to attention-only tuning (17.95% on GP vs. 19.23%). This raises a critical unanswered question: does attention-only tuning provide unique generalization benefits, or does it simply produce an implicit regularization effect similar to using smaller learning rates? The paper does not run learning rate ablations for attention-only tuning, leaving this alternative hypothesis untested. The observation that both mechanisms improve OOD performance similarly suggests update magnitude—whether achieved through fewer parameters or smaller learning rates—may be the operative factor rather than module type.

- **Extremely narrow experimental scope limits generalizability:** Only one model (Llama-3.2-Vision-11B) and two tasks (both rule-based symbolic reasoning benchmarks from Chu et al. 2025) are tested. Both tasks require precise rule-following rather than open-ended reasoning, creative generation, or dialogue. There is no evidence the findings transfer to text-only models, different model scales, encoder-decoder architectures, or task types where attention patterns serve different functions.

- **No statistical rigor:** The paper reports no error bars, standard deviations, or confidence intervals from multiple seeds. Given that the parameter-matched improvements are modest (10-15% vs. 6-8% on GP OOD), and the low absolute performance levels (see below), the robustness of these conclusions cannot be assessed.

- **Concerning absolute OOD performance undermines practical significance:** Even the best method achieves only ~19% OOD accuracy on GeneralPoints—a structured arithmetic task where random chance for valid expressions would be substantially lower, but meaningful rule transfer remains questionable. Differences at such low performance levels may not indicate meaningful reasoning capability.

- **Missing engagement with closely related literature:** The paper does not discuss PEFT methods like LoRA, which by default update attention projection matrices and are directly relevant to the module-selection question. If LoRA already achieves similar benefits by targeting attention layers, the novelty of the empirical finding is diminished.

- **Incomplete presentation:** The contribution list in Section 1 appears truncated (introducing "Our primary contributions are as follows:" followed by "1" then jumping to summary text). There is no limitations section acknowledging the narrow scope or discussing whether attention-only SFT affects other model capabilities.

## Nice-to-Haves
- Direct comparison with LoRA to establish whether attention-only tuning offers unique benefits beyond standard PEFT approaches
- Experiments on at least one additional model architecture or task type to demonstrate generalizability beyond rule-based symbolic reasoning
- Mechanistic analysis (gradient norms, weight change magnitudes per module) to strengthen causal claims about FNNs and memorization

## Removed Points
*These points are flagged to be removed, treat them with caution*

- Criticism that Mukherjee et al. (2025) is "only an arXiv preprint dated May 2025" and therefore problematic—the paper cites it appropriately, and recency of citation is not a substantive weakness.

- Criticism about "missing contribution bullet points"—while the introduction formatting appears incomplete, this is a presentation issue that does not invalidate the paper's substantive claims.

- Criticism about equation placement being "displaced and fragmented"—this appears to be a PDF parsing artifact and does not affect the mathematical correctness of the formulations.

- Requests for comparisons to "standard NLP benchmarks" like GSM8K, ARC, MMLU—the paper explicitly scopes its investigation to controlled reasoning tasks where memorization vs. generalization can be cleanly disentangled via rule manipulation, which is methodologically sound for the research question posed.

## Novel Insights
The paper provides a useful framing connecting SFT generalization dynamics to the functional roles of Transformer modules. The hypothesis that FNNs serve as "memory stores" whose updates disproportionately cause memorization—while attention layers support more flexible, context-sensitive adaptation—aligns with prior interpretability work (Geva et al. 2021, Meng et al. 2022) but applies this lens specifically to the OOD generalization problem. However, the learning rate findings introduce a significant confound: if low learning rates achieve similar OOD benefits for full fine-tuning, the mechanism may be fundamentally about update magnitude rather than module type. This alternative interpretation remains untested.

## Suggestions
1. **Run learning rate ablations for attention-only tuning** across the same range (1e-6, 1e-7, 1e-8) to determine whether module selection provides benefits beyond implicit regularization from smaller effective updates.

2. **Add statistical rigor** by running multiple seeds and reporting error bars or confidence intervals, particularly for the parameter-matched experiments where effect sizes are modest.

3. **Test on at least one additional model architecture** (e.g., text-only Llama or a smaller scale) to establish whether findings generalize beyond a single 11B vision-language model.

4. **Discuss LoRA and PEFT connections explicitly**—if LoRA already targets attention layers by default, situate the findings within that broader context.

5. **Add error analysis on OOD failures**—with attention-only achieving only ~19% on GP OOD, understanding the remaining 81% failure modes would clarify whether the method truly enables rule transfer or merely delays overfitting.

# Actual Human Scores
Individual reviewer scores: [2.0, 0.0, 4.0, 0.0]
Average score: 1.5
Binary outcome: Reject
