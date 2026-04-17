---
job_id: 6a910d8c-1fa8-4c91-95f8-1be477cbd924
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: JjJzbMDGsx.pdf
paper: Language Confusion Gate: Language-Aware Decoding Through Model Self-Distillation
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a decoding-time intervention for multilingual large language models, rooted in representation analysis and self-distillation, and evaluates it on standard ML benchmarks. This is squarely within ICLR’s scope (representation learning, multilingual LMs, decoding, interpretability).

## Minimum Quality
Pass ✅.  
All required sections are present (Abstract, Introduction, Related Work, Method, Experiments/Results, Discussion/Conclusion). The work is clearly written in English, technically coherent, and supported by nontrivial experiments and analysis. No obvious fatal methodological or theoretical flaws stand out that would justify desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any attempts to manipulate AI reviewers or hidden instructions in the main content.

---

# Expected Review Outcome:

## Summary

The paper studies “language confusion” in multilingual LLMs, where unintended scripts (e.g., Chinese characters in Hebrew output) appear during generation. The authors propose the Language Confusion Gate (LCG), a small MLP attached to a frozen LLM that predicts allowed language families at each decoding step and masks disallowed tokens. LCG is trained via norm‑adjusted self‑distillation, using top‑k/top‑p pseudo‑labels derived from logits debiased by output-embedding norms. Experiments across several open and commercial models show substantial reductions in confusion rates (often order-of-magnitude) with negligible impact on translation, reasoning, and general benchmarks, while largely preserving legitimate code-switching.

## Strengths

1. **Well-motivated, practically relevant problem.**  
   Language confusion is a real failure mode for multilingual LLMs, especially in production translation or localized applications. The quantitative evidence in **Table 2** and **Table 7** that many commercial systems still exhibit measurable CJ and Latin confusion reinforces the motivation and timeliness of the work.

2. **Clear mechanistic insight about embedding norms and confusion.**  
   The analysis in Section 3.2, decomposing logits as  
   \(\text{logits}_i = h \cdot e_i = \lVert h\rVert \lVert e_i\rVert \cos_{\text{sim}}(h,e_i)\),  
   and the simple adjustment  
   \(\text{logit}_{\text{adj},i} = \frac{h\cdot e_i}{\lVert e_i\rVert}\)  
   make a convincing case that output-embedding norms bias sampling toward high-resource languages. **Table 1** shows that CJ/Latin tokens are heavily overrepresented in the top‑5% norms across all tested models; **Figure 2** compellingly visualizes that after norm-adjustment the previously top-ranked confusion tokens drop out of the top-10, which directly supports the core training signal used for the gate.

3. **Simple, plug-in architecture with no base-model modification.**  
   The Language Confusion Gate is just a two‑layer MLP operating on the final hidden state, trained with BCE on language-family multi-label pseudo-targets (Section 4.2). It does not alter LLM weights, and can be bolted onto any model with minimal engineering. The intervention rules in Section 4.3 are straightforward, interpretable, and explicitly tuned to keep interference sparse.

4. **Strong empirical results with careful metrics.**  
   - On “no‑think” models, **Table 3** shows that LCG-adjusted cuts CJ confusion by roughly 10× in Qwen3‑8B on FLORES‑NO‑LATIN (4.5% → 0.1%) and Llama3.1‑8B (3.0% → 0.4%), while BLEU remains essentially unchanged or improves slightly.  
   - On INCLUDE, CJ confusion for Qwen3‑30B drops from 2.21% to 0.11% with negligible change in accuracy (71.12 → 70.83).  
   - For “thinking” models on Humaneval‑XL (**Table 4**), Qwen3‑30B’s CJ% is driven to 0.00% with Pass@1 and Pass@10 effectively unchanged, indicating that reasoning quality is preserved.

5. **Evidence that norm adjustment actually matters.**  
   The ablations between LCG-unadjusted and LCG-adjusted in **Table 3** are clean: norm-adjusted training consistently reduces both CJ and Latin confusion further, often without hurting BLEU. For instance, Llama3.1‑8B’s Latin confusion shrinks from 5.7% to 2.9% when adding norm adjustment, which lines up nicely with the geometric analysis in Section 3.2.

6. **Attention to code-switching rather than just “language purity”.**  
   The paper does not naïvely ban cross-script tokens. Section 3.3 and the qualitative examples in **Figures 5–7, 9, 10** emphasize that legitimate code-switching (e.g., code in Arabic prompts, explaining foreign phrases) is valuable. Empirically, **Table 5** shows that while LCG reduces code-switch rates on FLORES‑WITH‑LATIN (e.g., Qwen3‑8B from 46.34% to 25.90%), the post-intervention levels are still comparable to a strong reference model (Claude Sonnet 4 at 23.29%) and not far from ground-truth answer rates (38.36%). Moreover, the token-level analysis that LCG allows human-validated English tokens at 86.7% of confusion points is reassuring.

7. **Comparison with multiple baselines.**  
   The baselines in **Figure 3** (ICL, greedy decoding, ORPO, and “No Rule” variant) show that:  
   - Simple prompting or greedy decoding gives only marginal reductions in confusion.  
   - ORPO can reduce confusion but at the cost of general accuracy on INCLUDE (e.g., Qwen3‑8B drops from 61.4 to 57.3), whereas LCG achieves similar or better confusion reduction with no such degradation.  
   This supports the claim that a dedicated decoding-time gate is more targeted and less harmful than global preference optimization.

8. **Low overhead and practical deployment details.**  
   Section 6’s latency numbers (15.95 ms/token without LCG vs 15.99 ms with LCG on Qwen3‑30B) and Appendix F’s integration with speculative decoding argue convincingly that LCG is operationally cheap. Appendix B’s broader benchmark table (Table 6) suggests that general capabilities (MMLU, GPQA, AIME) remain stable under the gate.

## Weaknesses

1. **Script-level granularity is quite coarse and limits the scope of the solution.**  
   The gate operates over four families (CJ, Latin, Symbols, Low-Res). This handles “obvious” cross-script confusion (e.g., stray CJK in Arabic), but it does not address confusion between languages that share a script (e.g., English vs. Spanish, or English vs. code tokens), and the authors acknowledge this only briefly in Section 6. This is a significant limitation for real multilingual deployments where same-script interference is common, so the method solves only part of the broader “language confusion” problem. It would help to at least quantify how much confusion in real systems is purely cross-script vs. same-script.

2. **Pseudo-label construction and training objective are underspecified and not thoroughly analyzed.**  
   Section 4.2 defines pseudo-targets \(y_{t,i}^*\) via the indicator of whether any token from family \(i\) appears in \(S_{k,p}(\text{logits}_{\text{adjust}})\). However:
   - The exact \(k,p\) values used during training, and whether they are tuned or fixed across models, are not specified in the main text.  
   - There is no sensitivity or ablation study on these hyperparameters; a smaller top‑k might miss legitimate candidates, while too large a top‑p set makes almost every family appear, producing uninformative multi-labels.  
   - The BCE loss \(\mathcal{L}=\sum_{i=1}^n \text{BCE}(y_{t,i}^*,\sigma(z_{t,i}))\) treats all families and all time steps equally, but there is no analysis of class imbalance (Latin tokens vastly outnumber Low-Res in the vocabulary, as noted in Section 4.1) or of calibration quality (false positives vs. false negatives of the gate).  
   Without more detailed specification and analysis, it is hard to assess how robust the learned gate is, and whether its behavior is brittle to the specific norm-adjusted top‑k/p choice.

3. **Norm analysis is suggestive but not fully convincing as a causal explanation.**  
   The geometric decomposition and **Table 1 / Table 8** clearly show that high-resource languages have larger norms, and **Figure 2** illustrates one concrete confusion point where norm-adjustment helps. However:
   - The paper stops at correlation, not causation, and does not explore whether normalizing logits by norms at inference (or re-scaling \(W_{\text{out}}\)) would itself be a competitive baseline. A “Norm-normalized decoding” baseline, where logits are always replaced by \(\text{logit}_{\text{adj},i}\) before sampling, would be a direct test of how far norm bias alone explains confusion.  
   - The remark that norm bias “cannot fully explain” confusion between two high-norm languages (like English and Chinese) is plausible, but not quantified; an empirical breakdown of confusion cases by language pair and by norm statistics would sharpen the argument.  
   As written, the mechanistic story is compelling but somewhat incomplete, and the method could be better justified by adding these baselines and analyses.

4. **Evaluation setup and metrics have blind spots, particularly for Latin confusion.**  
   The Latin confusion metric is only computed on FLORES‑NO‑LATIN, where references have zero Latin characters. While this makes rule-based detection easy, it also implicitly assumes that any Latin usage is wrong for those sentences. In practice, systems sometimes translate a non-English proper noun or technical term that the reference chooses to transliterate, so the metric may count those as confusion. The FLORES‑WITH‑LATIN analysis in **Table 5** partially addresses this, but:  
   - The human annotation for “natural code-switching” is only quoted as a single aggregate number (LCG allows 86.7% of validated English tokens), with no details on annotator numbers, agreement, or sampling scheme.  
   - There is no analogous human audit for FLORES‑NO‑LATIN to estimate false positives in the confusion metric itself.  
   As a result, the claims about “order-of-magnitude reductions” in Latin confusion may be slightly overstated, since some proportion might be penalizing reasonable lexical choices.

5. **Limited diversity and granularity in evaluation languages and tasks.**  
   The experiments focus on a small set of target languages: Arabic, Hebrew, Korean, Thai, Chinese, Greek, Russian, Vietnamese (mainly via FLORES+ and INCLUDE) plus Python reasoning. This is respectable, but:
   - There is no breakdown of confusion rates by target language in **Table 3**, nor an analysis of whether LCG behaves differently for high-resource vs. genuinely low-resource targets.  
   - There are no experiments on “naturally code-switched” corpora (e.g., social-media code-switch datasets) where acceptable mixing is frequent; instead, the analysis is constructed by slicing FLORES, which was never designed around code-switch.  
   - The qualitative examples in **Figures 6–10** are useful but cherry-picked; some systematic quantitative measure of semantic adequacy for code-switched outputs would strengthen the claims about preserved multilingual behavior.

6. **Some methodological and implementation details are missing or only in the appendix.**  
   For reproducibility and to better understand capacity vs. performance trade-offs, more explicit detail in the main paper would help:
   - The dimensionality and hidden size of the two-layer MLP gate, regularization (dropout, weight decay), and training schedule (epochs, optimizer, learning rate) are not specified in Section 4.1/4.2 or 5.1, only loosely gestured at in Appendix C for compute.  
   - The exact rule for “Persistence of the previous token’s language” (Section 4.3) is a bit underdefined: it states “we always allow the language family of the immediately preceding non-symbol token”, but does not clarify how this interacts numerically with the gate’s own probabilities. Is this a deterministic OR over families, or does it shift logits/probabilities?  
   These omissions do not invalidate the results but make it harder for others to reimplement or adapt LCG to different family definitions.

7. **Related work around decoding-time control and self-distillation is underdeveloped.**  
   The Related Work section is focused almost exclusively on language confusion-specific methods and code-switching literature. There is limited discussion of:
   - Broader self-distillation techniques for LLMs (e.g., guiding fine-tuning or decoding with teacher-generated labels).  
   - Other decoding-time control methods that similarly attach small modules or adjust logits without modifying base weights.  
   Without this, the reader could come away with the impression that “norm-adjusted self-distillation of a gating MLP during decoding” is more conceptually isolated than it really is.

8. **Minor mathematical and notation issues.**  
   While the math is mostly straightforward, there are places where clarity could be improved:
   - In Section 3.2, the notation \(\cos_{-}\mathrm{sim}(h,e_i)\) in the norm-adjusted logit equation looks like a typographical artifact; presumably it is the same \(\cos_{\text{sim}}(h,e_i)\) as defined earlier.  
   - Equation-style descriptions of the intervention rule (Section 4.3) are textual only; given that the gate output is \(\mathbf{z}_t\in\mathbb{R}^4\), it would be useful to explicitly define the mapping from \(\sigma(\mathbf{z}_t)\) to a set of allowed families (e.g., via a threshold \(\tau\)) and then to a mask over logits, clarifying how multi-label predictions are handled.  
   These are not fatal but do slow down precise understanding.

Overall, the paper presents a strong and practically useful idea, but it stops short of fully exploring the underlying mechanisms and has some gaps in experimental and methodological depth.

## Potentially Missing Related Work

1. **Yang, Z., Pang, T., Feng, H. (2024): “Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning.”**  
   This paper studies self-distillation strategies for language model fine-tuning, which is closely related to the norm-adjusted self-distillation in Section 4.2. It would be appropriate to discuss it when introducing the training procedure for LCG (Section 4.2) and contrast how their self-generated labels are used to preserve distribution vs. how this work uses them to define language-family pseudo-targets.

2. **Kim, T., Kim, J., Lee, G. (2024): “Instructive Decoding: Instruction-Tuned Large Language Models are Self-Refiner from Noisy Instructions.”**  
   This work proposes decoding-time modifications that use auxiliary signals to refine outputs without changing model weights. It is conceptually aligned with LCG as a plug-in decoding module and should be referenced in Section 2 when discussing decoding-based interventions, and perhaps compared in Section 5.3 as another example of decoding-time control.

3. **Zhan, W., Jing, Y., Rutkowski, L. (2025): “What Makes Large Language Models Undistillable?”**  
   This paper analyzes pitfalls of knowledge distillation for LLMs, including failure modes when teacher outputs are misaligned or noisy. Given that LCG relies on self-distillation from the same model’s norm-adjusted logits, it would be useful to mention and discuss this work in Section 2 or 4.2, clarifying why the distillation here avoids or mitigates such traps (e.g., multi-label language-family supervision instead of full sequence-level imitation).

## Questions

1. **Norm-normalized decoding baseline.**  
   Can you report results for a simple baseline where, at inference, you always use \(\text{logit}_{\text{adj},i} = (h\cdot e_i)/\lVert e_i\rVert\) for sampling, without any gate? Even if this harms some tasks, quantifying its impact on confusion vs. BLEU/accuracy would greatly clarify how much of LCG’s benefit comes from the norm insight vs. learned gating.

2. **Details and sensitivity of pseudo-label generation.**  
   What exact values of \(k\) and \(p\) were used to define \(S_{k,p}(\mathbf{logits}_{\text{adjust}})\) during training? Have you tried other settings, and did that materially affect gate performance or the sparsity of interventions? Some small ablation on this would increase confidence in the robustness of the method.

3. **Gate architecture and decision threshold.**  
   Please specify the hidden size of the MLP, activation function, optimizer, and training hyperparameters in the main text (not only in code). Also, how do you convert \(\sigma(z_{t,i})\) to allowed/disallowed families at test time? Is there a fixed threshold (e.g., 0.5) or some learned or calibrated decision rule? How sensitive are results in **Table 3** to this threshold?

4. **Human evaluation of confusion vs. legitimate Latin tokens.**  
   Could you provide more detail on the human annotation setup for estimating the 86.7% “allowed” rate in FLORES‑WITH‑LATIN and, more importantly, any audit of FLORES‑NO‑LATIN outputs to estimate how often your automatic Latin confusion metric flags reasonable Latin words? Even approximate numbers would help interpret the magnitude of the claimed improvements.

5. **Per-language breakdown and low-resource behavior.**  
   Do you have per-target-language confusion statistics within FLORES and INCLUDE? In particular, does LCG behave differently on low-resource targets, and does the “never mask Low-Res” rule ever allow obvious confusion that could have been prevented?

6. **Failure modes and same-script confusion.**  
   Beyond the examples in Appendix J, can you characterize how often LCG incorrectly allows a confusion token (false negatives) vs. incorrectly blocks a valid token (false positives)? And could you comment on any preliminary experiments toward same-script, language-level gates (even if only for English vs. Spanish/French) to quantify how much harder that problem is?

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The method is conceptually straightforward, grounded in a reasonable mechanistic observation, and empirically supported on multiple models and tasks. Some aspects (pseudo-label construction, lack of a norm-normalized baseline, limited analysis of metric reliability) prevent a rating of “excellent,” but there are no obvious fatal flaws.

## Presentation Rating

3: good.  
The paper is generally clear, with helpful figures and tables. **Figure 1** effectively illustrates the high-level gating mechanism and how it filters tokens; **Figure 3** succinctly compares baselines. There are some missing implementation details and minor notation issues that, if fixed, would further improve clarity.

## Contribution Rating

3: good.  
The work offers a practically useful, plug-in solution to a real problem, backed by a coherent mechanistic explanation and solid experiments. While conceptually incremental in some respects (a small MLP + self-distillation + masking), the combination in the specific context of language confusion, and the demonstration that this can be done without harming performance or code-switching, makes it a valuable contribution.

## Overall Rating

8: Accept, good paper (poster).  
The paper addresses an important and under-served practical issue, proposes a simple and effective decoding-time mechanism, and supports it with thoughtful analysis and strong empirical evidence. Despite some missing baselines and limited granularity in evaluation and analysis, the work is clearly above the bar for ICLR and should be of interest to both practitioners and researchers working on multilingual LLMs and decoding control.

## Reviewer Confidence

4: confident.  
I am familiar with multilingual LLMs, decoding/control methods, and distillation techniques, and I checked the math and experimental design in reasonable detail. Some implementation details are not specified, so I leave a small margin for misinterpretation, but I am unlikely to change my overall assessment drastically.