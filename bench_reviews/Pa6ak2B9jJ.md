## Summary

AUTO-RT proposes a reinforcement learning framework for automated red-teaming of LLMs that discovers attack *strategies* rather than fixed prompts. The method decomposes attack generation into a strategy generator (AM_g) and rephrasing model (AM_r), introducing Dynamic Strategy Pruning (DSP) to eliminate redundant exploration paths and Progressive Reward Tracking (PRT) that uses downgraded target models to shape sparse rewards. Experiments across 16 white-box and 2 black-box models claim improvements in attack success rate, diversity, and efficiency.

## Strengths

- **Comprehensive empirical evaluation:** The paper tests AUTO-RT across 16 open-weight models spanning Llama, Vicuna, Yi, Mistral, Qwen, Gemma families (2B to 72B), plus commercial models (GPT-4.1, Claude Sonnet 4, Gemini-2.5-Pro). This breadth provides meaningful evidence of generalizability across architectures and safety training regimes.

- **Novel reward shaping mechanism:** The First Inverse Rate (FIR) metric provides a principled, data-driven criterion for selecting downgraded models to shape sparse safety rewards. This addresses a practical challenge in RL-based red-teaming and is a genuine methodological contribution (Section 2.3.3, Figure 4).

- **Defense Generalization Diversity (DeD) metric:** Evaluating second-round attack success after defense construction is a practically relevant evaluation that captures sustained discovery capability beyond one-shot jailbreaking. AUTO-RT achieves substantially higher DeD than baselines (e.g., 46.80% vs 20.10% RL baseline on Vicuna 7B), demonstrating meaningful improvement.

- **Clear ablation of components:** Tables 7-9 provide ablation results showing both DSP and PRT contribute to performance across most models, with PRT showing particularly strong impact on DeD.

## Weaknesses

- **Inconsistent best performance claims:** The abstract and paper claim AUTO-RT "significantly outperforms existing methods" and "consistently achieves the highest ASRtst," but Table 1 contradicts this: Gemma 2 9B (RL 44.85% > AUTO-RT 44.80%), R2D2 (FS 27.18% >> AUTO-RT 12.45%), and Mistral 7B (IL 54.88% > AUTO-RT 52.65%). These counterexamples should be acknowledged and analyzed rather than omitted from claims.

- **AutoDAN substantially outperforms AUTO-RT on raw attack success:** Table 3 shows AutoDAN achieves ASRtst=55.23% versus AUTO-RT's 38.38%—a gap of nearly 17 percentage points. While AUTO-RT wins on DeD (38.19% vs 17.88%), this tradeoff between first-round success and sustained discovery is not adequately discussed. Safety practitioners may prioritize raw ASR for vulnerability discovery, making this a consequential limitation.

- **ASRtst metric uses post-hoc selection:** Equation 6 defines ASRtst as the average ASR of the top-100 strategies by ASR, not the ASR of strategies sampled from the trained policy. This post-hoc selection inflates reported performance compared to what a deployed system would achieve without oracle access to strategy quality.

- **No statistical significance reported:** Despite visible variance in Figure 3's violin plots, no confidence intervals, standard errors, or statistical tests appear in any table. This makes it difficult to assess whether observed differences are meaningful, particularly for small gaps (e.g., Llama 3 8B: AUTO-RT 15.00% vs RL 14.55%).

- **Exploitability/severity framing not operationalized:** The introduction distinguishes exploitability (ease of triggering) from severity (harm caused) and claims this motivates the work, but all evaluations use standard ASR. No separate metric quantifies exploitability. This gap between motivation and evaluation is unexplained.

- **AM_r kept frozen without justification:** The rephrasing model (AM_r) is implemented as Vicuna-7B and kept frozen while only AM_g is trained. This design choice is not ablated or justified. Since attack query quality depends critically on AM_r's ability to instantiate strategies, this could be a bottleneck.

- **FIR selection procedure not fully automated:** The paper states "select the last model before a sharp increase of FIR" but provides no automated detection rule. Figure 4 shows examples where the spike is visually apparent, but no algorithm or threshold is specified for the other 12+ models tested.

- **Non-monotonic ablation behavior unexplained:** In Tables 7-9, combining DSP and PRT sometimes underperforms individual components (e.g., Yi 6B DeD: +PRT=50.94% > AUTO-RT=47.25%). The interaction between components deserves analysis.

## Nice-to-Haves

- Comparison with additional strong baselines (PAIR, TAP, GCG, AutoDAN-Turbo) to position against widely-used methods, with computational cost analysis.

- Ablation isolating the benefit of the hierarchical (AM_g + AM_r) architecture versus end-to-end generation to quantify the strategy-level contribution.

- Expansion of the ethics statement to address responsible disclosure and potential misuse risks of releasing effective jailbreak strategies.

- Sensitivity analysis on FIR selection: what happens if models beyond the FIR spike are selected?

## Removed Points

*These points are flagged to be removed, treat them with caution*

- **Ethics statement "inadequate"**: While brief, the ethics statement addresses the intended purpose (supporting robust LLM development). Demanding detailed responsible disclosure procedures and dual-use safeguards exceeds what is standard for AI safety research papers. The primary venue for such concerns is model release, not paper publication.

- **Missing PAIR/TAP/GCG baselines constitutes a critical flaw**: The paper compares against conceptually similar RL-based methods (IL, RL) and prompt-based approaches (AutoDAN, Human Template, Past-Tense). PAIR, TAP, and GCG use fundamentally different mechanisms (textual feedback, tree search, gradient-based optimization) that would require substantially different experimental setups. AutoDAN-Turbo comparison is a reasonable addition, but this is incremental rather than disqualifying.

- **Black-box "validation concerns"**: The reviewer claims "lack of detailed API usage constraints" but Table 10 in Appendix G clearly presents results on GPT-4.1, Claude, and Gemini using the ICL approach. The methodology is specified; reproducibility concerns about API access apply generically to all commercial model evaluations.

- **Reproducibility of prompts missing**: Figures 5-7 in the appendix DO contain the prompt templates for strategy generation, query rephrasing, and consistency checking. This criticism is factually incorrect based on the paper content.

## Novel Insights

The Progressive Reward Tracking mechanism introduces an interesting asymmetry: rather than shaping rewards toward a goal state (as in potential-based shaping), it uses a *weaker* model's failure modes to guide exploration. The intuition that "unsafe regions of TM are contained within those of TM'" (Figure 2) suggests a novel theoretical connection between model capability degradation and adversarial robustness. However, the paper does not prove this containment formally or validate it empirically across model pairs. The FIR metric indirectly operationalizes this by detecting when downgraded models become too corrupt to provide useful signal—a practical heuristic that could inspire more principled theoretical treatment.

## Suggestions

- Report average ASR across all generated strategies (not just top-100) alongside ASRtst to show realistic deployment performance.

- Add confidence intervals or standard errors to key results; conduct statistical tests for claims of "significant improvement."

- Provide an automated FIR threshold selection algorithm (e.g., gradient-based or statistical change-point detection) to remove manual inspection.

- Include wall-clock time and GPU-hour comparisons with baselines to verify efficiency claims.

- Acknowledge models where AUTO-RT underperforms baselines and analyze why (e.g., R2D2's adversarial training may specifically target strategy-based attacks).