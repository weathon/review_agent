Now I have all the information I need to write the final review. Let me consolidate my findings.

## Summary

CausalMM proposes a structural causal model (SCM) for MLLMs that treats visual and language priors as confounders between attention mechanisms and model output, applying back-door adjustment and counterfactual reasoning to derive a contrastive decoding strategy that perturbs attention maps at both visual encoder and LLM levels to mitigate hallucinations.

## Strengths

- **Attention perturbation as a contrastive signal is a reasonable and potentially useful technique.** Unlike VCD which perturbs the visual input (adding noise to the image), CausalMM perturbs the attention maps directly. This is a distinct and sensible approach: by replacing attention weights with random/uniform/reversed/shuffled alternatives, the method generates counterfactual logits that capture the bias introduced by the model's attention patterns. The ablation in Section 4.3 (Figure 6) shows random attention works best, providing practical guidance.
- **Dual-modality perturbation captures both bias sources.** The method addresses both visual priors (via visual encoder attention perturbation) and language priors (via LLM attention perturbation), and combines them in a multimodal collaborative setting (Eq. 12). This is a natural extension beyond methods that only address one modality.
- **POPE results show genuine improvements, especially in adversarial settings.** On GQA adversarial, CausalMM-multimodal achieves 79.53 accuracy vs. 75.16 regular (+4.37), outperforming VCD (+1.17) and OPERA (−0.16). Similar patterns hold on MSCOCO adversarial (83.70 vs 78.63 regular, +5.07) and A-OKVQA adversarial (77.86 vs 74.26, +3.60). These are consistent, meaningful gains.
- **Plug-and-play, training-free compatibility.** The method operates entirely at inference time by modifying attention maps, making it compatible with any MLLM architecture without retraining, as demonstrated across LLaVA-1.5 and Qwen2-VL.

## Weaknesses

### Fatal

**VLind-Bench headline claims are directly contradicted by the paper's own data.** The abstract claims "a maximum score improvement of **65.3%** on 6 VLind-Bench indicators" and the introduction claims "**143.7** points on 6 indicators of VLind-Bench." Yet Figure 3's data table shows that the multimodal attention variant (the paper's main proposed method) produces scores **identical** to the regular baseline across all five sub-metrics for both models:

- LLaVA-1.5: regular (Sck=22.5, Svp=35.0, Scb=48.8, Slp=65.0, LP=45.0) = multimodal (22.5, 35.0, 48.8, 65.0, 45.0)
- Qwen2-VL: regular (88.8, 98.0, 68.0, 82.0, 52.0) = multimodal (88.8, 98.0, 68.0, 82.0, 52.0)

Zero improvement. Furthermore, the vision-only and language-only variants actually **decrease** some scores (e.g., LLaVA-1.5 Sck drops from 22.5 to 15.0 with vision attention). The paper's own text compounds this problem: it claims the multimodal collaborative setting makes "a significant leap" (Section 4.2, paragraph 1) and that "CAUSALMM method significantly improves the model's score on VLind-Bench" (Figure 3 caption). No computation from the presented data can produce the 65.3% or 143.7-point figures. This is the paper's most prominent quantitative claim in the abstract, and it is unsupported.

### Major

- **The causal framework is decorative rather than derivational—the method reduces to contrastive decoding with attention perturbation.** The paper's central novelty claim is applying structural causal modeling with back-door adjustment and counterfactual reasoning to MLLMs. However, the actual implementation (Eqs. 9-12) computes token selection as $\arg\max \text{softmax}(\ell_i + \gamma(\ell_i - \ell_{cf,i}))$, which is structurally identical to VCD's contrastive decoding formula—the only difference is that counterfactual logits come from attention perturbation rather than visual input distortion. The SCM does not determine which attention to perturb, how to perturb it (random/uniform/reversed/shuffled), the decoding formula's specific form ($\max$, $\gamma$, $\log(\epsilon)$), or the intervention layer—all are chosen empirically. The paper never derives the token selection formula from the SCM or back-door adjustment. The claimed paradigm shift from "statistical correlations" to "causal relationships" is therefore unsupported: the method is contrastive decoding, and the causal framing adds no algorithmic content beyond what the perturbation mechanism already provides.

- **The SCM contains notational errors that undermine confidence in the causal modeling.** In Section 3.1, the graph lists "$T_l \rightarrow A_v$: Language token embeddings $T_l$ influence the MLLM's attention $A_t$" and "$P_l \rightarrow A_v$: Language priors $P_l$ inform the MLLM's attention mechanism $A_t$." The arrows point to $A_v$ (visual attention) but the descriptions reference $A_t$ (LLM attention)—these should be $T_l \rightarrow A_t$ and $P_l \rightarrow A_t$. This is not merely a typo; it creates confusion about the causal graph's structure and whether the authors have carefully validated their own model. Additionally, the graph lacks the $T_i \rightarrow A_t$ edge, which represents how visual tokens are inputs to the LLM's attention—a critical architectural path in MLLMs.

### Minor

- **POPE results are competitive but not uniformly dominant.** OPERA frequently achieves higher recall (e.g., MSCOCO Random: OPERA 85.26 vs CausalMM-multimodal 82.00; A-OKVQA Popular: OPERA 91.13 vs CausalMM 77.60). CausalMM's advantage is primarily in accuracy and precision, not across all metrics. The paper's claim of "superior performance" (Section 4.2) is overstated given the mixed results.

- **The GPT-4o evaluation (Table 3) shows trivially small improvements.** The multimodal variant scores 85.0 on the "All" metric vs. 84.7 for regular—a 0.3-point difference on a 10-point scale. This is within noise and does not substantiate meaningful improvement.

- **MME results are presented in figures with narrow y-axis ranges** (Figure 5: Perception 1450-1550, Cognition 420-460), which visually amplifies modest differences. Precise numerical comparisons are difficult from the bar charts alone, and the only detailed MME table (Table 2) covers just 4 sub-metrics for Qwen2-VL.

### Trivial

None worth noting beyond what's already covered.

## Nice-to-Haves

- A direct mechanistic analysis of how attention perturbation changes internal representations (hidden states, token representations) would strengthen the claim that the method addresses "causal effects" rather than simply providing a useful contrastive signal.
- Attention map visualizations before and after perturbation would help readers understand what the method actually does to the model's focus.
- Variance/significance reporting on POPE and MME results, given that many improvements are 2-5 percentage points.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Harsh Critic: "The back-door adjustment is never computed" and "Pv and Pl are fixed parametric properties, so summation is incoherent."** The paper references "Sec. A.1" for proof—the appendix (stripped from the parsed version) may contain this derivation. While the concern about fixed model properties is philosophically valid, many causal inference papers in ML treat model parameters as random variables in a population-level analysis, so this is not outright incoherent. The core issue—that the framework doesn't derive the actual method—is already captured in the Major weakness.

- **Harsh Critic: "Counterfactual reasoning formulas mix observational conditioning with interventional notation in violation of Pearl's calculus."** This is technically nuanced. After a $do$-intervention on $A_i$, conditioning on $I$ and $P_v$ (which have direct paths to $O$ via $T_i$) is not automatically inconsistent—it simply doesn't change the conditional for $A_i$ itself. The notation is confusing but may not be formally wrong.

- **Harsh Critic: "SCM omits the critical cross-attention path $T_i → A_t$."** While valid, the graph already has $T_i → O$ and $A_t → O$ as separate paths. The omission of $T_i → A_t$ is a simplification rather than a fundamental error, though it does reduce the fidelity of the causal model.

- **Harsh Critic: "Direct paths $T_i → O$ and $T_l → O$ don't exist in transformer architectures."** Causal graphs are abstractions; these paths can represent the cumulative effect of tokens passing through the processing pipeline. This is a standard simplification in causal modeling.

- **Strength Finder: "Principled causal framing over correlation-based approaches" and "CausalMM formalizes the relationship using SCM, enabling principled back-door adjustment."** These strengths are removed because they conflict with the verified Major weakness that the causal framework doesn't substantively contribute to the method. You cannot claim a "principled" framework as a strength when the framework doesn't derive the actual algorithm.

- **Strength Finder: "Dramatic improvement on VLind-Bench" with "Figure 3 shows substantial gains across all five sub-metrics."** This strength is removed because it is directly contradicted by the data in Figure 3, which shows multimodal = regular across all five sub-metrics for both models.

- **Harsh Critic: Demands for "direct comparison with attention-perturbation contrastive decoding without the causal framing."** This is a nice-to-have, not a required experiment. The method IS contrastive decoding with attention perturbation; the value-add of the causal framing (or lack thereof) is a conceptual point, not an experimental one.

## Novel Insights

The most revealing observation from this review is the disconnect between the paper's theoretical apparatus and its empirical substance. The paper builds an elaborate causal framework (SCM, back-door adjustment, counterfactual reasoning with do-calculus notation) but the actual method is a simple and reasonable technique—perturb attention maps and use the resulting logit differences as a contrastive signal during decoding. This technique could stand on its own merits without the causal dressing. The VLind-Bench data contradiction (multimodal = regular, yet claiming 65.3% improvement) is the most damaging finding, as it suggests either computational errors or selective reporting in the paper's headline results.

## Suggestions

- **Immediately reconcile the VLind-Bench claims with the data.** Either the 65.3% and 143.7-point figures are computed from a metric or comparison not shown in the paper, or they are incorrect. Provide exact derivations or retract the claims.
- **Honest framing:** Position the contribution as "contrastive decoding with dual-modality attention perturbation" rather than "a structural causal model with back-door adjustment." The attention perturbation idea is interesting and useful; the causal framework, as currently presented, weakens rather than strengthens the paper.
- **Fix the SCM notational errors** ($T_l \rightarrow A_v$ and $P_l \rightarrow A_v$ should be $T_l \rightarrow A_t$ and $P_l \rightarrow A_t$).

## Calibration

**Anchors compared against:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| TAME (anchor token intervention for MLLM hallucination) | `/home/wg25r/review_agent/human_reviews/zGb4WgCW5i.md` | 7.00 | Similar topic (attention-based MLLM hallucination mitigation, plug-and-play), but TAME has clean theory-meets-method alignment and no data contradictions. CausalMM is below this. |
| DoLa (contrastive decoding for LLM hallucination) | `/home/wg25r/review_agent/human_reviews/Th6NyL07na.md` | 7.25 | Same technique class (contrastive decoding) with principled derivation. CausalMM is well below this. |
| Visual Attention Sink (MLLM attention redistribution) | `/home/wg25r/review_agent/human_reviews/7uDI7w5RQA.md` | 5.75 | Similar topic (attention-based intervention), similar concerns about novelty vs. existing methods. But no data contradictions. CausalMM is below this. |
| GACD (gradient-based contrastive decoding for MLLM hallucination) | `/home/wg25r/review_agent/human_reviews/zgXGNXkC0F.md` | 4.75 | Similar profile: addresses MLLM hallucination via contrastive decoding with overclaimed novelty. CausalMM is comparable but the VLind-Bench data contradiction is worse. |
| DFITE (diffusion model for ITE, misapplied causal framework) | `/home/wg25r/review_agent/human_reviews/4u0ruVk749.md` | 3.00 | Similar issue: causal framework doesn't connect to the actual method. CausalMM is slightly above this because it has more substantial empirical results on POPE. |
| IRIS (misapplied causal discovery) | `/home/wg25r/review_agent/human_reviews/zgM66fu0wv.md` | 2.50 | Causal framework fundamentally misapplied. CausalMM is above this because the attention perturbation method itself is reasonable. |
| Overclaimed tabular learner | `/home/wg25r/review_agent/human_reviews/lf8QQ2KMgv.md` | 3.75 | Headline claims contradicted by own data. CausalMM is comparable—same core problem of data not supporting claims. |

CausalMM sits between the 3.0-3.75 range occupied by papers with misapplied causal frameworks or overclaimed results contradicted by data, and the 4.75-5.75 range for papers with reasonable methods but some issues. The VLind-Bench data contradiction pulls it firmly toward the lower end, but the genuine POPE improvements (particularly in adversarial settings) keep it from the very bottom. Final score: 3.5.

## Score and Decision

**Originality:** Low. The method is contrastive decoding with attention perturbation—structurally identical to VCD with a different perturbation target. The causal framework, claimed as the primary novelty, does not derive the algorithm.

**Importance of research question:** Moderate. Mitigating MLLM hallucination from modality priors is important and timely.

**Claims well supported:** Poor. The most prominent claim (65.3% VLind-Bench improvement) is directly contradicted by the paper's own data. The causal framework claims are not supported by the method's derivation.

**Soundness of experiments:** Mixed. POPE results are reasonable; VLind-Bench results contradict the claims; MME results are presented with limited numerical detail.

**Clarity:** Moderate. The paper is readable but contains notational errors in the SCM and confusing mixing of causal and observational language.

**Value to community:** Low-to-moderate. The attention perturbation idea is useful, but the paper's framing and unsupported claims may mislead rather than help.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>