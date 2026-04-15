Now let me search for calibration papers.Now I have sufficient information to write the final consolidated review. Let me synthesize everything.

---

## Summary

X-ALMA is a multilingual LLM-based translation system scaling to 50 languages via plug-and-play language-specific (LS) LoRA modules grouped by linguistic similarity. It uses a five-stage training recipe (three pre-training stages plus two post-training stages), culminating in a novel Adaptive-Rejection Preference Optimization (ARPO) method that mitigates "over-rejection" — a failure mode of generic preference optimization when preferred and dispreferred translation pairs are highly similar. X-ALMA reports state-of-the-art COMET-22 results across all 98 English-centric translation directions on FLORES-200 and WMT'23, surpassing Aya-101, Aya-23, LLaMAX3, and NLLB-3.3B.

---

## Claims and Support

**Claim 1: X-ALMA achieves SOTA open-source multilingual translation across 50 languages on FLORES-200 and WMT'23 by COMET-22.**
- *Partially supported.* Tables 2–4 show X-ALMA leading all group averages and WMT'23 directions. Per-direction results are in Appendix E. However, the SFT stage trains on FLORES-200 **dev** set, NTREx **test** data, and WMT'15–22 **test** sets. Using NTREx test data as training data is a legitimate concern. The FLORES concern is mitigated: (a) dev, not test, is used; (b) the paper's own ablation (Figure 3 right) shows adding FLORES dev data provides negligible gain over NTREx+WMT alone, suggesting results are not driven by FLORES dev proximity. WMT'15–22 training → WMT'23 evaluation is a clean temporal split. Overall, the benchmark contamination is less severe than the harsh reviewer suggests, but the use of NTREx test as training data is worth flagging.

**Claim 2: The LS module architecture prevents language conflicts and mitigates the curse of multilinguality.**
- *Unsupported as a causal claim.* No ablation isolates the LS module contribution versus a shared LoRA or dense fine-tuning baseline. Figure 1 compares heterogeneous systems with different sizes and recipes. The paper demonstrates the full pipeline works well but does not show the architecture is the cause.

**Claim 3: Each training stage provides crucial, consistent gains.**
- *Partially supported.* Figure 3 (left) shows cumulative improvement across stages for Group 6. Some increments are small (xx→en: Pre Stage 2 to Pre Stage 3 is 83.4 → 83.6). The ablation is limited to Group 6 only. "Crucial" slightly overstates the case.

**Claim 4: ARPO outperforms existing preference optimization methods for translation and mitigates over-rejection.**
- *Partially supported.* Table 5 shows ARPO clearly best on Group 6 across BLEU, COMET-22, and XCOMET-XL. The over-rejection narrative is supported by BLEU collapse in competing methods. However, the comparison is demonstrated only on one group (Group 6), and the over-rejection mechanism is inferred from BLEU drops rather than direct sequence-level analysis.

**Claim 5: Plug-and-play loading strategies offer practical memory advantages.**
- *Weakly supported as a design property.* No actual memory, latency, or throughput measurements are reported. This is an architectural inference rather than an empirically demonstrated advantage.

---

## Strengths

- **Strong and consistent empirical results**: X-ALMA achieves the highest group averages across all 8 language groups in both en→xx and xx→en on FLORES-200 (Tables 2–3), and leads on all 12 WMT'23 directions including for languages not covered by ALMA-R (Table 4). Beating Aya-23-35B (2.5× more parameters) in all group averages is a notable result.
- **ARPO is a genuine and useful contribution**: The observation that MT preference pairs are highly similar and that this causes BLEU collapse in standard methods (DPO, SimPO, etc.) is insightful and backed by Table 5. ARPO's adaptive weighting of the dispreferred term is simple, principled, and empirically effective. Section 6.1 provides a well-reasoned qualitative explanation of the over-rejection phenomenon with reference to concrete examples in Appendix F.
- **Practical plug-and-play architecture**: The hard-gated LS module design is computationally attractive: single-module loading for inference, module merging, and MoE-style joint loading are all viable deployment modes. This is not just cosmetic — it directly addresses GPU memory constraints compared to full MoE.
- **Careful training recipe with ablation support**: Figure 3 demonstrates that each stage adds measurable value. The SFT data ablation provides a useful insight: diverse English data (WMT) helps generalization beyond the NTREx domain, but adding multi-way-parallel FLORES dev adds little.
- **Public release of models and preference data**: The release of preference learning data for 50 languages is a meaningful contribution to the community.

---

## Weaknesses

### Fatal
*(None — the paper's core empirical claims are reasonably substantiated. No single issue entirely invalidates the paper.)*

### Major

- **No architectural ablation isolating the LS module contribution.** The paper's framing presents the plug-and-play LS module architecture as a core contribution to mitigating the "curse of multilinguality." Yet there is no experiment comparing: (a) X-ALMA with a single shared LoRA, (b) grouped LS LoRAs vs. ungrouped LoRAs, or (c) the training recipe with vs. without the modular design. Section 3.1 asserts "prevent conflicts between languages during training, such as gradient conflicts" (citing Wang et al., 2021) but offers no gradient-conflict measurement or interference analysis. Without this, the architecture's causal contribution is unvalidated; it is entirely possible that the five-stage recipe alone drives the gains. Since the modular architecture is one of two claimed primary contributions, this is a significant evidentiary gap.

- **ARPO evaluation is confined to a single language group.** Table 5 presents the preference optimization comparison only for Group 6. The paper's abstract claims ARPO "surpasses existing preference optimization methods in translation tasks" — a claim that requires broader evidence. Group 6 was specifically chosen as "the most challenging group" (Section 6), and the paper never shows whether ARPO's advantage holds in groups with different linguistic characteristics (e.g., the higher-resource Germanic or Romance groups where preference pairs may differ in character). This is the primary methodology gap for the ARPO contribution.

- **Language grouping anomaly undermines the linguistic motivation.** Table 1 lists Group 4 as "Southeast Asian Languages" yet includes `fr` (French) alongside `id`, `mg`, `ms`, `th`, and `vi`. French shares neither linguistic family nor geography with Southeast Asian languages. The paper's grouping criteria require languages to "be as similar as possible," which French manifestly does not satisfy relative to Group 4. This either indicates a pragmatic residual placement (for size balancing) that is not disclosed, or a mistake. In either case, the "human linguistic knowledge yields more accurate classification" justification for not using Lang2Vec is undermined by this example.

### Minor

- **Training recipe ablation is restricted to Group 6.** The claim that "each stage is crucial" is grounded only in Group 6 data. Since Group 6 is the hardest group by design, the incremental contributions of stages may look different for simpler groups. Broader ablations covering at least one more group would strengthen this claim.

- **NTREx test data used as SFT training data.** NTREx is commonly used as an evaluation resource; using its test portion as SFT training data could advantage X-ALMA on benchmarks that share provenance with NTREx. This is less severe than the harsh reviewer suggests (FLORES-200 test and WMT'23 are not NTREx), but it is still worth acknowledging in the paper as a potential confounder.

- **Over-rejection mechanism not directly demonstrated.** The BLEU-drops-but-COMET-stable pattern in Table 5 is good evidence for stylistic shift, but the conclusion that this reflects "rejection of preferred translation style" is inferred. Token-level or sequence-level analysis (e.g., n-gram overlap shifts between generated outputs and preferred data) would strengthen this claim.

- **Training recipe complexity and compute not discussed.** Five training stages, spanning monolingual pre-training, pseudo-monolingual, SFT, and preference optimization with separate base/module parameters, is expensive. The paper provides no estimate of total GPU hours or parameter counts during inference (base + all 8 modules), making it difficult to contextualize the gains.

### Trivial

- **English (en) is listed inside each group's language column in Table 1** (e.g., "(en)") without explanation in the table caption. The text explains it but the table notation is ambiguous.

---

## Nice-to-Haves

- **Architectural ablation**: A comparison of X-ALMA's LS modules against a single shared LoRA at equal parameter budget would strongly validate the modular design.
- **ARPO across all 8 groups**: Even a summary table showing BLEU/COMET for ARPO vs. CPO (best baseline) across groups would meaningfully broaden the ARPO evidence.
- **τ_θ dynamics across training**: Since τ_θ depends on the model's own log-likelihoods (which change as training progresses), tracking τ_θ distributions over training steps would reveal whether the adaptive penalty is behaviorally stable.
- **ARPO hyperparameter sensitivity (η)**: The paper fixes η=1.5 with no sensitivity analysis. A small sweep would confirm robustness.
- **Statistical significance or confidence intervals**: Many of the margins in Tables 2–4 are 0.1–0.5 COMET points. Reporting variance over a small set of decoding seeds would help distinguish meaningful gaps from noise.
- **Direct per-language comparison with ALMA-R on the 6 shared languages**: Figure 1 shows aggregate comparison but not whether ALMA-R languages individually regress under X-ALMA, which is central to the "no curse" claim.
- **xx→xx evaluation on a subset of language pairs**: X-ALMA explicitly scopes to English-centric translation, which is appropriate, but a brief appendix showing a subset of non-English-to-non-English directions would clarify the architecture's actual capabilities.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh critic (Claim 1, asymmetric size comparison with Aya-23-35B)**: The harsh reviewer noted that Aya-23-35B being larger "helps the authors." Under the hard rule, comparisons that favor the baseline (larger model, same or better task) are intentionally asymmetric to prove a stronger point; this is not a weakness. X-ALMA beating a 2.5× larger model strengthens, not weakens, the contribution. **Removed.**

- **Harsh critic: "Per-direction results deferred to appendix"**: The paper clearly states "Detailed results for each translation direction can also be found in Appendix E" (Section 5.5). This is normal practice for a paper with 98 directions. The group averages in the main paper and the full results in the appendix constitute adequate evidence. **Removed.**

- **Harsh critic: Aya-23-35B vs. 13B backbone**: Mentioned as evidence of unfair comparison. Per hard rules, if the asymmetry favors the baseline (larger model), the comparison only strengthens the paper's claim. **Removed.**

- **Harsh critic: Baseline decoding parity (re-decoded vs. taken from prior work)**: While decoding details are underspecified, this is standard practice at ICLR for multilingual evaluation papers and cannot be verified without external sources. **Removed as a nitpick.**

- **Spark: Non-English-to-non-English (xx→xx) evaluation as a core weakness**: The paper explicitly defines its scope as "98 English-centric translation directions" and does not claim xx→xx capability. This is scope creep rather than a flaw. Moved to Nice-to-Haves. **Removed as major weakness.**

- **Neutral reviewer: "insufficient comparison with encoder-decoder baselines"**: NLLB-3.3B is included. Requesting deeper analysis of why LLMs now beat encoder-decoder models is outside this paper's scope. **Removed.**

---

## Novel Insights

The ARPO contribution contains a genuinely underappreciated insight: MT preference data is structurally different from QA or dialogue data because high-quality dispreferred translations are necessarily close to preferred ones, making standard reward-margin-based objectives unstable. The BLEU-collapses-while-COMET-is-stable pattern across five different preference methods (Table 5) is striking evidence that MT preference learning has a specific failure mode not captured in existing methodology surveys. The proposed adaptive weighting mechanism — scaling down the dispreferred term when the model cannot yet discriminate the pair — is simple, does not require a reference model, and preserves the behavior-cloning objective as a stabilizer. This framing of over-rejection as a structural property of MT preference data (not a hyperparameter problem) is the paper's most original and transferable contribution.

---

## Suggestions

1. **Add an architectural ablation** comparing X-ALMA (grouped LS LoRAs) vs. a single shared LoRA trained with the same five-stage recipe. This is the single most important addition to validate the architecture claim.
2. **Extend the ARPO comparison to all 8 groups**, at minimum as a compact appendix table; this would substantially strengthen the methodological contribution.
3. **Relabel Group 4** more accurately or explicitly justify French's placement there (e.g., "pragmatic grouping for size balance"), and clarify that the criteria are not purely linguistic where balance constraints override.
4. **Add a "Limitations" section** that acknowledges the English-centric scope, the use of NTREx test data in training, and the absence of architectural ablations.
5. **Report inference-time parameter counts** (base + 1 LS module, base + all 8 modules) to enable fair comparison with dense baselines.

---

## Score and Decision

**Calibration anchors:**
- **Lingual-SMoE** (ySS7hH1smL): Accept poster, scores 8,8,6,8 (avg 7.5). A similarly-scoped multilingual MT modular paper with strong ablations and careful metrics. X-ALMA is empirically stronger but has a larger unsupported causal claim about its architecture.
- **ALMA original** (farT6XXntP): Accept poster, scores 8,5,6,8 (avg 6.75). This is the direct predecessor, accepted despite missing some baselines and limited reproducibility details. X-ALMA extends ALMA with 44 more languages and ARPO — a clear incremental advance.
- **Mufu** (0eMsrRMmCw): Accept poster, scores 6,6,8 (avg 6.67). Multilingual low-resource MT, modest scope.
- **DPO ties extension** (h71cSd2loX): Reject, scores 5,6,5,6 (avg 5.5). Incremental preference optimization variant with narrow empirical coverage — the ARPO-only portion of X-ALMA is at this level, but X-ALMA is a broader system paper.

X-ALMA sits above Mufu and the ALMA original due to its broader empirical coverage and the ARPO contribution. It sits below Lingual-SMoE because the core architectural claim is unsupported by ablation. The benchmark contamination concern (NTREx test → training) is real but limited in severity. Overall this is a solid accept-level system paper with one significant methodological gap (no architecture ablation) and one narrower-than-claimed contribution (ARPO on one group). Positioned at **6.5**, consistent with the ALMA-original tier of acceptance.

**Axis summary:**
- **Originality**: Moderate-high (ARPO is novel; LS modules are incremental over language-specific adapters but practically motivated)
- **Importance of research question**: High (50-language MT at high quality is genuinely impactful)
- **Claims supported by evidence**: Moderate (empirical results strong; architectural causal claim unsupported)
- **Soundness of experiments**: Moderate (strong main tables; ablations narrow to Group 6; ARPO not broadly evaluated)
- **Clarity of writing**: Good (well-organized; Group 4 anomaly needs clarification)
- **Value to community**: High (model + data release, ARPO insight, scaling blueprint)

**Score: 6.5 — Accept (poster)**

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>