Now let me search for calibration papers.Now I have sufficient information to write the final review. Let me compile it.

---

## Summary
This paper proposes SuMi, a method for **multimodal wild TTA** — a new setting where test data contains both weak OOD samples (single modality corrupted) and strong OOD samples (multiple modalities corrupted or one modality missing). SuMi consists of three components: (1) IQR smoothing, which dynamically gates samples during adaptation using a time-varying interquartile range criterion applied to feature representations; (2) unimodal assistance, which selects low multimodal-entropy samples that also exhibit sufficiently high unimodal entropy (indicating reliance on cross-modal interaction); and (3) mutual information sharing (MIS), a cross-modal KL-divergence alignment loss. The paper introduces two benchmarks (Kinetics50-C and VGGSound-C) covering 21 video corruption types, 6 audio corruption types, and 4 strong OOD scenarios, and shows that existing TTA methods fail catastrophically on strong OOD while SuMi maintains significant accuracy gains.

---

## Strengths

- **Novel and well-motivated problem formulation, concretely demonstrated.** The paper shows in Figure 1(b)–(d) that existing methods (Tent, SAR, SoTTA, DeYO, CEMA, and even the multimodal-specialized READ) fail dramatically under strong OOD and wild TTA settings, often degrading far below the source model. This establishes a genuine, unaddressed gap.

- **Strong empirical gains on the hardest setting (Tables 2 and 4).** On Kinetics50-C strong OOD, SuMi achieves 33.4% average vs. 29.1% for READ (+4.3%), with the largest gain on "Mix" scenarios (18.4% vs. 13.7%). On VGGSound-C strong OOD, SuMi achieves 19.7% average vs. EATA's 17.4%. These are substantial absolute improvements in a regime where most baselines are near chance.

- **Informative non-obvious empirical finding about unimodal entropy (Figure 3(c), Table 6).** The observation that unimodal entropy in the [20,40] percentile outperforms the [0,20] percentile — indicating that very-low-unimodal-entropy samples lack multimodal discriminative value — is experimentally well-supported and motivates the unimodal assistance component in a principled way.

- **Graceful degradation under wild mixing ratios (Figure 5).** The mixed-ratio evaluation across 10 different proportions of strong OOD samples provides a realistic assessment of wild TTA conditions and shows SuMi degrades more gracefully than all baselines.

- **Benchmark contribution.** Kinetics50-C and VGGSound-C with strong OOD scenarios is a concrete, reusable contribution that fills a genuine gap since prior benchmarks (including READ's) only consider single-modality corruption.

---

## Weaknesses

### Fatal
*None* — the core empirical results are not invalidated by the identified issues.

### Major

- **The paper's primary interpretive claim about IQR smoothing directly contradicts its own ablation data.** Section 4.3 states: *"IQR smoothing brings the most improvements to the model."* Table 5 (Kinetics50-C, severity 3) shows: IQR alone = 37.1, UA alone = 52.1, MIS alone = 49.4. IQR is the **weakest** single component. Marginal contribution analysis from the full model is equally clear: removing MIS costs **−11.9 points** (59.3→47.4), removing IQR costs −3.3 points (59.3→56.0), and removing UA costs −1.3 points (59.3→58.0). By every reasonable metric, **MIS contributes the most**. This mischaracterization is not minor phrasing — it is the paper's core narrative claim and is factually incorrect per the authors' own data.

- **Table 5 contains two rows with identical checkmarks (IQR ✓, UA ✓, MIS ✓) reporting different results (54.3/44.6/51.3 vs. 59.3/52.0/59.1) with no label distinguishing them.** Section 3.4 explains that MIS is applied for all iterations in weak OOD but only the first t₀ iterations in strong OOD — these two rows almost certainly correspond to those two cases. However, the table provides no such label, making it impossible for a reader to determine which row corresponds to the proposed method (SuMi) and which to an ablation variant. This is a presentation failure of a methodologically important design decision.

### Minor

- **Overclaiming in the comparison section.** The paper states SuMi "outperforms other methods consistently and significantly on all the four distribution scenarios." Table 4 shows EATA achieving higher accuracy than SuMi on Crowd noise (28.8 vs. 27.9), Rain (32.3 vs. 31.6), Wind (33.2 vs. 34.1, SuMi is better here actually), and importantly EATA's strong OOD average (17.4) is lower than SuMi (19.7, so SuMi is better on average here). But specifically on audio noise categories in Table 4, EATA beats SuMi on Crowd and Rain. The text should acknowledge this nuance rather than claiming uniform dominance.

- **No sensitivity analysis for the smoothing coefficient β, which varies dramatically between datasets (0.6 for Kinetics50-C vs. 0.9 for VGGSound-C).** Figure 7 only analyzes µ and λ. Given that β governs the core IQR smoothing behavior and takes very different values on the two datasets, its sensitivity is arguably more important to understand than λ.

- **No source model (no-adaptation) baseline row in Tables 1–4.** Although Figure 4 includes a "Source" bar for comparison, the main tables make it impossible to directly assess how much each method degrades from or improves over the pre-trained model. This is particularly important for Tables 2 and 4 where Figure 4 suggests many baselines drop dramatically below source on strong OOD.

- **The IQR mechanism applied to high-dimensional feature vectors is ad hoc.** Algorithm 1 computes scalar Q₁ and Q₃ over all dimensions of the concatenated representation batch, then uses a dimension-level voting threshold. The paper provides no formal argument for why this would preferentially select weak OOD samples over strong OOD ones — it relies entirely on the qualitative t-SNE in Figure 3(b). A quantitative analysis tracking what fraction of selected samples are weak vs. strong OOD as a function of iteration t would directly validate or refute the smoothing narrative.

### Trivial

- **Hyperparameter µ has opposite effects on the two datasets** (Figure 7(a): increases Kinetics50-C, decreases VGGSound-C). The explanation offered (Kinetics50 is video-dominant, VGGSound is audio-dominant) is reasonable but presented as post-hoc rationalization; µ=1.0 is used for both without acknowledged trade-off.

---

## Nice-to-Haves

- Quantitative selection statistics over adaptation iterations (fraction of weak/strong OOD samples selected at each step) would directly verify the IQR smoothing mechanism beyond the current t-SNE visualization.
- A hybrid baseline (e.g., EATA + IQR smoothing only) would help disentangle whether gains on VGGSound-C strong OOD come from IQR smoothing specifically or from the interaction with MIS.
- Since MIS emerges as the most impactful component in the ablation (when honestly read), a dedicated analysis of when and why MIS helps vs. hurts (especially in Mix scenarios where both modalities are degraded) would substantially strengthen the paper's contribution narrative.

---

## Removed Points
*These points were flagged for removal; treat with caution.*

- **"Mutual information sharing is not standard information-theoretic MIS"** (harsh critic). This is a cosmetic naming complaint. The method is clearly defined in Equation 6.
- **"Large hyperparameter differences between datasets imply per-dataset tuning" framed as reproducibility concern** (harsh critic). Different hyperparameters per dataset are normal in TTA literature and not a methodological flaw.
- **Weakness about the specific form of the KL target in Eq. 6 (½(p^u' + p^m) vs. p^u' alone) not being ablated** — partially addressed: the design motivation is clearly stated (robustness to corrupted modalities), and requiring every sub-design choice to have an isolated ablation is beyond standard norms.

---

## Novel Insights

The paper's most genuinely novel empirical observation — that the [20,40] unimodal entropy quantile outperforms the [0,20] quantile for cross-modal learning — is a non-obvious finding with clear practical implications: samples that are too easy in individual modalities carry little cross-modal information and are therefore less useful for multimodal model adaptation. This insight, validated in Figure 3(c) and Table 6, suggests a general principle for multimodal self-supervised learning: moderate unimodal uncertainty is a proxy for multimodal discriminativeness, not just noise. The paper also demonstrates, perhaps inadvertently through its ablation, that cross-modal prediction alignment (MIS) is by far the most powerful single component for handling strong OOD and missing-modality scenarios — a finding that the paper underplays but that has independent value for the community.

---

## Suggestions

1. **Correct the ablation interpretation in Section 4.3.** Per Table 5, MIS is the largest single contributor by marginal impact; IQR is the second largest. Rewrite the narrative to reflect this honestly. The method is still novel and effective — the story should be told accurately.

2. **Label the two full-model rows in Table 5.** Distinguish "IQR+UA+MIS (all iterations)" from "IQR+UA+MIS (first t₀ iterations only)" and explain that the latter is the proposed SuMi. Make Algorithm 1's conditional MIS logic explicit in the ablation table.

3. **Add a β sensitivity figure** alongside the existing µ and λ analyses in Figure 7. Given β=0.6 vs. 0.9 across datasets, this is the most underspecified hyperparameter.

4. **Add a "Source" row** to Tables 1–4 so readers can directly see how much baseline methods degrade below no-adaptation on strong OOD scenarios.

---

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| READ (high anchor) | TPZRq4FALB | 8.0 | First multimodal TTA paper; accepted 8/8/8/8. More principled (attention modulation + theory-backed loss). SuMi extends to a harder setting but with weaker methodology narrative. |
| MM-CTTA (medium anchor) | UhKkWHkvfg | 5.0 | Similar multimodal TTA extension; rejected. Comparable level of novelty, but weaker empirical gains. SuMi has larger gains over baselines than MDAA. |
| IDKR TTA (low anchor) | nc0XGK40dn | 4.67 | Similar "mixed distribution" TTA framing; rejected for unclear contribution/methodology. SuMi is clearly stronger with better empirical results. |
| AdapTable TTA (low anchor) | ws0F5NTzGw | 4.5 | TTA in a new domain with distribution shift issues; limited novelty. SuMi is methodologically more interesting. |

SuMi occupies a position clearly above the low anchors (nc0XGK40dn, ws0F5NTzGw) due to its genuine new problem setting, strong empirical results, and non-trivial method. Compared to the MM-CTTA paper (5.0), SuMi offers larger and more compelling empirical gains, a cleaner problem formulation, and more thorough evaluation — suggesting a modest upward shift. However, SuMi falls well below READ (8.0) due to the incorrect ablation narrative (a major interpretive flaw), the ambiguous ablation table, and a less methodologically principled design. The **major weakness** — the paper's stated core claim that "IQR brings the most improvements" is directly contradicted by the authors' own data — is serious enough to require substantial revision but does not invalidate the method's empirical value.

Positioning: slightly above the MM-CTTA medium anchor (5.0), warranting a **5.5**.

**Axis ratings:**
- *Originality*: Good — the wild multimodal TTA setting and the benchmark are genuine new contributions.
- *Importance of research question*: High — multimodal strong OOD robustness matters in practice.
- *Claims vs. support*: Weak — the core narrative claim about IQR is contradicted by the paper's own ablation.
- *Soundness of experiments*: Moderate — empirical results are strong, but ablation interpretation is wrong.
- *Clarity of writing*: Moderate — ambiguous Table 5 and incorrect Section 4.3 narrative.
- *Value to community*: Moderate-High — benchmarks and empirical findings are useful despite interpretive issues.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>