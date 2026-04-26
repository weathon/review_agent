## Summary

SigMap proposes a multi-modal foundation model for wireless localization that combines two innovations: (1) a cycle-adaptive masking strategy for self-supervised CSI pre-training that disrupts periodic shortcuts in channel data, and (2) a "map-as-prompt" framework that encodes 3D geographic information into learnable soft prompts via a GNN for parameter-efficient fine-tuning. Experiments on ray-tracing datasets (DeepMIMO and WAIR-D) show state-of-the-art localization accuracy and cross-scenario transfer with only 0.4% of parameters updated during fine-tuning.

## Strengths

- **Conceptually clean map-as-prompt mechanism**: Encoding 3D environmental structure through a GNN into soft prompts prepended to a frozen Transformer backbone is a principled approach to parameter-efficient adaptation, and the prompt-as-conditioning mechanism is more interpretable than naive feature concatenation (Section 3.4, Algorithm 1).

- **Well-motivated cycle-adaptive masking**: The identification of periodic shortcut learning in CSI data is technically sound, and the cross-correlation-based detection of periodicity followed by shift-aware masking provides a clear mechanistic rationale (Section 3.3, Equation 6). The ablation in Table 3 shows adaptive masking (0.673m MAE) outperforming grid (0.770m) and strip (0.753m) alternatives.

- **Strong empirical improvements in the few-shot generalization setting**: On unseen environments with ~100 fine-tuning samples, SigMap (w/ map) achieves 1.026m MAE on DeepMIMO O2 and 1.880m on WAIR-D, outperforming the best baseline (LWLM) by 53.2% and 44.3% respectively (Section 4.5). The map ablation shows the prompt mechanism provides genuine gains over CSI-only (Table 4).

- **Parameter efficiency**: Fine-tuning requires only 0.085M trainable parameters (0.4% of total) and 30 minutes, which is practically relevant for deployment scenarios (Table 5).

## Weaknesses

### Fatal

None.

### Major

- **Incorrect "zero-shot generalization" claim**: The abstract and contribution statement (Section 1.2) both claim "strong zero-shot generalization to unseen environments." However, Section 4.5 explicitly describes using "approximately 100 instances per scenario" for fine-tuning, which is few-shot transfer, not zero-shot. The paper's own text correctly uses the phrase "few-shot learning setup" in Section 4.5, but the abstract's headline claim is directly contradicted by the experimental design. This misrepresents what the experiments demonstrate and inflates the perceived contribution. — *This matters because the generalization capability is a primary selling point, and the distinction between zero-shot and few-shot is substantive: zero-shot would require no target-domain labels at all.*

- **No map-aware baselines**: All baselines (OMP, CNN, SWiT, LWLM) receive CSI only, while SigMap receives CSI + 3D map. The w/o-map ablation (Table 4) shows map input alone reduces MAE by 31% in single-BS (2.275m → 1.564m). The experiments establish that map information helps localization, but they cannot establish that the specific map-as-prompt mechanism is better than simpler alternatives (e.g., concatenating flattened map features, MLP over map embeddings). Without at least one map-aware baseline, the paper cannot disentangle the benefit of having map data from the benefit of the proposed integration method. — *This matters because the core contribution is supposed to be "how" to integrate maps, not merely "that" maps help.*

- **Main results evaluated in the same environment used for pre-training**: Section 4.1 states O1 3p5 is used "for both pre-training and fine-tuning," and Tables 1–4 report results within this scenario. The genuine generalization test (Section 4.5) shows weaker results (1.880m MAE vs. 1.564m in-distribution). The headline "state-of-the-art" claims rely primarily on in-distribution performance, which is less impressive for a framework explicitly motivated by cross-scenario transfer. — *This matters because the paper positions itself as a foundation model for cross-scenario generalization, yet its strongest results do not test this property.*

### Minor

- **Missing random masking ablation**: The stated motivation (Section 1.1) criticizes "generic masking strategies," yet Table 3 compares only grid and strip masking — not random masking, which is the standard MAE baseline. The ablation shows adaptive is better than grid/strip, but not that it improves over the simplest and most widely-used masking alternative, leaving the cycle-adaptive advantage incompletely validated.

- **Undefined NLoS attention mechanism**: Equation 11 introduces an "NLoS-aware attention mechanism" with learned parameters $\mathbb{W}_{\text{NLOS}}$, appearing only in Section 4.2 (results) rather than in Section 3 (methodology). The mechanism is described as a "key advantage" for single-BS localization, yet its formal definition, motivation, and relationship to the rest of the architecture are missing from the method section. An ablation isolating its contribution would clarify its role.

- **All experiments use ray-tracing data with no real-world validation**: Both DeepMIMO and WAIR-D rely on ray-tracing simulation (WAIR-D uses real city geometries but simulates channels). While ray-tracing is a common starting point, real-world CSI includes calibration errors, hardware impairments, and non-ideal propagation effects that may degrade performance significantly.

## Nice-to-Haves

- Adding a simple map-aware baseline (e.g., map features concatenated to CSI before the encoder) would directly test whether the prompt mechanism specifically adds value beyond simply having map data.
- Reporting true zero-shot results (no target-domain labels) as an additional datapoint, even if performance is weaker, would clarify the method's actual generalization capabilities and allow honest framing.
- Error maps or CDF curves for the generalization experiments (as shown for in-distribution results in Figure 5) would reveal whether map prompts correct specific spatial failure modes.

## Removed Points

*These points were flagged but removed from the main review. Treat with caution.*

- **Formatting/notation gripes removed**: Parsing artifacts (equation numbering inconsistencies, red-colored notation in Eq. 9, garbled text lines) are parser issues, not author errors — removed per instructions.

- **"Same train/test split" / data leakage concern removed**: The reviewer speculated about potential data leakage from missing train/val/test split details. The paper states results are "averaged over 5 independent runs" and the framework involves separate pre-training and fine-tuning stages, making this speculation without evidence. Removed.

- **"2D vs 3D map ablation undermines 3D motivation" removed**: The reviewer argued the 8% MAE degradation from 3D→2D undermines the 3D motivation. The paper's interpretation — that topological/LoS cues carry the majority of the benefit — is reasonable and actually informative. This is a minor point at best, not a weakness.

- **Concern about GNN parameter count growing with scene complexity removed**: This is speculative and the paper provides parameter efficiency numbers. Without evidence that this is actually a problem, this remains unverified.

- **Concern about multi-BS attention sharing weights removed**: The notation in Equations 9-10 is somewhat ambiguous, but this is a minor presentation issue, not a substantive methodological concern.

- **Missing related works concerns removed**: Per instructions, I do not flag missing related work citations.

- **Concern about ray-tracing realism removed**: While all data is simulated, the use of ray-tracing is standard in this community and stated clearly. This does not invalidate the paper's contributions but is worth noting as a scope limitation (listed under Minor).

## Novel Insights

The paper exposes an interesting asymmetry in its own results: the map-as-prompt mechanism provides 31% MAE improvement in single-BS settings (2.275m → 1.564m) but only 15% in multi-BS settings (0.789m → 0.673m). This suggests that geographic prompts are most valuable precisely when single-BS scenarios create the most ambiguity — a finding consistent with the NLoS motivation, but one the paper does not discuss. Additionally, the fact that the 2D map retains 92% of the 3D map's benefit suggests the prompt mechanism primarily encodes topological connectivity rather than precise 3D geometry, which has practical implications for deployment scenarios where 3D maps are unavailable.

## Suggestions

- Rename "zero-shot generalization" to "few-shot transfer" or "minimal fine-tuning transfer" throughout the abstract and contributions to match the actual experimental setup.
- Add a simple map-concatenation baseline to Tables 1–2 to demonstrate that the prompt mechanism specifically adds value beyond naive map integration.
- Define the NLoS attention mechanism (Equation 11) in Section 3 with proper motivation and notation, and add an ablation isolating its contribution.

## Score and Decision

**Calibration anchors**:
- High band (≥6): Wi-GATr (7.0, wireless simulation with strong methodology), ADePT (7.0, PEFT), HiRA (8.0, PEFT)
- Medium band (~5): masking strategy papers (4.5–5.5), SMPE (6.0, wireless resource allocation, rejected)
- Low band (≤4): WM5G2NWSYC (2.0, overclaimed zero-shot), ZaudLwn0Hm (2.5, overclaimed generalization), WiMTR (4.0, WiFi CSI localization with overclaimed results and unfair comparisons)

This paper is better than the low anchors — it has genuine technical ideas (cycle-adaptive masking, map-as-prompt), proper ablations within its framework, and achieves real improvements. However, it shares the overclaimed generalization pattern seen in the ≤4 anchors (zero-shot claim contradicted by its own few-shot design) and has a significant evaluation gap (no map-aware baselines). It falls below the medium anchors, which had cleaner evaluation. Compared to WiMTR (4.0), which was rejected for overclaimed generalization and unfair comparisons on WiFi CSI data, this paper has stronger technical contributions but similar evaluation issues. The zero-shot misrepresentation and the missing map-aware baselines are substantive problems, not nitpicks.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>