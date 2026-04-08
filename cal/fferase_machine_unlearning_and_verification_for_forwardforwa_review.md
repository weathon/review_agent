=== CALIBRATION EXAMPLE 2 ===

# Final Consolidated Review
## Summary

This paper introduces FF-Erase, the first machine unlearning framework designed specifically for Forward-Forward (FF) models. FF-Erase uses a guidance model to produce target goodness distributions, steering the original model's layer-wise goodness scores away from forgetting data via KL-divergence minimization while periodically recovering utility on remaining data. The paper also proposes G-MIA, a membership inference attack that leverages FF models' layer-wise goodness scores to verify unlearning effectiveness under a black-box access model.

## Strengths

- **Genuine problem novelty**: This is the first work to formalize and address machine unlearning for FF models. The two identified challenges—parameter tuning sensitivity causing model collapse, and layer-wise independent training complicating the effectiveness-utility trade-off—are well-motivated and empirically validated (Figure 5 shows GA causes collapse or fails to unlearn depending on λ).
- **Architecturally tailored solution**: The goodness-guided strategy directly addresses FF's unique training dynamics. Rather than applying brute-force gradient ascent, FF-Erase uses a guidance model to produce valid target goodness distributions, preventing the invalid distribution shifts that cause collapse. The ablation with a random guidance model (R.G.M in Table 1, test accuracy drops to 55.53%) cleanly demonstrates that guidance quality matters and that the method's success is not trivially obtained.
- **Layer-wise CKA analysis (Appendix C.2)**: Table 3 provides genuinely insightful layer-wise analysis, revealing that conventional methods like Bad Teacher leave high CKA in shallow/middle layers (0.9998 at Layer 1) while over-forgetting in deep layers—explaining *why* distillation-based GA fails for FF models. This goes beyond typical ablation depth.
- **Practical efficiency with flexible trade-offs**: The mini-retrained and fast-distilled strategies for guidance model generation offer practitioners tunable efficiency-performance trade-offs. Table 1 shows FF-Erase can achieve 1.9–3.1× speedup over retraining while maintaining competitive G-MIA scores, and the α₁/α₂ parameters provide clear knobs for different scenarios.

## Weaknesses

### Major:

- **G-MIA's access model is not truly black-box**: Section 5 states the attacker "can obtain the output of the target model... the goodness vectors from all layers." Standard black-box MIAs (e.g., Shokri et al., 2017) assume access only to final prediction outputs/probabilities. Requiring all-layer goodness vectors is a strictly stronger assumption—this is effectively a gray-box setting. The paper should either (a) reclassify G-MIA accordingly and discuss the practical implications (are goodness vectors typically exposed by FF model APIs?), or (b) explicitly argue why this access level is reasonable for FF model auditing scenarios. Calling it "black-box" without qualification is misleading relative to community norms.

- **Circular verification risk**: The primary metric for evaluating FF-Erase's unlearning effectiveness is G-MIA, which the paper itself proposes. While Section 6.1 demonstrates G-MIA outperforms existing MIAs as an *attack*, using a self-designed attack to verify a self-designed defense raises concerns about verification reliability. The paper would be significantly strengthened by showing that G-MIA-based verification conclusions (e.g., "FF-Erase effectively unlearns") are consistent with conclusions drawn from independent metrics. The accuracy-on-forgetting-data metric partially addresses this, but G-MIA ACC scores of 0.55–0.59 (vs. 0.50 for random guessing) leave room for interpretation about what constitutes sufficient unlearning—a threshold the paper never defines.

- **Potential guidance model information leakage in fast-distilled strategy**: Equation (8) trains the guidance model via distillation from the original model θ_o on D_remain. Since θ_o was trained on D_forget ∪ D_remain, it may encode membership signals about D_forget in its internal representations that transfer to the guidance model through distillation—even when distilling only on D_remain. If the guidance model retains such signals, it could provide "guidance goodness" that inadvertently preserves forgetting data influence. The paper does not investigate this potential leakage pathway, which is concerning given that fast-distilled guidance is the recommended strategy when D_remain is small (exactly the scenario where leakage effects could be most pronounced).

### Minor:

- **KL divergence direction and loss function choice not justified or ablated**: Equation (5) uses D_KL(g ∥ g*) rather than the reverse or a symmetric alternative (e.g., Jensen-Shannon divergence). KL divergence is asymmetric, and the choice of direction affects gradient magnitude and behavior when g and g* have disjoint support—a plausible scenario during unlearning. No justification or ablation against alternatives is provided.

- **Recovery forward frequency K not ablated**: The hyperparameter K controls how often remaining data is re-learned (every K epochs). Section 4.3 notes K affects efficiency, but Table 1 only varies α₁ and α₂ for guidance models while holding K fixed. The interaction between K and the unlearning-utility trade-off remains unexplored, despite K being described as critical for balancing utility maintenance against efficiency.

- **No minimum guidance model quality threshold established**: While Table 1 shows a gradient from R.G.M (catastrophic) to well-trained guidance models, and the ablation varies α₁/α₂, the paper provides no principled guidance on what constitutes a "sufficiently good" guidance model. Practitioners lack a diagnostic to determine whether their guidance model is adequate before committing to the unlearning process.

- **Limited architectural diversity in evaluation**: All experiments use CNNs (TinyCNN, AlexNet, VGG) on image benchmarks. The related work cites FF extensions to recurrent (FF-LSTM) and graph-based (ForwardGNN) architectures. While FF research is still early-stage, demonstrating that the goodness-guidance mechanism transfers to at least one non-CNN FF variant would significantly strengthen the generality claim.

### Trivial:

- **Termination thresholds (ε₁, ε₂) selection not discussed**: These determine when unlearning stops, yet no guidance is provided for setting them. However, this is a common issue in optimization-based unlearning methods and does not uniquely undermine this work.

## Nice-to-Haves

- **Theoretical analysis linking goodness distribution shifts to parameter influence removal**: A formal justification for why KL-divergence-guided goodness shifts correspond to actual data influence removal, rather than relying solely on empirical correlation with accuracy and MIA metrics.
- **Goodness distribution visualizations**: Histograms of layer-wise goodness before/after unlearning would make the distribution-shifting mechanism more interpretable and confirm shifts match guidance without collapsing activation magnitudes.
- **Confidence intervals on G-MIA metrics**: While single-run evaluation is standard in this area, variance estimates on G-MIA ACC/AUC would strengthen the empirical claims, particularly when differences between methods are small (e.g., 0.55 vs. 0.56).
- **Iterative unlearning evaluation**: Real-world RTBF compliance may require successive unlearning requests. Testing whether FF-Erase degrades over multiple sequential unlearning operations would address a practical concern.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Numerical discrepancies between abstract claims and Table 1 data**: The harsh critic claimed the 1.9–3.1× speedup was not supported by Table 1. However, D-(0.3,0.1) achieves 353.7s vs. RE's 1107s (≈3.13×), and D-(0.5,0.5) achieves 583.5s (≈1.9×), fully supporting the claimed range. The critic miscalculated.
- **Introduction's assertion about black-box MIA inaccuracy lacking upfront evidence**: This is a structural/writing concern—the evidence appears in §6.1. Standard paper structure delays detailed evidence to the experiments section. This is a formatting nitpick.
- **Notation in Equation (8)**: The critic flagged D_KL(Dref; θ ∥ θ_o) as confusing. This is a standard shorthand in the distillation literature (KL between output distributions on Dref). Formatting nitpick.
- **Final prediction aggregation unclear**: The paper explicitly addresses this in §3.1: "It is common to take a fully-connected layer on them as the predictor." The critic missed this statement.
- **Demanding ImageNet-scale or Transformer experiments**: FF algorithms are still in early development stages and have not been demonstrated at ImageNet scale. Requesting this is scope creep beyond the paper's reasonable ambitions.
- **Missing comparison with influence function/Fisher information methods**: The paper explains in §1, §2, and Appendix A why standard BP-based unlearning methods (including influence-based approaches) are structurally incompatible with FF's layer-wise training. The comparison with GA, BT, FYE, SURE, and FATS in Appendix C.3 is sufficient to establish this point.
- **Hyperparameter search overhead in efficiency claims**: This applies to any method including retraining (which also requires hyperparameter tuning). Asymmetric criticism.

## Novel Insights

The layer-wise CKA analysis (Table 3) reveals an underappreciated structural property of FF models: middle layers are the most "specialized" for individual training processes (lowest CKA under retraining, ~0.23–0.63), while shallow layers learn general features shared across data subsets (CKA ~0.92–0.99) and deep layers recover shared high-level features (CKA ~0.49–0.62). This inverted-U specialization pattern differs from the typical BP model understanding where feature specificity monotonically increases with depth. This suggests FF unlearning might be most effective when targeting middle layers specifically, and that the "shallow layer retention" observed in methods like Bad Teacher (CKA 0.9998 at Layer 1) is actually expected behavior rather than a failure mode—raising the question of whether FF unlearning verification should weight middle-layer goodness signals more heavily.

## Suggestions

- **Reclassify G-MIA's threat model**: Explicitly acknowledge that all-layer goodness access is stronger than standard black-box. Discuss whether FF deployment scenarios naturally expose goodness vectors (e.g., edge computing where the full model runs locally), and compare G-MIA against a truly black-box baseline that uses only final-layer predictions on FF models.
- **Add cross-validation of unlearning effectiveness**: Report unlearning results using at least one independent verification metric beyond G-MIA (e.g., distance from retrained model parameters, or a canonical MIA from the literature) to mitigate circular verification concerns.
- **Ablate KL divergence direction**: Compare D_KL(g ∥ g*), D_KL(g* ∥ g), and Jensen-Shannon divergence on at least one model/dataset to justify the loss function choice.
- **Test guidance model leakage explicitly**: For the fast-distilled strategy, evaluate whether the guidance model's goodness on D_forget differs significantly from a mini-retrained guidance model's goodness on D_forget, to confirm no information leakage from the original teacher.
- **Define an effectiveness threshold**: Provide a principled discussion of what G-MIA ACC/AUC level constitutes "effective unlearning" for compliance purposes, even if approximate. The current results (ACC ~0.55–0.59 vs. random 0.50) leave practitioners without clear guidance.

---

**Quality Assessment:**

- **Novelty**: High. This is the first work addressing unlearning for FF models, and the goodness-guided approach is architecturally motivated rather than borrowed from BP literature.
- **Technical soundness**: Moderate. The core algorithm is well-designed and empirically validated, but the G-MIA access model mischaracterization and circular verification risk are substantive concerns. The lack of loss function justification (KL direction) is a technical gap.
- **Empirical support**: Moderate-to-good. Experiments cover multiple datasets, models, and baselines with informative ablations. However, the primary evaluation metric (G-MIA) is self-proposed, and the gap between FF-Erase and RE on G-MIA scores is modest (~0.01–0.04 ACC). The CKA analysis and GA failure demonstrations are compelling.
- **Significance**: Moderate-to-high. As FF algorithms gain traction for edge computing and bio-plausible AI, unlearning becomes a practical necessity. This work provides the foundational toolkit, though its impact depends on FF models achieving broader adoption.
- **Clarity**: Good. The paper is well-organized with clear problem motivation, method description, and experimental structure. The illustrations (Figure 2) effectively communicate the unlearning workflow.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 4.0, 8.0]
Average score: 5.5
Binary outcome: Reject
