=== CALIBRATION EXAMPLE 2 ===

# Final Consolidated Review
## Summary
This paper identifies model capability privacy leakage as a critical vulnerability in existing Offsite Tuning (OT) methods: the emulator, while protecting model parameters, retains significant inference ability, enabling misuse. The authors propose LLEOT, a framework that employs Loss Landscape Elevation (LLE) to enforce a fixed loss margin between the emulator and the original model. Theoretically, LLE degrades emulator inference (increasing perplexity) while preserving gradient alignment for adapter optimization. Combined with Collaborative Prompt Knowledge Distillation (CPKD) for soft prompts, LLEOT aims to secure both model capability privacy and data privacy during adaptation.

## Strengths
- **Meaningful problem identification and motivation:** The paper clearly articulates and formalizes the overlooked risk of "capability privacy" leakage in offsite tuning (Figure 1, Section 1). This addresses a timely and practical security concern in LLM adaptation.
- **Novel, theoretically grounded methodology:** The core LLE mechanism is elegant and simple. Theorem 1 provides a clean theoretical foundation, guaranteeing exponential perplexity increase (degrading inference) and gradient preservation under an ideal condition, which is a principled approach to the dual objectives.
- **Comprehensive and rigorous empirical evaluation:** Experiments are extensive, covering three base LLMs (Qwen2-1.5B, Gemma-2-2b, Llama-3.2-3B), two compression rates, and four QA benchmarks. The proposed Capability Privacy Leakage (CPL) metric is consistently applied. Results (Table 1) demonstrate LLEOT consistently achieves better privacy protection (lower CPL) while matching or surpassing the adaptation accuracy of strong baselines (OT, CRaSh).
- **Strong ablation studies and analysis:** Ablations effectively validate the contributions of the CPKD and LLE stages (Table 2), the components of CPKD (Table 3), and the effect of the margin *H* (Figure 4). Additional analysis on compatibility with data-privacy techniques (Figure 5) and adapter size (Table 7) strengthens the practical claims.

## Weaknesses
### Major:
- **Significant gap between theoretical guarantee and practical implementation.** Theorem 1 assumes the ideal condition \(L_E(P;x) = L_M(P;x) + H\) holds for **all** prompts \(P\) and inputs \(x\). However, the practical optimization (Equation 7) only minimizes the *expected absolute deviation* over a finite dataset \(X_e\) and a distribution of proxy prompts. The paper does not analyze how well this condition is approximated in practice, nor does it provide empirical validation of the critical gradient equality claim (e.g., cosine similarity of gradients). This undermines the core theoretical assurance that LLE simultaneously degrades inference and preserves alignment.
- **Insufficient evaluation of model capability privacy claims.** The privacy claim ("secures model capability privacy") is evaluated solely via the CPL metric—zero-shot accuracy ratio on four QA tasks. This does not establish that a malicious data owner cannot extract useful capabilities from the emulator via more sophisticated means (e.g., fine-tuning the emulator itself, distillation attacks, or querying on other tasks). The privacy guarantee therefore remains narrow and unverified against realistic adversarial strategies.

### Minor:
- **Computational overhead and practicality of emulator construction are unanalyzed.** The emulator construction involves three sequential stages (LayerDrop, CPKD distillation, LLE elevation). The computational cost and time relative to baseline methods (OT, CRaSh) are not discussed, which is important for assessing the framework's practical deployment cost.
- **Method is developed and evaluated exclusively for soft prompt adapters.** While the authors claim theoretical applicability to various adapters, the CPKD technique is specifically designed for soft prompts (using proxy prompts), and no experiments verify the framework's effectiveness with other popular adapters like LoRA. This limits the perceived generality of the contribution.
- **Assumption of original model access during LLE.** The LLE stage (Equation 7) requires computing the original model's loss \(L_M\) on elevation data \(X_e\), implying the model owner must perform forward passes with their private model. The sensitivity of results to the size and nature of \(X_e\) is not explored, leaving open questions about efficiency and data requirements.

## Nice-to-Haves
- A direct empirical analysis of gradient alignment (e.g., cosine similarity between \(\nabla_P L_E\) and \(\nabla_P L_M\) across prompts and data) would strengthen the validation of the core mechanism.
- Experimentation on open-ended generative tasks (beyond multiple-choice QA) would better test the privacy-utility trade-off in more realistic LLM usage scenarios.
- A preliminary experiment applying LLEOT with a LoRA adapter would help demonstrate the framework's generality beyond soft prompts.

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **Strength/Weakness about missing comparison to recent (2024-2025) baselines:** The instruction states to remove criticisms about missing related works, as external sources cannot be confirmed. The paper adequately compares against established OT and CRaSh methods.
- **Weakness questioning the existence or release status of cited models (Qwen2, Gemma-2, Llama-3.2):** The hard rules state all cited models are assumed to exist and be available.
- **Weakness about "unfair baseline comparisons" in terms of adapter type asymmetry:** The paper adapts OT and CRaSh for the soft-prompt setting (see Section 5.1). While the adaptation could be suboptimal, this is a methodological detail, not an inherent asymmetry that unfairly favors the authors' method. The core comparison on the stated problem remains valid.
- **Generic strengths like "the paper is well-written" or "the topic is important":** These have been removed per the instruction to keep only specific strengths.

## Suggestions
- **Clarify the theory-practice gap:** In Section 4.2 or the proof appendix, explicitly discuss the approximation nature of Equation 7 relative to the ideal condition in Theorem 1. Include an empirical analysis showing the distribution of \((L_E - L_M)\) across a held-out set of prompts and data after optimization to demonstrate how closely the fixed-margin condition is met.
- **Strengthen privacy evaluation:** Design and include an experiment where a malicious data owner attempts to extract capability from the provided emulator (e.g., by fine-tuning or distilling a smaller model on the emulator's outputs) and measure the standalone performance of the extracted model. This would provide a more adversarial test of the "capability privacy" claim.
- **Add a brief practicality analysis:** Include a discussion or a small table in the appendix comparing the training time/flop cost of constructing the LLEOT emulator versus the baseline OT and CRaSh emulators, to help readers assess the overhead of the enhanced privacy.

# Actual Human Scores
Individual reviewer scores: [4.0, 2.0, 4.0]
Average score: 3.3
Binary outcome: Reject
