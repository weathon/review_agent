=== CALIBRATION EXAMPLE 55 ===

# Final Consolidated Review
Now I have a thorough understanding of the paper. Let me synthesize the review.

---

## Summary
Gradient Storm extends the Sleeper Agent (SA) backdoor attack to enable simultaneous injection of multiple triggers (each with a distinct source–target pair and trigger type) into models trained from scratch. The key mechanism distributes poison crafting across *R* rounds of model retraining per cycle, so that different poisoned samples are aligned with different checkpoints of the surrogate model's parameter trajectory. The authors evaluate on CIFAR-10 and GTSRB, showing improved single-trigger ASR over SA (99.76% vs. 89.73% on CIFAR-10) and high per-trigger ASR in 2- and 3-attack scenarios.

---

## Strengths

- **Meaningful single-trigger improvement over SA.** The gain from 89.73% → 99.76% ASR on CIFAR-10 (Table 1) is not trivial, and the GTSRB result (84.25%, Table 2) also exceeds SA (58.19%), demonstrating that the multi-round procedure yields a reliably stronger attack beyond the parent method.
- **Multi-trigger novelty in the from-scratch, black-box, clean-label setting.** Existing multi-trigger works (Gong et al. 2021; Xue et al. 2024) operate in different threat models (outsourced cloud, transfer learning). Gradient Storm is the first gradient-matching-based framework for planting multiple heterogeneous triggers during from-scratch training, which is a practically significant threat model.
- **Broad and diverse transferability.** Table 4 shows ASR ≥ 95% across ResNet18, ResNet20, ResNet34, MobileNetV2, VGG11, and VGG16 — a wide range of architectures spanning depth and family — under the black-box assumption, which is strong empirical evidence for the cross-architecture generalization of the crafted poisons.
- **Maintained per-trigger ASR with no catastrophic interference.** Tables 5 and 6 show that adding a second or third trigger does not catastrophically degrade earlier triggers' individual ASR (most entries remain above 90%), resolving a natural concern about gradient interference between sequentially planted backdoors.

---

## Weaknesses

### Fatal
None.

### Major

- **Overstated robustness claims in abstract and conclusion.** The abstract claims the method "demonstrates its robustness against eight different poisoning defense mechanisms," and the conclusion says it "shows strong resilience against a range of poisoning defense mechanisms." Table 3 directly contradicts this: ABL (2.1% ASR), Gradient Shaping (8.9%), and DP-InstaHide (6.47%) all reduce the attack to near-random. Three of eight evaluated defenses effectively neutralize the attack. This is not a minor mischaracterization — it misrepresents the safety profile of the proposed attack to readers who may rely on these claims when evaluating defense research.

- **No defense comparison to Sleeper Agent.** The paper evaluates defenses only for GS, never running the same defenses against SA. It is therefore impossible to determine whether GS is *more* defense-resistant than SA, *equally* vulnerable, or even *less* robust. The defenses that succeed against GS (ABL, Gradient Shaping, DP-InstaHide) are well-known to exploit gradient alignment, which is the shared mechanism between SA and GS. Without this baseline, the defense evaluation section provides no relative insight.

- **No quantitative comparison to existing multi-trigger attacks.** The central new contribution is multi-trigger capability, yet Gong et al. (2021) and Xue et al. (2024) — the closest prior multi-trigger works — are discussed in related work but never compared against in any experiment. Even acknowledging that they operate in different threat models, adapting them to the from-scratch setting (as the paper does for HTBA) or at minimum discussing the quantitative gap would be necessary to substantiate the significance of the multi-trigger contribution.

- **"Expanded parameter space coverage" is the titular claim but is never substantiated.** The mechanism is motivated intuitively (lines 192–194: "perturbations are dispersed over a wider region of the parameter space"), but no analysis — empirical or theoretical — demonstrates that multi-round optimization actually increases gradient diversity or parameter-space coverage relative to the SA baseline. The performance gain could equally be attributed to an increased effective compute budget or sampling more target-class images. Without an ablation that controls for total compute (e.g., GS with R=2 rounds of B/2 samples each vs. SA with 1 round of B samples), the advertised mechanism remains unverified.

### Minor

- **Multi-trigger experiments limited to a single dataset and architecture.** Tables 5 and 6 use only CIFAR-10 and ResNet18. Given that single-trigger experiments cover GTSRB and transferability covers six architectures, extending at least one multi-trigger scenario to GTSRB or a second architecture would substantially strengthen the core contribution.

- **Computational overhead not discussed.** The method requires S × R retraining runs per trigger triple (e.g., 4 × 2 = 8 runs per attack, and with 3 attacks up to 24+ retraining runs). No wall-clock time, FLOPs, or comparison to SA's cost is provided. This is relevant for practitioners evaluating the attack's practical feasibility.

- **No ablation for sample selection criterion.** Algorithm 1 (Step 10) selects the *q* target-class samples with the highest gradient norm to perturb. This is a non-trivial design choice. The paper references an ablation study in Appendix A, but this appendix is absent from the submitted manuscript. The sensitivity to this criterion — and whether random selection or other criteria would perform comparably — cannot be assessed.

- **Notation inconsistency in Eq. 1.** The constraint is written as `θ(δ) ∈ arg min_θ ...` (lowercase δ) while the rest of the formulation consistently uses uppercase Δ for the perturbation matrix. This makes it ambiguous whether θ is a function of the full matrix or a single vector; it should be θ(Δ).

### Tiny

- The conclusion states the method achieves "attack success rate exceeding 90% in both single-trigger and multi-trigger scenarios" without acknowledging the three defenses that reduce ASR to single digits — inconsistency that should be corrected.
- The label "Stronger Noisy Gradients" in Contribution 1 is a misleading descriptor; gradient matching does not inherently produce "noisy" gradients.

---

## Nice-to-Haves

- **Isolation ablation of R vs. compute budget.** Compare GS with R=1 (same total budget as SA) to GS with R=2 (splitting budget across rounds) to isolate whether the gain comes from multi-round dispersal or from effective compute increase.
- **Perceptual stealth metrics.** LPIPS, SSIM, or PSNR comparisons would complement the L∞ constraint to validate the "minimal modification" claim more rigorously.
- **Trigger interference analysis at scale.** Evaluate >3 simultaneous attacks to probe the scalability limits of the sequential poisoning strategy and whether per-trigger ASR degrades as the number of concurrent attacks grows.
- **Evaluation on Vision Transformers.** ViT-based models are increasingly common; testing transferability to a ViT architecture would extend the attack's relevance.
- **Defense mechanism analysis.** A brief analysis of *why* ABL, Gradient Shaping, and DP-InstaHide succeed (and others fail) would strengthen the security narrative and guide future work.

---

## Removed Points
*These points are flagged for removal — treat them with caution.*

- **LC and Refool ASR are suspiciously low (2.3%, 2.93%):** The harsh critic suggests implementation errors. However, both methods are designed for distinct settings (LC is a clean-label targeted attack; Refool relies on reflection effects), and genuinely low ASR in CIFAR-10 from-scratch training has been documented. The comparison favors the baseline (lower competing ASR makes GS look better), which per instructions should not be flagged as a weakness — this asymmetry actually provides a stronger baseline to compare against. REMOVED.
- **Statistical significance / error bars:** Single-run evaluation is standard practice in backdoor attack research benchmarks at this scale. Demanding confidence intervals over multiple seeds is not the norm in this community. REMOVED.
- **Missing related works not listed:** Per instructions, cannot confirm existence of unlisted works. REMOVED.
- **Requesting theoretical proofs for the bilevel optimization:** This is an empirical systems paper; demanding formal convergence guarantees is not standard for attack papers in this community. REMOVED.
- **HTBA adaptation criticism:** The paper explicitly describes and justifies the from-scratch adaptation of HTBA (Section 4.1). The adaptation is reasonable and disclosed. REMOVED.

---

## Novel Insights

The most genuinely novel observation across all three reviews — not surfaced by the paper itself — is the following: the multi-round poison dispersal in GS shares a structural similarity with *continual learning under distribution shift*, where each retraining cycle shifts the surrogate's parameters while earlier poisons must remain active. The fact that earlier poisons empirically survive later retraining rounds (Tables 5–6) echoes the "forgetting" dynamics studied in continual learning, and the parameter-space coverage claim is essentially a claim about gradient diversity analogous to Experience Replay buffers. Analyzing GS through this lens could provide a principled explanation for when multi-round poisoning succeeds and when interference degrades earlier backdoors — a theoretical connection that would substantially strengthen the paper's contribution.

---

## Suggestions

1. **Correct the robustness framing.** Replace "demonstrates robustness against eight defenses" with accurate language such as "evaluated against eight defenses, maintaining high ASR against five while being mitigated by ABL, Gradient Shaping, and DP-InstaHide."
2. **Run Table 3 for Sleeper Agent.** Applying the same eight defenses to SA and presenting the comparison directly addresses the most important missing experiment and validates whether GS actually improves over SA in defense resistance.
3. **Add a controlled ablation of R.** Include a row in Table 1/2 comparing GS with R=1, R=2 (current), and SA (equivalent compute), all with identical total poison budgets. This directly tests the "expanded parameter coverage" claim.
4. **Include the Appendix A ablation** in the final submission's supplementary material, specifically justifying S=4, R=2, T=0.006, and the gradient-norm sample selection criterion.
5. **Extend Table 5 or 6 to GTSRB.** Even a single two-attack scenario on GTSRB would substantially bolster the generality of the multi-trigger contribution.

---

**Overall assessment:** Gradient Storm presents a meaningful and practically relevant extension of Sleeper Agent with a genuinely novel multi-trigger capability demonstrated empirically. The single-trigger improvement is real and the transferability results are strong. However, the paper's credibility is undermined by overstated abstract claims that contradict its own Table 3, the absence of any defense comparison to SA (making the core relative claim unverifiable), and the lack of direct quantitative comparison to existing multi-trigger methods — the most natural baseline for its primary contribution. The titular mechanism ("expanded parameter space coverage") remains intuitive assertion rather than demonstrated fact. In its current form, the paper is below the ICLR bar primarily due to these experimental gaps and framing issues; addressing them would make it a solid contribution to the backdoor attack literature.

# Actual Human Scores
Individual reviewer scores: [3.0, 3.0, 3.0, 3.0]
Average score: 3.0
Binary outcome: Reject
