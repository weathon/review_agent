Now let me search for calibration anchors.Now I have enough calibration data. Let me compose the final review.

---

## Summary

VQ-Transplant is a two-stage framework that replaces the VQ module in a frozen, pre-trained visual tokenizer (VAR) without end-to-end retraining: Stage I substitutes the VQ module by minimizing quantization error, and Stage II adapts only the decoder (5–20 epochs) to realign its priors with the new quantized space. The paper also introduces MMD-VQ, a non-parametric distribution-alignment variant, and validates the overall approach across multiple VQ algorithms, codebook sizes, and four datasets.

---

## Strengths

- **Two-stage necessity is well-documented.** Table 3 clearly shows that Stage I substitution alone worsens r-FID (e.g., MMD VAR: 1.49 post-substitution vs. 0.92 original), while Stage II decoder adaptation recovers and surpasses it (0.81), demonstrating the decoder–quantization mismatch insight concretely and convincingly.

- **Broad multi-method validation.** Tables 3 and 7 test five distinct VQ algorithms (Vanilla, EMA, Online, Wasserstein, MMD) in both multi-scale and fixed-scale configurations, showing consistent behavior patterns across settings. This breadth makes the framework conclusions more generalizable.

- **Cross-dataset generalization with strong results.** Tables 8–10 show that Wasserstein VQ and MMD VQ under VQ-Transplant generalize to FFHQ, CelebA-HQ, and LSUN-Churches with substantial margins over baseline tokenizers (e.g., FFHQ r-FID 1.21 vs. VQGAN-LC's 3.81), validating that the adapted models transfer beyond the training distribution.

- **Adaptation epoch analysis (Tables 4–5).** The paper systematically tracks r-FID over 20 epochs, providing actionable guidance on the budget-performance tradeoff that researchers would need when applying the method.

---

## Weaknesses

### Fatal
None.

### Major

- **Misleading efficiency claim.** The abstract states "reducing the training cost by 95%" and Table 1 reports a "21.8× speedup," computed as VAR's 960 GPU-hours vs. VQ-Transplant's 44 GPU-hours. However, VQ-Transplant is architecturally dependent on VAR's pre-trained encoder–decoder, which required those 960 GPU-hours to produce. VQ-Transplant is an incremental adaptation procedure, not a standalone training pipeline. The accurate claim is: *given publicly available VAR weights, exploring one new VQ configuration costs ~44 GPU-hours rather than ~960*. That is a useful and honest practical contribution, but it is categorically different from a standalone 95% cost reduction. A researcher without prior access to VAR would pay ≥1004 GPU-hours total. Neither Table 1 nor the abstract provides this qualification, creating a structurally misleading comparison.

- **Token count confounding in Table 2.** The paper's central quality comparison pits VQ-Transplant (MMD VQ: 512 tokens; MMD VAR: 680 tokens) against baselines that almost universally use 256 tokens (VQGAN variants, DQVAE, DiVAE, Llama GEN). Token count is a first-order driver of reconstruction quality — this is explicitly confirmed in the paper's own Table 2, where RQVAE improves from r-FID 3.20 (256 tokens) to 2.69 (512 tokens) to 1.83 (1024 tokens). No matched-token comparison is provided anywhere in the main paper. While the performance gaps are large enough that token count alone cannot fully explain them (RQVAE at 512 tokens still reaches only 2.69 r-FID vs. MMD VQ's 0.86–1.05 at 512 tokens, because VQ-Transplant inherits VAR's powerful encoder–decoder), the absence of any controlled experiment leaves the "state-of-the-art reconstruction quality" headline claims insufficiently supported. The same issue affects Tables 8–10, where all baselines use 256 tokens while VQ-Transplant uses 512.

### Minor

- **From-scratch comparison limited to early training (Table 6).** Table 6 compares VQ-Transplant (22h) against from-scratch training for only 25–35 GPU-hours (5–7 epochs). The paper itself notes that "discrete tokenizers typically require hundreds of epochs to achieve high-quality visual reconstruction when trained from scratch," so the from-scratch entries (r-FID 1.26–1.40) are far from convergence. The comparison at matched compute budget is operationally valid, but the paper does not test whether VQ-Transplant achieves better *final* quality than a fully converged from-scratch model — which is the stronger claim implied by the framing.

- **MMD-VQ is not consistently the best performer.** On FFHQ (Table 8, K=32768 after adaptation), Wasserstein VQ achieves r-FID 1.21 vs. MMD VQ's 1.37. On LSUN-Churches (Table 10), Wasserstein VQ achieves r-FID 1.79 vs. MMD VQ's 1.87. The paper's text does not engage with these cases where the proposed method is not the winner. The theoretical motivation — that MMD handles non-Gaussian features better than Wasserstein VQ — is not empirically validated in a setting where features are measurably non-Gaussian, leaving the theoretical justification detached from the experimental evidence.

- **Cross-dataset experimental setup underspecified.** Section 5.3 does not explicitly state whether the VQ module trained on ImageNet-1k is directly applied to FFHQ/CelebA-HQ/LSUN-Churches or whether a separate per-dataset substitution is performed. Clarifying this is essential for interpreting the cross-dataset generalization claim.

### Trivial

- **r-IS directional notation inconsistency.** Tables 2 and 3 label r-IS with "↓" (lower is better), while Table 7 labels it "↑" (higher is better). The discussion text consistently treats higher IS as better. The Tables 2–3 notation appears to be erroneous; this should be corrected.

---

## Nice-to-Haves

- A matched-token comparison within the VQ-Transplant framework (e.g., running fixed-scale MMD VQ at 256 tokens) would substantially strengthen the quality claims in Table 2 by isolating the contribution of the VQ method from token count.
- Downstream generative quality evaluation (e.g., FID on an autoregressive or diffusion model conditioned on VQ-Transplant tokens) would close the loop from tokenizer quality to the practical motivation of visual generation research.
- A setting where features are demonstrably non-Gaussian (e.g., extreme out-of-distribution data) and where MMD-VQ's advantage is measurably larger would validate the theoretical motivation for MMD over Wasserstein VQ.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Stage I should include decoder reconstruction signal"** (Harsh Critic, Section 4.1): The two-stage design with frozen decoder in Stage I is the paper's core architectural choice. Ablating every possible design variant is out of scope for a paper presenting a framework, and the paper provides sufficient justification for the decoupled design. Removed as a strawman.

- **"Democratizing quantization research is overextended"** (Harsh Critic, Abstract): This is a rhetoric/framing concern rather than a substantive technical weakness. The practical value to researchers who do have access to VAR weights is real. Removed as pure style/scope nitpick.

- **"EMA VQ nearly matches MMD VQ"** (Harsh Critic): Observed in Table 7 at K=32768 (EMA VQ r-FID 0.99 vs. MMD VQ 0.97). A 0.02 gap is within normal variance. The claim that "the marginal contribution of the MMD objective is questionable" based on a 0.02 gap at one configuration is not well-supported; at K=65536, MMD VQ leads by 0.13 r-FID (0.86 vs 0.99). Removed as overstated.

- **"Cross-dataset token count inflated vs. baselines"** (partially): Retained as a Minor weakness rather than a separate removal, because the core FFHQ/CelebA-HQ/LSUN comparison with non-VQ-Transplant baselines at 256 tokens is a real concern — but it is less severe than the ImageNet-1k Table 2 issue because the gaps are larger.

---

## Novel Insights

The paper's sharpest empirical insight — that reducing quantization error alone can worsen reconstruction quality because the decoder's priors are conditioned on the original codebook distribution — is clearly documented and is more broadly useful than the paper's specific framework. This decoder–quantization mismatch phenomenon implies that any work replacing or fine-tuning VQ modules in pretrained tokenizers must account for decoder-side adaptation, a point largely absent from prior literature. The systematic demonstration across five VQ algorithms and two scale configurations strengthens the generality of this observation.

---

## Calibration

**Anchor papers compared:**
- `/home/wg25r/review_agent/human_reviews/GMwRl2e9Y1.md` — avg 8.0 ("Rotation Trick" for VQ-VAE). Deeper theoretical contribution (gradient propagation through VQ layer), cleaner comparisons across 11 settings, no efficiency overclaim. VQ-Transplant is below this level.
- `/home/wg25r/review_agent/human_reviews/gzqrANCF4g.md` — avg 8.0 ("Language Model Beats Diffusion"). Highly impactful video/image tokenizer with broad downstream validation. VQ-Transplant is substantially below this level.
- `/home/wg25r/review_agent/human_reviews/8ishA3LxN8.md` — avg 6.5 ("Finite Scalar Quantization"). Simple VQ replacement with honest competitive comparisons; clean contribution without efficiency overclaim. VQ-Transplant has a comparable breadth but weaker comparison setup.
- `/home/wg25r/review_agent/human_reviews/yGnsH3gQ6U.md` — avg 6.0 ("Binary Spherical Quantization"). New quantization method with some fairness-of-comparison concerns (parameter count). VQ-Transplant has analogous issues (token count) but also the efficiency overclaim on top.
- `/home/wg25r/review_agent/human_reviews/qOForsjh4q.md` — avg 4.0 ("Same Accuracy, Twice As Fast"). Efficiency claims mislead on what is new vs. warm starting; weak generalization. VQ-Transplant's efficiency issue is real but less fundamental — the core two-stage framework is novel and not just warm-starting.
- `/home/wg25r/review_agent/human_reviews/4zygH3k8Zr.md` — avg 4.4 ("Replacement Learning"). Parameter-efficient training with unclear positioning. VQ-Transplant has a cleaner, more clearly scoped contribution.

VQ-Transplant sits below the accepted tokenizer/VQ papers (6.0–6.5) due to the efficiency overclaim and token count conflation, but clearly above the rejected efficiency papers (4.0–4.4) because the core idea and two-stage validation are genuinely novel and well-executed. The cluster center of the medium-quality anchors (BSQ at 6.0, FSQ at 6.5) is the upper bound; the efficiency/comparison issues pull the score down. I land at **5.5**.

---

## Score and Decision

**Score: 5.5**
**Decision: Reject** — The paper contains a genuine and practically valuable contribution (cheap VQ iteration given pretrained tokenizers, two-stage mismatch correction). However, two major issues must be addressed before acceptance: (1) the efficiency framing systematically omits the prerequisite VAR training cost, making the headline "95% cost reduction" claim false as stated, and (2) the primary quality comparison (Table 2) conflates token counts, which is the main axis of the paper's quality advantage. These are not presentation fixes — they require either new controlled experiments or a fundamental reframing of the contribution scope. In its current form, the paper overclaims on both efficiency and quality axes.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>