Now I have all the information I need. Let me compile the final review.

## Summary

The paper proposes VQ-Transplant, a two-stage framework that enables plug-and-play replacement of vector quantization modules in frozen pre-trained visual tokenizers (e.g., VAR), followed by lightweight decoder adaptation (5 epochs) to resolve decoder-quantization mismatch. A secondary contribution, MMD-VQ, uses maximum mean discrepancy for distributional alignment between codebook and feature vectors. The paper demonstrates the framework across five VQ variants in both multi-scale and fixed-scale configurations, showing that distribution-alignment methods (Wasserstein VQ, MMD-VQ) are most compatible with the transplant approach.

## Strengths

- **The core VQ-Transplant idea is practical and well-motivated.** Decoupling VQ research from full tokenizer retraining addresses a real bottleneck. Table 3 demonstrates that at matched codebook size K=4096, MMD VAR adaptation achieves 0.91 r-FID vs. the original VAR's 0.92, confirming that decoder adaptation with just 5 epochs can recover (and slightly exceed) original performance—a genuinely useful empirical finding.

- **Systematic ablation across VQ variants and configurations.** The paper tests five VQ methods (Vanilla, EMA, Online, Wasserstein, MMD) under both multi-scale (Table 3) and fixed-scale (Table 7) settings, with substitution and adaptation phases separately reported. This breadth provides useful data on which VQ properties matter for transplant compatibility (distribution-alignment methods consistently outperform alternatives).

- **Informative adaptation epoch analysis.** Tables 4 and 5 show that r-FID continues to improve from 5 to 20 epochs (0.91→0.79 at K=4096; 0.81→0.74 at K=8192), characterizing the performance-compute tradeoff and confirming that the 5-epoch default is a practical choice, not a ceiling.

- **Cross-dataset applicability demonstrated.** Tables 8–10 show strong reconstruction on FFHQ, CelebA-HQ, and LSUN-Churches, with the transplanted models outperforming fully-trained baselines like VQGAN-LC (1.21 vs. 3.81 r-FID on FFHQ).

## Weaknesses

### Fatal
None.

### Major

- **The headline "0.81 r-FID" result is confounded by codebook size.** The paper's most prominent result—MMD VAR achieving 0.81 r-FID, emphasized in the abstract, introduction, and Table 2—uses K=8192, double the VAR baseline's K=4096 (0.92 r-FID). At matched codebook size K=4096, MMD VAR adaptation achieves only 0.91 r-FID—a marginal 0.01 improvement over the original VAR. The paper never evaluates the original VAR tokenizer with K=8192, making it impossible to determine whether the 0.81 result stems from MMD-VQ or simply from doubling the codebook. Any VQ method would likely benefit from a larger codebook. The abstract's claim of "near state-of-the-art reconstruction fidelity" and the introduction's claim of "superior reconstruction fidelity" (line 23, 28) are built on this confounded comparison. Without a VAR K=8192 baseline, the central empirical claim does not hold as stated.

- **The 21.8× speedup comparison conflates multiple independent factors.** Table 1 compares VQ-Transplant (22 hrs, 2×A100, ImageNet-1k ≈1.3M images) against VAR from-scratch training (60 hrs, 16×A100, OpenImages ≈9M images). This conflates dataset size (~7× difference), GPU count (8× difference), and starting from a pre-trained model vs. training from scratch. While the paper's premise assumes a pre-trained tokenizer already exists, the specific "21.8× speedup" and "95% training cost reduction" numbers are inflated by these confounds. A more meaningful comparison would normalize for dataset size or compare against fine-tuning the existing model on ImageNet-1k with a new VQ module jointly (which the paper discusses in Appendix C but does not include in Table 1).

- **MMD-VQ's claimed superiority over Wasserstein VQ is not empirically established.** Across all tables, MMD-VQ and Wasserstein VQ produce near-identical results: at K=4096 adaptation, 0.91 vs. 0.93 r-FID; at K=8192, 0.81 vs. 0.83 (Table 3). On FFHQ (Table 8), Wasserstein VQ adaptation (1.21 r-FID) actually outperforms MMD VQ (1.37 r-FID). No variance, confidence intervals, or statistical significance tests are reported. The paper argues MMD-VQ's advantage is being "nonparametric" compared to Wasserstein VQ's Gaussian assumption (Section 4.2), but provides no empirical evidence where this matters—feature distributions that are "multi-modal, heavy-tailed, or otherwise non-Gaussian" are never tested or visualized. MMD-VQ as a secondary contribution is not well-supported.

### Minor

- **Missing downstream generation evaluation.** The paper evaluates only reconstruction quality (r-FID, PSNR, etc.) but never tests whether transplanted tokenizers work for image generation—their ultimate purpose. A tokenizer with slightly better r-FID but different token distributions could produce worse generation quality when used to train an autoregressive model. At least one downstream generation experiment would substantially strengthen the paper.

- **"Cross-dataset generalization" terminology is misleading.** Section 5.3 claims "generalization" across datasets, but the adaptation phase is performed separately on each target dataset (FFHQ, CelebA-HQ, LSUN-Churches). This is per-dataset fine-tuning, not zero-shot generalization. The word "generalization" is misused and should be replaced with "applicability" or "transfer."

- **The from-scratch comparison (Table 6) is uninformative.** Training MMD VAR from scratch for 5–7 epochs and showing it underperforms VQ-Transplant is expected—the model hasn't converged. The paper acknowledges this ("typically require hundreds of epochs"), which raises the question of why this comparison is presented at all without a fully-converged from-scratch baseline.

- **Scope limitation with LDM tokenizers underreported.** The paper briefly notes VQ-Transplant achieves "reasonable performance" on LDM-16 but with "lower adaptability compared to VAR-based models" (line 175–181). If the framework's effectiveness depends on the tokenizer having been trained with adversarial loss (providing the discriminator for decoder adaptation), this should be stated as an explicit scope limitation rather than relegated to a half-sentence.

### Trivial
None.

## Nice-to-Haves

- Running VAR with K=8192 to properly control the codebook-size confound—this single experiment would resolve the paper's most critical weakness.
- Evaluating at least one downstream generation task (e.g., training VAR's autoregressive model on transplanted tokenizer tokens and reporting g-FID).
- Visualizing encoder feature distributions to test whether they deviate from Gaussianity, which would empirically motivate MMD-VQ's nonparametric advantage over Wasserstein VQ.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"The paper never evaluates with a genuinely novel VQ method"** — The paper's stated goal is enabling rapid exploration of novel VQ methods, but testing with established variants is sufficient to validate the framework. Demanding a novel VQ method as a test case is scope creep; the framework's value is in enabling future VQ research, not in producing new VQ methods itself.

- **"Joint optimization in Appendix C raises the question of why the inferior two-stage approach is the main proposal"** — The paper explicitly acknowledges this (line 175) and explains the tradeoff (increased training time). The two-stage approach is the simpler, more efficient default; joint optimization is a reasonable alternative for those with more compute. This is not a weakness.

- **"The two-stage procedure is presented without justification for why two stages are necessary"** — The paper provides clear justification: Stage I minimizes quantization error, and Stage II resolves decoder-quantization mismatch. The ablation in Table 3 shows that Stage I alone (substitution) produces worse r-FID than the original model, and Stage II (adaptation) recovers and exceeds performance. This justifies the two-stage design.

- **Reproducibility concerns about undisclosed hyperparameters** — Per the rules, these are trivial implementation details impractical to include in a submission.

- **"Not yet released" or availability concerns about cited models/tools** — Per the hard rules, cited entities are assumed to exist and be available.

## Novel Insights

The paper reveals an interesting asymmetry in the VQ-transplant setting: reducing quantization error (Stage I) does not directly improve reconstruction fidelity, because the decoder's learned priors are calibrated to the original quantization space. The decoder adaptation stage (Stage II) effectively converts the reduced quantization error into improved reconstruction. This two-stage finding—that quantization error and reconstruction fidelity are decoupled when the decoder is frozen—is a useful empirical insight for the community, even if the paper overclaims the magnitude of the final improvement.

## Suggestions

- **Run and report VAR with K=8192 as a baseline.** This is the single most impactful change—without it, the headline result cannot be attributed to the proposed method.
- **Reframe the primary result as the matched codebook-size comparison (K=4096: 0.91 vs. 0.92 r-FID)** and present the K=8192 result as a scaling study, not as evidence of method superiority.
- **Tone down the speedup claim** by normalizing for dataset size or comparing against fine-tuning the existing model, not just against from-scratch training on a larger dataset with more GPUs.
- **Acknowledge that MMD-VQ and Wasserstein VQ perform comparably** rather than claiming superiority; reframe MMD-VQ as a theoretically sound alternative with equivalent empirical performance.

## Score and Decision

**Calibration anchors:**

- **High-scoring anchors (>7):** StableToken (avg 7.50, Accept Poster) — novel multi-branch VQ tokenizer with strong downstream evaluation and clear problem framing; InfoTok (avg 7.33, Accept Oral) — principled ELBO-based VQ training with generation evaluation. These papers have stronger empirical validation including downstream tasks and cleaner claims.

- **Medium-scoring anchors (4–6):** VQBridge (avg 6.0, Accept Poster) — achieves 100% codebook utilization with generation evaluation, similar VQ tokenizer space but stronger empirical contribution; SSQ paper (avg 4.67, Reject) — principled tokenizer framework with reconstruction-generation paradox insight but limited scope; PinTok (avg 4.50, Reject) — overclaimed speedup with unfair baselines.

- **Low-scoring anchors (<3):** EfficientLLM (avg 2.0, Withdrawn) — performance gains from massive compute, not the proposed method; NoLoRA (avg 2.0, Withdrawn) — core idea identical to prior work with unsubstantiated claims.

The paper under review is more substantial than the low-scoring anchors (it has a real framework contribution, not fabricated or duplicated work). It is weaker than VQBridge (which includes generation evaluation and doesn't overclaim) and clearly weaker than the high-scoring anchors. It is most comparable to the SSQ paper (principled framework, reconstruction-only evaluation, limited scope) and PinTok (overclaimed speedups). However, VQ-Transplant does demonstrate a genuinely practical finding at matched codebook sizes (0.91 vs. 0.92 r-FID), which is a real contribution even if oversold. The paper is slightly stronger than SSQ/PinTok because the core framework idea is more immediately useful and the empirical finding at matched settings is valid.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>