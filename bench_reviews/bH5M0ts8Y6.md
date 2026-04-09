## Summary

VINCIE proposes learning in-context image editing directly from native video data by constructing interleaved multimodal sequences (frames, textual transition descriptions, and segmentation masks) and training a DiT model with three proxy tasks: next-image prediction, current segmentation prediction, and next segmentation prediction. The approach demonstrates scalability and achieves strong results on multi-turn editing benchmarks, while also showing emergent capabilities in story generation and multi-concept composition.

## Strengths

- **Novel data paradigm with strong scalability evidence.** The core insight—that video temporal dynamics naturally provide the multi-turn editing signal that curated image pairs cannot—addresses a genuine data bottleneck. Figure 5 provides compelling scalability evidence: later-turn success rates (Turn-4, Turn-5) show nearly log-linear improvement with data scale from 0.25M to 10M sessions, while Table 5 shows that video-sequence pretraining outperforms pairwise training by +16.4% at Turn-1 and +21.0% at Turn-5 on MSE-Bench. This is a concrete, quantitative demonstration that video structure provides something pairwise data cannot.
- **Well-designed proxy tasks with meaningful ablation.** The CSP and NSP tasks are not just auxiliary losses—they serve as controllable intermediate representations. Table 3 shows that the chain-of-editing strategy (CS→NS→I) improves Turn-5 success rate from 10.3% to 17.3%, and the paper demonstrates that segmentation prediction mitigates the subject position drift inherent in video training (Figure 7, Section 4.4). The practical benefit is clear: the model learns to ground edits before generating them.
- **In-context editing mitigates artifact accumulation.** Figure 6 and Table 4 provide evidence that incorporating full context from previous turns significantly reduces artifacts compared to sequential single-turn editing. Table 4 shows L1 distance nearly halving (0.155→0.086) when dummy context is added at Turn-1, confirming that the model actively leverages context rather than treating each turn independently.

## Weaknesses

### Major:

- **Main comparison table (Table 2) omits the proposed method's results.** MSE-Bench is the paper's own proposed benchmark, yet Table 2 lists 17 baselines without including "Ours." The text states "our method achieves a 25% success rate at turn-5," but the reader cannot verify this against the tabulated baselines. The method's MSE-Bench numbers appear only indirectly in the ablation table (Table 5, "sequence" row, Turn-5 = 22.0%). This presentation gap makes it impossible to directly compare VINCIE against the baselines it claims to outperform on the paper's own benchmark. Additionally, the "25%" claim in the text (Section 4.3) does not match the 22.0% shown in Table 5; this inconsistency further erodes confidence in the reported results.

- **"Trained exclusively on videos" claim is misleading for SOTA results.** The Abstract states the model is "trained exclusively on videos" and achieves "state-of-the-art results," but Table 1 shows the best-performing variants are "Ours* + SFT," which undergo supervised fine-tuning on curated image-pair datasets (OmniEdit, SEED-Data-Edit, etc.; see Appendix C.7). While the video-only model (Ours* without SFT) performs competitively, the SOTA claims rely on additional image-pair SFT. The paper should clearly distinguish VINCIE-Base (video-only) from VINCIE-SFT throughout, and the Abstract should not conflate the two when claiming SOTA. The Limitations section (Appendix F) also does not acknowledge this gap.

- **MSE-Bench contains only 100 test instances.** For a benchmark that the paper proposes to "advance research in this area," 100 instances is statistically fragile. Small benchmark sizes are vulnerable to outlier-driven swings in success rate metrics. The paper should acknowledge this limitation explicitly and, at minimum, report confidence intervals or variance estimates to support the comparative claims.

- **Key ablation (Table 3) uses an intermediate checkpoint, weakening the evidence for CSP/NSP.** The caption states: "This ablation study was conducted using an intermediate checkpoint, so the reported numbers may not be directly comparable to those in other tables." Since the CSP/NSP contribution is a central design claim, the ablation should be reproduced on the final checkpoint. Without this, the reader cannot confirm that the reported gains (e.g., Turn-5: 10.3%→17.3%) hold for the model whose results are advertised elsewhere.

### Minor:

- **VLM annotation accuracy is 75.14% (Appendix D.3), but the impact of this ~25% label noise is not analyzed.** The paper argues that scale compensates for noise (citing InstructPix2Pix's similar approach), but provides no ablation or analysis confirming this. A simple experiment filtering low-confidence annotations or injecting synthetic noise would clarify whether the noise imposes a performance ceiling.

- **GPT-4o evaluation has moderate correlation with human judgment (Pearson r = 0.4858, r² ≈ 0.24).** While GPT-4o is a reasonable proxy, it explains less than 25% of the variance in human scores. The human evaluation results (Table 6, Appendix D.1) should be elevated to the main text to corroborate the GPT-4o numbers, especially given the small benchmark size.

### Trivial:

- The scalability figure (Figure 5) has a large gap between 0.25M and 1.25M data points; finer granularity in the low-data regime would better illustrate sample efficiency.

## Nice-to-Haves

- Quantitative evaluation of "emergent" capabilities (story generation, multi-concept composition) rather than relying solely on qualitative examples (Figures 1, 19–21).
- Comparison against training on web-scraped interleaved image-text datasets (e.g., Obelics) to isolate whether the benefit comes from video temporal dynamics specifically or from the interleaved sequence format more generally.
- Error analysis breaking down performance by edit category (object addition/removal, attribute change, camera change) on MSE-Bench, to validate the "universal" editing claim and reveal where video-native priors help or hurt.
- Analysis of performance beyond 5 turns to test the long-form content claim, as context window limitations may degrade quality.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Not yet released" / reproducibility concerns about cited models (e.g., Nano Banana, GPT Image 1).** The paper cites them; they are assumed to exist per review rules. Removed.
- **Demand for comparison with text-conditioned baselines outside the in-context editing scope.** The paper explicitly scopes its contribution to in-context image editing; requesting comparison with a fundamentally different paradigm is scope creep.
- **Criticism that pixel-wise L1/L2 metrics in Table 4 are "outdated."** Table 4 also reports DINO and CLIP-I scores alongside L1/L2; the pixel metrics supplement rather than replace perceptual metrics. The concern is partially addressed.
- **Reproducibility concerns about undisclosed hyperparameters and training details.** Appendix C provides extensive implementation details including data construction, architecture, SFT datasets, and training configurations. Removed as nitpick.
- **Concern that the data construction pipeline is not reproducible because VLM/grounding model thresholds are not fully specified.** The appendix provides the prompting instructions and pipeline description. Standard practice in this area uses similar automated pipelines; demanding every threshold is a nitpick.
- **Criticism about the "teacher-student" dynamic limiting "universal" editing because GroundingDINO/SAM2 may fail on certain categories.** This is speculative without evidence that it actually limits the model's performance. The model shows generalization to uncommon edit types (Figure 14), suggesting this bound is not tight in practice.

## Novel Insights

The paper reveals an interesting asymmetry in how video data benefits multi-turn editing: earlier turns (Turn-1) saturate quickly with data scale, while later turns (Turn-4, Turn-5) exhibit nearly log-linear improvement. This suggests that video data's primary value is not in teaching *what* to edit (which can be learned from single pairs), but in teaching *how to maintain coherence across accumulated edits*—precisely the signal that pairwise data cannot provide. This finding has implications beyond image editing: any generative task requiring long-form consistency (story generation, multi-step reasoning) may benefit more from sequential training data than from scaling single-step pairs.

## Suggestions

- **Add VINCIE results directly to Table 2** (both base and SFT variants) with the exact numbers matching the text claims, and reconcile the "25%" vs. "22%" discrepancy explicitly.
- **Reproduce the CSP/NSP ablation (Table 3) on the final checkpoint** so that the proxy task contribution is verifiable against the main reported results.
- **Report confidence intervals or variance on MSE-Bench results** given the 100-instance size, and move the human evaluation (Table 6) to the main text to support the GPT-4o numbers.
- **Adopt consistent nomenclature** (e.g., VINCIE-Base vs. VINCIE-SFT) throughout the paper, and adjust the Abstract to clearly attribute SOTA results to the SFT variant rather than implying they come from video-only training.

## Quality Assessment

- **Novelty:** High. The video-to-editing paradigm shift is genuinely new; no prior work demonstrates learning in-context image editing solely from native video data with this level of success.
- **Technical soundness:** Moderate. The method is well-designed, but key ablations use intermediate checkpoints, and the "video-only" vs. SFT distinction is blurred in presentation.
- **Empirical support:** Mixed. Strong on MagicBrush and scalability; weaker on MSE-Bench due to small benchmark size, missing method entry in main table, and moderate eval metric correlation.
- **Significance:** High if the presentation issues are resolved. The scalability of video-based training and the artifact mitigation finding are impactful for the field.
- **Clarity:** Moderate. The core method is clearly described, but the conflation of video-only and SFT results, and the missing Table 2 entry, create confusion that undermines an otherwise well-written paper.