=== CALIBRATION EXAMPLE 21 ===

# Final Consolidated Review
## Summary
This paper proposes **Principal Spectral Regularization (PSR)**, a partial spectral treatment of momentum that shrinks only the dominant singular directions rather than fully orthogonalizing the update as in Muon. The paper’s core empirical message is that this lighter-weight spectral intervention can make SGD with momentum outperform AdamW in the reported LLaMA pretraining setups, while being substantially cheaper than full orthogonalization in memory and, at sufficiently large matrix sizes, runtime.

## Strengths
- **The paper identifies a concrete and potentially important optimizer design regime between Adam-style diagonal adaptation and Muon-style full orthogonalization.** The proposed PSR is not just “another optimizer tweak”: it explicitly targets the top singular directions of momentum via block Lanczos bidiagonalization and deflation (Alg. 1), which is a specific and technically meaningful interpolation between scalar and full-matrix spectral methods.
- **The spectral observations are specific and useful rather than generic.** Fig. 1 and Sec. 3.1 argue that momentum in LLM training exhibits a “spiked-head-heavy-tail” structure, with clear differences between attention and MLP layers. This gives a concrete mechanistic rationale for why shrinking only a small number of dominant directions might preserve useful tail structure better than full flattening.
- **The paper does a better-than-usual job of connecting mechanism, toy intuition, algorithm, and systems cost.** The toy Styblinski–Tang study is not sufficient evidence for LLM claims, but it does serve as a mechanistic illustration of the hypothesis that full equalization can be suboptimal while partial shrinkage can be better. The method is then instantiated in a clear algorithm, accompanied by complexity analysis and practical timing/memory measurements.
- **The computational trade-off against Muon is substantiated in a fairly concrete way.** Appendix E provides an explicit FLOP accounting, and Table 3 shows that PSR can be markedly cheaper in memory and runtime than Newton–Schulz orthogonalization for larger matrices (especially 7B/70B-scale shapes), even under a naive PyTorch implementation.
- **The authors are appropriately transparent that PSR does not dominate Muon.** Sec. 5.2 and the conclusion clearly state that Muon is better in long-horizon convergence and downstream performance in the reported extended-training setting. This honesty is valuable and avoids a common failure mode in optimizer papers.

## Weaknesses

### Major:
- **The paper’s framing overreaches relative to its own evidence, especially in how strongly it questions full orthogonalization.** The strongest empirical takeaway supported by the results is that PSR is a cheaper partial alternative that can beat AdamW and approach Muon in some regimes. The paper does *not* establish that full orthogonalization is broadly suboptimal. In fact, the paper itself reports that Muon overtakes PSR in longer training and on downstream average performance for the 1.3B extended run: Sec. 5.2 states that “SGD-M with PSR is still stricly worse than running Muon,” and the conclusion says “our PSR method does not match Muon in downstream performance or scaled-up training.” This does not invalidate PSR, but it does mean the title/abstract-level message is stronger than the evidence.
- **The empirical case for the headline “surpass Adam” claim is not yet fully convincing at ICLR standards because the LLM results appear to be single-run and the reported margins are modest.** The paper provides repeated-run statistics only for the toy function (Table 7), not for the main LLM experiments. Since optimizer differences in pretraining can be small and noisy, especially when the reported downstream average gains are only a few tenths of a point, the lack of variance estimates or multi-seed confirmation weakens confidence that PSR’s advantage over AdamW is robust rather than run-specific.
- **Hyperparameter robustness is under-supported for the scales where the paper wants to make practical claims.** PSR depends on at least three nontrivial choices: shrinkage strength, rank fraction, and update rescaling. The paper does provide some ablation (Table 9), which is good, but this is limited and does not really establish cross-scale robustness for 3B/7B training. The fixed RMS multiplier of `0.18` is motivated empirically from Table 6, not derived or stress-tested; this makes it hard to tell how much of the gain comes from the spectral idea versus hand-tuned normalization.
- **The efficiency narrative is incomplete because the paper’s practical speed advantage only appears clearly at larger scales, while some core training results are on smaller models where PSR is not faster in wall-clock terms.** The paper does acknowledge this limitation: Sec. 4 explicitly says PSR “can be more time-consuming” in practice for small-scale LLMs, and Table 3 shows PSR slower than Newton–Schulz for some 1.3B/3B attention shapes. That honesty is appreciated, but it means the practical story should be narrower: PSR is not uniformly efficient in end-to-end runtime; it is a favorable trade-off mainly at larger matrix sizes or under memory pressure.
- **The long-horizon evidence is too limited to support strong optimizer conclusions beyond early/mid training.** The paper itself notes that PSR’s early advantage diminishes and that Muon becomes better in later stages. The 350M/1.3B/3B main curves are only to 10k steps, and the sole extended-training experiment is one 1.3B run to 36B tokens. For optimizer papers, especially when making claims about replacing Adam-family methods in LLM pretraining, stronger long-run evidence is important.

### Minor
- **The toy optimization section is useful as intuition but too weak to carry much evidential weight.** The paper partially acknowledges this in Appendix D.1 (“the connection between mathematical function optimization and LLM pretraining is relatively vague”), which is the right stance. Still, the main text leans on it somewhat heavily when motivating claims about full orthogonalization being unnecessary.
- **There is internal inconsistency in reported PSR hyperparameters between prose and Algorithm 2.** In Sec. 4 the text says the “optimal regularization factor” is `η = 0.95`, but Algorithm 2 calls `PSR(..., η = 0.5, r = 32[m])`, which appears inconsistent with the textual description `r = min(m,n)/32` and with the table notation. This may be a notation mix-up between shrinkage and retained proportion, but as written it is confusing and should be corrected.
- **The paper would benefit from cleaner separation of claims about sample efficiency, downstream quality, and systems efficiency.** Right now these are somewhat blended, which contributes to overstatement. The evidence supports “beats AdamW in the reported settings” more strongly than “questions full orthogonalization” or “surpasses Muon-like approaches.”

### Trivial
- **The distributed-training implications are not discussed.** For very large-scale training, communication and parallelization details for Lanczos/QR/SVD matter, especially if one wants to claim systems relevance beyond local kernel costs. This is not a fatal omission, but it limits practical interpretability.
- **Wall-clock reporting is component-level rather than end-to-end throughput.** Table 3 is still useful, but tokens/sec or total training-time comparisons would make the practical trade-off much easier to assess.

## Nice-to-Haves
- Add multi-seed LLM results, at least for one or two representative models (e.g., 350M and 1.3B), to validate that PSR’s gains over AdamW are outside normal pretraining variance.
- Reframe the title/abstract/conclusion to emphasize the supported claim: PSR is a strong **compute/memory-efficient partial alternative** that can outperform AdamW and approach Muon, rather than evidence that full orthogonalization is generally unnecessary.
- Extend long-horizon training for 3B/7B models, since the paper’s own narrative suggests optimizer rankings may change later in training.
- Provide sensitivity curves for the fixed `0.18` RMS rescaling and clarify whether the gains persist under less tuned normalization.
- Clarify the apparent inconsistency between Algorithm 2 and the prose on PSR hyperparameters.
- Plot validation loss/perplexity against wall-clock time and total optimizer FLOPs to directly show the practical Pareto frontier.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **Claims about missing specific related-work baselines such as AdaSGD.** I cannot verify externally which omitted baselines are essential, and the review instructions explicitly say not to mention missing related works.
- **Criticism that the paper should use modern sequence lengths such as 2048–8192.** This is partly scope creep. The paper does include one 2048-length setup in Appendix B for the extended 1.3B run, and sequence length expansion would strengthen the paper but is not a core flaw.
- **Objections about code release status or reproducibility based on unreleased artifacts.** The paper states code will be released upon publication; per instructions, such concerns should not be treated as weaknesses here.
- **Formatting/parser issues or awkward phrasing.** The extracted text contains artifacts, and style-only complaints are not appropriate.
- **Claims that the paper demonstrates full orthogonalization is suboptimal in general.** This appears in one reviewer’s listed strengths, but the paper’s own evidence does not support such a broad statement; the supported insight is narrower and is reflected in the main review instead.
- **Fairness objections about SOAP comparison due to unequal asymmetry if the asymmetry favors the baseline.** The paper’s comparison is limited, but not in a way that clearly disadvantages the authors’ method more than the baseline, so this is not a strong review point under the given rules.
- **Speculative numerical-instability criticism about Lanczos/SVD degeneracy.** The paper does not present evidence of instability problems, and the review input raised this mainly by analogy to another paper rather than from direct evidence here.

## Novel Insights
The most useful synthesis is that this paper’s real contribution is **not** a refutation of Muon-style full orthogonalization, but the identification of a practically interesting middle regime: for LLM momentum spectra with a pronounced low-rank “head,” selectively shrinking only a few dominant directions can recover much of the benefit of spectral flattening while preserving heavy-tail structure and greatly reducing orthogonalization cost. The paper’s own results suggest a nuanced optimizer picture: head suppression seems particularly helpful in early-to-mid training and for beating AdamW, while fully tail-amplifying methods like Muon may matter more for late-stage convergence and final downstream quality. That trade-off is a genuinely interesting insight for optimizer design, even if the current title and framing oversell it.

## Suggestions
- Rewrite the framing around a narrower but stronger claim: **PSR is an efficient partial spectral method that beats AdamW in the reported setups and narrows much of the gap to Muon.**
- Add multi-seed confirmation for at least a representative subset of LLM experiments.
- Resolve the PSR hyperparameter inconsistency between Sec. 4 and Algorithm 2.
- Expand the long-horizon experiments beyond a single 1.3B case if resources permit.
- Include direct sensitivity analyses for the RMS rescaling constant and for `r` / shrinkage strength on larger models.
- Add end-to-end efficiency plots against wall-clock and total optimizer FLOPs, not just per-layer orthogonalization timings.

# Actual Human Scores
Individual reviewer scores: [4.0, 2.0, 4.0, 4.0, 2.0]
Average score: 3.2
Binary outcome: Reject
