Now I have a solid understanding of the landscape. Let me synthesize the review.

Key calibration anchors:
- SVD-LLM (accepted, poster): scores 8, 5, 6, 6 (avg ~6.25). Most similar method to AdaSVD.
- ASVD (rejected): scores 6, 8, 6, 5 (avg ~6.25) but rejected
- Dobi-SVD (accepted, poster): scores 8, 5, 6, 6, 6 (avg ~6.2)
- AutoTrunc (withdrawn/rejected): scores 6, 3, 3 (avg 4)
- LLRC (rejected): scores 3, 5, 5, 5 (avg 4.5)

Now let me carefully analyze the AdaSVD paper and synthesize the review.

Key things to verify from the harsh critic:

1. **Compression ratio definition ambiguity**: The paper uses "compression ratio" and "retention ratio" somewhat interchangeably. Looking at the paper: Table 1 uses "RATIO 40%, 50%, 60%", and the text says "truncating 60% or more parameters." The mrr and trr in Eq. (19) are explicitly "minimum retention ratio" and "target retention ratio." The RATIO in tables appears to mean compression (fraction removed), while the internal formulas use retention. This is potentially confusing but not fundamentally flawed since they define their terms in Section 3.2. The mapping is clear: at RATIO 60%, the target retention is 40% (keeping 40% of parameters). The paper could be clearer, but this is a presentation issue rather than a structural problem. The actual parameter counts after SVD are well-defined through Eq. (20).

2. **Theoretical vs implemented adaComp**: This is a legitimate concern. The derivation involves switching between closed-form updates (which are unstable) to LSE with Moore-Penrose pseudoinverse, with some skipped steps. The alternating update is heuristic—no convergence guarantees are provided. The empirical evidence (Figure 3c, Table 3c) shows it works but also shows overfitting at high iterations. This is a real but moderate weakness.

3. **adaCR importance metric**: Using cosine similarity as importance is simplistic and weakly justified. This is a valid criticism.

4. **No variance/statistical significance**: Valid concern but common in the field.

Let me now verify each specific claim of the harsh reviewer against the paper.

For the "Harsh Critic" point about compression ratio being "not rigorously specified" - checking the paper again: Eq. (20) clearly defines CR(W_i) as a retention ratio. The text describes "compression ratios" like 40%, 50%, 60%. From context (Table 1), RATIO of 60% means compressing 60% of parameters (keeping 40%). This is slightly confusing but not misleading in a way that undermines the claims. The baselines use the same convention. So this is a clarity concern, not a fundamental flaw.

The Harsh Critic also says "there is no explicit accounting of how many parameters are actually stored per layer after SVD" — this can be computed from the given compression ratio since SVD factorization clearly defines the stored matrices. Whitening matrices could add overhead but this is standard for all compared methods.

For the claim about "no inference latency/throughput or memory footprint measurements" — this is a valid gap. The paper claims to reduce "memory requirements" and enable "deployment on resource-constrained devices" but doesn't measure actual memory or speedup. SVD-LLM (the main baseline) was accepted at ICLR despite some reviewers noting this gap, so this is a moderate but not fatal weakness.

For EoRA comparison — the paper cites EoRA (Liu et al., 2024) in passing in Section 3.1 ("our adaptive compensation can be integrated with data whitening proposed by Wang et al. (2025) and Liu et al. (2024)"). EoRA is a concurrent/similar approach focused on eigenspace low-rank approximation. Not comparing with it is a valid concern but EoRA is not an SVD-based method per se; it's a compensation method for quantization. The comparison isn't as direct as the Spark reviewer suggests.

Let me verify the VLM evaluation claim - Figure 5 shows only qualitative captioning examples on COCO. No BLEU/CIDEr metrics are reported. This is genuinely weak.

Now regarding model scale — the paper only tests 7B-class models. SVD-LLM also initially tested similar scales and was still accepted. This is a minor weakness.

Calibration summary:
- SVD-LLM (accepted, avg ~6.25): Similar method, had issues with fairness of comparisons (LoRA finetuning) and lack of latest model evaluation but was still accepted
- AdaSVD is incrementally improving over SVD-LLM with two reasonable contributions
- But has weaknesses in justification of adaCR, limited theoretical grounding of adaComp, no hardware metrics, limited VLM evaluation

I'll give this a score around 5-6, leaning toward 5.5 since the empirical results are genuinely strong but the contributions have some gaps in justification.

Actually, looking more carefully:
- The adaComp idea (alternating pseudoinverse updates) is essentially Alternating Least Squares applied to the SVD truncation correction problem. This is well-known in the matrix factorization literature. The novelty is in applying it to LLM weight matrices after SVD truncation, which is incremental but still a contribution.
- The adaCR idea (layer-wise adaptive compression based on cosine similarity) is quite simple and the justification is thin.
- The empirical results are solid wins over SVD-LLM, especially at high compression ratios.
- The lack of actual memory/speed measurements and limited VLM evaluation are notable gaps.
- The theoretical grounding of adaComp is incomplete (no convergence analysis, no comparison to ALS literature).

Given the calibration papers (SVD-LLM ~6.25 poster acceptance, AutoTrunc ~4 rejected, LLRC ~4.5 rejected, Dobi-SVD ~6.2 poster), this paper is roughly in the 5-6 range. It's incrementally better than SVD-LLM, which is meaningful progress, and the empirical results are comprehensive and convincing. But the theoretical contribution is modest (adaComp is basically ALS for this setting) and adaCR is weakly motivated.</think>## Summary

AdaSVD proposes two complementary techniques for SVD-based LLM compression: **adaComp**, which compensates for SVD truncation error by alternately updating singular matrices U and V⊤ via Moore-Penrose pseudoinverse Least Squares Estimation, and **adaCR**, which assigns layer-specific compression ratios based on cosine similarity between layer inputs and outputs. Experiments across LLaMA2-7B, OPT-6.7B, Mistral-7B, and Vicuna-7B show consistent improvements over SVD-LLM, with especially large gains at high compression ratios (e.g., 44% PPL reduction on WikiText-2 at 60% compression).

## Strengths

- **Strong and consistent empirical improvements over SOTA SVD methods.** AdaSVD consistently outperforms SVD-LLM across 4 model families and 8 datasets (Table 1, 2), with the largest gains at high compression ratios where prior methods severely degrade (e.g., WikiText-2 PPL from 89.90 → 50.33 at 60% on LLaMA2-7B, a 44% relative improvement). The gains on downstream QA benchmarks are moderate but consistent in average accuracy. Table 4 further shows orthogonal combinability with GPTQ-INT4 quantization. These are non-trivial improvements over a strong baseline.

- **Two well-motivated and orthogonal ideas.** Post-truncation compensation (adaComp) and layer-wise adaptive compression ratios (adaCR) address distinct, real limitations of existing SVD-based methods. The ablation study (Table 3) cleanly validates that both components contribute independently, and the stack-of-batch strategy is a practical engineering solution for GPU memory constraints during calibration.

- **Broad evaluation across model families and compression ratios.** The paper tests 4 LLM families, 8 datasets, and 5 compression ratios (40–80%), plus a VLM (LLaVA). The ablation studies (Table 3a–d) systematically test adaComp, adaCR, iteration count, and minimum retention ratio.

## Weaknesses

### Major:

- **Weak justification for adaCR's importance metric.** The cosine similarity between layer input X and output WX is used as the sole measure of layer importance (Eq. 17–18), but the paper provides only one sentence of motivation: "The importance of W can be measured by its impact on the input." This metric inherently favors layers that act as near-identity maps (high input-output similarity), which may not correlate with layers that are crucial for downstream task performance. No comparison with alternative importance measures (e.g., gradient-based, Fisher information, activation variance, or even a simple heuristic like assigning higher retention to first/last layers) is provided. Since adaCR is one of the two named contributions, the thin justification is a meaningful gap. Table 3d shows that the minimum retention ratio (mrr) is a sensitive hyperparameter that requires tuning, and Table 3b shows adaCR's gains are modest at 40–50% compression, further suggesting the importance metric may not be robust.

- **Incomplete theoretical grounding for adaComp's convergence and optimality.** The adaComp procedure is essentially Alternating Least Squares (ALS) applied to the weighted reconstruction objective ∥U_k^σ(V_k^σ)⊤X − WX∥²_F. While reformulating as an LSE problem solved via Moore-Penrose pseudoinverse (to avoid instability of direct matrix inversion) is sensible, the paper (1) provides no convergence guarantees, (2) does not discuss whether alternating pseudoinverse updates converge to a local minimizer of the joint objective, (3) skips derivation details for the V_k update (dropping the dependence on X without justification in Eq. 13), and (4) the empirical evidence in Table 3c shows that more iterations can *hurt* performance (e.g., PPL rises from 14.76 → 15.84 at 40% compression going from 1→15 iterations), suggesting overfitting rather than convergence. The relationship between ALS convergence and the overfitting behavior is not analyzed. This is not fatal since the method demonstrably works, but it limits the claim that adaComp "effectively and stably minimizes compression error."

- **No actual memory footprint, latency, or throughput measurements.** The paper is motivated by deployment on "resource-constrained devices" and claims SVD "effectively accelerates model inference by reducing memory requirements" (Section 1), yet provides zero measurements of inference latency, throughput, or peak GPU memory. While SVD compression mathematically reduces parameter count, the actual speedup depends on hardware, kernel support, and whether the low-rank factorization can be efficiently executed. SVD-LLM (the primary baseline) demonstrated inference speedup; the absence of such evidence here is a notable gap for a compression paper with deployment motivations.

### Minor:

- **VLM evaluation is qualitative only.** Figure 5 shows image captioning examples on LLaVA-7B/COCO but reports no quantitative captioning metrics (CIDEr, BLEU, etc.). The claim of effectiveness on VLMs is therefore weakly evidenced.

- **Overfitting sensitivity of adaComp provides no principled iteration selection.** Table 3c shows more iterations help at high compression but hurt at low compression, attributed to "overfitting due to limited calibration data." No validation set, convergence criterion, or automated selection heuristic is proposed, leaving iteration count as a manually-tuned hyperparameter.

- **Calibration data used for both compression and evaluation.** The 256 WikiText-2 calibration samples are drawn from the same corpus used for perplexity evaluation. While this matches prior work (SVD-LLM, ASVD) for comparability, it raises concerns about overfitting to the calibration distribution, especially given the iterative nature of adaComp.

- **The compression/retention ratio terminology is inconsistent.** The paper uses "compression ratio" in Section 1 (referring to fraction removed, e.g., "truncating 60%"), "RATIO" in tables (apparently meaning compression), but uses retention-oriented terms mrr ("minimum retention ratio") and trr ("target retention ratio") internally. While decipherable, this inconsistency could confuse readers about whether reported ratios refer to compression or retention.

## Nice-to-Haves

- Report actual inference latency, throughput, and peak memory on real hardware to substantiate the deployment motivation.
- Provide quantitative VLM evaluation metrics (CIDEr, BLEU) rather than only qualitative examples.
- Evaluate on a larger model (e.g., 13B+) since the introduction explicitly motivates the problem using 70B/66B models.
- Compare adaCR's cosine similarity importance against alternative importance metrics (e.g., gradient norm, activation variance, or simple heuristic schedules) to strengthen the justification for this specific design choice.
- Provide a convergence or validation set-based stopping criterion for adaComp iterations rather than manual tuning.

## Removed Points

- **Claim that compression ratio definition undermines all comparisons.** While the compression/retention terminology is inconsistent, Equation (20) explicitly defines CR(W_i) as the retention ratio of parameters, and all baselines are compared at the same compression level. The actual parameter savings are well-defined by the low-rank factorization. This is a clarity issue, not a structural flaw that invalidates experiments.

- **Missing comparison with EoRA (Spark's point #2).** EoRA (Liu et al., 2024) is cited in the paper and focuses on eigenspace low-rank approximation for *quantization* error compensation, not SVD-based compression. While there is conceptual overlap in the compensation idea, EoRA is not directly an SVD-based compression method and operates in a different setting (quantization error correction). Including it would be informative but is not a baseline gap in the SVD-based compression comparison.

- **Missing comparison with non-SVD compression methods (Human Finder point #6).** The paper's scope is explicitly SVD-based compression methods, which is a coherent and well-defined research area. The paper does demonstrate combinability with GPTQ quantization (Table 4). Requesting comparison with entirely different compression paradigms (pruning, standalone quantization) goes beyond the paper's stated scope.

- **No evaluation on latest models (Human Finder point #1).** Testing on LLaMA2/OPT/Mistral/Vicuna is reasonable for the SVD-based compression community; SVD-LLM (accepted at ICLR 2025) tested similar models. This is a nice-to-have improvement, not a fundamental deficiency.

- **Perplexity as an imperfect evaluation metric (Human Finder point #2).** While PPL has known limitations, it remains the standard metric in this research area (FWSVD, ASVD, SVD-LLM all use it). The paper also reports 5 zero-shot reasoning benchmarks. The absolute PPL values at high compression ratios are indeed high, which the paper acknowledges implicitly—it shows relative improvements in a challenging regime.

- **Demand for variance/statistical significance (Harsh Critic point #4).** Single-run evaluation without standard deviations is the norm in this community (SVD-LLM, ASVD, FWSVD all report single values). This is a standard practice gap, not a unique methodological flaw of this paper.

- **Stack-of-batch distribution change (Neutral Reviewer point #4).** Averaging calibration samples does change the data distribution, but this is a practical trade-off for memory efficiency. The empirical results show it works; analyzing the information loss is a nice-to-have analysis rather than a core deficiency.

- **Computational cost of compression (Neutral Reviewer point #1).** While reporting compression time would be useful, post-training compression methods are typically run once and then deployed. SVD-based methods are already orders of magnitude faster than training-based methods. This is informational but not a critical evaluation gap.

## Novel Insights

The empirical observation that adaComp's value increases with compression ratio (1 iteration suffices at 40%, but more iterations help at 60%+) while potentially overfitting at low compression is an interesting finding that suggests the truncation error landscape differs qualitatively at different compression levels. This connects to the ALS literature where iterative updates are only beneficial when the initial approximation (ordinary SVD truncation) is sufficiently poor. The cosine-similarity importance curve showing a bowl shape for LLaMA-family models (Figure 4)—where early and late layers are more important—aligns with the broader pruning literature (e.g., ShortGPT, LLaMA Pruner) that identifies layer redundancy patterns, though the specific metric used here is novel for SVD allocation.

## Suggestions

- Provide at least one figure or table showing actual parameter counts, inference wall-clock time, and/or peak memory for a representative configuration to ground the deployment motivation.
- Add a principled stopping criterion for adaComp iterations (e.g., monitor validation loss or relative change in reconstruction error between iterations).
- Clarify the compression vs. retention terminology throughout, or add an explicit mapping table.
- For adaCR, include at least one comparison experiment against a simple heuristic (e.g., linear or bowl-shaped retention schedule) to show that the specific cosine-similarity metric provides value beyond just "non-uniform allocation."

## Score and Decision

Calibration papers:
- **SVD-LLM** (ICLR 2025, accepted poster, avg ~6.25): Most directly comparable. SVD-LLM introduced data whitening and sequential low-rank approximation; AdaSVD builds on top of whitening and adds post-truncation compensation and adaptive ratios. SVD-LLM had stronger theoretical grounding (theorems+proofs) but comparable empirical scope. AdaSVD's empirical gains are meaningful.
- **AutoTrunc** (rejected, avg ~4): Also proposes adaptive SVD truncation positions but with weaker novelty and presentation.
- **Dobi-SVD** (accepted poster, avg ~6.2): Differentiable SVD truncation with theoretical analysis. Comparable novelty level.
- **LLRC** (rejected, avg ~4.5): Differentiable rank selection but limited performance and novelty.

AdaSVD has a meaningful empirical contribution (consistent, sometimes dramatic improvements over SVD-LLM) with two reasonable but incrementally novel ideas. The theoretical grounding of adaComp is incomplete, and adaCR is weakly justified. Compared to SVD-LLM (which had theoretical proofs supporting whitening), AdaSVD offers less theoretical novelty but stronger practical results. The absence of hardware metrics and the thin VLM evaluation are notable gaps. Overall, this is a solid but not exceptional contribution—incremental progress on the current SOTA with some limitations in justification.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>