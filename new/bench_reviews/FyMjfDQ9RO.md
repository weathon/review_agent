Now I have enough calibration data. Let me synthesize the final review.

## Summary

Sylber proposes a self-supervised learning framework that produces speech representations with salient syllabic structure through "self-segmentation distillation" — iteratively refining pseudo-labels from an initial model (SDHuBERT) to create frame-wise embeddings that are flat within syllable spans and distinct between them. This structured representation enables a linear-time greedy segmentation algorithm and produces syllable-rate tokens (4.27 Tok/s) that support intelligible speech reconstruction and competitive spoken language modeling at significantly lower bitrates than prior SSL-based tokenizers.

## Strengths

- **Clear empirical improvements in syllable detection/discovery.** Table 1 shows consistent gains over prior methods (SDHuBERT, VGHuBERT, Komatsu & Shinozaki) across most metrics, with a notable improvement in syllable purity (59.4→64.0). The linear-time greedy algorithm achieves this without the O(n²) cost of prior methods, and Table 1's Sylber-Greedy vs. SDHuBERT-Greedy ablation cleanly demonstrates that the learned structure enables efficient segmentation.

- **Strong cross-domain and cross-lingual generalization.** Table 2 demonstrates that syllable detection generalizes to noisy conversational speech (Fisher), Spanish (MLS), and Mandarin (AISHELL-3) without any fine-tuning, with F1 scores within 1 point of in-domain English. This is an impressive and practically valuable result.

- **Substantial coding efficiency gains.** Table 4 shows Sylber tokens achieve 4.27 Tok/s and best coding-rate across all vocabulary sizes, with ~20% improvement over SDHuBERT tokens. The token rate reduction from 25-30 Hz (HuBERT) to ~4 Hz is significant for scaling spoken language models.

- **Well-matched uLM experiments.** Table 6 carefully controls for model size (125M) and training data, showing Sylber-uLMs match or outperform SDHuBERT-uLMs and GSLM at much lower bitrates. The sBLIMP score of 60.54 with 125M parameters on 66K hours is noteworthy, exceeding TWIST-13B (59.20).

- **Linguistically motivated categorical perception analysis.** The Discriminability Index experiment (Table 7, Figure 3) is an original contribution connecting speech SSL to psycholinguistic theory, and Sylber's lowest DI (0.112) provides a principled explanation for why its tokenization is efficient.

## Weaknesses

### Fatal
None.

### Major

- **The "syllabic" labeling of discovered units is stronger than the evidence warrants.** The paper's headline claim is that representations are "syllabic" and that this is "the first demonstration of validity and effectiveness of speech tokenization at the syllable level." While Table 1 shows good alignment with syllable boundaries (F1=72.2), and the ~4.27 Tok/s rate matches English syllable production rates, the evidence does not establish that tokens correspond to linguistically defined syllables (e.g., adhering to CV/CVC templates, preserving onset/nucleus/coda internal structure). The discovered units could plausibly be coarser-than-phoneme segments that happen to overlap with syllables in rate and boundary location. The cross-lingual generalization (Table 2) supports phonological structure, but boundary detection alone does not confirm syllable identity. The paper would be substantially strengthened by analysis of token-phoneme alignment (e.g., what fraction of tokens correspond to canonical syllables vs. other segment types). As currently presented, "robust segmental units at ~4-5 Hz with good phonological content and syllable-aligned boundaries" is better supported than "syllabic embedding representation."

- **Insufficient ablation isolating the contribution of self-segmentation distillation.** Sylber changes multiple factors simultaneously relative to SDHuBERT: a distilled regression loss to segment means, a reduced 9-layer architecture (initialized from SDHuBERT's 9th layer), a denoising objective, and a two-stage training procedure. The paper does not include an ablation where SDHuBERT is retrained with the same 9-layer architecture + denoising but without distillation, or where the distillation loss is applied with random/poor initial segments. The algorithmic ablations (Sylber-Greedy vs. SDHuBERT-Greedy, Sylber-MinCut) demonstrate the quality of learned features but do not isolate *why* they improve. This weakens the mechanistic claim that self-segmentation distillation is the key innovation, leaving open the possibility that the gains come from a better-tuned architecture or temporal smoothing inherent in the segment-mean regression.

- **Quantization causes notable intelligibility degradation, tempering coding efficiency claims.** While Sylber's continuous features achieve WER 4.88 (Table 3), the quantized 20K model reaches WER 7.95 — a 63% relative increase. HuBERT 2K achieves WER 5.04 with higher Tok/s but better intelligibility. The coding efficiency advantage (Table 4) is calculated on the quantized models, so this WER gap matters. The paper acknowledges that "the embedding space of Sylber is readily quantized" based on the DI analysis, but the quantization gap contradicts this: k-means loses substantial speaker and pitch information (pitch correlation drops from 0.918→0.774), suggesting the embedding space is less "ready" for quantization than claimed. This does not invalidate the contribution (4.27 Tok/s is genuinely efficient), but the efficiency-intelligibility trade-off needs more honest discussion.

### Minor

- **The causal claim about "emergent" categorical perception should be tempered.** Section 6 states that "self-segmentation distillation might be a natural learning algorithm that resembles human language learning." However, the MSE regression loss to segment-averaged targets inherently flattens within-segment variation, which would mechanically produce sharper interpolation boundaries regardless of whether the model has learned "categorical" representations. The DI improvement over SDHuBERT (0.112 vs. 0.131) is modest, and the analysis is conducted on continuous embeddings, not discrete tokens, so the connection to tokenization efficiency is indirect. This is an interesting analysis but the causal attribution is overstated.

- **Spoken language modeling evaluation is limited to proxy metrics.** The sWUGGY and sBLIMP tasks evaluate lexical and syntactic discrimination via likelihood, not actual speech generation quality. As the paper acknowledges, Sylber degraded on SUPERB tasks; sWUGGY/sBLIMP alone do not establish that syllabic tokens are suitable for full spoken language modeling.

- **Cross-lingual generalization is shown only for boundary detection, not for tokenization or resynthesis.** Table 2 demonstrates that Sylber's syllable detection transfers to Spanish and Mandarin, but no resynthesis or uLM experiments are conducted on these languages. This leaves open whether "syllabic tokens" generalize across languages or only the boundary detection aspect.

- **The silence token addition partially undermines the "purely syllabic" framing.** The Sylber-w/SIL variant (Table 6) inserts explicit silence tokens when inter-token gaps exceed 140ms, improving sWUGGY from 76.31→78.03. This shows that syllabic tokens alone are insufficient for optimal language modeling — the representation benefits from additional structural tokens.

### Trivial
- The term "syllabic embedding representation" (Sylber) implies stronger linguistic grounding than the evidence supports; a more neutral name might better reflect the contribution.

## Nice-to-Haves

- Comparison against modern neural audio codecs (EnCodec, DAC) on bitrate/intelligibility trade-offs would better contextualize the coding efficiency results, though this falls outside the paper's primary focus on SSL-based tokenization for language modeling.
- An ablation isolating the distillation loss from architecture/hyperparameter changes would clarify the contribution of self-segmentation distillation specifically.
- Analysis of token-phoneme correspondences (e.g., what fraction map to canonical syllable types) would directly address whether the units are linguistically syllabic.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Coding efficiency comparison is structurally biased because BPE is not optimal for speech coding"**: The paper is transparent about using BPE for HuBERT compression and cites prior work (Shen et al., 2024) that established this approach. Both methods are compared using their standard tokenization pipelines. The comparison is not biased in favor of Sylber — if anything, it uses the established compression method for HuBERT. Removed as unfair criticism.

- **"Missing comparison to concurrent/concurrent syllable-level methods"**: Per hard rules, I should not flag missing related works. Additionally, concurrent work may not have been available at submission time. Removed.

- **"SPARC dependency could bias results"**: All models in Table 3 use the same SPARC-based resynthesis pipeline, making the comparison fair. Removed as unfairly singling out one component of a controlled experiment.

- **"Reproducibility concerns about hyperparameters/thresholds"**: The greedy algorithm's thresholds are standard design choices, and the paper shows robust generalization across domains/languages. Removed as minor implementation detail per hard rules.

- **"Reporting standard deviation across seeds"**: Single-run evaluation is standard practice in SSL speech research. Removed as a generic weakness per soft rules.

- **"Demanding SUPERB results in main paper"**: The authors explicitly acknowledge this limitation and relegate it to the appendix with clear discussion. Removed as the authors already address this.

- **"The quadratic attention cost motivation is overstated since many works process 20-30 Hz sequences"**: The paper's motivation about efficiency is valid regardless of whether current systems technically can process such sequences; Sylber provides meaningful headroom. Removed as an overstatement of a minor framing concern.

## Novel Insights

The connection between the MSE regression-to-segment-means loss and categorical perception effects is an underexplored insight. The self-segmentation distillation framework essentially performs within-segment temporal averaging as a training signal, which naturally creates sharp inter-segment boundaries. Whether this constitutes genuine "categorical perception" in the psycholinguistic sense or is an artifact of the loss function deserves further investigation, but the observation that temporal structure induction also creates more discrete embedding spaces is a valuable finding that bridges SSL engineering with linguistic theory.

## Suggestions

- **Rename/recalibrate the "syllabic" claim.** Either provide direct evidence that tokens correspond to linguistic syllables (phoneme alignment, syllable-type distribution analysis), or reframe the contribution as "segmental units at syllabic rate with syllable-aligned boundaries and phonological content."
- **Add a controlled ablation.** Train a 9-layer SDHuBERT (same architecture as Sylber) with denoising but without distillation, and/or apply the distillation loss with random segments. This would isolate the contribution of the proposed training framework.
- **Discuss the quantization gap more honestly.** Acknowledge that while the continuous embedding space is efficient, current k-means quantization loses significant pitch/prosodic information, and characterize what types of content are most affected.

## Score and Decision

Calibration against similar papers:
- **SyllableLM** (accepted poster, scores 6,6,6,8): Directly comparable in topic (syllable-level speech units). Sylber has stronger evaluation (resynthesis, coding efficiency, cross-lingual generalization, categorical perception) but shares similar overclaiming issues about "syllabic" and "semantic" labels. Sylber's algorithmic contribution (O(n) segmentation) and empirical thoroughness are somewhat stronger.
- **DC-Spin** (rejected, scores 5,1,8,5): Weak comparison due to high WER (~18%) and limited novelty. Sylber is clearly superior in methodology and results.
- **RepCodec** (rejected, scores 6,5,5,6): Incremental contribution (k-means replacement). Sylber has a more novel framework and broader evaluation.
- **MR-HuBERT** (spotlight, scores 8,8,8,8): Stronger contribution with comprehensive evaluation and clear improvements. Sylber is not at this level due to overclaiming and incomplete ablation.
- **dMel** (rejected, scores 6,5,3,5,8): Simple discretization approach with limited novelty. Sylber has substantially more technical depth.

Sylber is a solid paper with real contributions (linear-time segmentation, efficient tokenization, cross-lingual generalization, categorical perception analysis) but with overclaimed "syllabic" interpretation and insufficient ablation. It sits above SyllableLM in technical depth but shares its weaknesses in overclaiming. I place it at 6.5 — a clear accept at the poster level but below spotlight due to the overclaiming and ablation gaps.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>