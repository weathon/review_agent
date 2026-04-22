Now let me consolidate everything into the final review.

## Summary

The paper introduces TAAE, a ~950M parameter transformer-based speech codec that uses Finite Scalar Quantization (FSQ) instead of the conventional Residual Vector Quantization (RVQ), achieving state-of-the-art speech quality at ultra-low bitrates (400 and 700 bps). A key novel contribution is a post-hoc residual decomposition method that allows a single FSQ-trained model to be flexibly configured into residual token hierarchies without retraining, with a mathematical guarantee that all resulting quantization levels remain within the training distribution.

## Strengths

- **Strong objective results at ultra-low bitrates:** TAAE at 400 bps achieves PESQ 2.96 and STOI 0.90, outperforming baselines operating at 2–3× the bitrate (e.g., Mimi at 1100 bps: PESQ 3.01, Encodec at 1500 bps: PESQ 2.36) as shown in Table 2. This is a meaningful compression-quality result regardless of parameter count.

- **Novel post-hoc FSQ residual decomposition (Sec. 3.2.1, Eqs. 3–4):** This is a genuinely clever contribution. The constraint L = 2^n + 1 (Table 1) and the Minkowski sum analysis provide a principled foundation for decomposing a single FSQ bottleneck into hierarchical residual tokens post-training, offering bitrate flexibility that RVQ-based codecs lack by design.

- **Multi-configuration deployment from a single trained model (Sec. 3.2.1, Table 2):** The same model can operate at 400 bps (single token), 700 bps (residual two tokens), or with continuous latents for diffusion-based generation, without retraining. This is a practical advantage over RVQ-based codecs.

- **Scaling behavior validated (Sec. 4.6, Appendix A.2):** Architectures at 250M, 500M, and ~1B parameters confirm that the transformer-based codec consistently improves with scale.

- **Causal variant with minimal degradation (Appendix A.4):** A streaming-compatible version outperforms Mimi (a dedicated streaming codec) on objective metrics, despite using significantly less training data, suggesting practical viability.

- **Generalization to unseen languages (Appendix A.5) and robust inference speed (Appendix A.9):** Despite 950M parameters, TAAE remains competitive in real-time factor with much smaller baselines.

## Weaknesses

### Fatal
None.

### Major

- **No FSQ-vs-RVQ ablation within the same architecture:** The paper explicitly motivates FSQ as addressing "inherent problems of VQ and RVQ quantization" (Sec. 1, Sec. 3.2), citing codebook utilization issues and hierarchical token stream complications. While Sec. 3.2.1 provides an elegant theoretical argument for FSQ's post-hoc flexibility, there is no direct experimental comparison of FSQ versus RVQ within the TAAE architecture at comparable bitrates. This gap means the paper cannot experimentally confirm that FSQ is superior to (or even competitive with) RVQ in this application, despite being positioned as a core contribution. The Appendix A.8 codebook utilization analysis shows FSQ has near-optimal utilization, but this does not establish the comparison against RVQ-based TAAE.

- **Conflation of scale and architecture in SOTA claims:** The paper's central claim—"the potential of scaling transformer-based codec architectures" (Sec. 4.5)—is primarily demonstrated by comparing a 950M-parameter model against models that are 10–100× smaller (Encodec ~15M, DAC ~80M). While the paper acknowledges this disparity (Sec. 4.4, "Parameter counts also vary widely") and includes scaling experiments (Appendix A.2 showing consistent improvement from 250M→1B), it provides no evidence that transformers specifically enable these gains rather than that any comparably-sized architecture would achieve similar results. The scaling experiments only show that TAAE scales within its own architecture family, not that it scales better than alternatives.

### Minor

- **MUSHRA evaluation lacks statistical rigor (Sec. 4.3, Fig. 2):** The subjective evaluation uses 24 participants recruited via public forums without screening for hearing ability, only 25 audio samples, and reports no confidence intervals or significance tests. While the apparent effect size (TAAE ~85 vs. next-best ~55 at comparable bitrates) is large enough that statistical significance seems plausible, this cannot be confirmed from the presented data. The ground truth scoring ~90 rather than 100 is expected behavior in MUSHRA (not a setup flaw).

- **Baselines pre-selected for MUSHRA based on objective metrics (Sec. 4.3):** Selecting which baselines to include in the subjective test based on prior objective performance risks biases, though the authors are transparent about this and the motivation (limiting test length) is reasonable.

- **Training data confound with Mimi (Sec. 4.1 vs Sec. 2.2):** TAAE trains on 105k hours of English-only data, while Mimi trains on 7M hours of multilingual data. This difference cuts both ways (Mimi has more data but must handle more diversity), making fair comparison difficult. The paper does not deeply discuss how this affects interpretability.

- **WavLM finetuning ablation is deferred to Appendix A.1:** Given that the authors state finetuning with the WavLM perceptual loss is "essential in producing intelligible speech" (Sec. 3.4), this critical ablation should appear in the main text rather than the appendix.

### Trivial
None.

## Nice-to-Haves

- An FSQ-vs-RVQ ablation within TAAE at matched bitrate and parameter count would directly validate the paper's core quantization claim.
- A compute-equivalent comparison (training FLOPs and inference FLOPs vs. quality) would help contextualize whether the quality gains are efficient regardless of architecture.
- Reporting confidence intervals for MUSHRA results would strengthen the subjective evaluation claims.

## Removed Points

- **"Ground truth scoring ~90 indicates test setup issues":** Ground truth rarely reaches 100 in MUSHRA tests; ~90 is actually a typical result and does not indicate a problem with the setup. This is a misunderstanding of MUSHRA norms.
- **"The 950M codec is not a small fraction of generative pipelines":** The paper's argument (Sec. 1) is that in pipelines with multi-billion parameter language models, 950M is not prohibitive. This is a reasonable position, not a misleading claim.
- **"SemantiCodec comparison is unfair because it requires BigVGAN for vocoding":** The paper already acknowledges this in Sec. 2.2 and omits SI-SDR for SemantiCodec due to temporal misalignment, handling the confound appropriately.
- **"No justification for why patching-based downsampling is better than progressive downsampling":** This is a design choice, not a claim requiring experimental justification. The paper references ViT-style reasoning and provides analysis in Appendix B.4.
- **"Uniform noise approximation in Eq. 2 is unjustified":** The paper states this is based on Brendel et al. (2024) and demonstrates effectiveness empirically. The approximation is a standard technique, not a novel theoretical claim.
- **"Formatting/presentation nitpicks, appendix placement complaints":** The parser strips appendices; these are not author errors.
- **"Missing related works":** Not evaluated per hard rules.
- **"Demand for larger datasets or more models":** The model zoo and training data (105k hours, 5 baseline codecs, multiple bitrate configurations) are adequate for the paper's stated scope.

## Novel Insights

The post-hoc FSQ residual decomposition is a genuinely novel and well-motivated idea that addresses a practical limitation of codec design: the need to commit to a token hierarchy at training time. By constraining level numbers to L = 2^n + 1 and leveraging the nested structure of quantization points, the paper creates an FSQ bottleneck that can be reconfigured at inference time between single-token, residual multi-token, and continuous latent modes—all from a single trained model. This eliminates a key rigidity of RVQ-based codecs and is independent of the transformer architecture itself, making it potentially applicable to any codec backbone.

## Suggestions

- Run a direct FSQ-vs-RVQ ablation within the same TAAE architecture at matched bitrate—this is the single experiment that would most strengthen the paper.
- Report 95% confidence intervals for MUSHRA scores or at minimum include error bars in Figure 2.
- Consider a brief discussion of training compute (GPU-hours) alongside parameter counts to help readers assess efficiency.

## Score and Decision

**Calibration anchors:**

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| FSQ (Mentzer et al.) | 6.5 | Directly relevant—also about FSQ vs VQ. TAAE builds on FSQ for a different domain (speech codecs) and adds the novel post-hoc decomposition, plus scaling experiments. More applied but with a similar quantization-ablation gap. |
| WavTokenizer | 6.5 | Comparable—also a neural audio codec proposing tokenizer improvements (single VQ vs RVQ) with similar missing ablations. TAAE has stronger empirical results. |
| SpeechTokenizer | 5.75 | Comparable domain (speech codec with RVQ). TAAE has stronger results and a more novel architecture. |
| Multi-resolution HuBERT | 8.0 | High-scoring speech model with thorough ablations—TAAE has weaker ablation coverage (no FSQ-vs-RVQ comparison). |
| Scaling law remote sensing paper | 2.2 | Low-scoring paper with unfair large-vs-small baseline comparisons and overclaimed scaling conclusions. TAAE shares the scale confound concern, but its other contributions (post-hoc FSQ decomposition, flexible multi-configuration deployment) are genuine and the scale concern is acknowledged. |

TAAE makes genuine contributions: the post-hoc FSQ residual decomposition is novel and well-formulated, the empirical results at ultra-low bitrates are strong, and the multi-configuration deployment is practical. However, the lack of an FSQ-vs-RVQ ablation leaves one of the two core claims (that FSQ is preferable to RVQ for this application) experimentally unsupported, and the scale confound leaves the other core claim (that transformers specifically enable these gains) partially weakened. The paper is above the WavTokenizer/FSQ baseline level because its results are stronger and the decomposition idea is more novel than either of those papers' contributions alone, but the ablation gaps prevent it from reaching the level of papers with thorough experimental validation. A score of 6 reflects this balance.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>