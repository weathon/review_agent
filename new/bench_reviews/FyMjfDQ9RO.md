Now I have all the information I need. Let me write the final consolidated review.

## Summary

Sylber introduces a self-segmentation distillation framework that learns syllabic speech representations from raw audio, producing features with clean piecewise-constant segment structure aligned with syllable boundaries. The method enables a linear-time O(n) segmentation algorithm (replacing prior O(n²) approaches), efficient tokenization at 4.27 tokens/second (6–7× lower than HuBERT-based baselines), and spoken language modeling competitive with much larger models, along with an analysis of categorical perception in the embedding space.

## Strengths

- **Substantial improvement in syllable discovery quality**: Table 1 shows Sylber achieves syllabic purity of 64.0, a large improvement over prior SOTA of 59.4 (Komatsu & Shinozaki, 2024), along with the best F1 (72.2) and R-value (75.9). This is a meaningful margin in a metric where prior methods had plateaued in the 53–59 range.

- **Linear-time segmentation enabled by clean feature structure**: The O(n) greedy algorithm matches the O(n²) MinCut algorithm on Sylber features (F1: 72.2 vs 72.2 in Table 1), while applying the same greedy algorithm to SDHuBERT causes major degradation (F1 drops from 67.5 to 61.2). This demonstrates that the self-segmentation distillation produces representations clean enough to obviate expensive post-hoc optimization — a genuine and practical contribution.

- **Competitive coding efficiency with intelligible reconstruction**: Tables 3–4 show Sylber operates at 4.27 Tok/s vs. 23.59+ Tok/s for HuBERT units while maintaining reasonable intelligibility (WER=7.95 at 20K vs. 5.04 for HuBERT-2K). The coding-rate metric confirms superior information-per-bit efficiency (0.0315 vs. 0.0283 for HB50-BPE at 5K vocab).

- **Cross-lingual generalization without tuning**: Table 2 shows syllable detection generalizes to conversational English (F1=71.9), Spanish (F1=71.7), and Mandarin (F1=71.3), nearly matching in-domain performance (F1=72.2), despite training only on English audiobooks.

- **Spoken language modeling at drastically lower bitrate**: Table 6 shows Sylber-uLM (125M, 1K hrs) outperforms GSLM at sBLIMP (58.04 vs 57.06) while using ~3.4× lower bitrate. At 66K hours, Sylber-uLM achieves sBLIMP of 60.78, exceeding TWIST (13B, 150K hrs) at 59.20 — a notable result given the 100× parameter gap.

- **Honest limitations discussion**: Section 7 explicitly acknowledges degradation on SUPERB tasks and positions Sylber as a coding/tokenization framework rather than a universal speech representation, appropriately scoping its claims.

## Weaknesses

### Fatal
None.

### Major

- **The "emergent categorical perception" claim (Section 6) is overclaimed.** The paper states "our loss objective does not involve any categorical learning at all" (line 277) and frames categorical perception as "unexpected." However, the self-segmentation distillation loss explicitly regresses all frames within a segment toward the same target (the segment's average embedding), producing a piecewise-constant embedding structure — flat within segments, discontinuous across boundaries. When interpolating between two syllables belonging to different segments, sharp transitions are a direct geometric consequence of this training objective, not an emergent linguistic discovery. The Discriminability Index is essentially measuring how well the model implements the very structure it was trained to produce. Comparing Sylber's DI (0.112) to SDHuBERT's (0.131) is more informative than comparing to non-segmental models like HuBERT (0.141), since SDHuBERT also has a segmental inductive bias and shows only modestly worse DI. The comparison to models without segmental training objectives sets up a straw man. This matters because the "emergent categorical perception" is presented as a headline contribution (bullet 6 in the contribution list, and a full Section 6), but it is better described as a structural property of the representation that follows from the training design.

- **The "minimal information loss" framing in the abstract overstates what the evidence supports.** The abstract claims "a compact sequence of tokens with minimal information loss," but Sylber 20K shows WER 7.95 vs. HuBERT 2K's 5.04 (Table 3) — nearly 3% absolute degradation. More significantly, pitch correlation drops sharply under quantization (0.918 → 0.774, Table 3), indicating systematic loss of prosodic information. While the paper acknowledges this in the body (Section 5.2, line 205-208), the abstract's "minimal information loss" characterization misrepresents the tradeoff. The coding-rate metric partially accounts for this but is an ad hoc combination of WER and bitrate that weights all bits equally regardless of what linguistic information they carry, potentially favoring models that discard prosodic information. This matters because the paper's central value proposition is efficient coding with reasonable fidelity, and the boundary between "minimal" and "meaningful" loss is precisely where the contribution's significance should be judged.

- **Cross-linguistic evaluation in Table 2 lacks ground-truth specification.** Table 2 reports syllable detection F1 for Spanish (MLS) and Mandarin (AISHELL-3), but how ground-truth syllable boundaries are defined for these languages is not described in the main text. Syllable boundaries are language-specific — Spanish and Mandarin have very different syllabification rules from English. Without knowing what constitutes "ground truth" in these corpora, the generalization claim "without any tuning" is difficult to interpret. Details are referenced to Appendix A.2.5 (stripped from this version), but for a headline contribution (contribution bullet 4), the main text should specify this.

### Minor

- **No ablation isolating the distillation objective from continued training.** Sylber is initialized from SDHuBERT and trained for additional stages. Without comparing against continued SDHuBERT training with a standard (non-distillation) objective from the same initialization, it is unclear whether the improvement comes specifically from the distillation loss or simply from additional training epochs. The paper notes the denoising is "not a primary source of learning" (Section 3.1) but does not isolate the distillation contribution.

- **The coding-rate metric (Section 4.2) is ad hoc.** It is defined as (1-WER/100) × #words / #bits, combining intelligibility and bitrate into a single scalar. This weights each bit equally regardless of linguistic content type, which may systematically favor models that discard prosodic information (as Sylber does — see pitch correlation drops in Table 3). The metric is useful but its design choices should be more critically examined.

- **The sBLIMP comparison with TWIST (Section 5.3) somewhat overclaims.** Sylber-uLM (125M, 66K hrs) achieves sBLIMP 60.78 vs. TWIST (13B, 150K hrs) at 59.20. The paper calls this "astonishing given the huge gap," but sBLIMP is a relatively narrow probe of syntax, while TWIST substantially outperforms on sWUGGY (84.10 vs. 76.31), which captures lexical coverage. Framing the sBLIMP edge as the primary finding while downplaying the sWUGGY gap gives an incomplete picture.

- **The categorical perception stimuli are manually adjusted to place the perceptual boundary at α=0.5.** The paper states (line 271) that endpoints are "manually adjusted to make the perceptual boundary drawn approximately in the middle." This means the stimuli are designed to have a categorical boundary at the midpoint, and DI then measures how sharply the model transitions at that same midpoint. While DI measures transition *sharpness* not just *location*, the manual adjustment reduces the generality of the probe.

### Trivial
None.

## Nice-to-Haves

- An intelligibility-matched bitrate comparison (comparing Sylber and HuBERT at approximately equal WER) would more directly quantify the bitrate advantage and strengthen the efficiency claim.

- Analysis of how the self-segmentation distillation converges across iterations beyond the two-stage design would validate the bootstrapping framework.

- Cross-linguistic evaluation with explicitly defined syllable boundary annotations (or using language-specific evaluation protocols).

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh Critic Claim 1 (full tautology version)**: The harsh critic's strongest version of the categorical perception argument — that the entire Section 6 is "undermined" because it is "tautological" — is overstated. While the piecewise-constant training objective does create an inductive bias toward categorical structure, the DI metric measures the *sharpness* of category boundaries across a continuous interpolation, not just within-segment constancy. SDHuBERT has a related but weaker segmental objective and shows DI of 0.131 vs. Sylber's 0.112, suggesting the distillation produces genuinely stronger categorical effects. The finding is not purely tautological but is overclaimed as "unexpected" emergence.

- **Harsh Critic: SDHuBERT's higher recall indicates oversegmentation without evidence.** The paper argues (line 162) that recall and cluster purity "can be inflated by having more segments." While direct evidence (e.g., segment count comparisons) is not provided, the argument is standard and plausible — oversegmentation typically increases recall at the cost of precision, which is exactly what Table 1 shows. This is a minor point, not a major flaw.

- **Harsh Critic: Pitch correlation drop invalidates "minimal information loss."** The paper actually discusses this explicitly in Section 5.2 and acknowledges that quantization flattens pitch. The critic treats this as a devastating flaw, but the paper positions Sylber as a coding framework (Section 7) and the prosodic loss is an inherent tradeoff of syllabic tokenization. The issue is with abstract framing, not with hidden information.

- **Strength Finder: "Self-segmentation distillation as a principled bootstrapping framework."** This is listed as a supporting strength, but the bootstrapping is limited to two stages with no analysis of convergence or additional iterations, which limits how "principled" it can be called. Moved to NTH.

- **Strength Finder: "Novel evaluation of categorical perception via articulatory interpolation"**: The methodology is creative, but since the categorical perception finding is itself overclaimed (see Major weakness 1), the metric's novelty is somewhat undermined. The DI metric definition is deferred to the appendix. Kept as minor supporting contribution.

- **Harsh Critic: Missing ablation of distillation objective.** This is real but minor — it would strengthen but not invalidate the paper, so moved to Minor rather than Major.

- **Harsh Critic: uLM comparison mixes models with different training data/parameters.** The paper clearly separates these comparisons (top vs. bottom section of Table 6 with different data sizes) and acknowledges the resource gap. This is not a flaw but a feature of showing scaling behavior.

## Novel Insights

The most interesting observation emerging from the intersection of the paper and reviews is that there appears to be a meaningful distinction between "emergent" and "designed-in" categorical structure that the paper elides. Sylber's piecewise-constant embeddings create an inductive bias toward categorical boundaries, but the *degree* of categorical effect (DI 0.112 vs. 0.131 for SDHuBERT) suggests that the specific distillation mechanism matters beyond just having a segmental objective. This raises a genuinely interesting question: does self-segmentation distillation approximate something linguistically meaningful about how syllabic categories organize the speech code, even if the categorical structure is a designed-in inductive bias rather than an emergent phenomenon? Acknowledging this distinction — rather than claiming surprise — would actually strengthen the paper's theoretical contribution.

## Suggestions

- Reframe Section 6: Instead of claiming categorical perception is "unexpected" or "emergent," acknowledge it as a structural property that follows from the training design and present the DI comparison with SDHuBERT as evidence for how *strongly* the distillation induces categorical structure. This is more honest and equally interesting.

- Qualify the "minimal information loss" claim in the abstract to "low bitrate with competitive intelligibility" or similar, and note the prosodic information loss as a known tradeoff.

- Specify in the main text how syllable ground truth is defined for the MLS (Spanish) and AISHELL-3 (Mandarin) evaluations in Table 2, even if briefly.

## Score and Decision

**Calibration anchors used:**

1. **Multi-resolution HuBERT** (avg score 8.0, spotlight): Strong speech SSL paper with comprehensive experiments and clear ablations, no overclaims. Sylber is narrower in scope (syllabic coding vs. general SSL) and has overclaim issues that MR-HuBERT does not.

2. **SpeechTokenizer** (avg score 5.75, poster): Unified speech tokenizer with RVQ, reviewers question necessity and overclaims. Sylber has stronger empirical evidence for its core contribution but similar overclaim issues around "emergence" and efficiency framing.

3. **WavTokenizer** (avg score 6.5, poster): Efficient acoustic codec tokenizer, extreme compression with competitive reconstruction. Most similar in spirit to Sylber (efficient tokenization). Reviewers noted overclaims about semantic information but accepted the core compression contribution.

4. **DC-Spin** (avg score 4.75, reject): Speaker-invariant speech tokenizer with small margins vs. baselines and overclaimed speaker-invariance. Sylber has larger performance gaps and more contributions but also more overclaim issues.

5. **JOOCI** (avg score 3.5, reject): Speech representation learning with limited baselines. Sylber has much stronger empirical contribution.

6. **Overlapped chunk segmentation** (avg score 2.5, reject): Presenting well-known facts as novel. Sylber clearly does not fall into this category.

Sylber sits between DC-Spin (4.75) and SpeechTokenizer (5.75). Like WavTokenizer, it has a genuine and substantial core contribution (efficient syllabic tokenization with strong empirical evidence) but overclaims on some headline findings. The categorical perception overclaim is the most significant issue. The core technical contribution — self-segmentation distillation enabling clean segment structure and linear-time segmentation — is solid and well-demonstrated. The paper would be clearly stronger with honest reframing, but the underlying methodological contribution is genuine.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>