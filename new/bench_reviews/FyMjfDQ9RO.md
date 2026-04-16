## Summary
This paper proposes **Sylber**, a self-supervised speech representation model that explicitly sharpens emergent syllabic structure via self-segmentation distillation from SDHuBERT-derived pseudo-segments. The resulting representation supports a simple **linear-time greedy segmentation algorithm**, produces very low-rate syllabic tokens (~4.27 tok/s), and is evaluated on syllable detection/discovery, resynthesis, coding efficiency, spoken-language-model probes, and a novel embedding-discreteness analysis.

## Strengths
- **Strong and clearly supported core contribution: better syllable-structured representations with faster segmentation.** Table 1 is convincing: Sylber improves syllable detection/discovery over prior methods while uniquely enabling an \(O(n)\) greedy segmentation algorithm. The Sylber vs. Sylber-MinCut comparison is especially important evidence that the representation itself has become cleaner, not just the downstream segmentation heuristic.
- **The method is novel but technically coherent.** The self-segmentation distillation objective is well matched to the stated goal: regress each frame toward its segment-level teacher embedding, thereby encouraging piecewise-constant within-syllable structure. This is a meaningful advance over prior work where syllable structure emerged only indirectly.
- **Practical compression gains are real.** The token rate of ~4.27 tok/s is dramatically lower than HuBERT-style frame-level tokens, and Table 4 shows clear gains in token rate, bitrate, and the paper’s task-driven coding-rate metric over the compared baselines.
- **Breadth of evaluation is commendable.** The paper does not stop at segmentation metrics; it also evaluates out-of-domain and cross-lingual transfer, speech resynthesis, coding efficiency, uLM probe tasks, and embedding geometry. This makes the contribution more substantial than a narrow segmentation paper.
- **The out-of-domain/cross-lingual segmentation transfer is interesting.** Table 2 shows that a model trained on English audiobook speech still achieves similar boundary-detection scores on conversational English, Spanish, and Mandarin, which is a nontrivial empirical finding.
- **The paper is refreshingly explicit about scope.** The limitations section clearly states that Sylber is intended more as a speech coding/tokenization framework than as a universal SSL representation, and it acknowledges degradation on some SUPERB tasks.

## Weaknesses

###: Fatal

None.

### Major:
- **The paper overstates some downstream conclusions about efficient spoken language modeling.**  
  The evidence in Table 6 is promising, but it does **not** cleanly isolate tokenization efficiency as the cause of the gains. Comparisons vary tokenizer, vocabulary, training corpus size, parameter count, and in one case silence handling. Thus the results support “Sylber is a promising tokenizer for uLMs at much lower token rates,” but not the stronger abstract/introduction framing that it has already established materially more scalable or more efficient spoken language modeling in a controlled sense. The most defensible claim is about lower-rate tokens with competitive probe performance, not a demonstrated LM-efficiency breakthrough.

- **The abstract’s “minimal information loss” claim is stronger than the resynthesis evidence supports.**  
  Table 3 shows a clear compression–fidelity tradeoff. Quantized Sylber tokens are impressive for their low rate, but they do not preserve information with only minimal loss relative to the stronger baselines: e.g., Sylber 20K has WER 7.95 versus HuBERT 2K at 5.04, and the paper explicitly notes substantial degradation in pitch/prosodic information after quantization (“flattened speech generation”). The continuous \(\infty\) setting is strong evidence that the representation is rich, but it does not fully validate the discrete-token compression claim. This is best framed as **strong compression with acceptable intelligibility**, not minimal-loss coding.

- **The “categorical perception” section supports embedding discreteness under a specific probe, but the paper sometimes interprets it too strongly.**  
  Section 6 provides an interesting synthetic interpolation probe and the DI metric, and Table 7 does support the empirical claim that Sylber embeddings show sharper transitions than baselines under this protocol. However, the paper sometimes slides from that result to a broader human-cognitive interpretation (“categorical perception emerges naturally,” “resembles human language learning”). Given the setup—TTS-generated endpoints, manual adjustment of boundaries toward the midpoint, articulatory interpolation, and a new metric—this is best treated as a **representation-space discreteness analysis**, not strong evidence of human-like categorical perception.

### Minor
- **Sylber is meaningfully narrower than a general-purpose SSL representation.**  
  The paper itself acknowledges degradation on some SUPERB tasks and states that the model is “not yet suitable for universal speech representation.” This does not invalidate the paper, since that is partly outside its chosen scope, but it is an important practical limitation: the gains in syllabic structure and compression appear to come with loss of some information useful for broader downstream tasks.

- **The method depends on SDHuBERT initialization and pseudo-segments, which narrows the conceptual novelty somewhat.**  
  The contribution is not de novo syllable discovery from scratch; it is a refinement/distillation framework built on top of an earlier syllable-inducing model. That is still a valid and useful contribution, but the presentation should be careful not to overstate how independently the syllabic structure is learned.

- **The uLM evaluation is limited to proxy discrimination metrics.**  
  sWUGGY and sBLIMP are standard and useful, but they are still narrow probes. Since the paper emphasizes spoken language modeling, the case would be stronger with direct generation-oriented evaluation or samples from the uLM pipeline rather than only zero-shot acceptability/lexical probes.

- **The paper gives limited analysis of what information is preserved versus discarded by syllabic tokens.**  
  Section 5.2 notes that Sylber appears to marginalize articulatory variation orthogonal to orthography and loses pitch detail after quantization, but this is only partially characterized. A more systematic analysis of preserved/lost phonetic, prosodic, and speaker-related information would sharpen the paper’s practical message.

- **The impressive multilingual/out-of-domain transfer in Table 2 lacks direct competing baselines in the same setting.**  
  The result is still valuable, but without comparable numbers for other methods on Fisher/MLS/AISHELL-3, it is hard to tell how unique this robustness is to Sylber rather than a broader property of syllable-oriented models.

### Minor
- **Some custom efficiency interpretations should be presented a bit more cautiously.**  
  The token-rate and bitrate results are straightforward and convincing; the custom “coding-rate” metric is less standard and partially downstream-ASR-dependent, so conclusions should rely more heavily on the raw rate/bitrate/intelligibility tradeoff than on this derived scalar.

### Trivial
- **Ablation coverage in the main paper is somewhat limited for headline robustness claims.**  
  The paper states that denoising is not a primary source of learning and that training is not sensitive to hyperparameters or initialization, but the main body provides little direct quantitative support for those statements.

## Nice-to-Haves
- Add a more controlled LM-efficiency study, e.g., match uLM architecture and training budget while varying only tokenizer/token rate.
- Include direct generation examples or evaluation from the trained uLMs, not only sWUGGY/sBLIMP.
- Expand the information-retention analysis to prosody, speaker traits, and finer phonetic detail.
- Temper the cognitive language in Section 6 and frame DI more explicitly as a probe of representational discreteness.
- Provide a small ablation on the self-distillation loop and on the role of the denoising objective.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Need missing related work / codec baselines such as EnCodec/SoundStream/DAC.”**  
  Removed under the instruction not to criticize missing related work. It is fair to say codec comparisons would broaden context, but not to penalize the paper for omitted external baselines whose necessity depends on scope.

- **“Baseline comparisons are unfair because HuBERT uses BPE-expanded vocabularies / different granularities.”**  
  Removed as a core criticism. The paper explicitly explains this design in Section 4.2.1: matching vocabulary sizes for HuBERT via BPE is intended to make the efficiency comparison fairer, not slanted toward Sylber. One can still note that the comparison is not definitive, but not treat it as an unfair-evaluation flaw.

- **“Claims about unreleased or unverifiable tools/APIs/benchmarks.”**  
  Removed per instruction. The cited systems/datasets/APIs are to be treated as real and available.

- **“The paper hides SUPERB results, so the limitation is unverifiable.”**  
  We should not treat absence of appendix text in this extracted copy as a paper flaw. The main paper explicitly states this limitation; the missing appendix here is an artifact of the review copy.

## Novel Insights
The most compelling synthesis is that the paper’s real contribution is **not** primarily “syllable tokenization beats all alternatives,” but rather that it demonstrates a practically useful design principle for speech representations: explicitly distilling toward self-discovered segment-level targets can transform noisy emergent structure into representations clean enough that a much cheaper inference algorithm becomes viable. That is a deeper contribution than the individual benchmark wins. At the same time, the paper also reveals a sharp tradeoff: imposing syllabic parsimony appears to improve segmentation regularity and coding efficiency, while discarding some of the fine-grained information needed for universal-purpose speech representations. Framed that way, the work is both more credible and more informative to the community.

## Suggestions
- Recast the headline claims more conservatively: emphasize **efficient syllabic tokenization with competitive downstream probe results**, rather than broad claims of minimal information loss or established scalable spoken LM.
- In Table 6 or an added experiment, control uLM architecture/training budget and vary only tokenizer to directly test the efficiency claim.
- Add a short main-paper discussion of which kinds of information are lost when imposing syllabic structure, especially prosody and fine phonetic detail.
- Reframe Section 6 around **embedding discreteness under synthetic articulatory interpolation**, and reserve stronger cognitive interpretations for clearly marked speculation.
- Include at least one ablation on the iterative self-distillation setup and one direct analysis of the denoising objective’s contribution to robustness.

## Score and Decision
**Assessment across axes:**  
- **Originality:** high. The self-segmentation distillation formulation and the emphasis on representation-induced fast segmentation are genuinely novel.  
- **Importance of the question:** high. Efficient speech tokenization for scalable spoken modeling is an important problem.  
- **Support for claims:** mixed-to-strong. The central segmentation/tokenization claims are well supported; some broader LM-efficiency and cognitive claims are overstated.  
- **Experimental soundness:** good overall, with strong segmentation evidence and broad evaluation, but some downstream interpretations are not cleanly controlled.  
- **Clarity:** generally strong; the motivation and method are clear, though some rhetoric should be softened.  
- **Value to the community:** high. Even with some overclaiming, this is a meaningful contribution to speech representation learning and tokenization.

**Calibration against human-reviewed anchors:**  
I compared this paper primarily against:
- **SyllableLM (dGSOn7sdWg.md, Accept (Poster), scores 6/6/6/8):** very close topic and similar strength pattern—novel syllable-level/tokenization idea with meaningful empirical support but not fully definitive on all downstream fronts. Sylber’s segmentation evidence feels stronger and more concretely grounded than the typical “promising” tokenizer paper, placing it at least in this acceptance band.
- **SpeechTokenizer (AF9Q8Vip84.md, Accept (poster), scores 6/6/3/8):** another tokenization-focused paper accepted despite some evaluation limitations. Sylber is comparably broad and arguably more distinctive in its representation-learning angle.
- **RepCodec (LfDUzzQa3g.md, Reject, scores 6/5/5/6):** useful lower anchor for tokenization papers that are competent but not convincing enough. Sylber is above this: its central contribution is cleaner, and Table 1 provides a stronger “why this matters” result.
- **DC-Spin (OW332Wh9S5.md, Reject, scores 5/1/8/5):** another lower anchor where downstream claims and evaluation mismatches contributed to rejection. Sylber shares some overclaiming tendencies, but its core evidence is much stronger and more self-contained than DC-Spin’s.

Relative to these anchors, this paper looks **solidly above the reject-side tokenization papers and in the lower-to-middle accept range**. The work has a real contribution, and the weaknesses mostly argue for **claim calibration**, not for overturning the central result.

**Final score: 7.0 / 10**  
**Decision: Accept**

MY FINAL SCORE: <pineapple>7.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>