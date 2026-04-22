Now I have thoroughly reviewed the paper and calibration anchors. Let me compose the final review.

## Summary

Delta proposes adapting Visual Contrastive Decoding (VCD) to text by randomly masking input tokens and subtracting the resulting logits from the original logits during inference. The core idea is that masked inputs amplify "hallucination-prone" responses (model priors), so subtracting them recovers context-grounded content. The method is evaluated on SQuAD v1.1, SQuAD v2, TriviaQA, Natural Questions, CommonsenseQA, and MMLU, with the strongest result being a 14.53 pp improvement in no-answer exact match on SQuAD v2.

## Strengths

- **Strong NoAns_EM improvement on SQuAD v2:** Delta achieves a +14.53 pp improvement (w/o sampling) and +11.81 pp (w/ sampling) on the no-answer exact match, demonstrating genuine effectiveness at helping the model abstain from answering when context doesn't support an answer (Table 1).

- **Inference-time, training-free approach:** Delta requires no fine-tuning, additional data, or external models, making it immediately deployable with any pre-trained LLM. This practical advantage is clear.

- **Transparent limitation reporting:** The authors explicitly evaluate on context-free benchmarks (CommonsenseQA, MMLU) and report the small degradations there (Table 2), clearly delineating the method's scope.

- **Consistent improvements on context-rich QA with sampling:** Under sampling, Delta improves SQuAD v1.1 (+4.4 pp EM), SQuAD v2 (+6.1 pp EM), TriviaQA (+7.8 pp EM), and NQ (+2.6 pp EM).

## Weaknesses

### Fatal
None.

### Major

- **No experimental comparison with CAD (Shi et al., 2024) or DoLA, the most directly comparable prior work.** CAD is acknowledged in Section 2 as achieving "a similar outcome," and the paper claims Delta is "more generalizable" because CAD is "mainly based on context-driven datasets." However, Table 2 shows Delta also fails on context-free tasks (CommonsenseQA: -0.25, MMLU: -0.29), directly contradicting this generality claim. Without head-to-head comparison, it is impossible to assess whether Delta offers any improvement over the most relevant baselines. DoLA (Chuang et al., 2024) is cited for contrastive decoding but never discussed or compared experimentally.

- **The central claim of "hallucination mitigation" is not evaluated with hallucination metrics.** The paper's title, abstract, and introduction all frame Delta as a hallucination mitigation method, but evaluation uses only standard QA accuracy metrics (EM, F1). While NoAns_EM on SQuAD v2 measures one aspect (abstaining when no answer exists), this alone does not justify the broad hallucination framing. No faithfulness, factuality, or dedicated hallucination benchmarks are used. The gap between the claimed contribution ("mitigates text hallucinations") and the actual evidence (QA extraction accuracy on a narrow set of tasks) is significant.

- **Delta hurts greedy-decoded performance on key datasets, and this inconsistency is inadequately addressed.** On greedy decoding (w/o sampling), Delta slightly hurts TriviaQA (48.27→48.13), NQ (14.88→14.57), and SQuAD v2 HasAns_EM (59.08→57.47). If Delta's mechanism works as theorized—subtracting hallucination-prone logits—it should not degrade greedy performance. The paper acknowledges "marginal" progress without sampling (Section 5.2) but does not honestly acknowledge or explain the performance degradation. This raises questions about whether the mechanism works as theorized, or whether improvements under sampling stem from a different effect.

### Minor

- **Single model, 4-bit quantized configuration only:** All results are on Llama 3.1 8B Instruct with 4-bit quantization. No full-precision results or other model families are tested, limiting generalizability claims. No justification is provided for the 4-bit choice.

- **APC explanation is confusing:** Section 3.5 states "the logit with probability higher than the particular threshold ... is not selected to the set V_head," but Equation 4 shows that tokens with probability *above* the threshold are *included* in V_head. This contradicting description makes the method harder to understand and reproduce.

- **Narrow ablation study:** The ablation only varies mask ratio and α on SQuAD v1.1. Key design choices—EOS token as MASK token, masking only input vs. input+generation, alternative masking strategies—are not ablated.

- **Notation inconsistency between Sections 3.3 and 3.4:** Section 3.3 defines masking on input x, but Section 3.4 applies mask(z) where z includes generated tokens y₁,...,y_{t-1}. Whether masking extends to previously generated tokens is a significant design choice that goes undiscussed.

## Nice-to-Haves

- Comparison with CAD and DoLA, which are the most directly relevant baselines
- Evaluation on a dedicated hallucination benchmark (e.g., TruthfulQA, a faithfulness metric for summarization or RAG)
- Testing on additional models and full-precision configurations
- Qualitative examples showing real cases where Delta correctly abstains vs. incorrectly suppresses valid answers

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Claim that DoLA is not cited/discussed at all:** DoLA (Chuang et al., 2024) IS cited in the introduction (line 21) and appears in the reference list, though it is not discussed in the Related Work section. The claim was overstated; the real issue is that DoLA is mentioned but not compared experimentally or discussed substantively in related work.

- **Demand for confidence intervals or multiple runs:** Large-scale benchmark evaluation with single runs is standard practice in this field; requesting confidence intervals is a generic minor nitpick.

- **Formatting/notation nitpicks about the MASK token choice and inconsistency of z vs x:** While the notation inconsistency is real, calling for extensive discussion of "why not a dedicated [MASK] token" is a design-choice nitpick beyond the scope of what the paper needs to establish.

- **Critic's claim that the paper "selectively reports favorable conditions":** The paper reports both w/ and w/o sampling conditions for all datasets in Table 1, including the decreases. The data is transparent, even if the narrative emphasizes improvements.

## Novel Insights

The most interesting observation emerging from the reviews is that Delta's strongest result—dramatic improvement on SQuAD v2's no-answer cases—suggests the method is fundamentally an "abstention enhancer" rather than a general hallucination mitigator. This reframing would make the paper's claims more honest and focused, since knowing *when not to answer* is a narrower and more precisely evaluable contribution than the broad claim of "hallucination mitigation."

## Suggestions

- Reframe the contribution as "improving context-dependent abstention" rather than general hallucination mitigation, which better matches the evidence.
- Add experimental comparison with CAD and DoLA to establish whether Delta offers any advantage over existing contrastive decoding methods.
- Honestly acknowledge and analyze why Delta hurts greedy-decoded performance on TriviaQA, NQ, and SQuAD v2 HasAns_EM, rather than characterizing it as "marginal."

## Score and Decision

**Calibration anchors:**

- **DoLa (avg 7.25, Accept poster):** Novel layer-contrastive method for factuality, strong TruthfulQA improvements (+12-17%), broader evaluation, better analysis. Delta is substantially weaker (narrower evaluation, missing baselines, inconsistent results).
- **Differential Transformer (avg 8.0, Accept oral):** Novel architecture contribution with thorough empirical validation. Far stronger than Delta.
- **RITUAL (avg 5.0, Reject):** Conceptually similar (random transformations + contrastive decoding for hallucination), incremental novelty. Delta has comparable novelty but weaker evaluation (missing most relevant baselines, narrower scope).
- **GACD (avg 4.75, Reject):** Gradient-based contrastive decoding for multimodal hallucination, limited experimental validation and questionable novelty claims. Similar weakness profile to Delta.
- **On inherent limitations of GPT/LLM Architecture (avg 2.0, Reject):** Fundamentally flawed theoretical paper. Delta is clearly much better than this.

Delta is comparable to RITUAL and GACD in novelty profile (incremental adaptation of contrastive decoding) but has additional weaknesses in missing baselines and inconsistent results. It falls below RITUAL because RITUAL at least compared against existing CD methods (VCD, etc.). A score of **4.0** reflects a borderline reject: the idea and some results are sound, but the missing baseline comparisons, overclaimed generality, and inconsistent evidence under greedy decoding significantly weaken the paper.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>