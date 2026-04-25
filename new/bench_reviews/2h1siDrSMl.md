Now let me run calibration searches to anchor my scoring.Now I have enough context. Let me compile the final review with verified claims.

The key findings from paper verification:

1. **Google Search as Stage-2 retrieval** (confirmed at line 84-88): "the second stage query-expanded text retrieval with a *Google Search* engine" — this is confirmed real, while all baselines use Wikipedia. This is a genuine structural concern.

2. **1M WikiWeb2M pre-training vs. "minimal training" claim** (confirmed at lines 126-131): The paper indeed uses 1M WikiWeb2M pre-training before 10K fine-tuning, and Table 3 shows this accounts for ~54% of the gain over LLaVA-v1.5 baseline.

3. **Table 1 vs. Table 2 discrepancy** (confirmed): Table 1 shows 25.10/27.34 for RoRA-VLM on InfoSeek Entity/Query, Table 2 shows 24.56/26.33 for the same model. Real discrepancy, unexplained.

4. **Stage-2 precision lower than Stage-1** (confirmed, Table 5): Stage-1 is 38.53/37.67, Stage-2 is 27.01/26.97. But these measure different things (entity match vs. answer retrieval), so this is not necessarily paradoxical — the paper doesn't explain the seeming paradox though.

5. **Training noise vs. inference noise mismatch**: Training has 1/3 noise, inference may have ~65% noise based on Stage-1 recall (~35%). The paper mentions Appendix A.4 analyzes robustness under varying noise, which partially addresses this.

6. **Equation 4 notation**: Minor — the subscript issue in Eq. 4 is present but the text explanation is clear.

7. **Domain transfer claim**: Table 4 shows RoRA-VLM domain transfer (20.26) beats LLaVA-v1.5 SFT (18.23), which is meaningful but perhaps overstated as "novel capability".

Now, the most critical question: how serious is the Google Search issue?

The Google Search issue is genuinely problematic for the cross-system comparisons (Table 1), but the internal ablations (Tables 2, 3, 4) remain valid as they compare variants within the same system. The paper does contribute:
- A concrete 2-stage retrieval design
- Adversarial noise injection that shows large gains in Table 2 (7+ points)
- Visual token refinement (moderate gains)
- Knowledge-intensive pre-training insights

The harsh critic calls this "Fatal" and says the paper "cannot be accepted." But I think this is too strong — the paper has real internal validity through its ablations. However, the comparison with baselines in Table 1 is genuinely misleading, particularly the "minimal training" claim when 1M WikiWeb2M pre-training is used. These are Major weaknesses.

For calibration:
- VvDEuyVXkG (multimodal RAG, accepted, 6.25): Comparable topic, sound methodology, uses proper static evaluation
- J2Jyp1SZ0n (MMSearch, accepted, 6.5): Uses web search as part of the pipeline for evaluation, but that's the paper's point, not a confound
- rGk0ur4Tfr (retrieval-based video VLM, rejected, 4.75): Uses retrieval for video QA, rejected for weak innovation/limited experiments

This paper has genuine innovations (adversarial noise injection, 2-stage retrieval, visual token refinement) but has the serious Google Search confound that undermines all cross-system comparisons. It's comparable to papers that got ~4-5.

Given:
- The internal ablations are solid (Tables 2, 3 show clear gains from the proposed components)
- The cross-system comparison is confounded by Google Search
- The "minimal training" framing is misleading (hides 1M pre-training)
- There's a Table 1/2 discrepancy
- The paper has 3 valid components with evidence

I would score this around 4.5-5.0. The paper has genuine contributions but the misleading framing and confounded comparisons are significant issues. Comparable to papers with avg ~4.75-5.

My final score: **4.5**Now I have everything I need to write the final review.

---

## Summary

RoRA-VLM introduces a retrieval-augmented framework for vision-language models addressing two challenges: (1) modality discrepancy in VQA retrieval queries, resolved via a two-stage image-anchored textual-query expansion, and (2) noisy retrieved knowledge, resolved via adversarial noise injection during training and query-oriented visual token refinement. The framework is evaluated on OVEN, InfoSeek, and Enc-VQA benchmarks, where a 7B model reports state-of-the-art results over all baselines including 55B models.

---

## Strengths

- **Image-anchored 2-stage retrieval design** (Section 3.2): Using the query image as an entity anchor for query disambiguation is a well-motivated design that cleanly addresses the anaphoric reference problem in text queries. Figure 5 qualitatively validates that retrieved images successfully identify the target entity across different perspectives (e.g., distinguishing Buland Darwaza from other monuments), and Appendix A.5 (referenced in §5) reports an 11.52% improvement in retrieval precision over single-stage retrieval.

- **Adversarial noise injection produces large, well-evidenced gains** (Table 2): Removing retrieved images while keeping all text knowledge ("text-only RAG") drops InfoSeek Entity from 24.56% to 17.29% and Query from 26.33% to 19.28%—a ~7-point swing. This strongly supports the claimed mechanism: the model learns to filter irrelevant knowledge by comparing retrieved vs. query entity images, not merely by leveraging retrieved text.

- **Attention visualization provides mechanistic interpretability** (Figure 4): The attention heatmaps clearly show RoRA-VLM attending to knowledge snippets whose associated images match the query entity (e.g., focusing on the Landwasser Viaduct snippet while ignoring an unrelated passage), directly evidencing the learned filtering behavior.

- **Entity-rich pre-training shown to be a key enabler** (Table 3): The ablation comparing WikiWeb2M vs. ShareGPT4V pre-training is well-controlled. WikiWeb2M yields 24.56/26.33 on InfoSeek while ShareGPT4V gives only 21.28/22.84, and removing pre-training entirely drops to 20.68/23.41, confirming that entity-rich data—not just more pre-training—drives the gains.

- **Broad evaluation across three diverse benchmarks** (Table 1): Covering entity recognition (OVEN), information-seeking VQA (InfoSeek), and encyclopedic VQA (Enc-VQA) provides a substantially wider evaluation surface than most retrieval-augmented VLM papers.

---

## Weaknesses

### Fatal
None.

### Major

- **Stage-2 text retrieval uses live Google Search while all baselines use static Wikipedia — this confounds all cross-system comparisons in Table 1.** Section 3.2 (confirmed at paper line 84) explicitly states: "the second stage query-expanded text retrieval with a *Google Search* engine, leveraging the vast resources of the web to enhance retrieval accuracy." Google's index is orders of magnitude larger than Wikipedia, continuously updated, and semantically optimized for precisely this factual lookup task. Every baseline (Wiki-LLaVA, PreFLMR, RA-CM3, PaLI) retrieves from Wikipedia. The reported margins over Wiki-LLaVA in Table 1 (e.g., 25.10 vs. 21.44 on InfoSeek Entity) are equally consistent with the hypothesis that Google Search returns better snippets than Wikipedia retrieval — no experiment in the paper disentangles the knowledge-source advantage from the method advantage. This is the most serious flaw because the headline claim that "RoRA-VLM outperforms state-of-the-art retrieval-augmented VLMs" rests almost entirely on Table 1. The internal ablations (Tables 2–4) retain validity since they compare system variants under identical retrieval conditions, but the competitive claim requires an experiment replacing Google Search with a static offline corpus at the same query setup.

- **The "minimal training" claim in the abstract misleadingly omits the 1 million-instance WikiWeb2M pre-training.** The abstract and Section 5 repeatedly claim competitive performance "with only a minimal number of training instances (e.g., 10,000)" and contrast against baselines "fine-tuned on up to 1 million instances." Yet Section 4 (confirmed at lines 126–131) describes a **1 million-instance WikiWeb2M pre-training stage** before the 1,000-instance downstream fine-tuning. Table 3 directly quantifies the impact: WikiWeb2M pre-training alone lifts LLaVA-v1.5 from 10.34→18.00 on InfoSeek Entity (7.66 points), accounting for ~54% of the total improvement over the base LLaVA-v1.5 (14.22 points to 24.56). Baselines such as Wiki-LLaVA and PreFLMR do not include an equivalent 1M pre-training stage. The "minimal training" framing therefore systematically misrepresents the training budget and conflates the benefit of entity-rich pre-training with the benefit of the proposed retrieval method.

### Minor

- **Unexplained discrepancy between Table 1 and Table 2 for the same model.** Table 1 reports RoRA-VLM on InfoSeek as 25.10 (Entity) / 27.34 (Query), while Table 2 (ablation on InfoSeek) reports 24.56 / 26.33 for "RoRA-VLM (ours)" — a gap of 0.54–1.01 points. The caption for Table 1 says "fine-tuned on less than 10,000 instances" while Section 4 says "randomly sampled 1,000 instances for lightweight fine-tuning." These may refer to different fine-tuning budgets, but this is not stated anywhere in the paper. Without explanation, the ablation baselines in Table 2 may not be comparable to the headline Table 1 numbers, undermining the ablation analysis.

- **Stage-2 retrieval precision (27.01%) is lower than Stage-1 (38.53%) without explanation** (Table 5). These are measuring different things (Stage-1: entity image match; Stage-2: answer inclusion), so the comparison is not directly apples-to-apples. However, a Stage-2 answer retrieval precision of only ~27% means the majority of retrieved text snippets do not contain the answer—worth analyzing given that this is the component responsible for actually providing the factual answer.

- **Training noise ratio (33%) does not match realistic inference noise (~62–65%).** Training uses 1 irrelevant snippet out of 3 total. Stage-1 retrieval precision is only ~35–38% (Table 5), meaning at inference roughly 62–65% of retrieved knowledge snippets may correspond to wrong entities—nearly double the training noise level. The paper mentions Appendix A.4 analyzes robustness under varying noise ratios, which partially addresses this, but the gap between training and inference noise distributions is not explicitly discussed.

### Trivial

- **Equation 4 has a minor subscript notation issue.** The score $s_j$ is defined in the text as $\sum_{i=1}^m (\mathbf{x}_{I,i} \cdot \mathbf{x}_{\tilde{I}_i,j})$ — i.e., the sum is already inside $s_j$ — but Eq. 4 writes $\sum_{i=1}^m s_j$ in the conditioning of the top-$m$ selection, making $s_j$ appear independent of $i$. The intended meaning is clear from the surrounding text, but the equation as written is technically inconsistent.

---

## Nice-to-Haves

- **An ablation replacing Google Search with a static offline text retriever (BM25 or dense retrieval over a Wikipedia snapshot) would directly quantify how much of the Stage-2 gain is attributable to query expansion vs. knowledge-source quality.** This is the single most impactful experiment the authors could add.

- **A random-token selection baseline in Table 2** (retaining $m$ tokens randomly rather than by average pooling) would cleanly isolate the contribution of *relevance-based* selection from mere token-count reduction.

- **Comparison of domain-transfer results using LLaVA-v1.5 domain-transfer as the primary reference** (17.18 vs. 20.26) rather than LLaVA-v1.5 full SFT (18.23 vs. 20.26) would give a cleaner picture of RoRA-VLM's actual generalization advantage.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "Non-reproducible results" as a separate structural flaw.** The reproducibility concern (Google's index changes over time) is real, but is downstream of the already-kept fairness concern. It does not constitute an independent fatal flaw beyond what the unfair comparison already captures. Folded into the Google Search major weakness.

- **Harsh Critic: "Visual token refinement uses a weak ablation comparator" (should use random token selection).** While the suggestion for a random-selection baseline is reasonable, the existing average-pooling comparison is a commonly used practical alternative in the literature and is not a misleading baseline per se. Moved to Nice-to-Haves.

- **Harsh Critic: "Stage-2 precision lower than Stage-1 means Stage-2 may add more noise."** The metrics measure different things—Stage-1 checks entity image matching (recall-like), Stage-2 checks answer coverage in text. A 27% answer coverage rate is not inherently "worse" than Stage-1 entity precision; it reflects the harder task of exact answer retrieval. The observation is noted as a Minor issue for explanation, but the claim that Stage-2 "introduces more noise than it removes" is not established.

- **Harsh Critic: "Domain transfer claim is not novel capability."** The observation that the gap (RoRA-VLM 20.26 vs. LLaVA-v1.5 domain-transfer 17.18) is modest is fair, but Table 4 does show RoRA-VLM domain-transfer (20.26) outperforming LLaVA-v1.5 full SFT (18.23) — this is a meaningful result even if not spectacular. The claim of "novel zero-shot domain transfer capability" is somewhat overstated but is not a core flaw; moved to Nice-to-Haves.

- **Strength Finder: "Minimal training data requirement as a strength."** Conflicted directly by the verified 1M WikiWeb2M pre-training. This is not a valid strength given the hiding of the pre-training cost.

---

## Novel Insights

The adversarial noise injection strategy—training with exactly one guaranteed-wrong retrieved snippet alongside correct ones—shows a 7-point accuracy drop when images are removed at inference, demonstrating that the model genuinely uses visual entity matching to filter textual knowledge rather than relying on text relevance alone. This is a concrete, verifiable finding that could inform future noise-robustness training strategies in retrieval-augmented multimodal systems. The observation that entity-rich image-text pre-training (WikiWeb2M) contributes roughly 54% of the total improvement over the base VLM—and is qualitatively distinct from generic image captioning pre-training (ShareGPT4V)—is a practically useful empirical finding about what pre-training data matters for knowledge-intensive VQA.

---

## Suggestions

1. **Replace Google Search with an offline static retriever (BM25 or dense retrieval over a Wikipedia snapshot dated before the evaluation period) and re-run Table 1.** Without this, the competitive results cannot be attributed to the proposed architectural innovations.
2. **Reframe the abstract and introduction to accurately describe the full training budget** (1M WikiWeb2M pre-training + downstream fine-tuning) so the "training efficiency" comparisons are honest.
3. **Clarify the Table 1 vs. Table 2 fine-tuning setup discrepancy** — explicitly state whether different numbers of fine-tuning instances are used, and if so, why the ablations use a different regime from the main comparison.
4. **Add a paragraph in Section 5 discussing the discrepancy between training noise ratio (33%) and the actual inference noise rate** inferred from Stage-1 precision (~62–65%), either justifying the training setup or showing that results are robust across noise levels (citing Appendix A.4 results in the main text).

---

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Score | Comparison |
|---|---|---|---|
| Multimodal RAG with Dynamic VQA (VvDEuyVXkG) | `/home/wg25r/review_agent/human_reviews/VvDEuyVXkG.md` | 6.25 | Accepted mRAG paper with clean methodology; no knowledge-source confound |
| MMSearch (J2Jyp1SZ0n) | `/home/wg25r/review_agent/human_reviews/J2Jyp1SZ0n.md` | 6.5 | Accepted; uses web search but as the system's stated feature, not a comparison confound |
| MMed-RAG (s5epFPdIW6) | `/home/wg25r/review_agent/human_reviews/s5epFPdIW6.md` | 7.0 | Accepted; strong multimodal RAG with clean baselines, stronger experimental design |
| Retrieval-Based Video LM (rGk0ur4Tfr) | `/home/wg25r/review_agent/human_reviews/rGk0ur4Tfr.md` | 4.75 | Rejected; simpler retrieval-based VLM, limited innovation, but no major evaluation confound |
| Text-oriented VQA with LLMs (N4gT8PIjHL) | `/home/wg25r/review_agent/human_reviews/N4gT8PIjHL.md` | 3.5 | Rejected; much weaker — no novel method, just a pipeline analysis |

**Reasoning**: The paper under review has genuine technical contributions (adversarial noise injection with 7+ point ablation evidence, 2-stage retrieval design, visual token refinement) well-validated by internal ablations. These contributions are more substantive than the rejected rGk0ur4Tfr (avg 4.75). However, the two major issues — the Google Search confound that invalidates all cross-system comparisons in Table 1, and the misleading "minimal training" claim hiding 1M pre-training — are more serious than anything in the accepted mRAG papers (6.25–7.0). The core competitive claim of the paper rests on Table 1, and that table's comparison is not clean. The paper sits clearly below the accepted mRAG papers and is closer to the rejected retrieval-VLM anchor (4.75). Given the stronger internal ablation evidence than rGk0ur4Tfr but the more problematic evaluation design, I place this at **4.5**.

**Axes:**
- *Originality*: Moderate — 2-stage image-anchored retrieval and adversarial noise injection are genuine novelties; visual token refinement is incremental.
- *Importance of research question*: High — knowledge-intensive VQA for VLMs is an important open problem.
- *Claims well-supported*: Partially — internal ablations are strong; cross-system comparisons are not.
- *Soundness of experiments*: Weak — Google Search confound and misleading training budget framing undermine the main results table.
- *Clarity of writing*: Adequate — structure is clear, but the training setup is ambiguous (10K vs. 1K fine-tuning discrepancy).
- *Value to research community*: Moderate — the adversarial noise injection finding and entity-rich pre-training insights are genuinely useful, but results cannot be fairly compared to the field's baselines.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>