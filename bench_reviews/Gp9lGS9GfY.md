## Summary
This paper presents **GAR**, a region-level MLLM for mask-conditioned understanding that aims to preserve both fine local detail and necessary global context via a simple **RoI-aligned feature replay** mechanism. It also introduces **GAR-Bench**, a benchmark targeting not only single-region perception but also multi-prompt interaction, non-entity recognition, and compositional reasoning, and reports strong results across captioning, VQA, and some zero-shot video transfer settings.

## Strengths
- **The paper identifies and directly targets a real limitation of many region-level MLLMs: local-region understanding without enough scene context.** This is not just claimed qualitatively; the architectural ablations in Table 8 are informative. In particular, the proposed global-image + RoI-aligned replay variant substantially improves GAR-Bench results over local-only, cross-attention, RoI-pooled, and crop-supplement baselines while remaining competitive on detailed captioning.
- **The benchmark contribution is more substantive than a standard “another caption benchmark.”** GAR-Bench explicitly evaluates difficult cases that are easy to miss in region-captioning-only setups, especially **non-entity recognition** (e.g., reflections), **position reasoning**, and **multi-prompt relations with distractors**. This is a useful diagnostic decomposition of region understanding.
- **The data ablations support the paper’s claimed capability progression.** Table 10 shows that adding the fine-grained dataset primarily improves detailed recognition, while the relation dataset drives the large jump on GAR-Bench captioning and VQA, matching the intended role of each data source.
- **The paper does more than report a single favorable benchmark.** Beyond GAR-Bench, it evaluates on DLC-Bench, Ferret-Bench, MDVP-Bench, LVIS/PACO recognition, and VideoRefer. Even if the strongest headline claims rely heavily on GAR-Bench, the broader evaluation does show that the model is not narrowly tuned only for one metric.
- **The authors proactively study evaluation robustness rather than treating their benchmark as unquestionable.** The subsampling analyses (Tables 12–13), cross-judge comparisons (Table 14), and input-format analysis for general VLMs (Table 15) are all useful and strengthen the empirical section.

## Weaknesses

###: Fatal
- **Potential train/benchmark contamination is a serious concern for the core multi-prompt reasoning claims.**  
  The paper states in Section 3.3 that Round 2 training data is built using the **PSG dataset** to generate relation-aware captions, QA pairs, and MCQs (“we incorporated the Panoptic Scene Graph (PSG) dataset ... We construct a relation dataset with 414K samples”). Appendix B.1 then states that GAR-Bench relation tasks also **source images from PSG** (“For the ‘relation’ tasks, we source images from the Panoptic Scene Graph (PSG) dataset”).  
  The paper does **not** clearly specify that GAR-Bench uses disjoint images or a held-out split relative to the PSG-derived training data, nor does it document deduplication. Because the benchmark is central to the headline claim that GAR excels at multi-prompt interaction and compositional reasoning, this missing split hygiene materially weakens the evidence. This is not yet enough to prove leakage, but it is a substantial unresolved threat to validity.

### Major:
- **The paper overstates the novelty of the core architectural idea.**  
  The method is effective, but the framing sometimes suggests a more fundamental architectural leap than the paper actually delivers. Section 3.2’s key mechanism is: encode the **full image**, derive a box from the mask, then apply **RoI-Align** on the global feature map to extract context-aware regional features. This is a sensible design and the ablations suggest it works well, but the paper’s narrative at times reads as if this resolves a long-standing architectural dilemma in a fundamentally new way. The real contribution is better described as a strong engineering synthesis and adaptation for region-level MLLMs, plus the training/data/benchmark package, rather than a deeply novel architectural principle.
- **The strongest “beats much larger models / beats proprietary models” claims rely heavily on the authors’ own benchmark, which is also LLM-judged in part and difficulty-filtered against strong models.**  
  GAR-Bench is valuable, but it is also curated in a way that can amplify benchmark-specific optimization. Appendix B.1 states that any question answered correctly by all four strong non-thinking MLLMs was removed. This makes the benchmark intentionally difficult, but it also means the test set is not a neutral sample of region-understanding tasks. In addition, GAR-Bench-Cap depends on LLM judging. The paper does include cross-judge consistency analyses, which helps, but that does not fully establish that the benchmark ranking translates into broad superiority over larger models. The external benchmark results are strong, yet the most aggressive comparative claims should be phrased more cautiously.
- **The “arbitrary number of prompts” claim is stronger than the presented evidence.**  
  The task formulation allows a set of \(N\) prompts, and Figure 6b notes examples with up to 7 and 9 prompts, but the paper does not provide a systematic performance breakdown versus prompt count. Since scalability to more simultaneous prompts is one of the paper’s conceptual selling points, the lack of stratified analysis leaves this claim under-supported.
- **The synthetic relational data pipeline is plausible but under-validated.**  
  Section 3.3 relies on a seed captioner plus an LLM merger to generate large amounts of relation-aware descriptions and QA. The paper mentions quality control and human curation for GAR-Bench, but does not provide quantitative error rates, agreement rates, or noise analysis for the 2.5M training corpus. Since the paper’s multi-prompt reasoning gains appear heavily driven by this data, more evidence about annotation fidelity would materially strengthen technical soundness.

### Minor
- **The video-transfer narrative is somewhat overstated relative to the paper’s own evidence.**  
  The authors do acknowledge the limitation, including in Appendix E, and Tables 6–7 indeed show weakness on temporal aspects such as temporal description and future prediction. So this is not a misrepresentation in the sense of being hidden. However, claims like “strong capabilities can be easily transferred to videos” should be tempered: the results support useful zero-shot transfer for some tasks, but not robust temporal understanding.
- **The paper lacks a direct controlled test showing that GAR truly uses global context, rather than merely benefiting from having extra tokens.**  
  The architecture and qualitative examples are consistent with the claim, and Table 8 supports the design choice empirically. Still, a cleaner context-ablation experiment—e.g., masking/randomizing background while keeping the prompt fixed—would more directly validate the core mechanism behind non-entity and position reasoning.
- **Efficiency reporting is limited.**  
  Table 8 reports first-token latency and ViT token counts, which is useful, but there is no fuller accounting of memory/FLOPs or sequence-length overhead versus alternatives. Given that one selling point is balancing performance with practicality, a somewhat sharper efficiency characterization would help.
- **There is at least one apparent inconsistency between text and table values.**  
  In Section 4, the text says “GAR-8B achieves an impressive overall score of 54.5” on GAR-Bench-VQA, but Table 1 reports **59.9**. This should be corrected.

### Trivial
- None.

## Nice-to-Haves
- Provide a **prompt-count-stratified** breakdown of GAR-Bench performance to substantiate scalability beyond two or three regions.
- Add a **context ablation** experiment (remove or corrupt the background while preserving the prompted region) to directly test whether the method uses global context for non-entity and positional reasoning.
- Report **training/inference memory and FLOPs** for RoI-aligned replay vs. crop-based and cross-attention alternatives.
- Include a **noise/verification analysis** for the synthetic relation data: e.g., sampled human agreement, estimated hallucination rate, or correction statistics.
- Test robustness to **imperfect prompts** such as noisy masks, SAM-generated masks, or boxes, since practical deployment will rarely have ideal manual masks.
- For the video extension, a small-scale **video fine-tuning** experiment would clarify whether the current limitation is architectural or purely data-related.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper fails to specify how multiple prompt features are ordered or tokenized before feeding to the LLM.”**  
  The exact routing details are indeed not deeply elaborated in the main text, but this criticism overreaches. The paper does provide a usable high-level architecture description, and the omission is more a detail-level reproducibility request than a substantive flaw under current standards.
- **“Prior work already solved this with identical mechanisms, so the contribution is not meaningful.”**  
  The paper may overstate novelty, but this stronger claim goes too far. The ablations show that this particular design choice matters in this setup, so the method should not be dismissed as vacuous.
- **“General models’ weak GAR-Bench performance is probably only due to format unfamiliarity.”**  
  This is too speculative. The paper partially addresses input-format concerns in Table 15 by trying several region-specification formats, so a pure format-mismatch explanation is not supported by the evidence provided.
- **Pure reproducibility nitpicks about missing implementation minutiae.**  
  Appendix C includes core implementation details and hyperparameters; remaining omissions are not substantial enough to be central review points.
- **Criticism that cited tools/models/benchmarks may not be available or verifiable.**  
  Removed per instruction.

## Novel Insights
The strongest reading of the paper is not “a radically new architecture beats giant models,” but rather that **region-level multimodal understanding may now be bottlenecked as much by evaluation/task formulation and relation-centric data construction as by backbone design**. GAR’s gains seem to come from the combination of (i) preserving context through a simple but effective feature extraction choice, (ii) explicitly training on relation-aware multi-prompt data, and (iii) evaluating on tasks that expose failures hidden by ordinary region captioning. In that sense, the benchmark and data pipeline may be at least as consequential as the model modification itself.

## Suggestions
- **Clarify split hygiene immediately.** Explicitly document whether PSG images/annotations used for GAR-Bench are disjoint from all PSG-derived training data; if so, say exactly how. If not, revise the claims and add held-out evaluation.
- **Reframe novelty more precisely.** Position RoI-aligned feature replay as an effective region-MLLM design choice validated by ablations, rather than as a fundamentally unprecedented mechanism.
- **Temper the largest comparative claims.** Replace broad statements about surpassing much larger or proprietary models with benchmark-scoped claims unless supported by more independent held-out evaluations.
- **Add prompt-count analysis.** Show performance as the number of prompts increases, especially on relation tasks.
- **Quantify training data quality.** Report sampled human validation for the synthetic relation corpus.
- **Strengthen the context claim with a direct causal test.** A background-removal/randomization experiment would be especially convincing.
- **Soften the video-transfer framing.** Present current results as promising zero-shot transfer for some video tasks, while clearly delimiting the absence of robust temporal modeling.