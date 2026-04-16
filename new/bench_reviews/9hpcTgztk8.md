## Summary

The paper proposes REPLM, a framework that reformulates document-level relation extraction (RE) as an in-context few-shot learning problem for large language models. It retrieves distantly supervised documents as relation-specific in-context examples, constructs multiple example sets, and aggregates model outputs via a weighted scoring procedure to generate subject–relation–object triplets without explicit NER or gradient-based fine-tuning. The authors evaluate across six datasets and multiple LMs, claiming state-of-the-art performance and arguing that REPLM reveals substantial missing annotations in DocRED.

## Strengths

- **Clear, well-motivated formulation:** The paper articulates the limitations of NER-dependent, fine-tuned document-level RE models (error propagation, inflexibility to new relations / LMs) and presents a coherent alternative: triplet generation via in-context examples, with per-relation prompting and probabilistic aggregation over multiple retrieved example sets (Sec. 3–4). The high-level pipeline is easy to follow and technically sensible as an ICL design.

- **Effective retrieval and aggregation design:** The combination of semantic retrieval from a distantly supervised pool (Sec. 4.1), random subsampling into multiple sets with similarity-based weighting (Sec. 4.2), and an aggregated scoring function over subject–object generations (Sec. 4.3) is well thought out. The ablation study (Table 5) convincingly shows that (i) “best context” outperforms random context and (ii) the full multi-set framework yields consistent gains across all six datasets and all five backbones, indicating that the multi-set design is an important methodological contribution.

- **Broad empirical coverage and portability:** The framework is instantiated with five different LMs (GPT-JT, Llama-3.1-8B/70B, GPT-3.5, GPT-4o) and evaluated on six datasets (DocRED, CDR, GDA, CoNLL04, NYT, ADE), with more than 30 baselines summarized in Table 4. REPLM generally improves as the backbone LM improves, and GPT-4o variants achieve very strong performance across datasets. This strongly supports the claim that the framework is portable and benefits from future LM improvements.

- **Insightful analysis of dataset issues:** The paper provides concrete evidence that DocRED’s human-annotated dev set is incomplete: REPLM generates many triplets that match Wikidata but are labeled as false positives under the original annotations (Sec. 6.1–6.2, Appendix F–G). Similar issues are identified on CDR and GDA for entity normalization (e.g., “complement receptor 1” vs. “CR1”) and on NYT for noisy distant labels. This is a valuable contribution for the RE community.

- **Interesting “learning vs memorization” probe:** The experiment in Sec. 8 replacing all entities with random names in CoNLL04, while not definitive, is a thoughtful attempt to test whether the system is merely retrieving KB facts or actually using textual context. The modest performance drop (72.9 → 70.47 F1) suggests the framework is indeed doing nontrivial extraction.

## Weaknesses

### Fatal

None of the identified issues reduce the work to “not even a paper” or directly falsify the core empirical finding that this pipeline can reach strong performance. However, there are serious problems in framing, evaluation fairness, and probabilistic interpretation that materially weaken the central claims.

### Major

- **Overstated “no human annotation” story and supervision framing**

  The paper repeatedly claims that REPLM “does not require human annotations of documents” and that the main DocRED experiment is “solely trained without human annotation” (Sec. 5). Formally, this is true in the narrow sense that the in-context pool uses the distantly supervised split, not the human-annotated train/dev splits, and REPLM’s core variant avoids gradient-based training.

  However, the method depends heavily on a large labeled corpus derived from a human-curated KB (DocRED’s distantly supervised split) that is specifically built for DocRED. It also introduces a REPLM (params adj) variant whose hyperparameters are tuned on the human-annotated train set. Moreover, some headline cross-dataset results (Table 4) use models like GPT-3.5/4o that are themselves trained on massive human-annotated corpora.

  This does not invalidate the method, but the narrative that REPLM is “annotation-free” is overstated. A more accurate characterization would be “no *task-specific gradient-based training*; uses distantly supervised labelled data at inference time.” As written, the supervision story can mislead readers about how much labeled structure the system really exploits.

- **State-of-the-art claims conflate methodology with massive model scale and weak supervision**

  The strongest claims—“demonstrate that our framework achieves state-of-the-art performance” (Abstract), “outperforming more than 30 baseline methods” (Abstract, Sec. 7)—are not supported by apples-to-apples comparisons:

  - On DocRED, Table 2’s main comparison is against REBEL / REBEL-sent (~BART-large) with 26–28 F1, while Table 4 lists many fine-tuned document-level and sentence-level methods with F1 up to 68.13 (DocRED-CLiP). REPLM with GPT-JT reaches 35.09 F1 on DocRED, which is far below these strong fine-tuned baselines. The stronger REPLM variants with GPT-3.5/4o and Llama-3.1-70B in Table 4 reach 59.66–68.35 F1, comparable to or slightly above the best fine-tuned models, but they do so with vastly larger backbones and additional weak supervision from the distantly supervised pool.

  - Across datasets, REPLM uses very large LMs (e.g., GPT-4o, Llama-3.1-70B) while many baselines use models in the 100M–400M range trained only on the task-specific training set. The paper does not normalize for model scale or access to distantly supervised examples. Thus, it is unclear how much of the gain comes from the REPLM framework versus simply leveraging much bigger, more data-rich models.

  - Some baselines (CodeIE) are distinguished in Table 4 as not requiring training; REPLM’s advantage over these is more meaningful. But the blanket “outperforms 30+ baselines” headline obscures that many of those baselines reside in a stricter resource regime (much smaller models, no distantly supervised in-context pool).

  Overall, the empirical results show that *given large, modern LMs and access to distantly supervised documents, REPLM can be very strong*, but they do not justify the stronger claim that REPLM as a *method* is decisively superior to strong fine-tuned methods under comparable resource budgets.

- **Evaluation with Wikidata-augmented labels is not a neutral gold standard**

  Section 6.2 augments DocRED dev labels by taking the union of all model predictions, checking each predicted triplet against Wikidata, and adding matched triplets as new “ground truth.” The authors then re-evaluate and report large F1 gains for REPLM over REBEL (Table 3, Fig. 3), claiming that REPLM “actually performs much better than the original labels from the development set.”

  This procedure suffers from two structural biases:

  - It is explicitly conditioned on all systems’ outputs; methods that output many triplets (REPLM averages 20.21 vs REBEL’s 4.93, Sec. 6.1) have a far larger pool of candidates to be “rescued” as true positives via KB matching. Thus, the augmentation inherently favors high-recall / high-output systems like REPLM.

  - The augmented labels are not strictly “truth”: a triplet might be in Wikidata but not be inferable from the document, in which case crediting it as correct conflates “factual correctness in the world” with “entailed by the document.” The paper frames the augmented set as “more closely reflect[ing] the ground-truth” (Fig. 3 caption), but this is not guaranteed.

  The result is that Table 3 and Fig. 3 overstate the strength of REPLM under a “better” ground truth. The qualitative finding that DocRED is incomplete is valid; the quantitative “SOTA under improved labels” claim is not rigorously supported.

- **Probabilistic formulation is conceptually mis-specified**

  In Sec. 4.3, the paper defines

  \[
  p(s \mid C_l,d_i,r) = \text{len}(s)\sqrt{\prod_{k=1}^{\text{len}(s)} p(s_k \mid s_{<k}, C_l, d_i, r)}
  \]
  \[
  p(o \mid s,C_l,d_i,r) = \text{len}(o)\sqrt{\prod_{k=1}^{\text{len}(o)} p(o_k \mid o_{<k}, s, C_l, d_i, r)}
  \]
  and sets
  \[
  p(s,o \mid C_l,d_i,r) = p(s \mid C_l,d_i,r)\, p(o \mid s,C_l,d_i,r).
  \]

  These quantities are not true probabilities: multiplying a scaled geometric mean of token probabilities by length can easily produce values > 1, and there is no normalization over candidate spans. Yet the paper then uses these as “probabilities” within the mixture in Eq. (1) and thresholds them as \(p(s,o \mid d_i,r) > \theta\).

  In practice, they are heuristic scores built from token-level log probabilities. The method likely works well as a ranking-and-thresholding scheme, but the probabilistic interpretation (Eq. (1) as a proper mixture model over context sets) is technically incorrect. The paper should present this explicitly as a scoring heuristic rather than as a valid probability model.

- **Claims about computational efficiency are unsupported and likely misleading**

  The abstract and introduction emphasize that fine-tuning is “computationally expensive” and position REPLM as a computationally lean alternative. However, REPLM’s inference procedure is heavy:

  - For each document–relation pair, it constructs L in-context sets of K examples and runs the LM; on DocRED, where \(|\mathcal{R}|=96\), this implies up to 96 × L LM calls per document (Sec. 4.2–4.3), each with a long prompt concatenating K documents plus triplets.

  - When using large models (Llama-3.1-70B, GPT-4o), this is substantially more expensive at inference time than running a single pass of a fine-tuned smaller model over the document.

  The paper provides no runtime, FLOPs, or wall-clock comparison, so the assertion that baselines suffer “large computational overhead (e.g., from fine-tuning)” while REPLM is cheaper is not substantiated and is likely false for many realistic deployment scenarios. The real trade-off is “less training compute, much more inference compute,” which needs to be acknowledged.

- **Evaluation and claims on “no NER” somewhat understate residual entity-span issues**

  A key selling point is that REPLM avoids explicit NER and its error propagation. It is true that REPLM does not depend on a separate NER pipeline and can, in principle, output arbitrary textual spans. However:

  - The evaluation requires exact string match of subject and object with ground-truth spans (Sec. 5–6), so entity boundary and normalization issues still matter. As the paper’s own analyses on CDR/GDA and NYT highlight, surface-form mismatches (e.g., “complement receptor 1” vs “CR1”) cause false negatives.

  - In practice, REPLM simply pushes entity span detection and normalization into the LM’s generation behavior, rather than eliminating the underlying problem. This is still a meaningful simplification from a system engineering perspective, but the narrative that it “eliminates the error propagation from named entity recognition” (Contributions, Sec. 9) is too strong.

### Minor

- **Learning vs. memorization experiment is interesting but limited**

  The random-entity experiment on CoNLL04 suggests that REPLM is not relying purely on memorized world knowledge to perform RE, which is a nice sanity check. However, it is conducted on a small, sentence-level dataset with limited reliance on external knowledge, and the distantly supervised in-context documents are correspondingly modified. The result supports the claim that the framework can learn extraction patterns, but it does not settle the broader question of how much REPLM leverages memorized KB patterns on more knowledge-heavy datasets like DocRED.

- **Some implementation details are deferred to appendices**

  For reproducibility and clarity, the main text could be more explicit about generation details: decoding strategy (sampling vs. greedy vs. beam), max sequence lengths, and how many triplets are generated per prompt. These appear to be in Appendix E, but high-level summaries in Sec. 4.3 would aid comprehension.

### Trivial

- Occasional small issues like duplicated figure captions and slightly confusing legends (e.g., Fig. 1 repetition, Fig. 3’s two heatmaps) are minor and do not affect the substance.

## Nice-to-Haves

- A zero-shot baseline where the same LMs (e.g., GPT-4o, Llama-70B) are prompted to extract relations without any distantly supervised in-context examples would help quantify the marginal benefit of the REPLM retrieval + aggregation scheme relative to the backbone LM’s raw capability.

- A concrete experiment demonstrating the “new relation type” capability—for example, introducing unseen relations at test time with only a small number of distantly supervised examples and comparing to retraining a fine-tuned model—would directly support one of the most practically interesting claims.

- Reporting macro-F1 in addition to micro-F1 on DocRED and other datasets would help assess performance on rare relations; Fig. 2 suggests heavy dependence on frequent relations, and macro metrics would clarify this.

## Removed Points

These points are flagged to be removed as primary criticisms; treat them with caution, as they either overreach or are already reasonably addressed in the paper.

- **“Using DocRED’s distantly supervised split is tantamount to training on the same dataset and invalidates ‘no-annotation’ claims.”**  
  The method *does* rely on DocRED’s distantly supervised split, which is tuned to the same domain and KB, and this should temper the “annotation-free” narrative. However, the paper is explicit that \(\mathcal{D}^{\text{dist}}\) is distantly supervised and that no gradient-based fine-tuning is performed; it also evaluates on additional datasets where REPLM uses analogous distantly supervised sources. Thus, while the framing is somewhat overstated, it is not accurate to say the work is “misleading” or that the core evaluation is invalid.

- **“Evaluation protocol fundamentally entangles extraction with KB access, so results are unreliable.”**  
  The Wikidata-based augmentation does introduce bias and cannot be treated as a neutral gold standard, but the paper’s main results (Table 2) are still based on the original human-annotated dev set, where REPLM already clearly outperforms REBEL variants. The KB-augmented evaluation should be treated as a supplemental analysis rather than as invalidating the core findings.

- **“Baseline choice in Sec. 5 is inherently unfair because other DocRED models could be rewritten to avoid NER.”**  
  The authors’ decision to restrict Table 2 to methods that do not require explicit NER (REBEL, REBEL-sent) is a defensible scoping choice, especially since they provide a much broader comparison in Table 4. While a stronger baseline set under the same assumptions would be ideal, the existing baselines are not obviously inappropriate.

## Novel Insights

The paper’s most valuable insight is that distantly supervised, relation-specific in-context examples, when retrieved and aggregated carefully, can make large LMs into competitive document-level relation extractors without explicit NER or fine-tuning, and that such systems systematically expose incompleteness and normalization issues in widely used benchmarks like DocRED, CDR, GDA, and NYT. This synthesis of ICL-based RE with distantly supervised retrieval and cross-dataset analysis is more than a straightforward application of prompting; it lays out a practical template for leveraging future LMs for RE while simultaneously stress-testing existing datasets.

## Suggestions

- Reframe the supervision narrative to clearly state: (i) the method uses distantly supervised labeled data as in-context examples; (ii) the “params adj” variant uses human-annotated training data for hyperparameter tuning; and (iii) the main benefit is avoiding task-specific gradient-based training, not being fully label-free.

- Temper state-of-the-art claims by explicitly acknowledging the role of massive backbone LMs and weak labels, and by clearly separating (a) comparisons within a fixed backbone family (e.g., GPT-JT with and without REPLM) from (b) comparisons across very different model scales. Highlight that REPLM provides a *framework* that can reach or surpass SOTA when coupled with strong LMs, rather than claiming the framework itself universally dominates.

- Present the scoring function in Sec. 4.3 as a heuristic constructed from token log probabilities, and avoid probabilistic language that implies a properly normalized mixture model. Clarify that the method empirically works as a ranking and thresholding scheme, without overclaiming probabilistic soundness.

- Add some form of computational cost analysis: approximate number of tokens processed and LM calls per document–relation pair, and compare with representative fine-tuned baselines’ training and inference costs. This will make the practical trade-offs transparent.

- Include macro-F1 and possibly per-frequency-band performance on DocRED to complement micro-F1 and Fig. 2. This will help readers assess how well REPLM handles rare relations.

- Consider adding a zero-shot baseline and/or a same-backbone fine-tuning baseline (e.g., fine-tuned Llama-3.1-8B vs REPLM with Llama-3.1-8B) to better isolate the added value of the REPLM framework itself.

## Score and Decision

**Calibration references:**

- **PromptNER (WDQ9ZzsgDL.md)** — LLM prompting for NER, solid empirical work but limited novelty, unfair comparisons to much smaller models; scores: 3, 5, 3, 3 (decision: Reject).  
- **When Does ICL Fall Short (Cw6lk56w6z.md)** — systematic study of ICL limits on IE-like tasks, methodologically interesting but with weaknesses; scores centered around 5 (decision: Reject).  
- **GoLLIE (Y3wpuxd7u9.md)** — guideline-following LLM for IE with stronger methodological and empirical grounding; scores: 8, 5, 6, 6 (accepted as poster).  
- **Bio-RFX (KskgLM728l.md)** — relation extraction with structural constraints; technically sound but with various experimental and positioning issues; scores: 5–6 (decision: Reject).

Relative to these:

- REPLM has **stronger empirical breadth and more systematic ablations** than PromptNER and IntentGPT, and a clearer, more impactful application domain (document-level RE). It is also more ambitious in benchmarking.
- However, like PromptNER and Cw6lk56w6z, it **overstates novelty and SOTA** and relies on comparisons across very different model scales, without sufficient resource calibration.
- It does not reach the conceptual and empirical maturity of GoLLIE, which had clearer methodological innovations and a more carefully controlled evaluation.

Overall, I place this paper above weaker rejected ICL/LLM-IE works (which often received ~3–4) but below borderline-accept posters like GoLLIE (~6–7). The core idea is promising and the experiments are substantial, but the framing and evaluation issues are serious enough that I would not recommend acceptance this round.

MY FINAL SCORE: <pineapple>5.5</pineapple>  
MY FINAL DECISION: <orange>Reject</orange>