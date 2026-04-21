Now I have a thorough understanding of the paper and the calibration anchors. Let me compose the final review.

## Summary

RoRA-VLM introduces a retrieval-augmented framework for vision-language models (VLMs) targeting knowledge-intensive VQA tasks. It proposes two main innovations: (1) a two-stage retrieval process that uses the query image as a visual anchor to retrieve similar images, extracts entity names to expand the textual query, and then performs a second-stage text retrieval via Google Search; and (2) a noise-resilient retrieval-augmented generation method combining adversarial noise injection (adding irrelevant knowledge snippets during training) and query-oriented visual token refinement (selecting the most relevant image patches via CLIP similarity). Evaluated on OVEN, InfoSeek, and Enc-VQA, the 7B-parameter model outperforms larger baselines including the prior SOTA Wiki-LLaVA.

## Strengths

- **Strong empirical improvements over larger models**: RoRA-VLM (7B) outperforms PaLI-X (55B) and PaLI-17B on InfoSeek (25.10/27.34 vs. 20.80/23.50 and 16.00/20.70), and surpasses Wiki-LLaVA on nearly all benchmarks (Table 1). The gains are consistent and substantial.

- **Well-motivated two-stage retrieval design**: The image-anchored entity retrieval → text-query expansion pipeline directly addresses the anaphoric reference problem in knowledge-seeking VQA (e.g., "this building" → "Castle of Good Hope"). Table 5 provides retrieval precision numbers (35–39% first stage, 27% second stage), and the 11.52% improvement over single-stage retrieval (reported in the introduction) is compelling.

- **Visual token refinement is properly ablated**: Table 2 shows that removing visual token refinement causes performance drops of 0.62% (Entity) and 1.48% (Query) on InfoSeek. Figure 3 provides strong qualitative evidence that the refinement correctly clusters on the key entity in matching images and scatters on mismatched images.

- **Knowledge-intensive pre-training analysis is thorough**: Table 3 cleanly isolates the benefit of entity-rich WikiWeb2M pre-training over generic ShareGPT4V and no pre-training (24.56/26.33 → 21.28/22.84 → 20.68/23.41), confirming that domain-relevant pre-training matters beyond generic visual instruction tuning.

- **Attention visualizations support the noise-resilience claim**: Figure 4 shows the model assigns higher attention to textual knowledge corresponding to images with matching entities, providing mechanistic evidence for selective knowledge utilization.

## Weaknesses

### Fatal
None.

### Major

- **Missing ablation for adversarial noise injection — one of the paper's two core contributions lacks direct empirical validation.** The paper presents adversarial noise injection as a key component of its noise-resilient generation method, claiming that "by training with adversarial noise, VLMs implicitly learn to compare the visual appearances of entities" (Section 5, lines 408–412). However, the only related ablation in Table 2 ("text-only RAG") removes *all* retrieved images from both training and inference, conflating the value of having images at all with the value of the adversarial training strategy. A proper ablation would train the full pipeline with retrieved images present but *without* injecting irrelevant snippets during training, then compare against the full model. Without this condition, there is no direct evidence that the adversarial noise injection training strategy itself contributes anything beyond the model simply having access to retrieved images during training. This gap partially undermines one of the paper's two claimed contributions.

- **Wiki-LLaVA comparison relies on an unvalidated reimplementation with narrow performance margins.** The paper discloses (Table 1 footnote) that Wiki-LLaVA's source code is not publicly available and marks it with an asterisk. No validation is provided — the reimplemented numbers are not compared against those reported in the original Wiki-LLaVA paper. The margins are narrow: only 0.65% on OVEN Entity (15.08 vs 14.43) and 3.66% on InfoSeek Query (27.34 vs 23.68). Even small reimplementation differences could affect whether the "constantly outperform SOTA" claim holds. This undermines the paper's central comparison claim, especially given that Wiki-LLaVA is the only directly comparable 7B retrieval-augmented VLM baseline.

- **Google Search as the second-stage retrieval backend introduces non-reproducibility and potential unfair advantage.** Section 3.2 and footnote 2 disclose that Stage 2 uses Google Search via the Serper API. This is proprietary, non-deterministic, and far more powerful than the fixed Wikipedia-based indexes used by baselines like Wiki-LLaVA and PreFLMR. The performance gains attributed to the two-stage retrieval design could partially derive from having a superior retrieval engine rather than from the image-anchored query expansion strategy itself. The appendix ablation for single-stage retrieval (A.5) does not control for the retrieval backend, making it impossible to disentangle the method's contribution from the search engine's capability.

### Minor

- **"Minimal training instances" claim is misleading.** The abstract states "with a minimal amount of training instance" and Section 5 says "fine-tuned on less than 10,000 instances per dataset" (line 136), but Section 4 reveals a 1M-instance pre-training phase on WikiWeb2M before the 1,000-instance fine-tuning. The actual total training data is ~1M+ instances, not 10K. While the fine-tuning step is lightweight, the framing obscures the full data requirements.

- **"Zero-shot domain transfer" claim is overstated.** Table 4 shows a leave-one-category-out experiment (training on 10 of 11 iNaturalist categories, testing on "Insect"). This is few-shot cross-category transfer at best — the model has seen closely related categories like "Amphibian," "Plant," and "Bird" during training. True zero-shot domain transfer would involve an entirely unseen domain with no related training categories.

- **Potential data overlap between training/retrieval sources and evaluation benchmarks.** WIT, WikiWeb2M, and the evaluation benchmarks (OVEN, InfoSeek, Enc-VQA) are all Wikipedia-derived. The paper does not analyze whether pre-training or retrieval data contains information that directly answers test questions, which is essential for interpreting benchmark results fairly.

- **The adversarial noise injection uses a weak form of adversarial sampling.** Footnote 3 states the irrelevant snippet is sampled by ensuring the entity is "mismatched with the target entity," but this does not guarantee visual difficulty. A more adversarial setting would include visually similar but semantically different entities (e.g., different penguin species), which would better test the model's ability to discriminate. The current strategy may be too easy, inflating the perceived effectiveness of noise resilience.

### Trivial
- Equation 4 uses subscript $i$ inside the sum $\sum_{i=1}^m$, which shadows the outer index $i$ for retrieved images $\tilde{I}_i$. The meaning is discernible but the notation is inconsistent.

## Nice-to-Haves

- Report results with a reproducible retrieval backend (e.g., BM25 over Wikipedia, or the same CLIP-based retrieval used in Stage 1) as the primary comparison, with Google Search reported as an upper bound.
- Validate the Wiki-LLaVA reimplementation by comparing its numbers on a shared benchmark against those reported in the original paper.
- Add a direct ablation for adversarial noise injection: train with images but without noise injection, and compare.
- Analyze failure cases of the two-stage retrieval and visual token refinement.
- Report standard deviations across multiple runs, especially for narrow margins.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Baselines not all fine-tuned the same way"** — The Harsh Critic claims it is implausible that PaLI-17B and PaLI-X were fine-tuned by the authors. While this is a reasonable suspicion, the paper states "all the baseline models are fine-tuned on the OVEN, InfoSeek, and Enc-VQA datasets respectively" and we have no direct evidence this is false. Some numbers may indeed come from original papers with compatible fine-tuning. The concern is speculative without proof.

- **"CLIP patch features not well-calibrated for fine-grained entity matching"** — This is a reasonable theoretical concern, but the paper provides empirical evidence that the approach works (Table 2 ablation showing improvement, Figure 3 qualitative results). The empirical results directly address this concern.

- **"No standard deviations or confidence intervals"** — Reporting confidence intervals is not standard practice for large-scale VQA benchmarks in this community. This is a nice-to-have, not a substantive weakness.

- **"Missing appendix, missing proofs in appendix"** — The parser strips appendices; the original submission includes them. Removed per hard rules.

- **Pure formatting/style nitpicks and notation issues beyond the Equation 4 shadowing** — Removed as trivial or parser artifacts.

- **Strength Finder's claim of "Effective adversarial noise injection validated by text-only RAG ablation"** — This strength is dropped because it mischaracterizes what the text-only RAG ablation actually validates. That ablation shows images are important, not that adversarial noise injection specifically is effective. This conflicts with the verified major weakness about missing adversarial noise injection ablation.

- **Strength Finder's claim of "Zero-shot domain transfer capability"** — This strength is dropped because the experiment is leave-one-category-out, not zero-shot, which conflicts with the verified minor weakness about the overstated claim.

## Novel Insights

The paper makes an important architectural observation that is easy to overlook: in knowledge-seeking VQA, the anaphoric reference problem ("this building," "this animal") creates a fundamental asymmetry between what the visual modality provides (entity identity) and what the text modality provides (information need). The two-stage retrieval design exploits this asymmetry by using the image as an entity anchor first, then expanding the text query — this is a principled decomposition of the multimodal query that goes beyond simply concatenating image and text features for joint retrieval. However, the paper's evaluation does not cleanly separate this insight's contribution from the contribution of using a more powerful search engine (Google Search) in the second stage.

## Suggestions

- **Most critical**: Add a direct ablation for adversarial noise injection — train the full pipeline with retrieved images present during training and inference but *without* injecting irrelevant snippets. This is the single experiment that would most strengthen the paper.
- Report results using a reproducible retrieval backend for Stage 2 to disentangle the method's contribution from Google Search's capability.
- Validate the Wiki-LLaVA reimplementation by reproducing at least one result from the original paper.
- Reframe "zero-shot domain transfer" as "cross-category generalization" to avoid overclaiming.
- Clarify the training data requirements by explicitly mentioning the 1M pre-training phase alongside the fine-tuning claims.

## Score and Decision

**Calibration anchors:**
- MMed-RAG (avg 7.0, Accept Poster): Similar RAG-for-VLM paper with thorough ablations and theoretical grounding. RoRA-VLM is weaker due to incomplete ablation coverage and unvalidated baseline.
- Self-RAG (avg 7.5, Accept Oral): Much stronger methodological contribution with clean self-reflection mechanism and comprehensive evaluation. RoRA-VLM is clearly below this.
- RepARe (avg 6.0, Accept Poster): VLM VQA improvement with solid but incremental contribution. RoRA-VLM is comparable in empirical strength but weaker in evaluation rigor.
- R-VLM (avg 4.75, Reject): Simpler retrieval-based video QA with limited evaluation. RoRA-VLM is stronger with more components and better results.
- EDU-RAG (avg 2.33, Reject): Very weak RAG benchmark with shallow analysis. RoRA-VLM is much stronger.
- Reward-RAG (avg 3.0, Reject): Unfair comparison concerns, lack of novelty. RoRA-VLM is stronger but shares some unfair comparison concerns.

RoRA-VLM sits below RepARe (6.0) due to the missing adversarial noise injection ablation undermining a core contribution, the unvalidated Wiki-LLaVA reimplementation, and the Google Search dependency. It sits above R-VLM (4.75) because the empirical results are stronger and the framework addresses a well-defined problem with multiple well-motivated components. The paper makes real contributions but the evaluation gaps are significant enough to raise doubts about whether both claimed innovations actually work as described.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>