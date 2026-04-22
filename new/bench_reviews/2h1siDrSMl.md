Now I have a thorough understanding of the paper and the reviewer claims. Let me synthesize my final review.

Key findings from verification:

1. **Google Search as Stage-2 retriever**: Confirmed — Section 3.2 explicitly states "query-expanded text retrieval with a Google Search engine" via the Serper API (footnote 2). The comparison with baselines like Wiki-LLaVA that use Wikipedia-based retrieval does create a fairness concern. However, the CRAG paper that was flagged for similar concerns (using web search while baselines don't) got avg score 3.75. The difference is that CRAG's core contribution was the web search itself (making it a bigger issue), while RoRA-VLM also has substantive non-retrieval contributions (noise injection, token refinement). The fairness concern is valid but the magnitude of its impact needs to be contextualized.

2. **Adversarial noise injection ablation**: This is a legitimate concern. The "text-only RAG" ablation (Table 2) removes ALL images from both training and inference — it doesn't isolate the effect of the noise injection during training. A proper ablation would compare "RoRA-VLM with adversarial noise injection" vs "RoRA-VLM without adversarial noise injection" (keeping retrieved images but without the randomly sampled irrelevant document). The paper does mention Appendix A.4 for "varying levels of retrieval noise" but we can't see it.

3. **"Minimal training" claim**: The abstract says "with a minimal amount of training instance" but Section 5 says "fine-tuned on less than 10,000 instances per dataset." However, this "10,000" refers to the fine-tuning stage only. There's a 1 million instance pre-training on WikiWeb2M (Section 4). The paper does disclose this in Section 4, but the abstract and the claim at line 136 ("fine-tuned on less than 10,000 instances") is misleading because it only counts the fine-tuning data while omitting the pre-training data.

4. **"Adversarial" terminology**: Footnote 3 says "We randomly sample an entity from our retrieval database" — this is random negative sampling, not adversarial in the optimization sense. The terminology is inflated.

5. **"Zero-shot domain transfer"**: The evaluation removes one category ("Insect") out of 11 from training while keeping the same dataset structure. This is held-out-category evaluation, not zero-shot transfer to a truly unseen domain.

6. **No variance/statistical significance**: Tables report single numbers. This is a fair concern but is standard practice in the field.

7. **Wiki-LLaVA reimplementation**: Marked with * as authors' own implementation. This adds some concern about fairness but the paper discloses it.

Let me now check the high-scoring anchor more carefully and decide on scoring.

For calibration:
- High anchors (7-8): MMed-RAG (7.0, similar multimodal RAG with overclaimed novelty concerns but accepted), Inference Scaling for RAG (8.0), Self-RAG (7.5)
- Medium anchors (4-6): R-VLM (4.75, retrieval-augmented VLM but with overclaimed contributions), EDU-RAG (similar Google Search issue, avg 2.33), VLB benchmark (6.25)
- Low anchors (<3): EDU-RAG (avg 2.33 for Google Search reproducibility), pLvh9DTyoE (2.5, limited contribution)

RoRA-VLM is better than EDU-RAG (which was purely a benchmark paper with minimal contribution). It has substantive technical contributions beyond the retrieval backend. But the Google Search unfairness and missing ablation for adversarial noise injection are significant.

Compared to MMed-RAG (7.0): MMed-RAG also had "overclaimed novelty" concerns and missing baselines but had solid ablations. RoRA-VLM has a more serious fairness concern (Google Search vs Wikipedia) and a missing critical ablation.

I'd place this around 5.5 — it has real contributions and strong results, but the Google Search fairness issue and the unaatablated noise injection prevent it from scoring higher.</think>Now I have enough information to write the final review. Let me also quickly verify the Strength Finder's key claim about Table 2 isolating adversarial noise injection:

The Strength Finder claims the text-only RAG ablation (Table 2) "isolates the contribution of the adversarial noise injection strategy." But this is incorrect — the text-only RAG ablation removes ALL images from both training and inference, not just the adversarial noise. This conflates two things: (1) the presence of images during inference and (2) the adversarial noise injection during training. Without an ablation that keeps images but removes only the noise injection, we cannot isolate its contribution.

## Summary

RoRA-VLM proposes a retrieval-augmented framework for VLMs on knowledge-intensive VQA tasks, with a 2-stage retrieval process that uses image-anchored entity names to expand text queries, and a noise-resilient training/generation approach combining adversarial noise injection during training and query-oriented visual token refinement during inference. The system outperforms prior retrieval-augmented VLMs on OVEN, InfoSeek, and Enc-VQA benchmarks using a 7B parameter model.

## Strengths

- **Strong empirical results with a small model**: RoRA-VLM (7B) outperforms much larger baselines including PaLI-X (55B) on InfoSeek (25.10 vs 20.80 Entity; 27.34 vs 23.50 Query) and surpasses Wiki-LLaVA across nearly all benchmarks (Table 1). This is notable given the model size disparity.

- **Well-motivated problem formulation**: The two challenges identified — modality discrepancy in retrieval queries and noise in retrieved multimodal knowledge — are genuine and under-addressed in the VLM RAG literature. The 2-stage retrieval concept (using image-matched entities to expand text queries) is a sensible and clearly illustrated approach (Figure 5).

- **Effective visual token refinement with clear qualitative evidence**: The query-oriented visual token refinement (Eq. 3–4) shows consistent improvement in ablation (Table 2: +0.62 Entity, +1.48 Query) and the qualitative visualizations in Figure 3 effectively demonstrate that the refinement concentrates on entity-relevant patches while scattering on mismatched retrieved images.

- **Informative pre-training analysis**: Table 3 shows meaningful comparisons — WikiWeb2M pre-training improves LLaVA-v1.5 from 10.34 to 18.00 on InfoSeek Entity, and further that entity-rich WikiWeb2M substantially outperforms generic ShareGPT4V pre-training (24.56 vs 21.28). This provides useful practical insights for the community.

- **Attention visualizations support the noise-resilience claim**: Figure 4 provides token-level attention heatmaps showing that RoRA-VLM learns to attend to knowledge snippets whose images match the query entity during generation, directly illustrating the model's learned noise discrimination.

## Weaknesses

### Fatal
None

### Major

- **Unfair comparison with baselines due to Google Search as Stage-2 retriever**: The 2-stage retrieval delegates Stage 2 to Google Search via the Serper API (Section 3.2, footnote 2), giving RoRA-VLM access to web-scale, continuously updated information. In contrast, the primary retrieval-augmented baselines (Wiki-LLaVA, PreFLMR) retrieve from fixed Wikipedia/WIT-based knowledge bases. The claimed performance improvements over these baselines (e.g., +3.66% and +3.69% over Wiki-LLaVA on InfoSeek) may be partly or largely attributable to the retrieval backend advantage rather than the proposed methodological contributions. Without a same-backend comparison (e.g., rerunning RoRA-VLM with Wikipedia-only Stage-2, or rerunning baselines with Google Search), the paper cannot validly attribute the gains to the 2-stage pipeline or noise-resilient generation alone. This is a structural design issue in the experiments, not an easily fixable presentation concern. (The CRAG paper, which had a similar web-search advantage over baselines, received avg score 3.75 from human reviewers for exactly this concern.)

- **Adversarial noise injection is never ablated in isolation**: The adversarial noise injection is identified as one of two key innovations (Abstract, Section 3.3), yet Table 2 only provides two ablations: removing VK-Refinement and removing all retrieved images (text-only RAG). The text-only RAG ablation simultaneously removes images from both training AND inference, conflating the training-time noise injection with the inference-time availability of visual context. There is no comparison between RoRA-VLM trained with adversarial noise injection vs. RoRA-VLM trained without it (keeping retrieved images but removing the randomly sampled irrelevant document). Without this ablation, there is no direct evidence that the adversarial noise injection strategy itself contributes to performance — a claimed central contribution remains unsubstantiated. (The paper mentions Appendix A.4 for "varying levels of retrieval noise," which may partially address this, but the key baseline of zero noise injection is not reported in the main text.)

### Minor

- **Misleading "minimal training" framing**: The abstract claims "with a minimal amount of training instance" and Section 5 says "fine-tuned on less than 10,000 instances per dataset." These statements refer only to the fine-tuning stage, omitting the prerequisite visual-knowledge alignment pre-training on 1 million WikiWeb2M instances (Section 4). Table 3 shows this pre-training alone accounts for the majority of improvement on InfoSeek (LLaVA-v1.5: 10.34 → LLaVA-v1.5 w/ WikiWeb2M: 18.00, a +7.66 absolute gain). The "minimal training" claim, while technically accurate for fine-tuning, is misleading about the overall data requirements. The paper does disclose pre-training in Section 4, but the abstract's framing obscures it.

- **"Adversarial" noise injection terminology is inflated**: Footnote 3 states "We randomly sample an entity from our retrieval database, together with its image and corresponding knowledge, as the irrelevant sample." This is random negative sampling, not adversarial optimization. The term "adversarial" typically implies gradient-based or optimization-driven worst-case perturbation. While the method is sound, the terminology overstates the nature of the contribution.

- **"Zero-shot domain transfer" claim is overstated**: Table 4 evaluates by removing the "Insect" category from the iNaturalist subset of Enc-VQA (1 of 11 categories) while training on the remaining 10 categories from the same dataset. This is a held-out-category evaluation within the same dataset and task format, not zero-shot transfer to a genuinely unseen domain. The model still benefits from the same question format, answer structure, and image distribution characteristics.

- **Non-reproducibility due to Google Search**: Google Search results are nondeterministic and change over time, meaning future researchers cannot replicate the reported numbers. While this is a practical concern, it is somewhat secondary to the fairness issue since the same backend could in principle be replaced with a frozen snapshot.

### Trivial
None

## Nice-to-Haves

- Ablation comparing RoRA-VLM with vs. without adversarial noise injection (keeping all other components including retrieved images) to directly validate this claimed contribution
- Same-backend comparison (e.g., rerun RoRA-VLM with Wikipedia-only Stage-2 retrieval, or give baseline methods access to the same Google Search backend) to isolate methodological vs. retrieval backend contributions
- Error analysis conditioned on Stage-1 retrieval success/failure to reveal whether the noise-resilient generation truly handles retrieval failures or simply benefits from high Stage-1 accuracy
- Scaling analysis of the noise injection (varying number of irrelevant documents: 0, 1, 2, 4)

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Wiki-LLaVA reimplementation concern (doubly unfair)**: The harsh critic claims Wiki-LLaVA reimplementation combined with Google Search creates a "doubly unfair" comparison. While the self-implementation of baselines is worth noting, the paper discloses it (Table 1, *). The primary fairness concern (Google Search vs Wikipedia) is already captured above. Adding a secondary complaint about reimplementation is piling on without adding new substance.

- **Comparison with non-retrieval models (PaLI, BLIP-2, InstructBLIP) inflates improvement**: These are included as scale references, not as direct competitors for the retrieval-augmented setting. Including both retrieval and non-retrieval baselines is standard practice and helps readers understand the value of retrieval augmentation itself.

- **Equation 4 notational ambiguity**: The harsh critic claims subscript i in the sum references tokens from both $\tilde{\mathbf{X}}_I$ and the selection criterion. Looking at Eq. 4 more carefully, the notation is consistent — $\sum_{i=1}^m$ sums over the already-selected query image tokens to compute similarity scores for retrieved image tokens. This is a minor notation choice, not a substantive error.

- **Missing comparison with token pruning methods (ToMe, etc.)**: Token pruning for efficient inference is a different research direction. The paper's goal is selecting relevant tokens for noise reduction, not generic token pruning for efficiency. This is scope creep.

- **No error bars or statistical significance tests**: While true, single-run reporting is standard practice in this research community for large-scale VLM benchmarks. This is a generic concern that applies to virtually all papers in the field.

- **Strength Finder's claim that "text-only RAG ablation isolates adversarial noise injection contribution"**: This is factually wrong. The text-only RAG ablation removes all images from both training and inference — it cannot isolate the effect of noise injection alone. Moved because it mischaracterizes what the ablation demonstrates.

## Novel Insights

The most interesting observation that emerges from combining the paper's results is that the relative contributions of the three components (retrieval backend, pre-training data, and noise-resilient generation) appear to follow a clear hierarchy. Table 3 shows pre-training alone gives +7.66 on InfoSeek Entity; Table 2 shows that removing VK-Refinement costs only ~0.62; and the text-only RAG ablation shows a large drop of ~7.3. This suggests that the visual-knowledge alignment pre-training and the presence of images during inference are the dominant factors, while the token refinement provides incremental improvement. The critical unresolved question is where adversarial noise injection falls in this hierarchy — the paper provides no data point to answer this.

## Suggestions

- Add a direct ablation of adversarial noise injection: compare RoRA-VLM (with noise injection) against the same system trained without the randomly sampled irrelevant document. This is the single most important experiment missing from the paper.
- To address the Google Search fairness concern, either (a) run RoRA-VLM with a fixed Wikipedia-based text retriever for Stage 2 and report those results alongside the Google Search results, or (b) clearly separate and discuss the contribution of the retrieval backend versus the model-level innovations.

## Evaluation

**Originality**: The 2-stage image-anchored retrieval and query-oriented visual token refinement are reasonable contributions. However, the "adversarial noise injection" is essentially random negative sampling dressed in adversarial terminology, reducing the perceived novelty. The overall framework design is sensible but the novelty of individual components is moderate.

**Importance of research question**: High. Knowledge-intensive VQA for VLMs is a timely and significant problem with clear practical implications.

**Claim support**: Partially supported. The main results are strong but the attribution of gains to methodological contributions vs. retrieval backend is confounded. The adversarial noise injection claim is unsubstantiated by a proper ablation.

**Soundness of experiments**: The experiments are incomplete in critical ways — the missing adversarial noise injection ablation and the Google Search fairness issue undermine the ability to draw firm conclusions from the comparisons.

**Clarity**: Generally well-written with clear problem formulation and good qualitative visualizations. The "minimal training" framing and "adversarial" terminology are misleading.

**Value to community**: Moderate. The framework and qualitative analysis provide useful insights, but the reproducibility concerns and confounded comparisons limit the reliability of the reported benefits.

## Calibration Anchors

| Paper | Avg Human Score | Comparison |
|-------|----------------|------------|
| MMed-RAG (s5epFPdIW6) | 7.0 | Multimodal RAG for VLMs with "overclaimed novelty" and missing baselines, but solid ablations. RoRA-VLM has similar overclaiming issues but with more serious fairness concerns (Google Search) and a missing critical ablation. Below this anchor. |
| Inference Scaling for RAG (FSjIrOm1vz) | 8.0 | Strong systematic study with extensive experiments on knowledge-intensive RAG. Much more thorough methodology and ablation than RoRA-VLM. Well above. |
| CRAG (JnWJbrnaUE) | 3.75 | Used large-scale web searches as retrieval augmentation while baselines didn't. Flagged for unfair comparison and limited novelty. RoRA-VLM shares the web-search fairness issue but has substantially more non-retrieval technical contributions (noise injection, token refinement, pre-training analysis). Above this anchor. |
| R-VLM (rGk0ur4Tfr) | 4.75 | Retrieval-augmented VLM with limited novelty over baselines and missing ablations. RoRA-VLM has more substantial contributions but shares the incomplete ablation problem. Slightly above. |
| EDU-RAG (a2rSx6t4EV) | 2.33 | Used Google Search for retrieval; flagged for search bias and non-reproducibility, plus shallow analysis. RoRA-VLM has much deeper technical contributions and analysis. Well above. |
| Low-resource multimodal NER (pLvh9DTyoE) | 2.50 | Very limited contribution and poor evaluation. RoRA-VLM is much stronger. Well above. |

RoRA-VLM sits between R-VLM (4.75, with incomplete ablations but fewer fairness issues) and MMed-RAG (7.0, with overclaimed novelty but solid ablations). The Google Search fairness issue is significant (similar to CRAG at 3.75), but RoRA-VLM has substantially more non-retrieval technical content than CRAG. The missing ablation for adversarial noise injection is a clear gap that undermines a claimed contribution. I place this at **5.5** — the paper has real contributions and strong empirical results, but the fairness and ablation gaps prevent confident attribution of gains to the claimed innovations.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>