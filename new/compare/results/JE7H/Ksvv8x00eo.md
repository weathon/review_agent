---
job_id: 1e9fd22e-5901-4dba-bd1e-de000d3d347b
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: Ksvv8x00eo.pdf
paper: CaTS-BENCH: Can Language Models Describe Numeric Time Series?
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper introduces a multimodal benchmark and evaluation suite for time-series captioning and reasoning with foundation models, squarely within ICLR’s “datasets and benchmarks”, multimodal representation learning, and numeric reasoning topics.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Methodology/Benchmark description, Experiments, Results/Analysis, Conclusion) are present and reasonably detailed. The work is technically sound overall, with non-trivial experimental evaluation and clear methodology.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see attempts to manipulate LLM reviewers or hidden instructions beyond the normal scientific content.


---

# Expected Review Outcome:

## Summary

The paper introduces CaTS-Bench, a multimodal benchmark for “context-aware time series captioning” and related Q&A tasks. It aggregates 11 real-world datasets (climate, health, crime, border crossing, etc.), applies a standardized windowing pipeline, attaches metadata, generates plots, and uses an “oracle” LLM (Gemini 2.0 Flash) to produce semi-synthetic captions, plus a small set (579) of human-revisited test captions. The authors also define numeric-focused evaluation metrics and a suite of 460 multiple-choice questions, and benchmark a wide range of VLMs (proprietary and open-source, pretrained and finetuned), including a PAL variant, analyzing their failures on numeric reasoning and utilization of visual inputs.

## Strengths

1. **Substantive, carefully engineered benchmark with rich multimodality.**  
   CaTS-Bench systematically combines numeric sequences, metadata, plots, and captions across 11 heterogeneous domains (Table 2, **Page 4**). The dataset is non-trivial in scale (20k samples, ~570k time steps; ~4k test samples; plus 460 Q&A items) and goes beyond trend labels or synthetic templates by using expressive, multi-sentence captions and realistic metadata. The samples shown in **Figures 19–23** (Appendix L) make clear that the provided captions are fairly detailed and numerically grounded, and the metadata is genuinely informative (e.g., all-time stats, region, sampling frequency).

2. **Thoughtful semi-synthetic caption generation and validation pipeline.**  
   The captioning pipeline in **Figure 2** (**Page 4**) is more careful than typical “LLM generates everything” approaches: the oracle sees the full series plus precomputed statistics (mean, std, min, max) and metadata but is explicitly instructed not to use external knowledge. The authors then run substantial quality checks: manual verification of ~2.9k test captions with 98.6% accuracy over stats and trends (**Table 9**, **Page 21**), a human indistinguishability test with ~41% accuracy at telling human vs. LLM captions, and multiple diversity/embedding analyses (**Tables 12–14**, **Pages 23–24**). This gives reasonable evidence that the semi-synthetic captions are factually and stylistically usable as references.

3. **Additional human-revisited subset and robustness checks on evaluation bias.**  
   The “HR” subset (579 test captions across agriculture, crime, demography, Walmart) is manually refined from multiple LLM candidates. The authors explicitly evaluate all models against both semi-synthetic (SS) and human-revisited (HR) ground truths (**Table 3**, **Page 7**, and **Tables 7–8**, **Pages 19–20**), and they run a paraphrasing robustness study (**Section H.3**, **Table 10**, **Figure 5**). The high Spearman rank correlations in **Table 11** (avg 0.9266) suggest that the metrics are not severely biased toward Gemini’s particular phrasing, which helps mitigate the concern that Gemini both generates and is evaluated on its own captions.

4. **Non-trivial, numeric-focused evaluation metrics and detailed analysis.**  
   Beyond standard DeBERTaScore, BLEU, ROUGE-L, METEOR, and SimCSE (**Section 3.5**, **Appendix F**, Equations (1)–(5)), the paper introduces two numeric metrics: (i) “Statistical Inference Accuracy” for mean / std / min / max with 5% tolerance; (ii) a “Numeric Score” that aligns generated numbers to ground truth with a weighted combination of accuracy and recall. While somewhat heuristic, this is a welcome step toward measuring numeric fidelity, and the analysis around **Table 4** and **Tables 7–8** is informative in separating “fluent but vague” from “fluent and numerically grounded” captions.

5. **Diagnostic Q&A tasks that expose model weaknesses in time-series reasoning.**  
   The 460 Q&A items span time series matching, caption matching, plot matching, and pairwise comparisons (amplitude, peak, mean, variance). **Figure 3** and **Table 17** show that even strong proprietary models like GPT-4o and Gemini 2.0 Flash often fail on plot matching (mostly near random) and are far below human performance on several tasks. The construction of stronger distractors, especially the artificially perturbed time series / captions (**Section 3.4**, **Section J.3**, **Table 19**), is a nice design choice that meaningfully increases difficulty.

6. **Clear evidence that current VLMs underuse visual inputs for time series plots.**  
   The visual ablation results in **Table 15** (**Page 25**) and summarized in **Figure 4** (**Page 9**) are striking: for most models, dropping the plot and keeping only numeric values + metadata either barely hurts or even *improves* metrics, indicating that the “vision” side is mostly unused or noisy. The attention visualization in **Figure 7** shows that LLaVA mostly attends to textual elements (axes, labels) rather than line geometry. The experiments with GAF and recurrence plots (**Figure 8**, **Table 16**) further support the conclusion that the bottleneck lies in current multimodal architectures, not just the choice of encoding. This is a solid, data-backed insight.

7. **Breadth of model baselines, including PAL.**  
   The benchmark includes a broad set of VLMs (Gemini 2.x, GPT-4o, Claude, LLaVA variants, InternVL 2.5, Qwen-VL, Phi-4 M.I., Idefics 2, SmolVLM, Llama 3.2 V, Gemma 3, etc., **Appendix E**) and evaluates both pretrained and finetuned versions. The PAL setup for Qwen-VL (Section 4.1, **Table 4**) is particularly interesting: it clearly boosts statistical inference accuracy (mean / max / min around 0.97–0.98), illustrating that explicit program execution is helpful for numeric-heavy captioning.

8. **Presentation quality and transparency.**  
   The paper is generally well written, highly structured, with explicit prompts in Appendix N, detailed dataset rules (Table 5), and training setups (Table 6). Figures like **Figure 1** (overview of tasks and domains) and **Figure 3** (radar plots of Q&A accuracies) are informative and match the claims in the text. The authors are unusually explicit about limitations and ethical considerations in Sections A and the Ethical Statement.

## Weaknesses

1. **Positioning vs. closely related recent benchmarks is incomplete, hurting novelty claims.**  
   While Table 1 compares CaTS-Bench with TADACap, TRUCE, and TACO, the paper does not mention several directly relevant recent works on LMs describing time series. In particular, “BEDTime: A Unified Benchmark for Automatically Describing Time Series” and “Can Language Models Infer Event Descriptions from Time Series?” (both by Sen et al., 2025) are absent from Section 2, even though they target extremely similar tasks (automatic NL descriptions of time series, event inference). This omission weakens the claim that CaTS-Bench is “the first large-scale, multimodal benchmark explicitly designed for context-aware time series captioning and reasoning” (Abstract and **Page 2**) and makes it difficult to gauge incremental vs. orthogonal contributions. The authors need to (i) explicitly compare dataset design choices (domains, label types, modality coverage) to these benchmarks; (ii) clarify what CaTS-Bench enables that BEDTime and similar efforts do not; and (iii) revise overstated “first” language.

2. **Heavy reliance on a single proprietary oracle for ground-truth captions and potential evaluation entanglement.**  
   Almost all semi-synthetic captions are generated from Gemini 2.0 Flash, which is also evaluated as a baseline. Although the authors perform paraphrasing-based robustness checks (**Section H.3**, **Table 10**, **Figure 5**) showing high rank correlation and include a human-revisited subset, the dependence on a single proprietary model is still a structural vulnerability:
   - It is plausible that Gemini’s stylistic and conceptual biases shape what is considered a “good” caption (e.g., emphasizing certain statistics or phrasing), which may overestimate its performance relative to other models.  
   - The human-revisited subset is only 579 samples concentrated in four domains (Table 2, **Page 4**), which is small compared to the 4k test set and cannot fully de-bias evaluation.  
   While the authors are honest about this limitation in Appendix A, it meaningfully affects the benchmark’s long-term robustness. A more balanced oracle ensemble (or mixing human-written and multi-oracle captions) would strengthen the scientific value.

3. **Semi-synthetic nature and limited scale of human-verified content raise questions about how “real” the benchmark is.**  
   The benchmark uses real numeric data, but nearly all captions are LLM-generated; only 579 are explicitly human-revisited. For tasks that purport to approximate real-world analytic workflows (e.g., business analysts describing Walmart sales, epidemiologists describing COVID curves), human-written domain narratives are arguably more representative, especially in terms of high-level abstractions, causal hypotheses, and domain-specific jargon. As **Table 9** and the human detectability study show, captions are accurate and stylistically human-like, but they may still miss the kinds of insights real analysts would include. The paper acknowledges this in Appendix A but does not attempt to quantify how “analyst-like” these captions are beyond statistical descriptors. This limits CaTS-Bench more to “statistical description” than to genuine domain-level explanation.

4. **Design of numeric metrics is somewhat ad hoc and under-analyzed.**  
   The Numeric Score metric (\(\lambda_A = 0.3\), \(\lambda_R = 0.7\)) and the 5% tolerance are only briefly justified as “widely accepted” (**Section 3.5**, **Appendix F.2**). There is no ablation/sensitivity analysis on these hyperparameters. For example:
   - How robust are model rankings if the tolerance is 2% or 10%?  
   - How does the balance between accuracy and recall affect what is rewarded? A model that writes very long captions with many approximate numbers might do well on recall but be unhelpful in practice.  
   - The metric matches each reference number to its closest generated number, which can produce optimistic scores if the caption includes many unrelated numbers.  
   Given that these metrics are a central claimed contribution, a more thorough quantitative exploration of their behavior would be valuable.

5. **Q&A suite is relatively small and partially task-misaligned with the training procedure.**  
   The final Q&A benchmark uses only 460 questions (Table 17, **Page 29**), which is quite small compared to the 20k captioning samples. This is especially an issue for sub-tasks with only 40 examples (amplitude, peak, mean, variance comparison), which makes the radar plots in **Figure 3** somewhat noisy; a few question idiosyncrasies can swing performance considerably. The authors themselves observe mixed effects of finetuning on Q&A (**Section 4.2**), likely due to task mismatch. However, they do not explore any multi-task training or simple modifications (e.g., joint captioning + Q&A finetuning on the much larger 38k unfiltered Q&A pool described in Appendix J.2). As a result, the Q&A evaluation feels more like a diagnostic add-on than a full-fledged benchmark.

6. **Limited exploration of non-visual baselines and representation-learning perspectives.**  
   The paper is framed as a benchmark for “context-aware time series captioning”, but nearly all baselines are VLMs that ingest plots, with less focus on *time-series-native* representation models. There is a PAL setup using Qwen-VL (Section 4.1) and numerically grounded prompts (Section 3.3), but no baseline where an LLM operates purely on learned time-series embeddings (e.g., using a pre-trained TS encoder like TST, PatchTST, or Chronos to produce a compressed representation passed to the LLM) or using classical statistical descriptors as structured input. Given that **Table 15** and **Figure 4** show that vision contributes very little, it would be very instructive to compare against (a) a strong text-only LLM that only receives the numeric series as text; (b) a TS encoder + LLM pipeline without any plot. Without such baselines, the representation-learning insights remain incomplete.

7. **Role of numeric input vs. visual input is under-quantified in the main text.**  
   Appendix K.1 (**Figure 16**) provides a qualitative example showing that including the numeric series reduces hallucination compared to using the plot alone. However, the main visual ablation experiment in **Table 15** conflates “visual + numeric + metadata” vs. “numeric + metadata only”. There is no systematic experiment where models receive (i) plot + metadata but *no numbers*, (ii) numbers + metadata but no plot, and (iii) both. This makes it hard to disentangle whether the numeric series itself or just the metadata is doing the heavy lifting in caption quality.

8. **Statistical inference behavior after finetuning is only qualitatively analyzed, yet appears quite fragile.**  
   Section K.2 (**Figures 17–18**) shows that finetuning can cause models like LLaVA 1.6 Mistral to become overconfident and hallucinate approximate means even when inaccurate, whereas others (Idefics 2) can infer means and std reasonably well. This is an important insight, but the paper mostly provides qualitative anecdotes. It would be helpful to quantify, for example, how often finetuned models mention approximate means or standard deviations and how often those are within 5%. Some of this is visible in **Table 4**, but not broken down as “mention vs. omit” behavior per model. Without more systematic measurement, the takeaway around “cross-entropy drives overconfident numeric guesses” remains somewhat speculative.

9. **Some missing detail/clarity in mathematical definitions and metric implementation.**  
   While Equations (1)–(5) are standard, the numeric metrics in Section 3.5 are only described verbally. For example:
   - For Statistical Inference Accuracy, the exact indicator function is not written, and it is unclear whether multiple mentions of a statistic are allowed (e.g., if a caption says “around 10 to 12” for the mean, how is that parsed and scored?).  
   - For the Numeric Score, the paper states: “Accuracy (mean of \(1 - \min\{\text{relative\_error, tolerance}\}\) over all matched numbers)” but does not give a formal formula or define how unmatched reference numbers are treated in the denominator. A concise mathematical definition (e.g., indexing ground-truth numbers \(g_i\) and matched generated numbers \(\hat{g}_i\), plus explicit recall computation \(R = \frac{\#\text{matched}}{\#\text{total}}\)) would improve clarity and reproducibility.  
   Since this metric is a central artifact others will re-implement, more explicit notation would be appropriate.

10. **Claims about being “not limited by CaTS-Bench” for vision usage are plausible but not fully substantiated.**  
    The authors argue that the visual modality under-utilization is a limitation of current VLMs, not the benchmark, because CaTS-Bench provides plots plus metadata and even alternative encodings (GAF, RP). However, **Table 16** shows that, for Idefics2-8B, *all* visual encodings sharply degrade DeBERTa F1 and Numeric scores compared to “No Plot”, whereas line plots modestly help some stats (e.g., max/mean inference). It remains possible that the particular plotting choices (e.g., scaling, font size, clutter, lack of color-coded meta-features) interact poorly with generic CLIP-like encoders. The paper could be more cautious here and acknowledge that benchmark design may also contribute to the difficulty, not just models.

## Potentially Missing Related Work

1. **Sen et al., “BEDTime: A Unified Benchmark for Automatically Describing Time Series”, 2025.**  
   This appears to be a highly relevant benchmark for time series description by language models, directly overlapping with the core task of CaTS-Bench. It should be discussed in Section 2 (Related Work) and compared in **Table 1**, including differences in data sources, caption styles, inclusion of metadata and images, and scale. It should also be mentioned in the Introduction where the authors claim to fill the gap in TSC benchmarks.

2. **Sen et al., “Can Language Models Infer Event Descriptions from Time Series?”, 2025.**  
   This work evaluates LMs’ ability to infer event descriptions from time series, which is conceptually close to the Q&A and descriptive captioning tasks here. It should be cited in Section 1–2 and discussed as a complementary benchmark focusing on event-level semantics rather than purely numeric/statistical description.

3. **Davies et al., “Language Models Do Not Embed Numbers Continuously”, 2025.**  
   This paper directly targets numeric reasoning limitations in LMs, which CaTS-Bench also aims to expose, especially in Sections 1–3. It should be cited near the discussion of numeric extrapolation and reasoning limitations in **Page 2**, and potentially in Section 3.5 when motivating numeric-focused metrics, to better ground the work in the broader literature on numerical representations.

(Other directly related works already appear to be cited, including VisText and various LLM-for-time-series papers.)

## Questions

1. **Comparison to BEDTime and other TSC-style benchmarks.**  
   Please provide a direct comparison table or paragraph contrasting CaTS-Bench with BEDTime and any other contemporaneous TSC benchmarks beyond TADACap/TRUCE/TACO. In particular, what is unique about CaTS-Bench’s multimodality and Q&A suite that those benchmarks do not provide?

2. **Numeric metric sensitivity.**  
   Could you report a small ablation varying (a) the relative tolerance (e.g., 2%, 5%, 10%) and (b) the (\(\lambda_A, \lambda_R\)) weighting in the Numeric Score to show that model rankings are stable? If not, can you at least comment based on pilot experiments whether the main conclusions (e.g., finetuning improves numerics; PAL helps) would hold under alternative settings?

3. **Entanglement of Gemini as oracle and baseline.**  
   Have you considered regenerating a small subset of ground-truth captions using a non-Gemini oracle (e.g., GPT-4o or a strong open-source VLM) to test whether Gemini’s relative advantage persists? This could help reassure users that CaTS-Bench does not inherently favor Gemini-style outputs.

4. **Numeric-only vs. plot-only vs. both.**  
   The current visual ablation in **Table 15** compares VL (plot + text) to L (text-only). Could you run a focused experiment on a subset where models receive: (i) plot + metadata *without* the raw numeric array; (ii) numeric-only + metadata; (iii) both. This would more cleanly disentangle the relative contributions of each modality and help future work understand whether vision alone can support descriptive captioning.

5. **Scaling up human or expert-authored captions.**  
   Do you have concrete plans or methodology for scaling the human-revisited (or human-authored) portion beyond 579 examples? For instance, would a smaller but fully human benchmark be feasible, and how might its insights differ from the semi-synthetic one?

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The dataset construction and experiments are carefully described, with extensive sanity checks and multiple baselines. The main concerns are about reliance on a single oracle and some under-specified metric details, but these do not fundamentally undermine the empirical conclusions.

## Presentation Rating

3: good.  
The paper is generally clear, well organized, and rich in figures and tables. Some claims about “firstness” and certain metric definitions could be sharpened, and related work is missing a couple of key benchmarks, but overall readability is high.

## Contribution Rating

2: fair.  
CaTS-Bench is a useful benchmark with a solid pipeline and insightful analyses, but (i) novelty is diminished by missing discussion of very similar recent benchmarks, and (ii) the work is primarily engineering/benchmarking rather than introducing fundamentally new modeling or evaluation paradigms.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  
The benchmark is carefully constructed and the experimental analysis is thoughtful, especially regarding numeric reasoning and vision under-utilization. However, incomplete positioning relative to closely related recent benchmarks and structural dependence on a single proprietary oracle meaningfully weaken the case for ICLR main-track acceptance as-is. With stronger related-work integration, clearer metric formalization, and some additional modality/metric ablations, this could become a solid contribution.

## Reviewer Confidence

4: confident.  
I am familiar with LLMs for time series, multimodal benchmarks, and numeric reasoning, and I read the paper (including key appendices and equations) carefully. There is some uncertainty regarding external contemporaneous benchmarks not discussed in the paper, but overall my assessment is unlikely to change dramatically.