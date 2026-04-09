## Summary

EmoSign introduces the first dedicated dataset for emotion recognition in American Sign Language (ASL), comprising 200 ASL video clips annotated with sentiment (7-point scale), emotion category presence/intensity (10 categories), and open-ended descriptions of emotion cues by 3 Deaf native ASL signers with professional interpretation experience. The paper benchmarks 4 multimodal LLMs across three tasks—sentiment analysis, emotion classification, and emotion cue grounding—revealing that current models rely heavily on text captions rather than visual cues and exhibit systematic positive/neutral biases.

## Strengths

- **Community-informed annotation design**: The recruitment of Deaf native ASL signers with professional interpretation experience (Section 3.2) directly addresses a documented failure mode in prior work—FePh used hearing annotators, which risks misinterpreting grammatical facial expressions as emotional ones (Lim et al., 2024). This is a substantive methodological choice that most emotion datasets do not make.

- **Rich, multi-layered annotation schema**: The three-layer annotation (sentiment, multi-label emotion intensity, and free-text cue descriptions) goes well beyond binary presence/absence labels. The open-ended cue descriptions (e.g., signing speed, specific non-manual markers) provide grounding supervision that is rare in emotion datasets and could enable future work on interpretable affective reasoning.

- **Clear empirical demonstration of modality imbalance**: The ablation across caption-only, video-only, and video+caption conditions (Tables 3–4) provides direct evidence that current MLLMs predominantly leverage text shortcuts rather than visual understanding for sign language emotion recognition. The finding that AffectGPT defaults to "Neutral" in the video-only condition (wF1 = 0.04) is a stark and informative result.

- **Identification of systematic model biases**: The paper documents specific, reproducible failure patterns—GPT-4o collapsing to happiness/frustration in the video-only condition, AffectGPT's neutral bias, and Qwen2.5 claiming to need audio context for sign language—providing actionable directions for model improvement.

## Weaknesses

### Major:

- **VADER-based selection creates a confound for multimodal evaluation**: The dataset was constructed by selecting the 100 most positive and 100 most negative utterances based on VADER sentiment scores computed on *English text captions* (Section 3.1). This means the text modality inherently carries the emotional signal that was used to curate the dataset. Consequently, the finding that "caption-only performance was similar to or slightly better than video-only results" (Section 5.1) is partially tautological—the dataset was selected precisely because the captions contained salient emotional content. This undermines the core comparison between modalities and makes it difficult to assess whether models are genuinely failing to perceive visual emotion cues or simply encountering a dataset where the text signal is overwhelmingly strong by construction. A more informative evaluation would include analysis of cases where VADER sentiment diverges from annotator labels, which the authors allude to in Section 6 but do not quantify.

- **Critically low inter-annotator agreement on multiple emotion categories**: Table 2 reports Krippendorff's alpha of 0.119 for surprise (negative), 0.166 for disgust, and 0.330 for frustration. Alpha values below 0.2 indicate agreement barely above chance, which calls into question whether these categories are reliably annotatable with the current protocol. Using such labels as ground truth for evaluation (Table 4 includes columns for disgust, surprise, frustration, and anger at α = 0.370) means that model performance on these categories is measured against unreliable targets. The paper does not discuss the implications of this for benchmark validity or propose remediation (e.g., merging low-agreement categories, flagging them separately, or excluding them from primary metrics).

- **Emotion cue grounding task lacks quantitative evaluation**: The paper introduces "Emotion Cue Grounding" as a benchmark task (Section 4.1), defining it as identifying "video frames and spatial regions relevant to sentiment analysis and emotion classification." However, Section 5.3 evaluates this task solely through manual inspection of "several randomly selected videos" with qualitative discussion. No quantitative grounding metric (temporal IoU, spatial overlap, precision/recall of identified cues) is provided. This means the grounding task cannot be reproduced or compared by future work, which is inconsistent with claiming it as a "benchmark" contribution.

### Minor:

- **Limited dataset scale and diversity**: With 200 utterances from 4 signers in a lab environment (Section 3.4), the dataset is small for training or robust evaluation. The authors acknowledge this and cite comparable small expert-annotated datasets (Arodi et al., 2024; Krojer et al., 2024; Li et al., 2024b), but those datasets serve different tasks (anomaly detection, image editing, graph analysis) where small size is more defensible. For multimodal video understanding, 200 clips may limit the ability to draw generalizable conclusions about model behavior, even in evaluation-only mode.

- **No human baseline for video-only condition**: The paper concludes that "current multimodal models fail to integrate visual cues into emotional reasoning" (Abstract), but without measuring human performance on the video-only task (i.e., Deaf signers viewing muted clips without captions), it is impossible to determine whether poor model performance reflects a fundamental limitation of current architectures or simply the inherent difficulty of the task. A human baseline would contextualize the model results and clarify whether the performance gap is surmountable.

- **Annotation confidence scores are collected but not analyzed**: Section 3.2 states that annotators rated their confidence on a 0–100 scale after each video, yet these scores are never reported or used (e.g., to weight labels, identify ambiguous clips, or correlate with agreement). Analyzing whether low-confidence clips correspond to low-agreement categories could clarify whether the low alpha values reflect inherent ambiguity or annotation errors.

### Trivial:

- The single-expression emotion classification task (Section 4.1) merges joy and excitement into "happiness" due to high Jaccard similarity (0.81), but the original 10-category schema already included both as separate annotation targets. The rationale for collecting them separately only to merge them could be clarified.

## Nice-to-Haves

- **Fine-tuning experiments**: Demonstrating that a model can improve on this dataset through training (even small-scale) would strengthen the claim that EmoSign is a useful resource, not just an evaluation probe.

- **Analysis of VADER–annotator discrepancy**: Quantifying how often and where VADER sentiment diverges from Deaf annotator labels would directly address the selection confound and identify the most valuable clips for visual emotion learning.

- **Comparison to specialized facial expression recognition models**: Testing vision-only FER models (e.g., emotion classifiers trained on facial action units) would help isolate whether the modality imbalance finding is specific to MLLMs or a broader property of current vision systems.

- **Multi-label emotion classification benchmark**: The paper acknowledges this gap in Section 6; given that emotions frequently co-occur (Jaccard similarity of 0.81 for joy/excited), this is a natural next step.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: Insufficient model variety (only 4 MLLMs tested)** — Four models spanning proprietary (GPT-4o) and open-source (AffectGPT, Qwen2.5-VL, MiniGPT4) is a reasonable starting benchmark. Demanding more models is a generic request for breadth that does not identify a specific flaw in the paper's conclusions.

- **Weakness: No statistical significance testing (confidence intervals, p-values)** — While desirable, single-run MLLM benchmarking without significance testing is the standard practice in current multimodal evaluation papers. The small dataset size does make this more concerning, but this belongs in nice-to-haves rather than as a core weakness for a dataset paper.

- **Weakness: No fine-tuning experiments** — The paper positions EmoSign as a benchmark and diagnostic dataset, not a training resource. Demonstrating learnability would strengthen the paper but is not a requirement for a dataset contribution at ICLR.

- **Weakness: Comparison to specialized FER models** — The paper's stated scope is evaluating multimodal LLMs on sign language emotion recognition. Testing monolithic FER systems is outside the paper's stated scope; the relevant question is whether MLLMs can handle this multimodal task, which the paper addresses.

- **Weakness: Disclosure of GPT-4o API costs** — This is a reproducibility nitpick about proprietary model costs, which is immaterial to the paper's scientific contributions.

- **Weakness: Grammatical error in Section 6** — The parser artifact or grammatical issue in the Limitations sentence is a formatting nitpick.

- **Weakness: No cross-signer train/test split** — The current evaluation is zero-shot with no training on EmoSign data, so cross-signer splits are not applicable. This would become relevant for fine-tuning experiments.

## Novel Insights

The most striking empirical finding is not just that models rely on text, but *how* they fail when deprived of it: they collapse to degenerate prediction patterns (AffectGPT → always Neutral; GPT-4o → happiness/frustration; MiniGPT4 → happiness), suggesting that current MLLMs have no structured representation of visual affective cues in sign language. The qualitative grounding analysis reveals a deeper pathology—models re-interpret the same visual cue in opposite directions depending on whether a caption is present (Figure 3), indicating that visual "reasoning" is being post-hoc rationalization of text-driven conclusions rather than independent perception. This is a more precise characterization than simply "models rely on text," and it has implications beyond sign language: it suggests that MLLM visual grounding for affective content is fundamentally confabulatory when textual context is available.

## Suggestions

1. **Quantify the VADER–annotator divergence**: Compute the correlation between VADER scores and annotator sentiment labels across all 200 clips. Identify the subset where they disagree, and report model performance on this subset separately. These "disagreement" clips are where visual emotion cues most diverge from text, making them the most informative for evaluating genuine visual understanding.

2. **Add a quantitative grounding metric**: Even a simple metric—e.g., overlap between model-identified temporal segments and a rough annotation of key frames—would make the grounding task reproducible and comparable. If frame-level annotations are not available, consider using the annotator cue descriptions to create pseudo-ground-truth for temporal localization.

3. **Flag or restructure low-agreement emotion categories**: Consider merging surprise(negative) and disgust into broader categories or marking them as "low-reliability" in the benchmark, with separate reporting. This would prevent misleading per-class accuracy comparisons in Table 4 where the ground truth itself is unreliable.

4. **Collect human video-only baseline**: Have Deaf signers perform the same sentiment/emotion task on muted clips (no captions) to establish an upper bound for visual-only understanding. This single addition would dramatically clarify whether the model failure is architectural or task-difficulty-related.