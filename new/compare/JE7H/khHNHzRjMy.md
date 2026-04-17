---
job_id: 0477fc94-069f-4330-b155-06322d8e6fb3
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: khHNHzRjMy.pdf
paper: EmoSign: A Multimodal Dataset for Understanding Emotions in American Sign Language
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The work introduces a new dataset and benchmarks for multimodal emotion recognition in ASL, squarely within ICLR’s scope on representation learning, multimodal modeling, datasets and benchmarks, and fairness / accessibility in ML systems.

## Minimum Quality
Pass ✅.  
The paper is complete and well-structured, with Abstract, Introduction, Related Work, Dataset/Methodology (Sections 3–4), Experiments/Results (Section 5), Limitations (Section 6), and Conclusion (Section 7). Claims are empirically grounded, the methods are technically sound at the scale they operate, and the exposition is adequate for peer review. While there are important weaknesses (small dataset, sampling bias, limited modeling depth), none rise to the level of a desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no hidden prompts, manipulative text, or instructions targeting automated review systems in the main content.

---

# Expected Review Outcome:

## Summary

The paper introduces EmoSign, a multimodal dataset of 200 American Sign Language (ASL) utterances annotated by three Deaf native signers with (1) sentiment on a 7-point scale, (2) intensities of 10 emotion categories, and (3) open-ended descriptions of visual emotion cues. The authors describe the collection and annotation pipeline, analyze inter-annotator agreement, and benchmark four multimodal LLMs (GPT‑4o, AffectGPT, Qwen2.5‑VL, MiniGPT4‑video) on sentiment classification and (single-label) emotion classification under caption-only, video-only, and video+caption conditions, plus a qualitative “emotion cue grounding” analysis. Results show that current models rely heavily on text, are weak at leveraging visual ASL cues, and exhibit a mild positive/neutral bias.

## Strengths

1. **Clear, well-motivated problem at the ASL × affective computing intersection.**  
   The paper focuses on a genuine gap: there are many ASL datasets for translation and recognition, but essentially none with fine-grained sentiment and emotion annotations linked to Deaf signers’ own interpretations. The discussion in Section 1 around facial expressions serving both grammatical and emotional roles in ASL, and the potential for misinterpretation in clinical/legal settings, makes the motivation compelling.

2. **Thoughtful dataset design with Deaf expertise and qualitative cue descriptions.**  
   The annotation strategy (Section 3.2) uses three Deaf native signers with professional interpretation experience, and includes both numeric labels and rich free-text descriptions of cues (movement speed, sign size, facial and body markers, etc.). The qualitative analysis on Page 5 summarizing themes from these descriptions is particularly valuable; for instance, the emphasis on non-manual markers and sign modifications aligns well with linguistic literature and provides concrete guidance for model design beyond just “add more data.”

3. **Transparent dataset construction pipeline and pre-processing.**  
   Figure 1 clearly summarizes the selection and annotation pipeline: using ASLLRP as the base corpus, applying VADER to English captions to select the top 100 positive and top 100 negative utterances, then collecting annotator scores and cues, and finally aggregating via majority vote with confidence-based tie-breaking. This clarity makes the dataset reproducible and easy to critique or extend.

4. **Solid basic statistics and reliability analysis.**  
   Table 2 reports Krippendorff’s alpha for each emotion and sentiment, with an overall mean of 0.593 and per-label values that are honestly low for some categories (e.g., 0.119 for negative surprise, 0.166 for disgust). The comparison with MELD and IEMOCAP helps situate these agreement levels. Figure 2 further provides useful descriptive statistics of clip duration (2A), sentiment distribution (2B), and binarized emotion frequencies (2C).

5. **Benchmarks clearly reveal modality imbalance and visual weaknesses of MLLMs.**  
   The ablations in Table 3 (sentiment) and Table 4 (emotion classification) under caption-only, video-only, and video+caption conditions are informative. For example, Table 3 shows AffectGPT’s wF1 plunging to essentially zero in the video-only 3-class sentiment condition (0.04), while rising to 64.37 with video+caption. Similarly, Table 4 shows GPT‑4o’s single-label emotion performance improving from total wF1 = 20.76 (video-only) to 55.09 (video+caption), and being similar in caption-only vs. video+caption, underscoring that the models rely heavily on text and struggle to exploit visual signing cues.

6. **Useful qualitative grounding analysis exposing reasoning failures.**  
   Figure 3 and the example in Appendix A.4/A.5 (Figures 6 and 8) are effective. They show for a specific clip (“If Mary gets home late, John will probably be upset”) that models flip their interpretation of the same visible behaviors depending on whether the caption is present: Qwen2.5 describes “frustration or anger” in the video-only condition but “worry” when the caption is available. This concretely illustrates that explanations are text-driven rationalizations rather than genuinely grounded in visual ASL cues.

7. **Clarity of writing and figures.**  
   Overall exposition is clear and reasonably concise. Figures like Figure 2 (dataset stats) and Figures 7–10, 23–27 (additional distributions and accuracy plots in the appendix) are easy to interpret and aligned with the text. Figure 5 (Jaccard similarity heatmap) in the appendix nicely justifies combining “joy” and “excited” into “happiness” based on a Jaccard similarity of 0.81.

## Weaknesses

1. **Very small dataset size relative to ICLR standards and claimed ambitions.**  
   EmoSign contains only 200 utterances (≈16 minutes) from 4 signers (Section 3.4), yet the paper positions it as “the first comprehensive dataset” and a “benchmark.” For a modern ICLR-scale benchmark, this is extremely small: even with 200×3 annotations, the effective sample size is limited for training or robustly evaluating neural models, especially with 11 emotion classes (10 emotions + neutral) on skewed distributions. This size essentially constrains the paper to probing off-the-shelf models, not enabling serious learning-based comparison. The authors cite similarly small, high-quality benchmarks in other domains (Arodi et al., 2024; Krojer et al., 2024; Li et al., 2024b), but those are typically accompanied by more exhaustive analysis and focus on domains where very small datasets are unavoidable; this justification is not fully convincing here.

2. **Sampling strategy introduces strong bias and limits generality.**  
   The dataset is constructed by first pre-selecting ASLLRP clips whose English captions are highly positive or negative according to VADER, then taking the 100 most positive and 100 most negative utterances (Section 3.1). This induces at least three issues:
   - It strongly biases the dataset toward text-emotion alignment, even though a core claim is that ASL visual cues can differ from textual sentiment. The authors mention they “found VADER results differed from the annotators’ results” in Section 6, but do not quantify this divergence; a simple correlation or contingency table between VADER scores and final annotated sentiment (the 7-point scale) is missing.
   - Neutral and subtle emotions are systematically underrepresented (Figure 2B, where labels around 0 are sparse), which makes the benchmark particularly ill-suited for nuanced or context-dependent affect.
   - It couples the dataset to an English-centric, lexicon-based sentiment estimator, embedding its biases into sample selection.  
   These limitations materially affect the kinds of models and questions the dataset can support.

3. **Limited coverage and imbalance of emotion categories, with low reliability for several labels.**  
   Table 2 shows Krippendorff’s alpha as low as 0.119 for negative surprise and around 0.166 for disgust, yet these labels are still used in the single-label classification benchmark. For some classes, counts are very small: Figure 24 (Appendix, single-expression distribution) shows several emotion categories with fewer than ~10 clips. This combination of low frequency and low inter-annotator agreement makes any quantitative evaluation per-class shaky. The paper should have:
   - More explicitly flagged which labels are reliable enough for benchmarking, potentially collapsing or discarding unreliable categories.
   - Provided per-class sample counts in Table 4 or the main text, not only in the appendix, to allow readers to interpret reported accuracies like “SP(N) Acc = 14” for MiniGPT4 (caption-only) or “DG Acc = 50” for GPT‑4o (video+caption) in light of their denominators.

4. **Benchmarks are shallow and not tailored to the sign/emotion setting.**  
   The evaluation only probes four pre-trained MLLMs with simple prompting. There is no attempt to:
   - Train or fine-tune any model (even a shallow classifier) on EmoSign, e.g., using pose-based features, optical flow, or facial landmarks, which the authors themselves call out as future work (Section 6).
   - Compare with any sign-language-specific models or feature extractors (e.g., pose-based encoders from prior SLT/SLR literature).  
   As a result, the quantitative results in Table 3 and Table 4 are mostly telling us that generic MLLMs, unadapted to ASL, are not good visual affect recognizers. This is not surprising, and with only 200 examples, the negative result is hard to interpret as anything more than anecdotal.

5. **Inconsistent and sometimes puzzling evaluation metrics and numbers.**  
   The metrics are defined only very briefly. For sentiment analysis, the authors say they use accuracy and weighted F1 (Section 4.1) but never specify whether “weighted” means per-class weighting by support (as in sklearn) or something else. For emotion classification, they talk about “weighted accuracy” wAcc (Table 4) but do not provide the formula. This is particularly important given some counterintuitive values:
   - In Table 3, MiniGPT4 caption-only has wAcc = 1.92 and wF1 = 5.92 on the 3-class sentiment task, both essentially random / broken, whereas the same model on video-only yields wAcc = 34.68, wF1 = 40.00. The paper does not discuss why caption-only is catastrophically worse than video-only for this model.
   - For several models in Table 4, video+caption yields *lower* total wAcc than caption-only (e.g., GPT‑4o: 41.16 vs. 35.97) while wF1 is almost the same (55.89 vs. 55.09). This contradicts the blanket claim that “Models performed similarly in caption-only and video+caption conditions and notably better than video as the only input,” but the text does not analyze which classes are hurt by adding video.  
   A more precise mathematical definition of wAcc and wF1, plus per-class supports and overall macro vs. micro views, would clarify whether these numbers reflect meaningful trends or evaluation noise on a very small dataset.

6. **Emotion cue grounding task is underspecified and lacks quantitative evaluation.**  
   Section 4.1 introduces “emotion cue grounding” as identifying relevant temporal/spatial regions of a video, framed in analogy to video QA grounding literature. However:
   - The dataset does not provide frame- or region-level ground-truth annotations; only free-form text descriptions of cues (e.g., “oooh mouth morpheme,” “emphasized ‘arriving’ with head shake and tilt” in Figure 3).
   - The “task” is evaluated only through a few hand-picked qualitative examples and narrative discussion (Section 5.3).  
   While the qualitative analysis in Figure 3 is insightful, branding this as a separate grounding task is misleading. As defined, there is no objective way to measure grounding, and the paper never proposes a concrete metric or algorithm (e.g., aligning model-generated explanations with expert descriptions using some similarity score). At minimum, the authors should clarify that this is exploratory qualitative analysis rather than a benchmarked task.

7. **Limited engagement with recent sign-language modeling literature.**  
   The related work in Section 2 mostly covers SLT/SLP datasets and a small number of translation/production papers, but it omits several highly relevant works on continuous ASL recognition and representation learning that bear directly on how one might design visual encoders or baselines for this dataset. Important missing references (see next section) include hybrid CNN‑HMM models, temporal convolutional and spatio-temporal graph approaches, and large multilingual sign datasets. Their omission weakens the argument that EmoSign sits in a broader trajectory of sign-language representation learning and undercuts claims about the novelty of focusing on affect.

8. **Potential overstatement of claims given limited dataset and analyses.**  
   Some statements are stronger than warranted. For instance, the abstract claims EmoSign is “the first sign video dataset containing sentiment and emotion labels,” and Section 3 calls it “comprehensive,” despite FePh already annotating facial expressions with emotions (albeit with important differences) and despite the clear size limitations. Similarly, Section 5.1 claims that “visual information can contribute meaningfully” because video+caption often outperforms caption-only, but Table 4 shows this is not systematically true for emotion classification, and the magnitude of improvements in Table 3 is modest relative to noise from such a tiny dataset. The paper would be stronger if more carefully hedged, explicitly acknowledging that these are preliminary indications on a small-scale dataset.

9. **Minor methodological and mathematical clarity issues.**  
   - The majority-vote aggregation with confidence-based tie-breaking (Section 3.3) is reasonable, but the exact rule when all three annotators choose distinct labels on the 7-point sentiment scale or on intensity scales {0,1,2,3} is not described. For example, if labels are \(-1, 0, 1\) with confidences \(c_{-1}, c_0, c_1\), is the highest-confidence label always chosen, or is there some tie-breaking toward neutrality?  
   - When combining “joy” and “excited” into “happiness,” the Jaccard similarity score of 0.81 (Figure 5) is cited, but the exact definition used (e.g., for each video, presence when intensity ≥1 vs. ≥2) is not explicitly specified. Since this transformation affects label space and subsequent metrics in Table 4, stating the precise threshold \(t\) used for binarizing intensities (\(I_e \ge t\)) before computing Jaccard would make the procedure reproducible.  
   - There is no explicit equation for Krippendorff’s alpha as used here or a note about which distance function is employed for ordinal data on the \([-3,3]\) and \([0,3]\) scales; while not strictly necessary, a short definition would help readers confirm that ordinal disagreements are treated appropriately.

Overall, these weaknesses do not invalidate the dataset or the qualitative insights, but they do materially limit the robustness, scope, and impact of EmoSign as an ICLR-level benchmark.

## Potentially Missing Related Work

Below are directly relevant works that are not cited but should be considered:

1. **Koller et al., “Deep Sign: Enabling Robust Statistical Continuous Sign Language Recognition via Hybrid CNN-HMMs”, 2019.**  
   - Relevance: Presents hybrid CNN-HMM models for continuous SLR, focusing on robust temporal modeling from video. Directly relevant for discussing potential visual encoders or baselines for EmoSign.  
   - Suggestion: Add to Section 2 (“Machine learning research on sign language”) to contextualize temporal modeling approaches that could be adapted to emotion recognition on sign streams.

2. **Camgoz et al., “Sign Language Transformers: A Survey”, 2020.**  
   - Relevance: Comprehensive survey of transformer-based and other modern architectures for SLR/SLT. Helps clarify where EmoSign sits relative to current architectural trends.  
   - Suggestion: Cite in Section 2 when discussing recent advances in SLT and SLP, and use it to argue which model components may be well-suited to capture emotional/non-manual cues.

3. **Albanie et al., “Babel-17: A Large-Scale Multilingual Sign Language Dataset”, 2021.**  
   - Relevance: Introduces a large-scale multilingual sign dataset; useful comparison point in Table 1 and discussion of dataset scale and diversity.  
   - Suggestion: Include in Table 1 and discuss in Section 2 as part of the broader ecosystem of sign datasets, emphasizing EmoSign’s unique emotional annotations vs. Babel-17’s scale.

4. **Zhou et al., “Spatial-Temporal Graph Convolutional Networks for Sign Language Recognition”, 2020.**  
   - Relevance: Proposes ST-GCNs tailored to modeling spatio-temporal joint dynamics in signing, directly relevant to capturing movement-based cues like speed, sign size, and emphasis that EmoSign annotators highlight.  
   - Suggestion: Cite in Section 2 when talking about pose-based or skeletal representations, and mention in Section 6 as a promising architecture class for future emotion recognition baselines on EmoSign.

5. **Li et al., “Word-Level Deep Sign Language Recognition from Video: A New Large-Scale Dataset and Methods Comparison”, 2020.**  
   - Relevance: Provides a large dataset and comprehensive methods comparison for word-level SLR, including CNN-RNN architectures; relevant as a reference for dataset scale and for potential feature extractors.  
   - Suggestion: Discuss in Table 1 / Section 2, emphasizing how EmoSign’s focus differs (emotion vs lexical content) and how these models could serve as backbones.

6. **Yin et al., “Sign Language Recognition, Generation, and Translation: An Interdisciplinary Perspective”, 2021.**  
   - Relevance: An overview paper covering recognition, generation, and translation of sign languages, giving a broad view of tasks and challenges.  
   - Suggestion: Reference in Section 2 to frame EmoSign as expanding the space of sign-related tasks from lexical/semantic to affective.

7. **Zhou et al., “Temporal Convolutional Networks for Sign Language Recognition”, 2019.**  
   - Relevance: Introduces TCNs for SLR, highlighting architectures for capturing temporal structure in signing.  
   - Suggestion: Add to Section 2 and mention in Section 6 as one of the established time-series backbones that could be used as an emotion recognition baseline.

8. **Camgoz et al., “Neural Sign Language Translation”, 2018.**  
   - Relevance: Early neural SLT model; relevant to discussions around sign-to-text translation and how emotion-aware models might be integrated into translation pipelines.  
   - Suggestion: Cite alongside Fang et al. (SignLLM) and Liang et al. (LLaVA-SLT) in Section 2.

9. **Koller et al., “Re-Sign: Re-Aligned End-to-End Sequence Modelling with Deep Recurrent CNN-HMMs”, 2019.**  
   - Relevance: Presents end-to-end sequence modeling tailored for sign language, again relevant to temporal modeling for emotion recognition.  
   - Suggestion: Add in Section 2 when discussing end-to-end SLT/SLR methods and sequence modeling.

10. **Zhou et al., “Learning Spatial-Temporal Representations for Sign Language Recognition”, 2019.**  
    - Relevance: Focuses on learning joint spatial-temporal representations from video of signing; highly pertinent to the “emotion cue” dimensions (e.g., motion speed, body posture) that EmoSign seeks to capture.  
    - Suggestion: Cite in Section 2 and possibly in the Limitations/Future Work section when discussing potential architectures for explicitly modeling emotional kinematics.

## Questions

1. **Quantifying divergence between text-based and signer-based sentiment.**  
   You mention in Section 6 that VADER-filtered captions “often contained rich non-manual markers that conveyed emotions differently than the text.” Can you provide quantitative evidence of this? For example:
   - Correlation between VADER scores and final annotated sentiment on the \([-3,3]\) scale.  
   - A confusion matrix of VADER polarity (positive/neutral/negative) vs. majority-vote signer sentiment (collapsed to 3 classes).  
   This would significantly strengthen your claim that EmoSign reveals modalities where visual affect diverges from text sentiment.

2. **Reliability thresholding for emotion labels.**  
   Given the low Krippendorff’s alpha for several categories (e.g., 0.119 for negative surprise, 0.166 for disgust in Table 2), did you consider merging or discarding low-agreement categories for the benchmark tasks? If so, what happened empirically? If not, could you discuss whether you believe these labels are sufficient for training/evaluating models, and whether you would recommend users to treat them as “softer” supervision?

3. **Exact definitions of wAcc and wF1.**  
   Please provide precise formulas (in LaTeX) for the weighted accuracy and weighted F1 used in Tables 3 and 4, including the class weights. Are you using per-class accuracy weighted by class frequency, macro-averaged F1 weighted by support, or something else? Clarifying this would help interpret unusual numbers like MiniGPT4’s 1.92 wAcc in the caption-only 3-class setting.

4. **Aggregation rule when annotator labels differ.**  
   In Section 3.3 you state that majority vote is applied and confidence is used in ties. Could you specify formally how this works for ordinal scales?
   - If annotators give three distinct sentiment labels \(s_1,s_2,s_3 \in \{-3,\dots,3\}\), do you always pick \(\arg\max_i c_i\) where \(c_i\) is confidence, or is there any consideration of label distance (e.g., preferring the median sentiment)?  
   - For emotion intensities \(I_{e} \in \{0,1,2,3\}\), is the procedure the same?  
   A short algorithmic description would clarify the final label distribution.

5. **Grounding evaluation: could you propose a quantitative proxy?**  
   Have you considered aligning model-generated textual explanations with annotator cue descriptions via some semantic similarity measure (e.g., using an embedding model) as a crude grounding score? Even with only free-text descriptions, this might allow reporting at least a rough quantitative comparison of models’ ability to mention specific cues (e.g., “furrowed brows,” “fast signing”) versus generic statements (“neutral expression”). If you view this as inappropriate, please clarify why in the paper so readers understand the intended use of the cue descriptions.

6. **Feasibility of releasing additional data or annotations.**  
   Given that 200 clips is quite small, are there practical or ethical constraints that prevent you from scaling EmoSign up (e.g., more ASLLRP utterances, more signers, or adding frame-level landmarks)? It would be helpful to know whether EmoSign is intended as a seed dataset that will grow over time, or a fixed benchmark.

Author responses addressing these questions and elaborating on the reliability and metric definitions would improve my confidence in the dataset’s utility and in the interpretation of your results.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A. The paper notes IRB approval, uses an existing ASL corpus with documented consent (ASLLRP), and compensates Deaf annotators. There is no evidence of privacy violations or unsafe applications in the current scope.

## Soundness Rating

2: fair.  
The dataset construction and basic analyses are generally sound, and the benchmarks are correctly executed at their modest scale. However, the very small dataset size, label reliability issues for several emotions, limited exploration of models, and underspecified metrics/grounding task prevent a higher rating.

## Presentation Rating

3: good.  
The paper is clear, logically organized, and most figures and tables (e.g., Figures 1–3, Table 1–4) are easy to interpret. Some methodological details (exact metric definitions, aggregation rules) and more cautious phrasing of claims would further improve clarity.

## Contribution Rating

2: fair.  
The main contribution is a small but carefully annotated ASL emotion dataset and some insightful qualitative analyses. This is valuable, but the scale, limited diversity, and relatively shallow benchmarking reduce the impact as an ICLR benchmark contribution.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  
The paper addresses an important and underexplored problem and does so with notable care in dataset design and qualitative analysis, especially through collaboration with Deaf native signers. However, the dataset is very small and somewhat biased by VADER-based text filtering, several emotion categories are low-agreement and low-frequency, the benchmarks are limited to prompting generic MLLMs, and evaluation metrics and the “grounding” task are underspecified. These issues collectively keep the work just below what I would expect for the main track at ICLR, although it could still be a useful community resource if accepted.

## Reviewer Confidence

4: confident.  
I am familiar with multimodal emotion recognition and sign-language modeling literature, have carefully read the paper, tables, and figures (especially Figures 1–3 and Tables 1–4), and I have reasonable confidence in my assessment, though I would welcome additional clarifications from the authors on metrics and label reliability.