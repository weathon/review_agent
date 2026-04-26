Now I have a thorough understanding of the paper. Let me write the final consolidated review.

## Summary

EmoSign introduces the first multimodal dataset for emotion recognition in American Sign Language (ASL), comprising 200 ASL video clips annotated by 3 Deaf native signers with sentiment labels, emotion intensity ratings across 10 categories, and open-ended emotion-cue descriptions. The paper also establishes benchmark results for sentiment analysis, emotion classification, and emotion cue grounding using four multimodal LLMs, finding that current models heavily rely on text captions and struggle with visual-only emotion understanding.

## Strengths

- **First-of-its-kind dataset addressing a genuine gap**: EmoSign is, to the authors' knowledge, the first ASL video dataset with fine-grained emotion and sentiment labels annotated by Deaf native signers. Table 1 makes the gap clear — all prior ASL datasets lack any emotion labels. The dual function of facial expressions (grammatical and affective) in sign language creates a genuinely under-studied and technically distinctive challenge.

- **Deaf native-signer annotators**: Recruiting Deaf ASL signers with professional interpretation experience directly addresses the known problem of hearing individuals misinterpreting ASL facial expressions (Lim et al., 2024). This is a substantive methodological contribution that strengthens annotation validity relative to FePh, which used hearing annotators on face crops only.

- **Systematic modality ablation reveals model failures**: Tables 3 and 4 cleanly demonstrate through caption-only, video-only, and video+caption conditions that models are dramatically worse with visual input alone (e.g., AffectGPT video-only sentiment wF1 = 0.04, near-chance) and that adding video to captions often does not improve emotion classification (GPT-4o emotion wF1: caption-only 55.89 vs. video+caption 55.09). This directly supports the claim about models' failure to leverage visual cues.

- **Open-ended emotion cue descriptions from native signers**: Section 3.4 documents how emotions manifest through manual and non-manual markers from native-signer perspectives (e.g., signing speed, body movement, mouth configurations). This is novel documentation of ASL-specific emotion expression that goes beyond categorical labels.

## Weaknesses

### Fatal
None.

### Major

- **VADER-based video selection creates a confound for multimodal evaluation claims**: The dataset was constructed by selecting the 100 most positive and 100 most negative utterances based on VADER sentiment scores of the English *text captions*. This creates a built-in correlation between text sentiment and human-annotated emotion labels. The core claim that "current multimodal models fail to integrate visual cues and heavily rely on text captions" is therefore partially confounded — the text was already used to select which videos are in the dataset. The paper acknowledges in Section 6 that "VADER results differed from the annotators' results" and that interesting videos conveyed emotions "differently than the text," but these were precisely the ones *filtered out*. While the near-chance video-only performance still demonstrates genuine model limitations, the confound weakens the paper's interpretation of caption-only success as purely a model deficiency rather than partly a dataset artifact. A stratified analysis showing performance on clips where VADER and annotator labels align vs. diverge would substantially strengthen the conclusions.

- **Unreliable ground truth for most emotion categories**: Krippendorff's alpha values for several emotions are far below accepted thresholds: disgust (0.166), surprise-negative (0.119), sadness (0.333), frustration (0.330). Values below 0.33 indicate agreement barely above chance, and below 0.667 is conventionally considered insufficient for drawing reliable conclusions. Nine of ten emotion categories fall below 0.667. Only sentiment (0.738) and joy (0.699) approach adequate reliability. The paper compares to MELD (Fleiss' kappa = 0.43) and IEMOCAP (0.48), but these use a different metric (Fleiss' kappa vs. Krippendorff's alpha), making the comparison misleading. Additionally, with only 1–3 annotators per clip and "majority vote" aggregation, clips with a single annotator have no majority to compute, creating potentially unreliable ground truth labels.

- **Emotion cue grounding task is purely qualitative with no quantitative evaluation**: Section 4.1 defines emotion cue grounding as identifying "video frames and spatial regions relevant to sentiment analysis and emotion classification," but Section 5.3 evaluates this solely through manual inspection of "several randomly selected videos" (the paper's own characterization: "To obtain a preliminary understanding"). No quantitative metrics, systematic evaluation protocol, or reproducible methodology are provided. This means one of the paper's three benchmark tasks lacks any empirical grounding.

### Minor

- **Small dataset with limited signer diversity**: With 200 clips from only 4 signers in a lab setting (ASLLRP corpus), the ecological validity and generalizability of findings is limited. The paper acknowledges this limitation and justifies the size by comparison to other small-but-valuable benchmarks, which is reasonable given the difficulty of recruiting Deaf expert annotators.

- **Missing 23 clips from partition description**: The single-expression set (140 clips) and multi-expression set (37 clips) total 177, but the full dataset contains 200 clips. The status of the remaining 23 clips is not clearly explained, which could affect the interpretation of class distributions and per-class metrics.

- **Intensity annotations collected but discarded**: Annotators provided emotion intensity on a 0–3 scale, but the binarization step discards this information, reducing what could have been a richer evaluation to binary presence/absence categories where agreement is already low.

## Trivial
None.

## Nice-to-Haves

- **Stratified analysis by VADER–annotator agreement**: Show whether model failures concentrate on clips where text sentiment and human labels diverge, which would directly test whether the VADER selection confound drives the observed text-reliance behavior.

- **Quantitative grounding evaluation**: Define formal metrics for the grounding task (e.g., temporal IoU with annotator-identified cue windows, or automated matching of described cues against annotator descriptions).

- **Release per-annotator labels**: Given the low agreement on many emotions, releasing individual-level annotations would enable soft-target or uncertainty-aware evaluation, significantly increasing the dataset's research utility.

## Removed Points

- **"VADER selection fundamentally confounds multimodal evaluation" (Harsh Critic, structural)**: The harsh critic overstates this as "fundamentally confounding" and claims the finding is "partially an artifact." Retained as a **major** weakness, but downgraded from fatal because the video-only near-chance performance still demonstrates genuine model inability to process visual ASL emotion cues — even with text-biased selection, models should have been able to learn from visual information if they could process it. The confound primarily affects the *interpretation* of why caption-only outperforms video-only, not the finding that video-only performance is poor.

- **"23 missing clips" (Harsh Critic)**: Retained as minor — the partition sizes (140 + 37 = 177) vs. 200 total clips is unexplained, but this is a presentation gap, not a methodological flaw.

- **"Statistical significance testing needed" (Harsh Critic)**: Removed — standard practice for benchmark papers at this scale does not typically include significance testing. This is a nice-to-have, not a weakness.

- **"Per-class accuracy on few samples is meaningless" (Harsh Critic)**: Partly valid but overstated — the paper reports per-class numbers to show model *biases* (e.g., GPT-4o defaulting to happiness/frustration), not just accuracy. This is partially addressed by using weighted metrics as the primary evaluation.

- **"Ecological validity concerns about ASLLRP lab setting" (Harsh Critic)**: This is a known tradeoff that the paper already discusses. Removed as a weakness — the paper explicitly chose ASLLRP for quality and reproducibility and acknowledges the limitation.

- **"Missing related works" (Strength Finder)**: Removed per instructions — no external source verification available.

- **"Reasonable inter-annotator agreement contextualized against MELD and IEMOCAP" (Strength Finder)**: Removed as conflicting with a verified major weakness (low alpha on most emotion categories and misleading metric comparison).

- **"Reproducible experimental setup" (Strength Finder)**: Downgraded to trivial — while the setup is documented, claiming "reproducible" when some details are in the (stripped) appendix is overstated. Removed from strengths.

## Novel Insights

The most striking finding is the asymmetry where adding video to captions provides a meaningful boost for sentiment analysis (GPT-4o: 49.53 → 76.72 wF1 in 3-class) but *not* for emotion classification (GPT-4o: 55.89 → 55.09 wF1). This suggests that current MLLMs can extract global affective polarity from visual cues when a textual anchor exists, but cannot leverage those same cues for finer-grained emotional distinctions — a finding consistent with the Deaf annotators' observation that ASL emotion cues require distinguishing grammatical from affective functions, a disentanglement that current models appear unable to perform.

## Suggestions

- Add a subgroup analysis splitting benchmark results by whether VADER sentiment and annotator labels agree or diverge, to disentangle dataset artifact from genuine model limitation.
- Define at least one quantitative metric for the grounding task (e.g., BERTScore or LLM-judge matching between model descriptions and annotator cue descriptions), so it can serve as a reproducible benchmark rather than a qualitative observation.
- Release per-annotator labels alongside aggregated labels to enable research into label uncertainty and multi-annotator training objectives.

## Calibration and Score Justification

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| MMAU | TeVAZXr3yv.md | 7.5 | Novel audio multimodal benchmark with expert annotations, comprehensive evaluation (10k clips, 18 models). EmoSign is substantially smaller, with weaker methodological validation and no quantitative grounding evaluation — clearly below this. |
| MediConfusion | H9UnNgdq0g.md | 6.25 | Small (352 pairs) expert-driven medical VQA benchmark. Similar profile to EmoSign: novel domain, expert-validated, reveals model failures. EmoSign has additional structural concerns (VADER confound, low agreement). Below this. |
| EndoAssistant | voYshhbWeJ.md | 5.0 | Large-scale medical dataset with expert validation but limited experimental novelty. EmoSign has stronger novelty (first ASL emotion dataset, Deaf annotators) but more severe methodological issues. Comparable. |
| MMMT-IF | ic153qXFfx.md | 4.0 | Multimodal benchmark derived from existing data, annotation reliability concerns, qualitative-only evaluation aspects. EmoSign has stronger domain novelty and community impact but similar methodological weaknesses. |
| AVCaps | FFUmPQM8c5.md | 4.0 | Small-scale multimodal dataset (2061 clips) with weak experimental validation. EmoSign has stronger motivation (Deaf community, ASL) but similar scope and evaluation limitations. |

EmoSign sits between EndoAssistant (5.0) and AVCaps/MMMT-IF (4.0). Its domain novelty and community significance (first ASL emotion dataset with Deaf annotators) are genuine strengths that push it above the 4.0 anchors. However, the VADER selection confound, low inter-annotator agreement on most emotions, and the purely qualitative grounding evaluation are significant methodological weaknesses that keep it below 5.0.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>