The paper content is already provided in the user message. Let me now carefully verify the key claims and synthesize the final review.

## Summary

EmoSign introduces the first ASL video dataset with fine-grained sentiment ratings, emotion intensity scores, and qualitative emotion cue descriptions, annotated by 3 Deaf native ASL signers. The dataset comprises 200 clips (~16 minutes) sourced from the ASLLRP corpus and provides baseline benchmarks with 4 multimodal LLMs, revealing that current models rely heavily on text captions and struggle to process visual emotion cues in sign language.

## Strengths

- **Novel and important domain contribution.** The paper addresses a genuinely understudied gap—emotion recognition in sign language—where facial expressions and body movements simultaneously serve grammatical and emotional functions. No prior ASL dataset includes fine-grained emotion labels and cue descriptions from Deaf native signers, making EmoSign a unique resource.

- **Culturally informed and methodologically sound annotation design.** Recruiting Deaf native signers with professional interpretation experience (rather than hearing annotators) is a significant methodological strength, given documented misinterpretation of signers' expressions by non-signers (Lim et al., 2024). The three-layer annotation (sentiment, emotion intensity, open-ended cue descriptions) provides richer information than prior datasets like FePh, which only offered binary labels on cropped face images.

- **Clear and interpretable ablation design.** The three-condition evaluation (caption-only, video-only, video+caption) provides straightforward diagnostic insights into how multimodal models use different input modalities, and the finding that video-only performance is dramatically worse than caption-only is a clear and important result.

- **Valuable qualitative analysis of cue descriptions.** Section 3.4's thematic analysis of annotator-provided emotion cue descriptions (facial expressions, sign modifications, role/context markers) provides genuine linguistic insight into how emotions manifest in ASL, going beyond surface-level labeling.

- **Ethical care in data collection.** IRB approval, months of community engagement, annotator training, and allowing annotators to skip uncomfortable content demonstrate responsible research practices.

## Weaknesses

### Major

- **Overstated claims relative to dataset scope and construction.** The paper describes EmoSign as a "first comprehensive dataset" and claims it "establishes a new benchmark for understanding model capabilities in multimodal emotion recognition for sign languages." However, the dataset contains only 200 clips (~16 minutes) from 4 signers in a single lab corpus, pre-selected as the 100 most positive and 100 most negative by VADER scores on English captions. This bimodal, text-preselected, single-source construction limits ecological validity and constrains what can be concluded about model capabilities broadly. The dataset is more accurately characterized as a valuable pilot/seed dataset that highlights important challenges—not a comprehensive benchmark. These overclaims should be tempered significantly.

- **VADER-based preselection structurally couples text sentiment to labels, undermining the central scientific claim.** The paper's key empirical finding is that "current multimodal models fail to integrate visual cues and heavily rely on text captions for emotion reasoning." However, the dataset was built by selecting exactly those clips whose English captions were flagged by VADER as having strong sentiment. This means: (1) the dataset systematically excludes cases where visual emotional nuance diverges from text—the very scenario the paper argues is most important—and (2) the captions provided to models are pre-correlated with the task labels by construction. The conclusion that models "fail to integrate visual cues" is confounded by this design, since captions were the basis for dataset inclusion. The weaker finding—that visual-only performance is poor—is valid, but the stronger claim about modality dominance is not well-supported without analyzing cases where text and visual cues diverge.

- **Small size and limited signer diversity raise generalizability concerns.** With only 4 signers from one corpus in a controlled lab setting, there is a substantial risk that models could learn signer-specific idiosyncrasies rather than generalizable emotion cues. The paper does not analyze performance variation across signers or discuss what generalization means with such limited diversity. This is a known issue in sign language research—the SignAvatars reviewers raised similar concerns about dataset scope limitations.

- **Low inter-annotator agreement for several emotion categories undermine labels as ground truth.** Krippendorff's alpha for surprise-negative (0.119), disgust (0.166), frustration (0.330), and sadness (0.333) are at or below conventional thresholds for reliable coding (0.333). The paper acknowledges these numbers (Table 2) but does not analyze what they mean for benchmark validity, whether these categories should be merged or excluded, or how disagreement patterns relate to the paper's core thesis about the difficulty of disentangling grammatical from affective expressions. This deserves explicit discussion.

### Minor

- **Emotion cue grounding task lacks quantitative evaluation.** Section 5.3 defines emotion cue grounding as one of three benchmark tasks but provides only manual inspection of selected outputs, with no quantitative metric (e.g., keyword overlap with annotator descriptions, temporal IoU). For a dataset paper that claims to "establish a new benchmark," this leaves one of three tasks without a measurable evaluation.

- **No fine-tuning or adaptation experiments.** All baselines are zero-shot evaluations of general-purpose MLLMs. While this reveals current out-of-the-box limitations, it does not explore whether the dataset's annotations could actually improve models through fine-tuning or adaptation. Including even one fine-tuning experiment would strengthen the paper's claim about the dataset's value.

- **Label aggregation and binarization are underspecified.** The paper uses "majority vote" on ordinal scales (0–3 for emotions, -3 to +3 for sentiment) with tie-breaking by "most confident annotator," but does not detail how presence/absence was binarized for Figure 2C or how the "single dominant emotion" was determined for the single-expression subset. These design choices directly affect the target labels and should be explicit.

- **No statistical significance tests.** Point estimates on 200 samples (with further subsetting to 140 for single-expression) are used to make detailed claims about model biases and modality contributions. Small differences like wF1=55.09 vs 55.89 (Table 4) are treated as meaningful without any confidence intervals or statistical tests.

## Nice-to-Haves

- Analyze model performance stratified by text–label alignment (clips where VADER agrees vs. disagrees with annotator sentiment) to directly test whether visual cues matter independently of text.
- Include at least one model specifically designed for affect recognition (e.g., facial action unit detector) as a baseline to distinguish sign-language-specific challenges from general MLLM weaknesses.
- Report per-signer performance to assess generalization risk.
- Expand the emotion cue grounding analysis with a quantitative metric to make it a proper benchmark task.

## Removed Points

- **"No dedicated affect recognition model baselines"** — While including specialized models would strengthen the paper, this is a dataset/resource paper and zero-shot MLLM evaluation is a reasonable starting baseline. Moved to Nice-to-Have since this falls outside the paper's stated scope of establishing initial baselines.

- **"Ethical considerations and community engagement insufficient"** — The paper describes IRB approval, sustained community engagement, and annotator choice protocols. Demanding more extensive ethics discussion is scope creep for a dataset paper; the existing ethical care is adequate.

- **"Missing related works"** — Not flagging missing citations per instructions.

- **"No demographic details about annotators"** — The paper states they are Deaf native signers with professional interpretation experience recruited through a third-party vendor. Demanding more detail risks identifying anonymous annotators. This is a minor reproducibility concern but not a substantive weakness.

- **"Distribution bias from VADER selection should be discussed more"** — This is partially addressed in Section 3.1 where they acknowledge VADER results "differed from the annotators' results often contained rich non-manual markers that conveyed emotions differently than the text." The deeper structural concern (text–label coupling) is kept as a major weakness above.

- **Formatting and presentation nitpicks** removed per instructions.

## Novel Insights

The paper's most insightful finding is that even when provided with video input, multimodal models often construct post-hoc visual justifications that align with their text-based sentiment judgments rather than genuinely grounding their reasoning in visual cues (demonstrated in Figure 3 where identical visual cues are interpreted oppositely depending on caption availability). This aligns with broader literature on multimodal grounding failures but documents it specifically in the sign language domain where visual cues are linguistically essential.

## Suggestions

- Reframe EmoSign as an initial/pilot dataset rather than a comprehensive benchmark. Temper claims like "first comprehensive dataset" to "first dataset with fine-grained emotion labels and cue descriptions."
- Add a dedicated analysis of cases where VADER text sentiment and annotator labels diverge. This directly tests the paper's core motivation about the importance of visual emotion cues beyond text.
- Either add a quantitative grounding metric or clearly position the cue grounding analysis as exploratory rather than a benchmark task.
- Merge or exclude emotion categories with very low inter-annotator agreement, and explicitly discuss what these low agreement levels mean for benchmark reliability.
- Report per-signer and per-condition confidence intervals to support comparative claims about model performance.

## Evaluation on Key Axes

**Originality**: High. This is the first dataset of its kind for ASL emotion recognition with Deaf annotator labels and cue descriptions. The domain is genuinely novel.

**Importance of research question**: High. Emotion recognition in sign language has real consequences (legal, medical settings) and is technically challenging due to the dual grammatical/emotional role of visual cues.

**Claims supported by evidence**: Partially. The core dataset contribution is solid, but several claims about model behavior (modality dominance, positive bias) are confounded by the VADER-based dataset construction, and the "comprehensive benchmark" framing overreaches.

**Soundness of experiments**: Moderate. The ablation design is clean, but small N, no statistical tests, and the structural confound of text-based preselection limit what can be concluded.

**Clarity**: Good. The paper is well-written and the motivation is clearly articulated.

**Value to the research community**: Moderate-to-High as a pilot dataset that opens an important direction; Lower if positioned as a definitive benchmark.

## Score and Decision

**Calibration**: Compared against EIBench (emotion benchmark with limited evaluation, scores 3-5, rejected), OV-MER (emotion dataset with label quality concerns, scores 3-8, rejected), SignAvatars (sign language dataset with scope concerns, scores 5-8, rejected), and YouTube-SL-25 (large-scale sign language dataset, scores 6-8, accepted). EmoSign has a genuinely novel and important domain contribution with culturally informed annotations, but its scientific claims are undermined by the VADER preselection confound and small scale. It sits above EIBench (which had more fundamental issues) but below YouTube-SL-25 (which had much greater scale and diversity). The structural confound is a serious issue that limits how much can be concluded from the benchmark results.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>