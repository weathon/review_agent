## Human Reviewer 1

### Summary
This paper introduces BANZ-FS, a dataset for two-handed BANZSL fingerspelling, combining ABC News Auslan broadcasts, lab multi-camera RGB-D recordings, and web-sourced videos. The dataset contains >35k annotated fingerspelling segments from 116 signers, with multi-level alignments (video ↔ subtitles ↔ letters ↔ lexicon). The authors define and benchmark four tasks (IFSR, FSD, FSD-R, FSR-Context), evaluate pose- and RGB-based baselines, present cross-domain experiments, and report qualitative case studies.

**Overall:** the dataset is useful and fills a real gap; however, the manuscript needs clearer documentation of procedures, metrics, and ethics before the benchmark claims can be fully trusted.

### Strengths
**Presentation:** The paper is clearly structured (Intro → Related Work → Data Collection → Statistics → Tasks → Benchmarks → Case Studies). Tables summarizing dataset comparisons and statistics (Table 1 & Table 2) are informative and well-labeled. The taxonomy of tasks (IFSR, FSD, FSD-R, FSR-Context) is clearly defined and consistent throughout.

**Contribution:** The proposed BANZ-FS dataset addresses a tangible linguistic and practical gap by focusing on two-handed BANZSL fingerspelling, a setting largely underrepresented in current literature. Its annotation depth (multi-level temporal alignment, OOFS/singleton reporting, and FS speed distribution) and multi-domain composition (news, lab multi-view, web) provide meaningful diversity and reusability for the community.

**Techniques:**

1. A well-designed multi-domain acquisition strategy combining professional broadcast data, lab-controlled RGB-D recordings, and in-the-wild web footage, enabling both realism and controlled benchmarking.

2. Rich, fine-grained annotations supported by comprehensive statistics (e.g., long-tail letter frequency, signer diversity, OOFS/singletons, FS speed).

3. A complete benchmark suite spanning isolated recognition, detection, joint detection-recognition, and contextual evaluation, covering both constrained and realistic use cases.

4. Practical qualitative case studies that transparently illustrate typical failure modes such as rapid fingerspelling or loose boundary segmentation, adding diagnostic insight to the quantitative results.

### Weaknesses
**1. Segmentation design lacks empirical justification:** Section 3.3 adopts a fixed 10s sliding window without ablation or rationale. An analysis of alternative window lengths (e.g., 5s, adaptive) is needed to confirm robustness.


**2. Annotation reliability not quantified:** Although multi-level verification is mentioned (Section 3.1), no inter-annotator agreement (e.g., κ, α) or adjudication detail is provided. This weakens confidence in temporal boundary quality.


**3. Evaluation criteria under-specified:** Metrics such as AP@IoU and AP@Acc (Section 5.2) lack exact definitions and sensitivity checks. Clear reporting of thresholds and pipeline details is essential for reproducibility.


**4. Cross-domain variability insufficiently analyzed:** Results in Table 3 - 4 reveal strong domain gaps, yet no normalization or domain ablation is provided. A domain-wise feature analysis (e.g., t-SNE) would clarify generalization behavior.


**5. Error analysis remains anecdotal:** Case studies (Section 5.3) illustrate failures qualitatively but lack quantitative breakdowns, e.g., per-letter confusion or temporal precision - which would turn examples into diagnostic insight.

**6. Signer-split clarity:** Section 3.3 does not specify whether the splits are signer-disjoint. Please clarify whether signer identities overlap across splits, and report this information explicitly.

**7. Ethical transparency missing:** No explicit consent, license, or release protocol is reported, though videos include identifiable individuals. An ethics statement is required for dataset release.

### Questions
Most of my concerns and suggestions are already reflected in the Weaknesses section. I believe that clarifications or additional experiments addressing these issues would substantially strengthen the paper.

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
4

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper introduces BANZ-FS, a new, large-scale dataset focused on two-handed fingerspelling for BANZSL (British, Australian, and New Zealand Sign Language). The authors highlight that existing sign language datasets rarely address the specific challenges of two-handed fingerspelling, focusing instead on single-handed systems like American Sign Language (ASL). The dataset features rich, multi-level annotations, including video-to-subtitle, video-to-letter, and video-to-lexicon alignments. It is designed to capture real-world linguistic phenomena like spelling errors, acronyms, and abbreviations , as well as visual challenges unique to two-handed signing, such as self-occlusion and rapid transitions. The authors benchmark state-of-the-art models on tasks like fingerspelling detection, isolated recognition, and recognition in context, demonstrating that the dataset presents significant challenges for current methods.

### Strengths
1. Novel and Specific Contribution: This is the first large-scale fingerspelling dataset specifically for the two-handed BANZSL system. It fills a critical gap, as most resources are for single-handed systems.

2. High Data Diversity: By combining news, lab, and web sources, the dataset captures a broad spectrum of signing styles. This includes formal signing from professionals, clean signing in controlled settings, and casual "in-the-wild" signing from 116 unique signers.

3. Linguistic Realism: A major strength is the inclusion and annotation of real-world linguistic phenomena often missing from clean datasets. This includes spelling errors (e.g., "Maguire" signed as "Maquire"), lexical abbreviations ("EQ" for "equipment"), acronyms ("GWS"), and self-corrections.

4. Rich, Multi-Level Annotations: The dataset provides comprehensive annotations beyond simple transcription. This multi-level alignment supports a variety of tasks, including detection (temporal boundaries), isolated recognition (letter sequences), and contextual recognition (alignment with spoken language lexicons).

5. Robust Benchmarking: The authors establish strong baselines by testing existing models on the new data. The results (Tables 3 & 4) show that current methods struggle with cross-domain generalization (e.g., training on Lab data and testing on Web data), proving the dataset is a challenging and valuable resource for advancing the field.

### Weaknesses
1. The methods evaluated in Tab.3 and Tab.4 are a bit old, which are mostly before 2023. For a paper submitted to ICLR 2026, new state-of-the-art methods could be incorporated.
2. As shown in Tab.2, the dataset is "heavily skewed toward Auslan" (Australian Sign Language). The BSL (British) and NZSL (New Zealand) contributions are "relatively limited" , primarily coming from the smaller "Web Videos" and "YouTube" sources.
3. While the proposed dataset highlight the "in-the-wild" characterisitic, this is not fully reflected in the data construction and composition. Most videos are still captured from News or Lab environments.
4. Some labelling steps remain confusing. For example, how to (2) identify the signer ID based on pose trajectories? Besides, only a little data is cross-checked. How to guarantee the data quality is a question.

### Questions
See above

### Soundness
4

### Presentation
3

### Contribution
4

### Rating
8

### Confidence
4

---

## Human Reviewer 3

### Summary
The authors introduce BANZ-FS, a new dataset for two-handed fingerspelling detection and recognition. The dataset originates from three sources, including web recordings, news broadcasts, and lab recordings in a controlled setting. Over 35k video-aligned instances are collected, aligned on a video-subtitle, video-fingerspelled letters, and video-target lexicon level.

This paper tackles a particular shortcoming of other papers in the field of fingerspelling recognition, which is that the datasets used in this field are too simple and not representative of real-world fingerspelling. That is, they are often based on images of individual fingerspelled letters, instead of videos of sequences of fingerspelling. The latter are far more difficult to recognize accurately due to co-articulation and fast spelling (and due to the fact that it's video data, not image data). In contrast, the former is a trivial task.

The authors detail their data collection, cleaning, and labelling techniques for the web data in sufficient detail. Subtitles, which are noisy annotations, are cleaned by the authors to obtain 30k cleaned subtitles. Then, signing experts verify alignment of subtitles and video data, refining it if necessary. Annotations are cross-checked by other annotators (5%). 

Lab data is collected with multiple RGB-D cameras, and participants were properly informed of their rights according to an ethical protocol. Signing levels were recorded, though it is not immediately clear if hearing status correlates to signing level.

### Strengths
The paper is clearly written and easy to follow.

The authors evaluate multiple models from the literature on their datasets, leading to a rigorous evaluation of not only the dataset, but also of these models. This is a valuable contribution to the field.

The authors also list several fingerspelling related tasks and evaluate models for each of them, which is also valuable. In fact, this makes this paper a good introduction to the field of fingerspelling recognition.

### Weaknesses
I would have liked to read more information on the datasets. More information is given in the "Questions" section below. But, for example, "volunteers" were selected for fingerspelling data collection. Were these volunteers in any way checked for signing proficiency? What if they submitted particularly bad data? Fingerspelling is not trivial at all.

Similarly, the lab collection of data could have had more information. For example, several RGB-D cameras are used. Are all data used? What about depth? This section could be expanded.

It is also not clear if spelling errors were kept in annotations or not. This seems like an important aspect to mention.

There are several spelling mistakes throughout the paper, but these could easily be fixed if accepted.

### Questions
How are spelling errors in labels handled? Should they remain or not?

Page 5. For the web dataset, you identify signer IDs based on pose trajectories. Did you use their Pose Flow system? If so, I would mention this explicitly in the paper.

Page 5. You say that the fingerspelling segments are annotated. How is this done? Are segments delineated in time? Is the entire word annotated? Do you keep the intended spelling or the spelling as it is (including possible errors?)

Page 5. Can you explain what "lexical forms of fingerspellings" are?

Page 6. How does hearing status relate to signing level?

Please perform a check for grammar and spelling. For example, on page 3, you write "recognition recognition" and in the appendix you misspelled "Isolated".

Did you in any way use the depth information from the lab setup? Would this have been useful? How do you combine information from multiple cameras?

### Soundness
3

### Presentation
3

### Contribution
4

### Rating
8

### Confidence
4

---

## Human Reviewer 4

### Summary
Here, the authors present a new dataset and benchmarks for fingerspelling detection and recognition, focused upon the two-handed manual alphabet common to the British, Australian and New Zealand Sign Languages, arguing that this covers a gap in fingerspelling datasets that largely focus upon one-handed manual alphabets (e.g. American Sign Languages). The authors curate and annotate fingerspelling segments from three sources (Auslan interpretations from a news broadcast, YouTube videos, and a lab-collected dataset generated by the authors), and assess both in-domain and out-of-domain performance across models.

### Strengths
The authors have curated a comprehensive dataset that addresses a critical gap in fingerspelling detection and recognition research: their curation of not just a large quantity of videos, but through different sources and annotated with various metadata is valuable. The authors correctly note that two-handed manual alphabets pose additional complications not seen in one-handed manual alphabets (e.g. occlusion), and so it is essential that they're releasing this resource. The benchmarks are comprehensive, spanning numerous state-of-the-art models, but also critically assess performance both within and outside of domains. This is overall strong data collection and benchmarking work.

### Weaknesses
1. Although the authors have gone to lengths to collect metadata about segments/videos, I wish there was a more thorough analysis of this. In Tables 3/4, the authors report overall performance, but it would have been really interesting to conduct stratified analyses of what factors impact fingerspelling recognition and detection performance beyond domain. For example, is detection/detection+recognition more challenging with NZSL and BSL, given that the contexts in which fingerspelling occurs is different than the majority of their training data across domains? Are tasks more difficult on Deaf signers, who might fingerspell faster/more fluidly than language learners? Are infrequent letters worse in their performance?

1a. The authors claim that their dataset spans three sign languages - but at least one dataset are exclusively Auslan. It's moreover unclear how much of the dataset is BSL/NZSL - the authors should report this, especially because it is central to their claim that their dataset spans three languages. Although the authors do acknowledge their dataset is imbalanced - to what degree? This should be reported.

2. More details about the stratification of datasets would be critical and are currently missing from the paper. Are the splits a purely random split, or is there some attempt to stratify within each data source such that scenarios/signers are unseen? If the splits are purely random, this might impact interpretation: e.g. if the same signer is in both the test and training dataset, and fingerspelling the same word, in-domain performance may be overinflated by interpretation. 

3. The authors provide some examples of fingerspelling phenomena in Figure 2 (abbreviations, acronyms, spelling errors, and self-correction), but it's unclear to me how they actually deal with them in the dataset - I can see arguments both ways. On one hand, it might be important for fingerspelling recognition systems to report the fingerspelling as-is. On the other hand, people perceiving signing will naturally correct and contextualize these issues in language recognition - and critically, providing written language transcriptions of fingerspelling can strip the necessary context to correct (for example, in self-correction, signers will often shake their heads, make facial expressions, or pause to indicate a correction, and these gestures won't make their way into the written language transcription), so it may be important to provide a corrected form. In other words - "recognition" and "translation" are sometimes at odds. Can the authors better document their design decisions and rationale here on what we consider as ground truth? 

4. There are places where the authors could be more culturally sensitive. In line 228, the authors refer to the community as "deaf and hearing-impaired" viewers, whereas they use "Deaf and hard-of-hearing" in the introduction. The latter is generally considered more appropriate by Deaf communities. In Figure 3, the authors distinguish between "Deaf individual" and "Auslan expert", but these are not mutually exclusive categories.

### Questions
(See weaknesses)

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
5