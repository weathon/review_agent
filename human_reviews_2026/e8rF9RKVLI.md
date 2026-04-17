# SVDF-20: A LARGE-SCALE MULTILINGUAL BENCHMARK FOR AI-GENERATED SINGING DETECTION

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 6, 2

## Abstract
As generative models replicate human singing with uncanny precision, detection systems must operate reliably across all languages, not just English or Mandarin. Current detectors fail catastrophically on unfamiliar languages, a critical gap we address with SVDF-20, the first comprehensive multilingual singing voice deepfake detection benchmark. Our contributions are threefold: (1) We provide a quality-controlled dataset of 24,421 songs ($1,475.6$ hours) across 20 languages, introducing $87%$ novel linguistic content compared to existing resources—including all 10 major Indic languages previously absent from singing voice deepfake detection research. (2) We demonstrate through experiments on eight architectures that multilingual training is essential: models trained on limited languages degrade to $45%$ Equal Error Rate (EER) on diverse languages, while SVDF-20-trained models achieve a $31%$ relative improvement, maintaining robust detection across all linguistic contexts. (3) We establish evaluation protocols with singer-disjoint splits and codec robustness tests that reveal how linguistic diversity fundamentally changes what models learn, shifting from language-specific patterns to universal synthesis artifacts. These findings establish that SVDF-20 enables the development of deepfake detectors capable of safeguarding musical authenticity globally, not just in data-rich languages.

Data and Code: \href{https://anonymous.4open.science/r/SVDF20-D328/}{https://anonymous.4open.science/r/SVDF20-D328/}

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces SVDF-20, which is a new large-scale dataset and benchmark for singing‐voice deepfake detection covering 20 languages. The dataset is created with matched bona fide vs. AI‐generated vocals, obtained via a controlled YouTube-based pipeline and vocal separation. The authors conduct experiments with various SVDD architectures, and ablate reseach questions relevant to the generalizability of monolingual models to multi-lingual songs.

### Strengths
1. The paper considers research questions regarding the generalizability of synthetic music detection models to unseen languages and provides a dataset and evaluation methodology for this. 
2. The multi-lingual music collection is generalized, providing a very good dataset for multi-lingual music generation and detection.

### Weaknesses
1. One of the core motivation for this paper is the question: "Can SVDD models trained on limited languages generalize to unseen ones?". However, there is no dataset split that directly measures this. T04 split measures unseen distribution, not strictly unseen languages. Thus, later results don't directly answer the core motivation question.
2. The deepfake samples come from heuristic YouTube queries. The paper lacks detail on how correctness of labels was ensured. If many samples come from a single synthesis method or user community, models might learn dataset-specific artifacts. The potential bias in what content appears on YouTube (genres, singers) is not addressed. Furthermore, there is no human evaluation of the dataset, which would have provided some validation for the dataset creation process. 
3. There is a discrepancy in the number of bonafide and deepfake songs for some languages. This could introduce bias in training and evaluation.
4. The evaluation shows benefits over training on SingFake, but it would be informative to see performance comparison over other datasets. For example, the performance of a model trained on SONICS or on CTRSVDD. Furthermore, only EER on the dataset level is shown. But extending this to other metrics such as F1 scores would be great.
5. The evaluation is done on vocal-only clips. This doesn't capture the nuances of a 'song', as music and relevant background information also provide essential information. So the background information can provide enough information for existing fake detection methods to detect the song is fake. This calls into question the necessity of multi-lingual training in this domain, without supporting evidence. Also, a model is used to extract the vocals which might add further artifacts.

### Questions
1. Did the authors conduct evaluation of existing methods on a unseen languages test-set?
2. Can the authors provide more details regarding their fake song generation setup? 
3. Did the authors conduct any human evaluation (both quantitative and qualitative) of the generated dataset, to ensure correctness?
4. Did the authors conduct cross-dataset training and evaluation experiments such as training on SONICS and evaluation on their multilingual dataset?
5. The paper shows that speech recognition and representation learning can generalize to unseen languages, but SVDD models don't. Can the authors provide any explanation for this?
6. Can the authors provide evidence that multilingual understanding is required for fake song detection when music is included in the vocal-only clips?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents SVDF-20, a multilingual singing-voice deepfake detection benchmark spanning 20 languages (10 Indic + 10 global), comprising 24421 songs (~1476 h) segmented into 388631 clips across codec variants. The work demonstrates that multilingual training yields 13.78 pp EER improvement on cross-domain evaluation versus monolingual baselines.

### Strengths
This is the first large-scale multilingual SVDD benchmark with substantive Indic representation; clear community need. Singer-disjoint splits, codec augmentation, and multi-tier evaluation protocol (T01–T04) demonstrate thoughtful construction from authors. Despite undertraining, the multilingual advantage appears across all eight architectures tested, suggesting a robust (if underestimated) effect.

### Weaknesses
(1) 25 epochs cannot be defended for a dataset of this scale. The paper should either (a) extend training to convergence and revise claims accordingly, or (b) explicitly reframe as a "dataset paper with baseline experiments" rather than claiming definitive conclusions about architectural comparisons.

(2) No per-language metrics to assess fairness or identify low-resource failure modes; No learning curves to validate that 25 epochs represents reasonable stopping; and No feature-space analysis (e.g., t-SNE by language, language-adversarial probing) to substantiate the invariance claims beyond aggregate EER. 

(3) Equations 4, 5 formalize the invariance hypothesis; but this hypothesis was never empirically measured. I recommend the authors to include quantitative divergence metrics (MMD, JS) on learned representations.

(4) While metadata-only release is stated, multi-jurisdictional sourcing from platforms requires clearer documentation of takedown handling, consent mechanisms (especially for lesser-known singers), and re-download rights for researchers.

### Questions
(1) Can you provide validation loss curves or per-epoch EER trajectories to demonstrate that 25 epochs approaches a plateau? Current absolute EERs (31–38%) suggest otherwise.

(2) What are EERs for each language independently? Are low-resource Indic languages disproportionately harmed?

(3) What happens if you train on only Indic-10 or global-10 subsets? Does the multilingual benefit scale linearly with language diversity?

(4) Real-world SVDD can also operate on full songs with accompaniment (the "mixture" setting). How do these models perform on non-vocal-isolated audio?

(5) Can you quantify Eq. 4–5 using representation divergence metrics across language pairs as previously mentioned?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduced a large-scale benchmark dataset for singing voice deepfake detection (SVDD). The authors proposed a dataset consisting of 20 languages (which includes 10 major Indic languages underexplored in SVDD research). They showed rigorous experiments across eight diverse architectures to show that the models trained on their dataset outperform on less diverse data.

### Strengths
The primary strength of the paper is the diversity of the languages in their SVDD dataset, which is expected to allow generalizability across more languages. 

The authors have conducted rigorous experiments to isolate the impact of the training data and utilized the Singfake dataset to compare the improved out-of-distribution generalizability to language.

### Weaknesses
While the paper introduces a new benchmark, it did not manage to strongly establish the importance of a multi-lingual dataset of this level of diversity. The authors utilized multiple models to compare the difference in performance between Singfake and SVDF-20. But the difference in their number of samples (Singfake dataset has a total of 16k samples, whereas SVDF-20 has a total of 388k samples) reduces the impact of this finding, since the improved generalizability can also come from simply having more samples rather than having diversity of languages. A stronger claim would require a control experiment of comparing against a less linguistically diverse or even monolingual dataset with a comparable size to strongly prove that the advantages are actually from the diversity of language, not simply due to having more training data.

Inadequate information regarding the generation methods (Singing Voice Conversion, Singing Voice Synthesis, or even end-to-end song generation) used in this dataset limits the utility of this dataset for diagnosing where the model is failing.

### Questions
1. Could the authors discuss how they can prove the observed performance gains are due to having diverse languages and not simply the higher data volume?
2. Could the authors provide more details regarding the generation methods present in their dataset?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces SVDF-20, a large-scale, multilingual dataset for singing voice deepfake detection (SVDD). It includes 24,421 songs across 20 languages (10 Indic and 10 global), covering ~1,475 hours of audio. The authors position this as the most linguistically diverse SVDD benchmark to date, aiming to answer three questions: (1) how multilingual data affects SVDD performance, (2) whether multilingual training improves cross-lingual generalization, and (3) whether the dataset helps bridge domain gaps compared to prior English or East-Asian–focused datasets such as SingFake, CTRSVDD, and SONICS. They train and evaluate several open-source audio models (AST, WavLM, AASIST, etc.) on these setups and conclude that multilingual training improves robustness, though performance naturally degrades when moving across unseen languages.

### Strengths
- **Data effort:** The authors clearly spent effort in curating and cleaning multilingual singing data. 

- **Transparency in setup:** The paper reports training settings, architectures, and language groupings in a reproducible manner. It’s easy for others to replicate or extend the experiments.

### Weaknesses
1. **Contribution is incremental.**  
   The paper mainly extends prior multilingual datasets rather than introducing a novel task or method. *SingFake*, *CTRSVDD*, and *WildDeepfake-SVDD* already include multilingual material. This work’s novelty is scale, not concept.

2. **Misalignment around SONICS.**  
   The paper repeatedly frames *SONICS* as an “advancement of SVDD,” but SONICS is a **synthetic song detection** task, both vocals and accompaniment are generated. SVDD, on the other hand, deals with *fake vocals over real music*. The authors conflate the two tasks, which weakens the motivation and literature positioning.

3. **No substantial modeling or algorithmic contribution.**  
   All experiments rely on off-the-shelf backbones (AASIST, WavLM, AST) and standard training setups. There’s no new detection technique, loss function, or data processing beyond dataset expansion.

4. **Research questions are shallow.**  
   RQ1 and RQ2, impact of multilingual data and multilingual training are intuitive. Everyone expects monolingual models to degrade cross-lingually, and multilingual training to help. The results merely confirm common sense without offering mechanistic or analytical insight (e.g., language-specific phoneme transfer, prosody shifts, etc.).

### Questions
My concerns are mostly about the scope and motivation rather than implementation details. One suggestion I would like to give is fix the confusion of SVDD and SONICS paper.

### Soundness
2

### Presentation
2

### Contribution
1
