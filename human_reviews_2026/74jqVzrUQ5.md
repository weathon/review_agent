# VocSim: A Training-free Benchmark for Zero-shot Content Identity in Single-source Audio

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 0, 8, 2, 2

## Abstract
The goal of general-purpose audio representations is to map acoustically variable instances of the same event to nearby points, i.e., to resolve content identity in a zero-shot setting. We introduce VocSim, a training-free benchmark that measures this capability directly on 125k single-source clips aggregated from 19 corpora spanning human speech, animal vocalizations, and environmental sounds. By restricting to single-source audio, VocSim isolates content representation from source separation confounds. We evaluate embeddings with two training-free measures: local Precision@k and a point-wise Global Separation Rate (GSR) that contrasts each item’s nearest inter-class distance with its mean intra-class distance. To calibrate GSR, we report lift over an empirical random baseline obtained by label permutation. Across diverse models, a simple pipeline—frozen Whisper encoder features with time–frequency pooling and label-free PCA—yields strong zero-shot performance. Yet VocSim also surfaces a consistent generalization gap: on blind, low-resource speech, local retrieval (P@k) drops sharply and the GSR lift over baseline is small, indicating that global class structure is only marginally better than chance. As external validation, top embeddings predict zebra finch perceptual similarity (80.9\% triplet accuracy) and improve downstream bioacoustic classification. We release data, code, and a public leaderboard to standardize evaluation of zero-shot audio similarity and to catalyze representations that better generalize across sound sources and recording conditions.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper introduces a benchmark composed of existing datasets designed for evaluating audio foundation models in a zero-shot context. The tasks specified by the benchmark involve single-source, non-soundscape audios, with the constraint that models are evaluated without access to supervised training data. As part of this work, the dataset collection is made available, a public leaderboard is planned for future release, and an initial evaluation of existing audio foundation models is provided.

### Strengths
- The paper introduces a large-scale dataset for 0-shot audio processing. A key strength is its size and straightforward accessibility via Hugging Face, lowering the barrier for entry for researchers. This increases the potential for reproducibility and follow-up work.
- The planned introduction of a public leaderboard might provide a clear framework for standardized evaluation, which might foster further research.
- The authors provide a comprehensive benchmark for zero-shot audio tasks.

### Weaknesses
**1. Critical flaws in references**

The paper's credibility is strongly undermined by what appear to be numerous hallucinated or incorrect references. This raises concerns about the validation process behind the related work section and, by extension, the rest of the paper (all datasets, results, etc.). A manual check of the bibliography revealed several major errors, including (but not limited to):

- L. 518-521 (Birb): The cited paper does not exist. The actual BIRB benchmark was published by different authors at a different venue.
- L. 588-591 (BirdSet): The citation is incorrect. BirdSet was presented at last year's ICLR by a different authors, and the provided link leads to some other source.
- L. 605-609 (MagnaTagATune): The citation lists the wrong authors, an incorrect title, and a wrong link.
- L. 626-628 (MTEB): Wrong title and wrong link.
- L. 662-664 (HEAR): Wrong authors, wrong link

**2. Clarity and Presentation**

The paper is difficult to follow due to structural and presentational weaknesses, which obstruct a clear understanding of the methodology and results.

- Undeclared abbreviations and inconsistent naming: Abbreviations for datasets (e.g., in Table 1) and models are used without being properly introduced, forcing the reader to guess their meaning. Model configurations are particularly hard to parse, with unclear terms like "pooling, label fre, opca D100 vs. 100D" used without clear definitions. The inconsistent use of terminology throughout the paper complicates comprehension.
- Incomprehensible figures and tables: The figures are not self-contained and lack sufficient explanation. Figure 1, for instance, is presented without a clear structure or narrative, making its purpose difficult to grasp. Figure 2 suffers from a poorly visible legend, rendering the results presented hard to interpret.
- Ambiguous terminology: Key concepts are not clearly defined or hard to follow. For example, the paper uses the term "acoustic-to-label consistency" without explaining what it means in this context, how it is measured, or how "inconsistent labels" are handled, especially if they are an intentional part of the benchmark design. Additionally, the naming convention of the models and techniques is very hard to follow.

**3. Methodological Choices**

The paper fails to provide a clear rationale for several design decisions, leaving the reader to question the basis for the experimental setup.

- Missing rationale for dataset selection: While the paper provides a high-level motivation for the types of audio it includes, it does not justify why these specific datasets were chosen over other potential candidates that fit the criteria.
- Unsupported claims regarding OOD evaluation: The authors claim to address OOD evaluation but fail to discuss the pre-training datasets of the foundation models. An OOD claim is only meaningful in relation to a model's training distribution. Without detailing what data the models were trained on, it is impossible to validate whether the benchmark datasets are truly OOD or to analyze the impact of pre-training data on the results (not only the models).
- Lack of motivation for model selection: The choice of models for the benchmark evaluation is not well-justified. Notably, more recent and powerful spectrogram-based models (e.g., BEATs, or EAT) are absent. The paper should explain why older models like MAE were chosen over these more modern baselines.
- Inadequate guidance on metric interpretation: The paper states the importance of reporting multiple metrics (e.g., to capture the geometry of the embedding space) but provides no guidance on how these metrics interact or how they should be interpreted together (also in context of the benchmark leaderboard). It is unclear what the combined results signify for model performance or how they should guide future development.

### Questions
- The references contains numerous significant errors (not existing papers, hallucinated authors, wrong links etc.), questioning the paper’s validity in general. Could the you explain the origin of these widespread errors and detail the process you could use to systematically audit, correct all citations and factual claims? (This is my main concern and the primary reason for the very low score.)
- The rationale for the experimental design is unclear. Could you please clarify: (a) the pre-training datasets for each model, which is essential to validate the out-of-distribution claims, (b) the specific justification for choosing these datasets over other alternatives, and (c) the reasoning for omitting more recent baseline models like BEATs or EAT from the evaluation?
- The benchmark's utility is limited by unclear terminology and interpretation. Could you provide a precise definition for key concepts, naming conventions, Figure legends (e.g., Figure 1) and offer explicit guidance on how the different evaluation metrics should be interpreted together to form a conclusive assessment of a model's performance and how they are handled in the leaderboard to rank a model.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper aggregates existing datasets into a single set allowing for the investigation of unsupervised clusters. They also introduce a new eval metric GSR.

### Strengths
- The open sourced data and code base are of a high quality. Allows for easy reproduction and further research in the area.
- It addresses a key need: better measurement of the acoustic latent space structure.

### Weaknesses
unstructured notes:
- Think you are missing some authors off:  https://arxiv.org/abs/2203.03022 (add et al)
- In the related work I think you should mention contrastive leaning methods: BYOL/DINO/Barlow twins e.g. https://arxiv.org/pdf/2209.14345
- I am skeptical on the quality of GSK as a useful metric. I agree with your points in lines 397. An addition point might be that as it is so sensitive to outliers, is the GSK not more of a measure of miss labelling rate? A label error will upper bound this metric. e.g. going through your dataset some of the mosquito sounds to me sound way more like a trumpet (for example). Another artefact might be a more complex cluster structure not described by the class labels. In fact we see this in figure 1 d) where the red class is split. (noted on strong correlation with other similar metrics)
- B2 details that the dataset is a collection of existing datasets. Please make this clearer in the main text. 
- A flow chart would be nice for the data curation.
- CIs for table 2 would be nice.
- For models which output a sequence of representations you average over the time or freq space. This seems like a significant limitation. You run ablation studies for with and without the PCA step over the sequence but some measure of the full sequence representation would be good OR evaluating on a model which outputs a signal vector per sample e.g. https://arxiv.org/pdf/2209.14345
-

### Questions
please see weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a new benchmark for evaluating audio encoders that focuses exclusively on their intrinsic representation capabilities through zero-shot evaluation, without any fine-tuning. Unlike existing benchmarks such as SUPERB, which assess encoder performance on downstream tasks after fine-tuning, this work isolates and measures the encoders’ generalization power directly. The paper details the benchmark’s design, its diverse dataset spanning speech, bioacoustics, and environmental sounds, and the evaluation methodology using metrics such as Precision@k (P@k) and GSR. Experimental results demonstrate that Whisper encoders exhibit notably stronger generalization performance compared with other leading audio encoders, including CLAP, WavLM, and others.

### Strengths
* The proposed method evaluates the **intrinsic representation capability** of audio encoders **without any fine-tuning or additional learnable parameters**. In contrast, benchmarks such as **SUPERB** rely on feature aggregation and trainable parameters, making their results sensitive to these design choices.
* A key strength of this approach is its use of **zero-shot evaluation**, which provides a more direct and unbiased measure of generalization performance.
* The benchmark encompasses a **diverse range of audio domains**, including **speech**, **bioacoustics**, and **environmental sounds**, ensuring comprehensive coverage across different acoustic contexts.

### Weaknesses
* **Limited task scope:** The benchmark focuses exclusively on **classification-oriented tasks**, which restricts its applicability. Audio encoders and their learned representations are widely used in other important areas, such as **text-to-speech (TTS)**, **speaker diarization**, **speech-to-speech (S2S, especially dialog) systems**, and **speech enhancement or separation**. As a result, the current setup provides only a partial view of encoder performance.
* **Incomplete domain coverage:** The benchmark omits **music and singing voice** data, which represent major audio domains and are essential for evaluating the generalization ability of modern audio encoders.
  * I also personally feel that limiting the single-source events and removing the spatial components make this work less realistic.
* **Lack of integration with existing benchmarks:** The study would benefit from including **subsets or comparisons with existing benchmarks** such as **SUPERB** and **HEARS**, to contextualize its findings and illustrate how its zero-shot results differ from established evaluation frameworks.
* **Uncontrolled experimental conditions:** A major limitation is the difficulty of drawing clear insights from the reported results. The compared audio encoders differ substantially in **training data scale, model size, architecture, domain coverage, and supervision type**, yet these factors are not controlled or analyzed. Consequently, it is unclear **why** certain models outperform others.

  * For instance, **SUPERB** carefully controls for such variables by initially restricting comparisons to **self-supervised learning (SSL)** models and providing detailed documentation of model sizes and training data, which helps reduce confounding factors.
  * Unfortunately, this benchmark lacks such design rigor. The comparison among **Whisper**, **WavLM**, and **CLAP** involves models that are too heterogeneous, making it impossible to draw meaningful conclusions about which aspects of their design contribute to performance differences.

### Questions
### Questions for the Authors

1. **Discrete vs. Continuous Representations**

   * Do you have any discussion or consideration of **discrete token representations**? Why does the paper focus exclusively on **continuous representations**, given that recent studies often explore both?

2. **Definition of the Goal of Audio Representations**

   * In the abstract, you state: *“The goal of general-purpose audio representations is to map acoustically variable instances of the same event to nearby points.”*
     While this is certainly one possible goal, it seems overly narrow and classification-oriented. Many audio applications (e.g., **reconstruction** or **regression**) do not primarily rely on this property. Could you clarify why this particular definition was chosen and how it aligns with broader audio representation learning objectives?

3. **Choice of Downstream Tasks**

   * Why was **bioacoustic classification** selected as the downstream evaluation task? Given the benchmark’s emphasis on generalization, it would be informative to include or compare results on a wider range of downstream tasks. Have you considered incorporating subsets or comparisons with **existing benchmarks** such as **SUPERB** or **HEARS** to strengthen the evaluation?

4. **Redundancy in Related Work (Section 2)**

   * The last paragraph of Section 2 reiterates the discussion of **HEAR** and **SUPERB**, which appears earlier in the same section. Could this be streamlined to avoid redundancy?

5. **Clarification of the “Blind” Test Set**

   * The term *“blind test set”* is somewhat confusing. Do you mean that only the **labels** are withheld while the **audio files** are released? Since the paper also reveals the source of the test set, this appears closer to an **out-of-domain evaluation set** rather than a fully blind one. Could you clarify what is meant by “blind” in this context?

6. **Selection and Control of Audio Encoders (Section 4.1)**

   * In Section 4.1, the selected encoders (e.g., Whisper, WavLM, CLAP) vary greatly in **training data scale, model architecture, and supervision type**. How did you control for these factors, or design ablation studies, to draw meaningful and interpretable findings from such heterogeneous models?

7. **Silence Handling During Pooling (Section 4.1)**

   * During the pooling process, how are **silence regions** handled? Was any voice activity detection (VAD) or "silence detection" applied to exclude silent segments, or are they included in the mean pooling? Including silence might bias the pooled representations.

8. **Evaluation Metrics (Section 4.2)**

   * While Section 4.2 is clearly written, have you considered incorporating additional evaluation metrics, such as **mutual information** or **purity** (as used in the HuBERT paper)? These could provide complementary insights into representational quality.

9. **Layer Selection (Section 5)**

   * Which **layer outputs** are used for evaluation? Since prior SSL probing studies have shown strong **layer dependency** of learned representations (that's why SUPERB uses the weighted layer sum to avoid this layer dependency problem), a discussion of this choice would add valuable context.

10. **Table 2 – VAE Performance**

    * Could you elaborate on why the **VAE model** performs worse than the **Log Mel baseline**? Was the VAE trained without access to the blind set, or are there other factors influencing this outcome?

---

### Additional Comments

* Please avoid **line breaks in the abstract**. Abstracts are often indexed or parsed by external systems, and maintaining plain text formatting ensures consistency.
* Consider adding **encoder probing analyses** to provide deeper interpretability. For instance, studies such as *Pasad et al. (ASRU 2021)* analyze layer-wise correlations between encoder activations and phoneme labels; similar probing could help clarify what linguistic or acoustic information your encoders capture.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces VocSim, a training-free benchmark designed to evaluate zero-shot audio content identity. The dataset comprises 125k single-source audio clips aggregated from 19 existing corpora, spanning across different types of audio. And the authors propose evaluating zero-shot similarity via two metrics: Precision@k and  Global Separation Rate (GSR). A series of models are evaluated under frozen feature settings on VocSim with pooling strategies and PCA, where a simple Whisper-encoder + pooling + PCA pipeline yields strong performance. At the same time, all models show significant generalization gaps on blind OOD subsets (low-resource speech languages).

### Strengths
The paper contains two strengths:

First, the motivation is clear as the benchmark focuses specifically on content identity under zero-shot settings in single-source audio, distinguishing it from scene analysis or supervised classification. The data processing is decent. The curated audio is segmented by type, and includes clean vs. noisy domains, different durations, and varied class granularities. This design encourages generalization testing across different acoustic conditions.

Second, the empirical evaluation is broad. The authors benchmark many widely-used audio encoders, including Whisper, CLAP, wav2vec, WavLM, AudioMAE, and neural codecs, providing a useful diagnostic comparison for the community. The validation result is insightful, it highlights clear generalization collapse on blind low-resource speech subsets, a compelling finding about the limitations of current representation models.

### Weaknesses
However, there are two major weaknesses that hinder the paper's acceptance.

First, the novelty is limited. 

The benchmark is mostly an aggregation of previous datasets (19 existing corpora) rather than a newly collected dataset with fresh labeling, annotations, or clearly designed structure. Similar aggregation-based benchmark efforts in audio already exist, as also mentioned by authors (e.g., HEAR, SUPERB). Other works have also pursued large aggregated datasets for general-purpose audio similarity and content representation (e.g., AudioSet; Fusion-Audio; WavCaps; Laion-Audio-630K). The paper does not demonstrate conceptual novelty beyond these data aggregations and two evaluation metrics that are incremental. From the data perspective, this introduces some concerns of the data reliability and motivation. These datasets spans many domains, the annotations originate from fundamentally different semantics: phones, words, and vocal imitations belong to speech; syllables belong to birds; and event-type labels belong to ESC-50. These represent different classification tasks with different acoustic and semantic structures. The paper does not sufficiently justify why aggregating these distinct sources into one evaluation is intrinsically meaningful. From the huggingface demo page provided in the appendix, the sound quality also varies a lot. So the verification of label reliability, annotation quality, or consistency across sources are also necessary.

The model innovation is yet another side. The paper only evaluates existing models and basic clustering / feature pooling strategies. There is no proposal of new embedding methods, training paradigms, or pre-training objectives. As a result, the work reads primarily as a resource + evaluation paper rather than a technical contribution. This should be upon more discussion but it might be inadequate for ICLR.

Second, the experimental design needs improvement. 

The paper should compare VocSim with existing benchmarks to demonstrate why it is superior or insightful from other perspective. Such comparison can involve: applying the same models but calculating metrics in HEAR, SUPERB, AudioSet, WavCaps, and Fusion-Audio (not all but at least 1-2 of them). Missing comparisons to other content-identity datasets is critical here. 

Some further concerns might involve the model training details. Some models (like CLAP and Whsiper) may have been originally trained on subsets of the same data sources used in VocSim. Thus, the performance comparisons (and claims about “zero-shot”) may not be fully fair. A careful treatment of data overlap is necessary.

In this case, while the benchmark gives a structured way to evaluate zero-shot content identity, the paper does not convincingly demonstrate that VocSim demonstrates better quality and usability than previous benchmarks, nor it clarifies all evaluated models are free from the potential overlapping issues that might harm the "zero-shot" statement. 

In conclusion, the paper is cleanly written and the empirical evaluation is thorough. However, the novelty is limited, and key comparisons to similar aggregated datasets are missing. The benchmark’s reliability and generalizability remain unclear. With stronger justification,  and more evidence of benchmark stability and comparison, the work may become a valuable contribution.

### Questions
1. Is there quantitative validation of label quality across subsets, particularly on vocal imitations and human words/utterances where annotations are noisy or semantic?

2. How does VocSim differ concretely from prior aggregated corpora such as AudioSet, HEAR, SUPERB, WavCaps, Fusion-Audio, and LAION-Audio-630k? Why should the community adopt VocSim over these?

3. Were the foundation models (esp. Whisper, wav2vec2, CLAP) trained on subsets overlapping VocSim sources? If so, how is “zero-shot” interpretation justified?

### Soundness
3

### Presentation
3

### Contribution
2
