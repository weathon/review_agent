# RUBATO: A Multi-Version Benchmark for Robust Music  Analysis and Transcription

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4

## Abstract
Robustness is a fundamental challenge for deep learning, as models frequently inherit dataset biases and fail to generalize across real-world variability. Models for music audio analysis and transcription—machine-learning tasks of particular difficulty and data scarcity—often lack robustness to changes in instrumentation, interpretation, or recording conditions. In contrast to text and vision, robustness in music remains underexplored. To address this gap, we introduce RUBATO, a manually curated, fully open music dataset and benchmark. Our central idea is to exploit the unique opportunities of Western classical music where we find famous works free of copyright and with an abundance of available recordings, which follow the same score but differ in interpretation and recording conditions, supplemented by arrangements and adaptations for other instrumentations. For RUBATO, we collected and recorded 14 canonical works in up to 54 versions, totaling 560 audio tracks and 42 hours of audio, including original recordings, arrangements and adaptations, controlled piano renditions, and synthesized versions. We further curated symbolic scores and expert annotations for various tasks. Ensuring structural coherence for the majority of versions, we transfer annotations between versions using state-of-the-art alignment techniques, which we evaluate for the heterogeneous version pairs in RUBATO. The resulting high-quality annotations allow for benchmarking music understanding models, which we demonstrate for two selected tasks—automatic music transcription and local key estimation. Going beyond standard metrics, the multi-version design of RUBATO enables systematic evaluation not only of models' efficacy but also of their consistency across versions of the same work. We formalize this notion as cross-version consistency, which allows to assess model robustness along various dimensions of music data. Testing current machine-learning systems for different variants of such consistency measures, we find that most of these systems struggle to generalize under real-world variability, highlighting the need for more robust models and for benchmarks as RUBATO capable of measuring such robustness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a benchmark dataset, RUBATO, that is specialized for multi-version related tasks on classical music. The authors have put significant effort creating this dataset including recording new versions. The data consists of carefully chosen works and versions, which are aligned semi-automatically. The datasets also includes expert labels of the structure, pitch, instruments, key, beats, and chords.

### Strengths
The paper and the proposed dataset definitely seem to be of high quality. As far as I can tell, many musicological aspects are considered in the design of the dataset. Multi-version is a unique, interesting aspect of classical music and the proposed dataset would significantly help evaluating the aspect.

### Weaknesses
* Title: "Robust" is a much more general term than how it is used in this paper. I do think the proposed dataset provides a way to test robustness of models, but "Robust Music Analysis" sounds too general when what it actually means is a cross-version consistency.
* Section 4: RUBATO Benchmark: These experiments are interesting, but I disagree with some of the interpretations. I find the conclusion of "This suggests that medium-sized, fully convolutional models obtain slightly lower efficacies but are more robust and, therefore, universally applicable than large Transformers." (on L 426-427) a bit premature. The models are different from various aspects and we can't simply attribute the different behaviors to the difference of model architectures. 
* Impact: The proposed dataset is unfortunately limited to 1) evaluation of 2) classical music, and especially on 3) the consistency between versions. I firmly believe it took a significant amount of effort and consideration to build this dataset, and for the purpose it would be a great dataset. But that seems really narrow, otherwise I would expect the consistency related discussion in 4.2 Result - the last 5 lines - would have deeper discussion. However, that discussion is rather simple and a bit questionable. Similarly, I am not convinced about the discussion in Section 5 (more details on Questions.)

### Questions
In Section 5 - Results: 
- On "Figure 4 shows the key recall rate for Octave-LSTM. Efficacy is generally high as compared to a typical inter-annotator agreement.": I find it a bit confusing that the authors discuss the performance of the model by comparing it with inter-annotator agreement. To be strict, the efficacy being high or low has nothing to do with inter-annotator agreement. Wouldn't it make more sense to compare GEC or other CVC measures with a typical inter-annotator agreement?
- The key detection and transcription would have very different random prediction baselines, and they're essentially very different tasks. This would make all the CSC not comparable across tasks.
- On "This suggests that for LKE, the challenge is less the variety of versions but to learn the musical notion of local key": I kind of get it, but not fully. It is as if implying there are only two aspects of solving a task: inter-version variety or the core task itself? And can we actually separate them? Isn't "to learn the musical notion of local key" == "be able to do it with different styles (that are present in different versions)" -- i.e., isn't the latter include the former? 
  - Would it be the case that, as the model prediction score goes higher, in reality LEC and LPC would go higher as well? I'm asking this question as I wonder if we could simply conclude "LKE is a relatively easy task and both of the models perform quite consistently well." In other words, the cross-version difference might be still challenging as much as the problem is, and we can't really argue if the challenge is more or less the variety of versions vs. learn the musical notion of local key."

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
RUBATO is a multi-version dataset and benchmark for evaluating the robustness of music analysis and transcription models. The dataset comprises 14 canonical works with up to 54 versions each (560 tracks, 42 hours), including original recordings, arrangements, controlled piano renderings, and synthesized audio. A key contribution is the provision of high-quality, structurally coherent annotations, transferred across versions using a rigorously evaluated alignment strategy. Beyond standard evaluation, the authors propose Cross-Version Consistency (CVC) metrics (GEC, LEC, LPC) to quantify model robustness. It demonstrated significant robustness gaps in current state-of-the-art models through experiments on Automatic Music Transcription (AMT) and Local Key Estimation (LKE).

### Strengths
The dataset is  constructed. The systematic inclusion of different version types (original, arranged, controlled) enables targeted studies on specific domain shifts. The effort to ensure structural coherence and the rigorous alignment process are commendable. It is an open resource dataset with clear structure and the formalization of CVC provides a valuable, additional lens for evaluation beyond standard accuracy, better reflecting the requirements for real-world, musicologically-grounded applications.

### Weaknesses
A significant omission is the evaluation of large, generative, or foundation models in music. The benchmark is limited to established, discriminative models for AMT and LKE (e.g., MT3, Basic Pitch). The field is rapidly moving towards large-scale models and instruction-tuned models capable of multiple tasks (e.g., Qwen2.5 omni, Gemini). Perhaps ABC notation can also be provided for this. The benchmark's relevance is diminished by not including such models. For instance, it is unclear how these models would perform on tasks like version retrieval, structure analysis, or zero-shot transcription across RUBATO's heterogeneous versions. This limits the paper's impact on the current research frontier. 

The exclusive focus on 18th- and 19th-century Western classical music, while practical for copyright reasons, constrains the generalizability of the findings. Robustness challenges in pop, jazz, or non-Western music may differ significantly.

### Questions
The benchmark currently evaluates single-task, discriminative models (e.g., MT3). However, the field is rapidly advancing towards large, multi-task audio foundation models capable of both transcription and analysis, often in a generative or instruction-following setting (e.g., "transcribe the violin part," "describe the musical structure"). In this new context, how do the authors envision the RUBATO benchmark evolving? Specifically: 1) How can the CVC metrics be adapted to evaluate the consistency of a single, multi-task model's outputs across versions, as opposed to comparing dedicated, single-task models? 2) Does the concept of a fixed, frame-level output representation (e.g., a 72-bin pianoroll) become a limitation when evaluating a model that might natively output a symbolic score, a text description, or a combination thereof?

The paper's tasks and annotations are firmly grounded in the paradigm of Western Common Practice notation, where discrete "notes" and a fixed set of "keys" are fundamental concepts. So how do the authors view the generalizability of their proposed evaluation metrics (e.g., F-measure on note events, MIREX score for key) to musical cultures where the very concept of a "note" is different (e.g., Raga music, genres with continuous pitch slides) or where the perceptual foundation of harmony does not align with the Western key system?

Also, given that robustness is a universal goal, what steps could be taken in the future to extend the RUBATO framework—or to inspire a similar benchmark—that is agnostic to a specific notational culture, perhaps relying more on perceptual similarity or culture-specific ground truths?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces RUBATO, a new multi-version benchmark designed to evaluate robustness in music audio analysis and transcription.
It leverages Western classical works that share the same musical score but vary across performances, instrumentations, and recording conditions, providing a systematic basis to test model generalization.

RUBATO contains 560 recordings (42 hours) of 14 canonical works by 12 composers, covering original performances, arrangements, controlled piano renderings, synthesized versions, and adaptations—all aligned to a shared musical time axis through a validated hybrid alignment pipeline. The dataset ensures structural coherence across most versions and includes annotations sufficient for evaluating diverse music analysis tasks.

Beyond the dataset, the paper proposes the RUBATO Benchmark, introducing Cross-Version Consistency (CVC) metrics that quantify how stable a model’s predictions remain across versions of the same work. Using these metrics, the authors evaluate state-of-the-art systems for Automatic Music Transcription and Local Key Estimation, showing that while these models perform well on individual recordings, they often fail to generalize consistently across different versions—revealing a clear robustness gap in current music models.

### Strengths
S1:Well-engineered and reproducible data pipeline.The dataset construction is technically solid and clearly described, including systematic collection, controlled re-recordings, MIDI-based renderings, unified formatting, and a multi-stage alignment process (manual anchors → audio–audio → score–audio). Quantitative alignment error analysis adds credibility and reproducibility.

S2:Clear formalization of cross-version evaluation.The paper explicitly defines robustness as Cross-Version Consistency (CVC) and decomposes it into GEC, LEC, and LPC. This makes robustness measurable and avoids conflating accuracy with stability.

S3:Alignment quality is quantitatively validated.Synchronization accuracy is reported with concrete AUC values and human-level tolerance references, demonstrating methodological rigor and providing realistic expectations of annotation noise.

### Weaknesses
W1: 

The overall idea—evaluating robustness through cross-version consistency—is conceptually reasonable but not new. Similar robustness and consistency frameworks have been widely explored in other domains since around 2019–2022 (e.g.,*ImageNet-C*,*WILDS*,*BIG-bench*).What this paper mainly contributes is an adaptation of those ideas to the music domain, without introducing genuinely new methods or theoretical insights.

W2: While the three proposed measures (GEC, LEC, LPC) are well defined, the paper never demonstrates that they truly capture robustness. There is no evidence that these metrics correlate with human perception, alignment reliability, or downstream generalization. The authors even acknowledge that a model could score high on CVC while being consistently wrong, which raises serious doubts about the interpretability and usefulness of the metric itself.

W3: Only two tasks—Automatic Music Transcription and Local Key Estimation—are evaluated, although the dataset contains annotations for several others (e.g., beat tracking, structure analysis). This makes it difficult to justify RUBATO as a general benchmark for “robust music analysis.”

W4: The dataset is limited to Western classical music, which simplifies alignment and annotation but prevents generalization to popular, modern, or non-Western styles where variability is higher. The scope is narrow and misses the opportunity to test robustness in more diverse musical conditions.

W5: The observation that Transformer-based models are less consistent than convolutional ones is interesting but remains purely descriptive. The paper does not explore potential causes such as timbre sensitivity, training-domain bias, or decoding instability.

W6:  All experiments are run on small, task-specific architectures. It remains unclear whether similar robustness issues would appear in large pretrained or foundation models, which are increasingly standard in audio and MIR research.

W7:  Cross-version consistency analysis is unlikely to be used in real-world music applications. At present, it serves mainly as a diagnostic research tool rather than an applied benchmark. The paper doesn’t convincingly explain how the proposed metrics or dataset would translate to practical or industrial benefit.

W8：From a methodological and temporal perspective, the contribution feels incremental and somewhat out of step with where robustness research has moved. The work will likely be useful for the MIR community as a reproducible resource, but the contribution remains incremental rather than forward-looking.

### Questions
Q1: The dataset includes annotations for other MIR tasks such as beat tracking and structure analysis, yet the experiments focus only on AMT and LKE.
Could you clarify whether the proposed CVC framework can be meaningfully extended to these additional tasks, and if so, what limitations or adjustments would be required?
Q2: The finding that Transformer-based models show lower cross-version consistency than convolutional ones is intriguing, but the discussion remains quite descriptive.
Could you provide further quantitative or qualitative analysis to explain the source of this instability—such as the influence of timbre sensitivity, domain bias, or decoding thresholds?
Q3: Have you considered whether recent music-pretrained representation models—such as MERT—exhibit similar cross-version consistency patterns at the representation level?
Even if these models are not designed for low-level transcription, testing them could help determine whether the robustness patterns you report also appear in higher-level semantic or structural representations.

### Soundness
2

### Presentation
3

### Contribution
2
