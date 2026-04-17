# Neuroprobe: Evaluating Intracranial Brain Responses to Naturalistic Stimuli

- Decision: Reject
- Scores: 8, 4, 4, 6

## Abstract
High-resolution neural datasets enable foundation models for the next generation of brain-computer interfaces and neurological treatments. The community requires rigorous benchmarks to discriminate between competing modeling approaches, yet no standardized evaluation frameworks exist for intracranial EEG (iEEG) recordings. To address this gap, we present Neuroprobe: a suite of decoding tasks for studying multi-modal language processing in the brain. Unlike scalp EEG, intracranial EEG requires invasive surgery to implant electrodes that record neural activity directly from the brain with minimal signal distortion. Neuroprobe is built on the BrainTreebank dataset, which consists of 40 hours of iEEG recordings from 10 human subjects performing a naturalistic movie viewing task. Neuroprobe serves two critical functions. First, it is a mine from which neuroscience insights can be drawn. The high temporal and spatial resolution of the labeled iEEG allows researchers to systematically determine when and where computations for each aspect of language processing occur in the brain by measuring the decodability of each feature across time and all electrode locations. Using Neuroprobe, we visualize how information flows from key language and audio processing sites in the superior temporal gyrus to sites in the prefrontal cortex. We also demonstrate the progression from processing simple auditory features (e.g., pitch and volume) to more complex language features (part of speech and word position in the sentence tree) in a purely data-driven manner. Second, as the field moves toward neural foundation models trained on large-scale datasets, Neuroprobe provides a rigorous framework for comparing competing architectures and training protocols. We found that the linear baseline on spectrogram inputs is surprisingly strong, beating frontier foundation models on many tasks. Neuroprobe is designed with computational efficiency and ease of use in mind. 
We make the code for Neuroprobe openly available and will maintain a public leaderboard of evaluation submissions, aiming to enable measurable progress in the field of iEEG foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces **Neuroprobe**, a multitask iEEG decoding benchmark (15 tasks spanning audio/vision/language) built on BrainTreebank, with standardized **within-session / cross-session / cross-subject** splits and AUROC as the primary metric. It establishes linear baselines and compares “frozen” foundation-model representations (BrainBERT, PopulationTransformer).

### Strengths
1. **High potential community impact.** Well-structured benchmark with clear splits, standardized electrodes per subject, and a plan for a public leaderboard, enabling fair and reproducible model comparison on diverse naturalistic tasks.

2. **Comprehensive evaluation.** The benchmark spans **15 decoding tasks** across auditory, visual, and linguistic domains, providing broad support for probing multimodal neural processing and enabling cross-modal insights under naturalistic stimuli.

3. **Strong and Transparent Baselines.** The benchmark presents some baseline models (from raw-signal linear decoders to frozen foundation-model representations) , which provide honest performance references that will help guide future progress.

4. **Insightful Spatial and Temporal Analyses**. Figs 4–7 effectively demonstrate how Neuroprobe enables neuroscientific discoveries, including modality-specific cortical organization and the temporal evolution of linguistic and perceptual feature processing.

### Weaknesses
1. **Task overview missing in the main text.** A schematic grouping the 15 tasks by modality/subtasks would improve immediate understanding.

2. **Foundation models only used as frozen encoders**, without justification or discussion on whether fine-tuning changes the ordering.

3. **Model size & compute not reported** in Fig. 3 & Fig. 7, limiting efficiency comparison.

4. The **counter-intuitive observation** that linear ≥ foundation models is not discussed in the paper.

### Questions
1. Please **add a compact visual** summarizing all 15 tasks (audio/language/vision groups and label generation).

2. Why **freeze foundation models** instead of fine-tuning? Were they pretrained on **BrainTreebank**? Any **subject or task overlap**? A brief description of each model in the main text would help.

3. Please **report parameters and size** for linear vs. foundation models in Fig. 3 & Fig. 7.

4. Please discuss **why** the tuned linear baseline outperforms foundation models—e.g., representation alignment, SNR improvements from Laplacian referencing , etc.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Neuroprobe, a benchmark suite for evaluating machine learning models on naturalistic stimuli decoding tasks to understand the neural computation mechanism of multi-modal language processing from human intracranial EEG (iEEG) data. Neuroprobe builds upon the BrainTreebank dataset, offering 15 well-defined binary classification tasks, standardized evaluation splits (within-session, cross-session, and cross-subject), and a leaderboard to track progress in iEEG foundation models. Through extensive analyses across spatial and temporal domains, the authors explore the flow of information in the brain and present comprehensive baseline comparisons (including linear models and state-of-the-art pretrained models).

### Strengths
1. Neuroprobe’s benchmark is carefully constructed from the movie tasks in Brain Treebank [1], featuring pre-defined tasks, rigorously constructed train/test splits, and standardized evaluation rules to evaluate previous baselines [2,3].

2. The authors conduct in-depth spatial and temporal analyses (based on linear decoding), using Neuroprobe to visualize how language, audio, and visual features are represented and evolve in the brain, where the analysis methods are widely adopted in neuroscience research. The discussion is substantively grounded, providing interpretable neuroscientific insights, such as the mapping of features to specific brain regions and the timing of language/visual processing.

3. The manuscript thoroughly documents the experimental setup and choices, making these tools (including open-source code and a leaderboard) publicly available to ensure reproducibility and community usability.

### Weaknesses
1. The motivation for expanding the original 4 decoding tasks (i.e., pitch, volume, sentence onset, and speech/non-speech in BrainBERT [2]) to 15 tasks is unclear. It’s unclear whether these manually defined concepts (e.g., number of faces, global optical flow) align with the neural encoding mechanism [4-6]. Could the author provide some neuroscience literature to ensure the neuroscientific significance of these decoding tasks?

2. The linear decoding results differ significantly from the results in [2] (e.g., 0.90 v.s. 0.60 on sentence onset), which greatly challenges the effectiveness of current iEEG foundation models. This work is similar to DLinear [7] in the time series forecasting field. If this work aims to challenge the representation learning of current iEEG foundation models (as the evaluation focuses on linear probing), more baselines [8-10] are needed for evaluation.

Overall, I appreciate the efforts in benchmark formulation and running so many baselines, but I am not satisfied with the novelty of the method.

**References**:

[1] Wang C, Yaari A, Singh A, et al. Brain treebank: Large-scale intracranial recordings from naturalistic language stimuli[J]. Advances in Neural Information Processing Systems, 2024, 37: 96505-96540.

[2] Wang C, Subramaniam V, Yaari A U, et al. BrainBERT: Self-supervised representation learning for intracranial recordings[J]. arXiv preprint arXiv:2302.14367, 2023.

[3] Chau G, Wang C, Talukder S, et al. Population transformer: Learning population-level representations of neural activity[J]. ArXiv, 2025: arXiv: 2406.03044 v4.

[4] Mesgarani N, Cheung C, Johnson K, et al. Phonetic feature encoding in human superior temporal gyrus[J]. Science, 2014, 343(6174): 1006-1010.

[5] Luo A F, Henderson M M, Tarr M J, et al. Brainscuba: Fine-grained natural language captions of visual cortex selectivity[J]. arXiv preprint arXiv:2310.04420, 2023.

[6] Bouchard K E, Mesgarani N, Johnson K, et al. Functional organization of human sensorimotor cortex for speech articulation[J]. Nature, 2013, 495(7441): 327-332.

[7] Zeng A, Chen M, Zhang L, et al. Are transformers effective for time series forecasting?[C]//Proceedings of the AAAI conference on artificial intelligence. 2023, 37(9): 11121-11128.

[8] Jiang W B, Zhao L M, Lu B L. Large brain model for learning generic representations with tremendous EEG data in BCI[J]. arXiv preprint arXiv:2405.18765, 2024.

[9] Wang J, Zhao S, Luo Z, et al. Cbramod: A criss-cross brain foundation model for eeg decoding[J]. arXiv preprint arXiv:2412.07236, 2024.

[10] Zhang D, Yuan Z, Yang Y, et al. Brant: Foundation model for intracranial neural signal[J]. Advances in Neural Information Processing Systems, 2023, 36: 26304-26321.

### Questions
See the above concerns.

### Soundness
2

### Presentation
2

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
This paper introduces Neuroprobe, a benchmark built upon the BrainTreebank iEEG dataset designed to evaluate brain decoding models under naturalistic stimuli. Neuroprobe defines 15 classification tasks spanning audio, vision, and language domains, offering standardized data splits (within-session, cross-session, cross-subject) and carefully controlled baselines. Using this resource, the authors analyze the spatiotemporal organization of language and perceptual processes in the brain and compare a range of linear and state-of-the-art neural decoders. The paper includes extensive visualization and performance analysis, and aims to establish a public leaderboard to support progress in iEEG-based representation learning.

### Strengths
1. Relevance and Timeliness: The introduction of Neuroprobe as a standardized benchmark for iEEG decoding addresses a critical and widely recognized gap in the field. The lack of rigorous, standardized evaluation frameworks has hindered reproducible progress in iEEG foundation models. This benchmark, built on the publicly available BrainTreebank dataset, provides a much-needed common ground for comparing model performance and tracking advances.

2. Rigorous and Controlled Benchmarking: The paper thoughtfully defines three distinct evaluation splits (within-session, cross-session, cross-subject) that systematically address critical issues like temporal autocorrelation and test generalization across different levels of difficulty. The methodology for preventing data leakage (e.g., using contiguous blocks for cross-validation) is a particular strength, ensuring the integrity of the reported results.

3. Rich, Neuroscientifically Interpretable Analysis: A major strength of the work is the extensive analysis that translates model performance into neuroscientific insights. The spatial (Figure 4) and temporal (Figures 5, 6) visualizations of decodability provide a clear, data-driven picture of how and where different multimodal features are processed in the human brain. This dual utility—as both a machine learning benchmark and a tool for neuroscience discovery—significantly enhances the paper's impact.

### Weaknesses
1. Limited Scope of Model Benchmarking: While the authors benchmark a linear baseline and two foundation models (BrainBERT, PopulationTransformer), the evaluation lacks a broader range of modern deep learning architectures. For instance, non-foundation models like supervised CNN-Transformer or RNNs, which have shown strong performance in related neural signal decoding tasks [1-3] , are absent. A more diverse set of baselines would provide a clearer picture of the current state-of-the-art and the specific challenges posed by this benchmark.

2. Insufficient Analysis of Anatomical Variability Impact: The benchmark acknowledges the challenge of variable electrode placements across subjects but does not quantitatively analyze how this anatomical heterogeneity affects decoding performance, especially in the challenging cross-subject split. An ablation study examining performance relative to inter-subject electrode coverage overlap or a more detailed regional analysis would strengthen the conclusions about the spatial localization of decodable features.

3. Incomplete Contemporary iEEG Literature: The related work section provides a good overview of the benchmark landscape but could be more thoroughly integrated with recent advances in iEEG/ECoG deep learning. For example, several recent works focusing on deep learning for functional mapping, unsupervised representation learning, or subject-transfer in iEEG are highly relevant [4-6]. Engaging more directly with this literature would better contextualize Neuroprobe's contributions and suggest immediate avenues for future benchmarking by the community.

**References:**

[1] Willett, F. R., Kunz, E. M., Fan, C., Avansino, D. T., Wilson, G. H., Choi, E. Y., ... & Henderson, J. M. (2023). A high-performance speech neuroprosthesis. *Nature, 620*(7976), 1031-1036.

[2] Metzger, S. L., Littlejohn, K. T., Silva, A. B., Moses, D. A., Seaton, M. P., Wang, R., ... & Chang, E. F. (2023). A high-performance neuroprosthesis for speech decoding and avatar control. *Nature, 620*(7976), 1037-1046.

[3] Song, Y., Zheng, Q., Liu, B., & Gao, X. (2022). EEG conformer: Convolutional transformer for EEG decoding and visualization. *IEEE Transactions on Neural Systems and Rehabilitation Engineering, 31*, 710-719.

[4] Chen, X., Wang, R., Khalilian-Gourtani, A., Yu, L., Dugan, P., Friedman, D., ... & Flinker, A. (2024). A neural speech decoding framework leveraging deep learning and speech synthesis. *Nature Machine Intelligence, 6*(4), 467-480.

[5] Zheng, H., Wang, H., Jiang, W., Chen, Z., He, L., Lin, P., ... & Liu, Y. (2024). Du-IN: Discrete units-guided mask modeling for decoding speech from Intracranial Neural signals. *Advances in Neural Information Processing Systems, 37*, 79996-80033.

[6] Singh, A., Thomas, T., Li, J., Hickok, G., Pitkow, X., & Tandon, N. (2025). Transfer learning via distributed brain recordings enables reliable speech decoding. *Nature Communications, 16*(1), 8749.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces what appears to be the first public benchmark that systematizes high-fidelity naturalistic iEEG into a multi-task, cross-session, and cross-subject evaluation suite. The design is clear, the implementation is practical, and the results are reproducible with code and a leaderboard. 
This paper introduces a public benchmark that turns high-fidelity naturalistic iEEG into a coherent evaluation suite: it builds on BrainTreebank’s 40+ hours from 10 participants, defines 15 standardized decoding tasks spanning speech, vision, and language, and ships unified protocols, metrics, code, and a leaderboard.  It formalizes three splits—within-session, cross-session, and cross-subject—with safeguards against temporal leakage and a default focus on cross-session testing. 

It lowers the barrier to entry with a Neuroprobe-Lite subset, a cap of 120 electrodes per subject, probe reuse for multiple re-referencing strategies, and a clear submission workflow. Finally, it benchmarks linear models, BrainBERT, and PopulationTransformer, and finds that Laplacian-re-referenced spectrograms with a simple linear model are surprisingly strong and already yield clear spatiotemporal decoding maps and processing order.

Several tasks are binarized and a few evaluation choices may limit external validity. The paper would be stronger if the main text added motivation for binarization, robustness checks across thresholds, and small ablations to show conclusions are stable.

### Strengths
Originality. First standardized benchmark from high-fidelity, natural iEEG. It turns movie-watching data into 15 clear decoding tasks across speech, vision, and language, and tests three settings: within-session, cross-session, and cross-subject.

Quality. Careful and reproducible setup. Splits prevent temporal leakage, inputs are fixed at one second, and a 120-electrode cap keeps things practical while allowing re-referencing. Baselines span simple linear models to pretrained transformers, with documented spectrogram choices and a public leaderboard.

Clarity. Easy to follow. Figures show how raw data become tasks, what each split measures, and how models perform across all 15 tasks. Labels and windowing are defined plainly, and the appendix table removes ambiguity.

Significance. Provides a reusable yardstick for iEEG foundation models. Simple Laplacian-based spectrograms with linear decoding already do very well and yield clear spatiotemporal maps of decodability, which will guide future methods and fair comparisons.

### Weaknesses
1 Many tasks are reduced to binary by percentile thresholds or one-vs-rest, which can distort difficulty and hide effects. Please justify the choice in the main text and add robustness checks: threshold sweeps, and where possible, regression or multi-class versions with AUROC/PR-AUC or R². 


2 Authors cross-subject protocol trains only on Subject 2, Trial 4, then tests on others. This is a very narrow source distribution and may not generalize. Add alternatives: train on a different subject/session, a pooled multi-source train set, and report how rankings and absolute scores move. 

3 For efficiency authors cap each subject at 120 electrodes and choose them starting with those that show the highest linear decoding performance. That can create circularity and make simple linear probes look stronger than they would under anatomy-only or random selection. Please clarify the selection is done using train-only data, and add sensitivity analyses: anatomy-driven, random, and performance-driven selection compared side by side. 

4. Baseline parity and “why linear wins” are under-analyzed.
The strongest baseline is linear decoding on Laplacian-re-referenced spectrograms, helped by careful spectrogram sweeps. The pretrained models use different inputs and may be handicapped. To make the comparison fair and more informative, add runs where all models share the same pre-processing (e.g., Laplacian spectrograms), and a small ablation quantifying the contribution of re-referencing and spectrogram hyperparameters to linear gains.

### Questions
1 Are task windows anchored appropriately for each modality?
All evaluations cut a 1-second window after each word onset. That is natural for language, but visual tasks (e.g., optical flow, brightness) may not align to word timing. 
Would it be fair to test modality-specific anchors (e.g., frame onsets or peak optical-flow times) and add a jitter-robustness check that shifts windows ±250–500 ms to confirm conclusions hold. 

2 Does the Lite subset’s truncation introduce bias?
Neuroprobe-Lite keeps the first 3,500 annotations per task and selects subject–trial pairs that hit this limit. That “first-N” rule can skew content distributions and label balance.
 Please report how class proportions and movie segments differ before vs. after trimming, and consider stratified sampling across the full session to avoid temporal bias. 

3 Is the cross-subject region mapping discarding informative data?
For cross-subject, authors average activity within atlas regions and only keep regions shared by both subjects. This simplifies alignment but may drop informative electrodes and change task difficulty. Please quantify how many regions are discarded per pair, show performance vs. fraction of overlapping regions, and compare with alternative alignments (e.g., functional correspondence or optimal transport). 

4 How will the leaderboard guard against overfitting?
Submissions are unverified beyond formatting and receive a “reproducible” tag if a code link is provided. To curb test-set overfitting, consider a sequestered hidden test split, periodic refreshes, and mandatory config files with fixed seeds.

### Soundness
3

### Presentation
2

### Contribution
3
