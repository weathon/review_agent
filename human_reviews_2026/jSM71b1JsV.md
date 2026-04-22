# Battery Fault: A Comprehensive Dataset and Benchmark for Battery Fault Diagnosis

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 4, 4, 6

## Abstract
With the accelerated popularization of electric vehicles (EV), battery safety issues have become an important research focus. Data-driven battery fault diagnosis algorithms, built on real-world operational data, are critical methods for reducing safety risks. However, existing battery datasets have limitations such as insufficient scale, coarse-grained labels, and lack of coverage of real-world operating conditions, which seriously restrict the development of data-driven fault diagnosis algorithms. To address these issues, this paper introduces a large-scale benchmark dataset named CH-BatteryGen, which is, to the best of our knowledge, the first EV battery system fault diagnosis dataset based on real-world operating conditions. This dataset integrates real on-board operation data with mechanism-constrained generative modeling technology, balancing authenticity and scalability. It covers two mainstream battery chemistries, namely nickel-cobalt-manganese (NCM) lithium batteries and lithium iron phosphate (LFP) batteries, and involves charging, discharging, and operation data of 1000 electric vehicles. It provides four fault labels (normal, self-discharge, high-resistance, low-capacity) and three severity level annotations, supporting two benchmark tasks: fault classification and fault grading. Through systematic validation using traditional machine learning methods (random forest (RF), support vector machine (SVM)) and deep learning models (long short-term memory (LSTM), convolutional neural network (CNN)), the results show that the CNN model performs best in the fault classification task, achieving an F1-score of 0.9280 in the LFP discharging scenario; in the fault grading task, the F1-score reaches 0.8813. The CH-BatteryGen dataset has been open-sourced, aiming to provide a standardized evaluation platform for battery fault diagnosis algorithms, promote research development in this field, and contribute to the transformation of sustainable transportation systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this paper, a new benchmark (real-calibrated, synthetically generated) for battery fault diagnosis is presented. The authors use real data for calibrating their data generation method for two different battery chemistries: lithium iron phosphate (LFP) and nickel-cobalt manganese (NCM). They use this benchmark to evaluate different baseline architectures on two tasks: fault classification and fault grading.

### Strengths
- Novel benchmarks that are easily accessible online are an important contribution to the academic community.
- Claimed open release in a standard format allows for comparisons in the community

### Weaknesses
- My main criticism is that ICLR may not be the right venue for this work. Given its specialized nature concerning batteries, I believe more specialized journals (such as those heavily cited in the references) would be a better fit.
- The data generation pipeline is quite unclear. On one hand, it states that it is based on real data, but then it mentions that simulations and data-driven methods, such as diffusion models and convolutional wavelets, are used. It is unclear: a) the motivation behind these choices, and b) how this dataset was validated. How can we tell whether this data is actually realistic?
- In general, the entire manuscript is hard to follow.
- The benchmark section uses only standard architectures (RF, SVM, LSTM, and CNN) with many arbitrary design choices. The exclusion of more specialized architectures makes the baselines rather weak, while the multitude of arbitrary design choices makes it hard to develop a clear intuition of what is needed for improvement.

### Questions
Please address the points I have raised in the weakenss directly

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents CH-BatteryGen, a benchmark dataset and evaluation suite for electric vehicle battery fault diagnosis. It combines real-world EV data with physics-constrained generative modeling to ensure both realism and scalability. The dataset spans two chemistries (LFP, NCM), four fault types, three severity levels, and includes multi-modal time series signals such as voltages, current, temperature, and state of charge. It supports two tasks: four-way fault classification and three-way fault grading.
Data are derived from large-scale EV operations, augmented by AI-generated current profiles and voltage signals from RC and DCWT-based models. The dataset includes 1,000 EVs with diverse conditions and fine-grained per-cell and pack-level signals. Both classical and deep learning baselines are evaluated, with CNNs performing best overall. The authors also propose BatteryMultiModalCNN, which processes voltage time series as grayscale images via a ResNet-50 with attention and feature fusion.
CH-BatteryGen is positioned as more fine-grained and realistic than prior datasets like EVBattery, BatteryML, NASA PCoE, and HNEI, which lack fault labels or real-world coverage. The dataset is open-sourced to promote standardized evaluation, with visualizations provided for signal patterns, voltage images, and model performance.

### Strengths
- The paper addresses a critical application in battery fault diagnosis, an area where publicly available, fault-labeled datasets remain scarce. The inclusion of both LFP and NCM chemistries, along with data from both charging and discharging conditions, enhances the dataset’s relevance to real-world electric vehicle operations.

- The paper clearly defines two tasks: fault classification and severity grading. These tasks are evaluated using standard metrics such as F1 score, recall, and accuracy, supported by confusion matrices that facilitate interpretability. The addition of a grading task represents a meaningful advancement beyond traditional binary fault detection.

- The data generation process combines AI-generated current profiles with physics-constrained voltage modeling, using an RC model for LFP and a DCWT-based mapping for NCM. This approach balances scalability with domain fidelity by incorporating electrochemical structure into the synthetic data.

- The empirical study includes a range of classical and deep learning methods, from random forests and SVMs to LSTMs and a CNN-based architecture with CBAM attention. The comparison provides a useful baseline for future work, and the CNN’s performance, particularly under discharging conditions, offers practically valuable insights

### Weaknesses
- While the reported voltage deviation under 10 mV is promising, more comprehensive tests are missing. These include distribution-level comparisons, condition-dependent realism (e.g., by temperature or state of charge), and consistency across sensor modalities. No validation is performed using real-world fault labels, which weakens claims of external validity.

- There is ambiguity regarding dataset scale and class balance. The claim of 1,000 EVs conflicts with the small global label counts reported (e.g., 400, 30, 30, 40), and it is unclear whether these refer to subsets, chemistries, or scenarios. This uncertainty affects class imbalance, grading reliability, and the credibility of performance metrics.

- The current data split risks train-test leakage. If segments from the same vehicle appear in both sets, this could inflate results. A vehicle-level split should be provided alongside the current file-based split, and metrics should be reported for both.

- The experimental protocol would benefit from stronger validation strategies. Given the dataset's moderate scale, leave-one-vehicle-out cross-validation would provide a more robust assessment of generalization across vehicles and chemistries.

- The cross-domain transfer analysis is underdeveloped. Although the paper suggests cross-chemistry and cross-mode degradation, no explicit train-on-A/test-on-B experiments or confidence intervals are presented. Such comparisons are essential to support generalization claims.

- Baselines do not reflect the current state of the art. Recent time-series models such as transformers or state-space models are absent, despite their known performance advantages in noisy, long-horizon data. Their inclusion could affect conclusions about modality and architecture choices.

- The definition of severity levels lacks operational grounding. Although threshold formulas are provided, the paper does not fully explain how they are calibrated, normalized per battery pack, or made robust to sensor noise and cell variability.

- Details on dataset release and ethical considerations are incomplete. Although the dataset is described as open-source, no repository URL or license is provided in the paper. There is also no discussion of data provenance, real versus synthetic composition, consent, or privacy protections for vehicle identifiers.

### Questions
- The reported total of “1000 EVs × 20 segments” appears inconsistent with the global label counts (400, 30, 30, 40). Are these figures per scenario, per chemistry, or based on labeled subsets? Clarifying this would help assess class imbalance and distribution, considering per-task, per-chemistry breakdowns along with severity distributions.

- For single-file experiments, was the data split by vehicle to prevent leakage? If not, re-running experiments with vehicle-level grouping and reporting the resulting changes in performance would be interesting.

- To support claims of cross-chemistry and cross-mode degradation, explicit train-on-LFP/test-on-NCM (and vice versa), as well as charging-to-discharging transfer experiments may be included. Report macro-F1 scores along with confidence intervals.

- Realism validation. Beyond the reported ≤10 mV voltage deviation, can you provide distributional comparisons between real and generated data, such as MMD or spectral distance? Fault-conditioned precision-recall curves stratified by temperature and SOC would further support realism claims.

- How are the R/R95 and Q/Q95 thresholds computed? Are they defined globally or on a per-vehicle or per-chemistry basis? An ablation under simulated sensor noise or inter-cell variability would help assess their robustness.

- Could you clarify the rationale for excluding recent time-series models such as transformers or state-space architectures? Including such baselines, as well as non-image-based temporal models, would help evaluate whether the CNN-based voltage image approach is optimal.

- Real BMS data often include sensor noise, dropout events, and timestamp misalignments. How are these artifacts modeled during data generation, and how are they handled at training and inference time?

- Have you evaluated model performance under extreme ambient conditions (e.g., –20 °C, +45 °C) or across varying pack topologies (e.g., 28, 92, 96, 124 cells) to test robustness to deployment variability?

- Do you have access to any real-world, fault-labeled datasets, either from lab studies or field deployments, that could be used to assess out-of-distribution generalization and validate the utility of the synthetic data?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a new dataset for classifying the failure mode and severity for lithium-ion batteries.

### Strengths
- A new dataset for the community, suitable for battery data analysis and also time-series classification research

### Weaknesses
- Why classifying battery failure modes or severity is an important task? I have noted that the authors mentioned battery faults may lead to severe consequences in lines 047 - 051. But in order to avoiding that consequences, what we really need is to predict battery failures in advance? How could judging failure modes after existing battery failures help to prevent these problems?
- As the experimental results show, some simple techniques using CNN can deliver pretty good classification performance. So what are the real benefits if the classification accuracy increases by 1%?
- The dataset released seems to be synthetic data, why not releasing real data? How could the model built on the synthetic data perform on real conditions?

### Questions
See weakness.

### Soundness
3

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
3

### Summary
This paper introduces CH-BatteryGen, a large-scale AI-generated dataset designed for fault diagnosis of electric vehicle (EV) power batteries. It integrates real on-board operational data from 1,000 EVs—spanning two major chemistries (NCM and LFP)—with generative modeling to synthesize realistic voltage, current, and temperature signals. The dataset is annotated with four fault categories (“normal,” “high internal resistance,” “low capacity,” and “self-discharge”) and three severity levels, supporting two benchmark tasks: fault classification and fault grading.
A unified evaluation framework is proposed comparing traditional ML (RF, SVM) and deep learning models (LSTM, CNN). Experiments show that CNNs achieve the best overall performance, with F1-scores up to 0.9280 in classification and 0.8813 in grading tasks

### Strengths
- The work presents one of the first comprehensive attempts to generate a large-scale, labeled EV battery fault dataset combining real and synthetic data. The dual-task benchmark (classification and grading) provides a novel framework for fair model comparison under diverse operating modes.
- The experimental design is systematic, evaluating both conventional and deep learning methods across chemistries (LFP/NCM) and conditions (charge/discharge). Quantitative metrics are clearly reported and confusion matrices help illustrate model limitations.
- The dataset could be impactful for both academic and industrial research, especially given the scarcity of publicly available EV fault data. If properly validated and open-sourced, CH-BatteryGen could become a key benchmark for safety-critical battery diagnosis.

### Weaknesses
- Unclear data provenance and labeling process.
The dataset claims to “integrate real on-board operational data with generative modeling methods to build a comprehen-
sive dataset covering 1,000 EVs”, However, it doesn't clarify where generative models are used. The definition and thresholds for these fault types are not rigorously described.

- Synthetic–real data gap.
While the authors state that generated data match real measurements within 30–50 mV deviation, it is unclear whether the generative process preserves electrochemical consistency or simply fits statistical distributions. There is no evidence that the generated sequences capture real DoD (Depth of Discharge) dynamics or aging mechanisms.

- Generalization experiment
Although the paper mentions that model performance significantly drops when transferring across chemistries (LFP ↔ NCM), no detailed results, tables, or quantitative metrics are provided. To support this claim, the authors should include a clear cross-domain evaluation matrix (e.g., train-on-LFP/test-on-NCM and vice versa) with corresponding F1-scores and confusion matrices to assess the true generalization ability of the models.

- Data availability inconsistency.
Although the paper claims the dataset and code are “open-sourced,” the GitHub repository does not actually contain the full data or scripts, making reproducibility and verification difficult.

Simplistic baselines and missing alternatives.
The study focuses on CNN/LSTM but omits more suitable paradigms like contrastive learning or anomaly detection methods, which may be better for rare fault discovery under limited labels.

### Questions
- Dataset composition:
What kind of data is the generated model used to generate?

- Labeling procedure:
How were the four fault labels defined and verified?
Are thresholds for “high internal resistance” or “low capacity” derived from SOH metrics or engineering heuristics?

- Reproducibility:
Please confirm whether the dataset and code will be released publicly before the camera-ready deadline.
If open-sourced, include a data card specifying counts per label and per chemistry.

- Methodology extensions:
Why didn't you consider using contrastive learning or other commonly used anomaly detection methods? Could you provide ablations with raw time-series input to assess whether image preprocessing (super-resolution) affects fairness?

- Denoise: Real battery data contains a lot of noise. Have you used any noise reduction methods?

### Soundness
3

### Presentation
3

### Contribution
3
