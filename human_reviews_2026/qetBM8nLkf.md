# PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Irregular temporal data, characterized by varying recording frequencies, differing observation durations, and missing values, presents significant challenges across fields like mobility, healthcare, and environmental science. Existing research communities often overlook or address these challenges in isolation, leading to fragmented tools and methods. To bridge this gap, we introduce a unified framework, and the first standardized dataset repository for irregular time series classification, built on a common array format to enhance interoperability. This repository comprises 34 datasets on which we benchmark 12 classifier models from diverse domains and communities. This work aims to centralize research efforts and enable a more robust evaluation of irregular temporal data analysis methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a unified preprocessing framework for irregular time series (ITS). The framework standardizes common steps and ships with curated datasets and baseline implementations. According to the paper, the pipeline supports multiple ITS tasks and provides a consistent and computationally efficient interface that lowers the barrier to reproducing baselines and evaluating new models across diverse benchmarks.

### Strengths
+ It handles a real problem in ITS learning research, despite not being a traditional ICLR paper
+ The framework covers multiple ITS settings, improving comparability across papers
+ The proposed pipeline to Raw ITS into model-ready tensors can accelerate experimentation and deployment

### Weaknesses
+ As the main contribution is the framework itself, the paper could be more focused on how it was implemented instead of evaluating the included models.
+ As it was proposed for unified ITS, some relevant families seem absent from the main implementation/benchmark (e.g., latent ODE[1]/RNN variants beyond NCDE, state-space models/SSMs, and foundation models as TabPFN[2]) despite being used for ITS with relevant results.
+ TIMESNET is reported as a transformer-based model, but it is not based on the attention mechanism.


[1] Rubanova, Yulia, Ricky TQ Chen, and David K. Duvenaud. "Latent ordinary differential equations for irregularly-sampled time series." Advances in neural information processing systems 32 (2019).

[2] Hollmann, Noah, et al. "Tabpfn: A transformer that solves small tabular classification problems in a second." arXiv preprint arXiv:2207.01848 (2022).

### Questions
+ As the MIMIC-III is available only to credentialed users, does the framework preprocessing code support the whole dataset?
+ What determined the models chosen for inclusion? Were there any barriers to adding existing models from other families?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces pyrregular, a unified framework designed to standardize the representation, analysis, and benchmarking of irregular time series (ITS) data. It proposes a clear taxonomy of irregularity types, implements an interoperable array structure that combines the flexibility of xarray with the memory efficiency of sparse COO tensors, and compiles a large repository of 34 naturally irregular datasets. The authors conduct a comprehensive evaluation of 12 classifiers spanning diverse modeling paradigms (including statistical, neural, and differential-equation-based approaches), offering the first systematic and reproducible benchmark for ITS classification. The work fills an important gap by centralizing disparate research efforts on irregular time series and providing a standardized, extensible foundation for future studies.

### Strengths
1. The paper clearly identifies a major gap in the field: the lack of interoperable tools and standardized benchmarks for irregular time series classification, which has long hindered cross-domain reproducibility and comparison.

2. The proposed array format elegantly combines xarray and sparse COO representations to achieve both flexibility and memory efficiency, while supporting multiple types of irregularity (uneven sampling, partial observation, raggedness).

3. The authors assemble a substantial suite of 34 real-world ITS datasets and evaluate 12 state-of-the-art models across multiple irregularity types, dataset scales, and sequence characteristics. The empirical analysis is thorough and yields valuable insights into the comparative strengths of classical versus deep models for ITS classification.

4. The framework, benchmark design, and dataset curation will likely become an important community resource for reproducible research and future extensions beyond classification.

### Weaknesses
1. While the benchmark covers a diverse set of classical and neural classifiers, recent developments in foundation or LLM-based time-series models (e.g., Time-LLM, CALF, or multimodal pretraining frameworks) are not discussed. Including such models, even conceptually, could contextualize where pyrregular fits within the broader trend toward generalist temporal modeling.

2. The paper briefly mentions runtime comparisons, but a deeper discussion of computational efficiency, scalability with data size, and memory footprint across model classes would strengthen the benchmarking narrative—especially given that one of pyrregular’s core motivations is interoperability and resource efficiency.

3. The paper focuses exclusively on classification, leaving forecasting, anomaly detection, and imputation for future work. A brief demonstration or pilot benchmark on another task could have further illustrated the generality of the proposed framework.

4.  For complex datasets such as MIMIC-III or PhysioNet 2019, a more in-depth analysis of model behavior and domain-specific challenges (e.g., handling clinical missingness patterns or label imbalance) would make the study more informative for practitioners in healthcare and other applied domains.

### Questions
see above

### Soundness
3

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
This paper studied the fragmentation in irregular time series research by proposing a unified framework and standardized dataset repository for irregular time series classification. Irregular time series, characterized by uneven sampling, partial observation, and raggedness, are common in many applications

### Strengths
The paper focuses on long-standing pain points in irregular time series research such as fragmented tools, lack of standardized benchmarks, and reliance on artificially induced irregularity.  The benchmark design is rigorous and comprehensive. Authors have evaluates a total of twelve methods over 34 datasets.

### Weaknesses
The paper studies exclusively on classification tasks and excludes other important time series tasks  such as forecasting, anomaly detection from benchmarking. While the paper mentions potential extensions, authors did not provide any technical details or preliminary results for these tasks.  Despite emphasizing practicality, authors provide incomplete details on computational ciost while it reports training/inference delay for classifiers, this paper did not analyze how the framework scales with increasing dataset.

### Questions
Can authors explicitly define how it distinguishes natural irregularity from artificial irregularity ?

### Soundness
3

### Presentation
3

### Contribution
3
