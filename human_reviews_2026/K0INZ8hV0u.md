# LEMMA-RCA: A Large Multi-modal Multi-domain Dataset for Root Cause Analysis

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 2

## Abstract
Root cause analysis (RCA) is crucial for enhancing the reliability and performance of complex systems. However, progress in this field has been hindered by the lack of large-scale, open-source datasets tailored for RCA. To bridge this gap, we introduce LEMMA-RCA, a large dataset designed for diverse RCA tasks across multiple domains and modalities. LEMMA-RCA features various real-world fault scenarios from Information Technology (IT) and Operational Technology (OT) systems, encompassing microservices, water distribution, and water treatment systems, with hundreds of system entities involved. We evaluate the performance of six baseline methods on LEMMA-RCA across various settings, including offline and online modes, as well as single and multi-modal configurations. Our study demonstrates the utility of LEMMA-RCA in facilitating fair evaluation and promoting the development of more robust RCA techniques. The dataset and code are publicly available at https://www.lemmarca.info.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
he paper introduces LEMMA-RCA, a large, multi-domain, multi-modal benchmark for root cause analysis. It aggregates real faults from IT microservices and OT water systems, with per-fault timestamps, entity-level metrics, and logs. The dataset supports offline and online evaluation and single or multi-modal inputs. Six baselines are evaluated.

### Strengths
1. The paper makes a strong case that RCA lacks large, open, realistic datasets across domains and modalities, then directly addresses this gap with IT and OT data at second-level granularity and millions of log events.

2. The dataset enables metric only, log only, and multi-modal settings, and provides a concrete online protocol with streaming snapshots. This is timely because most RCA works are offline and single-modal.

3. Six public baselines are run with fixed hyperparameters. Results show consistent gains from multi-modality and expose sharp drops online, which motivates future methods. Tables are clear.

### Weaknesses
1. Many IT scenarios are induced on in-house platforms. OT segments are standardized to two-hour windows and may concatenate normal data around attacks. These choices aid benchmarking but can shift distributions and simplify temporal context, which could bias methods tuned to the benchmark

2. Root cause labels are described at entity level, but the paper does not deeply detail labeling procedures, annotator reliability, or ambiguity handling when multiple entities co-cause failures.

3. Using default hyperparameters improves fairness but may understate some methods. No per-dataset tuning, no ablations on window sizes, log features, or OT KPI construction. Results might change with modest tuning.

Minor: Is a pure dataset paper appropriate for ICLR?

### Questions
1. How are ambiguous multi-cause incidents labeled and scored. Are partial credits or grouped causes considered.

2. How sensitive are results to the log feature pipeline choices, especially template frequency and golden-signal keywords. Any robustness checks.

3. What licensing and redaction steps ensure reproducibility for logs while protecting sensitive info. The dataset is CC BY-ND, but are raw logs fully available or partially sanitized.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces LEMMA-RCA, a multimodal dataset for root cause analysis that includes multiple domains. The proposed dataset aims to address the scarcity of publicly available RCA datasets that contain diverse and realistic fault scenarios. LEMMA-RCA includes metrics and log data collected from IT and OT systems. The authors evaluate 6 RCA methods across offline and online, single- and multi-modal settings. Experimental results show the dataset’s utility for benchmarking different RCA approaches.

### Strengths
- The paper constructs a multimodal RCA dataset, which is meaningful given the data scarcity in this field.
- The proposed dataset is collected from multiple systems (IT + OT) and supports both offline and online RCA evaluation.
- The experimental section compares multiple baselines in both online and offline settings

### Weaknesses
- The microservice failures are injected, while the paper states that the dataset contains "real faults". In Appendix D, the authors describe in detail the steps used to generate failures in microservices systems. If my understanding is correct, this indicates that the faults in the microservices are artificially injected rather than collected from real-world cases. However, in Table 1, the authors state that their dataset contains real faults. These two statements appear contradictory to me. The authors need to clarify whether the faults are injected and define what is meant by real faults.

- The paper’s statements about existing RCA methods are inaccurate. The paper conflates offline RCA with supervised data-driven methods, which is conceptually inaccurate. Many RCA algorithms (e.g., CIRCA, TraceRCA, BARO, and many LLM-based methods, which are cited in this paper) are unsupervised methods that do not require any training or retraining when new faults occur. However, the paper states that “most RCA methods are designed for offline use, requiring extensive data collection and full retraining for new faults,” which clearly does not reflect the current state of research. Furthermore, the authors’ definition of "data-driven RCA" is unclear and misleading. In RCA, “data-driven” refers to inferring causal relationships directly from observed data, not to methods that perform better with more data. The current words ignore the dominant unsupervised approaches in both single-modal and multi-modal RCA.

- The number of failure cases in each dataset is not explicitly given. It appears that the Product Preview and Cloud Computing sub-datasets each contain only four failure cases. This is insufficient for a comprehensive evaluation of RCA. I suggest that the authors clearly state the number of failure cases in the paper.

- The advantages of Lemma-RCA over other datasets are not discussed in detail. As mentioned by the authors in the paper, there are already many open-source RCA datasets, including multimodal datasets and datasets obtained through failure injection. However, the authors only provide a vague comparison of different datasets in Table 1. This is not sufficient to demonstrate the advantages of the LEMMA-RCA dataset over others. The authors should present more cases to illustrate the advantages of LEMMA-RCA in terms of quality, quantity, and other aspects compared with similar datasets.

- Figures 3 and 5 only provide a simple presentation of monitoring metrics without sufficient explanation, which makes them confusing. Moreover, they do not clearly show the fault propagation relationships among the metrics. I suggest that the authors provide more detailed explanations.

- Section 3.1 includes many technical details that are not directly related to data collection. The authors should move the details in this section to the appendix and provide a more detailed description of the characteristics and advantages of the dataset itself.

- In Section 3.2, the authors mention that logs from some system entities were excluded. However, the paper does not describe the specific filtering criteria.

### Questions
1. What's the definition of "real faults"? Are the microservices failures in LEMMA-RCA injected?
2. What's the difference between data-driven and non-data-driven RCA methods? Why do the offline methods need per-case retraining?
3. What's the number of failure cases in each sub-dataset?
4. What are the advantages of LEMMA-RCA compared with the other RCA datasets?
5. What's the criteria for filtering the logs?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces LEMMA-RCA, a new large-scale, multi-modal, and multi-domain dataset designed to facilitate research in RCA. The authors argue that progress in data-driven RCA has been hampered by a lack of suitable public benchmarks, with existing datasets often being small, proprietary, synthetic, or limited to a single domain/modality.

LEMMA-RCA addresses this gap by providing a comprehensive collection of data from real-world systems, including IT operations (microservices) and Operational Technology (water treatment/distribution systems). The dataset features diverse and realistic system faults, involves hundreds of system entities, and contains multiple data modalities, primarily time-series metrics and textual logs. The authors provide a detailed description of the data collection process, preprocessing steps, and fault scenarios. To demonstrate the dataset's utility, they evaluate 6 baseline RCA methods in both offline and online settings, highlighting the performance differences across modalities and settings. The dataset is made publicly available to encourage fair comparisons and advance the field.

### Strengths
1. The paper successfully identifies and fills a major gap in the RCA research landscape by providing a large-scale, public benchmark.
2. The dataset contains real system faults (or realistic induced faults) across different domains (IT and OT), which is a major step up from datasets with purely synthetic or simplistic faults.
3. The inclusion of both time-series metrics and textual logs allows for the development and evaluation of multi-modal RCA methods. The data is structured to support both offline and online evaluation settings.
4. The paper provides clear and detailed descriptions of the data sources, preprocessing, fault scenarios, and ground truth labeling process, which is essential for a dataset paper.

### Weaknesses
1. The evaluated baselines are primarily traditional causal discovery or statistical methods. Given the recent surge of interest in using LLMs for diagnostics and RCA, the absence of any LLM-based baseline is a notable omission. Including even a simple zero-shot LLM baseline would have provided a valuable modern reference point.
2. The feature extraction pipeline for logs is quite specific and multi-faceted (combining template frequency, keyword signals, and TF-IDF). This introduces a potential dependency, as future work might achieve different results based on their own feature engineering. It would be helpful if the authors clarified whether these preprocessed features are part of the release.
3. The adaptation of offline methods for online evaluation uses a simple stopping criterion ("similar results appear three times"). This is heuristic and lacks the rigor of more standard online evaluation protocols like prequential evaluation. While it demonstrates the challenge, the protocol itself could be stronger.
4. The paper focuses on single root causes tied to induced faults. Real-world incidents often involve multiple interacting root causes or complex cascading failures. It's unclear if the dataset captures this complexity or how such scenarios are labeled and evaluated.

### Questions
See weaknesses.

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
4

### Summary
The paper describes a dataset that the authors are releasing on root cause analysis (RCA) spanning 4+ different domains and including both time-series of various metrics and textual logs.  The datasets are marked with root causes, that are real, and not simulated. The paper evaluates various existing RCA methods in both online and offline mode.

### Strengths
1.  Releases a useful data resource for the community on root-cause analysis.which is distinguished from existing datasets by being multi-modal (textual logs and time-series), and real.

2.  Evaluates six existing methods on the released datasets.

### Weaknesses
1.  It is difficut to justify an ICLR paper just on the basis of releasing a dataset.  There are conferences with special tracks on datasets and benchmarking.  The paper is best submitted to such tracks.

2.  Most of the methods evaluated are not mainstream to the AI/ML/DL community, so relevance to ICLR is of question.  Here are some papers that are missed:

    2a:  Root cause analysis of outliers with missing structural knowledge  N Okati, SHG Mejia, WR Orchard, P Blöbaum… -NeurIPS 2025
     
     2b. Budhathoki, Kailash, et al. "Causal structure-based root cause analysis of outliers." International conference on machine learning. PMLR, 2022.

    2c.  Nagalapatti, Lokesh, et al. "Robust Root Cause Diagnosis using In-Distribution Interventions." ICLR 2025.

3.  Presentation issues:

     3a: The sentence in line 183 claiming that "fault ran the microservice" is confusing.  How can a fault run a service?
     
      3b. The term multi-modal may be a bit mis-leading in current AI/ML community.  

       2c.  Citations are sloppy.  For example, Li 2022 is repeated.  Citation Yu 2023 does not mention venue.

### Questions
Q1: The description of the type of data and attach for the SWaT and WADI datasets misses important details like the type of the attack, and what are the recorded metrics, and logs.

Q2.  Around line 278, the method of discretizing data into continuous values is not well-justified.  Isn't it better to directly do the RCA on the discrete data?

Q3: The description of baselines is quite poor quality.  A paper like this, should ideally first define a common framework, and describe each method with common termilogy.  Currently, it comprises disconnected snippets which may have been lifted directly from individual papers without much attempt to actually explain and contrast each baseline.

.

### Soundness
3

### Presentation
3

### Contribution
2
