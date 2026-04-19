# BMAD: Benchmarks for Medical Anomaly Detection

- Decision: Reject
- Scores: 3, 3, 5, 6

## Abstract
Anomaly detection (AD) is a fundamental research problem in machine learning and computer vision, with practical applications in industrial inspection, video surveillance, and medical diagnosis. In medical imaging, AD is especially vital for identifying anomalies that may indicate rare diseases or conditions. Despite its significance, there is a lack of a universal and fair benchmark for evaluating AD methods on medical images, which hinders the development of more generalized and robust AD methods in this specific domain. To bridge this gap, we introduce a comprehensive evaluation benchmark for assessing AD methods on medical images. This benchmark encompasses six reorganized datasets from five medical domains (i.e. brain MRI, liver CT, retinal OCT, chest X-ray, and digital histopathology) and three key evaluation metrics, and includes a total of fifteen state-of-the-art AD algorithms. This standardized and well-curated medical benchmark with the well-structured codebase enables comprehensive comparisons among recently proposed anomaly detection methods. It will facilitate the community to conduct a fair comparison and advance the field of AD on medical imaging.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces BMAD (Benchmarks for Medical Anomaly Detection), a benchmark dataset for anomaly detection (AD) containing six reorganized public datasets from five medical domains, three evaluation metrics, and fifteen AD algorithms. Analysis of the relative advantages and disadvantages of the AD algorithms is included. A stated motivation is the lack of existing uniform, comprehensive, standardized and fair benchmarks for medical anomaly detection.

### Strengths
-	Contributes a well-organized benchmark dataset to the community
-	Careful evaluation and analysis of implemented algorithms including hyperparameter optimization and multiple executions

### Weaknesses
-	Limited technical novelty
-	Definition and scope of anomalies possibly more meaningful if further extended, especially in the medical domain

### Questions
1. The overarching claim of a “universal and fair” benchmark appears difficult to substantiate, as representation from five domains necessarily remains a small subset of possible image anomaly cases in medicine. The degree of fairness regarding domain selection might be further justified. For example, thin structures appear relatively less represented.
2. As raised in the Introduction, a common practical concern for anomaly detection in the medical domain is when some (more-common) disease classes are known and labelled/annotated, but other rarer classes not in the initially known set of diseases may occur. Some analysis of such tasks might thus be relevant.
3. A number of minor spelling/grammatical errors might be addressed, e.g. “a fair comparing among these methods”, “well-structure, easy-use codebase”, “samll tumor localization”, “large pre-train models”, etc.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a benchmark for anomaly detection in medicine. The paper presents six medical datasets as an evaluation benchmark for anomaly detection. The paper evaluates multiple state-of-the-art algorithms on these datasets using the evaluation metrics of AUROC, Per-Region Overlap, and Dice. The benchmark is integrated in a code base and aims to facilitate and simplify the use of the data and algorithms in the medical image analysis community. The advantages and disadvantages of the evaluated algorithms are evaluated and discussed.

### Strengths
- the paper proposes a standardized access to multiple medical image datasets
- standardized implementation of multiple algorithms
- paper is well written and easy to follow

### Weaknesses
I appreciate the efforts put into this submission, but I find it challenging to pinpoint any technical, methodological, or experimental contribution that aligns with the standards of acceptance for ICLR. For example, the proposed datasets have been studied in light of anomaly detection before. Overall, I think the initiative can be very useful for the medical image analysis community, and I would therefore encourage resubmission to a more applied medical image analysis venue.

### Questions
-

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Anomaly detection is crucial in fields like medical imaging, but there's a lack of standardized evaluation benchmarks. To address this, a comprehensive benchmark for assessing anomaly detection methods in medical images was introduced in this paper. This benchmark includes datasets from five medical domains, 15 state-of-the-art algorithms, and provides a robust framework for advancing anomaly detection in medical imaging.

### Strengths
The paper addresses the lack of a universal and fair benchmark for evaluating anomaly detection (AD) methods on medical images. This is good contribution to the field, as it fills a significant gap and provides a standardized platform for evaluating AD methods in the context of medical imaging.

The paper demonstrates a good level of quality in terms of dataset curation and algorithm integration. It organizes six datasets from five different medical domains, ensuring diversity and representation of real-world medical scenarios. 

The paper is clear in its presentation. It is well-structured and provides detailed information about the benchmark, including the datasets, evaluation metrics, and algorithms used.

The benchmark is very importance for the field of anomaly detection in medical imaging. Medical imaging have a crucial role in diagnosing / monitoring various diseases. A standard benchmark can lead to the development of more reliable and robust AD methods, ultimately benefiting healthcare and patient outcomes.

The presence of Table 3, which provides insights into both the inference times and the performance of the AD models, is interesting as it not only facilitates a more robust comparison of the models but also helps researchers when assessing the efficiency of their proposed methodologies in the context of anomaly detection.

### Weaknesses
Regarding table 2, it would be highly recommended to represent the results in graphs for each data set (showing the relationship between methodology and performance). This would allow a more clear visualization and facilitate the comparison of the results between the different methodologies, improving understanding and the capacity to identify trends and variations.

While the paper mentions a well-structured codebase, it's crucial to provide insights into how this codebase is organized and made available to the community. Detailed documentation and code accessibility are essential for other researchers to replicate and extend the experiments.

The paper acknowledges that most of the data used in the benchmark is collected in advanced countries, which may introduce geographical and sampling biases. This limitation could potentially affect the generalizability of the benchmark to a broader range of medical imaging scenarios. To address this, the paper could suggest ways to mitigate these biases.

The paper mentions that the hyperparameter settings for the evaluated algorithms were based on the original works, and not all hyperparameters achieved their optimal values for specific datasets. This could potentially lead to suboptimal performance for some algorithms. To address this, the paper could provide recommendations or guidelines for tuning hyperparameters to improve the performance of the AD algorithms on the benchmark datasets. This would make the benchmark more valuable.

In addition to the mentioned quantitative metrics, incorporating qualitative analysis and insights from medical professionals regarding the benchmark's practical applicability in real-world anomaly detection scenarios would enhance its credibility and utility.

### Questions
As more algorithms and datasets become available, how will the benchmark be updated and maintained?

What are the plans for including new datasets and algorithms in the future?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a benchmark study for medical anomaly detection including 1) reorganizing 6 existing medical datasets and 2) running 15 SOTA algorithms evaluated on 3 metrics.

### Strengths
1. The authors utilize proper preprocessing techniques to reorganize the six datasets and systematically run baselines with reasonable evaluation metrics.

### Weaknesses
1. As also noted by the authors, ethical and fairness concerns exist among the datasets and need to be addressed properly.

2. Some part of the writing needs to be improved (e.g. "defeat->default parameters setting").

### Questions
1. It would be more helpful to discuss the impact of hyper-parameters especially for methods that the authors find hard to achieve optimal values reported in the original paper. 

2. From the section 4.1 implementation details, the authors seem to follow default settings first and then tune the learning rate/threshold only. Other important hyper-parameters, such as the weighting terms in losses, seem to be missing. It would also be very helpful to document the results using each combination so readers would know what works and what doesn't, which will also improve the reproducibility of a benchmark study.

3. It would be interesting to have a section discussing the robustness of each algorithm analyzed in training (e.g. which is easier to converge).

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
