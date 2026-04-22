# The Seismic Wavefield Common Task Framework

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 8, 2, 6

## Abstract
Seismology faces fundamental challenges in state forecasting and reconstruction (e.g., earthquake early warning and ground motion prediction) and managing the parametric variability of source locations, mechanisms, and Earth models (e.g., subsurface structure and topography effects). Addressing these with simulations is hindered by their massive scale, both in synthetic data volumes and numerical complexity, while real-data efforts are constrained by models that inadequately reflect the Earth's complexity and by sparse sensor measurements from the field. Recent machine learning (ML) efforts offer promise, but progress is obscured by a lack of proper characterization, fair reporting, and rigorous comparisons. To address this, we introduce a Common Task Framework (CTF) for ML for seismic wavefields, demonstrated here on three distinct wavefield datasets. Our CTF features a curated set of datasets at various scales (global, crustal, and local) and task-specific metrics spanning forecasting, reconstruction, and generalization under realistic constraints such as noise and limited data. Inspired by CTFs in fields like natural language processing, this framework provides a structured and rigorous foundation for head-to-head algorithm evaluation. We evaluate various methods for reconstructing seismic wavefields from sparse sensor measurements, with results illustrating the CTF's utility in revealing strengths, limitations, and suitability for specific problem classes. Our vision is to replace ad hoc comparisons with standardized evaluations on hidden test sets, raising the bar for rigor and reproducibility in scientific ML.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a Common Task Framework (CTF) for benchmarking machine learning methods on seismic wavefield modeling tasks. The framework aims to provide standardized evaluation and fair comparison in seismology with forecasting, reconstruction, and generalization problems. It includes three datasets (global wavefields, crustal simulations, and fiber-based DAS recordings), defines twelve evaluation metrics, and compares 14 different highly-cited ML models.

### Strengths
1. The paper introduces a benchmark suite for seismology, trying to address reproducibility and comparability challenges in scientific problems.

2. It offers a diverse range of datasets, covering different scales and difficulties.

3. It provides a comprehensive metric design with 12 evaluation metrics, which reflect multiple facets of the modeling: forecasting accuracy, robustness to noise, data limitation, and parametric generalization.

4. The paper provides a benchmark including 14 different methods.

### Weaknesses
1. The paper uses inconsistent names: “ctf4seismology,” “Seismic Wavefield CTF,” and “seismo dataset.” This inconsistency is confusing. 

2. The details of the datasets are unclear. A summary table should formally describe each dataset's size, format, train/validation/test splits, etc.

3. Important details are missing for each task, such as noise levels, limited data size (M), and number of forecasted time steps (m).

4. The paper would benefit from data visualizations and formal descriptions to help readers better understand the data and tasks. In particular, a more detailed explanation of the underlying physics is needed to highlight what makes seismic wavefield data different from general time-series data. Section 2.1 (called “Seismic Wavefields as Spatio-Temporal Systems”) does not clearly show what the system is and how spatial and temporal structures look like. To encourage broader community participation in the proposed CTF, the description should be more helpful to those non-seismologists in understanding the physics and challenges of the data.

5. No training hyperparameters and model settings are provided for the evaluated methods.

6. Table 1, Figures 3 and Figure 4 show almost the same information in different formats, consuming space without adding insight.

Overall, I appreciate the work to propose a CTF for a scientific problem. However, the paper currently reads more like an introduction or announcement of the upcoming Kaggle competition rather than a formal scientific paper. Many details are missing, making it hard to fully assess the paper.

### Questions
Please refer to the Weakness

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a Common Task Framework (CTF) for machine learning in seismology, specifically targeting the modeling and reconstruction of seismic wavefields. The authors present three curated datasets (global, crustal, and local scales) and 12 task-specific metrics, aiming to standardize evaluation and comparison of ML methods in this domain. The framework is inspired by successful CTFs in other fields (e.g., ImageNet) and is demonstrated through benchmarking a variety of ML models on the global wavefields dataset.

### Strengths
1. Meaningful Contribution:
Seismology is a domain where ML adoption is accelerating, but progress is hampered by inconsistent evaluation, weak baselines, and reporting bias. 
2. Well-Designed Evaluation Protocol:
The authors propose a multi-metric scoring system that captures different aspects of model performance (forecasting, reconstruction, noise robustness, limited data, parametric generalization). 
3. Benchmarking and Transparency:
The paper benchmarks a diverse set of ML models (RNNs, neural operators, DMD, etc.) and provides detailed results, including cases where complex models fail to outperform simple baselines. This transparency is valuable for the community and highlights the challenges ahead.
4. Open Platform and Reproducibility:
The datasets, code, and evaluation scripts are made publicly available, with plans for a Kaggle competition. This openness will foster community engagement and accelerate progress.

### Weaknesses
Performance and Impact:
The results show that most ML models struggle to outperform trivial baselines on the provided tasks. While this highlights the difficulty and the need for better methods, it also suggests that the immediate impact of the platform may be limited until more sophisticated models or richer datasets are available. It will be helpful to classify the dataset based on the difficulty level, so that the ML area can design solutions from easy to difficult task gradually.

### Questions
N/A

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
3

### Summary
This paper proposed a Common Task Framework (CTF) to evaluate machine learning algorithms for seismic wavefields. Three distinct datasets across various scales are included. The authors also established a series of tasks and proposed their corresponding evaluation metrics. They further provided the benchmarking results of 16 methods on the global wavefield dataset.

### Strengths
The authors provided helpful descriptions of the models and the seismic wavefields in the datasets.

### Weaknesses
While the idea of establishing a CTF is interesting, I feel that the current work has limited novelty and the experimental evaluation appears incomplete. My details comments are as follows.
1. Although three datasets are included in the proposed CTF, the authors did not clearly explain how these datasets relate to one another or why they should be integrated into a single framework.
2. The authors didn’t highlight the challenges involved in generating these datasets. For instance, the global wavefield dataset is generated using a public Earth model, and the crustal dataset is an extension of the existing work. Besides, the volume of each dataset is also relatively small.
3. In the experiment section, the authors presented benchmarking results only for the global wavefield dataset, and the results on the other two datasets are missing.
4. The related works section is missing.

### Questions
1. For the DAS dataset, could the authors provide a description of the models used for the generation of wavefields?
2. In the experiment section, could the authors provide some visualizations of the predictions of various methods to give readers more intuition about tasks and model performance?  
3. [Minor] Figure 1 is organzied in a different order from the introduction of three datasets in Section 2.1.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work presents a new common task framework for seismic wavefield estimation tasks. The framework seems to be hosted on github, and is proposed to be hosted on Kaggle platform in future. The framework currently seems to incorporate three global scale datasets from seismology domains. The tasks are defined with multiple perspectives with 12 subtasks and scoring metrics. Surprisingly, modern models tend to perform worse than zero-knowledge baseline model of predicting only zeros of all tasks.

### Strengths
This manuscript is presenting a new aggregated datasets and unifying evaluation system for tasks related to seismic wavefields. Seismology contains diverse tasks and domains, yet the proposed task has very large scale (i.e. planetwise), and immediate extensibility from Earth to Mars. The three datasets proposed seem to be well accepted and analyzed by domain experts. As an ML expert with seismological application research experience, I see the value underlying this work as a domain-expert-curated dataset-evaluation combination which takes care of two key issues: 1) is the data trustworthy, and 2) is the metric meaningful. These two questions had been and will be of greatest concern for a serious AI/ML driven interdisciplinary research.

### Weaknesses
Seismology contains lots of distinctive tasks from various aspects of solid earth (be it from the Earth, Moon, or Mars). Planet-wide seismic wavefield tasks, which are the primary focus of this manuscript, represent a portion of the wide field. The claimed name 'ctf4seismology' may be what the authors want to reach, but as of current form, the name is too assuming and overstating what the current dataset-evaluation-task proposal contains. I think one more spoonful of factual representation would be a good addition.

Also, perhaps due to the page limit, the materials presented in the 9 page limit did not provide sufficient information for me to gauge what would be a reasonable method and how to use the framework if I have a suitable method. This is the perspective of ML practitioner, who'd like to get an idea of 'how I can use it'. If there is a public repository, it would be much better to have a link to it explicitly shared in the manuscript.

### Questions
Are the twelve tasks equally important for seismic wavefield research? The current proposal seems to suggest a simple averaging of all subscores --- but I suspect that depending on the research direction some tasks will have greater relevance than others. Is this something left for seismologists? As it is presented in the current form, it seems like a call for ML practitioners to play with this new set of tasks.

Is there a publicly available repository to share the baseline codes and/or solicit collaboration, as alluded by the future directions given in the text? Would the proposed CTF extend well to incorporate wide range of distinct data-driven tasks in seismology (not only global but also local and/or heterogeneous)? This question may be the most important one if the authors plan to claim the audacious name of 'ctf4seismology'.

### Soundness
3

### Presentation
2

### Contribution
3
