# Scaling Wearable Foundation Models

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 8, 5

## Abstract
Wearable sensors have become ubiquitous thanks to a variety of health tracking features. The resulting continuous and longitudinal measurements from everyday life generate large volumes of data. However, making sense of these observations for scientific and actionable insights is non-trivial. Inspired by the empirical success of generative modeling, where large neural networks learn powerful representations from vast amounts of text, image, video, or audio data, we investigate the scaling properties of wearable sensor foundation models across compute, data, and model size. Using a dataset of up to 40 million hours of in-situ heart rate, heart rate variability, accelerometer, electrodermal activity, skin temperature, and altimeter per-minute data from over 165,000 people, we create LSM, a multimodal foundation model built on the largest wearable-signals dataset with the most extensive range of sensor modalities to date. Our results establish the scaling laws of LSM for tasks such as imputation, interpolation and extrapolation across both time and sensor modalities. Moreover, we highlight how LSM enables sample-efficient downstream learning for tasks including exercise and activity recognition.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper describes self-supervision training methodology (infilling of randomly-masked patches) applied to a very large data set of multi-modal time series data collected by wearable devices under natural usage.   The authors describe a comprehensive analysis of the model performance as a function of training data volume, compute resources and model complexity across a set of generative tasks (imputation, extrapolation/forecasting and interpolation).  Additionally they also evaluate performance of the resulting pre-trained encoder (and compare this with a supervised learning baseline) on a pair of discrimination tasks: binary exercise/non-exercise classification and multi-class activity classification.

### Strengths
## This paper represents strong contributions in the following areas: 

### Originality:  This paper is the first to evaluate self-supervised learning methods on a multi-modal physiological data set at this scale.  The authors describe a novel modeling approach as well as a comprehensive description of their ablation experiments. 

### Quality:  The authors present rigorous, systematic and thorough evaluation of performance on the generative tasks as a function of pre-training data volume, and model size. 

### Significance: The analysis of generative tasks is comprehensive and may be of interest both within the problem domain (models for multi-modal data from wearable devices) as well as more broadly.  However, the lack of compelling evidence for model generality across a range of downstream tasks may limit the broad interest. 

### Clarity:  The paper is fairly clear regarding the data source (including featurization), modeling approach, performance analysis and related ablation experiments.   Some details (discussed in the section below) are missing, but this is not likely to impact the general conclusions.

### Weaknesses
# Brief Summary #
This paper had some important weaknesses that are listed briefly below, with more detailed comments and some suggested steps to address these weaknesses following the list:

1. [Most concerning] Lack of variety in downstream task evaluation.  The paper evaluates only two closely-related downstream tasks (binary and multi-class activity classification) that utilize the learned representations.   Given that one of the main characteristics of ‘Foundation Models’ is that they are readily adaptable to a wide variety of downstream tasks, it is a stretch to consider the LSM model to be a foundation model rather than a tool for time series interpolation/extrapolation. 
2. No analysis of model performance in a manner that is agnostic to the downstream task (e.g. using measures of unsupervised embedding quality)
3. Incomplete comparison of model performance on downstream tasks against baselines— LSM performance was compared with SL-trained model having the same ViT architecture, but not against other classifier types.
4. Absence of some details and explanation on the problem setup for downstream classification tasks.
5. The authors are missing discussion/citation of some past publications that represent closely-related research (learning general-purpose representations from multi-modal physiological time series captured by wearable devices) 


# Additional Discussion Details #

## Issue 1: [Most concerning] Lack of variety in downstream task evaluation.
Other than an extensive analysis of the interpolation/imputation/extrapolation tasks (which are useful proxies), the paper evaluates performance on only two closely-related downstream tasks utilizing the LSM embeddings: binary exercise/non-exercise classification, and multi-class activity classification.  This makes it hard for the reader to have confidence that LSM can truly be considered a “foundation model”.  Per the proposed terminology set forth in Bommasani, et al., ‘On the Opportunities and Risks of Foundation Models’, in addition to being trained via self-supervision on diverse large-scale data, a FM “can be adapted to a wide range of downstream tasks”.    Moreover, there would be particular value and impact in demonstrating that the LSM embeddings are useful for ***physiologically-relevant*** downstream tasks.  

### Suggestions to address Issue 1:
The data source described by the authors contains some additional metadata/labels that could be used as targets for downstream tasks, building evidence for the general-purpose utility of the LSM embeddings.   The metadata includes subject-reported age, sex, and weight (possibly BMI, which is listed in table 2a) that can be used as regression and/or classification targets.  

Other targets, particularly those with some direct physiological relevance, would also provide additional evidence for general-purpose utility of the model.  I recall that Fitbit generates a ‘Cardio Fitness Score' that represents an estimate of VO2max.  If VO2max is available it could be utilized as either a regression target or a classification target (simple binning into high/low or tertiles/quartiles).  This would be a good choice of task, since there is a first-principles argument that the temporal relationship between physical effort and heart rate response *should* contain information about VO2max.  

## Issue 2: No analysis of model performance in a manner that is agnostic to the downstream task
In a complement to suggestions for issue 1 above (more downstream task variety) providing some task-agnostic analysis of embedding ‘quality’ would also help the argument that LSM is in fact a general-purpose foundation model. The authors provide some t-SNE visualization of embedding distribution in Figures 8 and 9, but no accompanying quantitative analysis.  

### Suggestions to address Issue 2:
Include some quantitative measure of how well-distributed the model embeddings are, and/or how well 5h data segments generated by the same individual can be distinguished from segments generated by other individuals.  

Multiple measures of embedding “quality” have been reported that are useful for evaluating and comparing models in a manner that is agnostic to the downstream task (for a recent review/comparison, see Tsitsulin, *et al.* ‘Unsupervised Embedding Quality Evaluation’ https://arxiv.org/abs/2305.16562).  

## Issue 3: Incomplete comparison of model performance on downstream tasks against baseline.
The authors compare the performance of their pre-trained model against a supervised learning baseline for the two downstream classification tasks.   However, this baseline (ViT with 100M parameters) may not be optimal, particularly for the relatively simple task of exercise vs. non-exercise binary classification.   It’s worth noting that based on the test set summary in Table 2b simply guessing the majority class (non-exercise) every time would produce ~65% accuracy, but the 100M-parameter ViT only achieves 70.9% with supervised learning (Table 3b).  

### Suggestions to address Issue 3:
Compare performance on both downstream classification tasks using an alternate modeling approach such as random forest or regularized logistic regression (or other boilerplate ML classifiers).  Given the problem statement (classify a period of time as exercise vs. non-exercise, or by exercise type, using a time series of 27 features), it is possible that RF or LR could outperform the SL ViT.  Even if they don’t, it could give an indication of the performance gap between these ‘non-deep’ supervised ML methods and the ViT SL baseline.  
 
## Issue 4: Absence of some details and explanation on the problem setup for downstream classification tasks.
Section 4.2 on discriminative tasks is very light on details, to the extent that it would be difficult or impossible for other researchers to reproduce the findings (using their own data and models).  The data segments are 5 hours long (so presumably only a subset of that time contains the activity/exercise of interest) but it is not clear how the activity period within the segment is labeled or how the classification is evaluated. Tables 3b and 3c report mAP as a performance metric, which makes me think that there is some time segmentation or drawing of boundary-boxes involved in the downstream task, but this is not explained in the text.  In contrast, Appendix section B.3 states “This is likely as there are significant periods of walking in the 5-hour inputs, even if the activity is labeled otherwise”, which implies that the entire 5h inputs are assigned a single class label.  This is confusing (or at least unclear) to the reader.  

### Suggestions to address Issue 4:
Provide more detail on the downstream classification problem setup, accompanied by a diagram or illustration (at least in the Appendix) that illustrates visually how the model outputs are evaluated against the ground truth labels. 

## Issue 5: The authors are missing discussion/citation of some past publications that represent closely-related research.  

Some relevant published work that is fairly similar in scope (learning general-purpose representations from multi-modal physiological time series captured by wearable devices) exists, but these are not cited or discussed.  One clear example is  D. Spathis *et al.*, Self-supervised transfer learning of physiological representations from free-living wearable data (2020, https://arxiv.org/abs/2011.12121) which utilized pretext tasks to pre-train models for multi-modal inputs from wearables (heart rate and raw IMU), then evaluated the resulting embeddings on a variety of downstream tasks.  

### Suggestions to address Issue 5:
At least include the Spathis *et al.* paper (listed above) among the citations.  Given the close match in scope, this paper should probably also be included in Table 1 despite not referring to its work as a “foundation model” (being from 2020, it predates the popularization of the “foundation model”).    It may also be worth doing an additional literature review specifically looking for pre-2021 papers that do not use the term “foundation model” but are clearly pursuing the same objective.

### Questions
1. In section 3.1 (data preprocessing) provide an explicit statement of which physiological metrics ("features") are calculated on-device vs, which are calculated on the stored centralized data.  This section mentions broadly that the raw sensor signals are not stored centrally or on-device (which seems to imply that all of the features are calculated on-device), but whether this is the case should be stated clearly. 
2. Toward the end of section 3.1 (data preprocessing) the authors state “Within each 300-minute window, missing data between valid data points was linearly interpolated, and leading missing minutes were backfilled.”  Was the **test set** that was used for evaluating generative tasks free of these missing values that had been infilled by linear interpolation/backfilling?  If so, these time points should be excluded from the calculation of generative task performance (MAE/MSE) because they do not represent real measurements (but rather synthetic values created in data preparation using a human-chosen method).  Please comment on how this issue was handled.  
3. Typo(?) on line 202.  “Kurtosis is the kurtosis of the BFP signal”.  Should (I think) be BPF.
4. Why was the data input length of 5h chosen?  What other lengths were considered, and how were these compared to ultimately land on the choice of 5h.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The authors have introduced a multimodal foundation model for sensor data. The input is mix of raw sensor samples and some preprocessed discrete derived samples due to some understandable practical constraints. They have set up a frame work to prove generative and inferential capabilities allowing the model to be regarded as foundational. With a good representative choice of model architecture, scales of data and a diverse set of scaling experiments the authors have derived a first proof of power scaling laws showing validation of feasibility for the concept of sensor foundation models.

### Strengths
An original research work focused on a multimodal model with true large scale of data and model size that is comparable to language foundation models. The scaling laws validate proof of principle for this area. Sample applications chosen have practical significance.

The article is demonstrating high level of original research due to the scaling law contributions. They also gain additional validation by doing so with multi modal sensor streams. There have been other notable work showing the power of pre training on single or multiple sensor streams in the past at sufficient scale but nothing addressing the scaling question.

The results and analysis were presented with sufficient clarity. Some room for improvement has been suggested in the following section.

### Weaknesses
While the work is generally comprehensive, some areas of improvement remain. 

The limited choice of downstream tasks raises questions on the true foundational nature of the model and its few shot capability. It seems important to understand the ability to generalize to tasks beyond classification eg. regression tasks or system identification tasks.

The choice of generative evaluation goals while interesting fails to address the practical utility of the performance shown, especially with large errors in imputation. Also, there exist sophisticated imputation and interpolation schemes that are more performant comparators eg. MICE which might be the real performance benchmarks for the application. 

The definition of some of the features and preprocessing is incomplete and not in the language of mathematics. Some descriptions are highly redundant eg “Kurtosis is kurtosis”. Recommend use of the language of accurate mathematical description to define the features clearly.

### Questions
There is a lot of work that directly deal with accelerometer data in scale and recently also PPG data. The claim that these streams are hard to interpret and therefore not in scope - doesn’t seem a valid explanation. I do understand practical constraints in collecting these streams at a high enough sampling rate and so should suffice to explain the use of intermediates such as steps or heart rate.

The graphs with the scaling laws with the marker size and color simultaneously varying and with different y-axis scales is confusing. In fact, some marker size differences are small enough to not even stand out while accounting for a significant scale increase. Request to use different marker types instead of just size and make it more obvious, and also maintain y axis scale to compare more effectively.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This study scales a multimodal foundation model, LSM, across compute, data, and model size. Using a dataset of 40 million hours from 165,000 individuals, covering heart rate, heart rate variability, electrodermal activity, accelerometer, skin temperature, and altimeter data. The paper built a sensor foundation model and investigated strategies for training such models with considering data size, computation cost and model size.
The data processing steps are clearly explained and the the whole framework is tested on a variaty of tasks including generative and classification types of tasks.
The authors’ consideration of data processing and task diversity is a valuable contribution to the wearable and ubiquitous computing community, positioning this work as a significant step forward in multimodal sensor modelling. However, there are important areas for improvement, particularly regarding challenges specific to wearable computing and comparisons with state-of-the-art (SOTA) techniques.

### Strengths
- The data processing and model training steps are clearly outlined, and the study systematically scales the model across key factors such as data volume, computational resources, and model size.
- The paper includes a variety of tasks that effectively demonstrate the model’s utility across different contexts, enhancing the applicability of the LSM model in both generative and classification domains.

### Weaknesses
**Areas for Improvement**

- Background and Related Work: The paper is thin in covering prior works in multimodal wearable models. The chosen model (ViT) and baselines focus on non-wearable modalities, overlooking well-established approaches tailored to multimodal sensor data such as (but not limited to):

> [1] Saeed, A., Ungureanu, V. and Gfeller, B., 2021. Sense and learn: Self-supervision for omnipresent sensors. Machine Learning with Applications, 6, p.100152.

> [2] Deldari, S., Spathis, D., Malekzadeh, M., Kawsar, F., Salim, F.D. and Mathur, A., 2024, March. Crossl: Cross-modal self-supervised learning for time-series through latent masking. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining.

> [3] Haresamudram, H., Essa, I. and Plötz, T., 2021. Contrastive predictive coding for human activity recognition. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 5(2), pp.1-26.


- Dataset Clarity and Statistical Insights: The paper does not clearly specify the dataset utilized. The statement, “In this paper, we present the results of our scaling experiments on the largest and the most diverse wearable dataset published to date…” implies that the dataset is previously published, yet it lacks explicit details. The authors should clarify the dataset's origin and provide comprehensive information regarding its attributes. Clarification on the dataset’s availability and publication status is necessary.
Detailed statistical insights, including class distributions and sampling strategies, would enhance understanding of the dataset’s structure. This information would be particularly useful for evaluating how well the model might generalize across various activities and sensor modalities.

- Data Sampling and Task Appropriateness: The paper lacks clarity regarding the time window chosen for training and inference (e.g., It is infered that 5-hour windows is used for both generative and classification tasks). This is a critical point, as activities typically span only a few minutes, making it unclear whether such extended windows are suitable for classification. Details on data sampling strategies and the selection of window sizes should be provided to better assess model performance across tasks.

- Specific Challenges in Ubiquitous Computing: As a foundation model for wearable sensing, LSM should ideally address unique challenges in ubiquitous computing. For example:
  - the authors could discuss handling missing data at various stages (pre-training, fine-tuning, and inference), as highlighted by [2]. This would enhance the model’s robustness for real-world applications.
  - To demonstrate the model’s generalizability, it would be beneficial to extend experiments to include additional datasets covering diverse tasks, devices, and activities to analyse adaptability to a wider range of scenarios in wearable sensing.

### Questions
As mentioned above.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents a foundational model training for daily activity recognition from wrist worn portable devices. It uses a multimodal approach including 26 features out of four modalities (Acc, PPG, skin Temp  and conductance).
The model presents an improved performance compared to other methods in classification tasks of activities after user-based self-labelled classes (mostly on sports activities), as well as in interpolation and imputation tasks. 
The paper analyzes the model efficiency, and complexity needed by comparing on number of subjects, and samples needed.

### Strengths
1. Several tasks have been defined for multimodal model evaluation: imputation, interpolation
2. A robust set of activities and extensive dataset for training 165k people.
3. Comparative analysis across model parameters and data size through an ablation study.
4. Analysis of multiple tasks: Classification, interpolation, reputation and extrapolation.

### Weaknesses
1. 16 features extracted from PPG, Acceleration, Skin temperature and conductance, and altimetry - seems a very reduced space for learning. 
2. Temporal interpolation at the scale of 1 minute is not a very accurate task for wearable human data, as it only holds under strong assumptions of human behaviour. E.g. continuous activity, unchanged environment, no external inputs, among others.
3. It needed to be clarified the number of individuals used for training and the methods for data labelling. 
4. Baseline method comparison for imputation, interpolation and extrapolation was rather simple and did not included other advance methods, see 1.
[1] Maksims Kazijevs, Manar D. Samad. Deep imputation of missing values in time series health data: A review with benchmarking. (2023) Journal of Biomedical Informatics
5. The title is too broad and uninformative --> this model is trained for sports activity and proof given in that space only of the wrist wearable data.

### Questions
1. How was decided the feature space?
2. How many individual training samples per person were used? 
3. Did you use user-stratified splits? Or what is the data split method?
4. Why are the baseline methods for interpolation and imputation only Mean, NN and Linear?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This work proposed a foundation model aiming to serve as a general data encoder for wearable time series data. The author leverages signals capturing a variety of physiological activities for pre-training including activities of heart, skin, and motion. The model is evaluated on the tasks including time series imputation, forecasting, and human activity recognition. The author also shows that the scaling law also applied on large models for the modality of wearable signals.

### Strengths
Strengths:
1. The experiments are conducted on a large scale, and modeling a variety of signals including activities of heart, skin, and motion.
2. This work empirically shows that the scaling law of modeling can also be applied to the modality of wearable signals, in terms of scaling up computability, size of dataset, and size of model. 
3. The work includes reasonable baseline comparison with vision-based models.

### Weaknesses
Weaknesses:
1. The model has fixed shape input, which raises concern about generalization. For example, it is very common that different devices and scenarios have different sets of sensors with different sampling rate. Some wearables have PPG + inertial sensor data only. Some have other more sophisticated sensors like Galvanic Skin Response and Electrocardiogram. It’s unclear exactly how such a model would work with different sets of input modalities with diverse sampling and data resolution settings. 

2. It is not clear why modeling the time series wearable signal as an image only as opposed to other alternatives like spectral representations (fourier transformation, wavelet transformation). Wearable signals are essentially time series, and are more similar to audio modality. What is the intuition behind this design choice? Also, no comparison has been provided.
[I] Huang, Po-Yao, et al. "Masked autoencoders that listen." Advances in Neural Information Processing Systems 35 (2022): 28708-28720.
[II] Ansari, Abdul Fatir, et al. "Chronos: Learning the language of time series." arXiv preprint arXiv:2403.07815 (2024).

3. Insufficient description of modeling strategy and motivation with lack of model description. The paper can benefit from adding a more comprehensive comparison with general time series models like Chronos [2].

4. The key take away “the larger the model the better the performance” “the larger the dataset the better the performance” are fairly standard statements in the AI/ML area at this point. The paper can be improved by bringing in more insights with respect to the model size, scalability in different conditions (e.g., resource constraint/edge computing setting for example, or a distributed setting). Also, large offline dataset during training is hard to come by and often comes with significant privacy concerns. So when it comes to scalability, the computational resource and privacy are the main bottleneck. I think a bit more discussion around the key takeaways and the model’s scalability contributions would be important. 

5. Only a single real world downstream task of activity recognition has been explored in the current version of the work. While I highly appreciate the large-scale data collection efforts (that enabled the work) and the large computational model, only a single downstream task fails to highlight its capability. The paper can benefit from incorporating other real world mobile health applications such as stress/fatigue modeling with similar wearable data or may be other physiological conditions modeling.

6. Activity recognition is a well studied area. One additional task that could be interesting is whether such a model generalizes across unseen classes (e.g., dancing, cooking, and other activities for daily living). This could further highlight the capabilities of the model.

7. One of the core strengths of the paper is its large dataset. Making it available to the broader community could help research and development in this area. I could not find any mention about open sourcing the dataset. Also, no mention about open sourcing the model either. I understand that often there are IRB restrictions but I think making the model and data available could be a game changer in this area.

8. Relatively minor comment but there are some repetitive statements. The paper can be shrunk a bit and can be made it a bit more compact.

### Questions
Please refer to the weaknesses for detailed description of each points.

1. How the model would generalize to additional wearable activity tracking tasks? or event wearable health tasks?
2. Would the model and the representation generalize to unknown classes?
3. Can the authors provide more comparisons with time series models?
4. Describe the significance of the key take away of the paper. How it benefits the scalability of the model/approach in different settings.
5. Can the model be use for more diverse down stream tasks?
6. Clarify if you would open source the data and model?
7. Polish the write up and compress the text a little bit.

### Soundness
2

### Presentation
3

### Contribution
2
