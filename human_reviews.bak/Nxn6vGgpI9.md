# WEAR: An Outdoor Sports Dataset for Wearable and Egocentric Activity Recognition

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 6, 3

## Abstract
Though research has shown the complementarity of camera- and inertial-based data, datasets which offer both egocentric video and inertial-based sensor data remain scarce. In this paper, we introduce WEAR, an outdoor sports dataset for both vision- and inertial-based human activity recognition (HAR). The dataset comprises data from 18 participants performing a total of 18 different workout activities with untrimmed inertial (acceleration) and camera (egocentric video) data recorded at 10 different outside locations. Unlike previous egocentric datasets, WEAR provides a challenging prediction scenario marked by purposely introduced activity variations as well as an overall small information overlap across modalities. Benchmark results obtained using each modality separately show that each modality interestingly offers complementary strengths and weaknesses in their prediction performance. Further, in light of the recent success of temporal action localization models following the architecture design of the ActionFormer, we demonstrate their versatility by applying them in a plain fashion using vision, inertial and combined (vision + inertial) features as input. Results demonstrate both the applicability of vision-based temporal action localization models for inertial data and fusing both modalities by means of simple concatenation, with the combined approach (vision + inertial features) being able to produce the highest mean average precision and close-to-best F1-score. The dataset and code to reproduce experiments is publicly available via: https://www.anonymous.edu/anon

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduced a new outdoor sports dataset, namely WEAR, containing egocentric video and sensor data for human activity recognition (HAR), corresponding to vision and inertial-based tasks. There are 18 participants involved engaging in 18 different workout activities at 10 different locations. This dataset provides the potential to research the effectiveness of vision, inertial, and combined inputs on HAR. Besides, the authors also did tons of experiments as benchmark to show the strengths of each locality.

### Strengths
1. The tables, figures, and writing in the paper are clear and understandable.
2. The authors did a large number of experiments to show the features of each modality, i.e. vision and inertial, in HAR. Specifically, the vision-based temporal action localization, and Inertial-based wearable activity recognition, and multimodal HAR.

### Weaknesses
1. Compared to the vision-based action recognition datasets, the scale of WEAR is relatively small, with only 18 people operating 18 workout activities, and the total duration of each activity is 90 seconds, which does not show the minimum times the activity appears in this period.
2. One of the features of this dataset is that it is recorded in various conditions, such as location, and weather, with limited information overlapping. This proposed data collection method makes it really challenging, due to its variety and small number. This may make the model hard to extract enough meaningful information from this dataset.
3. The participants are young people, which may lead to bias. And there are multiple similar activities among these 18.

### Questions
The data collection used a head-mounted camera and smartwatches on the legs and arms. What is the motivation to collect data under these settings? Especially, there are few wearable products on legs in outdoor scenarios.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces WEAR dataset, an outdoor sports dataset for both vision- and inertial-based human activity recognition.

The dataset contains data from 18 participants performing a total of 18 different workout activities with untrimmed inertial (acceleration) and camera (egocentric video) data recorded at 10 different outside locations.

WEAR dataset provides a challenging prediction scenario marked by purposely introduced activity variations as well as an overall small information overlap across modalities.

### Strengths
+ The concepts are clearly introduced and explained, nice motivation.

+ Nice and interesting comparisons presented in Table 2.

+ Nice visualisations presented in the experimental section.

### Weaknesses
Major:

- The contributions highlighted in the Introduction section are a bit vague. What are the major contributions w.r.t. dataset collection, benchmarks and baselines etc? 

- The collected dataset seems still quite limited w.r.t. the number of classes and the number of performing subjects, which is a noteworthy shortcoming. The major differences compared to existing datasets are not clearly highlighted, or only limited discussions provided. More insights in terms of different aspects/perspectives/views should be added and discussed.

- The experiments w.r.t. benchmarks and baseline results are very limited w.r.t. different scenarios/tasks. This is also a noteworthy shortcoming.

- The evaluation metrics are also very limited. What are the other scenarios that this dataset can be used for? It is suggested to dig deeper into the problem rather than just propose a dataset and simply setup benchmarks and baselines.

- I noticed that the authors have a separate section talking about the limitations of the work; however, this section is not very well-written and it is suggested to improve it. For each limitation/weakness, it is suggested to provide detailed plans and solutions etc.

Minor:

- Page 1 spelling mistakes, e.g., timeseries ->  time series 

- Fig. 1 some fonts are too small to read.

- Left quotation marks require correction e.g., page 5 ‘dataset collection & structure’ section.

### Questions
Please refer to weakness section

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The manuscript introduces a new dataset called WEAR. This dataset contains ego-centric head-worn video data in combination with IMU data from each limb. The dataset is recorded by 18 participants. Each participant recorded 18 different activities. The data contains a total of 15h of this multi-modal data.

### Strengths
The dataset seems constructed well for a novel set of tasks in the fitness space. 

The recording of IMU signals from all limbs allows interesting experiments and ablations of which limb or modality yields the most signal on which activity. 

There is a wealth of baseline experiments of state of the art models that can use IMU or vision or both imu and vision data. This sets useful baseline performance on the dataset. 

The Oracle experiments convincingly show that there is quite a gap from the current performance to an "optimal" combination of the two modalities. This clearly shows there is more research to be done (even if the dataset at hand may be to small to actually help train models to close that gap).

The manuscript is well written and illustrated except for a few details (see weaknesses and questions).

### Weaknesses
The primary weakness of the dataset for ML is its relative small size of only 15h. (Compare that to 3670h of data in Ego4d). This makes it not well suited for training generalizable models solely on it (as evidenced by the need to pretrain on Kinetics 400). It may still be able to serve as a validation dataset for how well modalities can be combined on for fitness activity recognition.

Temporal alignment is critical for this kind of multi-sensor multi-modal data. The use of 4 jumps to get alignment signal is a bit questionable for alignment, given that IMU as running at 50Hz and camera at 60Hz. It will depend a lot on the implementation how good this alignment is. Unfortunately this details is not provided.

From Fig 2 it seems like the IMU-based models have low confusion, whereas the camera-based ones have more. Combining them does not seem to change much with confusion in general except for helping with the null-class. It kind of raises the question what vision is good for? 

The implicit assumption in the evaluation and dataset collection as presented is that in the future data will be available from both wrists and both feet which seems doubt full. Another interesting experiment would be to ablate which limb location makes the biggest difference in the action localization task. It would also be interesting to see the differences between confusion matrices between the different limbs. If only IMU data from one limb is used maybe the camera will also show a larger impact on performance.

### Questions
- whats cls in table 1?
- whats the number of hours collected for each dataset in table 1? 
- It is a bit unclear for the different models how they were trained on the relatively small dataset (12 participants data is only 10h of data which is very small for Transformer-based models) or whether they were pretrained on a different dataset before finetuning on the WEAR dataset. It does sound like only the vision based models were pretrained on the Kinetics 400 dataset. Otherwise all models were trained only on the WEAR dataset?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduce a new benchmark dataset, called WEAR, for human activity recognition using both inertial (acceleration) and camera (egocentric video) data. The dataset includes data recorded by 19 participants performing 18 different sports activities at 10 outdoor locations. The paper presents three benchmark tasks : inertial-based activity recognition, vision-based temporal action localization, and multimodal (inertial + video) activity recognition. Experiments show that recent action localization models (e.g., ActionFormer) can be extended for multimodal activity recognition and achieves superior results compared with single-modality or early fusion based methods.

### Strengths
The paper introduces a new dataset for outdoor physical activity recognition with multimodal (e.g., video and inertial) data, which is limited in the existing work. The dataset can be used for developing and evaluating inertial based or multimodal based models for physical activity recognition.

### Weaknesses
The scale of this dataset is too small for sufficient training of recent models and evaluation of model generalization. The dataset only involves 18 subjects (12 for training and 6 for testing), with only 15 hours in total (untrimmed) and roughly 8 hours for foreground activities. This data scale is too small to train a video model especially considering the increasing scale of the recent models, and it also makes the evaluation less convincing. The dataset is also less valuable for developing strong, generalizable multimodal activity recognition models due to the limited variation (only 18 activities with fine-grained differences, 10 locations and 18 subjects).

### Questions
N/A

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
