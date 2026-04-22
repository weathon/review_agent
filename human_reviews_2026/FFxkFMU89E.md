# EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 6

## Abstract
Imitation learning for manipulation has a well-known data scarcity problem. Unlike natural language and 2D computer vision, there is no Internet-scale corpus of data for dexterous manipulation. One appealing option is egocentric human video, a passively scalable data source. However, existing large-scale datasets such as Ego4D do not have native hand pose annotations and do not focus on object manipulation. To this end, we use Apple Vision Pro to collect EgoDex: the largest and most diverse dataset of dexterous human manipulation to date. EgoDex has 829 hours of egocentric video with paired 3D hand and finger tracking data collected at the time of recording, where multiple calibrated cameras and on-device SLAM can be used to precisely track the pose of every joint of each hand. The dataset covers a wide range of diverse manipulation behaviors with everyday household objects in 194 different tabletop tasks ranging from tying shoelaces to folding laundry. Furthermore, we train and systematically evaluate imitation learning policies for hand trajectory prediction on the dataset, introducing metrics and benchmarks for measuring progress in this increasingly important area. By releasing this large-scale dataset, we hope to push the frontier of robotics, computer vision, and foundation models. EgoDex is publicly available for download at https://github.com/apple/ml-egodex.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a large-scale egocentric human video dataset that captures diverse manipulation behaviors in everyday scenarios, annotated with 3D hand poses and tracking information. To demonstrate the dataset’s utility for robot manipulation learning, an imitation learning policy is trained to predict hand trajectories, and a set of benchmark tasks and evaluation metrics are introduced.

### Strengths
Leveraging human videos to learn dexterous manipulation policies is a promising direction, especially since collecting teleoperation data is often costly and time-consuming. 

The authors’ plan to open-source the dataset will further benefit the research community and foster future developments in this area.

### Weaknesses
The evaluation is limited, focusing solely on the trajectory prediction task. 

The paper lacks experiments demonstrating that progress on trajectory prediction tasks can effectively transfer to real-world robotic manipulation. Additional empirical studies validating this transferability would further strengthen the work and better justify the practical usefulness of the proposed dataset.  

Moreover, based on the provided sample data, the captured human behaviors appear somewhat unnatural---for instance, frequent use of pinch grasps---and the backgrounds are relatively clean and uncluttered, which may limit the dataset’s realism and representativeness of everyday human activities.

### Questions
What are the failure modes of the learned models? Could the authors provide more qualitative results to illustrate these cases? 

Beyond the trajectory prediction task, do the authors plan to include additional evaluation tasks to further assess the dataset’s utility?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces EgoDex, a new large-scale egocentric video dataset focused on human dexterous manipulation. Comprising over 300,000 episodes, 90 million frames, and 800 hours of video across 200 tasks and 500 objects, the dataset is meticulously labeled with 3D hand skeleton annotations and calibrated camera parameters. The data collection focused on spontaneous, natural human actions rather than scripted or unnatural movements. This resource is intended to foster advancements in video understanding, embodied AI, and robot manipulation by providing a rich, high-quality foundation for learning dexterity.

### Strengths
1. **Largest Scale and Rich Annotation**: EgoDex represents the largest-scale egocentric video dataset of human hand manipulation to date. The inclusion of precise, quantified camera parameters and 3D hand pose labels is a significant contribution, providing an invaluable resource for both video understanding and the development of embodied agents.

2. **Uniquely Naturalistic Data**: The data is collected from spontaneous, active human execution rather than unnatural, deliberately posed or slow-moving scripted actions. This naturalistic quality gives the dataset a unique advantage and is crucial for training robust models that generalize to real-world human behavior.

3. **Novel and Engaging Task Formulation**: The proposed task of predicting human hand trajectories from an egocentric viewpoint is highly intriguing. This task is both academically interesting and potentially beneficial for mutual promotion and integration with robotic motion prediction and control tasks in the future.

### Weaknesses
1. **Limited Auxiliary Sensor Data**: The current collection is primarily focused on RGB video. The study could have been significantly enhanced by incorporating additional hardware information during data collection, such as depth maps or synchronized foreground/background segmentation masks. This supplementary data would facilitate lifting observations to 3D and enable subsequent researchers to confidently develop tasks involving procedural background randomization/generation.

2. **Insufficient Detail on Data Collection Protocols**: The manuscript lacks rigor in specifying certain critical aspects of the data collection procedure. Key missing details include the exact number of subjects involved, the strictness of the policy regarding the hand always remaining within the field of view, and a statistical analysis of the duration distribution of tasks. Furthermore, the handling of noisy or erroneous keypoint predictions (e.g., jitter or catastrophic failure) is not discussed.

3. **Inadequate Downstream Task Validation**: The true value of a dataset lies in its utility for downstream tasks. Despite mentioning the positive impact EgoDex could have on robotic manipulation, affordance learning, or video generation, the research does not provide sufficiently solid experimental evidence to prove these claims. For a top-tier conference like ICLR, this lack of rigorous downstream validation significantly weakens the overall persuasiveness of the work.

### Questions
1. **Collection Rigor**: Provide exact details on the number of subjects and the protocol for handling and filtering noisy or failed 3D hand pose annotations in the dataset.

2. **Depth/3D Utility**: Conduct a small-scale pilot study demonstrating the utility of the dataset for a 3D-related task (e.g., 3D lifting or affordance modeling) to justify the investment in 3D labels.

3. **Robotics Validation**: Present a clear, quantitative baseline experiment on a robotic manipulation task (e.g., using a SOTA method fine-tuned on EgoDex) to solidify the dataset's direct value to the robotics community.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a large-scale egocentric manipulation video dataset. It includes comprehensive annotations and is more suitable for robotics task. The data is used to train and evaluate imitation learning policies.

### Strengths
1. The paper is well-written and well-organized. 
2. The proposed dataset is very useful. 
3. The dataset statistics are extensively provided.

### Weaknesses
1. Some more pilot experiments should be provided, as mentioned in Section 6. It would be great to see that the downstream experiments are *actually* deployed, not just discussed, to prove the significance of the dataset.  
2. More baseline or backbone networks should also be evaluated in the benchmark experiment. 
3. Two recent works [1, 2] should be discussed and compared, especially [1].

*Refs*:  
[1] MEgoHand: Multimodal Egocentric Hand-Object Interaction Motion Generation. Zhou et al.  
[2] TASTE-Rob: Advancing Video Generation of Task-Oriented Hand-Object Interaction for Generalizable Robotic Manipulation. Zhao et al.  CVPR 2025.

### Questions
The reviewer believes that some more pilot studies will strengthen the paper's significance, but may not influence its rating to be accepted.

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
5

### Summary
The paper introduces EgoDex, a massive egocentric dataset for dexterous manipulation: ~829 hours, 90M frames, 338k demonstrations, covering ~194 tabletop tasks, recorded at 30 FPS/1080p with paired 3D head/upper-body/hand (25 joints/hand) poses via Apple Vision Pro (with on-device SLAM + calibrated cameras to improve the estimation accuracy). The authors also define two benchmarks—trajectory prediction and inverse dynamics—with a best-of-K evaluation, and report baselines across encoder–decoder vs. decoder-only and BC/DDPM/Flow-Matching models, plus ablations on horizon, goal conditioning, model/dataset size. Visual goal conditioning notably improves final error (−53%), and performance scales with more data. A fixed held-out test split is provided for reproducibility. Lastly, the authors promise to open source all data in the future.

### Strengths
The strengths lie in the benchmark design and the extensive effort to collect a well-calibrated dataset. Specifically, they are

- Big and diverse. Way larger than prior human/robot sets, with language, camera extrinsics, and dense dexterity labels—covering ~200 tasks and ~500 objects.

- Clean paired signals. Synced ego RGB + full 3D skeleton (wrists/fingertips/head/arms) at 30 Hz, which is much cleaner than post-hoc hand pose from internet videos.

- Benchmark-ready. Two well-defined tasks (trajectory prediction, inverse dynamics) with a fixed held-out test split, so results are easy to compare apples-to-apples.

- Useful takeaways. The experiments surface a few clear patterns—for example, goal conditioning helps consistently.

### Weaknesses
The weaknesses include:

- Benchmark scope is narrow. The evaluation focuses on human motion prediction only, without assessing the robot side (e.g., retargeting quality and policy performance). Human trajectory prediction is only an intermediate signal; good imitation error does not necessarily translate to task success—this disconnect has been noted before [1], and that’s based on robot data—all the more so for human trajectory prediction as the retargeting might also amplify the error. A more direct metric like robot task success rate (in sim or on hardware) would better capture end-to-end utility. I agree trajectory metrics are reproducible, but they’re not decisive for manipulation.

[1]. Mandlekar, Ajay, et al. "What matters in learning from offline human demonstrations for robot manipulation." arXiv preprint arXiv:2108.03298 (2021).

- Practicality of goal images. Minor but important: how is the goal image obtained at inference? If it’s required, please discuss practical acquisition (e.g., user-provided snapshot, retrieval from a library, last-frame of a setup video) and report results (or at least ideas and any preliminary explorations) with vs. without a goal image to gauge real-world usability during inference.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
