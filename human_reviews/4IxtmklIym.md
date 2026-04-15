# FruitBin: A tunable large-scale dataset for advancing 6D Pose estimation in fruit bin picking automation

- Decision: Reject
- Scores: 5, 8, 3, 3

## Abstract
Bin picking is a ubiquitous application spanning across diverse industries, demanding automated solutions facilitated by robots. These automation systems hinge upon intricate components, including object instance-level segmentation and 6D pose estimation, which are pivotal for predicting future grasping and manipulation success. Contemporary computer vision approaches predominantly rely on deep learning methodologies and necessitate access to extensive instance-level datasets. However, prevailing datasets and benchmarks tend to be confined to oversimplified scenarios, such as those with singular objects on tables or low levels of object clustering. In this research, we introduce FruitBin. It emerges as an unparalleled resource, boasting an extensive collection of over a million images and 40 million instance-level 6D poses. Additionally FruitBin differs with other datasets whith its inclusive representation of a wide spectrum of challenges, encompassing symmetric and asymmetric fruits, objects with and without discernible texture, and diverse lighting conditions, all enriched with extended annotations and metadata. Leveraging the inherent challenges and the sheer scale of FruitBin, we highlight its potential as a versatile benchmarking tool that can be customized to suit various evaluation scenarios. As a demonstration of this adaptability, we have created two distinct types of benchmarks: one centered on novel scene generalization and another focusing on novel camera viewpoint generalization. Both benchmark types offer four levels of occlusion to facilitate the study of occlusion robustness. Notably, our study showcases the difficulty of FruitBin dataset, with two baseline 6D pose estimation models, one utilizing RGB images and the other RGB-D data, across these eight distinct benchmarks. FruitBin emerges as a pioneering dataset distinguishing itself by seamlessly integrating with robotic software. That enable direct testing of trained models in dynamic grasping tasks for the purpose of robot learning. Samples of the dataset with its associated code are provided in the supplementary materials. FruitBin promises to be a catalyst for advancing the field of robotics and automation, providing researchers and practitioners with a comprehensive resource to push the boundaries of 6D pose estimation in the context of fruit bin picking and beyond.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper introduces a large-scale PickSim-based synthetic dataset FruitBin for 6D object pose estimation in fruit bin picking. The dataset features comprehensive challenges and devised benchmarks for scene and camera view generalization as well as occlusion.

### Strengths
This is the first 6D object pose estimation dataset tailored for fruit bin picking although it is synthetic.

### Weaknesses
-- One drawback of Gazebo is that it can not do photorealistic rendering for objects and scenes with PBR textures. Although the generated dataset is large, without photorealistic textures, the transfer ability to real world is limited compared with other simulators such as BlenderProc and Kubric even the domain randomization techniques have been leveraged.

-- For real-world fruits, the size and shape of different instances of the same category vary to different degrees. However, it seems for FruitBin, these factors are not taken into consideration.

-- There is no real test set for the dataset, which is essential for sim2real and real-world applications.

-- The benchmarking methods are a bit outdated. PVNet and DenseFusion are from 2018-2019, but it is 2023 now. 

-- It would be better to showcase some robotic applications like bin picking using this dataset, since it is targeted for fruit bin picking.

-- It would be better to mark symmetric objects with "*" in Table 2.

-- Table 1 is too wide. 

-- There are some minor issues in the writing, more thorough proofreading is required.

### Questions
1) In the experiments, does PVNet use GT bboxes for cropping the objects in order to handle multiple instances of the same object class?

2) How does the diffusion generated backgrounds contribute to the performance?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
* This paper tackles novel research direction of fruits (or generalized any grocery item) using a robo-arm. 
* Dataset uses RGB and depth cameras for curating and annotatings the dataset.

### Strengths
* This industry really needs a good dataset to further explore the problem, this paper just targeted that. 
* This paper generalizes scenes as well as camera position for wider acceptability of it. 
* Good reference to prior work on datasets.

### Weaknesses
* I would have preferred to see even more robust baselines.

### Questions
NA

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents FruitBin, a 6D pose estimation dataset for fruit bin picking with benchmarking over scene generalization, camera generalization and occlusion robustness. It contains over a million images and 40 million instance-level 6D poses.

### Strengths
- This paper proposes a large-scale dataset, which may facilitate future research for bin-picking tasks.
- The technical details are clearly presented.

### Weaknesses
- Limited Contribution
    - It seems that the technical contributions of this paper is just replacing the assets in PickSim with fruits. I don't think this contribution is sufficient for an ICLR paper.
    - All the data are collected in the simulator. It seems that no data is collected in the real world.
- Inconvient Platform
    - This paper uses ROS+Gazebo as its simulator platform, and claims it's for "seamless robot learning". However, I would think mujoco, PyBullet, or Isaac Gym are some more popular options in the robot learning community.
- Format Issues
    - Table 1 and the references are with format issues.
    - The supplementary materials should not be attached to the main paper.

### Questions
- Will the dataset include more samples collected in the real world?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces a novel and extensive dataset designed for the task of fruit bin picking. This dataset is entirely synthetic and comprises 3D meshes of eight distinct fruits arranged in randomized configurations within bins, with varying lighting conditions and camera perspectives. The research employs this dataset to train two distinct models, one utilizing RGB data and the other incorporating RGB-D information, to serve as exemplary methods for 6-DOF pose estimation.

### Strengths
1. The paper is well-written and easy to understand. It explains its ideas clearly, making it accessible to a broad audience.
2. The dataset is extensive regarding images, configurations, and annotations. 
3. The paper also offers detailed insights into the dataset, providing readers with a comprehensive understanding of its composition. This helps other researchers in utilizing the dataset effectively.
4. The synthetic nature of the dataset allows for the extraction of highly detailed annotations, which can be challenging to obtain in real-world scenarios.

### Weaknesses
1. One primary concern regarding the paper pertains to its real-world applicability. While the synthetic dataset's ability to provide detailed annotations is a strength, it also raises questions about the practical utility of algorithms trained on it in real-world scenarios. The paper should delve into the broader implications and limitations of applying such models to real-world fruit-picking scenarios.

2. A related concern is the limited variety of objects in the dataset. With only 8 types of fruits, and a significant majority of them being spherical (75%), the need for 6DOF pose estimation for these objects may be questionable. The paper should address the relevance of 6DOF pose estimation for objects that might not require such detailed positioning information.

3. The paper should explore the broader question of whether 6DOF pose estimation is necessary for fruit picking, particularly when considering that many real-world fruit-picking applications rely on suction grippers, making pose estimation less critical.

4. It is important to clarify the specific scenarios that the dataset targets. Random mixing of different fruits in bins may not represent common real-world scenarios, where fruits are typically harvested in monocultures and packed separately. The paper should outline the dataset's intended use cases and their alignment with real-world applications.

5. While the paper claims diversity in the dataset, I would argue that diversity should be measured by the variety of objects rather than the sheer number of images and annotations. The paper should address these concerns and clarify how the dataset's diversity aligns with its practical usefulness.

6. In my opinion, the representative images in the paper all look similar, and the lighting variations are synthetic without showing real-world visual phenomena (shadows, reflection). The paper should discuss how these factors affect the dataset's applicability to real-world scenarios and consider potential improvements.

7. Some of the language choices throughout the paper, such as the use of "comprehensive" to describe the evaluation using two models, are overly grandiose, in my opinion. The paper should adopt more precise and measured language to accurately represent the extent of the evaluation and avoid overstating its findings.

8. As this dataset targets robotic grasping of fruits, I would have liked to see a comparison of using the dataset on 6DOF grasping with a robotic gripper, not only pose estimation.

### Questions
1. In the intro, the paper mentions that the dataset contains delicate fruits like bananas and apricots that require haptic feedback for grasping, yet it is not mentioned how this is modeled and incorporated in the benchmark. Is this only in reference to exact pose estimation?
2. In Table 1. how does the presented dataset compare to other 6DOF datasets regarding object diversity?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
