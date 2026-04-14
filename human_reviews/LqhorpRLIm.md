# What Matters in Learning from Large-Scale Datasets for Robot Manipulation

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Imitation learning from large multi-task demonstration datasets has emerged as a promising path for building generally-capable robots. As a result, 1000s of hours have been spent on building such large-scale datasets around the globe. Despite the continuous growth of such efforts, we still lack a systematic understanding of what data should be collected to improve the utility of a robotics dataset and facilitate downstream policy learning. In this work, we conduct a large-scale dataset composition study to answer this question. We develop a data generation framework to procedurally emulate common sources of diversity in existing datasets (such as sensor placements and object types and arrangements), and use it to generate large-scale robot datasets with controlled compositions, enabling a suite of dataset composition studies that would be prohibitively expensive in the real world. We focus on two practical settings: (1) what types of diversity should be emphasized when future researchers collect large-scale datasets for robotics, and (2) how should current practitioners retrieve relevant demonstrations from existing datasets to maximize downstream policy performance on tasks of interest. Our study yields several critical insights -- for example, we find that camera poses and spatial arrangements are crucial dimensions for both diversity in collection and alignment in retrieval. In real-world robot learning settings, we find that not only do our insights from simulation carry over, but our retrieval strategies on existing datasets such as DROID allow us to consistently outperform existing training strategies by up to 70\%.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper explores the key factors that influence the success of robot manipulation policies trained on large-scale datasets. It introduces a synthetic data generator, MimicLabs, to emulate common variations in real-world datasets such as camera poses and object arrangements. The study focuses on identifying the types of diversity that should be emphasized when collecting new datasets or retrieving relevant demonstrations from existing ones to maximize downstream task performance. Using simulation and real-world tasks, the authors demonstrate that factors like camera pose and spatial arrangement are critical to both dataset diversity and alignment. These insights are validated through experiments on the DROID dataset, leading to improvements in policy learning by up to 70%.

### Strengths
1. The paper tackles an important problem—optimizing large-scale datasets for robotic manipulation—by systematically analyzing how various dimensions of variation (DVs) in the data affect learning. It provides a novel approach through its MimicLabs synthetic data generator, which offers a controlled environment for testing different dataset compositions, addressing a key gap in robot learning research.
2. This work has the potential to significantly impact the future of large-scale robot dataset collection and utilization. By offering actionable insights on how to prioritize data collection and improve policy learning through targeted dataset retrieval, it lays the foundation for more efficient robot learning pipelines, reducing the cost and complexity of data collection efforts.

### Weaknesses
1. Some conclusions are well-known: The paper emphasizes that “camera poses and spatial arrangements are crucial dimensions for both diversity in collection and alignment in retrieval,” which is somewhat redundant with existing research. Camera pose, in particular, has long been recognized as a challenging factor to generalize. The generalization issues of spatial arrangements are also quite intuitive since they are linked to training distribution. If the training distribution is narrow, out-of-distribution (OOD) failure at test time is expected.
2. Dataset Dependency: The experiments and conclusions are closely tied to the DROID dataset, raising concerns about their generalizability to other robotic platforms or datasets. More varied datasets should be included in future experiments to ensure the robustness of the insights across different domains.
3. Difficulty in real-world implementation of “Dimensions of Variation” (DVs): While the paper proposes the concept of DVs, it might be challenging to apply this approach in real-world scenarios where it is impractical to enumerate all possible variations. For example, factors such as object material or lighting conditions are not sufficiently addressed. Furthermore, real-world robot performance is often contingent on handling the complex interplay of multiple variations simultaneously rather than generalizing to each factor individually.

### Questions
1. Have you tried using only wrist cameras? One question that could have been explored more thoroughly is whether using dynamic wrist cameras, which vary poses constantly, would act as a strong form of data augmentation and mitigate the problem of overfitting to fixed camera poses. If this technique were used, it could potentially reduce sensitivity to specific camera poses and improve generalization.
2. Generalization beyond single variation dimensions: The current analysis focuses on individual DVs (like camera pose or object arrangement), but in the real world, robots often face scenarios where multiple variations occur simultaneously. Have you considered experiments that explore the interaction effects between multiple DVs, and if so, how do the insights change?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work conducts a thorough investigation on creating and utilizing robotics dataset for co-training robot policy with imitation learning. It demonstrates the most useful types of diversity for dataset collection and proposes insightful yet effective retrieval methods from existing datasets to improve task performance.

### Strengths
1. This work tackles a popular yet significant problem concerning usage of large-scale robotics dataset.
2. The authors conduct extensive experiments on collector and retriever perspective with various DVs.
3. The paper is well-written and easy to follow.

### Weaknesses
1. The experiments in Section 5.1 only consider one task *clear table*, making the conclusions less convincing. Results on more simulation tasks or clarification on the representativeness of this task should be added.
2. The co-train setting in this paper largely focuses on using generated/retrieved data from the same task as the test task (except for some results in Table 3). However, aligning the task setting between the target task and the tasks in a public large-scale dataset is difficult in most cases, especially for real-world tasks.  More experiments and analysis on co-training with different tasks can add to the practicality of this work.
3. Some of the results may be confusing. Please refer to the Question section.

### Questions
1. In Table 1, it can be confusing why training and testing on only target distribution will yield such poor results, since overfitting to a rather small range of variation seems not to be that hard in imitation learning.
2. In Table 1, I am confused why co-training with varied camera poses can improve policy performance to such a great extent, because there are only several (~5) poses and they are very different. Maybe some videos and extra explanations can help clarification.
3. In Table 2 "Object spatial" part, why larger co-training distribution produces lower performance?
4. In Figure 5, I am curious whether the improved performance is attributed to the co-training setting or alignment between target and training task. For example, how is the performance if the policy is trained with retrieved Droid data without target data?

### Soundness
2

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
This paper presents a systematic study of what factors matter most in large-scale robotics datasets for manipulation tasks. The authors introduce MimicLabs, a framework for generating controlled datasets with various dimensions of variation (DVs), and analyze dataset composition from both collector and retriever perspectives. They identify camera poses and spatial arrangements as crucial factors while finding object textures less important. The study's insights are validated on real robots and existing datasets like DROID.

### Strengths
The paper addresses the crucial question of dataset composition in large-scale robotic manipulation through a systematic analysis. The two-perspective approach (collector and retriever) offers practical insights for both dataset collection and utilization. The authors conduct comprehensive experiments and clear validation in both simulation and real-world settings. The definitions and methodology based on "DV" are novel.

### Weaknesses
1. Results heavily rely on MimicGen's simplified environments with basic textures and lighting. The conclusion that "texture alignment is less important" may not hold in real-world scenarios like those in DROID with complex lighting, shadows, and material properties (as in computer vision, lighting effects are often considered as a special kind of texture). Limited scene diversity compared to real datasets like DROID (8 simulated scenes vs. 52 real buildings)

2. Limited Policy Evaluation: Experiments primarily use Diffusion Policy, with only BC-RNN as comparison in MimicLabs. Performance gains might be attributed to Diffusion Policy's characteristics rather than dataset composition. Missing evaluation with modern approaches like VLA or transformer-based policies.

3. No clear metric for measuring skill similarity (e.g., is "picking a bowl" similar to "picking a cup"?) Lack of systematic handling of compound skills. Binary treatment of skill alignment without considering partial similarity.

4. Overlooked DV Interactions: No consideration of cases where a tejectory has conflicting values across different DVs (e.g., good camera pose but poor spatial arrangement). Missing analysis of trade-offs between different DVs when retrieving demonstrations. No principled approach for weighing the relative importance of different DVs when they conflict.

The limitations affect the generalizability and practical applicability of the paper's conclusions, particularly when scaling to more complex real-world scenarios or different learning algorithms.

### Questions
1. How do you handle cases where a demonstration might be valuable according to one DV but less useful according to another? Is there a principled way to make such trade-offs?
2. How do your findings generalize across different imitation learning approaches? Would VLA or other transformer-based policies show similar sensitivity to DVs?
3. How does your conclusion about texture alignment being less important translate to real-world scenarios with complex lighting conditions?
4. Given that similar objects often share manipulation strategies, why not consider 'picking a cup' and 'picking a bowl' as the same skill? Would grouping such similar skills during retrieval improve policy performance or potentially harm it?
5. Since DROID contains diverse task variations across 52 buildings while your simulation covers 8 scenes, could the limited scene diversity affect your DV importance rankings? Specifically, would more diverse scenes reveal additional important DVs?

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
5

### Summary
This paper conducts a comprehensive empirical study on key factors in scaling robot manipulation datasets, offering valuable insights for both collectors and users.

### Strengths
1. This paper collects a large-scale robot manipulation dataset in simulation with diverse dimensions of variations to enhance understanding of each factor’s impact.

2. This paper provides practical suggestions on how to collect and leverage the data, and conducts real-world experiments on the DROID dataset to verify its effectiveness.

### Weaknesses
1.	In section 5.1, from the collector’s perspective, there’s only one task, “clear table”, which seems insufficient to draw conclusions (especially given that more tasks are actually used in the following sections).

2.	As illustrated in the paper, object geometry is vital to robot data. However, it is a pity that it is not included in the DVs of the experiments.

3.	Although the MimicLabs dataset generates different textures for the experiments, the textures are still primarily pure colors according to the appendix. More complex textures, such as plaid patterns or irregular textures, aren’t included.

4. There are a large number of experiments and various settings in this paper, but some of them are not explained clearly, which might be confusing (see the questions for details).

Overall, this paper conducts a nice trial to investigate the key factors in large-scale robot learning and provides many useful results and practical suggestions. However, more thorough experiments and more accurate explanations of experiments would make the conclusions more convincing.

### Questions
1.	How many tasks are included in the MimicLabs dataset? Different sections use various tasks from the dataset, but I am unsure how many tasks are there in total. According to the appendix, there seem to be five, but this is not shown in the main text or images. 

2.	What exactly are the four co-training settings in Figure 2? There seems to be no exact definition of these settings, and different experiments actually use different numbers/types of co-training, making it harder to understand. It would be helpful to have a better description of all co-training settings. 

3.	Why are all the co-training datasets in section 5.1 misaligned with the target? Can’t it contain the target distribution (as I saw in some experiments in other sections)? 

4.	In section 5.3, the authors have an important finding that full retrieval might be less effective than other strategies using less data. However, a more common strategy used in other areas is to first pretrain on the full dataset and then fine-tune on the target dataset. I am wondering have the authors tried this setting?

### Soundness
4

### Presentation
2

### Contribution
3
