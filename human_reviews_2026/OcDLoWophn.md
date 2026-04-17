# Two-Stage Prototypical Networks Reveal Mosquito Flight Patterns

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Understanding behavioral movements in mosquitoes is fundamental for monitoring arbovirus transmission. Most existing Artificial Intelligence (AI) methods recognize tiny insects as background and fail to extract correct features from video frames. To address this issue, we propose a two-stage few-shot classification by Movement Density Map (MDM) prototyping called two-stage prototypical networks. A novel approach that integrates object detection with two-stage prototype training to analyze and identify mosquito behavior from videos. In the first stage, mosquitoes are detected using a fine-tuned YOLO, achieving a maximum mean Average Precision (mAP50) of 97.8%. The detected areas with eliminated backgrounds are then aggregated into MDMs. This mechanism enables encoding hundreds of frames into a single spatiotemporal representation that reveals biologically meaningful flight patterns over time. The MDMs are then mapped into a Vision Transformer (ViT) embedding environment, where class-level prototypes are generated for few-shot classification under 1 and 5 exposures using prototypical networks. Results on  datasets of dengue and Zika-carrier mosquitoes, as well as non-carrier ones, collected over 13 days and nights show that our approach significantly extracts more accurate features than a common single-stage prototypical network, leading to an overall performance accuracy of 85.86% . These findings reveal that two-stage prototyping is a reliable and scalable solution for analyzing tiny-object biological videos and holds promise for other spatiotemporal recognition tasks where motion aggregation is critical.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a novel two-stage prototypical network framework for few-shot classification of mosquito flight behaviors in videos, aiming to distinguish dengue and Zika virus-infected mosquitoes from non-infected ones. The core innovation lies in using Movement Density Maps (MDMs) to compress spatiotemporal information by first applying YOLO-based object detection to eliminate background noise, then generating MDM prototypes that summarize movement patterns.

### Strengths
1.	The integration of MDMs with two-stage prototyping is a significant contribution.
2.	Experiments are sufficient.
3.	This paper is well organized.

### Weaknesses
1.	The two-stage pipeline may be computationally heavy for resource-constrained settings. The paper omits discussion of inference time or efficiency, which is critical for deployment.
2.	I suggest add a unified pipeline diagram depicting: input videos → YOLO detection → background elimination → MDM aggregation → ViT embedding → prototype classification. 
3.	Limited Generalization Validation：Tests on the MosquitoFusion dataset use static images concatenated as pseudo-videos, which do not reflect real-flight dynamics. I remain skeptical about its robustness for real-world deployment.It is recommended to test on more real video datasets.

### Questions
please refer to weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a deep learning–based approach for classifying mosquito diseases from video data. The core intuition is that mosquito flight patterns, which can be recorded on video, may reveal anomalies related to their health condition. This motivates an end-to-end framework that processes raw videos and outputs health-status predictions.
Technically, the proposed pipeline is designed to mitigate potential bias and interference from the background. It begins with a detection phase, followed by background removal. The remaining pixels are then transformed into spatial Gaussian distributions that move across the image plane as video frames progress. This intermediate spatiotemporal representation is subsequently collapsed over time and fed into an image-based feature extractor.
The authors evaluate their method on mosquito datasets, demonstrating satisfactory results. In the ablation studies, they primarily investigate local design choices, such as the choice of optimizer and backbone network (with the ViT architecture yielding the best performance).

### Strengths
- The approach is technically sound and results look promising. Every block of the chain is well-justified and well-introduced. Probably, it is hard to think to a better way to handle this task.
- The paper is clearly written. 
- Relevance and impact: The problem addressed by the authors is of clear practical importance. Monitoring and diagnosing mosquito-borne diseases through automated visual analysis can have a tangible impact on public health, especially in areas where early detection of infected insects can prevent outbreaks of vector-borne illnesses. Beyond its immediate application, the proposed pipeline also demonstrates how computer vision and deep learning can be applied to entomology and epidemiology, potentially inspiring further interdisciplinary research.

### Weaknesses
- Contributions. The work is primarily an application of existing deep learning techniques to a specific real-world problem. While the pipeline is well-engineered, it does not introduce novel methodological or theoretical contributions that could be generalized or transferred to other domains.he techniques employed (detection, background removal, and the use of deep feature extractors) are well established in the literature, and the paper mainly demonstrates their effectiveness in a new context rather than proposing new insights or innovations. For this reason, while the paper may be very well suited for applied or interdisciplinary journals focusing on AI for health, environmental monitoring, or bioinformatics, it may be less aligned with the expectations of top-tier machine learning conferences like ICLR, where novelty and theoretical contribution typically play a central role in the evaluation process.

- 4.2 THE SECOND STAGE OF PROTOTYPING. This section was somewhat unclear. It is not entirely clear how the training process is carried out. For instance, whether it involves a standard gradient-based optimization procedure, whether the model starts from a pretrained checkpoint, and whether, during a single training step, the model has access to multiple samples from the same class. This part of the paper would benefit from additional clarification and polishing to make the training protocol more understandable.

- Finally, the ablation studies are not particularly insightful. They mainly focus on comparing different architectural components or optimizers, rather than investigating the core design choices of the proposed pipeline. It would be more informative to analyze aspects more directly tied to the method itself — for example, the use of Gaussian representations, or the assumption that temporal information can be aggregated through simple averaging. Such analyses could be complemented by comparisons against models explicitly designed to handle spatiotemporal features, such as LSTMs, Transformers, or non-local blocks.

**Minor**
- Figure 4 and Figure 5 are difficult to read due to the very small font size; improving their readability would significantly enhance the overall presentation quality.

### Questions
No questions.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a two-stage prototypical network framework that integrates YOLO-based mosquito detection with movement density map (MDM) aggregation to reveal spatiotemporal flight patterns and improve few-shot behavioral classification accuracy.

### Strengths
The visualizations—particularly Figure 2—are highly insightful, effectively demonstrating how MDMs can capture meaningful spatiotemporal motion cues for tiny-object classification.

The model achieves better performance than existing benchmarks, indicating the potential of MDM-based representations for biological video analysis.

### Weaknesses
The description of the MDM model and overall problem setup could be clearer and more detailed.


The MosquitoFusion dataset seems to have been introduced in a prior short paper; however, more explanation about the labeling procedure and specifics of the 1-shot and 5-shot experiments should be included for reproducibility.


The technical novelty is somewhat limited, alternative formulations, such as grid-based spatial encoding, might enhance classification performance.


Comparisons with related methods that jointly model appearance and motion (e.g., ClusterNet by LaLonde et al., and Multiple Object Tracking with Motion and Appearance Cues by Li et al.) would strengthen the evaluation.



The ablation study is too minimal—rather than simple layer or optimizer variations, evaluating different methods or design choices for each of the two stages would yield deeper insights.

### Questions
I wonder if an end-to-end integration of detection and classification stages could provide a more robust and generalizable framework?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes a method to detect the flight patterns of mosquitoes. The approach begins by utilizing an object detector (YOLO) to identify mosquitoes. Then, a density map is created by averaging the detected coordinates along the temporal dimension. This information is subsequently fed into a Vision Transformer (ViT) to generate embeddings, which are then passed to a prototype network. While the topic is indeed interesting, there are several concerns worth addressing, especially regarding novelty and experimental results:

### Strengths
1) An interesting application of ML/CV for studying mosquito flight patterns

### Weaknesses
1) What is the motivation for using a few-shot learning technique? Although I understand the advantages of few-shot learning over conventional supervised learning paradigms, it would be helpful to clarify any additional motivations for focusing on few-shot learning. This could be discussed in the introduction.

2) What are the variables in Equations 1 and 2? Understanding the proposed approach is challenging without this crucial information.

3) The novelty of the work appears to be limited. The processes of creating the density map and generating embeddings are standard practices. It would be beneficial for the work to discuss the necessity of these steps for the specific task presented.

4) The experimental results are weak. There are no standard SOTA methods discussed and compared in this work. I would suggest exploring methods from crowd density estimation for a comparison? There are some similarities—such as scale imbalance and complex backgrounds—between the two studies that warrant investigation.

5) Which two classes were used for training, and which were used for validation and testing? Additionally, what was the configuration for the 1-shot and 5-shot modes?

### Questions
Please refer to my comments above especially regarding novelty and experiments.

### Soundness
2

### Presentation
2

### Contribution
1
