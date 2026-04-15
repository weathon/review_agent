# Sensor-Invariant Tactile Representation

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
High-resolution tactile sensors have become critical for embodied perception and robotic manipulation. 
However, a key challenge in the field is the lack of transferability between sensors due to design and manufacturing variations, which result in significant differences in tactile signals. 
This limitation hinders the ability to transfer models or knowledge learned from one sensor to another. 
To address this, we introduce a novel method for extracting Sensor-Invariant Tactile Representations (SITR), enabling zero-shot transfer across optical tactile sensors. 
Our approach utilizes a transformer-based architecture trained on a diverse dataset of simulated sensor designs, allowing it to generalize to new sensors in the real world with minimal calibration. 
Experimental results demonstrate the method’s effectiveness across various tactile sensing applications, facilitating data and model transferability for future advancements in the field.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work proposes a sensor-invariant tactile representation method that learns cross-sensor representations, enabling zero-shot generalization to new tactile sensors. To achieve this, the authors collect a large set of simulated tactile images paired with human-annotated images, where sensor contacts with the same objects in same pose are matched for positive pairs. Using a small amount of tactile data for downstream tasks, they fine-tune a task-specific head while keeping the pre-trained encoder frozen. They also demonstrate zero-shot generalization across unseen tactile sensors. The key contributions include: (i) an open-source large-scale dataset, (ii) zero-shot generalization to novel sensors, enabled by the model design, and (iii) a well-written, accessible paper.

### Strengths
The strengths of this paper are:

Originality: The problem is novel and relevant to practical applications of tactile-based learning.

Quality: he proposed method is well-founded, built upon established principles in the field, and presents a coherent theoretical framework. The experimental results, while only partially demonstrating the method's effectiveness, provide a solid initial validation of the approach. These findings indicate the potential for further development and optimization, suggesting that with additional refinement, the method could yield even more impactful results. However, a more comprehensive evaluation, including a broader range of experiments, would strengthen the evidence for the method’s effectiveness and applicability.

Clarify: The writing is clear and easy to follow.

Significance: Achieving zero-shot generalization to new tactile sensors is a valuable contribution to the field, addressing a critical challenge in tactile-based learning and manipulation. This capability has the potential to greatly expand the applicability of tactile sensing technologies across diverse robotic systems and tasks. By demonstrating that a model can effectively generalize to previously unseen sensors, the authors open up new avenues for research and development, ultimately advancing the state of the art in robotics and sensory integration. However, the tasks considered in this work is a bit limited.

### Weaknesses
There are several concerns:

Major:

1). The authors highlight the high cost of data collection, but recent studies (e.g., [1], [2]) have developed low-cost tools for gathering data across various manipulation tasks. I would like to see the authors' perspective on whether cross-sensor representation learning (SITR) remains necessary if data collection for each new sensor becomes straightforward. In other words, if sensor-specific data for any downstream task can be obtained easily, might single-policy approaches be more effective than a cross-sensor approach for certain tasks? Including a comparison in experiments could clarify this point, such as assessing whether the proposed method enhances sample efficiency in downstream task policy learning. This would help determine if easy access to sensor-specific data could potentially weaken the contribution of the cross-sensor approach in this work.

[1]. Chi, Cheng, et al. "Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots." arXiv preprint arXiv:2402.10329 (2024).

[2]. Yu, Kelin, et al. "MimicTouch: Leveraging Multi-modal Human Tactile Demonstrations for Contact-rich Manipulation." 8th Annual Conference on Robot Learning.

2). For the SCL loss, collecting labeled data in simulation requires additional effort, as human experts must ensure correspondence in data collection (i.e., matching contact poses on the same object). Could advanced unsupervised methods, such as the approach in [3], be used instead to avoid direct label generation? My concern is that while such controlled data collection is feasible in simulation—where we can enforce contact poses and objects—it would be challenging in the real world, as replicating data collection across different tactile sensors is cumbersome.

[3]. Xu, Mengda, et al. "Xskill: Cross embodiment skill discovery." Conference on Robot Learning. PMLR, 2023.

3). The zero-shot experiments focus on estimation results, but how does the method perform in manipulation tasks that require control inference? Such tasks could be challenging, as the pre-training phase is limited to static simulated tactile images, lacking the temporal reasoning essential for dynamic manipulation tasks. Additionally, the authors generate 100 simulated sensor configurations, which may introduce bias from the rendered images during simulation pre-training. This setup could potentially increase the sim-to-real gap, particularly for complex manipulation tasks.

4). The dataset is extensive, with 1 million samples. It would be helpful to know if such a large dataset is necessary for achieving the reported results. An ablation study on dataset size could provide valuable insights into how dataset scale impacts performance.

5). The baseline evaluation may be somewhat unfair. It appears the authors used existing models for T3 and UniT, but these models may not have been trained on a dataset as large as the one used for the proposed method. If so, this could lead to a performance drop for the baseline models. Ensuring comparable training conditions would provide a more accurate assessment of model performance.

6). The calibration process partially reflects sensor characteristics. However, even without calibration images, the model’s performance remains significantly higher than T3, as shown in Table 1 and Figure 8. This raises concerns that the large-scale dataset and loss design may contribute more to performance than the calibration process or the model architecture itself. To better understand the actual impact of calibration and the proposed model architecture, it would be essential to train T3 models using the same dataset and loss.

7). In Figure 9, the performance without SCL loss is not substantially lower, even for classification tasks, though Figure 7’s t-SNE results show a visibly worse latent space without SCL. Could the authors clarify this discrepancy? Given that the SCL loss is central to this work, it’s surprising that in pose estimation tasks, performance without SCL is almost identical. This raises the question of whether SCL is as critical as suggested, since the reconstruction loss alone—commonly used in other works—seems sufficient to achieve cross-sensor learning if the model can reliably reproduce contact images. If so, this would suggest that the simulation data collection pipeline and use of contact IDs may not be as effective as proposed.

Minor:

8). Figure 2 is incorrect; it’s not the object ID that should be close in the latent space, but rather the contact ID, which encompasses both the object ID and a similar contact pose.

9). Is the learned decoder for reconstruction still utilized in downstream tasks? Since there is a fine-tuned task-specific decoder for each task, I assume the answer is no. It seems that the decoder used during pre-training is solely for the reconstruction loss to facilitate encoder training. Please clarify this in the paper.

### Questions
See weakness section. If my concerns are decently addressed, I am happy to increase the rating.

### Soundness
3

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
5

### Summary
This study introduces a novel approach for extracting Sensor-Invariant Tactile Representations (SITR) aimed at enhancing the transferability of models across different vision-based tactile sensors. The authors employ a transformer-based architecture combined with supervised contrastive learning to create a robust tactile representation that allows for zero-shot transfer across various optical tactile sensors, specifically within the GelSight series. Experimental results validate the approach’s effectiveness across multiple tactile sensing applications, indicating potential advancements in model generalization and transferability for similar sensor designs.

### Strengths
1. Originality and Innovation: The integration of a transformer-based architecture with supervised contrastive learning to achieve sensor-invariant tactile representation is a novel approach within the tactile sensing field. The method is original and demonstrates the potential for addressing the long-standing issue of sensor-specific dependency in tactile applications.
2. Technical Rigor and Methodological Soundness: The paper presents a technically robust framework, comprising a two-stage process of normal map reconstruction and contrastive learning. The experiments cover a range of scenarios, lending empirical support to the effectiveness of the proposed method. The methodology is well-designed to achieve sensor invariance, and the use of transformers in tactile representation learning is a valuable contribution.
3. Potential for Real-World Impact: Achieving zero-shot transfer with minimal calibration across sensors is valuable in robotics and other domains that involve tactile sensing. By minimizing the need for sensor-specific calibration, the proposed method has potential implications for resource efficiency and ease of deployment in tactile applications.

### Weaknesses
1. Limited Generalizability: The approach is specifically tailored to GelSight sensors, limiting its generalizability to other types of tactile sensors. GelSight sensors are vision-based, and the proposed method may not be easily transferable to non-vision-based tactile sensors, such as capacitive or resistive types. This restriction diminishes the broader applicability of the findings, potentially limiting its impact within the ICLR community.
2. Practical Constraints of GelSight Sensors: GelSight sensors, due to their relatively large size and offline nature, present challenges for real-time or embedded applications, especially in constrained environments such as robotic end-effectors. The physical limitations of GelSight sensors may limit the practical deployment of the method in real-world applications, where compactness and real-time processing are often required.
3. Insufficient Justification for Calibration Object Selection: The study uses a ball and cube corner for calibration, yet lacks a theoretical explanation for why these shapes are specifically suited for achieving sensor invariance. A more detailed discussion of the rationale and potential generalization of these calibration objects would improve the clarity of the methodology and support its robustness across varied sensor designs.
4. Parameter Tuning Transparency: The paper does not provide adequate information on the parameter tuning process, particularly for the temperature parameter in the contrastive learning phase. This parameter significantly influences the performance of contrastive learning, and a more transparent tuning discussion would enhance reproducibility and provide insight into the method’s robustness.
5. Computational Complexity and Real-World Feasibility: Transformer models, while powerful, are computationally intensive and may not be optimal for real-time applications. The authors do not discuss computational demands or potential limitations in deploying the model on resource-constrained platforms. An assessment of computational requirements would provide a more comprehensive view of the method’s practicality.

### Questions
1. Calibration Object Choice: Could you elaborate on the theoretical and practical reasoning for selecting a ball and cube corner as calibration objects? How do these specific shapes contribute to achieving sensor invariance, and would the method remain effective with other shapes?
2. Generalizability to Non-Vision-Based Sensors: Has the SITR approach been tested or considered for use with non-vision-based tactile sensors (e.g., capacitive or resistive)? If not, what challenges or limitations do you foresee in generalizing this approach to other sensor types?
3. Temperature Parameter in Contrastive Learning: Could you provide insights into how the temperature parameter was tuned in contrastive learning? This parameter plays a crucial role in the learning process, and understanding its tuning would aid in assessing the approach’s sensitivity and reproducibility. Additionally, given that GelSight sensors tend to generate considerable heat during use, with noticeable temperature changes within a minute, how might such temperature variations impact the performance or stability of your model?
4. Feasibility for Real-Time Applications: Given the computational intensity of transformer-based architectures, have you assessed the feasibility of this approach in real-time applications? Would it be possible to optimize the model for deployment on resource-constrained platforms?
5. Given that GelSight sensors are tested independently in your experiments rather than being integrated on a robotic platform (e.g., mounted on both sides of a gripper), could you discuss any potential modifications or limitations of your method for practical deployment in space-constrained environments, such as on robotic end-effectors?
Thank you for your thorough and thoughtful work in this area. The study is innovative and contributes valuable insights to the tactile sensing field. We look forward to your clarifications and insights, which I believe will further enrich the impact of this research and its relevance to the ICLR community！

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
To address the problem of current vision-based tactile sensors' lack of transferability, this paper proposes a transformer-based method called SITR. SITR is trained on simulated tactile data, and is generalizable to real-world sensors though calibration. Experiments show the zero-shot performance on three downstream tasks: shape reconstruction, object classification and contact localization.

### Strengths
1. Sensor variance, which is the problem this paper tries to solve, is very important in the field of vision-based tactile sensing. Recently, different methods [1, 2, 3] have been proposed to address this problem, and this work proposes a model that is complementary to the prior works.

2. The framework proposed by this paper does not require any sensor-specific design in the model architecture, which seems to be its major contribution. Instead of designing different encoders or tokens for specific sensors, SITR only relies on calibration images to distinguish between different sensors. This can both reduce the model size and enable transferability to entirely novel sensors (as long as calibration images can be obtained).

3. The zero-shot results on downstream tasks show that the model transfers well across various sensor settings, which is a significant improvement upon the prior works.

References
[1] Zhao, Jialiang, Yuxiang Ma, Lirui Wang, and Edward H. Adelson. "Transferable Tactile Transformers for Representation Learning Across Diverse Sensors and Tasks." arXiv preprint arXiv:2406.13640 (2024).
[2] Rodriguez, Samanta, Yiming Dou, Miquel Oller, Andrew Owens, and Nima Fazeli. "Touch2touch: Cross-modal tactile generation for object manipulation." arXiv preprint arXiv:2409.08269 (2024).
[3] Higuera, Carolina, Akash Sharma, Chaithanya Krishna Bodduluri, Taosha Fan, Patrick Lancaster, Mrinal Kalakrishnan, Michael Kaess et al. "Sparsh: Self-supervised touch representations for vision-based tactile sensing." In 8th Annual Conference on Robot Learning.

### Weaknesses
1. The dataset the model is trained on all comes from simulation. The tactile simulator uses 3D meshes of the object to synthesize RGB tactile images, which assumes that all objects are rigid and without micro-geometry on the object surface (e.g., wood grain). Although this might be sufficient for geometry-based tasks such as shape reconstruction and object classification, the features extracted from this model may lack the ability to classify the fine-grained details (e.g., smoothness, hardness) in real-world setting. See the Questions section for more suggestions.

2. Despite being a sensor-agnostic model architecture, the calibration images are necessary for the SITR model. It's also shown in the ablation study that the model performance will drastically drop when calibration images are removed, and more calibration images usually lead to better performance. This leads to two potential concerns: (i) the calibration of new sensors requires specific objects (ball and cude) that might not be easily available to some of the users, which limits the application of the proposed model; (ii) the calibration images can lead to significant computation overhead, and most of the regions in the calibration images are redundant, i.e., not contacted by the object (shown in Fig. 3). See the Questions section for more suggestions.

### Questions
1. Referring to my first concern in Weaknesses section, it would be interesting to test the model's transferability to real-world datasets with more complicated settings. For example, Touch and Go [1] proposes three material understanding tasks that evaluate the model's understanding of material categories/ smoothness/ hardness given GelSight images. It would be interesting to see if the features learnt from pure simulation data can be transferred to these tasks.

2. Referring to my second concern, I'm curious about the computation overhead caused by using 18 calibration images. Also I wonder if a more careful design can be used in encoding the calibration images to reduce redundant information. As is shown in Fig. 3, only one out of the nine grids contains useful information. One simple potential improvement would be cropping the useful regions out from the 9 calibration images and concatenating them into a single image.

References
[1] Yang, Fengyu, Chenyang Ma, Jiacheng Zhang, Jing Zhu, Wenzhen Yuan, and Andrew Owens. "Touch and go: Learning from human-collected vision and touch." arXiv preprint arXiv:2211.12498 (2022).

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
4

### Summary
This paper aims to learn a tactile representation that can be used across different tactile sensors, including the same sensor design but with different instances, or different sensor designs. To achieve this, it collects a large dataset in simulation and uses contrastive learning to learn the representation. It divides a tactile image into patches and encodes it using a transformer. Meanwhile, it also passes simple calibration images, which contain simple objects pressing on different corners, to this transformer. The feature is further refined by learning to predict the normal map. The learned representation can be fine-tuned on downstream tasks and significantly outperforms baselines and other state-of-the-art methods.

### Strengths
The idea of using simple calibration images is interesting and inspiring. I believe this design should become a standard technique in future tactile representation learning.

The performance of the learned representation is quite good. On the three downstream tasks they evaluate, it significantly outperforms baselines and other methods in this area.

It also presents comprehensive ablation experiments on the role of calibration images and contrastive learning losses.

The paper provides nice visualizations of the learned feature space.

### Weaknesses
I believe there should be another important ablation experiment: training UniT/T3 on the simulated images collected in this paper. This is because there are two major differences between previous state-of-the-art methods (UniT/T3) and the proposed method. The first difference is the specific architectural design, such as contrastive learning, the transformer, and the use of calibration images. The second is the use of simulation data versus real-world data. This paper already presents experiments on architectural design but does not show whether the good performance mainly comes from a large simulation dataset. Specifically, I think the authors could train a representation with the loss proposed in UniT/T3 and compare its performance with the proposed method. This would provide more insight into where the performance improvement comes from.

It seems the authors do not compare the performance gain achieved by predicting the normal map.

Another weakness of this work is that it only evaluates sensors with flat gel pads, which have very similar optical designs. In contrast, other work, such as T3, also includes curved sensors. This point should be clearly discussed in the paper.

### Questions
What is the performance of UniT or T3 when trained on the simulation dataset gathered in this paper?

What is the effect of using the normal map prediction loss?

Is the statement “increasing the number of calibration images does not incur additional inference costs, as calibration tokens are computed only once per sensor” correct? Although it does not introduce additional computational cost for tokenization, the self-attention operator cost is O(n^2), where n is the sequence length. If you increase the number of calibration images, that cost will grow quadratically.

How is the “No SCL” baseline trained? Is it trained only with the normal prediction loss?

### Soundness
3

### Presentation
3

### Contribution
3
