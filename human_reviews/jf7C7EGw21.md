# VTDexManip: A Dataset and Benchmark for Visual-tactile Pretraining and Dexterous Manipulation with Reinforcement Learning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 5, 6, 5

## Abstract
Vision and touch are the most commonly used senses in human manipulation. While leveraging human manipulation videos for robotic task pretraining has shown promise in prior works, it is limited to image and language modalities and deployment to simple parallel grippers. In this paper, aiming to address the limitations, we collect a vision-tactile dataset by humans manipulating 10 daily tasks and 182 objects. In contrast with the existing datasets, our dataset is the first visual-tactile dataset for complex robotic manipulation skill learning. Also, we introduce a novel benchmark, featuring six complex dexterous manipulation tasks and a reinforcement learning-based vision-tactile skill learning framework. 18 non-pretraining and pretraining methods within the framework are designed and compared to investigate the effectiveness of different modalities and pertaining strategies. Key findings based on our benchmark results and analyses experiments include: 1) Despite the tactile modality used in our experiments being binary and sparse, including it directly in the policy training boosts the success rate by about 20\% and joint pretraining it with vision gains a further 20\%. 2) Joint pretraining visual-tactile modalities exhibits strong adaptability in unknown tasks and achieves robust performance among all tasks. 3) Using binary tactile signals with vision is robust to viewpoint setting, tactile noise, and the binarization threshold, which facilitates to the visual-tactile policy to be deployed in reality. The dataset and benchmark are available at \url{https://github.com/LQTS/VTDexManip}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces a dataset and benchmark to address the limitations in current robotic manipulation learning frameworks by incorporating both vision and tactile modalities. It collects human manipulation data with 182 objects and 10 daily tasks and uses low-cost tactile gloves to create a large-scale dataset. The dataset is used to study multi-modal pretraining and dexterous robotic manipulation using reinforcement learning. The authors benchmark 17 pretraining and non-pretraining methods to evaluate the effectiveness of visual and tactile inputs in dexterous manipulation tasks.

### Strengths
1. This paper create the first large-scale visual-tactile dataset for dexterous manipulation tasks, addressing a gap in the existing robotic learning literature. The dataset covers a wide variety of tasks and objects, which makes it valuable for generalizing robotic manipulation skills.

2. The use of both vision and tactile information for joint pretraining boosts performance in complex dexterous tasks.

3. The benchmark evaluates various pretraining and non-pretraining methods, which provides a comprehensive understanding of how different modalities affect task performance.

### Weaknesses
1. The contribution to the community is not super clear to me. Currently, what kind of tactile sensor should be used on the Dexterous Hand is still an open question. Although this paper discuss the Gelsight in the section 5.5, there are still a lot of different sensors. Optical tactile sensor [1, 6] has been used for dexterous manipulation, where "they have not been evaluated for dexterous manipulation with complex dynamics and coordination between fingers" shown in the paper might be incorrect. Also, magnetic tactile sensor [2, 3], force tactile sensor [4] are used in dexterous manipulation tasks. It's not clear why this type of tactile data is chosen for pre-training. Why does this benchmark could be widely used since it's still an open question?

2. There are some paper uses visual-tactile representations for reinforcement learning [5, 6]. I think compare using the pre-trained with those SOTA online-learning model is helpful (not only using simple ResNet and MLP).

3. Section 5.4 shows that this framework can do Sim2Real and show some impressive demos. However, no quantitative results have been shown in the paper.

4. A new CoRL paper [7] present a method to learn tactile-guided control policy from human hand demonstrations by embedding tactile sensor on the human finger tip, where the idea is similar to collect human-tactile data with tactile glove shown in this paper. What's the difference between these two approaches? It's helpful if analyzing it in related work.

5. In section 5.2, the paper didn't show which task is it.

[1]. Lambeta et al., DIGIT: A Novel Design for a Low-Cost Compact High-Resolution Tactile Sensor With Application to In-Hand Manipulation, 2020

[2]. Bhirangi et al., ReSkin: a versatile, replaceable, low-cost skin for AI research on tactile perception, 2021

[3]. Bhirangi et al., AnySkin: Plug-and-play Skin Sensing for Robotic Touch, 2024

[4]. Yin et al., Rotating without Seeing: Towards In-hand Dexterity through Touch, 2023

[5]. Sferrazza et al., The Power of the Senses: Generalizable Manipulation from Vision and Touch through Masked Multimodal Learning, 2024

[6]. Qi et al., General In-Hand Object Rotation with Vision and Touch, 2023

[7]. Yu et al., MimicTouch: Leveraging Multi-modal Human Tactile Demonstrations for Contact-rich Manipulation, 2024

### Questions
1. This paper only compare the VT-JointPretrain with V-Pretrain, V-Pretrain+T, T-Pretrain, and V+T. How about using V-Pretrain+T-Pretrain separately?

More questions have been shown in Weakness. I will consider increasing my score if the contribution can be explained more clearly.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a vision-tactile dataset and a novel benchmark for evaluating the vision-tactile pre-training methods. The experiments show that the tactile signals can provide guidance for the deployment of the tactile modality.

### Strengths
1. The experiments demonstrate both simulation and real-world settings.

### Weaknesses
- This MAE-based method is not novel. There is also another paper try to use the MAE to fuse the vision and tactile information: The power of the senses: Generalizable manipulation from vision and touch through masked multimodal learning. So the author at least should compare with it. 

- The comparison is not fair. The pre-trained model-based methods only use an MLP to concatenate with tactile information. This operation cannot illustrate that these models cannot leverage the tactile information better.

- Could you give me some intuitions about why VT-Joint training can improve the robustness of viewpoint changing? I think the reason might be these tasks heavily reply on tactile rather than vision.

- Why Voltron performs so bad?

- Where is the real-world experiment results?

### Questions
Seen above in weakness.

### Soundness
2

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
3

### Summary
This paper introduces a new visual-tactile dataset for training and benchmarking dexterous manipulation skills. It includes 10 tasks with 182 objects collected by human operators and provides a comprehensive benchmark of 6 challenging dexterous manipulation tasks. Additionally, the paper presents an extensive evaluation of over 17 methods on the benchmark, showcasing the efficacy of the dataset. Overall, this research is a valuable contribution, offering a well-structured dataset and benchmark for advancing visual-tactile manipulation.

### Strengths
- The paper introduces a highly diverse visual-tactile dataset for complex dexterous manipulation tasks, enhancing the realism and diversity of training data.

- Detailed analysis and evaluation of the proposed dataset underscore its robustness and potential impact on advancing dexterous manipulation.

- The proposed VT-JointPretrain method is promising, effectively integrating visual and tactile inputs for manipulation tasks.

- The extensive experimental analysis provides insights into the impact of pre-training on dexterous manipulation, with comparisons between different pre-trained and non-pre-trained methods on the benchmark.

### Weaknesses
- The tactile data in the proposed dataset lacks high-resolution force measurements, providing only binary force information per sensor. This limitation may impact tasks requiring precise force control, such as in-hand rotation or object flipping, where the exact contact force is critical.

- An ablation study of the visual-tactile representation learning framework is necessary to understand its efficacy further. Specifically:
(i) Examining performance variations with different masking rates of image and tactile data would clarify the framework’s robustness.
(ii) Analyzing performance with only image or only tactile data masked could highlight the independent contributions of each modality.

- Although the authors demonstrate successful deployment of the trained policy on real robots, the evaluation of its real-world performance is limited and lacks detailed analysis. A more rigorous evaluation of real robotic setups would strengthen the claims of real-world applicability.

### Questions
1- How well do the learned skills transfer to real-world dexterous manipulation scenarios? Have you evaluated the performance on a real robotic setup, and if not, are there plans for such an evaluation?

2- Could you elaborate on the rationale for using binary force information in the tactile data? Do you have plans to incorporate higher-resolution force data in future versions of the dataset to improve its applicability for tasks requiring fine-grained force control?

3- Have you considered conducting an ablation study with varying mask rates or with only image/tactile modalities masked? If so, could you share preliminary results or insights on how these variations impact the performance of the VT-JointPretrain framework?

4- Given the variety of tasks in the benchmark, how well does the proposed framework generalize across different types of manipulation tasks? Are there specific tasks or object types where it struggles, and if so, what might be the underlying reasons? (further discussion on failures could improve the paper)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The author introduces a unique dataset for visual-tactile with 10 daily tasks.  A new benchmark with six complex dexterous manipulation tasks is proposed, using a reinforcement learning-based vision-tactile skill learning framework.
The authors have key findings that incorporating tactile data boosts task success rates by up to 40%.

### Strengths
1. I appreciate the authors' efforts to collect a larger visuo-tactile dataset.
2. Sufficient experiments as well as real experiments.
3. The visualization for t-SNE is good to show how tactile differ in different tasks.

### Weaknesses
Major Weaknesses:
1. The overall presentation and low readability.

2. The contribution 2 is questionable. Half of the tasks have been already widely used in previous benchmark tasks[1], like bottlecap turning, faucet screwing, reorientation, and bimanual hand-over. Even though previous work is state-based not using visual-tactile, the authors cannot mention their contribution to creating tasks. 

3. The authors do real-world experiments, but there is no table of experiments to show the number, which cannot make any convincing conclusion.


[1] Chen, Y., Wu, T., Wang, S., Feng, X., Jiang, J., Lu, Z., ... & Yang, Y. (2022). Towards human-level bimanual dexterous manipulation with reinforcement learning. Advances in Neural Information Processing Systems, 35, 5150-5163.

### Questions
1. [line 382-387] Please make clear the definition of robustness to the viewpoint. Usually, when we refer to robustness to the viewpoint, we are talking about training policy in one viewpoint and test on another viewpoint.

2. Please specify the meaning and describe a little bit for each subfigure of Figure 1 and 2, instead of just putting on sentence in the caption.

3. [Line 16]. Only 6 tasks have been found in the paper, but the authors mention they have 10 tasks.

4. The real-world experiments use teacher-student to train the policy. However, this is not mentioned in the method section, which may cause confusion.

5. In section 5 benchmarking study, please specify how many seen objects are used and how many unseen objects are used. It would be better for the author to show all the objects that are used for each task.

### Soundness
3

### Presentation
2

### Contribution
3
