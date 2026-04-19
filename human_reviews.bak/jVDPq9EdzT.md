# UniDrive: Towards Universal Driving Perception Across Camera Configurations

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Vision-centric autonomous driving has demonstrated excellent performance with economical sensors. As the fundamental step, 3D perception aims to infer 3D information from 2D images based on 3D-2D projection. This makes driving perception models susceptible to sensor configuration (e.g., camera intrinsics and extrinsics) variations. However, generalizing across camera configurations is important for deploying autonomous driving models on different car models. In this paper, we present UniDrive, a novel framework for vision-centric autonomous driving to achieve universal perception across camera configurations. We deploy a set of unified virtual cameras and propose a ground-aware projection method to effectively transform the original images into these unified virtual views. We further propose a virtual configuration optimization method by minimizing the expected projection error between original and virtual cameras. The proposed virtual camera projection can be applied to existing 3D perception methods as a plug-and-play module to mitigate the challenges posed by camera parameter variability, resulting in more adaptable and reliable driving perception models. To evaluate the effectiveness of our framework, we collect a dataset on CARLA by driving the same routes while only modifying the camera configurations. Experimental results demonstrate that our method trained on one specific camera configuration can generalize to varying configurations with minor performance degradation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Camera-only 3D object detection is sensitive to camera settings. It has been proved that a detection model trained on a camera setting fails to provide satisfactory performance in another camera setting. The authors propose a method, UniDrive, to unify this. The proposed UniDrive comprises a set of virtual cameras, which wrap real cameras into virtual space and thus bridge the gap between different camera settings. It is worth noticing that the number of real and virtual cameras does not need to be the same. The authors also proposed a virtual projection error, which enables an optimization process to find the optimal virtual camera setting. Experiments are conveyed on the CARLA simulator and the BEVFusion-C model. It is shown that, with the proposed method, model performance on various camera settings is significantly improved.

### Strengths
- The article is clear and easy to follow.
- The motivation of the article is a long-standing problem in the 3D detection domain, and the authors propose a feasible way for solving the problem.
- Mathematical details are given for the projection from virtual cameras to real cameras.
- In the experiments, the proposed method is testified on seven different camera settings, showing its generalization to different camera settings.

### Weaknesses
- Only BEVFusion-C is used in the experiments. In line 407, the authors mentioned that BEVFusion-C is one of the SOTA methods in popular public leaderboards, which is not currently the case. The effectiveness of the proposed method should show its universal capability on multiple methods with at least 2-3 different camera settings, such as some methods spercifically designed camera-only detection like BEVFormer [1] and Sparse4D [2].
- It is not spercifically described that how to adopt the proposed method to exsiting network. It would be helpful as the proposed method serves as a plug-and-play component.


[1] Li, Zhiqi, et al. "Bevformer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers." European conference on computer vision. Cham: Springer Nature Switzerland, 2022.

[2] Sun, Wenchao, et al. "SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation." arXiv preprint arXiv:2405.19620 (2024).

### Questions
- As mentioned in weaknesses, the effectiveness of the proposed method should be testified on multiple camera-only 3D detection methods.
- It is mentioned in line 302 that the virtual camera configuration is obtained through an optimization process based on a given set of camera systems; it is curious to know if the proposal method also works toward a randomly picked virtual camera setting. This would testify to the effectiveness of the proposed metric and optimization process in Sections 3.3 and 3.4. It is recommanded to compare the optimized virtual camera configuration to randomly selected configurations and evaluate performance across different camera settings.

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
I greatly appreciate this paper; in fact, I had similar ideas that I hadn't implemented. Seeing the author perfectly execute this idea and achieve excellent results is truly exciting. The paper proposes transforming images into a unified virtual camera space with a virtual configuration optimization strategy, thereby enhancing robustness to different camera configurations.

### Strengths
1. The paper proposes to transform images into a unified virtual camera space, improving robustness across various camera configurations.
2. Additionally, the paper proposes a virtual configuration optimization strategy that minimizes projection errors.
3. The paper also presents a systematic data generation platform and a benchmark for evaluating perception models under different camera configurations.

### Weaknesses
See the Questions section.

### Questions
1. Does the virtual configuration optimization strategy actually work effectively? （Core concern: I am highly willing to increase the score if this issue is explained thoroughly.）

   1.1 In certain camera configurations, the results from a single dataset in the paper are not as good as those of the original BEVFusion-C (e.g., 6x60 69.3 vs. 64.6, 6x80b 69.1 vs. 63.1). Why can't the virtual configuration optimization improve to achieve a virtual camera space as good as or better than the original BEVFusion-C?

   1.2 In some camera configurations, the results of individual optimization are not as good as those from transfer learning, such as 60x80b (train on 60x80b 63.1 vs. train on 60x70 67.9). Why can't the virtual configuration optimization achieve a virtual camera space as good as or better than the transfer setting?

   1.3 For different camera configurations, what do the optimized Virtual Cameras look like? Can you provide the optimized intrinsics and extrinsics?

   1.4 How well does the original view warp to the virtual view? Are there any areas that are not properly warped? Can you provide visual results?

2. While BEVFusion is popular, the paper focuses on vision-centric autonomous driving. It would be beneficial to include experiments with popular detectors in the camera domain, such as BEVFormer and PETR. These differ from methods like BEVFusion-C and BEVDet, as they are based on query-based transformer solutions.

3. Were the number of virtual cameras optimized together? If not, could you provide ablation experiments?

4. The paper discusses that Inconsistent Intrinsics have a significant impact. Since real autonomous vehicles also have inconsistent heights, could you include an experimental analysis?

   4.1 Were the heights of virtual cameras optimized together?

5. Given the various camera configurations, would training on datasets that mix various configurations to optimize a unified virtual camera space lead to improved results?

6. All experiments are conducted on simulation data. Incorporating transfer experiments with real datasets would be more convincing, like the real dataset transfer experiments conducted in 3D Domain Generalization/Adaptation.

Note: I have provided several questions. The authors can prioritize addressing the core issues based on their time constraints, and it is not necessary to conduct all experiments.

### Soundness
4

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
3

### Summary
The paper addresses the challenge of achieving consistent 3D perception in autonomous driving across different camera configurations. It proposes the UniDrive framework, which creates a unified virtual camera space using ground-aware projections and configuration optimization to reduce projection errors. Experiments in simulation demonstrate that UniDrive improves perception robustness across varied camera setups with minimal performance loss.

### Strengths
The paper addresses a valuable and relevant problem in autonomous driving, focusing on camera configuration generalization.

The proposed approach sounds generally reasonable, employing a unified virtual camera space and ground-aware projection to help manage variability in camera setups.

Experimental results show clear performance gains, with UniDrive maintaining high perception accuracy across diverse camera configurations and outperforming baseline models.

### Weaknesses
W1: The innovation is somewhat limited, as the framework mainly leverages established virtual projection and optimization techniques without presenting substantially novel concepts.

W2: The approach relies heavily on simulated environments (e.g., CARLA), raising concerns about its applicability and robustness in real-world conditions.

### Questions
To what extent does the reliance on simulated data affect the generalizability of the results to real-world settings, particularly in urban driving scenarios?

Does the framework introduce any latency or computational overhead that could impact its feasibility for real-time applications in autonomous driving?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses the challenge of maintaining robustness in perception algorithms for autonomous driving systems as camera configurations—such as intrinsics, extrinsics, and the number of cameras—evolve. The authors introduce a novel re-mapping approach that transforms pixels from the original camera view to a unified virtual view, allowing training and inference to be conducted in this virtual space. This approach improves robustness across different configurations and integrates seamlessly into existing perception pipelines. 
In addition, the paper proposes a virtual camera configuration optimization method, enabling the identification of an optimal virtual configuration based on real-world multi-camera systems to further enhance robustness across configurations. 
Experimental results on synthesized data show that the method not only enhances cross-configuration performance but also unexpectedly boosts performance on the same configurations.

### Strengths
- This paper addresses a valuable real-world demand. As autonomous driving sensor configurations (e.g., camera intrinsics, extrinsics, and the number of cameras) evolve, it becomes crucial to ensure the perception algorithms are robust to such changes.
- The authors propose a re-mapping approach that transforms pixels from the original view into a unified virtual view, enabling training and inference to be conducted on this virtual view. This approach enhances robustness to cross-configuration shifts in camera systems. Furthermore, the method is modular, making it straightforward to integrate into existing perception pipelines.
- A virtual camera configuration optimization method is also introduced, which can quantitatively find a better virtual camera configuration based on a set of real-world multi-camera system to enhance the cross-configuration robustness and perception performance. 
- Experimental results demonstrate that the proposed approach significantly improves robustness across different configurations. Interestingly, performance also improves on configurations identical to the training setup. This is surprising, as one might assume additional transformations could lead to information loss rather than enhancement. This raises questions about whether these improvements are dataset-dependent (I posed in Weaknesses).
- The paper presents a well-formulated problem and demonstrates effective solutions under simulated data conditions.

Despite some limitations, especially those related to simulated data, this paper introduces compelling ideas, offering a well-defined problem formulation that could benefit the autonomous driving industry and potentially embodied AI.

### Weaknesses
- As the authors noted, one main limitation is that all analyses were conducted in a simulated environment. While this controlled setting is useful for evaluating the method’s potential, it raises concerns about real-world applicability, as training-based detectors are often sensitive to dataset-specific factors. Given the paper’s real-world motivation, validation on actual datasets would be valuable.
- As I posed in Strengths, it is a little counterintuitive that the transformation from the original view to a virtual view may cause information loss but can bring improvements. The authors should consider proposing hypotheses to explain why this transformation improves performance.

Apart from the issues I posed above, I also have some suggestions that will not prevent this paper to be accepted if unresolved. In real-world applications, there are several datasets of different camera configurations. As the proposed method can leverage the virtual view to unify these different configurations, I am curious whether the proposed method can leverage these datasets to train a better-to-all detector on each dataset.

### Questions
Please check weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
