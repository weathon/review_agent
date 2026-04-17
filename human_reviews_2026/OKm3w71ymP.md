# OpenFly: A COMPREHENSIVE PLATFORM FOR AERIAL VISION-LANGUAGE NAVIGATION

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Aerial Vision-Language Navigation (VLN) seeks to guide UAVs by leveraging language instructions and visual cues, establishing a new paradigm for human-UAV interaction. However, the collection of VLN data demands extensive human effort to construct trajectories and corresponding instructions, hindering the development of large-scale datasets and capable models. To address this problem, we propose OpenFly, a comprehensive platform for aerial VLN. Firstly, OpenFly integrates 4 rendering engines and advanced techniques for diverse environment simulation, including Unreal Engine, GTA V, Google Earth, and 3D Gaussian Splatting (3D GS). Particularly, 3D GS supports real-to-sim rendering, further enhancing the realism of our environments. Secondly, we develop a highly automated toolchain for aerial VLN data collection, streamlining point cloud acquisition, scene semantic segmentation, flight trajectory creation, and instruction generation. Thirdly, based on the toolchain, we construct a large-scale aerial VLN dataset with 100k trajectories, covering samples of diverse scenarios and assets across 18 scenes. Moreover, we propose OpenFly-Agent, a keyframe-aware VLN model emphasizing key observations to promote performance and reduce computations. For benchmarking, extensive experiments and analyses are conducted, where our navigation success rate outperforms others by 14.0\% and 7.9\% on the seen and unseen scenarios, respectively. The toolchain, dataset, and codes will be open-sourced.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The authors introduce a dataset and method for aerial Vision–Language Navigation (VLN). The OpenFly dataset contains 100K automatically generated trajectories, synthesized using multiple rendering sources such as Unreal Engine and Google Earth. Compared to existing aerial VLN datasets, OpenFly is the largest in terms of trajectory count. The proposed OpenFly-Agent is a keyframe-aware model based on the OpenVLA framework. By selecting representative frames and merging redundant visual tokens from high-frequency inputs, OpenFly-Agent efficiently predicts navigation actions using a Llama2-7B backbone. Experimental results demonstrate significant improvements compared to existing VLN baselines, such as NaVila  (from 14.2% to 32.2% success rate on seen scenarios).

### Strengths
The OpenFly dataset's scale (100K trajectories) represents a significant contribution despite its automated generation. Additionally, the performance of OpenFly-Agent is noteworthy, as it substantially outperforms recent baselines.

### Weaknesses
A primary limitation of the paper is its limited technical novelty. Since the model builds upon the existing OpenVLA framework, the novelty mainly lies in incremental enhancements, specifically the addition of keyframe selection and token pruning mechanisms. Several experimental issues raise further concerns:

(1) How were the Navid and NaVila baselines trained?

Implementation details and training procedures for these baselines are not provided in the paper. This makes it challenging to validate whether OpenFly-Agent indeed outperforms them in a fair comparison.

(2) Why was OpenVLA selected as the backbone?

The performance of OpenVLA baseline (2.3% in Table 3) is significantly lower than that of Navid (13.0%) and NaVila (20.3%) in Table 1. Given this substantial gap, the authors' choice of OpenVLA as the backbone over more performant alternatives such as NaVila requires further justification.

(3) Does OpenFly-Agent generalize to other datasets?

Considering that trajectories are automatically generated, direct comparisons in terms of dataset scale to handcrafted datasets such as TouchDown and CityNav become less relevant. Rather, the automated data-generation process could be viewed as part of the training methodology itself. Thus, evaluations on external datasets (e.g., AerialVLN, CityNav, OpenUAV) are necessary to substantiate the generalization capability of OpenFly-Agent.

### Questions
Please refer to the questions listed under Weaknesses. I like the approach of this paper. However, clarification and a generalization evaluation are needed. The paper itself is well-written and easy to follow.

### Soundness
2

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
5

### Summary
This proposes a large-scale dataset for aerial visual-language navigation.  The dataset comprises a high diversity of simulation scenes collected from 4 rendering engines, and it annotates a large number of navigation trajectories using autmated toolchain.  This paper demonstrates that the proposes dataset enhances performant robot policies.  It trains an OpenVLA model on the collected trajectories, which shows stronger performance on in-domain scenes and better generalization to out-of-domain scenes, during testing.  The authors commit that the dataset and codes will be publicly released.

### Strengths
1. This paper is well written.  It illustrates clearly the limitations of previous methods and how it addresses them in the newly proposed dataset.
2. The proposed dataset contains a high diversity of scenes (18 in total), while some correspond to real-world scenes
3. This paper proposes a strong baseline based on the performance OpenVLA model, outperforming other baselines by a large margin
4. The trained OpenVLA agent shows strong generalization to unseen scenes during testing
5. The trajectory collection pipeline is neat, which generates navigation trajectories without the need of manual effort

### Weaknesses
1. Although this paper provides details evaluation of OpenVLA agents, it lacks analysis on the effectiveness of the proposed dataset. In other words, this paper does not address the question "Do the diversity and quality of collected navigation trajectories facilitate more performant navigation agents, compared to other datasets?". I'd recommend the authors train OpenVLA on both OpenFly and OpenUAV datasets, and test on unseen scenes to highlight the effectiveness of the proposed dataset.
2. This paper utilizes A* algorithm to generate navigation trajectory.  However, this paper does not comment or provide any analysis on the potential biased introduced by A* algorithm.  For example, are the generated trajectories smooth, energy efficient, or safe? These are key factors that should be considered when deploying simulation-trained agents to the real world.
3. Likewise, this paper selects 4 DoF action space, which sounds reasonable for long-horizon navigation.  However, such a simplified action space is insufficient for drones to navigate in highly-occluded scenes (e.g. street scenes).  This paper doesn't discuss such limitations.
4. I'm not sure if training OpenVLA models for navigation is a right reseaerch direction, since recent studies show that navigation tasks are inherently visual grounding tasks.  Since VLMs excel at visual grounding, one can directly use VLM models to generate 2D waypoints for 3D navigation [1].  I'd strongly recommend the authors to test these zero-shot navigation policies on the proposed dataset.

---

Reference:
[1] Hu, Chih Yao, et al. "See, Point, Fly: A Learning-Free VLM Framework for Universal Unmanned Aerial Navigation." Conference on Robot Learning. PMLR, 2025.

### Questions
1. Do the diversity and quality of collected navigation trajectories facilitate more performant navigation agents, compared to other datasets?  A comparison of OpenVLA models trained on OpenFly and OpenUAV and tested on unseen scenes is needed.
2. What are the biases introduced by A* algorithm for generating action trajectories?
3. What are the limitations of 4 DoF action space in aerial navigation tasks?
4. What is the performance of VLM-based navigation models (See-Point-Fly) on the proposed dataset?

I'm open to increase my rating if the authors are able to address these concerns.

### Soundness
3

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
The paper "OpenFly: A Comprehensive Platform for Aerial Vision-Language Navigation" presents OpenFly, a novel platform aimed at addressing the challenges in aerial vision-language navigation (VLN) for UAVs (unmanned aerial vehicles). It tackles the issues of limited data diversity, high data collection costs, and small dataset scales that have hindered previous aerial VLN efforts. The key contributions of this paper are:
1.Platform Design: OpenFly integrates four rendering engines (Unreal Engine, GTA V, Google Earth, and 3D Gaussian Splatting), significantly enhancing the diversity of simulated environments used for VLN tasks. The 3D Gaussian Splatting technique, in particular, allows for realistic real-to-sim rendering, making the environments more photorealistic.
2.Automated Data Generation: The paper introduces a toolchain for automating the collection of aerial VLN data, which includes steps like point cloud acquisition, scene semantic segmentation, trajectory creation, and instruction generation. This toolchain reduces the need for manual annotation, thus enabling the large-scale generation of high-quality data.
3.Large-Scale Dataset: OpenFly's platform results in a large-scale dataset consisting of 100,000 trajectories, making it one of the largest aerial VLN datasets to date. This dataset spans 18 different scenes and includes a wide variety of environmental features and flight scenarios.

### Strengths
1. Innovative Data Generation Platform:
   OpenFly integrates four rendering engines (Unreal Engine, GTA V, Google Earth, and 3D Gaussian Splatting), which enhances the diversity of training environments for aerial vision-language navigation (VLN). This combination provides a wide range of realistic simulation environments for training models.
2. Automated Data Generation Toolchain:
   The platform features an automated toolchain for data collection, semantic segmentation, trajectory generation, and instruction creation. This reduces the reliance on manual annotations and makes it easier to scale data collection processes, allowing the creation of large-scale VLN datasets.
3. Real-World Experimentation and Model Performance:
   OpenFly-Agent is tested in real-world scenarios, and the model shows strong performance in both simulated and unseen environments, highlighting its potential for real-world application in aerial navigation tasks.

### Weaknesses
1. Why use such an old model, Llama2-7b, as the baseline?
2. The paper cites OpenUAV, but why isn't there a comparison with their method?
3. VTM appears to be a pooling layer, but there's no explanation for why the performance improvement is so significant.
4. Are keyframes selected based on rules? What would be the difference if they were selected uniformly? Would the performance decrease if important frames are missed?

### Questions
same as weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents OpenFly, an Aerial VLN platform addressing data scarcity. It integrates four rendering engines (using 3D GS for Real-to-Sim) and an automated data toolchain to generate a 100k-trajectory dataset. A novel keyframe-aware model, OpenFly-Agent, achieves SOTA performance in simulation and 23 real-world scenarios, validating its strong sim-to-real transfer.

### Strengths
1.	The paper directly tackles the most significant bottleneck in Aerial VLN. The automated toolchain is a highly valuable and practical contribution, drastically lowering the barrier to data collection.
2.	The 100k-trajectory dataset is the largest to date. More importantly, the integration of four distinct rendering engines, especially the use of 3D GS for real-world reconstruction, ensures exceptional environmental diversity and realism.
3.	The OpenFly-Agent, with its keyframe-aware selection and token merging, presents an intelligent approach to handling redundant video data in VLN, improving both efficiency and performance.

### Weaknesses
1. Relying on the A* data-generation pipeline introduces a two problems: (1) The path style is unnatural, filled with "robotic sharp turns" instead of smooth, human-like flight. (2) The data is pure "expert demonstration", meaning the model only learns to follow perfect paths, not recover from deviations. This makes the model extremely fragile to real-world disturbances.
2. The system uses discrete motion control, which in practice, is more like performing control classification based on visual input. During actual deployment, the UAV is unable to achieve precise flight maneuvers, such as orbiting a small object, because the current turning angles are insufficient

### Questions
1. The paper mentions in Section 3.2 that collision-free training trajectories are generated using A*. However, during the simulation and real-world testing in Section 6, no collision detection mechanism or related metrics (e.g., number of collisions) are mentioned. Please clarify: (1) During evaluation, are the agent's generated trajectories checked for collisions? (2) If a collision occurs, is the task considered a failure, or does it continue? (3) Does the OpenFly-Agent itself possess any form of dynamic or reactive collision avoidance capability?

2. Could the authors clarify the exact movement mechanism for the agent in the simulation? The description suggests the model chooses between a series of discrete waypoints or actions (e.g., Forward 9m). This raises a critical question: after executing an action, does the agent "teleport" to the next state, or does it execute a simulated continuous flight action between points? If it is "teleportation," the task essentially degenerates from a "navigation control" problem into a "sequential VQA" problem. This is fundamentally different from the real world, where a UAV must use continuous motor control to counteract inertia, wind, and various physical disturbances. Please clarify this setup, as it profoundly impacts the sim-to-real difficulty and the model's true generalization capabilities.

3. The ablation study shows a massive performance leap when moving from "KS" (9.2% SR) to "KS + VTM" (34.3% SR), suggesting the VTM module is critical. Could the authors elaborate on the reasons why this token merging strategy results in such a dramatic improvement? 

4. How to deal with the problem in weakness one?

I think the paper's overall contribution to be valuable, particularly the development of a pipeline for building UAV environments. Even if the trajectories have certain issues, the continuous videos are still useful for non-control applications, such as VLM pretraining or developing world models.
However, the paper is positioned as a comprehensive 'platform.' This implies that in addition to the environment, the trajectories and control mechanisms must also be sound and well-justified.
Therefore, I expect the authors to specifically address my questions on these points. I will consider raising my score upon receiving a satisfactory response.

### Soundness
3

### Presentation
3

### Contribution
2
