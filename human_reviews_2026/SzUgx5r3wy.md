# Self-Improving Loops for Visual Robotic Planning

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Video generative models trained on expert demonstrations have been utilized as performant text-conditioned visual planners for solving robotic tasks. However, generalization to unseen tasks remains a challenge. Whereas improved generalization may be facilitated by leveraging learned prior knowledge from additional pre-collected offline data sources, such as web-scale video datasets, in the era of experience we aim to design agents that can continuously improve in an online manner from self-collected behaviors.  In this work we thus propose the Self-Improving Loops for Visual Robotic Planning (SILVR), where an in-domain video model iteratively updates itself on self-produced trajectories, and steadily improves its performance for a specified task of interest.  We apply SILVR to a diverse suite of MetaWorld tasks, as well as two manipulation tasks on a real robot arm, and find that performance improvements continuously emerge over multiple iterations for novel tasks unseen during initial in-domain video model training.  We demonstrate that SILVR is robust in the absence of human-provided ground-truth reward functions or expert-quality demonstrations, and is preferable to alternate approaches that utilize online experience in terms of performance and sample efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes **SILVR (Self-Improving Loop for Visual Robotic Planning)**, a framework that treats a video generative model as a visual planner and **iteratively improves it with self-collected experience**. Concretely, an in-domain video model produces future-frame plans given a text prompt; a separately trained **inverse dynamics model (IDM)** converts consecutive planned frames into executable actions that are run in the environment, after which successful rollouts are filtered and used to **finetune both the in-domain video model and the IDM**—then the loop repeats.

### Strengths
1. The authors propose a simulator-driven self-optimization approach for video generation, enabling the success rate on corresponding tasks to increase with additional iterations.
2. They evaluate SILVR’s generalization and robustness across multiple environments. In simpler settings, SILVR can accomplish the target tasks to a certain extent and shows a degree of robustness and generalization.

### Weaknesses
1. Requires an initially solvable task; limited gains when the starting success rate is very low.
2. Real-world gains depend on an internet video prior and are modest/unstable without it, with few reported iterations.
3. Evaluation focuses on short-horizon/simple tasks, leaving feasibility on harder or long-horizon tasks unclear.
4. Minor presentation issue: Some figures are hard to read in the PDF due to lack of detail explaination for some signals.

### Questions
1. Why do some tasks trained from suboptimal initial data outperform the expert-initialized setting?
2. Why does the SILVR-distilled Diffusion Policy (DP) achieve a higher success rate than the final SILVR visual planner?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this work, the authors propose a method named Self-Improving Loops for Visual Robotic Planning (SILVR), which iteratively updates in-domain video model on self-produced trajectories to improve its performance for a specified task of interest. The authors thoroughly demonstrate how SILVR establishes connection between in-domain video model and high-performing visual planner for specified novel robotic control task. Extensive evaluations on the MetaWorld task suite and real-world robot arm verify the effectiveness of SILVR.

### Strengths
1. The proposed SILVR is straightforward and easy to follow, leveraging iterative fine-tuning of a visual planner on self-collected experience without complex reward engineering.
2. The paper provides extensive experiments across both simulated (MetaWorld) and real-world (Franka Panda arm) settings, demonstrating broad applicability and robustness.
3. SILVR shows significant performance improvement on unseen tasks (up to 285% in MetaWorld) and outperforms reinforcement learning and behavior cloning baselines in sample efficiency.
4. The method is robust to suboptimal initial data, does not require ground-truth rewards (works with VLM-based filtering), and can effectively incorporate internet-scale video priors for real-world generalization.

### Weaknesses
Since that I am not an expert in this field, here are a few potential weaknesses I observed from the paper:
1. While the method is well-executed and effective, the core idea of a self-improving loop—fine-tuning a model on its own successful outputs—is a well-established concept in other areas like large language models. Applying this established concept to visual planning, while practical, may be perceived as a solid incremental advancement rather than a foundational shift in methodology.
2.  The real-robot experiments (pushing a colored cup, opening a colored drawer), while important for demonstrating real-world applicability, are relatively simple table-top manipulations. It remains unclear how well SILVR would scale to tasks with more complex dynamics, longer time horizons, or tasks requiring intricate multi-step reasoning and tool use.
3. The self-improving loop implicitly assumes the initial model can achieve a non-trivial success rate to collect useful data. For tasks that are too distinct from the initial training distribution, the model might fail completely at iteration zero, preventing any improvement and creating a "cold-start" problem that SILVR fails to address.

### Questions
See weaknesses

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
4

### Summary
This paper proposes to iteratively update a text-&-image conditioned video model that generates visual imaginations of robot trajectories via online RL (instantiated as filter-BC), towards constructing a self-improving robot policy. The policy is formulated as a high-level video generator and an inverse dynamics model that can predict actions given consecutive frames. In Meta-World and two real-world experiments, the authors explore running filter-BC to improve the high-level policy, where in each iteration data is collected by the current combined policy, a success detector is called (either human ground truth, or the authors also explore using a VLM success detector) to label the self-collected data, then the video model is trained on the successful split of this data. The authors run 10 iterations of self-improvement in Meta-World, observing an average performance increase from 15% success rate to 56%, and run 2 iterations of self-improvement on two real-world tasks, observing an improvement from 50% to between 60 and 70%. For real-world experiments they find that their approach can generalize/perform better if their video model is combined with frozen pre-trained video model using a CFG-like score composition. They also run a few ablations, finding that with a VLM success detector that may not be as accurate, performance of their method still improves, and their method can also work with no success detector at all, and when the BC pre-training data is suboptimal. They only compare to baseline approaches on Meta-World (not the real-world), but compare to DSRL and regular filter-BC (without a video model), and find that their approach does better. The authors also explore distilling the two-level policy into a one-level policy after RL training.

### Strengths
(1) As far as I'm aware the idea of doing self-improvement of a high-level video generation policy in the robotic setting has not been explored yet, making this work novel in this regard.

(2) The performance of the approach is shown to be better than two relevant self-improvement baselines, DSRL and regular filter-BC, which is a promising result. 

(3) The authors ablate many components of their approach, making the experimentation fairly thorough

### Weaknesses
(1) While self-improvement is obtained, success rate on both MetaWorld and the real-world seems to cap out at around 60 to 70%, and more self-improvement iterations do not improve performance. Intuitively it would seem that there should be no cap on performance. Why does the approach not seem to be able to improve past this success rate?

(2) The authors do not make a convincing argument about why filter BC of a video generative high-level policy should be better than filter-BC of a regular single-level policy. *This leaves the reader unclear about what they should take away from the paper.* On line 362 the authors mention that "we hypothesize that the separately learned environment visual dynamics is easier to transfer when solving a novel task, leading to stronger base generalization performance through visual planning". Why should a visual dynamics model generalize better than a direct action prediction policy? Even if it does generalize better, the better generalization should only affect initial performance. Why is the proposed approach able to *improve* better than the baseline?

### Questions
(1) The IDM model is updated alongside the high-level policy only for the MetaWorld experiments (as stated in line 347). Why not also for the real-world experiments?

(2) Instead of doing score composition do combine the pre-trained video model and in-domain one at inference time, can the pre-trained one be finetuned on in-domain robotic data, and then passed to the self-improvement process? 

(3) For score composition to work, does the internet pre-trained model need to generate videos that look sufficiently similar to the in-domain model? It would seem that if the pre-trained model generates very different videos, score composition would not work well. How do you assess whether the required similarity is achieved?

(4) DSRL is used as a baseline in Table 1, but DSRL doesn't have a concept of "iterations" of self-improvement in the same manner that the proposed approach does. Rather a gradient step can be taken any time new observations are added to the replay buffer. Can you explain then how DSRL results were mapped to iterations 1 through 4?

(5) What causes the approach's performance to go down (as opposed to up) when just the in-domain model is used in the real-world experiments? What do the authors think specifically prevents the method from working in this case?

(6) How can success rate go up without filtering in Figure 5? What is the natural success rate percentage of the data being collected?

(7) How do baseline approaches (e.g., filter BC) perform in the real-world? I noticed the real-world experiments do not have any baseline approaches.

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
The paper studies how to make video-based visual planners automatically adapt and generalize to novel robotic tasks by learning from self-collected online experience. The proposed method (SILVR) creates an iterative loop: adapt an in-domain text-to-video model with an internet-scale prior, roll out generated visual plans via an inverse dynamics model, and use the filtered successful trajectories to update both the in-domain video model and the inverse dynamics model. Experiments in both the MetaWorld simulation and the real world demonstrate that SILVR can improve success rates and outperforms other online improvement baselines.

### Strengths
- Using online interaction to improve visual planning is a novel and interesting research problem.
- The approach is intuitive and performs well in practice.
- Thorough experiments in both simulation and on real robots. The training tasks and test tasks are separated clearly, allowing a strict assessment of how well the method adapts to unseen tasks.

### Weaknesses
- I treat the method as two complementary parts: (a) improve the visual planner by successful trajectories rolled out with the help of a good IDM, and (b) improve the IDM with meaningful online behavior collected under the guidance of a good visual planner. Though the paper is mostly presented from perspective (a), it would be valuable to do some ablations to understand whether the empirical gains come mainly from (a), (b), or their combination. My feeling is that the answer may differ across settings. On the real robot, the training and test tasks use the same motions, so the IDM might easily learn to infer actions by focusing on the robot's visual differences. Whereas MetaWorld test tasks require novel motion patterns, which means improvements to the IDM might matter more.
- I am also curious about the planning performance of AnimateDiff in the real-robot experiments. Does this experiment simply "distill" the capabilities of the general model into the in-domain model, or can the final improved in-domain model actually surpass the general model?

Understanding the above questions would further strengthen the paper’s arguments. However, I believe the current version is already sufficiently complete to merit acceptance.

### Questions
Please refer to the weakness section. In addition, I have one minor question:
- Besides successful trajectories, have the authors considered using failed rollouts to update the IDM as well? I suspect this could be particularly useful when collecting complete successful trajectories is difficult (e.g., in long-horizon tasks).

### Soundness
3

### Presentation
4

### Contribution
3
