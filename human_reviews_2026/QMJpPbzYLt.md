# Vision-Language Models Unlock Task-Centric Latent Actions

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Latent Action Models (LAMs) have rapidly gained traction as an important component in the pre-training pipelines of leading Vision-Language-Action models. However, they fail when observations contain action-correlated distractors, often encoding noise instead of meaningful latent actions. Humans, on the other hand, can effortlessly distinguish task-relevant motions from irrelevant details in any video given only a brief task description. In this work, we propose to utilize the common-sense reasoning abilities of Vision-Language Models (VLMs) to provide promptable representations, effectively separating controllable changes from the noise in unsupervised way. We use these representations as targets during LAM training and benchmark a wide variety of popular VLMs, revealing substantial variation in the quality of promptable representations as well as their robustness to different prompts and hyperparameters. Interestingly, we find that more recent VLMs may perform worse than older ones. Finally, we show that simply asking VLMs to ignore distractors can substantially improve latent action quality, yielding up to a six-fold increase in downstream success rates on Distracting MetaWorld.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper describes how vision-language models (VLM) representations can be used to train latent-action models (LAM) that are robust to “action-correlated distractors”. Extensive experiments are conducted on a modified version of the MetaWorld benchmark, where the background is artificially modified to include distractors, called Distracting MetaWorld. First, the weakness of current LAM on the task is demonstrated. Then, a study on the best VLMs for the task is conducted. Finally, the effectiveness of the approach is demonstrated by studying success rate on Distracting MetaWorld.

### Strengths
- The robustness of latent-action model to action-correlated perturbation is a fundamental problem that differentiates humans to current systems and that should be explored by the community. This paper attempts to find a minimal setup with artificial perturbations, which is a valid and interesting approach.

- The experimental results provide a clear signal on the proposed task that VLM representations help overcome the challenge of action-correlated perturbations.

- The paper clearly acknowledges several limitations such as the simplicity of the constructed task: Perturbed MetaWorld, and the use of VLM representation as opposed to a semantic segmentation approach.

### Weaknesses
- The clarity of the paper could be improved. First, the notion of promptable representation is not properly introduced, or at least way too far in the paper, in section 3, L123, which might confuse readers not aware of the literature. I would suggest putting a proper definition earlier in the paper. Then, the experimental setup is not formalized properly, the author describes inputs and outputs of their IDM and FDM in section 4 without describing the specific architecture they use. It might be useful to include a Figure summarizing the pipeline. There are other presentation issues: line 107: BC is not defined, finally in Figure 2, 3 a) b), 4), it is very unclear what the y-axis represents; both for people only looking at the figure and reading the captions, and for people reading the text, paragraph “results”, line 183, should be where the metric is explained.

- Although the problem of action-correlated distractors is correctly identified, the specific construction of synthetic distractors proposed by the paper is not fully-convincing and feels a bit too artificial to be convinced that it would transfer to real world problems. Only the background is altered, with completely random and unrealistic patterns. More interesting distractors would have been: occlusion of the action, other robots in the same scenes, humans performing actions in the scene, or lightning parameters such as luminosity.

- I have a fundamental criticism about VLM representations. It is well known that representations from generative models such as LLM/VLM are not very good and it seems odd to push to use them. The paper mentions these representations should “be minimal by filtering out visual details irrelevant to the prompt”. This is exactly what self-supervised learning representations and in particular representation from joint-embedding predictive architectures (JEPA) is trying to achieve, and it should be much better suited compared to VLM representations that have the disadvantages, as shown in the paper, to be sensitive and to rely on good prompts, as shown in the paper: “Ironically, the best prompt is “Do not describe background features. Focus on the robot arm and the [task-obj]”. SSL representations also do not have the disadvantages of segmentation as described in the limitation section.

- It is not clear in this work or similar work on LAM, what is the importance of the labeled trajectories. What if we do not fine-tune on these trajectories ? What if we only train on these trajectories ?

- In conclusion, the paper reasonably answers its own hypothesis, but the setting chosen and specific distractor + the choice of VLMs representation do not convince me that this is the best way to train LAMs.

### Questions
- Paragraph “Why does Molmo perform so well?” explains why the data might be the primary factor that makes good promptable representation, what do you think is the right data to train a VLM in order to get strong promptable representations for LAMs ?

- In line 257 mentions the use of “64 instead of full 5k” trajectories. How reliable is it to use so few trajectories ? Later the paper mentions “we found Phi-4 to outperform GraspMolmo, despite having worse probes on small datasets”, is that the result of validating hyper-parameters on so few trajectories ?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper explores the efficacy of promptable representations from VLMs as reconstruction targets for Latent Action Models (LAM). The author’s key idea is that the common sense reasoning abilities of VLMs can better enable LAMs to learn latent actions corresponding to the controllable changes while filtering out exogenous noise (e.g., scene details, camera motion). The authors explore representations from a wide range of VLMs and perform an extensive ablation study to validate the optimal strategy to obtain promptable representations from VLMs. Performance is evaluated over 20 recent VLMs and 29,000 experiments on the MetaWorld benchmark with distractors, both latent action quality (via linear probing) and downstream task success rates after fine-tuning are measured. The authors find that these promptable representations can effectively improve the performance of LAMs in the presence of distractors.

### Strengths
Leveraging the common-sense reasoning capabilities of VLMs to learn stronger latent actions centered on controllable changes is well motivated and addresses an important challenge facing current LAMs

The paper conducts an extensive empirical study to validate an optimal strategy for extracting proptable representations from VLMs. This study is conducted across a wide range of recent SOTA VLMs

The promptable representations are effective and consistently improve performance over baseline LAMs in scenes with distractors

### Weaknesses
Overall the novelty seems limited to the reviewer. As referenced by the authors, Chen et al. [1] proposed promptable representations, and much of the evaluation setting follows Nikulin et al. [2].

While the motivation is reasonable, the underlying reason why promptable representations lead to such improvements remains unexplored and poorly understood to the reviewer
* It is unclear why promptable representations from VLMs should be preferred to other methods like UniVLA that aim to disentangle task-specific cues and exogenous noise. The authors discuss UniVLA in Section 8, but it was unclear to the reviewer why the noted differences are relevant. Why is the proposed method preferred to UniVLA for learning task-centric latent actions?
* While it is clear that promptable representations improve performance of LAMs, it is not convincing that the improvement stems from the common sense reasoning abilities of VLMs themselves. To the reviewer it seems the improvement can stem from the larger and more diverse pretraining data that VLMs have seen compared to the VQ-VAE used in the baseline. Why would representations from a model like DINOv2 (i.e., a non-promotable model with large pre-training data) be a worse reconstruction target?

The goal of latent action learning is to unlock large scale human videos (e.g., from the web), but there are no experiments (or proof of concepts) showing that promptable representations are effective for learning latent actions from real-world videos. Datasets like EgoDex [3], which contain egocentric actions with annotated hand poses, could be helpful for this validation (as the availability of hand poses enables action probing)

Minor comments on formatting
* Line 72: should be “Latent Action Models (LAM)”
* Line 111: “linear probing” is listed as one of the evaluation metrics, but in the experiments it is reported as action probing (e.g., Figure 2 and Figure 3)
* Figure 2: bolding “ResNet” and “Trans” will prevent confusion on the x-axis label in the first plot
* Figure 5: Currently this figure is a bit difficult to read and there is a lot of repetition. In future versions readability could be improved, for example through assigning letters to specific prompts, or presenting only mean and std instead of the entire box plot

[1] William Chen, Oier Mees, Aviral Kumar, and Sergey Levine. Vision-language models provide promptable representations for reinforcement learning. Arxiv, 2024

[2] Alexander Nikulin, Ilya Zisman, Denis Tarasov, Nikita Lyubaykin, Andrei Polubarov, Igor Kiselev, Vladislav Kurenkov. Latent Action Learning Requires Supervision in the Presence of Distractors. ICML, 2025

[3] Ryan Hoque, Peide Huang, David J. Yoon, Mouli Sivapurapu, Jian Zhang. EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video. Arxiv, 2025

### Questions
(Line 118) Does fixing the latent action dimension to 128 really guarantee minimality? Regardless of the dimension of the latent action, it can still end up encoding noise rather than controllable changes

What distinguishes this work conceptually or experimentally from prior studies like Chen et al. [1] and Nikulin et al. [2]?

an the authors provide analysis or evidence clarifying why promptable representations improve latent action learning performance beyond empirical gains?

### Soundness
2

### Presentation
2

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
In this work, the authors tackle the problem of Latent Action Models capturing noise in noisy environments. Instead of learning the actions related to an agent's movements, the latent actions can focus too much on background noise, rendering them unusable for downstream tasks. To alleviate this, the authors propose to use a VLM to extract the relevant information from the video and use it as a target for LAM training. Exhaustive experiments on VLMs show an increase in performance of the model when distractors are present.

### Strengths
- This work tackles an important problem of disambiguating actions in LAM. Often, considered robotics setups are too simplistic for it to matter, but this is crucial in more complex environments.

- The authors present a clear experiment to demonstrate the targeted problem.

- The experiments across VLMs are exhaustive, both across models and prompt choices.

- Most of the performance compared to the distractor free setting is able to be recovered by the model.

### Weaknesses
1) Saying that the method is “without any supervision” (line 57) is slightly misleading as it relies both on task information and the use of a VLM to extract relevant information. It is however a weaker supervision than previous works using action labels.

2) As pointed out by the authors, the linear probing evaluation does not guarantee the minimality of actions and one could imagine that both the robot action as well as the “noise” are captured. While the proposed bottleneck to 128 dimensions may help (even if using a continuous space), this would need to be measured in some capacity.

3) Section 4: Observations with distractors are used as input to the IDM and FDM. But this means that the FDM has to go from a noisy observation to a clean one. In that case either the latent actions can encode this “noise removal”, or the FDM must focus only on relevant part of the data and filter the noise itself (which may be aided by using the promptable representations). The former is obviously undesirable but this is never discussed or analyzed in the paper.

4) How promptable representations are used should be made clearer. Right now the paper assumes a lot of knowledge of [1]. Perhaps a figure describing the whole process would help make this clearer.

5) Lack of formal comparison to other approaches such as UniVLA. Even if the UniVLA pipeline is more complex, it makes similar assumptions (access to task information) and should be compared to the proposed approach.

6) Why were 128 unconstrained dimensions chosen for the latent? While the VQ used by LAPO has limitations, using it (or another kind of regularization) would help with minimality of latent actions. This minimality can also have an impact on how noise is captured, as it requires significantly more capacity to learn than the robot actions that are common across videos. A proper comparison of behaviors under different latent constraints would help understand the problem better.


[1] Chen, William, et al. "Vision-language models provide promptable representations for reinforcement learning." arXiv preprint arXiv:2402.02651 (2024).

### Questions
1) Does the baseline LAPO also use the 128 continuous dimensions ? or is it using the original VQ ?

2) How much do you think that training in pixel space or with a small encoder is making the problem of distractors more prevalent ?

3) If promptable representations are better at filtering out noise, could they be used for everything? The VLM could be used as a general encoder for both the IDM, FDM, and FDM target


Typo line 309 : conveneince -> convenience

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes using promptable representations from Vision-Language Models (VLMs) as a clean training target for Latent Action Models (LAMs), enabling them to learn task-centric latent actions from videos containing action-correlated distractors, without requiring ground-truth action labels.

### Strengths
+ Well-Motivated Solution: Directly addresses a known failure mode of LAMs.

+ The "twin observation" experiment is a powerful motivator, and the large-scale VLM benchmark provides strong, data-driven insights.

+ Experimental results demonstrates a substantial improvement in success rates on the Distracting MetaWorld benchmark.

### Weaknesses
- The core concept of using VLM embeddings as representations for control was previously explored by Chen et al. [9] and Huang et al. [24]. This work applies a similar idea to a different, albeit important, problem (LAM robustness).

- The benchmarking is almost exclusively against the baseline LAPO [41] and its own variants. It lacks comparison to other contemporary LAMs (e.g., UniVLA [8]) or alternative distractor-handling techniques, limiting the claim to a broader state-of-the-art.

- The analysis is heavily empirical and lacks theoretical grounding. It lacks a theoretical framework or in-depth analysis (e.g., of the embedding space) to explain why the method works beyond the initial "twin" experiment.

### Questions
- How does the method compare to other proposed solutions for robust LAM training, such as UniVLA, on larger-scale benchmarks?

- Beyond pre-training data, what specific properties of the VLM's representation space are critical for its success, and can this be analyzed more formally?

### Soundness
3

### Presentation
3

### Contribution
3
