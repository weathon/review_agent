# MindPilot: Closed-loop Visual Stimulation Optimization for Brain Modulation with EEG-guided Diffusion

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Whereas most brain–computer interface research has focused on decoding neural signals into behavior or intent, the reverse challenge—using controlled stimuli to steer brain activity—remains far less understood, particularly in the visual domain. However, designing images that consistently elicit desired neural responses is difficult: subjective states lack clear quantitative measures, and EEG feedback is both noisy and non-differentiable. We introduce MindPilot, the first closed-loop framework that uses EEG signals as optimization feedback to guide naturalistic image generation. Unlike prior work limited to invasive settings or low-level flicker stimuli, MindPilot leverages non-invasive EEG with natural images, treating the brain as a black-box function and employing a pseudo-model guidance mechanism to iteratively refine images without requiring explicit rewards or gradients. We validate MindPilot in both simulation and human experiments, demonstrating (i) efficient retrieval of semantic targets, (ii) closed-loop optimization of EEG features, and (iii) human-subject validations in mental matching and emotion regulation tasks. Our results establish the feasibility of EEG-guided image synthesis and open new avenues for non-invasive closed-loop brain modulation, bidirectional brain–computer interfaces, and neural signal–guided generative modeling.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents MindPilot, an innovative closed-loop EEG-guided visual stimulus optimization framework.The core concept of this framework is to treat the human brain as an indivisible black-box system, iteratively guiding diffusion models to generate images through EEG feedback signals.This framework introduces a pseudo-model to provide surrogate gradients, enabling gradient-free optimization for various neural objectives such as semantic or spectral EEG features.The paper validates MindPilot through agent-based model simulations and human experiments, demonstrating its potential in EEG-guided image retrieval, generation, and real-time human brain control.

### Strengths
Novelty: Innovation: First to propose a closed-loop image generation framework based on EEG signal diffusion models.

Technical soundness: Technical Rationality: The pseudo-model guidance mechanism is clearly designed and effectively replaces explicit gradient computation.

Experimental breadth: Demonstrates stable convergence characteristics in both simulated and human experiments.

Potential impact: Openes up new avenues for non-invasive neural modulation and human-machine collaborative adaptation studies.

### Weaknesses
EEG Individual Variability: Insufficient discussion of inter-subject EEG variability; a brief explanation could be added to emphasize the model's adaptability and generalization capability.

### Questions
1.  Could the authors elaborate on whether MindPilot could be generalized to other modalities (e.g. fMRI)?

2. How do pseudo-model guidance and traditional gradient-free reinforcement learning methods differ in their convergence properties?

3. Additionally, there are minor inconsistencies in the formatting of references within the text. For example, several NeurIPS/ICLR-style citations only provide the conference name without volume, issue, or page information (e.g., Bashivan et al., 2019; Black et al., 2024; Luo et al., 2024a). Similarly, the citation for “Transactions on Machine Learning Research, 2024” (Oquab et al., 2024) omits both volume and page numbers. We recommend that the authors carefully review these entries and align them with the journal article format, ensuring that volume, issue, and page numbers are included where available.

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
4

### Summary
he paper presents MindPilot, an approach to control image generation from EEG-BCI. The area is under active research, and the present paper is a continuation of previous work.

### Strengths
Strengths:

- The paper tackles a challenging and interesting problem.
- The paper is well-written and aims for rigorous experimentation.

### Weaknesses
Weaknesses:

- The paper claims to be the first to tackle the problem, but the principles of the approach have been extensively explored in the last years: 

Many core-related works are missing, which hinders the novelty statement of the papers. Examples (not exhaustive list) of a few recent references of well-known papers and some even using similar CLIP approaches, generative modeling, and diffusion processes:

https://proceedings.neurips.cc/paper_files/paper/2024/hash/84bad835faaf48f24d990072bb5b80ee-Abstract-Conference.html
https://ieeexplore.ieee.org/abstract/document/10798967
https://arxiv.org/abs/2306.16934
https://arxiv.org/abs/2506.11151
https://dl.acm.org/doi/abs/10.1145/3379337.3415821
https://openaccess.thecvf.com/content/CVPR2022/html/Davis_Brain-Supervised_Image_Editing_CVPR_2022_paper.html
https://arxiv.org/abs/2308.02510
https://www.nature.com/articles/s42003-025-07731-7
https://dl.acm.org/doi/full/10.1145/3716553.3750786
https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html
https://arxiv.org/abs/2508.20705

- The reward and spreading operations are somewhat ad-hoc. I understand that the reward model cannot directly find the best matches, but this makes the robustness and replication of the approach problematic as the connection across the models is not straightforward. Figure 3d also shows that the alignment task is not easy and the results are not clearly demonstrating that simple CLIP with the underlying ad-hoc process is suitable.

- There are good attempts to evalute the approach. 

- It is known that EEG does not generally contain fine-grained semantic codes; it means that the pattern of responses across time and sensors partially aligns with the structure of semantic space for their geometry. Thus, I feel that some of the wordings in the paper are not in line with this general understanding.

- The closed loop experiment has results that do not support the message of the paper (Fig 6 C&D). In fact, when tested  “in-vivo” the model does not seem to perform.  This is also problematic as the paper claims novelty in closed-loop, but does not demonstrate that convincingly.

Minors:

- I don’t understand why it is highlighted that a brain is non-differentiable. I think we do not know the exact mechanisms of learning in the biological/cellular system, especially when making such simplified mathematical statements for analyzing EEG.  There is emerging evidence (https://academic.oup.com/pnasnexus/article/3/7/pgae261/7702306), but I am not sure how meaningful this is for the submitted paper.

- Closed-loop is only a small portion of the paper, while most of the paper relies on a single existing dataset.

### Questions
1. I would like to stress novelty compared to many previous works that are "not closed loop" -- as I think the present paper also has very limited closed-loop contribution

2. I think the results in the most critical parts of the paper are weak and the authors are trying to oversell them.

### Soundness
3

### Presentation
4

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
This paper focuses on a novel problem: given neural activity recordings evoked by a visual stimulus (e.g., EEG), how can we find or generate the most likely visual stimulus? I believe this problem has potential value for related research in neuroscience. In this work, the authors first propose an iterative optimization method that progressively updates the probability of each image in an image database by computing the correlation between EEG representations and the images, thereby identifying the most likely visual stimulus. Subsequently, the paper introduces an EEG-guided image generation model capable of attempting to generate the possible visual stimulus. The authors conducted corresponding evaluations, and I believe the evaluation is also sufficient.

### Strengths
1. The paper addresses a research problem that I believe has considerable practical value.

2. According to the authors’ evaluation, the method proposed in this paper is effective.

3. To facilitate a better evaluation, the authors conducted a visual rating experiment with human participants and demonstrated that the results obtained by their method are highly correlated with the ground truth provided by the participants.

4. The evaluation in the paper is thorough.

5. The paper is well-constructed, with figures and formatting presented nicely.

### Weaknesses
This paper has some limitations, but I think the authors have provided a thorough discussion of them in the “Limitations” section on page 9.

### Questions
1. Normally, better visual encoders and representation learning strategies are expected to learn more accurate and robust visual representations. However, according to the results in Table 1, representations obtained from ResNet trained on image classification tasks perform the best. What could be the reason for this?

2. Continuing from the above question, I have noticed a paper [1] emphasizing that visual representations obtained through contrastive learning are more consistent with human brain activity. Although that study focused on fMRI data, why does a different conclusion appear when it comes to EEG signals?

3. How should we understand ATM-S as the performance upper bound for image generation in Table 2?

4. In Figure 4C, what do the images in each row represent?

[1] Better models of human high-level visual cortex emerge from natural language supervision with a large and diverse dataset. Nature Machine Intelligence.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes MindPilot: a closed-loop visual stimulation optimization framework based on EEG feedback. It shows a participant a batch of images, records their EEG, and calculates a score indicating "how similar" their brain state is to a target state.

### Strengths
Novel and significant problem: Closed-loop visual modulation using non-invasive EEG, advancing towards "neural feedback-guided generative modeling."

Practical implementation: Gradient-free black-box guidance + simple score propagation + roulette wheel sampling, resulting in low engineering cost.

Multi-level validation: Proxy simulation (Tables 1, 3), dual targets of semantics and PSD (Figs. 3–5), small-scale real-time human closed-loop experiments (Fig. 6), and the provision of anonymous code.

### Weaknesses
1. Motivation and advantages of the "pseudo-model" need highlighting: The current discussion is insufficient regarding why Gaussian Process was chosen as the pseudo-model over other black-box optimizers (e.g., Bayesian Optimization). It is recommended to add a brief discussion or comparative experiment in the main text or appendix to explain the rationale for selecting GP and its advantages relative to other methods.


2. The hyperparameters α and β in Eqs. (3) and (4) are set as fixed values, but their specific chosen values or selection criteria are not reported, nor is there any sensitivity/ablation analysis. It is recommended to supplement the specific values, search ranges, and their impact on performance in the main text or appendix. Furthermore, the ambiguity caused by using the same notation as the crossover/mutation ratios in §4.3 should be clarified (or the symbols should be changed).


3. The core premise of the paper is that the proxy model can substitute for the real human brain in closed-loop optimization. However, the data in Table 1 severely weaken this premise: the maximum Pearson correlation between any proxy model and real EEG is only ~0.17. This value is too low. Successful optimization demonstrated on the proxy model does not necessarily generalize to the real human brain. The authors are advised to provide evidence demonstrating a systematic link between the optimization trends observed in the proxy model and those in the real human brain.


4. Figure captions could be more detailed. For instance, how exactly is the ordinate "EEG semantic feature similarity" in Fig.3.D calculated? What do the bold and underlining in Table 1 signify? The authors are recommended to briefly explain this.


5. The current experiments cannot distinguish whether the success of the closed-loop optimization stems from the unique neural information in the EEG or from visual information related to CLIP features that the proxy model learned from the images. To conclusively prove that EEG drives the optimization, the following control experiment is essential: A CLIP-only baseline: Remove the EEG feedback and directly optimize the image similarity to the target within the CLIP space. This baseline serves to quantify MindPilot's performance gain over pure visual semantic optimization.

### Questions
Human experiment details need more transparency. The paper mentions excluding 4 participants due to poor data quality but does not specify the exact exclusion criteria (e.g., artifact proportion exceeding a certain threshold). It is recommended to clearly state the data quality threshold criteria to enhance the experiment's reproducibility and rigor.

### Soundness
3

### Presentation
3

### Contribution
4
