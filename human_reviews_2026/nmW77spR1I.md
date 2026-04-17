# Vision-Language-Action Pretraining from Large-Scale Human Videos

- Decision: Reject
- Scores: 2, 6, 6, 8

## Abstract
Existing Vision-Language-Action models (VLA) struggle with complex manipulation tasks requiring high dexterity and generalization, primarily due to their reliance on synthetic data with significant sim-to-real gaps or limited teleoperated demonstrations.
To address this bottleneck, we propose leveraging human hands as a ``manipulator template'', capitalizing on the rich dexterity and scalability present in web data of human manipulation.
Our approach centers on physical instruction tuning, a novel training paradigm that combines large-scale VLA pretraining from human videos, perspective spatial alignment for reasoning in a unified physical
space, and post-training adaptation in physical environment.
Additionally, we introduce a part-level motion tokenization method which achieves millimeter-level reconstruction accuracy to model precise hand trajectories for action learning. 
To support our paradigm, we develop a comprehensive data curation pipeline that integrates heterogeneous sources --- including motion capture, VR, and RGB-only videos --- into a large-scale dataset with millions of motion-based instructional instances.
We empirically show the excellence of our model in hand motion generation and instruction following, and it also scales well with model and data sizes.
Importantly, we observe the expected gains in robotic dexterous manipulation as physical instruction tuning is applied.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes Being-H, a dexterous Vision-Language-Action (VLA) model trained through Physical Instruction Tuning, which integrates large-scale human video data with explicit 3D hand motion modeling. The approach leverages a newly curated dataset (UniHand) and introduces a part-level motion tokenizer for millimeter-level precision.

### Strengths
- [S1] Physical Instruction Tuning effectively extends visual instruction tuning to the physical domain, an interesting idea that unifies VLA, physical space alignment, and post-training adaptation for robotic tasks.
- [S2] The authors conduct comprehensive analysis, including quantitative and qualitative experiments (simulation and real-robot tasks).

### Weaknesses
- [W1] The manuscript, in its current form, lacks sufficient clarity and is not ready for publication. The inconsistent notation and disorganized presentation of formulas significantly hinder understanding. Substantial revision and careful polishing are required to improve the logical flow and overall readability. I also have several specific questions regarding unclear points and expect the authors to address them thoroughly.

- [W2] Scalability limitations of Physical Instruction Tuning. Although the authors frame their method as scalable, it still depends on datasets with explicit 3D hand motion annotations, which are costly and non-trivial to obtain. This reliance contradicts the claimed scalability and restricts the applicability of the approach to specialized domains where such annotations exist.

- [W3] Complex pipeline with modest gain. The training and alignment processes (including GRQ-based motion tokenization and physical space alignment) add substantial system complexity, yet the improvements over strong baselines are relatively modest compared to existing VLA model, such as GR00T.

### Questions
- [Q1] What are $\mathcal{X}_Q$ and $y_i$? How are they related to motions $\mathbf{m}$?
- [Q2] Do $\mathbf{m}$ and $m$ indicate the same thing?
- [Q3] What is the exact form of $\mathcal{L}_{\text{recon}}$ in Equation 3? 
- [Q4]  What is the exact form of $\mathcal{L}_{\text{commit}}$ in Equation 3? The manuscript does not describe the details about this objective.

### Soundness
2

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
5

### Summary
This paper proposes a paradigm that leverages large-scale human videos for VLA pretraining. The core method involves pretraining a model to generate detailed human hand motions from videos, and then adapting it to control a robot hand via post-training. The pretrained VLA successfully transfers human dexterity from internet videos to robots, demonstrating superior performance and data efficiency on real-world manipulation tasks compared to prior methods.

### Strengths
This paper's strength is its well-motivated idea to leverage human videos as a scalable resource for robot manipulation, a novel pretraining pipeline featuring part-level motion tokenization for high-fidelity control, and demonstrated strong performance with superior data efficiency on dexterous real-world tasks.

### Weaknesses
1. The approach is heavily reliant on the MANO hand model, which may not perfectly capture the full complexity and contact dynamics of real-world manipulation, potentially limiting the fidelity of the transferred skills. I suggest having some analytical experiments to assess the impact of potential errors in MANO on the entire pipeline.

2. The chosen tasks in the experiment section are not uniquely dependent on dexterous hands and could largely be accomplished with parallel grippers. The paper does not include functional grasping and in-hand manipulation tasks (e.g., reorienting a pen, spinning a key, or precise tool use) that would rigorously demonstrate the necessity of a dexterous hand and the specific advantages of the learned fine-grained finger control. This narrow task scope limits the claim of achieving general dexterous manipulation.

I would like to improve my score if the authors address my concerns during rebuttal.

### Questions
Please refer to the Weaknesses part.

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
This paper introduces Being-H, a large-scale vision-language-action (VLA) pretraining framework for dexterous robot manipulation.
The key idea is to treat human hand motion as a transferable prior for dexterous robot. Being-H first trains a transformer model on 2.5 M human video–text–motion triplets (the proposed UniHand-2.5M dataset) using MANO-based 3D hand parameters tokenized via a part-level grouped residual quantizer (GRQ). The pretrained model aligns vision, language, and hand motion tokens and is later adapted to robots through a lightweight projection and regression head for robot action prediction. Evaluations show that Being-H outperforms prior VLA baselines such as GR00T and InternVL3 on hand motion modeling, RoboCasa simulated tasks, and real-world dexterous manipulation, achieving higher success rates with only a fraction of teleoperation data.

### Strengths
1. Leveraging human-hand motion as pretraining data for dexterous robot control is a promising idea that directly targets the challenge of transferring human manipulation priors to robotic hands.

2. The work conducts large-scale multimodal pretraining on millions of human video–text–motion pairs and demonstrates clear empirical gains on hand motion modeling, simulated and real-world dexterous manipulation experiments.

3. The paper provides thorough experiments and analyses for the pretraining stage, including ablations on quantization design, model scale, and data efficiency, which make the technical contribution solid and well-validated.

### Weaknesses
1. The discussion of the embodiment gap between human and robot hands is not sufficiently clear. It remains unclear what specific aspects of human-hand pretraining help dexterous robot control, how large the embodiment gap actually is, and under what conditions the transfer succeeds or fails. The current framework behaves largely as a black box that relies on large-scale data to yield useful priors, without a mechanistic explanation.

2. The real-world evaluation appears incomplete and under-documented. For example, in Table 4, it is unclear how many trials were used to compute the success rate and how post-training performance scales with different amounts of robot data (e.g., 10, 50, or 100 demonstrations). It would also be important to compare against strong imitation-learning baselines such as Diffusion Policy, not just VLA-style models. Moreover, the paper does not provide any real-robot videos in the supplementary material, which makes it difficult to assess qualitative behavior or reproducibility.

### Questions
The paper appears to focus only on hand-centric representation learning and is applied to dexterous robotic hands. It would be helpful to clarify this scope explicitly in the title, so readers understand that the method is focusing on (dexterous) hand.

In Table 3, the performance on the Insertion task seems higher for GR00T than for Being-H.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a dexterous VLA training framework from large-scale human hand manipulation videos. It introduces a hand tokenization to tokenize the human hand for pretraining. Next, a post-training on the detextrous hand data is adapted. To support the training, a large-scale dataset is introduced. The experiments are conducted in different tasks.

### Strengths
1. The paper is well-written. The reviewer enjoys reading the introduction part of the paper, where some substantial challenges are discussed. 
2. The whole training pipeline is well-designed, insightful and reasonable. 
3. The dataset contribution is a great bonus, which is meaningful to the community. 
4. The completeness of the paper is very good, with sufficient experiments (including real-world experiments) and supplementary material.

### Weaknesses
1. Based on the knowledge of the reviewer, there is still a large embodiment gap between human hands and robot manipulators, especially when the grasping ways or contact points are distinct between human and robot manipulators. In these extreme cases, does the proposed framework, especially the proposed hand motion representation/tokenizer, still work well?  
2. Beyond the physical adaptation from the spatial perspective, is it possible to add a physical adaptation module considering the embodiement transfer (from human hand representation to robot manipulator representation) during post-training, to improve the better utilization of the pretrained priors?
3. Some failure case analysis and limitations should be discussed for future work.   
4. The so-called "*physical space* alignment for 3D reasoning" or "physical tuning" is much overclaimed, as "physical" generally means a lot (not just depth or camera, also including reflection, force, material, dynamics, and so on). However, the proposed alignment is substantially a data distribution normalization/alignment for better pretraining. This should be addressed in the final version.  
5. The ablation of post-training, such as scaling up the post-training data, using different post-training datasets (different gaps with pretraining dataset, also evaluate if can generalize to different post-training dataset) or networks, is not provided, which is similarly important for providing a clearer picture of the proposed framework.

### Questions
The reviewer expects some discussions about the questions raised in the weaknesses, and is glad to keep the rating if they are well-discussed.

### Soundness
3

### Presentation
4

### Contribution
3
