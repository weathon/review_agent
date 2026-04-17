# EquAct: An SE(3)-Equivariant Multi-Task Transformer for 3D Robotic Manipulation

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 10

## Abstract
Multi-task manipulation policy often builds on transformer's ability to jointly process language instructions and 3D observations in a shared embedding space. However, real-world tasks frequently require robots to generalize to novel 3D object poses. Policies based on shared embedding break geometric consistency and struggle in 3D generation. To address this issue, we propose EquAct, which is theoretically guaranteed to generalize to novel 3D scene transformations by leveraging SE(3) equivariance shared across both language, observations, and action. EquAct makes two key contributions: (1) an efficient SE(3)-equivariant point cloud-based U-net with spherical Fourier features for policy reasoning, and (2) SE(3)-invariant Feature-wise Linear Modulation (iFiLM) layers for language conditioning. Finally, EquAct demonstrates strong spatial generalization ability and achieves state-of-the-art across $18$ RLBench tasks with both SE(3) and SE(2) scene perturbations, different amounts of training data, and on $4$ physical tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a multi-task SE(3) equivaraint robot policies.  The model consists of (1) a SE(3)-equivariant encoder that transforms input point cloud into spherical Fourier features, followed by (2) SE(3)-invariant feature-wise linear modulation layer that facilitates conditioning on task instructions.  The proposed method sets a new state-of-the-art on RLBench simulation benchmark, while demonstrating strong generalization to spatial variety.

### Strengths
1. The proposed method sets a new state-of-the-art on the challenging RLBench simulation benchmark.  In particular, it demonstrates superior performance on testing scenes where objects are randomly SE(3) initialized.
2. The design choice of using SE(3)-equivariant point cloud encoder is techinically sound.  Its efficacy is also supported by the ablation study.
3. This paper reports both training time, inference speed and required GPU mem, facilitating better understanding of the proposed method.
4. The authors have released the code base, facilitating reproducibility of the proposed method.

### Weaknesses
1. This paper only evaluates the proposed method on one RLBench simulation benchmark and a real-world benchmark.  I'd highly encourage the authors to evaluate on other benchmarks, such as SimplerEnv [1] or LIBERO [2] or CALVIN [3], to further demonstrate the generality and effectiveness of the proposed method .
2. It's widely known that 3DDA has poor sim2real generalization due to the visual appearance gap between the simulation and the real world.  This paper uses a SE(3)-equivariant point cloud encoder, and point cloud has stronger sim2real generalization than RGB image encoders.  Along with the SE(3)-equivariance, it'd be interesting to see if the proposed method trained in simulation can be directly deployed in the real world.  Such results would intrigue the community and attract more attention in SE(3)-equivariant policies.

---

Reference

[1] Li, Xuanlin, et al. "Evaluating real-world robot manipulation policies in simulation." arXiv preprint arXiv:2405.05941 (2024).

[2] Liu, Bo, et al. "Libero: Benchmarking knowledge transfer for lifelong robot learning." Advances in Neural Information Processing Systems 36 (2023): 44776-44791.

[3] Mees, Oier, et al. "Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks." IEEE Robotics and Automation Letters 7.3 (2022): 7327-7334.

### Questions
1. What is the performance of EquAct on SimplerEnv or LIBERO or CALVIN simulation benchmark?
2. How does the proposed method work in sim2real setup?

I'm open to adjust my rating based on the author's rebuttal.

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
4

### Summary
The paper proposes a SE(3)-equivariant multi-task policy network with several architectural designs: (i) an equivariant U-net, (ii) an invariant FiLM layer, and (iii) an equivariant field network. The equivariance is mathematically proven. The network shows good performance in both simulation and real-world experiments.

### Strengths
- Multi-task learning with generalization is a much harder problem than per-task learning, while most prior works with equivariance could only tackle the latter.
- The proposed network has both theoretical foundations to guarantee its equivariance and engineering designs to improve its performance and efficiency.
- The experiments are well-designed and well-conducted. Especially some small but important details are well-studied in the experiments:
	- I appreciate that the authors also show the training/inference time and memory consumption for the proposed network, and it's good that the results match the baselines -- usually, equivariant networks can easily consume more time or memory as a cost for their "for-free" generalization to unseen poses
	- The robustness and empirical equivariance errors are tested.

### Weaknesses
- Some details, especially the experimental setups, can be made clearer. See the minor points and questions below.


Some minor points:
- Line 157: $x = \{x_T, x_{\text{open}\}, x=\{e, a\}$ looks confusing at first glance -- I understood it after a while, but I think it can be expressed in a clearer way.
- Table 1 is a bit hard to read at first, and it took me a while to understand its configuration: (i) The two small regions for the avg. success rate and time/memory consumptions, perhaps mark them in different background colors, so that they look more distinct from the per-task success rates. (ii) "10* 10 100" looks too abbreviated for training setups, maybe make it a bit more concrete, like "SE(2)/10" or something, and I think the table still has some space for that.
- Perhaps say "real-world experiments" instead of "physical tasks"? It made me confused before reading Appendix F -- what are "physical tasks"? force control? physical parameter related? ...

### Questions
- How are the query actions sampled when training the equivariant field network? Is it a uniform sampling? Is the sampling space too large: rotation + translation, in a large scene?
- In Section 4, it says: "During training, we also augment the dataset with respect to equation 1 by randomly rotating the
point cloud and the action simultaneously with [±5◦,±5◦,±45◦] rotation along [x, y, z] axis." How does it relate to the SE(3)/SE(2) randomizations in Section 5.1 Experiment settings (line 409-414)? Also, why are these numbers chosen?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes EquAct, an SE(3)-equivariant multi-task Transformer for open-loop robotic manipulation.
The method introduces explicit geometric inductive bias into 3D point cloud and language-conditioned policy learning, aiming to improve generalization and sample efficiency under spatial perturbations.
The model consists of three main components: (1) an SE(3)-equivariant Point Transformer U-Net for geometric encoding, (2) an Invariant Feature-wise Linear Modulation (iFiLM) layer for language conditioning, and (3) an Equivariant Field Network for action candidate evaluation.
Experiments on 18 RLBench tasks and several real-robot setups demonstrate clear advantages over non-equivariant baselines under SE(3)/SE(2) perturbations. The paper also provides quantitative analyses of equivariance error.

### Strengths
(1)The work systematically integrates explicit SE(3) equivariance into a multi-task vision-language manipulation Transformer with modular and theoretically grounded design.
(2)The introduction of SE(3) initialization in RLBench contributes an independent and valuable benchmark for studying spatial generalization.
(3)The method achieves strong performance under SE(3)/SE(2) perturbations, demonstrating the benefits of geometric inductive bias in low-data regimes.

### Weaknesses
(1)Lack of comparison with other equivariant models. The paper only compares with non-equivariant baselines, leaving unclear whether EquAct’s design offers advantages within the broader class of equivariant inductive biases.
(2)It is unclear whether explicit SE(3) constraints remain beneficial when data scale or pretraining increases, or if the performance gains would saturate.
(3)In real robotic systems, camera pose variations, viewpoint shifts, and lens distortions break strict SE(3) symmetry. The model’s robustness to such non-ideal conditions has not been examined.

### Questions
1. Could the authors include comparisons or analyses with other SE(3)-equivariant models such as [1][2]? This is not mandatory, but discussing why previous methods are limited to single-task settings while EquAct handles multi-task scenarios would clarify whether this advantage arises from the keyframe-based formulation rather than the equivariance design itself.
2. When scaling to larger datasets, does the model still rely on explicit equivariance for good performance? Are there signs of saturation?
3. In real-world conditions, camera viewpoint shifts and optical distortions are common. It would be valuable to include experiments testing robustness under such non-ideal equivariance scenarios.
If the authors can effectively address these concerns, I would consider raising the score.
References
[1] Tie, Chenrui, et al. “Et-seed: Efficient trajectory-level SE(3) equivariant diffusion policy.” arXiv preprint arXiv:2411.03990 (2024).
[2] Qi, Yu, et al. “Two by two: Learning multi-task pairwise objects assembly for generalizable robot manipulation.” CVPR 2025.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
The paper proposes a novel approach for adapting visuomotor policies to be SE(3) equivariant to scene changes and SE(3) invaraiant to similar language commands. This is achieved by using an SE(3)-equivariant point transformer that adapts to changes in the observation space, while having an SE(3)-invariant FiLM layers to adapt (or consequently not adapt) to changes in the language commands. They authors show the improved performance of the proposed approach for learning a next-best keyframe manipulation policy in a multi-task imitation learning setting.

### Strengths
The proposed approach does a good job at ensuring SE(3) equivariance/invariance through different parts of the system, such as the equivariant transformer arhitechture for encoding the points, and the Field network. The proposed novelties are experimentally evaluated and show good performance especially under high SE(3) variability with just a low number of samples (10 demos) and scales well with more number of demonstrations.

### Weaknesses
The policy architecture and the corresponding action refinement/selection is unclear. How is the implicit action value function used to predict the actions? Is it by randomly sampling actions and evaluating them? Or is it by having a separate policy network trained in an actor-critic approach? 
Training details about the implementation of the cross entropy losses are missing to make the paper more complete and to improve the reproducibility aspects of the paper.
Some discussion on not using direct behavioural cloning instead of an implicit action-value function would benefit the paper.

### Questions
In Lines 201-202, does the Q function take in a single action sample? Why does the gripper Q function take in only the translational action and not the rotational one as well? Moreover, how are the cross-entropy losses calculated?
The SO(3) Perturbation (r, p) is unclear in Table 4. are they the deviations along XYZ and roll pitch yaw? if so, should it not be SE(3) Perturbations in that case?

### Soundness
4

### Presentation
4

### Contribution
4
