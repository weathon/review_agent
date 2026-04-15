# GeneOH Diffusion: Towards Generalizable Hand-Object Interaction Denoising via Denoising Diffusion

- Decision: Accept (poster)
- Scores: 6, 6, 8, 6

## Abstract
In this work, we tackle the challenging problem of denoising hand-object interactions (HOI). Given an erroneous interaction sequence, the objective is to refine the incorrect hand trajectory to remove interaction artifacts for a perceptually realistic sequence.  This challenge involves intricate interaction noise, including unnatural hand poses and incorrect hand-object relations, alongside the necessity for robust generalization to new interactions and diverse noise patterns. We tackle those challenges through a novel approach, GeneOH Diffusion, incorporating two key designs: an innovative contact-centric HOI representation named GeneOH and a new domain-generalizable denoising scheme. The contact-centric representation GeneOH informatively parameterizes the HOI process, facilitating enhanced generalization across various HOI scenarios. The new denoising scheme consists of a canonical denoising model trained to project noisy data samples from a whitened noise space to a clean data manifold and a ``denoising via diffusion'' strategy which can handle input trajectories with various noise patterns by first diffusing them to align with the whitened noise space and cleaning via the canonical denoiser. Extensive experiments on four benchmarks with significant domain variations demonstrate the superior effectiveness of our method. GeneOH Diffusion also shows promise for various downstream applications. We include [a website](https://meowuu7.github.io/GeneOH-Diffusion/) for introducing the work.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper tackles the problem of denoising HOI in images. The proposed method, GeneOH Diffusion, uses a novel approach that includes a contact-centric HOI representation named GeneOH and a new domain-generalizable denoising scheme. The contact-centric representation GeneOH parameterizes the HOI process, facilitating enhanced generalization across various HOI scenarios. The new denoising scheme consists of a canonical denoising model trained to project noisy data samples from a whitened noise space to a clean data manifold and a denoising via diffusion. The experiments on four benchmarks show the superior effectiveness of the proposed method.

### Strengths
* GeneOH that encodes hand trajectories, hand-object spatial relations, and hand-object temporal relations to faithfully capture the interaction process.
* Extensive experiments on four benchmarks with significant domain variations show the effectiveness of the method.
* Codes are provided in the supplementary materials which is good for reproduction.

### Weaknesses
* The framework is complicated with three denoising stages. Complexity and running time discussion is missing in this paper.

* Comparison and discussion with more recent works are missing, e.g. reference [1] is also a diffusion based HOI method.

* Whitened noise space is constructed by diffusing the training data towards a random Gaussian noise. However, this technique is a common in diffusion model based applications.

* No user studies or applications presented to demonstrate the practical utility of the denoised results. Quantitative metrics alone do not prove the outputs are naturally refined.


[1] Diffusion-Guided Reconstruction of Everyday Hand-Object Interaction Clips

### Questions
See the weaknesses part.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper tackles a complex issue - how to remove errors in hand-object interactions to make interaction sequences appear more realistic. This problem involves challenges such as unnatural hand positions, incorrect hand-object relationships, and the need to work well in various interaction scenarios and noise patterns. To address these challenges, the author proposes the framework called GeneOH Diffusion, which includes two key components: GeneOH, a method for representing interactions, and a denoising scheme that works well across different scenarios. The experiments show that the method outperforms others in various tests, suggesting its potential for a wide range of applications.

### Strengths
- The writing of this paper is quite good. 
- The motivation is clear and the HOI denoising framework is well-designed.
- The experiments are sufficient, where the framework shows promise for novel HOI scenarios.

### Weaknesses
I don’t have a big concern with this article since I’m not directly an expert in this area.

### Questions
N/A

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
- The authors introduce GeneOH Diffusion as a solution to address the challenge of denoising generalizable HOIs.
- The authors develop GeneOH, a novel HOI representation that enhances generalization capabilities.
- The authors propose a canonical denoising model for domain-generalizable denoising and a 3-stage progressive approach for HOI denoising.
- This framework holds significant value for various downstream tasks. Experimental results on four test sets showcase the generalization ability of GeneOH Diffusion.

### Strengths
- On the technical side, the paper successfully employs diffusion models to achieve domain-generalizable denoising of hand-object interaction.
- The progressive execution of denoising through multiple stages, as mentioned in the article, has merit and aligns with the representation capabilities of GeneOH.
- The author offers a clear definition of a rational hand-object interaction trajectory.
- The paper additionally conducts a comprehensive comparison with various methods using multiple test datasets.

### Weaknesses
- The description part of GeneOH can also be further optimized. It is recommended to begin with a summary of the description, including the considered features and the composition of the items. Subsequently, the symbol description can be expanded upon. It is advisable to avoid delving into excessive detail at the initial stages.
- The authors provide experimental evidence demonstrating the generalizability of GeneOH Diffusion to the denoising of hand-object interaction sequences captured by real sensors. However, the majority of the experiments focus on adding known distribution noise to clean ground truth (GT) data and subsequently denoising it. This approach falls short in adequately analyzing the distinction between noise in real hand-object interaction trajectories and artificial noise, as well as evaluating the model's performance in real-world "in-the-wild" scenarios.

### Questions
- Some typos: Fig.3 caption: denoosing -> denoising
- Why is it necessary to optimize $J^{stage2}$ using $\mathcal{T}$ to obtain $J^{stage3}$ in the TemporalDiff stage instead of obtaining it through denoising? Have you conducted any ablation experiments to investigate this?
- The authors' ablation experiments demonstrate that removing a certain stage of the diffusion model does not lead to a deterioration in the overall results in terms of IV or Penetration Depth. However, there is a significant degradation observed in HO Motion Consistency. The reviewer requests an explanation from the authors regarding this phenomenon.
- Please consider citing: "H2O: Two Hands Manipulating Objects for First Person Interaction Recognition" in ICCV2021.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a HOI framework with a strong denoising capability to refine the hand trajectories in interaction sequences. The proposed GeneOH Diffusion method has two main key parts, one is a contact-centric HOI representation called GeneOH and the other is a domain-generalizable denoising scheme. Qualitative and Quantitative results show the methods outperform previous methods.

### Strengths
- The proposed method has strong spatial and temporal denoising capability and it generalizes well to novel HOI scenes. Other than existing methods, this paper explicitly parameterizes both the spatial and temporal relations between hands and objects.
- The proposed progressive HOI denoising process is designed to use a separate stage to learn hand trajectory, and hand-object spatial and temporal relations collaboratively.
- The proposed method outperforms the previous method TOCH on all 4 test sets by a large margin.

### Weaknesses
- In Fig 5 last column middle row, when the hand manipulating the scissors, the grasp does not make sense and the hand shown is penetrated into the scissors.
- The method has the assumption that it requires accurate object pose trajectory to work.

### Questions
- In Sec 3.2 Fitting for a hand mesh trajectory, the method denoises hand key points in the diffusion process and in the end optimizes to MANO parameters. How does the method ensure there is no penetration after going from keypoints to MANO mesh?
- In Sec 4.1 Training dataset, why did the authors use different standard deviations to noise the MANO parameters for translation, rotation, and pose parameters Does the way to noise the training sets affect the learning?
- In Sec 4.1 Evaluation dataset, what is the reason to use different noise on different test sets? Hope the authors give more explanation about it.

### Soundness
3 good

### Presentation
3 good

### Contribution
4 excellent
