# InfBaGel: Human-Object-Scene Interaction Generation with Dynamic Perception and Iterative Refinement

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Human–object–scene interactions (HOSI) generation has broad applications in embodied AI, simulation, and animation. Unlike human–object interaction (HOI) and human–scene interaction (HSI), HOSI generation requires reasoning over dynamic object–scene changes, yet suffers from limited annotated data. To address  these issues, we propose a coarse‑to‑fine instruction‑conditioned interaction generation framework that is explicitly aligned with the iterative denoising process of a consistency model. In particular, we adopt a dynamic perception strategy that leverages trajectories from the preceding refinement to update scene context and condition subsequent refinement at each denoising step of consistency model, yielding consistent interactions.
To further reduce physical artifacts, we introduce a bump‑aware guidance that mitigates collisions and penetrations during sampling without requiring fine‑grained scene geometry, enabling real‑time generation. To overcome  data scarcity, we design a hybrid training startegy that synthesizes pseudo‑HOSI samples by injecting voxelized scene occupancy into HOI datasets and jointly trains with high‑fidelity HSI data, allowing interaction learning while preserving realistic scene awareness. Extensive experiments demonstrate that our method achieves state‑of‑the‑art performance in both HOSI and HOI generation, and strong  generalization to unseen scenes. Project page: [yudezou.github.io/InfBaGel-page](https://yudezou.github.io/InfBaGel-page/).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents InfBaGel, a unified framework for human–object–scene interaction generation that combines a coarse-to-fine consistency model with bump-aware guidance for physically plausible motion (mostly for collision free). It also introduces a hybrid data training strategy that mixes real and synthetic data to reduce annotation needs and improve generalization across diverse scenes.

### Strengths
1. The motivation and contribution is clear: to generate higher quality HOSI data and to tackle the data scarcity of this field.
2. Experiment shows that the method is significantly better than previous works in terms of collision avoidance.

### Weaknesses
1. Limited qualitative results. Only two action sequences are presented in the paper, which makes it difficult to fully assess the model’s performance.
2. The method part confuse me a bit. Specifically, for Section 3.2 (“Motion Consistency Model”), does this section correspond to the Auto-regressive Consistency Model shown in the figure? Additionally, in line 277, how are human joints and object points constructed from human motion only? Is this perhaps a typo?
3. In Figure 2(c), there is a noticeable penetration between the hand and the box. Is this due to training solely on the OMOMO dataset?

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The problem this paper aims to tackle is the (1) limited annotation in dataset and the (2) dynamic scene variation (the movement of objects changes the scene as well), for a task termed as human-object-scene interaction generation. The proposed approach consists of several contributions/effective modules, termed as the dynamic perception, bump-aware guidance, the consistency guided iterative refinement, with the help of hybrid-data-training strategy. It seems that using the proposed method achieves good results in the new benchmark HOSI and the HOI as well.

### Strengths
Strength
-	The technical method seems correct
-	The motivation is well-presented

### Weaknesses
Weakness 

-	In table 3, it seems that the C+B provides inferior results. It does not convince me that all components achieve better performance, esp in the task accuracy metric. Maybe more explanations should be provided. 

-	It seems that the baseline methods in comparison seem limited. More evaluations should be performed comparing with more methods. 


-	The proposed contributions seem a lot. These makes the contributions diluted for some perspectives. The relationship among these contributions, and evaluation of their effectiveness one by one is necessary. Otherwise we cannot understand how each component tackles which portion of data.

### Questions
See weakness

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
The paper introduces a coarse-to-fine, instruction-conditioned generation framework using a consistency model that iteratively refines the human-object-scene interaction (HOSI). 

A dynamic perception strategy is key, where scene context is continuously updated using trajectories from the preceding refinement step to condition the subsequent refinement, ensuring consistent interactions.

Experiment results show that the hybrid data training strategy overcomes data limitations by combining
real-scene HSI data with synthesized HOI data, achieving zero-shot scene generalization.

### Strengths
The core idea is to integrate a dynamic perception strategy and a coarse-to-fine iterative refinement scheme, which is aligned with a consistency model's denoising process. 

This novel approach ensures that the scene context (which changes due to object and human movement) is updated at each step, leading to more physically consistent and realistic HOSI.

Such a method is robust, allowing the model to learn diverse and complex interactions without the dependence on a massive, fully annotated HOSI dataset, enabling strong generalization to unseen scenes.

### Weaknesses
Need to show some benefits on the downstream tasks such as robot learning, gaming, humanoid motion learning, e.t.c.

How is the model integrated with real physics? 

And how is the model transfer to different simulation socially-inteactive environments (indoor vs outdoor, household, store, hospital, e.t.c.)?

### Questions
See weaknesses

### Soundness
4

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
3

### Summary
The paper proposes a method to generate paired human-object interaction motion in the context of a static scene. The input is the static scene geometry, interaction object geometry, text instruction, and goal position of the object. The dynamic human-object-scene interaction is represented as a sequence of five voxel grids, representing the scene and object occupancy at the start position, goal position, and (optionally) 3 intermediate timesteps along the human-object interaction trajectory. Voxel grids at intermediate timesteps can be masked out for unconditional trajectry generation. Then, an auto-regressive consistency model is trained via consistency distillation to generate human and object motion conditioned on these inputs. To ensure strict collision avoidance, the denoised trajectories outputted by the consistency model are iteratively refined to avoid colliding with the static scene geometry, by introducing guidance from each colliding voxel to the nearest free-space. The method is trained on a combination of real-world HSI data (LINGO) and synthetic HOI data (OMOMO).

### Strengths
- Human-object-scene interaction generation is important and challenging problem.
- The paper achieves substantially improved results compared to LINGO and TRUMANS.
- The method is technically sound and the proposed components are effective: the dynamic perception encoder helps improve task success rate, the consistency model improves generation speed, and the bump-aware guidance reduces scene penetration.

### Weaknesses
- Despite the dynamic scene encoder and bump-aware guidance, the method does not seem to generate physically plausible manipulation motions. In Fig2(c) and Fig2(e), the crate is at a strange angle, hand poses do not seem to be predicted, and the hands are either penetrating with the crate or not in contact.
- The results are shown for a single task of moving an object from the start location to the goal location through a static scene. Therefore, it is not clear to me that the text instruction is necessary or useful in the problem formulation. More qualitative or quantitative examples for diverse interaction types (such as those seen in OMOMO - kicking and dragging objects, lifting over your head or in different manners) would strengthen the applicability of the method

### Questions
- The exact output format of the dynamic perception encoder has many missing details and is not fully clear in Sec. 3.2. It would be helpful to refer to the appendix in the main text and provide mathematical notations with the shape of each variable. My understanding is, each "voxel grid" is a 3D array {0,1,2}^(NxNxN) where N is the size of the voxel grid? And there are five voxel grids, corresponding to the start position, goal position, and three intermediate timestamps? How are the intermediate timestamps sampled?

### Soundness
3

### Presentation
3

### Contribution
3
