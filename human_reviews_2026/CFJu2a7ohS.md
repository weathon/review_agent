# BEVCA: Effective and Transferable Camouflage Attack against Multi-View 3D Perception in Autonomous Driving

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Multi-view 3D perception models are widely adopted by leading car manufacturers due to their highly competitive performance. However, existing adversarial camouflage techniques primarily focus on single-view 2D detectors, limiting their effectiveness against multi-view 3D perception models. In the paper, we propose BEVCA, the first framework to generate adversarial camouflage that effectively attacks multi-view 3D perception models by exploiting the Bird's-Eye-View (BEV) representation used across various 3D perception models. Our framework introduces a new differentiable multi-view neural renderer to enable end-to-end gradient-based camouflage optimization. Furthermore, we propose a novel BEV-feature-based adversarial loss to achieve effective and transferable attacks. Extensive experiments on 3D object detection and segmentation scenarios demonstrate that BEVCA outperforms existing baselines, achieving attack improvements of 36.2\% and 21.6\% in black-box settings, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Unlike prior works that optimize adversarial textures for 2D detectors or localized patches, BEVCA introduces a multi-view neural renderer to ensure geometrically consistent textures across multiple cameras and a BEV-feature-based adversarial loss to improve transferability. Experiments on 3D object detection and segmentation tasks show significant improvements over prior baselines in both white-box and black-box settings.

### Strengths
1. Attacking multi-view BEV-based perception is important, since these models dominate current autonomous driving pipelines.
2. The paper evaluates across both detection and segmentation tasks, under white-box and black-box settings, with ablation studies on losses and hyperparameters.

### Weaknesses
1. Shifting the attack target from final outputs to early/intermediate feature spaces to improve transferability has been widely validated across vision tasks (e.g., [a-c]). Therefore, presenting “attacking BEV features” as a primary contribution is insufficient.

[a] Wang, Pengfei, et al. "Left-right Discrepancy for Adversarial Attack on Stereo Networks." arXiv preprint arXiv:2401.07188 (2024).
[b] Inkawhich, Nathan, et al. "Feature space perturbations yield more transferable adversarial examples." CVPR. 2019.
[c] Huang, Qian, et al. "Enhancing adversarial example transferability with an intermediate level attack." ICCV. 2019.

2. Whether the optimized textures are transferable to non-BEV-based 3D perception models, even 2D models, has not yet been discussed.

3. Lack of physical-world validation. All experiments are conducted in CARLA simulation, making the practicality and feasibility of BEVCA in the real world uncertain, especially given the additional noise and disturbances from real environments and sensors.

4. This paper should more thoroughly discuss the real-world feasibility of covering an entire vehicle with adversarial texture:
   * Low stealthiness. Large, conspicuous repainting or full-vehicle wraps are not as stealthy as a small patch. In practice, the attacker would likely need exclusive control of the camouflaged vehicle (e.g., their own car) to deploy the attack — it is not something an attacker can covertly apply to someone else’s vehicle. If the camouflaged vehicle is the attacker’s own car, any reduction in other vehicles’ perception increases collision risk to the attacker as well.
   * Large changes to vehicle appearance (paint, wraps, decals) are restricted or regulated in many jurisdictions and may require permits or registration changes. This creates legal barriers for real-world deployment.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
I like the proposed ideas in this paper, e.g., BEV masking and incentivizing the features to get to zero. I also like that it seems from the results to be transferrable. however I would like further expansion as to why this attack is transferrable unlike other attacks.

The results are solid; however experiments confined to Carla are not ideal considering the many prior works that worked directly on realistic scenes. This puts into question the sim2real gap and the applicability of this work in the real world.

### Strengths
The strengths of this paper lie in:
1. I like the idea that the adversarial loss is on the BEV feature instead of the detection output. Focusing on zeroing out that BEV feature is pretty cool and makes sense; though there should've been a lot more expansion about why that might be a good idea and how the authors came up with it and how it helps in black-box settings. 
2. I like the novelty of the black box attack which isn’t very common and a much harder problem. 
3. The focus on camouflage instead of patching is good, but the authors should mention other approaches as well like 3D objects [1] which can rendered a lot easier in real scenes instead of needed models of Cars to fit the camouflage on.
4. The results are pretty good and attractive, though the dataset is synthetic.

### Weaknesses
The paper has some nice results, but in my opinion doesn’t have wide applicability for a few reasons:
1. Everything is simulated in Carla no real life datasets (the work [1] from 4 years ago was also 3D rendered but on the real KITTI dataset) this could put into question the applicability of this work to real life threats.
2. The dataset is synthetic and custom made and baselines are evaluated against this custom 500 scene dataset which can make the result hard to interpret considering there are many existing baseline out there.
3. The qualitative examples seem to mostly show the attacking car in isolation and no other cars or objects around. Also the cars are placed awkwardly and randomly in the scene. This is extremely unrealistic and again puts into question if this method could pose a real threat relative to the numbers.
4. Level of novelty: camouflage attacks have been around but the novelty is mainly in the BEV adversarial loss (expanded on in strengths).

[1] Abdelfattah, Mazen, et al. "Adversarial attacks on camera-lidar models for 3d car detection." 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2021.

### Questions
What would’ve made this result very strong is the rendering of this car on real multi-view NuScenes scenes and reporting how SOTA models struggled.

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
This paper introduces a new method that creates camouflage patterns to fool the multi-camera 3D perception systems used in self-driving cars. By attacking the system's internal BEV representation, the attack remains effective across different models and tasks without needing to know their specific designs.

### Strengths
1. Good attack performance: The proposed BEVCA method largely outperforms all existing baselines.

2 . Clear presentation: The paper's core idea and methodology are explained clearly, and the writing is easy to follow.

### Weaknesses
1. Experimental comparison issues: (1) The comparison results with methods like ACTIVE are not meaningful, because these methods were trained using 2D object detection loss functions. Directly applying them to attack 3D object detection is unfair. Therefore, the Multiview Neural Rendering proposed in this paper does not necessarily outperform previous methods. (2) The paper cites NeRF-based 3D adversarial texture methods (e.g., Li et al., 2024) but does not compare their method to them.

2. Physical-world validation: As noted in Section 5 (Conclusion & Limitation) of the paper: "Our current work is limited… as we have not conducted physical experiments…." However, physical-world testing is essential in this field.

3. Feature visualization: The visualization of the BEV difference features in the paper could be enhanced. It is suggested to add more intuitive indicators and a more detailed explanation.

### Questions
see weaknesses

### Soundness
2

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
3

### Summary
This work proposes BEVCA, the first work to generate adversarial camouflage to attack multi-view 3D BEV-based detection models. Its framework consists of two core novel modules: multi-view neural rendering and BEV-feature-based attack modules. The former module conduct various coordinate system transformation to generate ego-vehicle multi-view images and use differentiable rendering library PyTorch3D to support gradient-based attack. The latter module introduces novel attack on the BEV feature to increase effectiveness and transferrability. Experiments on Carla simulated environment with both white-box and black-box setting show non-trivial success rate and reveal severe security threat in existing multi-view 3D perception models.

### Strengths
Study on attacking 3D detectors with camouflage is a very practical and valuable research topic with lots of applications in autonomous driving. This work is the first to generate adversarial camouflage that effectively attacks multi-view 3D perception models. On simulated data, BEVCA shows non-trivial attack success rate, revealing clear security loophole in existing autonomous driving systems.

### Weaknesses
1. This paper only studies simulated environment, but never present any real-world attack successful case. For instance, the neural rendering module is specifically designed for simulation data, making it hard to bridge the sim2real gap. If this work cannot be applied to real-world data, its value and practibility is greatly compromised. I wonder if the authors have given any thoughts on how to adapt BEVCA to real data (e.g. nuScenes)?

2. Only BEVFormer is studied in this work. As another line of representative BEV-based multi-view 3D detector, BEVDepth[1] should also be included.

**References**

[1] Li, Yinhao, et al. "Bevdepth: Acquisition of reliable depth for multi-view 3d object detection." Proceedings of the AAAI conference on artificial intelligence. Vol. 37. No. 2. 2023.

### Questions
1. The authors use a synthetic fine-tuning dataset of 500 samples to bridge the real2sim domain gap, which is several orders magnitude less than the original nuScenes dataset. I wonder if that is enough to adapt the model to the Carla Simulator domain.

2. It is a bit counter-intuitive to see that optimizing L_det or L_bev + No Mask leads to almost no performance degradation in Table 3. In my previous experience, white-box attack almost always break a non-adversarial model after sufficient optimization steps. The authors claim that substantial noise signals leads to ineffective noise signals, which I find rather hard to believe. Well-optimized attack images should cover that even it contains a lot of background parts. I wonder if the authors could give some explanation on this result?

3. In the black-box setting, how was the transferability evaluated? I've only seen a bunch of 3D object detection and BEV segmentation
models being used (line 335), but what models are used as source model, and what is used as target model?

### Soundness
3

### Presentation
3

### Contribution
3
