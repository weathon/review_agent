# Plug-and-Play Label Map Diffusion for Universal Goal-Oriented Navigation

- Decision: Reject
- Scores: 6, 6, 0, 6

## Abstract
In embodied vision, Goal-Oriented Navigation (GON) requires robots to locate a specific goal within an unexplored environment. The primary challenge of GON arises from the need to construct a Bird's-Eye-View (BEV) map to understand the environment while simultaneously localizing an unobserved goal. Existing map-based methods typically employ self-centered semantic maps, often facing challenges such as reliance on complete maps or inconsistent semantic association. To this end, we propose Plug-and-Play Label Map Diffusion (PLMD), which defines a novel map completion diffusion model based on Denoising Diffusion Probabilistic Models (DDPM). PLMD generates obstacle and semantic labels for unobserved regions through a diffusion-based completion process, thereby enabling goal localization even in partially observed environments. Moreover, it mitigates inconsistent semantic association by leveraging structural consistency between known and unknown obstacle layouts and integrating obstacle priors into the semantic denoising process. By substituting predicted labels for unobserved regions, robots can accurately localize the specified objects. Extensive experiments demonstrate that PLMD \textbf{(I)} effectively expands the region of unknown maps, \textbf{(II)} integrates seamlessly into existing navigation strategies that rely on semantic maps, \textbf{(III)} achieves state-of-the-art performance on three GON tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Plug-and-Play Label Map Diffusion (PLMD), a diffusion-based map completion module to address the "partial observation" problem in Goal-Oriented Navigation (GON). PLMD leverages Denoising Diffusion Probabilistic Models (DDPM) to generate obstacle and semantic labels for unobserved regions, with structural constraints from known obstacles to ensure semantic consistency. It is designed as a plug-and-play module that integrates seamlessly with existing navigation strategies without retraining. Experiments on three GON tasks (ObjectNav, Instance-ImageNav, Multi-Robot ObjectNav) across HM3D and MP3D datasets show PLMD achieves state-of-the-art performance in success rate, path length-weighted success rate, and map completion quality.

### Strengths
1. PLMD can be integrated into existing navigation strategies without retraining, reducing the cost of upgrading real-world systems — a key advantage for industrial adoption.

2. Comprehensive validation: The authors test PLMD on three GON tasks (ON/IIN/MRON) across three datasets (HM3D_v0.1/v0.2, MP3D), providing sufficient evidence of cross-task and cross-environment generality.

3. Unlike most simulation-only works, the paper verifies PLMD on Jetson AGX Orin, demonstrating feasibility for resource-constrained embedded robots.

4. Multi-dimensional evaluation: Beyond standard navigation metrics (SR/SPL), the use of PSNR to measure map completion quality links the module’s intermediate performance to final navigation results, enhancing analysis depth.

### Weaknesses
1. PLMD’s "obstacle-guided diffusion" is a minor modification of existing diffusion-based map completion (Ji et al., 2024). The paper does not highlight how obstacle constraints solve fundamental limitations of diffusion models in navigation (e.g., mode collapse in sparse environments) — it merely adds a heuristic prior.

2. High computational overhead: PLMD’s FLOPs (34.5G) are an order of magnitude higher than lightweight navigation models (e.g., SemExp’s 3.1G). The paper only mentions "periodic activation" (every 50 steps) as an optimization but does not propose model compression (e.g., quantization, pruning) to improve real-time performance, which is critical for robot navigation. More experiments are required.

3. The authors admit most failures stem from localization errors due to bad semantic segmentation, but PLMD does not include mechanisms to mitigate this (e.g., multi-modal fusion, uncertainty estimation for segmentation). This makes the module fragile in scenarios with low-quality RGB-D inputs.

4. Some related and important works are missing citations: [1] Weakly-Supervised Multi-Granularity Map Learning for Vision-and-Language Navigation [2] IGL-Nav: Incremental 3D Gaussian Localization for Image-goal Navigation [3] Gridmm: Grid memory map for vision-and-language navigation

### Questions
See weakness.

### Soundness
3

### Presentation
3

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
This paper proposes a new Plug-and-Play Label Map Diffusion (PLMD) method, which aims to utilize Denoising Diffusion Probabilistic Models (DDPM) to generate a complete obstacle and semantic label map for unobserved regions, enabling goal localization in unknown indoor environments.
Specifically, it designs a label-guide denoising process that leverages obstacle distributions as structural constraints, ensuring consistent and reliable semantic reconstruction at the pixel level.
A clustering algorithm is utilized to identify potential navigation goals in the generated map, forming a candidate goal set.
The map generation is repeated until the candidate goal set is identified or the navigation strategy locates the goal.
Extensive experiments on HM3D and MP3D demonstrate the effectiveness of the proposed method.

### Strengths
1)The proposed Label-level Map Completion is delicate and reasonable, which leverages known obstacles and object semantic information in an explicit label map to rebuild unknown regions.
2)The proposed PLMD does not rely on a specific navigation strategy and can be integrated seamlessly into existing navigation strategies that rely on semantic maps.
3)Extensive experiments demonstrate the effectiveness of the proposed method in assisting navigation strategies to locate the goal.

### Weaknesses
1) In line 13, Biew -> View.
2) In line 198, the equation to construct the label map dataset does not seem to be fully accurate, since the incomplete label map is stored every 25 steps, t should not be from 0 to F, and the summation symbol is not very appropriate.
3)During training, are the obstacle map network and the semantic map network updated simultaneously after pretraining the obstacle map network? It is not very clear in the current manuscript. Since the obstacle map is first predicted, does it mean the obstacle map is simpler than the semantic map for prediction? Why don't you predict them simultaneously?
4)Why is the time efficiency computed with the method based on LLM/LVM? If LLM/LVM is introduced as the navigation strategy, its potential for predicting goal location should be evaluated and compared, not singly as a navigation strategy. Compared with RL-based methods, the time cost of the proposed diffusion-based method may be too large.
5)The idea of completing a map to predict an object goal is a little similar to PONI, which is not compared in the main manuscript.

### Questions
please try to address the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper addresses the problem of navigating to a goal in an unknown environment. The paper uses diffusion to generate a prediction of the structure of the unknown part of the map for the purposes of navigation. The paper shows evaluations on Habitat-Matterport3D and Matterport3D and show that the approach compares favourably to a set of baselines.

### Strengths
The paper addresses an interesting and important problem in robotics, which is goal-directed navigation in unknown or partially unknown environments. The idea of using diffusion to predict the unknown parts of the map is interesting, and worth studying.

### Weaknesses
Unfortunately, there is very little to recommend this paper for acceptance. The primary objection I have is that the training process for the diffusion model appears to be entirely in the same environment as the test evaluation, as given by "we generate training and validation data from label maps collected during the interaction of robot with the environment. First, we randomly initialize a starting position within indoor environments" and "To train the PLMD, we collect obstacle and semantic maps of size 256×256 through the Habitat simulator". Allowing the system to go and build a diffusion model of the environment is fundamentally no different (and arguably harder) than allowing the robot to go and build a map ahead of time. This paper could only be of interest if there had been any attempt to test in one environment and evaluate in another, different environment. The authors do not even appear to have partitioned the environment into disjoint train and test areas.

Secondly, the experimental results are very unclear. How is the map updated?  What (simulated) sensor data is used? It is not clear why success is not 100% -- if the goal object is not at the predicted location, does the mission terminate? That is not a particularly useful setting for this problem -- a far more useful (and common) setting is where diffusion model is used as a prior, and expected costs are calculated using the model as a prior, iterating via replanning until the goal is found.

The paper focuses on finding goals in unknown maps, but does not compare against the fairly large literature in robotics that addresses that problem, and primarily compares itself against RL and other diffusion approaches. It would be interesting to compare against the planning under uncertainty work that leverages structured models (e.g., work by Greg Stein, Nikolay Atanasov, Luca Carlone, etc.).

If the paper had focused entirely on the problem of map prediction and not the navigation problem, I might be more in favour of this paper, but even viewed in that light, the contribution is relatively modest. The paper presents results that indicate that the two-level label map captures "the contextual relationship between obstacles and semantic features" but this seems to be a statement that explicitly modelling the relationship between obstacles and semantics provides a better model overall, which has been known for sometime in the semantic SLAM community.

### Questions
- Why does the robot not succeed 100% of the time? Do the trials terminate early? 
- How well does the approach generalise across environments? Can it be trained in one environment and succeed in another?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes PLMD (Plug-and-Play Label Map Diffusion), a diffusion-based framework for semantic and obstacle map completion to enhance embodied navigation.
- The method performs diffusion inference under joint conditioning of semantic and obstacle labels for structural map completion and introduces a clustering-based integration strategy for selecting long-range navigation goals.
- Experiments are conducted on the HM3D dataset, covering ObjectNav (ON), ImageNav (IIN), and Multi-Robot ObjectNav (MRON) tasks.

### Strengths
- The model can be seamlessly integrated into existing navigation systems, demonstrating strong modularity and generality.
- PLMD achieves performance improvements across multiple subtasks (ON, IIN, MRON), and the generated maps provide intuitive visual evidence.

### Weaknesses
- The integration strategy for cluster core selection relies on fixed, manually tuned weights (0.5 / 0.4 / 0.1) that linearly combine cluster size, semantic confidence, and distance penalty. However, the paper does not provide any theoretical justification or empirical validation for this formulation. Such empirically chosen parameters may overfit to specific datasets (e.g., HM3D), limiting generalization and reproducibility. It is recommended to include a weight sensitivity study, showing how different combinations affect SR/SPL performance, and to clarify the individual contributions of each component to navigation quality.
- Diffusion-based models typically involve lengthy iterative inference, and PLMD requires optimal reconstruction under the joint conditioning of obstacle and semantic labels, which may further increase inference time. However, the paper does not report the model’s average inference time, computational resource consumption, or its impact on real-time performance during online navigation. The absence of these measurements limits the assessment of the method’s engineering practicality and deployability. It is recommended to include additional validation experiments reporting average inference latency and GPU resource usage to demonstrate the real-time feasibility of the proposed approach in embodied intelligence scenarios.
- The paper claims that PLMD is a “general-purpose navigation enhancement method,” but all experiments are conducted primarily on closed-set semantic datasets. The model has not been evaluated on open-vocabulary tasks, zero-shot or few-shot transfer scenarios, nor has its performance been verified across different datasets or real-world environments. This weakens the evidence supporting the claim of generality. Since real-world environments typically contain open semantics and unknown structures, the current experiments do not demonstrate the model’s actual adaptability. It is recommended to conduct evaluations on open-vocabulary tasks or cross-dataset transfer experiments to substantiate the core claim of “general navigation.”

### Questions
- The idea of using generative models for map completion in goal-oriented navigation is not new — for example, “Imagine Before Go” and “Distilling LLM Prior to Flow Model” have explored similar directions.
Please clarify how PLMD differs from these works in terms of the generation mechanism, type of prior, and the key insight or advantage it brings beyond existing generative map completion methods.
- Why were OpenFMNav, FBE, and MCoCoNav chosen as the base navigation models instead of other alternatives?
- How was the choice of 100 diffusion steps determined? Was it selected as a trade-off between inference time and generation quality?

### Soundness
3

### Presentation
3

### Contribution
2
