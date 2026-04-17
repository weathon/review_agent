# EnvSocial-Diff: A Diffusion-Based Crowd Simulation Model with Environmental Conditioning and Individual-Group Interaction

- Decision: Accept (Poster)
- Scores: 2, 8, 6, 6

## Abstract
Modeling realistic pedestrian trajectories requires accounting for both social interactions and environmental context, yet most existing approaches largely emphasize social dynamics. We propose EnvSocial-Diff: a diffusion-based crowd simulation model informed by social physics and augmented with environmental conditioning and individual-group interaction. Our structured environmental conditioning module explicitly encodes obstacles, objects of interest, and lighting levels, providing interpretable signals that capture scene constraints and attractors. In parallel, the individual-group interaction module goes beyond individual-level modeling by capturing both fine-grained interpersonal relations and group-level conformity through a graph-based design. Experiments on multiple benchmark datasets demonstrate that EnvSocial-Diff outperforms the latest state-of-the-art methods, underscoring the importance of explicit environmental conditioning and multi-level social interaction for realistic crowd simulation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a diffusion-based crowd simulation model that jointly models structured environmental context and multi-level social interactions. EnvSocial-Diff introduces an environmental encoder that represents obstacles, points of interest, and lighting as conditioning signals for the denoising process. On the social side, an Interaction module models both interpersonal relations and group coherence using a graph-based design. These modules are fused with historical trajectories and a destination attraction term to guide generation toward socially compliant, context-aware motion. Experiments on GC and UCY show gains over state-of-the-art baselines.

### Strengths
- The paper presents a thorough ablation study that isolates the contribution of each module.

- The figures are clear and informative.

- The authors clearly mention the motivation behind their approach and highlight how their method differs from prior work.

### Weaknesses
- The paper lacks a notation explanation section. Introducing a concise notation summary and problem setup at the start of the methods section would significantly improve clarity.

- The methods section feels rushed and disorganized. Several terms and notations are introduced without being defined. The presentation lacks a logical progression, and makes it difficult to follow how each component connects to the overall framework.

- No sensitivity analysis is provided for key hyperparameters, and leaves it unclear how robust the model is to tuning or parameter changes.

- Implementation details are insufficient, and more information about training configurations, architectures, and computational setup would enhance reproducibility. Although the authors mention that code will be released in an anonymous repository, the link is not provided to reviewers.

### Questions
- How is the collision count defined? Does it refer to cases where predicted locations completely overlap, or is there a distance threshold?

- Are collisions computed only between people, or do they also include interactions with obstacles?

- For each dataset, what is the average number of people present per scene, and what are the minimum and maximum crowd sizes observed?

- The authors mention that their method produces "produce socially compliant, context-aware, and realistic trajectory predictions". However, it is unclear how these qualities are defined or quantified. What specific criteria or metrics are used to measure social compliance or realism in the results?

### Soundness
2

### Presentation
1

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
The paper introduces EnvSocial-Diff, a diffusion-based crowd simulation model informed by social physics and augmented with two novel modules:
(1) a structured environmental conditioning mechanism that explicitly encodes obstacles, objects of interest (OOI), and lighting conditions, and
(2) an Individual–Group Interaction (IGI) module that captures both fine-grained interpersonal dynamics and group-level conformity via graph neural networks.

The model extends the Social Physics Informed Diffusion Model (SPDiff, Chen et al., AAAI 2024) by incorporating richer environmental signals and multi-level social reasoning into the generative diffusion process.
Experiments on the GC and UCY datasets demonstrate state-of-the-art performance across multiple trajectory prediction metrics (MAE, FDE, OT, MMD, DTW, and collision count). Ablation studies confirm the contributions of environmental factors and the IGI module, while qualitative visualizations support the interpretability and realism of simulated trajectories.

### Strengths
+ Novel Integration: Elegant fusion of social physics and diffusion modeling with explicit environmental conditioning.

+ Interpretability: Maintains physically grounded meaning for forces and accelerations.

+ Comprehensive Evaluation: Multiple datasets, metrics, and ablations validate both performance and generalization.

+ General Applicability: Applicable to domains such as simulation, safety planning, and digital twin environments.

+ Strong Theoretical Foundation: Builds directly on the Social Force Model while extending its scope through learnable conditioning.

### Weaknesses
- Computational Complexity: The paper does not report training/inference times or resource comparisons versus SPDiff or data-driven baselines. This limits understanding of scalability in real-time simulation.

- Limited Dataset Diversity: Experiments rely mainly on GC and UCY datasets. These are standard but relatively small; inclusion of additional or synthetic datasets (e.g., ETH, SDD) would strengthen generalization claims.

- Lighting Factor Validation: The contribution of the lighting module is modest and potentially dataset-specific. A more detailed justification (e.g., psychophysical rationale or ablation under controlled illumination changes) would improve the argument.

- Minor Clarity Issues: Some notation (e.g., the dual use of f_{\text{light}} and \tilde{f}_{\text{light}}) could be clarified (e.g., f_{\text{light}}^{\text{raw}} and f_{\text{light}}^{\text{enc}}) for better readability.

### Questions
> Could the authors clarify whether the environmental encoders are trained jointly with the diffusion network or frozen (especially the ResNet and BERT backbones)?

> How does EnvSocial-Diff perform under dynamic environments (e.g., moving obstacles)?

> Have the authors considered testing the model’s capacity for long-horizon rollouts (>5s) to evaluate accumulation of social-environmental errors?

> Could the lighting conditioning be replaced or augmented by other perceptual features (e.g., crowd density maps, saliency maps)?

> Is the approach compatible with large-scale agent-based simulations (e.g., thousands of pedestrians), or does the GNN limit scalability?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes EnvSocial-Diff, a diffusion-based crowd simulator that augments social-physics (SFM) with (1) structured environmental conditioning (obstacles, objects of interest, lighting, etc.) and (2) an Individual–Group Interaction (IGI) module that mixes pairwise and group-conformity signals via a GNN. On GC and UCY, the method reports consistent gains over SFM/PCS/SPDiff and other baselines across various metrics, with ablations for each factor.

### Strengths
- Many prior pedestrian prediction models are purely data-driven and overlook structured environmental and social factors. There are important factors in pedestrian simulation. This work tries to bridge the social-force ideas with modern generative modeling and scene modeling, which is refreshing.
    - Accurate modelling the prior scene information will be essential for future crowd simulation works. 

- The decomposition into destination force + diffusion refinement is clean and easy to follow. It preserves interpretability while still benefiting from generative diversity.
- The group similarity and alignment logic, aggregated via GNN, is conceptually simple but grounded in real social-navigation dynamics. The ablations suggest it brings in meaningful improvement.

### Weaknesses
- The **demo video** is very difficult to interpret. This is currently the biggest presentation gap. As it stands, it is hard to tell what is happening, which agents belong to which group, or how environment cues influence behavior. Since one of the main claims is improved realism and responsiveness to context, the qualitative visualization should make these effects obvious. Overlays, legends, visual callouts, and side-by-side comparisons would help a lot.
- The framework layers several components (diffusion, semantic encoders, GNN, physics prior). While individually reasonable, it would be good to comment on compute cost and whether a simpler scene-encoder and diffusion baseline might achieve similar gains.
- GC and UCY are standard dataset in crowd simulation, but they are relatively small scenes. Since the paper emphasizes complex social and environmental structure, a brief discussion or qualitative example in a denser or more varied environment would help support the generality claims.

### Questions
- How sensitive is the method to the accuracy and granularity of scene annotations (objects of interest, lighting, obstacle maps)? For example, if these are noisy, incomplete, or generated automatically from imperfect scene-understanding systems, how does performance degrade?

### Soundness
3

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
3

### Summary
The paper introduces EnvSocial-Diff, a diffusion-based model for crowd simulation. Conditioned on both environmental factors such as obstacles, objects of interest, and lighting and Individual–Group Interactions (IGI), the model aims to generate realistic walking trajectories for multiple agents.
Unlike prior approaches, EnvSocial-Diff explicitly models group conformity and leverages environmental cues beyond simple repulsive forces or binary traversability maps. Notably, it incorporates lighting conditions, which have been shown in behavioral studies to influence human motion but are largely overlooked in existing work. Trajectories are synthesized (as accelerations) using a diffusion model, conditioned jointly on the environment, IGI, and agent motion history.

### Strengths
(+) The proposed architecture effectively unifies and jointly models three environmental factors (obstacles, OOI, and lighting) with a social model (IGI), resulting in a powerful and more nuanced conditioning signal for the generative process, than ones used in prior works

(+) the paper shows significant design effort in the Individual-Group Interaction (IGI) module, which elegantly captures social dynamics at three distinct, complementary signals: approach tendency, motion alignment and crucially group conformity.

### Weaknesses
(-) the definition of several components is lacking. This includes 
equation 1 doesn't expalin what m, v and mu are
(-) line 237: where is this global scene feature coming from? 
(-) also the bias term as a function of the relative position between actor and obstacle isn't exaplined. why if the relative distance is larget, should the attention between actor and obstace grow?

(-) the setup should be spelled out in the beginning -- what do the authors mean by scene for example -- a single BEV image with annotated objects, obstacles, their locations and type, etc. what exactly is the output -- is it for all trajectories or just a sigle 
accordingloy the dimensino of obstacle position for example should be states (2 i guess?)

(-) computational cost: i didn't see a runtime analysis and suspect that since each actor requires a full denoising scheme per waypoint the number of model activations is quite large. An analysis of NFE and wall clock time would be useful. Also memory consumption. 

(-) the demo didn't really help me see the differences between the suggeted method and the baseline -- what are failrue cases in the baseline that are fixed or avoided byt he suggested model?

minor: 
(-) the transpose op in equation 8 escaped

### Questions
(-) in figure 1 the store is marked in dotted red but i believe it should be an object of interest which should then be marked in green?

(-) how would the model handle spawning or removing of agents? I noticed that in the demo about 12 second in a pair of new nodes appears out of no where -- what is happening there? 

(-) the collision count in the proposed method is reduced but i would argue stil seems quite high. What is the cause of that? could re-introducing the repulsive forces help?

### Soundness
3

### Presentation
3

### Contribution
2
