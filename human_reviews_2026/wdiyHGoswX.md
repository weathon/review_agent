# ControlTac: Force- and Position-Controlled Tactile Data Augmentation with a Single Reference Image

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
Vision-based tactile sensing is widely used in perception, reconstruction, and robotic manipulation, yet collecting large-scale tactile data remains costly due to diverse sensor-object interactions and inconsistencies across sensor instances. Existing approaches to scaling tactile data—simulation and free-form tactile generation—often yield unrealistically rendered signals with poor transfer to highly dynamic real-world tasks. We propose **ControlTac**, a two-stage controllable framework that generates realistic tactile images conditioned on a single reference tactile image, contact force, and contact position. By grounding generation in these important physical priors, **ControlTac** produces realistic samples that effectively capture task-relevant variations. Across three downstream tasks and three real-world experiments, the augmented datasets using our approach consistently improve performance and demonstrate practical utility in dynamic real-world settings. Project page: https://controltac.github.io/

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a two-stage conditional diffusion framework that augments vision-based tactile datasets by generating synthetic GelSight images from a single real tactile image, while explicitly conditioning on contact force (3-D vector) and contact position (2-D mask). The authors evaluate the augmented data on three downstream tasks (force estimation, 2-D pose estimation, and object classification) and report improved accuracy relative to baselines that rely on real data only, classic geometric augmentation, or simulation. Three real-robot experiments (pushing, tracking, and peg insertion) are included to demonstrate sim-to-real transfer.

### Strengths
Novelty: The idea of physically grounded, force- and position-conditioned tactile generation is new. The two-stage design (force first, then position via ControlNet) is intuitive and modular.
Technical quality: The diffusion backbone (DiT + ControlNet) is appropriate for high-frequency tactile textures; ablations against hybrid and separate baselines are thorough.
Experimental breadth: The paper trains downstream networks from scratch, mixes real/generated data, and evaluates on unseen objects and a second sensor instance—an unusually complete pipeline.
Real-robot validation: Insertion with 3 mm tolerance and 75–90 % success rates is convincing evidence that the generated images do not suffer from the “texture hallucination” problems common in earlier Vis2Tac work.
Reproducibility: Training details, hyper-parameters, and code URLs are provided; datasets are public.

### Weaknesses
Physical faithfulness of force conditioning
The force label is a single 3-D vector measured at the robot wrist. GelSight deformation is driven by the distributed contact pressure field, not by the resultant force alone. Two contacts with identical resultant forces but different pressure profiles (e.g., flat vs. edge) produce different tactile images. The paper does not discuss whether the generator implicitly learns this mapping or simply interpolates intensities. A small ablation that perturbs the pressure distribution while keeping the resultant force fixed would clarify physical plausibility.
Position representation and generalisation
The contact mask is a rigid binary template transformed by 2-D translation + rotation. This works for the convex/curved PLA objects used in evaluation, but fails for:
(a) objects whose contact area changes with force (soft materials),
(b) non-convex or articulated geometries (keyhole, USB-C shield) where the mask topology varies.
Fig. 12 already shows degraded quality on a banana and a flattened ring. The claim “generalises to unseen objects” is therefore overstated; generalisation is shown only for rigid, PLA-printed shapes with similar size and curvature.
Baseline fairness
TAXIM is forced into an unfair setting: the authors manually convert z-displacement into normal force and add position control, although TAXIM was not designed for this. The reported MSE gap (1054 vs. 23) is therefore inflated.
The Vis2Tac baselines (Li et al. 2019, Dou et al. 2024, etc.) are single-image translators; they cannot vary force/position. Comparing them without allowing them to generate multi-modal outputs makes the superiority of CONTROLTAC trivial. A fairer baseline would be a stochastic Vis2Tac model fine-tuned with the same DiT backbone and force/position conditions.
Dataset bias and failure modes
All training objects are PLA, Lambertian, and 3-D printed. The generator inherits these biases: it cannot predict high-frequency surface texture (fabric, leather) or subsurface scattering (skin, silicone). Fig. 12 shows ringing artifacts on the banana. The ethical statement claims “no risks”; however, if downstream users rely on synthetic data for safety-critical insertion, dataset bias becomes a safety issue. The paper should state limitations more prominently.
Statistical significance
Real-robot experiments report success rates on 20 trials per object. With only 3 objects, the 95 % confidence interval on the 85 % success rate is ≈ ±15 %. The difference between “real-data-only” and “augmented” policies is not statistically established. Either increase trials or provide confidence bounds.
Clarity and typos
Table 1 caption uses “↓” for MSE but “↑” for SSIM without defining arrows.
Section 4.2 claims “competitive performance with 1/3 of the real data” but the plotted curves cross inside one standard deviation—wording should be softened.

### Questions
Include a 1-D pressure line-scan plot (real vs. generated) for the same resultant force to verify that the generator does not simply scale overall brightness.
Report inference time and GPU memory; tactile augmentation is only useful if it is faster than collecting real data.
Discuss whether the ControlNet branch can be removed at test time to accelerate generation.
Clarify copyright and consent for the Type-C connector photo (Fig. 21).

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
4

### Summary
This paper introduces ControlTac, a two-stage controllable tactile data augmentation framework that generates realistic tactile images from a single reference image, conditioned on contact force and position. ControlTac produces diverse and physically plausible tactile samples that significantly improve performance across downstream tasks such as 3D force estimation, contact pose estimation, and object classification.

### Strengths
1. The paper introduces a controllable and physically grounded tactile data generation method, enabling fine-grained control and physical plausibility.
2. ControlTac can generate thousands of realistic tactile images from just one reference image.
3. This work conducts extensive experiments to validate the effectiveness of ControlTac, including real-world experiments.
4. Compared to simulation-based and free-form generative methods (e.g., Text2Tac, Vis2Tac), ControlTac produces more realistic and varied outputs
5. This modular framework is easy to follow.

### Weaknesses
There are several main issues in this paper that remain unaddressed:

1. In the downstream tasks, is ControlTac further fine-tuned, or does it directly use the FeelAnyForce pre-trained model in a zero-shot manner? If it is the latter, can a model trained on only 20,000 frames from FeelAnyForce truly support generalization to a wider range of more complex objects in more open environments? In the failure cases shown in the appendix, the model performs worse on objects with flat surfaces, rich textures, or varying hardness. Could this issue be addressed by introducing a more diverse set of objects into the training dataset? I believe this is crucial to substantiating the paper’s claimed “cross-object generalization capability.” 
2. In the data generation process for downstream tasks, how is the force used to control generation determined? If it is randomly sampled, how is its range defined? Could this potentially lead to contact mask–force pairs that are physically implausible?
3. The description of the object pushing downstream task is unclear. What is the specific objective of this task? What do the numbers in Table 7 represent, and how are they computed? Is the applied force static or dynamic during the pushing process?
4. The real-world experiments have certain limitations, especially since the Insertion Task does not demonstrate clear real-time interaction characteristics. Can this work improve performance in more complex dexterous manipulation tasks?

### Questions
1. It would be better to add labels of “generated images” and “error maps” to the upper and lower parts of Figure 4 for clearer presentation.
2. Is it possible to perform zero-shot generation on heterogeneous sensors (e.g., DIGIT)?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This work proposes to train a DiT+ControlNet to generate images corresponding to what a vision-based tactile sensor will see. The generation will be conditioned by a reference image from the real tactile sensor of the object of interest, the value of the force with which the sensor is touching the object and a contact map (a binary image) outlying the position of the object. By varying force and contact map the model can generate many visual tactile images from a single reference one better than what a simulator would do. The model can be used as data augmentation for real datasets and improve performances on 3 downstream tasks: force estimation, object pose estimation and object classification (among 6 classes). The specific models for each task are tested on offline datasets, but also in real condition with the use of a robotic harm. In general the data augmentation strategy based on the trained model is somewhat useful when the amount of available real data is small.

### Strengths
+ A quite interesting application of DiT+ControlNet to some under-explored touch. The way the authors designed the conditioning signals for their model makes a lot of sense in the context of the work and what they are trying to achieve.

+ According to Tab. 1 the method can recreate significantly more faithfully images from a tactile sensor for seen objects wrt to a separate simulator.

+ I appreciated that the authors took their proposal for a real world test including experiments with a real robot and various objects

### Weaknesses
## Major 

A. **Experiments in Sec. 4.1 are all in-domain:** Per my understanding the results in Sec. 4.1. Cover only “in domain” experiments, meaning generations of images for known objects belonging to the same dataset used to train ControlTac. This puts the method and the ablations at an unfair advantage against Taxim that has not been fine tuned for that specific category of objects. The gap is big enough that the proposed method might still be better, but I would have expected a generalization experiment by removing from the training data a certain type of object and testing generation quality against it.

B. **Technical novelty:** While from an application perspective I feel like this paper checks all boxes, from a technical novelty perspective there is fundamentally limited innovation. The contribution is an application of DiT and ControlNet to this very specific robotic domain and showing that it can achieve promising results. There is not a strong technical novelty per se. I think it might be also a matter of the type of avenue where the work is submitted, I would find this work way more fitting for a robotic conference. I’m not sure what the broader ML community would find interesting from this work. 


## Minor


C. **Minimal contributions to performance when enough real data are available:** Sec. 4.2 - Fig. 5 kinda shows that the learned model does not really add anything to a real dataset with enough samples. Note that the same real dataset is the one used to train ControlTac, so it is not very relevant to check performance when only a subset of them are available. Results seem more promising in Sec. 4.3 and Sec 4.4, but ties to Weakness A on the fact that without having a good measure of the generalization performance of the method it’s hard to say whether images generated with Taxim would have equally (or better) helped in these 2 experimental settings.

D. **Not very self-contained:** The paper presentation could be improved by providing a bit more background on tactile sensors and on what are the representations that are learned in the paper. The current draft assumes a significant amount of previous knowledge in the reader. I would suggest adding a section before 3.1 explaining what is a visual tactile sensor, what is the “background”, what is the “force vector” etc.

### Questions
1. What’s the impact of the choice of the reference image on the generation? If I pick 2 reference images from the same object and condition the generation with the same force and position control how much are the generated images changing? 

2. Can you comment on weakness A?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposed a tactile data augmentation method based on a single reference image called ControlTac. 
In the first stage, authors applied a DiT that takes tactile image representation from a pre-trained SANA encoder and a 3D force vector as condition, to generate a corresponding target tactile image that reflects the desired force.
In the second stage, they reuse the DiT to allow the decoder to generate the same object under a 2D affine transformation.
With this pipeline, the authors could perform effective data augmentation. 
Results have shown that this augmentation helps downstream offline perception and real-world manipulation experiments.

### Strengths
1. It is true that the lack of high-quality tactile images is a bottleneck to the community. A reliable augmentation method will benefit the tactile research community.
2. The authors designed and conducted extensive downstream experiments, providing qualitative and quantitative results to show the effectiveness of their method.

### Weaknesses
1. Fig 1 is not clear. From the context, I infer the "Data Augmentation" block is saying "generate tactile images for the same object with different 3D force and contact pose", but this invariance is not mentioned here or in the related context.
2. Regarding Fig 2 and related discussion: Text2Tac and Vis2Tac are obviously NOT tactile data augmentation methods. These two are not giving accurate tactile signals, but they try to align some tactile properties with other modalities in a generative model. It's inappropriate to put them in the context of this manuscript.
3. The definition of "tactile latent representation" is misleading for different settings. It's unclear when you mention the one from SANA, or after force injection. 
4. It is still unclear why you use a contact mask to augment, given that we literally have the same object augmented with a 2D rotation. Shouldn't a 2D rotation matrix (3DoF) do the same job compared to a mask? When the object is larger than the sensor's size, how can masks help then? Also, how is edge detection involved in this stage?
5. On the motivation level, for a certain GelSight sensor, its deformation-to-force projection is determined by gel properties. This makes the cross-sensor claim questionable -- replacing background will only help when transferring across physically similar data with a bit of optical variance, limiting the actual contribution of this work. 
6. Compared to simulation, I think one big advantage of your augmentation method is that you should be able to perform augmentation on objects with rich texture. However, only objects with simple geometry are shown in the figures. Is that due to the limitations of your encoder/decoder?

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
