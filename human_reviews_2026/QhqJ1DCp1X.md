# H2OFlow: Grounding Human-Object Affordances with 3D Generative Models and Dense Diffused Flows

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Understanding how humans interact with the surrounding environment, and specifically reasoning about object interactions and affordances, is a critical challenge in computer vision, robotics, and AI. Current approaches often depend on labor-intensive, hand-labeled datasets capturing real-world or simulated human-object interaction (HOI) tasks, which are costly and time-consuming to produce. Furthermore, most existing methods for 3D affordance understanding are limited to contact-based analysis, neglecting other essential aspects of human-object interactions, such as orientation (e.g., humans might have a preferential orientation with respect certain objects, such as a TV) and spatial occupancy (e.g., humans are more likely to occupy certain regions around an object, like the front of a microwave rather than its back). To address these limitations, we introduce **H2OFlow**, a novel framework that comprehensively learns 3D HOI affordances ---encompassing contact, orientation, and spatial occupancy--- using only synthetic data generated from 3D generative models. H2OFlow employs a dense 3D-flow-based representation, learned through a dense diffusion process operating on point clouds. This learned flow enables the discovery of rich 3D affordances without the need for human annotations. Through extensive quantitative and qualitative evaluations, we demonstrate that H2OFlow generalizes effectively to real-world objects and surpasses prior methods that rely on manual annotations or mesh-based representations in modeling 3D affordance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a pipeline for learning object affordances from 3D Human-Object interactions obtained using 3D generative AI. First, the method uses a pre-trained generation pipeline to generate human-object interaction motions, then uses it to train a diffusion model that, given an object, predicts plausible humans interacting with it using Dense Flow. The paper studies three kinds of affordances: contact, spatial, and orientational-based, and compare against several baselines and objects with geometry unseen at training time, showing good generalization.

### Strengths
- I found the paper interesting, proposing a direction orthogonal to the ones in the previous methods. The study of different kinds of affordances could serve to define object classes more effectively based on their spatial and orientation qualities.
- Improvement over competitors is significant, and the results on real objects acquired with a phone camera look exciting

### Weaknesses
- Generate interaction data using a pre-trained 3D network specifically designed for generating 3D HOI might actually incorporate biases from such methods. For example, as mentioned in the paper, the majority of methods focus on contact-based interactions, and hence, the generated data are also affected. Additionally, one might wonder why not use the original datasets used to train CHOIS as training data for the Dense Flow. Finally, the Dense Flow module works at a frame level, while CHOIS generates sequences, which can be quite redundant in terms of nearby frames.

- The point clouds used in the real-world setting look particularly clean. I assume it is part of the post-processing done by the capturing device, but it would also be useful to test on more noisy data, such as point clouds from kinect fusions, as those available in the BEHAVE dataset. 

- Qualitative results are not well presented. Figure 3 illustrates, for example, contact over the object; however, although the table is symmetric, the legs have different affordances, highlighting a skewed distribution in the dataset. On the human body, contact is spread on legs, which I do not find intuitive. Orientational and spatial are difficult to decode. I would suggest reporting a colormap for the visualized colors, describing better what is visualized, and using a mesh when available (e.g., SMPL)

### Questions
- Could you comment on your choice of using generative AI, instead of directly using the data on which they are trained?
- Is it possible to provide an example on a more noisy point cloud, maybe discussing how much outliers cause degradation in the network prediction?

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
This paper proposes H2OFlow, a novel framework for learning comprehensive 3D human-object interaction (HOI) affordances, encompassing contact, orientation, and spatial occupancy. The method leverages a pre-trained 3D generative model to create a synthetic dataset, thus avoiding manual annotation. Its core innovation is a diffusion model trained to predict dense flow, a probabilistic, point-based representation of human interaction poses conditioned on an object's point cloud. In inference, sampling these flows generates a distribution of plausible interactions, from which the three affordance maps are statistically derived.

### Strengths
* The paper introduces an innovative framework that learns comprehensive affordances from synthetic HOI samples generated by 3D generative models. This approach cleverly eliminates the need for manual annotation and avoids the error-prone 2D-to-3D uplifting process used in prior work.
* A key contribution is the use of "dense diffused flows" as a probabilistic, point-based representation for human interaction. This design, learned via a diffusion model, elegantly circumvents the dependency on manifold meshes and is directly responsible for the method's outstanding robustness to the noisy and partial point clouds typical of real-world sensors. The superiority of this representation is further validated by strong experimental results and thorough ablations.

### Weaknesses
* The framework's performance is fundamentally tethered to the quality and diversity of a single upstream generative model (CHOIS). The paper appears to neglect any strategy for analyzing, filtering, or augmenting synthetic data. This raises critical questions: How does the model mitigate the risk of inheriting and amplifying potential biases from its sole data source—such as limitations in interaction patterns, insufficient object diversity, and unnatural poses? What are the upper limits of the affordance knowledge defined by CHOIS?
* The proposed method excels at exploring the general distribution of possible geometric interactions but lacks semantic control or decoupling mechanisms. This leads to two issues: (a) whether it is conditioned on specific task instructions (real-world scenarios should prioritize concrete interactions); (b) whether it can distinguish between objects that are geometrically similar yet functionally distinct. The model appears to learn more about geometric matching between humans and objects than a deeper, semantic understanding of functional accessibility.
* This method relies on a single standard human pose (H0) as a fixed baseline for flow prediction. However, significant variations in human morphology and dimensions directly influence how individuals interact with objects. Can the method be generalized to different people's post H0?  How would the predictability and reliability change if the standard pose H0 represented individuals of varying body types?

### Questions
See the Weaknesses

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
This paper presents H2OFlow, a framework for learning comprehensive 3D human-object interaction (HOI) affordances that goes beyond simple contact-based analysis. The authors address a key limitation in current affordance understanding methods, which typically focus only on contact regions while neglecting other critical aspects of human-object interactions, such as spatial orientation preferences and occupancy patterns.

The core motivation stems from the observation that human-object interactions involve rich 3D spatial relationships—for example, humans maintain characteristic distances and orientations when interacting with objects. Building on the concept of "comprehensive affordance" from Kim et al. (2024), which models probabilistic distributions over 3D spatial and orientational relations, the authors propose a novel approach that operates directly on point clouds rather than requiring mesh-based representations or 2D-to-3D uplifting.

Key contributions include:

A point-cloud-based affordance representation that captures both explicit contact and implicit non-contact interaction patterns from raw point cloud inputs
A synthetic data generation pipeline leveraging 3D generative models and dense diffusion flows (inspired by Eisner et al. 2022) to learn flexible affordances without manual annotations
A probabilistic formulation that operates directly on human-object point cloud pairs, eliminating the dependency on watertight meshes
The authors claim that H2OFlow generalizes effectively to real-world objects and outperforms existing methods that rely on manual annotations or mesh-based representations. The framework uses only synthetic data generated from 3D generative models, potentially addressing the scalability issues associated with labor-intensive dataset creation in this domain.

### Strengths
Originality: 
1. This paper introduces a novel paradigm shift from manual contact annotation to direct learning from point cloud datasets.
2. The integration of dense diffusion flows with 3D generative models represents an innovative approach to synthetic data generation that eliminates dependency on high-quality mesh inputs. 
3. It extends beyond the traditional binary contact-based affordance definition by incorporating spatial orientation and occupancy patterns, providing a more comprehensive and nuanced understanding of human-object interactions.

Quality: This paper achieves better qualitative and quantitative results on affordance learning compared with previous methods.

Clarity: The training method, inference time method, and dataset acquisition methods are clear.

Significance: This work addresses a critical challenge in computer vision, robotics, and AI by providing a more scalable and comprehensive approach to affordance learning. The elimination of manual annotation requirements and mesh dependencies has substantial practical implications for real-world applications. The comprehensive affordance representation (contact + orientation + spatial occupancy) offers significantly more flexibility for downstream tasks and could enable more sophisticated robotic manipulation and scene understanding capabilities. The potential impact extends beyond affordance learning to broader areas of 3D scene understanding and human behavior modeling.

### Weaknesses
1. Although it uses the comprehensive affordance representations for better affordance definition, all three representations are proposed in previous methods, lacking original definitions and contributions. Additionally, there is no ablation study on whether the two additional representations actually yield better results for affordance learning. 
2. Compared with only one previous baseline COMA, lacking experiments.
3. Most of the main figures are in the supplementary, indicating that the paper is not well organized.

### Questions
1. In Section 4.1, in the first sentence, the pretrained 3D model is used to generate meshes. However, in the last sentence of the same paragraph, the outputs are videos. So what does the pretrained model actually output?
2. Why use FPS to sample points? Have you tried other sampling policies?
3. For unseen objects, can you generate reasonable affordances for this object? For example, in Figure 5, most of the training data is human hands interacting with the object, and then for the unseen object, the chair, it also gives information about how hands interact with it. However, for chairs, maybe "Hips sitting on the chair" is a more common affordance. How can the method handle this situation?
4. Are the additional affordance representations (except contact) useful? Where is the ablation study about this?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies the problem of understanding the affordance of human object interaction. To this end, the authors propose a representation that captures not only contact affordance, but also orientational affordance and spatial affordance. The authors propose a pipeline to generate synthetic data using 3D generative models and employ a dense 3D-flow-based representation. The authors claim that, through extensive quantitative and qualitative experiments, the effectiveness and practical utility of the learned affordances on both synthetic datasets and real-world data are demonstrated.

### Strengths
1. The introduction of the affordance representation that captures both explicit contact and implicit non-contact interaction patterns is novel to me.

2. The problem the paper studies is important. If we want to move to spatial and physical AI in the future, it is important to understand human-object affordance.

### Weaknesses
1. The organisation of the paper needs to be improved. The overview figure as referred in Section 3 is very important for the understanding of the problem formulation, but the authors put it in the appendix. In Section 4, the authors describe a lot about their method, and it would be a lot better if there is a figure to demonstrate the whole method. 

2. Insufficient real-world results. The authors claim that their method can be well generalised to unseen real-world objects. It would be more convincing if the authors can provide quantitative results, given that at the moment there is only qualitative results.

### Questions
Please see the weaknesses part of my review. I would suggest the authors to address my concerns in the rebuttal.

### Soundness
2

### Presentation
2

### Contribution
2
