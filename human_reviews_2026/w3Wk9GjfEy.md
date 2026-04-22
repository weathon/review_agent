# Hierarchical Scoring with 3D Gaussian Splatting for Instance Image-Goal Navigation

- Avg Score: 5.20
- Decision: Reject
- Scores: 4, 6, 4, 6, 6

## Abstract
Instance Image-Goal Navigation (IIN) requires autonomous agents to identify and navigate to a target object or location depicted in a reference image captured from any viewpoint. While recent methods leverage powerful novel view synthesis (NVS) techniques, such as 3D Gaussian splatting (3DGS), they typically rely on randomly sampling multiple viewpoints or trajectories to ensure comprehensive coverage of discriminative visual cues. This approach, however, creates significant redundancy through overlapping image samples and lacks principled view selection, substantially increasing both rendering and comparison overhead. In this paper, we introduce a novel IIN framework with a hierarchical scoring paradigm called GauScoreMap that estimates optimal viewpoints for target matching. Our approach integrates cross-level semantic scoring, utilizing CLIP-derived relevancy fields to identify regions with high semantic similarity to the target object class, with fine-grained local geometric scoring that performs precise pose estimation within promising regions. Extensive evaluations demonstrate that our method achieves state-of-the-art performance on simulated IIN benchmarks and real-world applicability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes GauScoreMap for Instance Image Navigation (IIN). The method reconstructs a 3D Gaussian Splatting scene, “lifts” CLIP visual features into the 3D domain, then applies a global semantic scoring phase based on class-text embeddings to identify candidate regions, followed by a local geometric scoring phase that uses cross-attention between ray features and target-image DINOv2 features for fine-grained pose estimation. Experiments on the HM3D IIN validation set report the best SR (0.784) and SPL (0.605), supported by ablations, efficiency studies, and limited real-world demonstrations.

### Strengths
**Hierarchical scoring reduces redundancy**

* The global–local scoring structure effectively filters candidate regions via CLIP-based text–image similarity before precise geometric refinement, significantly reducing computation and redundancy.

**Empirical advantages over strong baselines**

* On the HM3D IIN benchmark, the method surpasses GaussNav by +5.9% SR and +2.7% SPL, demonstrating consistent gains under the same 3DGS representation.
* The method remains robust when 20–60% of target-region Gaussians are removed, showing resilience to incomplete reconstructions.

### Weaknesses
**Mathematical clarity and consistency issues**

* The intermediate formula between Eq. (5) and Eq. (6), ( $\ell = \max((O - v_o) \cdot v_d, , 0)$ ), contains a typographical comma error and unclear notation.

**Strong dependence on object category detection**

* The global stage depends on Mask R-CNN for class labels; failure in detection or class confusion among nearby instances may bias candidate generation, yet no failure or robustness analysis is reported.

**Lack of reproducibility and openness**

* No code or pretrained models are provided, and there is no stated plan or timeline for release.

### Questions
* How do these training settings influence convergence speed or stability?
* Have the authors observed any failure patterns when CLIP’s text embeddings misalign with visual attributes?
* Would introducing instance-level image embeddings, rather than purely text-based class embeddings, help reduce bias?
* Are there plans to release the implementation and pretrained weights? If so, when?
* How sensitive is the performance to Gaussian count, rendering resolution, or backbone selection (e.g., DINOv2 vs. CLIP)?
* Beyond empirical improvement, what conceptual or theoretical insight does GauScoreMap contribute regarding representation alignment between 2D semantics and 3D geometry?
* How does this approach advance understanding of the trade-off between global semantic priors and local geometric consistency in 3D neural fields?
* Could the authors share any empirical or theoretical insights explaining why this two-stage design improves over a unified cross-modal model?
* Finally, could you explicitly articulate the motivation behind the hierarchical scoring and the insights gained?

### Soundness
2

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
3

### Summary
This paper addresses the task of Instance Image-Goal Navigation (IIN), where an embodied agent must locate and navigate to a target object depicted in a single reference image captured from an arbitrary viewpoint.
While recent approaches have adopted 3D Gaussian Splatting (3DGS) to represent scenes with continuous geometry and photorealistic detail, they typically rely on dense or random multi-view sampling for target matching, resulting in redundant renderings and high computational costs.

To overcome these limitations, the authors propose GauScoreMap, a hierarchical scoring framework that integrates both semantic and geometric reasoning within 3DGS-based navigation. The method first performs global semantic scoring using CLIP-derived relevance fields to identify candidate object regions, and then conducts local geometric scoring through a learned ray-image cross-attention model (based on DINOv2 features) to estimate the 6D camera pose corresponding to the target image. This design effectively eliminates redundant sampling, significantly improving both navigation success and efficiency.

### Strengths
### Practical motivation

The paper addresses a bottleneck in 3DGS-based IIN, i.e., viewpoint redundancy and heavy computation, and proposes a principled hierarchical solution. The decomposition into global semantic localization and local geometric refinement is elegant and intuitively aligned with how humans perform visual search.

### Technically reasonable design

The integration of CLIP for high-level semantics and DINOv2-based cross-attention for fine-grained matching is well justified and methodologically coherent. The ``feature lifting'' procedure, which aggregates 2D CLIP features into 3D Gaussians, is also consistent with recent literature such as LUDVIG (Marrie et al., 2024).

### Comprehensive experiments

The method achieves state-of-the-art performance on the Habitat-Matterport3D benchmark (SR 0.784 / SPL 0.605), surpassing the strong GaussNav baseline by 5.9% in SR.
Ablation studies clearly isolate the contribution of each component: removing global semantic scoring or local geometric scoring leads to substantial drops (−17.6% and −36.3% in SR, respectively). Efficiency results and robustness to partial scene deletion further strengthen the claim.

The paper also demonstrates the method on a real-world Unitree G1 humanoid robot, utilizing LiDAR-based Gaussian reconstruction, which shows good transferability and suggests potential for embodied deployment.

### Weaknesses
### Positioning against existing semantic-augmented 3DGS

The use of CLIP-derived semantics in Gaussian representations is becoming common, as seen in Gaussian Grouping (Ye et al., ECCV 2024) and LUDVIG (Marrie et al., 2024).
I believe that the novelty of this paper lies not merely in semantic embedding, but in its hierarchical fusion of global semantics and local geometry for viewpoint optimization in navigation.

The authors might emphasize more explicitly that existing semantic 3DGS works target scene segmentation or editing, while GauScoreMap tackles cross-view instance localization and pose estimation, a substantially different objective. To this end, the authors could highlight the technical differences between existing Gaussian-grouping-like methods and the proposed one more clearly. 

### Dynamic or partially-captured environments
As stated in the limitation section, the method assumes static environments, and dynamic or partially captured environments remain unaddressed. Although the authors mention this as future work, a discussion of possible online or incremental extensions (like NeRF-SLAM or Magic-SLAM) would be helpful.

### Questions
### Novelty to semantic 3DGS (e.g., Gaussian Grouping)

Could the authors clarify whether those methods could, in principle, be adapted for IIN, and what challenges would arise? For example, segmentation-oriented semantics might lack the geometric correspondence reasoning required for 6D pose estimation.

On generalization:
Since CLIP and DINOv2 are trained on large-scale internet data, how well does GauScoreMap generalize to unseen object categories or visually ambiguous instances in indoor scenes?

### Computation & scalability
While the paper reports time and VRAM usage, can the method scale with the larger number of Gaussians, i.e., larger-scale environments?

### Soundness
4

### Presentation
2

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
The paper targets Instance Image-Goal Navigation (IIN): given a reference image of a target, the agent must find it in the environment. Recent IIN methods that pair 3D Gaussian Splatting (3DGS) with random/massive viewpoint sampling waste rendering and comparison budget. The authors propose GauScoreMap, a hierarchical scoring pipeline:

(1) a semantic stage uses CLIP-derived relevancy fields to highlight promising regions/views;

(2) a local geometric stage does fine-grained pose estimation within those regions to finalize the match. They claim SOTA on simulated IIN benchmarks and show real-world viability.

### Strengths
Reframes IIN over 3DGS as a view selection problem with semantic→geometric two-level scoring rather than brute-force rendering.

Reports SOTA on simulated IIN (HM3D/Habitat) and demonstrates real-world deployment (humanoid platform).

### Weaknesses
1. Section 3.4.1 Local scoring for region selection — clarity & notation

  a. Motivation/examples for ray selection. Could you add a short motivation and one concrete example of what kinds of rays are expected to receive high scores vs. low scores? 

  b. what is “ground-truth rays” and “ground-truth scores"

  c. Please clarify which tensor is the query and which are key/value in the cross-attention (Eq.5)

  d. Case and notation consistency. In 3.4.1, there are mixed upper/lower-case usages for the same symbols. Some symbols have subscripts while others don’t—please clarify. 

2. Missing sensitivity ablations on Hyperparameters, Global/Local score introduce some hyperparameters, how they affect final performance?

### Questions
1. Why rely on Mask-R-CNN for class labels? What about unseen classes?

2. Can you provide the time and hardware for 3D Gaussian reconstruction per HM3D episode, along with basic quality metrics?

3. How does the pipeline disambiguate when several objects of the same class appear?

### Soundness
3

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
4

### Summary
This paper proposes a new hierarchical scoring paradigm called GauScoreMap, which estimates optimal viewpoints for target matching for the Instance Image-Goal Navigation (IIN) task. Specifically, the global semantic scoring leverages CLIP embedding similarity to identify coherent regions with high semantic similarity to the target object class. Then the local geometric scoring employs a two-stage approach: first, perform region-level scoring by comparing sampled rays from candidate regions with the goal image’s DINOv2 features through cross-attention, then conduct precise pose estimation within the most promising region. The experiments on the simulation and real-world benchmarks demonstrate the effectiveness of the proposed method.

### Strengths
1)	The research topic on the IIN task is valuable, the authors aim to improve the existing 3DGS-based IIN method by introducing a hierarchical scoring approach, which is reasonable.
2)	In the hierarchical scoring approach, both high-level semantic alignment and fine-grained geometric matching are utilized to recognize the target area, which obviates the need for exhaustive or random sampling through the environment in existing methods.
3)	The experiments on both simulation and real-world benchmarks demonstrate the effectiveness of the proposed method.

### Weaknesses
1)	The memory cost of saving the CLIP feature Gaussian field for a large-scale IIN task may be too large. From table 3, it seems that only the memory cost of 3DGS is compared, so is the CLIP feature field included?
2)	About the time efficiency, does the hierarchical scoring cost more time than existing methods during inference? Since there is no comparison of inference time and memory cost.
3)	In sec. 4.5, I am curious about why deleting so many Gaussians can still localize the target object? Does it mean there are many redundant Gaussians? How do you remove the Gaussians?

### Questions
Please try to address the weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a method to leverage a 3D Gaussian Splatting representation efficiently in the context of Instance Image-Goal Navigation (IIN), where an agent must navigate to an object given a picture of it. Previous methods leveraging novel view synthesis methods either rely on a discrete representation of the scene, e.g. a graph, or use a fully continuous 3D representation but query it inefficiently with some random sampling-based strategies to find the location of the target object. This work proposes to use a 3D continuous representation to avoid constraints related to discretisation (e.g. a limited number of viewpoints per location) but also to query it more efficiently based on a hierarchical scoring method. The latter follows multiple steps: (i) promising regions are detected based on CLIP-feature similarity, then (ii) a local geometric scoring is applied to promising locations. The local geometric scoring can also be decomposed into 2 steps, i.e. region-level scoring followed by fine-grained pose estimation in the top-1 retrieved region. Such a method allows authors to reach satisfying performance on IIN while being more computationally efficient than counterpart methods.

### Strengths
* S1: The limitations of previous work are clearly presented and the proposed contributions are thus properly motivated.
* S2: The proposed method is clearly explained, and leads to high performance and more efficient runtime than other methods.
* S3: Real-world experiments are conducted (in appendix), which is appreciated.

### Weaknesses
- W1: [Major] The proposed method requires a first exploratory rollout to build the scene representation, which is a quite important limitation. However, authors conduct some experiments where they evaluate the performance of their approach from partial scene representations, simulating an unfinished exploration of the scene. This is emulated by randomly pruning gaussians. Unfortunately, this does not exactly simulate incomplete scene exploration as whole parts of the scene would be unknown.
- W2: [Major] Authors are using CLIP and DINOv2 features in the first and second stages of their method respectively. Their should be some experimental evidence about why each backbone is better in each step of the process.
- W3: [Minor] Authors report the performance of baselines from previous work so this is not a direct weakness of this paper. However, they should discuss a bit more the lack of fairness in comparison with MultiON baselines that are only given the semantic category of the target object as input.
- W4 [Minor] The real-world experiment is appreciated, but should be moved to the main paper for a more direct access to it.

### Questions
- Q1: [Related to W1] Are other baselines this method is compared to also requiring a first explorative rollout of the scene before navigating towards the goal? Is the explorative rollout taken into account when computing the SPL metric?
- Q2: [Related to W1] To evaluate the robustness of the method to incomplete scene exploration, could authors rather either optimise different 3DGS representations from only the first N collected views, varying N? This would be a much more convincing experiment.
- Q3: [Related to W2] Could authors ablate the relevance of CLIP and DINOv2 features in the 2 different steps of their process?
- Q4: [Related to W3] Could authors elaborate a bit more on the limitations of the comparison with MultiON baselines?

### Soundness
3

### Presentation
3

### Contribution
3
