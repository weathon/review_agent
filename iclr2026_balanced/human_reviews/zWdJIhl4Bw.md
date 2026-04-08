## Human Reviewer 1

### Summary
This paper adapts VGGT that is trained for pinhole cameras to equirectangular panoramas via a projection-interface design. Concretely, it introduces (i) ray field alignment that embeds the ERP ray direction to image tokens, (ii) dual branch, head only LoRA (token LoRA + ray LoRA), and (iii) a latitude aware depth uncertainty loss. The backbone is frozen; only heads are adapted (<0.5\% trainable params). Experiments on a curated synthetic subset of Matrix 3D show the LoRA variant approaches full fine tuning at much lower compute.

### Strengths
+ Practical problem and clean formulation. Adapting VGGT to panoramic (ERP) imagery is a meaningful and practical problem. The paper clearly articulates the projection mismatch and addresses it with minimal modifications to the base model.
+ Simplicity and clarity. The proposed components are simple, modular, and easy to implement, making the method accessible for practitioners.
+ Efficiency. Head-only LoRA achieves performance close to full fine-tuning while using less than 0.5 % of parameters and dramatically lower training cost.

### Weaknesses
- Novelty is limited. While the technical components are tailored to the panoramic adaptation task, they rely on established mechanisms (direction encoding, spherical weighting, and parameter-efficient fine-tuning). Their combination here is primarily an engineering solution rather than a new methodological or theoretical contribution. The backbone, training pipeline, and geometric reasoning remain identical to VGGT. As a result, the work reads more like a clean application note of VGGT on panoramic imagery than a genuine research advance.
- Narrow empirical scope. Evaluation is restricted to a curated synthetic subset (2,196 scenes) of Matrix 3D; no real 360° benchmarks or other projection types (fisheye, catadioptric) are tested, weakening claims of generality for “panoramic world models.”

### Questions
- Can the authors evaluate on real-world 360° datasets (e.g., indoor ERP depth/pose benchmarks) to confirm transferability beyond synthetic data?
- The ablation table (Table 4) is difficult to interpret. It mixes module removals, hyper-parameter variations, and tuning schemes without clear grouping or visual hierarchy. There is no cue highlighting the main results, making it hard to see which components actually drive the performance gains. Reorganizing the table into separate sections (e.g., component ablations vs. efficiency studies) and emphasizing the primary metric would significantly improve readability.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper adapts the 3D foundation model VGGT for panoramic image inputs by introducing a method called Projection-Domain Adaptation. This technique is designed to resolve the uneven mapping between panoramic and perspective views, utilizing the ray-field as the core mechanism for transformation and alignment.
To account for the non-uniformity inherent in equirectangular projection, the method also incorporates latitude-aware depth uncertainty. The results demonstrate that the adapted model significantly outperforms its unadapted counterparts. Furthermore, the proposed finetuning strategy is proven to be both efficient and computationally low-cost.

### Strengths
1)	The paper is clearly motivated, focusing on the adaptation of existing perspective-view reconstruction models for panoramic imagery.
2)	The efficacy of the proposed methods is effectively demonstrated through comprehensive benchmarking results.
3)	The manuscript is well-organized and easy to follow, presenting a clear and logical progression of ideas.
4)	Most of the experimental results are well-organized and supported.

### Weaknesses
Overall, this paper is of good quality. I have some minor concerns to point out.
1) The paper provides inadequate details on its synthetic data generation, which is crucial for reproducibility and assessing the benchmark's validity.
•	It is unclear how the pinhole camera intrinsics are determined (e.g., are they constant, or sampled from a distribution?).
•	The paper does not state whether errors are modeled in the camera parameters.
•	Given that the experiments are purely synthetic, the benchmark's value hinges on its realism. The current design appears to overlook significant real-world artifacts common to panoramic videos, such as lens distortion and focal length inaccuracies, which limits its practical relevance.
2) The proposed "latitude-aware uncertainty" is presented without sufficient validation.
•	The paper provides no empirical evidence (e.g., an ablation study) to demonstrate that this uncertainty metric quantitatively correlates with per-pixel error.
•	Furthermore, the underlying assumption linking uncertainty purely to latitude seems fragile. It is questionable whether this assumption holds in dynamic scenarios involving significant camera rotation, which is a common use case for panoramic video.
3) The method for projecting panoramic images to perspective views is underspecified. The paper fails to define how the resolution of the projected plane is determined. This is a critical omission, as a coarse projection resolution could introduce significant discretization and reprojection errors, potentially confounding the evaluation results.
4) The paper's primary weakness is its limited technical contribution. The core methodology mainly leverages existing insights from the panoramic image processing literature. The use of LoRA is also a standard and well-established technique for training efficiency, not a novel component. Despite this lack of technical novelty, the work holds clear value from an engineering and application perspective.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper proposes a fine-tuning to the recent VGGT 3D transformer in order to adapt it to equirectangular projections (ERP) (i.e. panoramic images).
To do so, the authors propose a fine-tuning pipeline which consists of three main ideas: (i) embedding explicit 3D rays into token representations (ray-field alignment module), (ii) a "ray-enriched" LoRA fine-tuning consisting of two branches (for tokens and rays), and (iii) a latitude-aware depth uncertainty that takes into account the non-isotropic token statistics in ERP images to balance the spherical supervision.

The authors propose two main models, the first being a fully fine-tuned VGGT, and the second being a ray-enriched LoRA-finetuned VGGT.
Both models outperform other recent 3D reconstruction models (e.g. DUSt3R, VGGT) on depth estimation, camera pose estimation, and 3D point quality.
The LoRA fine-tuned model shows *almost* equivalent performance for notable gains in number of training parameters and training time.

### Strengths
- (s.1) The problem tackled in this paper, "Projection-Domain Adaptation", is well formulated and motivated for adapting perspective-trained 3D transformers to panoramic data. It is also well presented, especially in the discussion around varying ray distributions for different lens types (section 5).

- (s.2) The problem is addressed in a principled way, with design choices solving precise problems.

- (s.3) The overall pipeline is intuitive and justified via ablation studies.

- (s.4) The performances achieved by the proposed model are notable, especially on the LoRA fine-tuned variant, considering the training time and additional parameters required.

### Weaknesses
- (w.1) The proposed model is only evaluated on a single manually curated subset of Matrix-3D, which limits the conclusiveness of the results.
    - (w.1.q1) Why is this manual curation necessary? The authors state that it's because Matrix-3D contains many extreme long shots with high camera-to-building distances. What would happen if the model is trained and evaluated without this manual curation?
    - (w.1.q2) For reproducibility purposes, could the authors provide details regarding this manual curation either in the paper or by releasing their pre-processing code? (i.e. what scenes were selected exactly?)
    - (w.1.q3) While I understand that panoramic dataset are scarce, the evaluation test-bed is limited. Notably, could the authors provide evaluations on out-of-distribution performance (e.g. on Stanford2D3D [1], Matterport3D [2]) on one or more tasks? 

- (w.2) Certain sections could be improved through clearer and more precise writing. Here are some suggestions below.
    - In section 3.3, $\mathbf{x}_c(u,v)$ and $\mathbf{P}(u,v)$ should be defined. 
    - In equation (6), what does "$\oplus$" indicate? I assume concatenation but could the authors state this explicitly?
    - It would be clearer for the reader if the notations used in the text are integrated in figure 1 (e.g. $t_i^{(0)} (u,v)$, etc.).
    - In equation (7), the definition of $h$ should be clarified.

If the authors can address these concerns, I would be willing to re-consider my overall evaluation.

### Remark on ICLR formatting

According to the ICLR formatting guidelines, table numbers, titles, and captions must appear **above** the tables.
The authors **must** ensure that all tables conform to this requirement.
Additionally, there seems to be some space formatting errors in page 8. 
I strongly advise the authors to correct these errors.

### References 

[1] Armeni, I., Sax, S., Zamir, A., & Savarese, S. (2017). Joint 2D-3D-Semantic Data for Indoor Scene Understanding. ArXiv, abs/1702.01105.

[2] Chang, A., Dai, A., Funkhouser, T., Halber, M., Niessner, M., Savva, M., Song, S., Zeng, A., & Zhang, Y. (2017). Matterport3D: Learning from RGB-D Data in Indoor Environments. International Conference on 3D Vision (3DV).

### Questions
- (q.1) Are the authors planning to release the code and/or weights for their fine-tuning?
- See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
4

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper proposes a framework for adapting perspective-trained VGGT to Equirectangular Projection (ERP) inputs. The method is built on three components: (1) Ray-Field Alignment, (2) a head-only, Ray-Enriched LoRA, and (3) a latitude-aware depth uncertainty loss. Experiments on the synthetic Matrix-3D dataset show improvements over zero-shot VGGT and plain finetuning baselines, with the LoRA variant achieving comparable performance to full finetuning at substantially reduced computational cost.

### Strengths
1. Well-motivated problem formulation. The paper clearly identifies geometric inconsistencies when applying perspective-trained models to ERP inputs. The analysis of failure modes F1 (geometric-measure mismatch) and F2 (proxy focal problem) in Section 3.2 provides valuable insights into why naive coupling fails, offering a pedagogically sound diagnostic approach.

2. High Parameter Efficiency: The core practical contribution is demonstrating that head-only LoRA (0.5% of parameters) can achieve performance comparable to full finetuning. This is a valuable finding for resource-constrained scenarios.

3. Systematic ablation studies. Table 4 provides comprehensive ablations examining each component's contribution, effectively demonstrating that ray-field alignment and ERP-consistent lifting are critical while uncertainty modeling and gradient regularization provide incremental gains.

### Weaknesses
1. Limited Novelty and Architectural Justification: The framework is an integration of existing techniques (ray encoding, LoRA, spherical weighting) and lacks conceptual innovation. Crucially, the head-only LoRA design is not justified. The paper provides no ablations to support this choice over adapting other layers (e.g., in the backbone).

2. Lack of Real-World Validation: This is a critical flaw. The evaluation is confined to the synthetic Matrix-3D dataset. This limitation invalidates the claim of a "generalizable pathway" in Abstract, which is unsupported without validation on real-world benchmarks (e.g., Matterport3D [1]) and addressing the synthetic-to-real gap.

3. Missing Comparisons to Domain-Specific Methods: The paper fails to compare against direct competitors in panoramic-specific depth estimation (e.g., BiFuse [2], OmniFusion [3], HRDFuse [4]). These domain-specific methods are cited but critically omitted from the experiments. 

4. Incomplete 3D Reconstruction Evaluation: The evaluation of 3D quality is insufficient. To support the "world model" claims, it should be benchmarked against standard video-based reconstruction methods(e.g.,  video-to-3D models).

5.  Poor Clarity on Model Architecture: Key components from VGGT, such as the "camera token" and "register tokens," are used without explanation

[1] Chang, Angel, et al. "Matterport3d: Learning from rgb-d data in indoor environments." arXiv preprint arXiv:1709.06158 (2017).

[2] Wang, Fu-En, et al. "Bifuse: Monocular 360 depth estimation via bi-projection fusion." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.

[3] Li, Yuyan, et al. "Omnifusion: 360 monocular depth estimation via geometry-aware fusion." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

[4] Ai, Hao, et al. "Hrdfuse: Monocular 360deg depth estimation by collaboratively learning holistic-with-regional depth distributions." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

### Questions
Please see the critical issues and required clarifications detailed in the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
4