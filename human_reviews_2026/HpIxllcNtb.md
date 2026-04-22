# MoGen: Detailed Neuronal Morphology Generation via Point Cloud Flow Matching

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Biological neurons come in many shapes. High-fidelity generative modeling of their varied morphologies is challenging yet underexplored in neuroscience, and crucial for the subfield of connectomics. We introduce MoGen (Neuronal Morphology Generation), a flow matching model to generate high-resolution 3D point clouds of mouse cortex axon and dendrite fragments. This is enabled by an adaptation that injects local geometric context into a scalable latent transformer backbone, allowing for the generation of high-fidelity, realistic samples. To assess MoGen's generation quality, we propose a dedicated evaluation suite with interpretable geometric and topological features tailored to neuronal structures that we validate in a user study. MoGen's practical utility is showcased through controllable generation for visualization via smooth interpolation and a direct downstream application: we augment the training set of a shape plausibility classifier from a production connectomics neuron reconstruction pipeline with millions of generated samples, thereby improving classifier accuracy and reducing the number of remaining split and merge errors by 4.4%. We estimate this can reduce manual proofreading labor by over 157 person-years for reconstruction of a full mouse brain.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents MoGen, a flow matching model for generating high-resolution 3D neuronal morphologies as point clouds. The authors injects local geometric context into a scalable transformer backbone, significantly improving the fidelity of fine details. The authors validate their model with a custom, interpretable evaluation suite and, most importantly, demonstrate its practical utility by augmenting a neural reconstruction pipeline's training data, which leads to a measurable reduction in neuron reconstruction errors.

### Strengths
1. The paper employs a state-of-the-art generative architecture, underpinned by a well-motivated modeling insight. The introduction of local geometric context effectively addresses a critical information bottleneck, leading to a demonstrable and significant improvement in the fidelity of generated morphological details.
2. The proposed evaluation suite is a notable contribution in itself. It moves beyond inadequate standard metrics by introducing a set of biologically salient and highly interpretable features that are tailored to the domain-specific properties of neuronal structures, a claim robustly validated through a user study.
3. The work provides a comprehensive and rigorous ablation study that systematically dissects the contribution of each core component. This thoroughly validates the necessity of the proposed architectural adaptations, the model's capacity, and the training schedule.

### Weaknesses
1. **The generated data represents a surface/volumetric structure rather than a topological tree.** The outputs of this model are dense point clouds representing surface geometry, which diverges from the classical definition of a neuronal morphology as a binary tree structure (skeleton) [1,2,3,4]. If the authors' goal is to generate detailed cellular surfaces with volume，the work should be more appropriately positioned and compared against prior works that also generate volumetric cell structures, such as [5]. 
2. **The evaluation protocol for topological features may be compromised by undersampling.** The authors compute two key topological metrics based on a Minimum Spanning Tree (MST) constructed from a heavily subsampled point cloud of only 256 points. This sparse sampling density is likely insufficient to faithfully capture the intricate branching patterns of neurites, potentially leading to the erroneous merging of distinct branches and a significant loss of fine-grained branch morphology details[2]. The validity and sensitivity of these metrics are therefore questionable.
3. **There is no validation of the fundamental biological validity of the generated structures.** A core anatomical property of neurons is that their arborization (excluding the soma) forms a binary tree[6]. The authors do not provide any evaluation to guarantee that the generated point clouds adhere to this essential topological constraint. 

**Reference**

[1] Cuntz, Hermann, et al. "One rule to grow them all: a general theory of neuronal branching and its practical application." *PLoS computational biology* 6.8 (2010): e1000877.

[2] Laturnus, Sophie C., and Philipp Berens. "MorphVAE: Generating Neural Morphologies from 3D-Walks using a Variational Autoencoder with Spherical Latent Space." *International Conference on Machine Learning*. PMLR, 2021.

[3] Yang, Nianzu, et al. "MorphGrower: A Synchronized Layer-by-layer Growing Approach for Plausible Neuronal Morphology Generation." *International Conference on Machine Learning*. PMLR, 2024.

[4] Zhu, Tianfang, Hongyang Zhou, and Anan Li. "MorphoGen: Efficient Unconditional Generation of Long-Range Projection Neuronal Morphology via a Global-to-Local Framework." *Proceedings of the IEEE/CVF International Conference on Computer Vision*. 2025.

[5] Wiesner, David, et al. "Implicit neural representations for generative modeling of living cell shapes." *International Conference on Medical Image Computing and Computer-Assisted Intervention*. Cham: Springer Nature Switzerland, 2022.

[6] Peng, Hanchuan, et al. "Morphological diversity of single neurons in molecularly defined cell types." *Nature* 598.7879 (2021): 174-181.

### Questions
1. The cells generated in this paper seem much smaller than those in other papers like MorphGrower (as shown in Figure 2). Why is there such a big difference in size? Both are based on mouse brain data, but your generated structures look about ten times smaller.
2. The experimental results show that using the generated data as negative samples for SHAPE classifier training leads to performance improvement. However, if the generated 3D point clouds were highly realistic and biologically plausible, introducing them as *negative* examples would be expected to confuse the classifier and cause performance degradation. The positive result suggests that the generated data **is not biologically realistic**. Instead, it provides a useful set of specific, artificial patterns for the classifier to learn.
3. What is the biological justification for using the mean x-coordinate as a conditioning feature? The manuscript indicates an interpolation range from -0.75 to 0.75. Were all neuronal fragments decentralized during pre-processing? The absolute position of a neuron in the brain is tied to its specific brain region and is unlikely to be confined to such an arbitrary, normalized range. Please clarify the pre-processing steps and the neuroanatomical meaning of this feature.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This manuscript introduces a flow matching generative model for synthesising high-fidelity neurite (axon, dendrite) fragments in the mouse cortex using 3D point clouds. The model adapts the PointInfinity architecture to better model neuronal fragment structures at high resolution, whilst minimizing undesirable discontinuities. This is achieved by injecting local geometric context into the latent cross-attention mechanism that compresses individual point tokens to extract global geometric structures.  The primary practical contribution of this work is to generate additional axon fragment samples to train and improve the performance of a shape plausibility classifier, enabling more reliable merging decisions during neuron reconstructions from EM data.

### Strengths
-	The idea of using 3D point clouds for high-fidelity neurite fragments generation is novel and interesting, intuitively suited to the sparse yet expansive volume of the neurites.

-	This is a promising application that could be utilized to improve the final performance of a production-level reconstruction pipeline.

-	The paper is generally well structured and presents the arguments in a logical order. The generation and interpolation visualisations demonstrate the effectiveness of MoGen.

### Weaknesses
-	The motivation of this paper is unclear. The paper would benefit from a detailed justification for its narrow focus on neurite fragment generation, neglecting the importance of full neuron topology in characterising neuron identity. From my domain knowledge, in single-neuron classification tasks, the overall structural organization of axons and dendrites often provides more discriminative information than local surface geometry. 

-	The value of interpolating from axon to dendrite in the morphological space needs more motivation from a neuroscience perspective. Could the authors explain why this is beneficial?  Biologically, axons would not develop into dendrites and vice versa. Do the smoothly interpolating axon and dendrite morphologies tend to co-exist in real neurons?  Please provide more ablation studies and analysis to address this concern.

-	In the ablation study, the authors validate the effectiveness of classifier-free guidance (cfg). However, flow-matching generative models conventionally use cfg<1.0 to indicate that it doesn’t use the cfg mechanism for sampling. Please explain this further.

-	It is unclear how the axons were categorised into “positive” and “negative” types in Section 4.1. It seems to suggest that negative cases are created by combining real axon morphologies in implausible ways. Additional visualization of these comparisons could help to understand.

-	This manuscript only compares different configurations for the component of their proposed method. How are the comparison results of the existing methods, such as MorphVAE, MorphGrower, and MorphOcc?

-	The downstream application task on improving the SHAPE classifier only supports the usefulness of generated axon fragments and neglects the arguably more intricate dendritic morphologies. Since assessing the model’s impact on a downstream system is another way to validate the generation quality of the neurite fragments, excluding the synthetic dendrites from such evaluation leads to a lack of more rigorous validation for its generation quality, beyond the custom metric via the MMD. The authors may provide further justification for this potential problem.

### Questions
•	Why is MoGen only used to generate negative axon samples for the downstream SHAPE co-training task? This seems to suggest that there are enough positive axon examples for training. In this case, they should support a sufficient number of implausible combinations to produce enough negative samples, making the usage of MoGen seem redundant here?

•	Following from the above question, Page 18 states, “In early experiments we found that the downstream benefits are maximized when co-training with synthetic examples added to the negative class only”. Why is this the case? Does this not suggest a lack of fidelity in neuron fragments generated by MoGen? 

•	The justification for the 10% negative sample ratio in the downstream task is not clearly expressed. Specifically, what does it mean by “this class of erroneous merges”? Are there different classes of negative examples? If so, why do 10% of the negative examples originate only from one class?  

•	Please explain how you define the classifier-free guidance mechanism. Why are there <1.0 values on the cfg?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Paper introduces flow-matching method that allows to generate 3D point cloud neuron morphologies by adapting and augmenting standard flow-matching methods e.g. used in image generation. They show significantly improved generations compared to prior work and demonstrate downstream simplifications in human annotation in neuron reconstruction.

### Strengths
- Well motivated problem
- Well written paper
- Thorough methods

### Weaknesses
- Rather niche and not of broad relevance to ML community. However, I am unsure whether this is a relevant weakness.

### Questions
what are other relevant use cases apart from synthetic data generation for tool improvement?

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
3

### Summary
This paper introduces MoGen, a flow-matching generative model for synthesizing high-fidelity 3D neuronal fragments, including axons and dendrites, represented as point clouds. The method adapts the PointInfinity architecture by injecting local geometric context into the latent cross-attention module, improving structural coherence and reducing discontinuities in generated morphologies. The model is trained on EM-derived fragments and evaluated using a custom set of interpretable geometric and topological metrics tailored to neuronal structures. Its main practical contribution lies in augmenting training data for a shape plausibility classifier within the PATHFINDER connectomics pipeline, which enhances automated neuron reconstruction performance.

### Strengths
- The paper is clearly written and well-structured, with informative visualizations that effectively illustrate the concepts and generation results.
- The integration of MoGen into the PATHFINDER connectomics pipeline and the demonstrated reduction in reconstruction errors show practical real-world impact.
- The evaluation suite provides interpretable and domain-relevant validation, and its consistency with expert judgments through a user study supports the reliability of the results.

### Weaknesses
- Lack of proper discussion with previous 3D neuron synthesis [1] and point cloud representation [2] methods.
- The neuron generation method MoGen is only limited for training shape plausibility classifier within the PATHFINDER setting. How about its application in broader interest such as boosting segmentation/reconstruction directly? 
- Since the dataset originates from a specific mouse cortical volume, it is uncertain about MoGen's generalization ability across different datasets.


[1] Tang, Zihao, et al. "3D conditional adversarial learning for synthesizing microscopic neuron image using skeleton-to-neuron translation." 2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI). IEEE, 2020.

[2] Zhao, Runkai, et al. "PointNeuron: 3D neuron reconstruction via geometry and topology learning of point clouds." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2023.

### Questions
Please see weakness.

### Soundness
3

### Presentation
3

### Contribution
2
