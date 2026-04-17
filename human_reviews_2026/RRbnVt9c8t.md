# Pulp Motion: Framing-aware multimodal camera and human motion generation

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 4

## Abstract
Treating human motion and camera trajectory generation separately overlooks a core principle of cinematography: the tight interplay between actor performance and camera work in the screen space. 
In this paper, we are the first to cast this task as a text-conditioned joint generation, aiming to maintain consistent on-screen framing while producing two heterogeneous, yet intrinsically linked, modalities: human motion and camera trajectories. 
We propose a simple, model-agnostic framework that enforces multimodal coherence via an auxiliary modality: the on-screen framing induced by projecting human joints onto the camera. This on-screen framing provides a natural and effective bridge between modalities, promoting consistency and leading to more precise joint distribution.
We first design a joint autoencoder that learns a shared latent space, together with a lightweight linear mapping from the human and camera latents to a framing latent. We then introduce Auxiliary Sampling, which exploits this linear map to steer generation toward a coherent framing modality. 
To support this task, we also introduce the PulpMotion dataset, a camera-motion and human-motion dataset with rich captions, and high-quality human motions.
Extensive experiments across DiT- and MAR-based architectures show the generality and effectiveness of our method in generating on-frame coherent camera-human motions, while also achieving gains on textual alignment for both modalities. Our qualitative results yield more cinematographically meaningful framings setting the new state of the art for this task.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the problem of jointly generating human motion and camera trajectories from textual descriptions. The authors argue that treating these modalities separately often leads to poor cinematic results (e.g., the camera losing track of the actor). They propose a novel framework that uses "on-screen framing" (the 2D projection of human joints onto the camera plane) as an auxiliary modality to bridge the gap between the two. They also contribute "PulpMotion," a large-scale dataset with paired camera-human motions and rich captions for both.

### Strengths
- The core idea of using on-screen 2D framing as the "bridge" between 3D camera and 3D human motion is cinematographically sound.
- I appreciate that the authors tested this on two different state-of-the-art architectures (DiT and MAR). This convincingly demonstrates that their sampling method is model-agnostic and broadly applicable
- The PulpMotion dataset seems like a significant step up,, particularly with the motion refinement steps to fix occlusions and the inclusion of detailed VLM-generated captions for both modalities.

### Weaknesses
-  The method relies heavily on the linear transform W to map from the joint (x,y) latent space to the framing latent z. While the results are good, I wonder if this linear assumption is too restrictive for highly complex camera moves or unusual perspectives where the relationship between 3D positions and 2D projections might be highly non-linear, even in a latent space.

- The evaluation heavily relies on FD metric in various feature spaces. While standard, these can sometimes be disconnected from perceived quality. Qualitative video examples are necessary to fully trust that the "refined" motions in the new dataset don't contain artifacts from the diffusion inpainting.

- Quantitative results: Authors present metrics only for their own trained models. There is no comparison against other baselines/SOTA works that: 
(1) Create human motion given a camera trajectory - Report motion metrics and in-frame joints here. 
(2) Create camera trajectories given a human motion - Report camera trajectories metrics and in-frame joints here.

- Videos are required: A motion generation paper is highly expected to also present videos showing its results. I expect the authors to submit: 
(1) Videos that show the steps of MotionPulp dataset generation, including the 3D pose estimation and camera trajectories + dataset post-processing to reach the final result.
(2) Videos of the generated results from the trained model. 
(3) Comparison against other SOTA methods.

### Questions
1. Did you experiment with non-linear mappings for W? I am curious if a simple MLP would provide better framing guidance than the linear transform, or if it would break the elegant decomposition used in the sampling derivation.

2. Regarding the dataset refinement: When you use diffusion inpainting to fix out-of-frame body parts, how often does this result in plausible but essentially "hallucinated" motions that might not match the original real-world action?

3. "Since the final sampling does not explicitly condition on z, there is no need to include the bridging modality during training. This reduces training cost and yields a more general approach." (L264) - If this is your conclusion, why W and Dec_z are needed? You can drop those from your architecture and paper.

4. Are you planning to release the PulpMotion dataset and the model for reproducibility?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper addresses an interesting problem of generating human motion in conjunction with camera trajectories. The paper describes a method for this generation, along with a dataset to be published for training and testing. While this all sounds interesting, there is no material to demonstrate the quality of the data, the method, or the generated results, and hence evaluating the quality of the idea and its implementation is unfortunately impossible at this stage it seems.

### Strengths
Ideas are good. There seems to be a large effort put into this work.

### Weaknesses
There are some questions regarding the implementation. For example, how well does the inpainting motion idea of occluded footage looks like, and what is the overall quality of the data in terms of foot skating etc.
There is not way to evaluated the quality unfortunately

### Questions
Am I missing something?

### Soundness
1

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
3

### Summary
This paper presents two main contributions.
First, it proposes an auxiliary projection (Aux) method that enhances cross-modal alignment in joint human–camera generation. The approach introduces a mathematically motivated projection mechanism that can be plugged into existing diffusion frameworks to improve multi-modal consistency without architectural changes.
Second, the paper introduces PulpMotion, a large-scale dataset containing paired 3D human motions, camera trajectories, and textual captions. Compared with previous datasets such as E.T., PulpMotion offers greater scale, richer modality coverage, and improved motion quality, providing a valuable resource for future research.
Experiments demonstrate that the proposed method achieves consistent performance gains across multiple metrics and subsets.

### Strengths
1. The proposed auxiliary projection method is novel and well-motivated, offering a clear mathematical way to improve cross-modal alignment by decomposing the latent space and re-projecting noise predictions, and it could be applied to other multi-modal diffusion frameworks.
2. The method shows consistent, measurable improvements across metrics like framing accuracy, camera-text alignment, and motion consistency, with robust gains on both pure and mixed subsets.
3. The PulpMotion dataset is large and high-quality, combining 3D human motion, camera trajectories, and detailed text captions, roughly twice the size of previous datasets, which is  valuable for future multi-modal and generative research.

### Weaknesses
1. Lack of qualitative visualization, analytical ablations and comparison. (1) The paper presents extensive quantitative results but lacks visual comparisons of generated outputs across baselines like ReDi and dual-modality generation (x,y). Without qualitative visualization, it is difficult to judge whether the reported improvements in metrics correspond to genuinely better visual framing or motion quality. (2) While the paper compares against multi-modal frameworks such as ReDi and its own variants, it does not evaluate performance against established single-modality models like text-to-motion or camera-only baselines. Even if joint models might not surpass unimodal ones on single-modality metrics, such comparisons would make the results more convincing and provide clearer explanations of the trade-offs between alignment and fidelity in multi-modal generation. (3) Although the auxiliary projection P_{//}​ is mathematically well-motivated, the paper provides little empirical insight into how it improves framing or cross-modal alignment. The ablations only vary sampling weights (w_c, w_z​) and modality setups, without deeper analysis of the mechanism. Including a simple latent-space visualization (e.g., t-SNE before and after applying Aux) would help illustrate what representation changes the auxiliary term enforces. At present, the improvement appears empirical rather than well-explained.
2. The dataset itself includes human motions synthesized or refined by a diffusion-based editing model (Fig. 10). However, there is no quantitative or qualitative validation of the realism or accuracy of these generated motions. Since they serve as ground-truth for training and evaluation, this raises concerns about data fidelity.

### Questions
1. Can the proposed auxiliary sampling method be applied to ReDi, and would it bring similar improvements?
2. What exactly do the “pure” and “mixed” subsets in PulpMotion represent, and how are they separated?
3. What are the sources of the raw videos in PulpMotion, and were any filtering or selection criteria applied during dataset construction?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work examines the problem of joint synthesis of human motions and camera motions. To train the model, the paper introduces a dataset of containing samples of human motions, camera motions, and text descriptions which are gathered from videos of human motions using SOTA pose, camera motion, and captioning models. The generation model consists of two parts: an autoencoder which encodes human motions and camera motions into a latent space, and a diffusion model which generates camera and motion latents which are decoded into the final camera and motion samples. The autoencoder also models the on-screen framing in the latent space as a linear transformation of the human and camera latents. A variant of CFG that is based on orthogonal projection on the kernel space of the framing latent linear projection is introduced to promote strong alignment with the text description. Experiments are conducted to investigate the ability of the proposed model to jointly generate human motion and camera motions given a text description.

### Strengths
* Joint generation of human and camera motion is an important problem which could lead to generative models that are suitable for achieving cinematic or artistic effects.
* The proposed dataset has a large number of samples covering diverse situations. The proposed dataset is the first to focus equally on the gathering and annotating both poses and camera motions.
* Experimental results show that the proposed dataset can be used to learn joint human and camera motions.

### Weaknesses
* The submission did not include example videos of the collected data or samples in a supplementary material file. This makes it very hard to judge the quality of the data or the sampling results. Given the lack of established metrics in this domain, human visual inspection remains the most reliable way to determine quality.
* The importance of the framing variable $z$ is unclear to me. As far as I can tell, $z_{raw}$ is never defined in the text. Furthermore, z only appears to affect the final sampling result via the encoder, and does not influence the training of the diffusion model beyond this. In the comparisons in Figure 4 and 5, including z generally reduces the quality of results compared only using x and y when auxiliary sampling (essentially a variant of CFG) is not used. I do not feel that the importance of z has been fully validated.
* I am not sure if the importance of the auxiliary sampling in equation (8) is fully validated compared to standard CFG in equation (4). I did not see an experiment comparing the effect of (8) vs. (4) given the same trained model. Am I missing this, or can it be added? Furthermore, was CFG used for the baseline models in Section 5.2? If not, the comparison with the proposed method is not fair because guidance makes a major impact on sampling results.

### Questions
* Can visualizations of data samples and model samples be shown for qualitative analysis?
* Why does including z make results worse in Figure 4 and 5, but better when auxiliary sampling is used?
* Can you compare auxiliary sampling to standard CFG for the same trained model?
* Was CFG used for the baseline models in Figure 4 and 5?
* What is the definition of $z_{raw}$?

### Soundness
2

### Presentation
2

### Contribution
2
