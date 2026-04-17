# Animating the Uncaptured: Humanoid Mesh Animation with Video Diffusion Models

- Decision: Accept (Poster)
- Scores: 4, 10, 8, 2

## Abstract
Animation of humanoid characters is essential in various graphics applications, but require significant time and cost to create realistic animations. We propose an approach to synthesize 4D animated sequences of input static 3D humanoid meshes, leveraging strong generalized motion priors from generative video models -- as such video models contain powerful motion information covering a wide variety of human motions. From an input static 3D humanoid mesh and a text prompt describing the desired animation, we synthesize a corresponding video conditioned on a rendered image of the 3D mesh. We then employ an underlying SMPL representation to animate the corresponding 3D mesh according to the video-generated motion, based on our motion optimization. This enables a cost-effective and accessible solution to enable the synthesis of diverse and realistic 4D animations

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new method for animating static 3D humanoid meshes using video diffusion models. Instead of relying on costly motion-capture datasets, proposed approach leverages motion priors learned by large-scale text-to-video diffusion models, which inherently capture diverse human movements. Given a 3D mesh and a text prompt, the model generates a synthetic video of the mesh performing the motion, then reconstructs 3D motion by fitting and optimizing a SMPL body model to track motion cues such as 2D landmarks, silhouettes, and dense DINOv2 features. The optimized SMPL parameters are transferred to the mesh. Experiments on the CAPE dataset show that this method outperforms existing baselines in motion tracking accuracy and smoothness, and a perceptual study confirms that users find the generated animations more realistic and better aligned with textual descriptions.

### Strengths
This introduces a simple yet effective framework that leverages the motion priors of existing video diffusion models to animate static 3D humanoid meshes without relying on expensive motion-capture data. Its integration of generative video priors with SMPL-based optimization enables realistic, temporally coherent, and diverse motion synthesis from simple text prompts. This combination, including feature-based optimization, registration, and reparameterization, makes the approach both scalable and generalizable, offering a practical and accessible solution for creating 4D humanoid animations. Also, its performance on pose fitting outperforms the previous works.

### Weaknesses
A main concern is that its novelty and contribution are somewhat limited, as the overall concept of using video diffusion models for motion generation has already been explored in prior works such as MotionDreamer [1] and AnyMoLe [2]. Similar to MotionDreamer and AnyMoLe, the proposed approach extracts dense features and generated videos to guide motion reconstruction. Although this paper integrates these components into a clean and unified framework for humanoid, it primarily extends existing ideas rather than introducing a fundamentally new mechanism for motion extraction or representation. Furthermore, while the concepts overlap with [2], this work is not referenced in the paper.


[1] MotionDreamer: Exploring Semantic Video Diffusion Features for Zero‑Shot 3D Mesh Animation. Uzolas et al., 3DV 2025

[2] AnyMoLe: Any Character Motion In‑betweening Leveraging Video Diffusion Models. Yun et al., CVPR 2025

### Questions
Why are untextured meshes rendered for video diffusion model inference? It seems that leveraging video diffusion models with textured meshes could yield better performance due to a smaller domain gap with the models’ training distribution. This untextured setting could also effect largely to the pose fitting performance.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
This paper described a method to generate 3D humanoid motion using 2D video generation models. Video models have seen significantly more data compared to motion generation models (that are restricted to motion-capture data typically), and hence seem to be less expressive. The method is well engineered, with individual steps that make perfect sense, and produces convincing results.

### Strengths
- excellent results.
- clear and reasonable method.
- some nice steps, such as the combined modalities of the tracking.
- seems to surpass the SOTA even when compared to dedicated motion generation models.

### Weaknesses
- Applies to humanoid characters only.
- Depth is not explicitly addressed.

### Questions
- How do you not get motions that are too flat in terms of depth?
- Why use VPoser, which is a rather old prior instead of newer ones?
- Why not use the texture of the mesh as well, wouldn't that help the video model to be more expressive?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a novel method to animate a static 3D humanoid mesh from a text prompt, addressing the limitations of costly MoCap datasets by instead leveraging the rich, generalized motion priors from large-scale Video Diffusion Models (VDMs). The pipeline first generates a video by conditioning a VDM on a rendered image of the mesh and the text prompt. Then, it performs a robust motion transfer by first registering a SMPL body model to the input mesh as a "deformation proxy" and re-parameterizing the mesh vertices. A tracker then optimizes the time-varying SMPL parameters to match the generated video, guided by a combination of sparse 2D landmarks, dense silhouettes, and DINOv2 features. To ensure temporal smoothness, these motion parameters are predicted by shallow MLPs, and experiments show this approach outperforms baselines in tracking and is significantly preferred by users over traditional MoCap-based methods for realism and prompt alignment.

### Strengths
- The paper's primary strength is its hypothesis that generative video models, trained on massive, diverse, "in-the-wild" video data, contain superior and more generalizable motion priors than the small, clean MoCap datasets currently used by most text-to-motion methods. The strong user study results (Fig. 4) convincingly validate this hypothesis.
- The method for transferring 2D video motion to the 3D mesh is very well-designed. It correctly identifies that regression-based pose estimators (like HMR) would fail on synthetic VDM-generated videos, and thus wisely opts for a more robust optimization-based tracking approach. The use of multiple, complementary tracking cues (sparse landmarks, dense silhouettes, and semantic DINOv2 features) provides strong guidance for the optimization.
- The paper does an excellent job of evaluating its claims. The authors wisely isolate and evaluate their tracking component on a controlled task (recovering GT motion from the CAPE dataset). The results show it is not only more accurate (lower MPJPE/PVE) but significantly smoother (much lower "Accel" error) than strong baselines. The perceptual user study is the main payoff. The fact that users overwhelmingly preferred this method's animations to those from a strong MoCap-based model (MDM) on realism, prompt alignment, and overall quality is a very strong result.
- While the method combines several existing tools (VDMs, SMPL, DINO), it does so in a novel pipeline that solves a practical problem. The "SMPL-as-proxy-rig" approach makes the method applicable to a wide range of static humanoid meshes that lack their own skeletons or rigs, which is a common use case.

### Weaknesses
- The method is a "Garbage In, Garbage Out" system that places full trust in the VDM's output. The paper acknowledges that VDMs can produce artifacts or "morphing effects," but does not fully address how the tracker would handle them. If the VDM generates a physically impossible motion, a distorted body part, or a character that morphs into the background, the optimization-based tracker will likely fail or produce an equally nonsensical 3D animation. (I am personally a believer of VDM as world simulators, so hopefully this will be less of an issue over time).
- The tracker works from a single 2D video, which is an inherently ill-posed 2D-to-3D problem. While the SMPL/VPoser prior helps, the method is still susceptible to ambiguities in depth and self-occlusion. The (very specific) prompt engineering in Appendix A (e.g., "Wide angle shot," "Fixed camera," "No zoom in") suggests that the VDM output must be carefully constrained to be "trackable," which limits the range of dynamic camera motions that can be animated.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents Animating the Uncaptured, a method for animating 3D humanoid meshes from text prompts. Given an input mesh and textual description, the approach first generates motion videos via a text-to-video (T2V) diffusion model, then leverages the SMPL parameterized human model as a deformation proxy to track and reconstruct character motion from the generated video, which is subsequently transferred back to the 3D mesh.
To enhance reconstruction quality, the authors integrate multiple cues of body keypoints, silhouettes, and dense DINOv2 features as optimization constraints. Experiments on the CAPE dataset show that the method outperforms SMPLify-X, WHAM, and Multi-HMR baselines. A user study further indicates that the generated animations achieve higher perceived realism and better text-motion consistency.

### Strengths
1. Novelty: 
The paper explores a promising direction by introducing a generalized motion prior from large-scale video diffusion models (VDMs) to animate static 3D meshes. The proposed “generation–tracking–deformation” pipeline bridges generative video synthesis and 3D motion reconstruction, leveraging the strong expressive capacity of modern VDMs. This cross-domain integration offers an extensible framework for text-driven 3D animation.

2. Experimental Evidence: 
The paper provides abundant qualitative examples, and quantitative results in Table 1 demonstrate clear improvement on CAPE sequences, particularly in the Accel metric, compared with existing registration and reconstruction baselines such as SMPLify-X, WHAM, and Multi-HMR.

### Weaknesses
1. Unclear motivation: 
While the paper claims to focus on animating humanoid meshes, the methods and experiments seem more centered on registration and tracking from images or videos.
The distinction between animation generation and motion fitting is not clearly articulated, making the actual novelty somewhat ambiguous.

2. Method clarity: Several parts of the method are under-explained.
- line 159 mentions "we use the encoding $Z \in R^32$ of the variational autoencoder VPoser", 
but no further elaboration is given. 
- lines 256–269 are vague: it is unclear whether $v_i^SMPL$ refers to the corresponding face or vertex on the SMPL template.
The function $\Psi$ is introduced without a precise definition or computational description.
also lacks a clear definition or computational description. 
- Equation (3) refers to L1, but the written form corresponds to an L2 prior, indicating inconsistency between text and formulation.

3. Resource requirements: 
The paper mentions 1,000 iterations for registration and over 4,000 for video tracking, yet omits device type, runtime, or memory requirements. Without such details, the practical feasibility and scalability of the method remain unclear.

4. Lack of ablation on MLP optimization: 
The implicit MLP for temporal modeling appears to be a design simplification rather than a fundamental requirement. Since the MLP is optimized per sequence and does not generalize across meshes, it limits both efficiency and scalability. A shared temporal model (e.g., RNN, Transformer, or motion prior) might offer better generalization and faster inference.

5. Strong Dependence on the Video Diffusion Model: 
The presented animations rely heavily on the pretrained Kling AI VDM. The authors neither fine-tune the VDM for animation-related content nor test robustness across different video generators. This raises concerns about reproducibility and generalization to varied video outputs.

### Questions
1. Limited quantitative evaluation:
The CAPE dataset provides only narrow evaluation scenarios on untextured meshes. 
Why not assess performance on broader human pose and shape benchmarks such as 3DPW, RICH, or EMDB?  Additionally, would textured meshes influence generation quality or tracking efficiency?

2. Tracking accuracy:
Has the tracking accuracy been quantitatively evaluated on editable or rigged humanoid meshes to validate applicability in real animation pipelines?

3. Use of SMPL motion generation:
Since the method already performs mesh-to-SMPL registration, why not leverage existing SMPL-based motion generation techniques to animate the mesh directly, instead of relying on text-to-video tracking? This would seem to be a more straightforward way to drive the mesh using well-established motion priors.

4. Failure cases: 
What are the observed failure scenarios?

### Soundness
2

### Presentation
2

### Contribution
2
