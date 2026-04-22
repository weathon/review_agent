# Focusing: View-Consistent Sparse Voxels for Efficient 3D VAE

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 8, 2, 4

## Abstract
High-fidelity 3D generation remains difficult. Although some methods have proposed converting raw meshes to SDFs, it remains a lossy process. TripoSF presented a VAE training paradigm based on a rendering loss to circumvent this lossy SDF conversion, achieving high-precision surface reconstruction. However, because the rendering loss cannot supervise all the VAE outputs in the same way as SDF supervision, it limits detail and scalability.We present  **Focusing**, a 3D VAE that improves efficiency by activating only the voxels that matter for a given view. Our key idea is a depth-driven voxel carving performed in the structured latent space: voxels inconsistent with the rendered depth are pruned before decoding. This concentrates learning  on locally relevant geometry, reduces attention and decoding costs, and lowers video random access memory (VRAM) usage. To stabilize training and capture fine details, we further introduce an adaptive zooming strategy that adjusts camera intrinsics to keep the number of active voxels within a target range. The VAE is trained with a render-based loss on depth, normals, masks, and perceptual terms, and we add simple regularizers (e.g., sparse-voxel TV and a short warm-up with TSDF supervision) to reduce small holes and speed up convergence. Across standard reconstruction benchmarks, Focusing improves geometric accuracy (CD, F-score) over strong baselines while cutting VRAM consumption, which allows for training the $1024^3$ resolution VAE on as little as 50GB of VRAM. These results show that local, view-consistent sparsity is an effective route to higher-resolution, more efficient 3D VAEs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a 3D VAE (Focusing) designed to improve the efficiency and detail of 3D shape generation. It proposes to mainly address the problem of existing methods to compute all voxels in the camera frustum, which could be irrelevant to the final review, hindering the scalability of high resolution. It introduces two core contributions, including depth-driven voxel carving and adaptive zooming to deal with this problem.

### Strengths
1. The contributions of carving and adaptive zooming are logical optimizations. Pruning occluded voxels before the decoding step and zooming to supervise fine-grained details seem to be two effective approaches .
2. The claim of training a $1024^3$ model on < 50GB VRAM is very useful to make high-resolution 3D more accessible.

### Weaknesses
The primary weakness is the experimental validation, which lacks sufficient comparison and ablation studies to support the main claims.
1. The authors introduce several components simultaneously, but only compare the final model against the baseline where this work is developed. There is no way to disentangle the effect of each component.
2. Although the paper claims it's more efficient compared to previous works, there is no quantitative evaluation based on it 
3. Limited comparison over generative experiments, such as the presentation of Section 4.3 as a validation of VAE's utility, but results are purely qualitative; There are also unsupported claims on regularization.

### Questions
1. The comprehensive ablation study could help isolate the individual contributions. For example, it could include the baseline, baseline+depth-driven voxel carving (demonstrating that carving does not affect 3D accuracy), baseline+carving+adaptive zooming (zooming could improve fine-grained details), and the full model. 
2. The VRAM comparison or any efficiency quantitative results could help: (a) compare with previous approaches, (b) compare before and after using carving.
3. There is only one comparison of this approach with the TripoSF baseline. Could the model be compared with others, such as the TRELLIS, Sparc3D, mentioned in the paper?
4. For the image-to-3D generation experiments, how does the model quantitatively compare to other generative models (TripoSF, TRELLIS, at least)?
5. For some parameters, such as the depth-carving threshold r = 2 / resolution, what's the model's sensitivity to this hyperparameter? What would happen visually if over-caving or under-caving

### Soundness
2

### Presentation
2

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
The paper proposes FOCUSING, a 3D VAE that introduces depth-driven voxel carving in the latent space to prune voxels that are inconsistent with a rendered depth map before decoding. This concentrates computation on view-relevant geometry, reduces attention/decoding cost, and enables 1024^3 resolution training with manageable amount of VRAM using a render-based loss plus simple regularizers. To stabilize the number of active voxels, the authors add an adaptive zooming scheme that modifies intrinsics to keep active voxels within a target range. On reconstruction benchmarks, the method outperforms TripoSF at comparable resolutions, and produces strong qualitative image-to-3D results.

### Strengths
1. Carving structured latents via depth consistency (as opposed to frustum culling only) is a clear, targeted idea that reduces both decode and render cost. Prior works (TRELLIS SLAT, SparseFlex/TripoSF, etc,) prunes by camera frusta or quotas; carving by per-voxel depth agreement in latent space appears novel in this context and is well-motivated by the render-supervision regime where only visible surfaces receive strong gradients.
2. The proposed efficient latent pruning enables 1024^3 resolution meshes under render-loss supervision, addressing a real bottleneck in scalable 3D VAEs and complements the trend toward structured 3D latents (TRELLIS) and SparseFlex-style meshing. This could potentially be a useful plug-in for render-supervised VAEs and image-to-3D pipelines.
3. The idea is well-motivated. The paper is well-written and clearly presents a concrete algorithm, detailing the loss design and features relevant ablation studies.

### Weaknesses
The paper presents a clean and novel idea. I do not have a major concern here, several smaller suggestions and possible improvements:
1. Comparative scope is narrow. Quantitative comparisons focus on TripoSF; there is no direct comparisons with other SLAT VAEs on reconstruction quality. At least Dora-VAE and its followups can be tested for comparison on Dora-Bench. Including more relevant baselines and include CD/F-score across complexity strata (possibly with some case studies) would better demonstrate the gain over previous baselines.
2. Clarity on gradient flow through carving- the voxel-carving mask appears non-differentiable; the paper does not discuss how gradients propagate to pruned voxels, whether there is bias introduced by hard masking, or whether stochasticity/annealing (e.g., soft masks) was explored.
3. Efficiency claims need fuller evidence. The abstract claims <50 GB VRAM for 1024³; however, peak VRAM vs. resolution and batch size are not plotted, nor compared to SparseFlex’s frustum-aware training. A standardized memory/time breakdown (encode/decode/render) against TripoSF/TRELLIS would strengthen the efficiency story.
4. Potential view-bias / consistency issues. Because pruning depends on the current rendered depth, supervision may under-regularize self-occluded or thin structures. A systematic evaluation (multi-view consistency metrics, topology errors, hole rates) and failure case study is absent.

### Questions
Please check the weaknesses for details.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on the task of 3D VAE training for 3D object generation. It builds upon and extends the concept of SparseFlex, aiming to further improve training efficiency. The paper introduces two key techniques—depth-based voxel carving and adaptive zooming—which effectively reduce and stabilize the sparse voxels involved in decoding and rendering, thereby decreasing and stabilizing VRAM usage. These techniques also enhance the overall quality of 3D VAE training.

### Strengths
1. **Technical Soundness:**
   I find the proposed techniques of voxel carving and adaptive zooming to be both well-motivated and technically sound. Although the ideas represent relatively straightforward engineering improvements, they serve as a valuable complement to SparseFlex.

2. **Clarity and Presentation:**
   The paper is generally well written and easy to follow. It provides sufficient context and is largely self-contained. I also appreciate the well-designed figures, which effectively help readers grasp the core ideas.

3. **Experimental Results:**
   The paper successfully demonstrates its superiority in terms of fitting error.

### Weaknesses
While the paper is relatively straightforward and engineering-oriented, I am not against that—I believe the idea is a meaningful and well-motivated complement to existing work. However, my major concern lies in the completeness of the evaluation. As a scientific paper submitted to ICLR, the current version’s experimental validation is far from sufficient. I understand that the authors may have completed the paper in a hurry and plan to add more experiments during the *long-period* rebuttal stage of ICLR. Nevertheless, I personally feel that this practice is unfair to other authors who submitted fully developed experiments before the deadline. As a result, I assign a **reject** rating here primarily due to the paper’s **incompleteness** and **fairness concerns**.

1. **Lack of Efficiency Analysis:**
   While the main claim is that the proposed techniques improve efficiency, there is no direct analysis of training or inference time, VRAM usage, or the resolution of supervision images. The only provided experiment concerns the VAE’s fitting quality. As a scientific paper, we not only care about fitting error but also about how efficiency-related metrics are affected by the proposed techniques. Without these analysis, the claim of efficiency improvement remains unsubstantiated.

   For instance, why does the proposed VAE achieve better fitting error? Is it because the reduced VRAM usage allows training with higher-resolution supervision images? Such quantitative evidence and analysis are essential. Furthermore, if the proposed method’s main advantage is efficiency, one might question whether similar improvements could be achieved simply by using smaller batch sizes and more GPUs. 

2. **Missing Ablation Studies:**
   The paper introduces two main techniques—depth-based voxel carving and adaptive zooming—as well as additional loss terms. However, no ablation studies are provided. For a scientific paper, extensive ablations are necessary to understand the contribution of each component and loss function to VRAM usage, training/inference time, and fitting quality. Without such experiments, it is difficult to attribute the observed performance improvements to specific innovations.

### Questions
0. **Limited Decoding Capability Compared to Trellis:**
   The original *Trellis* framework supports multiple decoding heads, such as decoding 3D Gaussians for texture generation. In contrast, the proposed method appears to decode only 3D meshes. How does the method handle mesh textures, and is there a plan to extend the approach to support texture decoding as in Trellis?

1. **Unexplained Term in Equation (2):**
   The loss term ( L_{\text{occ}} ) in Equation (2) is not explained. Please provide a clear definition and motivation for this term.

2. **Lack of Clarity in Lines 317–320:**
   The description of the self-pruning upsampling module is not self-contained and requires more detailed explanation. Additionally, the meaning of ( O_{\text{pred}} ) should be clarified.

3. **Missing Symbol Definitions in Equation (6):**
   Several symbols in Equation (6) are not defined. Please include explanations for all variables used.

4. **Ambiguous Description in Line 348:**
   The phrase “inject them into a Diffusion Transformer (DiT) via cross-attention” is unclear. From my understanding, the method trains an image-conditioned rectified flow model, rather than injecting image features into the latent space prior to rectified flow training. Please clarify this process.

5. **Clarification Needed in Line 369:**
   Why is the number 518 chosen? Please justify this specific design choice.

6. **Unsubstantiated Claim in Line 378:**
   The statement “Our method demonstrates results comparable to TripoSF’s 1024-resolution output even at 512 resolution and captures geometry details better at 1024 resolution” is not clearly supported by quantitative or qualitative evidence. Please provide corresponding results or analysis to substantiate this claim.

### Soundness
3

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
5

### Summary
FOCUSING targets high-fidelity yet efficient 3D generation by training a 3D VAE with render-based supervision to reconstruct meshes from sparse voxel latents, enabling 1024³ results with modest memory. It is adapted based on TripoSF. Its core idea is depth-driven voxel carving in the latent space that prunes voxels inconsistent with the rendered depth before decoding, concentrating learning on view-relevant geometry. An adaptive “zooming” strategy further adjusts camera intrinsics to keep active-voxel counts within a target range, stabilizing VRAM and sharpening details. On Objaverse-XL–scale training, the method cuts VRAM to ≲50 GB for 1024³ and improves Chamfer Distance and F-score over TripoSF at matched resolutions, with visual detail comparable to higher-res baselines. Overall, the work shows that local, view-consistent sparsity contributes to higher-resolution and more efficient 3D VAEs.

### Strengths
- Better efficiency and higher resolution than TripoSF for a 3D-VAE.
- Practical training design: render-based supervision (depth/normal/mask + perceptual losses), sparse-voxel TV, and a short TSDF warm-up stabilize learning and sharpen details.
- Adaptive “zooming” to keep active-voxel counts in a target range, improving memory predictability and fine-structure fidelity.
- The paper is clearly written, and the figures and formulas are clean.

### Weaknesses
- Internal structure is largely neglected: the efficiency comes from more selective surface sampling, but an informative 3D-VAE ideally encodes both exterior and interior—methods like SPARC3D highlight this. The trade-off may be suboptimal for full 3D reconstruction use cases.
- Limited novelty: much of the pipeline builds on TripoSF; the main additions (carving + zooming) feel like incremental efficiency improvements.
- Insufficient baselines: Only VAE reconstruction comparison (Table 1) to TripoSF. Missing head-to-head results against other single-image 3D reconstruction methods (e.g., Sparc3D). Even within VAE reconstruction, more baselines and Dora metrics should be included.
- Marginal quantitative gains: the reported improvements over TripoSF appear small ( <0.1σ in the table), which weakens the practical significance of the contribution.
- No qualitative ablations isolating design choices (carving thresholding, zoom schedule, TSDF warm-up length) to show when each component helps or hurts.

### Questions
- Please include a clear VRAM and wall-clock table (peak and average) across resolutions and batch sizes, alongside TripoSF and any additional baselines.
- How sensitive are results to the carving threshold/heuristics and zoom schedule? Provide qualitative ablations on thin parts and hollow objects.

Minor issues:

- L42: there is a missing space before “We.”
- Duplicate references of PointNet

### Soundness
2

### Presentation
3

### Contribution
2
