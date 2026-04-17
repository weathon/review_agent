# Scalable and Generalizable Autonomous Driving Scene Synthesis

- Decision: Reject
- Scores: 8, 4, 6, 6

## Abstract
Generative modeling has shown remarkable success in vision and language, inspiring research on synthesizing driving scenes. 
Existing multi-view synthesis approaches typically operate in image latent spaces with cross-attention to enforce spatial consistency, but they are tightly bound to camera configurations, which limits model generalization.
We propose BEV-VAE, a variational autoencoder that learns a unified Bird’s-Eye-View (BEV) representation from multi-view images, enabling encoding from arbitrary camera layouts and decoding to any desired viewpoint. 
Through multi-view image reconstruction and novel view synthesis, we show that BEV-VAE effectively fuses multi-view information and accurately models spatial structure. 
This capability allows it to generalize across camera configurations and facilitates scalable training on diverse datasets.
Within the latent space of BEV-VAE, a Diffusion Transformer (DiT) generates BEV representations conditioned on 3D object layouts, enabling multi-view image synthesis with enhanced spatial consistency on nuScenes and achieving the first complete seven-view synthesis on AV2.
Compared with training generative models in image latent spaces, BEV-VAE achieves superior computational efficiency.
Finally, synthesized imagery significantly improves the perception performance of BEVFormer, highlighting the utility of generalizable scene synthesis for autonomous driving.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper focuses on multi-view generation in driving scenes. Previous works use image-level latent representations, relying on cross-view attention to maintain cross-view consistency. This work proposes encoding multi-view images into a unified and compact BEV-latent. This explicit latent representation directly guarantees cross-view consistency. The proposed method can be trained across datasets (with different camera layouts) and demonstrates strong cross-dataset generalization capability and high image quality.

### Strengths
The motivation and idea of this work are solid. The BEV latent representation not only explicitly ensures cross-view consistency, as the paper emphasizes, but I also guess it can largely mitigate the subjectivity issues of generative models (For example, the consistency and move/changes of the 3D content are reasonable only within the camera view). The authors could consider validating this point.
The designs of the BEV latent encoder, decoder, discriminator, and training pipeline are reasonable and well-founded. The writing is clear.
Experiments are extensive and solid. The model's capability for cross-dataset training and its few-shot generalization ability are impressive. Visualization results show that the model achieves high accuracy in reconstructing views under the highly compressed BEV latent representation.

### Weaknesses
Recommend defining F_stt in Section 3.1.2.

The title "Scalable and Generalizable Autonomous Driving Scene Synthesis" doesn't fully capture the paper's key feature (BEV latent representation / multi-view synthesis). The experiments primarily demonstrate the method's cross-dataset training capability (which is good) rather than its scalability. Consider adjusting the title to make it more distinctive？

### Questions
The current method doesn't seem to involve temporal modeling. Will explore it in future work?

The paper discusses the proposed method's relatively lower FID scores (which, given the difficulty of latent representation in BEV space compared to image-level representation, I find understandable). Could increasing the BEV spatial resolution / the number of CFV sampled rays improve the view resolution/realism?

Are there any plans to open-source the code?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on autonomous driving scene synthesis and presents BEV-VAE, a variational autoencoder that unifies multi-view driving images into a compact bird’s-eye-view (BEV) latent representation, allowing flexible encoding and decoding across arbitrary camera configurations. By incorporating a Diffusion Transformer conditioned on 3D object layouts, the method achieves spatially consistent and generalizable multi-view synthesis on nuScenes and AV2. The synthesized data further enhances BEVFormer’s perception performance, highlighting the value of scalable and generalizable scene synthesis from a training data perspective.

### Strengths
1. The paper presents a clear motivation and is generally well written.
2. In the autonomous driving domain, due to the inherent need for multi-view perception, a BEV-based VAE offers greater practical value than image-space VAEs.
3. Using BEV representations makes it easier to transform different camera layouts, and simulate training data for different vehicle configurations.

### Weaknesses
1. The paper does not clearly articulate the advantages of BEV-VAE over Image-VAE. In terms of generation quality, both rFID and gFID are inferior to those of Image-VAE. Moreover, recent image-based multi-view generation methods also achieve strong spatial consistency. The potential benefits of BEV-VAE, in my view, may lie in two aspects—information compression and better compatibility with 3D editing—but the paper does not appear to emphasize either of these points. The results in Table 6 also raise questions — if the primary application value lies in train data generation, the improvement in detection performance appears comparable to that achieved by BEVGen, making it difficult to identify a clear advantage of BEV-VAE in this aspect.
2. The technical novelty of the paper is weak, as the BEV-VAE architecture largely follows that of BEVFormer. The use of BEV representations is also quite similar to BEVWorld, yet the paper lacks a detailed discussion of their differences. In addition, the rendering process to images resembles existing approaches such as self-occ. It would be beneficial for the authors to more clearly articulate the technical innovations, as the method section currently appears to primarily combine components from prior works.

### Questions
1. From Table 3, doesn’t the comparison with SD-VAE suggest that BEV-VAE has weaker zero-shot generalization ability than image-based latent representations?
2. Why does BEV-VAE use only 256×256 image resolution? Would scaling up the resolution introduce any potential issues or challenges?
3. How much improvement in generation quality does using DiT with BEV-VAE provide compared to using BEV-VAE alone?
4. During the model training process, which modules, if any, use pre-trained parameters, and which are trained entirely from scratch?
5. GAN losses are usually sensitive to hyperparameter settings. Could the authors comment on potential issues regarding hyperparameter sensitivity and training stability in their setup?

### Soundness
2

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
4

### Summary
This paper introduces BEV-VAE, a novel variational autoencoder framework designed for autonomous driving scene synthesis. The core contribution is a model that unifies multi-view images into a compact and camera-agnostic Bird's-Eye-View (BEV) latent representation. This approach decouples the scene's 3D structure and semantics from the specific camera configurations, enabling the model to be trained on diverse datasets with varying camera layouts and to generalize to arbitrary new viewpoints. For generative tasks, a Diffusion Transformer (DiT) is trained within the learned BEV latent space, conditioned on 3D object layouts represented as occupancy grids. The authors demonstrate the model's effectiveness through multi-view reconstruction, novel view synthesis, and cross-dataset generalization. While the proposed method achieves state-of-the-art multi-view spatial consistency, it shows a trade-off in per-image generative fidelity (gFID) compared to prior work. Finally, the practical utility of the synthesized data is validated by improving the performance of a downstream perception model, BEVFormer.

### Strengths
1. **Generalization:** The paper provides compelling evidence for the model's ability to generalize across different datasets (nuScenes, AV2, nuPlan) and camera setups. The experiments showing successful reconstruction of scenes from one dataset (e.g., nuPlan) using the camera intrinsics and extrinsics of another (e.g., AV2) are particularly impressive and strongly support the claims of generalizability. The demonstrated performance gains from training on a large, mixed dataset (PAS) validate the model's scalability.
2. **Superior Multi-View Spatial Consistency (MVSC):** The model achieves a state-of-the-art MVSC score. This is a crucial metric for autonomous driving applications, where maintaining the correct 3D geometry and spatial relationships between objects across views is often more important than perfect photorealism. The architectural design, which generates all views from a single, coherent 3D representation, naturally leads to this strength.
3. **Demonstrated Downstream Task Improvement:** The experiment in Section 4.8, showing that data augmentation using the proposed method improves the NDS score of BEVFormer, is a very strong point. It demonstrates that the synthesized data is not just visually plausible but also practically useful for training and improving perception models, closing the loop between generation and perception.

### Weaknesses
1. **Lower Generative Fidelity (gFID):** The most apparent weakness, which the authors acknowledge, is the relatively high (worse) gFID score compared to state-of-the-art methods like MagicDrive and DriveWM. While the paper frames this as a trade-off for better spatial consistency, the gap is substantial (20.7 vs. ~13-16). This indicates that the generated images may lack the fine-grained texture and realism of other methods, which could limit their utility in certain applications.
2. **Low Image Resolution:** All experiments are conducted at a 256x256 resolution, which is quite low for modern autonomous driving datasets and applications. While the authors suggest using super-resolution models as a post-processing step, this feels like an external fix rather than an integrated solution. The paper would be stronger if it discussed the challenges and potential architectural changes required to scale BEV-VAE to higher resolutions (e.g., 512x512 or higher).
3. **Overstated "Zero-Shot" Capability:** The term "zero-shot" in Section 4.6 seems too strong given the quantitative results in Table 3. The zero-shot performance on WS101 is very poor (PSNR 16.6, rFID 56.7). The real strength demonstrated here is in *fast adaptation* or *efficient fine-tuning*, where the pre-trained model provides a strong prior that allows for rapid convergence on a new dataset. The terminology should be more precise to reflect this.
4. **Static Scene Limitation:** The current framework operates on static scenes. The real world is dynamic, and the ability to model temporal evolution and generate coherent video sequences is a key direction in this field. While mentioned as future work, this is a significant limitation compared to the broader goals of full-world simulation.
5. **Mismatched Framing of Contribution and Lack of Efficiency Analysis:** The title "SCALABLE...SCENE SYNTHESIS" may be slightly overstated, as the paper's core innovation lies not in the generative model itself—which is a standard Diffusion Transformer—but in the preceding VAE architecture for learning a unified BEV representation. A significant, yet underexplored, benefit of this design is its potential for computational efficiency; by compressing the multi-view scene into a compact latent space, the subsequent diffusion process should be substantially less demanding in terms of memory and latency. To truly validate the "Scalable" claim and better frame the work's practical contribution, the paper would be significantly strengthened by a quantitative comparison of GPU memory usage and inference times against other leading methods.

### Questions
Regarding the FID/MVSC Trade-off: Could you elaborate on why you believe there is this trade-off? Is the lower FID an inherent consequence of the VAE's information bottleneck regularizing the latent space, potentially smoothing over high-frequency details? Have you experimented with alternative VAE formulations, such as a VQ-VAE, which might allow for sharper reconstructions while maintaining the unified BEV structure?

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
This paper address the problem of novel-view-synthesis (NVS) in driving scene, where the novel view are camera viewpoints around the cameras. 

The author propose to address this problem through a 3D-aware Birds-eye-view VAE. 

This VAE will encode multiple images together to 3D BEV latents, and then decode it back to images.  Afer training this VAE with MSE, perceptual loss and GAN loss over multiple datasets, the author showed that NVS can be down by using different camera extriniscs and intrinsics when decoding.  Which is a very neat idea. 



Also the author showed that this 3D-aware VAE has relatively OK reconstruction PSNR compared with vanilla image space VAE used by stable diffusion. 



The author also showed that the proposed method can be used to generate augmented data when trainingperception model: BEVFormer, increasing the performance, which is quite impressive.

### Strengths
1. The idea is so interesting.  3D aware VAE can be used for NVS by adjusting the camera parameters used in decoding process. Quite cool!
2. The evaluation is quite comprehensive (even though used proxy metric for NVS), I understand that the PSNR is lower than those of image based VAE, e.g. SD-VAE. 
3. Using the proposed methods for data-augmentation is also very interesting!

### Weaknesses
I think the major weakness I have is about runtime efficiency. It seems that the deformable attention would be very slow compared with flash-attention style attention implementations.  Can the author provides more details about it?  I understand that it might be slow without a well-optimized flash-attention style kernel.

### Questions
1. Is it possible to evaluate NVS without using proxy metrics as done in the paper (using reconstruction metric as proxy metric for NVS)
2. Will dropping input images during training improves the NVS performance?  Seems that if you drop one image during the encoding stage, but still compute the reconstruction loss on that image, this process will resemble a NVS training.

### Soundness
3

### Presentation
3

### Contribution
3
