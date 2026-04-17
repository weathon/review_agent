# UniHand: A Unified Model for Diverse Controlled 4D Hand Motion Modeling

- Decision: Accept (Poster)
- Scores: 6, 2, 8

## Abstract
Hand motion plays a central role in human interaction, yet modeling realistic 4D hand motion (*i.e.*, 3D hand pose sequences over time) remains challenging. 
Research in this area is typically divided into two tasks: 
(1) Estimation approaches reconstruct precise motion from visual observations, but often fail under hand occlusion or absence; 
(2) Generation approaches focus on synthesizing hand poses by exploiting generative priors under multi-modal structured inputs and infilling motion from incomplete sequences.
However, this separation not only limits the effective use of heterogeneous condition signals that frequently arise in practice, but also prevents knowledge transfer between the two tasks.
We present **UniHand**, a unified diffusion-based framework that formulates both estimation and generation as conditional motion synthesis. 
UniHand integrates heterogeneous inputs by embedding structured signals into a shared latent space through a joint variational autoencoder, which aligns conditions such as MANO parameters and 2D skeletons.
Visual observations are encoded with a frozen vision backbone, while a dedicated hand perceptron extracts hand-specific cues directly from image features, removing the need for complex detection and cropping pipelines.
A latent diffusion model then synthesizes consistent motion sequences from these diverse conditions.
Extensive experiments across multiple benchmarks demonstrate that UniHand delivers robust and accurate hand motion modeling, maintaining performance under severe occlusions and temporally incomplete inputs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents UniHand, a unified diffusion-based framework for 4D hand motion modeling that jointly addresses motion reconstruction and generation. It reformulates both tasks as a conditional motion synthesis problem, bridging the gap between estimation and generation. A Joint Variational Autoencoder (Joint VAE) aligns heterogeneous conditional inputs such as 2D and 3D keypoints and MANO parameters into a shared latent space, enabling robust modeling under incomplete or inconsistent data. The Hand Perceptron module extracts hand-specific features directly from full visual inputs without requiring explicit detection or cropping, while a canonical coordinate system ensures spatial and temporal consistency under dynamic camera settings. Experiments on DexYCB, HO3D, and HOT3D datasets demonstrate that UniHand achieves state-of-the-art results, especially under severe occlusion or partial input, validating its effectiveness and generalization for unified hand motion estimation and generation. However, despite its strong performance, the method suffers from algorithmic complexity and high computational cost, and the exploration of the unique challenges in multi-modal alignment for 4D hand pose estimation remains insufficient.

### Strengths
- The paper proposes a unified diffusion-based framework that integrates both estimation and generation for 4D hand motion modeling, offering a fresh formulation of conditional motion synthesis that extends beyond task-specific designs.
- The technical design, including the Joint VAE and Hand Perceptron modules, is well-motivated and validated through comprehensive experiments on multiple datasets, showing robustness under occlusion and dynamic camera motion.
- The paper is clearly written and systematically structured.

### Weaknesses
- Real-world deployment may be limited without efficient preprocessing of input modalities.
- Heavy computational and data requirements for training.

### Questions
- While UniHand demonstrates strong performance, the diffusion-based generation pipeline and multimodal latent alignment introduce high computational cost. Can the authors quantify the training and inference time compared to existing methods, and discuss possible simplifications or acceleration strategies?
- The paper acknowledges that UniHand struggles to maintain globally consistent trajectories under large camera movements due to the lack of explicit camera extrinsics. Could the authors elaborate on potential ways to address this—e.g., by incorporating implicit camera modeling or scene-aware constraints?
- Although UniHand integrates visual, 2D, and 3D conditions through a joint VAE, the paper could further investigate how multimodal alignment contributes to 4D hand pose estimation quality. Would ablation or visualization of the latent space help clarify how modalities interact and which provide the most benefit?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper aims to unify both hand pose estimation and generation using a single diffusion-based framework.
The proposed model, UniHand, handles both structured MANO and key points (as conditions for generation), as well as images (for estimation). 
Separate encoders are used to bring different hand conditions into the same latent space, then the method runs denoising diffusion process in the latent space, aiming to to produce smooth 4D motion output.

### Strengths
This paper is well motivated: it aims to bring the estimation and the generation model together. 
The proposed model indeed can do these two task in a unified way. 
This paper does not propose a new problem formulation, but the effort to introduce an unified solution for two problems are interesting and valuable.

The writing is clear in general: we can understand how the proposed framework achieve the proposed goal at the high level.

### Weaknesses
The main issue of this paper is it does not address the proposed goal: unifying both estimation *and generation*. The generation ability of the framework is not tested/reported. 
The paper is motivated by unifying the estimation and generation into the same framework. The motivation is sound, and the de-noiser in the framework is indeed a generative model. However, all results are reported **only on the hand pose estimation task**. 
The sole focus of estimation deviates from what is described in the title, abstract and intro, as well as in the related works where the authors talk about unifying generation under their framework in Line 134-136.
The authors could have shown their results on any generation problems, e.g. [Zuo et al., 2023] in Line 128 or [Zuo et al., 2024] in Line 115-116 of the grasp generation problem. I understand that extending the system for these generation tasks can involves much more work, but since the authors claims the unified method, one would expect the generation task results.

In Line 347-348, the authors proposes to evaluate the generation via estimation:
> Hand pose estimation in the camera coordinate space provides the most direct way to evaluate the quality of motion generation conditioned on visual observations.

but above is saying that the estimation and the generation are the same task, which contracts what the authors say in Line 013 and throughout the whole introduction!

The author should really test their framework on a few tasks listed in their Sec 2.2, without the results on the generation task, otherwise this paper gives a wrong impression to the audience. 

Regarding the estimation evaluation setup, it is unclear to me when $c_{2D}$ and $c_{3D}$ are provided to the model? In my understanding, the standard estimation task setup does not provide $c_{2D}$ and $c_{3D}$ to the model.

Since providing experiment results on the real generation task will lead to substantial changes to the paper and is beyond what can be done during rebuttal, I suggest rejection.

### Questions
Apart from the mismatch between what is claimed and what is experimented, the writing of this paper is good in general.

The points that is unclear:

1. Figure 1, bottom left, what are two red boxes represent at the bottom left? are they $g$?

2. Shouldn't motion encoder  be called hand pose encoder? 

3. Line 207-208, better to say "At each _autoregression_ step" instead of simply "At each step", which can mean diffusion step.

4. Line 883, DeepSpeeds need reference.

5. Line 118, EASY-HOI is not relevant to Hand Motion Generation.

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The work seeks to unify the hitherto distinct domains of hand motion generation and hand motion reconstruction. This is achieved by a variational autoencoder embedding various non-visual modalities (hand trajectories, human body keypoints) into a shared latent space. In parallel, a hand perceptron extracts frame-wise features from visual inputs. The extracted non-visual latent representations are then used by a latent diffusion model to generate the hand pose sequence, with visually informative features being provided at every layer of the motion denoiser. The work compares the proposed method with multiple hand motion generation and reconstruction baselines on three datasets, consistently outperforming them.

### Strengths
The proposed method shines when evaluated against numerous baselines, even in the presence of significant occlusion (Table 1). It is able to handle multimodal conditioning input, making it flexible and able to benefit from various types of known information at inference time.
The proposed method is thoroughly ablated with respect to its components and possible input modalities.
The work includes an honest discussion of its limitations.

### Weaknesses
The submission could benefit from more qualitative examples in the supplementary material. This is especially relevant for generative models.

### Questions
How does the method differentiate between the left and the right hand in its output?
How does the method perform in the presence of feet, which are often misdetected as hands in egocentric videos?

### Soundness
3

### Presentation
4

### Contribution
3
