# DiffuPhyGS: Text-to-Video Generation with 3D Gaussians and Learnable Physical Properties via Diffusion Priors

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2, 4

## Abstract
Generating realistic 3D object videos is crucial for virtual reality and digital content creation. However, existing 3D dynamics generation methods often struggle to achieve high-quality appearance and physics-aware motion, relying on manual inputs and pre-existing models. To address these challenges, we propose DiffuPhyGS, a novel framework that generates high-quality 3D objects with realistic and learnable physical motion directly from text prompts. Our approach features an LLM-Chain-of-Thought-based Iterative Prompt Refinement (LLM-CoT-IPR) method, which obtains prompt-aligned 2D and multi-view 3D diffusion priors to guide Gaussian Splatting (GS) to generate 3D objects. We further enhance 3D generation quality with a Densification-by-Adaptive-Splitting (DAS) mechanism. Next, we employ a material property decoder that utilizes a Mixture-of-Experts Material Constitutive Models (MoEMCMs) to predict the mixed material properties of the 3D object. We then apply the Material Point Method (MPM) to deform 3D Gaussian kernels, ensuring physics-grounded motion guided by implicit and explicit physical priors from the video diffusion model and a velocity loss function. Extensive experiments show DiffuPhyGS outperforms other methods in generating realistic physics-grounded motion across diverse materials.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces DiffuPhyGS, a framework that can generate dynamic 3D objects with physical information from text prompts. The improvements mainly include: a prior part, which proposes LLM-CoT-Iterative Prompt Refinement to make the prior more aligned and detailed; a visual modeling part, which proposes Densification-by-Adaptive Splitting to improve the shape and appearance quality of GS modeling; and a motion modeling part, which proposes a Mixture-of-Experts Material Constitutive Model to encompass different physical properties.

### Strengths
1. The problem this paper aims to solve is critical and important, considering that most current video & 3D & 4D generation lacks physical explanation and rationality.
2. The paper improves upon existing methods from several aspects (i.e., prompt refinement, visual modeling, motion modeling). This consideration is comprehensive, but there may also be incremental.

### Weaknesses
1. I think the core issue is the visual quality. I watched all the qualitative comparisons and videos provided in the authors' supplementary material. These results have many flaws, and I feel the physical properties do not meet expectations. For example:
The various parts of the hamburger all have a similar jelly-like texture, lacking the specific texture of each part.
The physical effects of the jar are very strange, feeling more like sand.
2. I didn't see a significant visual improvement of the method compared to other baselines. The quantitative results have the same problem.
3. My personal feeling is that the various modules proposed in the paper are not a holistic and critical improvement, but only some small, incremental improvements, and the effect of each component is difficult to be convincing given the current number of cases and visual quality.
4. I think that a more important task for binding physical properties to GS is editing, for example, after generating an object, whether reasonable physical effects can be produced by initializing different heights or rotating a certain angle, or applying different forces (such as stretching). This paper (or similar papers) lacks these experimental results.

### Questions
1. Why can the FVD metric be computed among the qualitative metrics, given that there are no ground-truth videos?

2. Why are the first three metrics exactly the same in the ablation studies of w/o Velocity Loss and w/o Video Diffusion Prior? This seems illogical, since both the noise sampling and optimization process of SDS introduce randomness and uncertainty.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes DiffuPhyGS, a framework that generates high-quality 3D objects with realistic and learnable physical motion. The authors proposed using LLM-CoT-IPR and SDS to generate high-quality 3DGS objects and employ MPM to optimize the final 4D dynamic results.

### Strengths
This paper presents a complete pipeline from text prompts to 3DGS generation, and finally to 4D motion generation.

### Weaknesses
1. Lack of novelty: The techniques used in the paper, including LLM-CoT-IPR, 2D-SDS, MV3D-SDS, and MPM-based 4D-SDS, are all from existing methods, making it difficult to identify any technical contributions in the proposed approach.
2. Insufficient experiments: The authors only used 5 cases for comparison, and the types of motions generated are too simple, with most of them involving objects falling from a high place.

### Questions
1. Velocity Loss: How is the expected velocity change calculated? If the MPM simulation is already based on physical equations, why does the result not align with the expected velocity?
2. Quantitative Evaluation: How are the ground-truth results obtained for the evaluated cases? For example, in Figure 3, the rubber burger is unrealistic in real-world settings. How can the authors determine which generated result is more reasonable across different methods?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
DiffuPhyGS presents an end-to-end text-to-video pipeline that generates 3D Gaussian objects with physics-driven motion. The system refines prompts using an LLM loop, stabilizes geometry with multi-view diffusion guidance and Densification-by-Adaptive-Splitting, and learns per-Gaussian material mixtures. Motion is driven through an MPM simulator using both implicit diffusion cues and an explicit velocity loss. On a small set of handcrafted prompts, the method reports higher metrics scores and is preferred in a small user study over PhysDreamer, OmniPhysGS, and PhysGaussian.

### Strengths
- The paper tackles joint generation of text-aligned appearance and dynamics in a single 3D Gaussian framework, combining prompt processing, diffusion-based 3D synthesis, and differentiable physics.
- Leveraging video diffusion priors with Score Distillation Sampling (SDS) to model motion and material is an interesting idea.

### Weaknesses
- The evaluation is very limited, four template prompts plus qualitative figures, without standardized datasets or complex multi-object interactions. It’s hard to assess generalization beyond curated cases.
- Baselines depend on geometry produced by DiffuPhyGS, and the main metrics blend appearance and motion. There are no physics-grounded metrics, so claims of physical fidelity aren’t well supported.
- The 3D generation component lags behind recent text-to-3D/image-to-3D methods. SDS-based approaches tend to produce lower quality and run slowly for both geometry and appearance, which makes it hard to judge the pipeline’s full potential since results are limited by the underlying generator.
- There isn’t a clear novelty claim on the 3D generation side. Using 2D and multi-view diffusion with SDS for 3D Gaussians has been explored before, and the LLM-CoT-IPR module mainly uses an existing LLM to refine prompts without introducing new techniques.
- I would suggest the authors apply the motion and material learning techniques on more recent and higher-quality 3D generation methods such as Trellis[1].

[1] Structured 3D Latents for Scalable and Versatile 3D Generation. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents DiffuPhyGS, a framework for generating high-quality 3D objects and realistic physical motion from text prompts. It addresses some limitations in current text-to-video generation methods concerning visual appearance and physical behavior. 
One of the central innovations of DiffuPhyGS is the LLM-CoT-IPR method, which employs a stepwise prompting refinement approach using large language models (LLMs). This technique enhances the alignment between textual inputs and the generated content, ensuring that the outputs closely adhere to the original prompts. Additionally, the framework incorporates a Hybrid Expert Material Constitutive Model (MoEMCM) that accurately predicts the properties of heterogeneous materials. This integration allows for improved fidelity in physical simulations, enhancing the realism of the generated objects.

### Strengths
### Originality
- The integration of LLM-CoT-IPR into the generation process effectively leverages textual information.
- The combination of multi-view diffusion priors, video diffusion priors, and predictive models for physical properties establishes an efficient pipeline for generating videos that adhere to physical laws from text inputs.
- Unlike previous methods that manually incorporate physical properties, the introduction of Mixture-of-Experts Material Constitutive Models (MoEMCMs) allows for adaptive estimation of physical properties for local Gaussian primitives.。
### Quality
- The experimental metrics are set up broadly, considering various factors.
- The ablation study is well-structured and effectively demonstrates the impact of components such as LLM-CoT-IPR, velocity loss, and the Mixture-of-Experts Material Constitutive Models (MoEMCMs) on the generation results. The qualitative comparisons provided in the appendix are particularly noteworthy.
### Clarity
- There are no significant grammatical or spelling errors; the writing is relatively clear.
### Significance
The research direction of this paper holds practical significance for video synthesis and augmented reality applications.

### Weaknesses
The qualitative results do not always demonstrate an advantage over the quantitative metrics, which significantly diminishes the confidence and impact of the paper.
There is a lack of comparison regarding efficiency and memory usage.
There are uncertainties regarding the specific implementation of LLM-COT-IPR; does it participate in the optimization of video generation?

### Questions
1. Why is there no direct comparison with existing video models to determine whether they can accurately represent the corresponding physical laws?  
2. In Table 1, there are multiple comparisons with baseline metrics; however, outside of the average metrics, there is no significant advantage in the individual metrics. Furthermore, how does OmniPhysGS achieve a score of 0.2 for the FVD metric on Jelly, which significantly surpasses all other methods?  
3. In Figure 3, the "Pancakes melting" example displays a noticeable scale deformation. What causes this? Is it related to the method employed?  
4. The user study results are considerably better than the baseline; could these be included in the main text? This might enhance the persuasiveness of the paper.  
5. Could the authors summarize the fundamental differences between the method proposed in this paper and previous methods? Particularly concerning OmniPhysGS, this would help me gain a clearer understanding of the contributions of this paper.  
6. How do the various methods compare in terms of efficiency and memory usage? It would be beneficial to present this comparison in a table.  
7. I have some questions regarding the specific implementation of LLM-COT-IPR. According to Algorithm 1, it appears to involve scoring images. I am curious whether LLM-COT-IPR participates in the iterative process of video generation or is limited to optimizing image prompts.  

If the authors can effectively address my concerns, I would consider improving my rating.

### Soundness
3

### Presentation
2

### Contribution
2
