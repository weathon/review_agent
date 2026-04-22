# TransLight: Image-Guided Customized Lighting Control with Generative Decoupling

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 2, 6, 6

## Abstract
Most existing illumination-editing approaches fail to simultaneously provide customized control of light effects and preserve content integrity. This makes them less effective for practical lighting stylization requirements, especially in the challenging task of transferring complex light effects from a reference image to a user-specified target image. To address this problem, we propose TransLight, a novel framework that enables high-fidelity and high-freedom transfer of light effects. Extracting the light effect from the reference image is the most critical and challenging step in our method. The difficulty lies in the complex geometric structure features embedded in light effects that are highly coupled with content in real-world scenarios. To achieve this, we first present Generative Decoupling, where two fine-tuned diffusion models are used to accurately separate image content and light effects, generating a newly curated, million-scale dataset of image–content–light triplets. Then, we employ IC-Light as the generative model and train our model with our triplets, injecting the reference lighting image as an additional conditioning signal. The resulting TransLight model enables customized and natural transfer of diverse light effects. Notably, by thoroughly disentangling light effects from reference images, our generative decoupling strategy endows TransLight with highly flexible illumination control. Experimental results establish TransLight as the first method to successfully transfer light effects with geometric structure across disparate images, delivering more customized illumination control than existing techniques and charting new directions for research in illumination harmonization and editing.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents TransLight, a novel framework for image-guided light effect transfer, which enables high-fidelity transfer of complex lighting effects from a reference image to a target image while preserving content consistency. The core lies in the proposed Generative Decoupling strategy, which employs two diffusion models to disentangle content and light effects in real images, thereby constructing a large-scale dataset of image–content–light triplets. Having these data, the authors design a two-stage training framework: the first stage applies LoRA fine-tuning to preserve content integrity, while the second stage leverages ControlNet to learn light effect injection and geometric control. The final model supports controllable transformations of lighting effects, including translation, rotation, and intensity adjustment, achieving customizable illumination transfer that surpasses existing methods in both Light-FID and perceptual quality.

### Strengths
- The paper introduces a new problem, image-guided light effect transfer, which is distinct from traditional relighting or style transfer tasks, demonstrating originality and novelty.
- The proposed Generative Decoupling framework effectively separates content and light effects using two diffusion models, and the subsequent two-stage training with LoRA and ControlNet achieves a good balance between content stability and controllable lighting manipulation.
- The authors construct a large-scale dataset of over one million image-content-light triplets and conduct extensive experiments showing significant improvements over existing methods in both quantitative metrics and qualitative visual fidelity.
- The model supports flexible control of lighting position, direction, and intensity, and the results exhibit natural integration between content and light, indicating that the network has successfully learned a disentangled representation of lighting effects.

### Weaknesses
- The method focuses on perceptual-level lighting transfer (e.g., lens flares, streaks) without modeling scene geometry, material properties, or physically consistent illumination. As a result, the generated lighting effects may not correspond to physically plausible relighting outcomes.
- The million-scale triplet dataset is constructed through automatic generation and filtering using a VLM, yet the paper does not provide sufficient details about data diversity, quality control, or potential biases, and the cost of VLM is not cheap, raising concerns about reproducibility and dataset reliability.
- The light extraction module plays a key role in generating training triplets, but the paper does not analyze how imperfect or incomplete extraction affects TransLight’s training and final generation quality. No quantitative or failure-case analysis is provided.
- The paper presents relatively few qualitative examples compared to prior works such as IC-Light. The lack of diverse visual results weakens the empirical support for the method’s generality.

### Questions
In Section 3.3, Stage 1 is described as “a simple addition of the content image and the background light effects image.” Could the authors clarify whether the model receives this summed image as a single input, or two separate input channels (content + light concatenated)?

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
This paper transfers light effects from a reference to a target via “generative decoupling” with two diffusion models: a light-removal model producing content-only and a light-extraction model producing an effects-only map. They mine images with a VLM to form image/content/effects triplets and filter by features. Training is two-stage: (1) LoRA to enforce content preservation (suppress lighting), (2) ControlNet to inject the extracted effects map. At inference, the effects map is injected and can be rigidly transformed (translate/scale/flip) to harmonize illumination, not requiring explicit geometry or light-source modeling.

### Strengths
1) The method is simple and is an interesting decomposition into content vs. light effects

2) Scalable data pipeline for building triplets without manual labels.

3) Practical for creative workflows like image harmonization, where exact physical relighting is unnecessary.

4) The lighting removal is perhaps the most interesting part of the pipeline that can suppress strong lighting phenomena like glares, specular effects, etc.

### Weaknesses
1) One of the main claimed contributions of the paper is that it is“First to transfer light effects with geometric structure across disparate images,” which is not correct. Prior work already transfers lighting in a physically grounded way: Latent Intrinsics (cited) and its generative adaptation -- LumiNet: Latent Intrinsics meets diffusion models for indoor scene relighting (CVPR 2025; not cited) transfer lighting codes from a source to a target while handling large geometric mismatches in complex indoor scenes. Also, these comparisons are missing in the current paper.

2) While the generative decoupling construction is interesting, the method does not transfer lighting. It is illumination effects harmonization via a 2D effects layer. There is no discussion in the paper on how the proposed method differs from diffusion-based relighting methods that are physically motivated (like Latent Intrinsics, LumiNet, or Diffusion Renderer (CVPR 2025), which is again not cited in the paper. 

3) No evaluation on standard relighting datasets (MIT Multi-Illumination, BIG-Time Time Lapse Dataset, etc.).  

4) Does the method understand lighting semantics, such as transferring the lighting from switching on a light bulb in one scene to another? The proposed approach seems to be spatially biased, as it focuses on pixel-aligned matching and then harmonization between the source and target images, rather than employing global illumination reasoning.

5) “Light FID” is unvalidated for lighting structure; FIDs are known to be biased and are not a good measure of lighting phenomena. Missing structure-aware metrics using a standard dataset and a user study.

6) Missing limitations and failure examples.

### Questions
1) What exactly do authors mean by “light effects with geometric structure”? Which phenomena are included (cast shadows, speculars, bloom, flares, caustics), and which are excluded?
2) Can the authors elaborate on how this decomposition differs from traditional intrinsic image decomposition?
3) How does this approach differ technically from diffusion-based, physically motivated relighting methods (Latent Intrinsics, LumiNet, Diffusion Renderer)?

4) Can this method demonstrate source-level semantics (toggle a lamp, move a point light, change intensity/color) and report accuracy under controlled edits?

5) Under a large geometry mismatch (indoor vs outdoor), what are the failure modes of the 2D effects transfer? Please quantify and show examples. Also, how does the method perform on indoor scenes?

6) The paper mentions content leakage multiple times. If the extraction is “content-free,” is it possible to quantify the content leakage?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a method to transfer light with two considerations: 1. the scene illumination and 2 the lighting effects. The two perspectives are explicitly decomposed with independent training data preparation.

### Strengths
1.	The problem modeling is good. Lighting effects are important for real photo but less discussed in relighting research.
2.	The data processing looks satisfying, and I encourage authors to release it.
3.	The model design looks reasonable and foreseeably easy to reproduce (if with proper data)

### Weaknesses
1.	From the results, it seems that rotational/directional illumination is less discussed. But I can understand that the main purpose is to achieve those lighting effects. 
2.	The model seems relatively weak in changing the illumination direction in its input images – many results display the inability in changing light direction.
3.	Most results show very strong artistic lighting expressions but in real photos they are relatively rare, so the scope may be a bit narrow. But I understand that this is the whole point of this framework: to realize those lighting expressions.

### Questions
1. Why we need both  stage 1 and 2 in figure 2 c? Are they just ablation or do we have special considerations?

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
2

### Summary
This paper introduces TransLight, a novel framework for image-guided light effect transfer. The key goal is to transfer customized light effects with complex geometric structure from a reference image to a target image, while preserving the target image content. The core idea is a generative decoupling strategy, where two fine-tuned diffusion models are used to separate content and light in natural images. Using this, the authors construct over one million image-content-light triplets, which are then used to train a two-stage generation pipeline consisting of LoRA-based fine-tuning and ControlNet conditioning. The method outperforms existing relighting and editing techniques both quantitatively (e.g., Light-FID) and qualitatively.

### Strengths
- The formulation of the “light effect transfer” task—distinct from conventional relighting or style transfer—is novel and well-motivated. The generative decoupling framework using two diffusion models is an innovative approach to address content-light entanglement, which has not been effectively resolved in prior work.
- The pipeline is well-engineered, with thoughtful design choices such as DINOv2-based filtering and multi-source relighting in training. Ablation studies confirm the effectiveness of LoRA pretraining and data filtering. The method achieves impressive improvements over IC-Light in both PSNR/SSIM and Light-FID.
- The paper is generally well-written and logically organized. Figures are clear and informative, especially those visualizing decoupling and transfer effects. The proposed framework is well explained across training and inference phases.
- The ability to transfer localized light effects from arbitrary references has strong applications in artistic editing, AR/VR, and photography enhancement. This work may serve as a foundation for future light-aware generative modeling.

### Weaknesses
- Inconsistent training supervision for decoupling: The light removal model is trained with both synthesized (IS) and relighted images (IR), while the light extraction model uses only IS. This inconsistency is not justified and may lead to a domain gap in model robustness. A clearer rationale or ablation would strengthen the method section.
- Threshold γ lacks theoretical justification: The cosine similarity threshold γ=0.98 is crucial in triplet filtering. However, the choice seems empirical and lacks sensitivity analysis or theoretical lower bounds. It remains unclear how this affects recall vs. precision in data quality.
- Two-stage training not rigorously compared: While the authors adopt a two-stage pipeline (LoRA → ControlNet), there is no analysis of whether joint training or alternative one-stage schemes could achieve similar or better performance. The added complexity needs clearer justification.

### Questions
- Can the authors provide a quantitative comparison between the light extraction model trained with and without the IR augmentation used in the light removal model? How significant is the performance drop in complex lighting scenarios?
- How sensitive is the performance to the choice of threshold γ in the triplet filtering stage? Could the authors include a small ablation or discussion of the impact of lower or higher γ? Does the hard threshold γ have a lower limit?
- In inference, does TransLight use both LoRA and ControlNet models? Could a single unified network achieve similar results? How is the runtime efficiency impacted?
- How does TransLight perform when the reference image contains subtle or ambient lighting effects rather than high-intensity structured light?
- Would using a unified prompt-to-light map generation strategy (e.g., text-guided diffusion) eliminate the need for large-scale triplet construction?

### Soundness
3

### Presentation
3

### Contribution
3
