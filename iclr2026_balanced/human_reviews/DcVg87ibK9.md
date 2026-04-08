## Human Reviewer 1

### Summary
This paper addresses the challenge of physically plausible image composition, particularly the failure of existing models to handle complex lighting, shadows, and reflections. It propose SHINE, a "training-free" framework that is effectively a test-time optimization process. 
The framework consists of three main components: 
1) Manifold-Steered Anchor (MSA) Loss, a guidance loss that uses a pretrained adapter (e.g., IP-Adapter) and the base model's predictions as an anchor to steer the latent variable; 
2) Degradation-Suppression Guidance (DSG), a form of negative guidance achieved by blurring specific query features ($Q_{img}$) inside the DiT to avoid artifacts; 
3) Adaptive Background Blending (ABB), a technique that uses cross-attention maps in early sampling steps to create smoother blends. 

Additionally, the authors contribute a new benchmark for evaluating composition in complex lighting conditions.

### Strengths
1.  Good Qualitative Results: The qualitative results presented (e.g., Fig. 1, 6) are impressive, demonstrating a high degree of physical realism in challenging scenarios.
2.  Benchmark Contribution: The newly proposed ComplexCompo dataset is a solid contribution, offering a valuable tool for future research.

### Weaknesses
1.  Impractical Computational Cost: The "training-free" claim is misleading, as the method is a test-time optimization with *extreme computational costs*. Algorithm 1 and Table 5 indicate that it requires $k=10$ optimization iterations *per* denoising step $t$. For a 20-step process, this amounts to ~*200* forward and backward passes. On a 12B parameter model like FLUX, this cost is computationally prohibitive and makes any comparison against single-forward-pass baselines impractical.
2.  Brittle Pre-processing Pipeline: To avoid image inversion, the paper introduces a (Sec 3.1) pre-processing pipeline that depends on *two* external models: a VLM for captioning and an inpainting model for creating $x^{init}$.  If the VLM caption is wrong (e.g., wrong color), the entire composition fails (as indicated in Sec 5-Limitation). It seems that this merely trades one failure point (inversion) for two new, uncontrolled failure points (VLM failure, inpainting failure)?
3.  Limited Novelty and Model-Specific:
    * MSA Loss: Re-naming a known technique (SDS) applied between an adapter and a base model does not constitute a significant novel contribution.
    * DSG: This is an empirical, model-specific hack (via brute-force experimentation) for the FLUX architecture, it seems to lack a theoretical justification and guarantee of generalizability.

### Questions
1.  Compute Cost Quantification: Could you provide a fair, wall-clock time comparison on identical hardware, e.g., how much time does SHINE (with $k=10$) take to generate one 1024x1024 image, versus the time required by baselines like AnyDoor or UniCombine?
2.  DSG Generalizability: You claim in Appendix E that DSG applies to SDXL, SD3.5, and PixArt. Was the "blur $Q_{img}$" heuristic applicable *out-of-the-box* to all these architectures, or did each one require a *new* round of brute-force experimentation to find a *different* hack (e.g., "blur $K_{txt}$")?
3.  Pre-processing Failure Rate: I am curious about the actual failure rate of your [VLM + Inpainting pre-processing pipeline], when running the 300-sample ComplexCompo benchmark.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 2

### Summary
The paper presents SHINE, a training-free strategy for image composition using FLUX. Since previous inversion-based and attention manipulation–based approaches struggle to achieve robust image composition, the authors introduce Non-Inversion Latent Preparation, Manifold-Steered Anchor (MSA) Loss, Degradation-Suppression Guidance (DSG), and Adaptive Background Blending (ABB). FLUX.1-dev is a CFG-distilled model, which causes high inversion errors. The subject image is captioned by a VLM, and an inpainting model creates an initial image based on the caption. Building on this, the authors perform a one-step forward diffusion to obtain and manipulate the latent. The Manifold-Steered Anchor (MSA) Loss utilizes the velocity predicted by the original noisy latent as anchor noise, along with the adapter-guided velocity prediction using the subject image’s latent. Degradation-Suppression Guidance (DSG) is inspired by negative prompting and constructs a negative velocity prediction by blurring the attention component $Q_{img}$ of FLUX. Adaptive Background Blending (ABB) selects between the attention mask and a user-provided mask depending on the timestep. Experiments demonstrate the effectiveness of the proposed framework for image composition, both quantitatively and qualitatively. The authors also propose ComplexCompo, a benchmark for challenging composition scenarios.

### Strengths
1. The design of the Manifold-Steered Anchor Loss, which takes into account the flow-matching process between FLUX.1-dev and the adapter model (InstantCharacter), is well-motivated and technically sound.
2. The in-depth analysis of the attention blocks in FLUX.1-dev (Sec. C & D) and the derivation of the subsequent improvement, Degradation-Suppression Guidance, are particularly interesting.
3. The Adaptive Background Blending method is simple yet powerful, and it appears broadly applicable to various image editing tasks.
4. The paper offers numerous insights drawn from a comprehensive set of dense experiments.
5. The proposed methods are model- and adapter-agnostic, making them broadly applicable across different architectures and integration settings.

### Weaknesses
Overall, the paper is still a strong and well-executed, however, it could be further improved if the authors provide additional details and clarifications on certain points regarding the following aspects:

1. In L159-161, the term “inpainted background image” appears somewhat inappropriate in this context. Phrases such as “scene image” or “image to which the subject is attached” would be more accurate and better convey the intended meaning.

2. In L186-187, the authors state that they abandon the inversion process and instead apply one-step forward diffusion. I interpret this as implying that “the effect of inversion can be approximated by directly adding noise corresponding to a one specific timestep”. If this interpretation is correct, a theoretical justification should be provided. Additionally, the authors should clarify how this approach differs from the inversion-skipping technique used in EEdit [1].

3. While numerical improvements are observed in Config A and C of the ablation study, it remains unclear whether DSG truly acts as a negative guidance vector when blurring the attention component $Q_{img}$ derived from the text embedding $c$ extracted by the VLM. Although FLUX is a CFG-distilled model where non-sensical or explicit negative prompts are often ineffective (as shown in Fig. 4(a)), this behavior may differ in other MM-DiT–based models such as SD3.5. It would strengthen the paper to include an analysis comparing simple negative prompting and Degradation-Suppression Guidance in SD3.5 to better quantify their differences.

4. Please specify which inpainting model (e.g., Brushedit, BrushNet, PowerPaint, Flux.1 fill-dev) and which VLM (e.g., BLIP-3, InternVL, LLaVA-1.5) were used in the proposed frame work and evaluation.

5. In Alg. 1, the origin of $z^{bg}$ is unclear. Is it defined as $z^{bg} = (1 - M^{user}) \odot z^{init}$ and $z^{subject} =  M^{user} \odot z^{init}$? Please clarify this in the text or Alg. 1.

6. In Tab. 1 & 3, it would be beneficial to include results for Config A (the input image generated by the inpainting model). This addition would help readers better understand the performance gap and the impact of subsequent modules at a glance.

[1] EEdit : Rethinking the Spatial and Temporal Redundancy for Efficient Image Editing, ICCV 2025

### Questions
1. In L260-262, how are the attention components systematically perturbed? Which 2D Gaussian filter $G$ (kernel size) is used?
2. How much time and GPU memory are required to generate a composited image using FLUX.1-dev + Adapter and + LoRA? Also, for the other baselines?

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
5

---

## Human Reviewer 3

### Summary
SHINE is a training-free framework designed for seamless, high-fidelity insertion with neutralized errors. It introduces the manifold-steered anchor loss, which leverages pretrained customization adapters to guide latent representations, ensuring accurate subject representation while maintaining the integrity of the background. Additionally, SHINE incorporates degradation-suppression guidance and adaptive background blending to prevent low-quality results and visible seams, enhancing the overall output quality.
In response to the lack of rigorous benchmarks in this domain, the authors propose ComplexCompo, a benchmark suite that tests the framework under challenging conditions.

### Strengths
1. The article presents a clear expression of its ideas, with a thorough visualization analysis of the motivation and proposed modules.

2. The experiments are comprehensive, and the results appear highly promising.

3. The work makes notable breakthroughs in addressing issues related to lighting realism and resolution rigidity in image synthesis.

### Weaknesses
1. As shown in Table 1, the method performs well in Subject Identity Consistency, but there is room for improvement in Background Retention. Could you provide further insights and possible approaches for enhancing this aspect?

2. In the Image Quality evaluation, the models used for assessment are somewhat limited. Could you consider incorporating more diverse and widely-adopted image quality evaluation models, such as UnifiedReward [1] and HPSv3 [2], to provide a more comprehensive analysis?

[1] Unified Reward Model for Multimodal Understanding and Generation. 2025

[2] HPSv3: Towards Wide-Spectrum Human Preference Score. 2025

### Questions
See Weakness.

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
4