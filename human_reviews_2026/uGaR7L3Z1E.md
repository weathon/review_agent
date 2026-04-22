# Follow-Your-Shape: Shape-Aware Image Editing via Trajectory-Guided Region Control

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
While recent flow-based image editing models demonstrate general-purpose capabilities across diverse tasks, they often struggle to specialize in challenging scenarios---particularly those involving large-scale shape transformations. 
When performing such structural edits, these methods either fail to achieve the intended shape change or inadvertently alter non-target regions, resulting in degraded background quality. 
We propose $\textbf{Follow-Your-Shape}$, a training- and mask-free framework that supports precise and controllable editing of object shapes while strictly preserving non-target content. 
Motivated by the divergence between inversion and editing trajectories, we compute a $\textbf{Trajectory Divergence Map (TDM)}$ by comparing token-wise velocity differences between the inversion and denoising paths. 
The TDM enables precise localization of editable regions and guides a $\textbf{Scheduled KV Injection}$ mechanism that ensures stable and faithful editing.
To facilitate a rigorous evaluation, we introduce $\textit{\textbf{ReShapeBench}}$, a new benchmark comprising 120 new images and enriched prompt pairs specifically curated for shape-aware editing.
Experiments demonstrate that our method achieves superior editability and visual fidelity, particularly in tasks requiring large-scale shape replacement.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces EditAnyShape, a training- and mask-free image editing framework designed for precise, shape-aware object editing while strictly preserving background content. Existing editing methods struggle with large-scale shape modifications—they either fail to achieve intended shape changes or degrade undesired regions. The authors propose a novel Trajectory Divergence Map (TDM), which localizes editable regions by analyzing token-wise velocity differences between model inversion and prompt-driven editing paths in latent space. This map guides a scheduled Key-Value (KV) injection mechanism, stabilizing initial trajectories, enabling region-selective edits, and maintaining non-target regions. A new benchmark, ReShapeBench, is introduced, containing 120 curated images and paired prompts to evaluate shape-aware editing models. Experiments show that EditAnyShape achieves state-of-the-art performance for both editability and visual fidelity, especially in challenging large-scale shape transformation tasks.

### Strengths
- This paper addresses the challenge of large-scale shape transformations in image editing, which is a well-defined problem
- EditAnyShape is a training and mask free that automatically localizes editable regions (using TDM), also new benchmark ReShapeBench is introduced improving the evaluation of shape-aware editing
- This framework achieves state-of-the-art quantitative performance across multiple metrics

### Weaknesses
While EditAnyShape presents a well-structured approach to large-scale shape transformation, several aspects remain underexplored or insufficiently analyzed.

- **Lack of analysis on TDM interpretability and reliability.**
    
    The paper claims that the Trajectory Divergence Map (TDM) can precisely localize editable regions, unlike the noisy cross-attention maps. However, it remains unclear how effectively TDM identifies the user-intended foreground. Although the authors argue that cross-attention–based attention injection is noisy and inconsistent, they do not provide sufficient experimental evidence demonstrating that TDM is indeed cleaner or more reliable.
    
    It would strengthen the paper to include visual or quantitative comparisons between TDM and cross-attention maps (e.g., attention heatmaps, spatial correlation, or IoU-based analysis). This would clarify whether TDM truly captures semantic differences rather than inheriting similar noise patterns. Conceptually, since TDM is derived from latent velocity differences between source and edit trajectories, it is not immediately intuitive why it would be less noisy or more foreground-focused than traditional cross-attention maps. More empirical validation is needed to justify this claim.
    
- **Potential limitations of KV injection and region blending.**
    
    In Section 3.2.2, the framework performs Key-Value (KV) injection across stages 1–3, blending features from inversion and editing trajectories. However, this mechanism may face challenges in cases involving significant structural layout changes or large scale variations (e.g., when a large object is transformed into a much smaller one) Because KV injection reuses structural information from the source trajectory, there is a risk that unintended layout features from the original image could be reintroduced into the edited regions, potentially interfering with the desired transformation.
    
    A detailed ablation or visualization showing how KV injection affects spatial consistency and layout preservation in such extreme transformation cases would improve the paper’s rigor. It would also be helpful to discuss whether TDM-based modulation sufficiently suppresses unintended KV influence in non-foreground areas.

### Questions
- As mentioned in the Weaknesses, TDM is computed as a latent-level difference between the inversion and editing trajectories. However, it is not intuitively clear how such a latent-space divergence can yield a spatially clear and human-interpretable distinction of editable regions. Could the authors provide further analysis or visualization to support this claim? For instance, how does TDM behave compared to traditional cross-attention maps in terms of localization clarity or noise robustness?
- Regarding the Key-Value (KV) injection process: in cases involving significant structural layout changes (e.g., when a large object is transformed into a smaller one or when the object’s spatial footprint drastically shifts), how does the model prevent source layout information from unintentionally being injected into non-target areas? An ablation or visual example illustrating this effect would help clarify the robustness of the proposed injection mechanism.

### Soundness
3

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
This paper proposes **EditAnyShape**, a training-free and mask-free method for shape-aware image editing using flow-based generative models. The method aims to enable large-scale shape manipulation while maintaining background fidelity. The main technical contribution is the Trajectory Divergence Map (TDM), a token-wise spatial map computed from the divergence between the velocity fields of a source inversion trajectory and its edited counterpart. TDM highlights regions of greatest semantic deviation, effectively predicting “where to edit” without an external mask or cross-attention supervision. Editing is then achieved through a Scheduled KV Injection mechanism in three stages: 1) Early-stage unconditional KV reuse for stability; 2) Middle-stage TDM aggregation; and 3) Late-stage localized KV injection guided by the final TDM (optionally assisted by a ControlNet).

In the experimental results, the authors introduce ReShapeBench, a curated dataset of 120 images with 290 prompt pairs emphasizing geometric and structural changes rather than texture edits.

### Strengths
1. The paper identifies a significant and challenging problem. Large-scale shape editing is a major failure case for most SOTA methods.
2. Using divergence between source and target flow trajectories to infer editable regions is original and well-motivated. It moves beyond cross-attention saliency or explicit user masks toward a model-intrinsic notion of semantic locality.
3. The proposed approach can be applied to existing pre-trained flow models without finetuning or additional training data. It integrates smoothly into existing editors, e.g., FLUX and KV-Edit.
4. The mathematical definition of TDM in Equations (1-- 4) is consistent, and the scheduled KV injection is clearly described and empirically ablated.
5. The provided visual examples demonstrate strong control over shape deformation and good background preservation, especially in scenes where previous mask-free editors distort context.
6. The contributed benchmark, ReShapeBench, is expected to fill an evaluation gap by focusing explicitly on shape-aware editing. The dataset and evaluation protocols are carefully designed and could be useful to the community.

### Weaknesses
1. The proposed method employs ControlNet guidance (depth/canny maps) during the final editing stage to preserve structure and edges. However, none of the baselines (FlowEdit, RF-Solver, KV-Edit, MasaCtrl, DiT4Edit, etc.) use ControlNet or equivalent structural conditioning. As ControlNet introduces a strong external geometric prior, the resulting improvements in PSNR, LPIPS, and boundary fidelity cannot be attributed solely to the proposed TDM mechanism. As a result, the unfair bassline comparison could invalidate the main quantitative claims in Table 1.
2. The technical novelty is incremental. The Trajectory Divergence Map is a new signal, but the rest of the framework (KV injection, flow editing, scheduled feature blending) extends existing works such as KV-Edit and Stable-Flow. Hence, the novelty lies primarily in the source of the region-control signal (velocity divergence), not in the editing pipeline itself.
3. The constructed ReShapeBench dataset, while valuable, is relatively small (120 images). Its representativeness for general photo editing, especially non-synthetic images, remains uncertain.
4. The presentation lacks runtime and memory analysis. In particular, TDM computation requires per-timestep velocity differences and KV caching, likely increasing GPU memory. The paper also does not quantify overhead relative to baselines.
5. As the proposed method is a training-free, language-driven system, it would be helpful to show cases where TDM misidentifies regions or when prompt wording yields ambiguous edits.

### Questions
The authors are suggested to respond to those raised in **Weaknesses.**

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
The paper introduces EditAnyShape, a training-free, mask-free framework for shape-aware image editing built on the FLUX.1 rectified-flow model. It targets large geometric and structural changes of foreground objects driven by text prompts, while maintaining high fidelity in non-edited regions. The core idea is a Trajectory Divergence Map (TDM) that measures token-wise velocity differences between inversion and editing trajectories to localize where semantics should change, which then guides staged KV injection and ControlNet conditioning. The authors also propose ReShapeBench, a curated benchmark of 120 images and 290 shape-editing cases, and show that their method outperforms diffusion-based and flow-based baselines on aesthetic quality, text alignment, and background preservation.

### Strengths
- Trajectory-based region control. TDM offers a conceptually grounded way to localize editable regions by exploiting velocity differences in rectified-flow trajectories, avoiding reliance on noisy attention maps or external segmentation.

- Training-free and mask-free pipeline. The method operates purely with a pre-trained FLUX model and does not require hand-drawn or model-generated masks, which reduces annotation overhead and simplifies deployment.


- Consistent empirical gains. Across ReShapeBench, the approach improves aesthetic scores, maintains better background similarity (PSNR/LPIPS), and achieves stronger text-image alignment than diffusion- and flow-based baselines.


- Targeted ablations. Parameter studies on early stabilization length and ControlNet timing clarify how different components contribute and highlight reasonable default settings.


- Task-focused benchmark. ReShapeBench explicitly emphasizes shape and structural transitions with clear criteria and curated prompts, helping the community study this specific editing regime.

### Weaknesses
- Global PSNR and LPIPS cannot disentangle background preservation from foreground edits, so the core claim of “preserving non-edited regions” is only indirectly tested. Some region-restricted metrics would provide stronger evidence.

- The paper does not empirically compare TDM to simpler region-selection strategies (e.g. DiffEdit-style prediction differences, cross-attention masks) when used within the same staged KV injection scheme, leaving the unique benefit of TDM somewhat under-quantified.

- Runtime and memory costs introduced by TDM accumulation and extra passes are not summarized in the main text.

- The paper primarily shows successful edits and does not systematically discuss where TDM fails (e.g. thin structures, cluttered scenes, overlapping objects), which would be valuable for practitioners.

### Questions
- Could you report background-only and foreground-only PSNR/LPIPS (e.g. using TDM masks or manual masks on a subset) to more directly validate background preservation?

- How does TDM compare to a DiffEdit-style prediction-difference mask or to cross-attention-based masks when combined with your staged KV injection and ControlNet schedule?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on improving precision of large-scale shape editing in images. In contrast to binary segmentation masks or cross-attention masks, which tend to be noisy and unreliable for shape transformation, TDM is based on the semantic difference between source and target denoising trajectories.

The core contributions are:
(1) EditAnyShape -- a novel training-free and mask-free approach for controllable object shape editing in images.
(2) A selective KV injection mechanism based on a Trajectory Divergence Map (TDM). This TDM serves as a guide for more precise and localized KV injection, preventing unintended edits.
(3) A Scheduled KV Injection approach to only introduce TDM-guided editing in later stages of the denoising process, in order to prevent instability in the early, high-noise denoising stages.
(4) A new benchmark, ReShapeBench, consisting of 120 (image, prompt) pairs for shape-aware editing.

### Strengths
- The general insight of the paper to use magnitude of trajectory difference for localizing edits is both clever and intuitive.
- The methods were generally well-motivated and clearly explained.
- The paper is overall well-written and polished.

### Weaknesses
- Figure 2 is quite confusing. For example, is that row in the top right a legend for left and right sides of the figure? If so, it could be labelled more clearly. Also, for the bottom figure, the outline colors of the frames (particularly blue and orange) are very hard to notice.
- The quantitative evaluation only includes results on the paper's custom evaluation dataset, but does not report results of a third-party editing dataset, such as PIE-Bench. Although PIE-Bench does not isolate shape changes, it would be more convincing to see if this method can perform on-par with other editing methods on an unbiased outside dataset.
- In some qualitative comparisons of Figure 6, it looks like other methods seem similar to the EditAnyShape framework. For example, in the third-to-last row, KV-Edit looks like it is making the figure 8 ball and preserving the background well. What is the inherent benefit over kv-edit (also training-free)?
- The ablation study doesn't actually compare to the base Flux.1-dev model, without adding EditAnyShape framework. It would be informative to know how well does the base model do for editing by default?

### Questions
- In Figure 2 (left), what is "Other shape editing"?
- Is there some way to visualize the TDM and compare it to the ground truth difference in foreground masks of the source and target images?
- Perhaps Section 4.2 about benchmark construction would better belong in the Methods section or its own section, rather than in Experiments?
- Section 4.2: What is the purpose of the third "general evaluation set"? Is it that subsets 1 and 2 are for training and subset 3 is for testing?
- Since this approach is training-free, it would be great to see it applied to other open-source flow-matching models, like

### Soundness
3

### Presentation
3

### Contribution
4
