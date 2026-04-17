# W-EDIT: A Wavelet-Based Frequency-Aware Framework for Text-Driven Image Editing

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4, 4

## Abstract
While recent advances in Diffusion Transformers (DiTs) have significantly advanced text-to-image generation, text-driven image editing remains challenging. Existing approaches either struggle to balance structural preservation with flexible modifications or require costly fine-tuning of large models. To address this, We introduce W-Edit, a training-free framework for text-driven image editing based on wavelet-based frequency-aware feature decomposition. W-Edit employs wavelet transforms to decompose diffusion features into multi-scale frequency bands, disentangling structural anchors from editable details. A lightweight replacement module selectively injects these components into pretrained models, while an inversion-based frequency modulation strategy refines sampling trajectories using structural cues from attention features. Extensive experiments demonstrate that W-Edit achieves high-quality results across a wide range of editing scenarios, outperforming previous training-free approaches. Our method establishes frequency-based modulation as both a sound and efficient solution for controllable image editing.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents W-Edit, a training-free text-driven image editing framework using wavelet-based frequency decomposition. It separates structural (low-frequency) and detail (high-frequency) features, enabling controllable edits without retraining diffusion models. By integrating frequency modulation into attention and inversion processes, W-Edit achieves high-quality, structure-preserving edits. Experiments show superior fidelity and realism compared to prior training-free methods.

### Strengths
W-Edit introduces a novel training-free framework for text-driven image editing that eliminates the need for costly retraining, making it efficient and practical for real-world applications. By leveraging wavelet-based frequency decomposition, it effectively disentangles global structural features from fine details, enabling precise edits while preserving the overall layout and coherence of the reference image. The method integrates seamlessly with pretrained diffusion models and demonstrates strong generalization across multiple architectures, achieving superior visual quality, realism, and alignment with textual instructions in both quantitative benchmarks and user studies.

### Weaknesses
1. The experimental setup for the block-wise frequency analysis of Diffusion Transformers is not clearly provided. Which dataset was used for this analysis? If it was conducted using only the image in Figure 1, can the results be considered generalizable?

2. In Figure 4, there is insufficient interpretation as to why the high-frequency components in the SingleStreamBlocks appear only in a specific region.

3. It is unclear whether the roles of early and later blocks in Diffusion Transformers are directly comparable to those in U-Net, where early blocks encode structural information and later blocks encode fine details. More explanation is needed on whether this functional separation holds consistently in DiTs.

### Questions
1. In the block-wise frequency analysis shown in Figure 4, why do the high-frequency components in the SingleStreamBlocks appear only in a specific region?

2. In Figure 5, it would be helpful to include comparisons with existing methods for a variety of edits, such as object removal or attribute modification.

3. Stable-Flow emphasizes that injecting features only into the Vital Layers is key for image editing in the DiT architecture. It would be informative to provide quantitative measures showing how well the layers selected in this study align with the Vital Layers identified in Stable-Flow.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes W-Edit, a DiT(FLUX)-based image editing model using wavelet-based frequency-aware feature decomposition. The authors point out that entanglement of global semantics and local signals creates difficulty in preserving structure and making precise modifications. Typically, low-frequency components in features encode layout and semantics, while high-frequency components capture texture and detail. They analyze DiT features using wavelet decomposition and show that early blocks focus on low-frequency and later blocks on high-frequency information. Based on this, they propose energy-aware feature fusion: replacing DiT features by taking wavelet components from the inversion branch and editing branch, respectively. Experiments show that W-Edit achieves strong performance in training-free image editing, preserving structure, prompt-aligned edits with minimal computational overhead.

### Strengths
- This paper addresses the issue of semantic entanglement between global structure and local details in image editing, which improves the understanding of existing model limitations.
- It presents a wavelet-based frequency decomposition approach to analyze frequency characteristics in DiT feature. Specifically, similar to UNet, they experimentally show that early blocks focus on low frequency and later blocks on high frequency details, which seems to give useful insight to the community.
- The method works in a training-free manner, making it computationally efficient and easy to apply to various pre-trained diffusion models.

### Weaknesses
- This paper aims to disentangle the preservation of object structure (low-frequency) from detailed texture (high-frequency) during editing. However, it remains unclear how effectively the method actually disentangles these components. The current energy-aware adaptive frequency fusion is designed to take low-frequency information from the inversion trajectory and high-frequency details from the editing trajectory. This works well when the goal is to preserve the original structure while changing the details. But what about cases where the structure should change but the details are retained (e.g., non-rigid edit, pose change, or view change without altering the object itself)? In such cases, shouldn’t high-frequency detail information be taken from the inversion trajectory, rather than low-frequency? Overall, I think the adaptive frequency fusion method should be more precisely designed depending on the editing type.
- Fig. 4’s frequency progression analysis seems dependent on the input image and prompt. For example, editing tasks with significant changes should yield different frequency profiles compared to minimal edits. If frequency analysis were performed across various editing cases, the claim would be much stronger.
- In Fig. 4, in the frequency analysis for “FLUX”, the axis labels are not clearly visible.
- For Eq. 4, the explanation for r_mid is insufficient. Why is only mid-frequency energy calculated, and not the total energy? The actual value of r_mid used in the experiments is also missing.
- For Eq. 9, it’s unclear what value was used for the threshold η, and how it is determined during experiments. I believe this threshold should be adaptively chosen according to the editing type for better results.
- There are some missing citations for image editing methods utilizing frequency analysis, such as “FlexiEdit: frequency-aware latent refinement for enhanced non-rigid editing” (ECCV 2024) and “FDS: Frequency-Aware Denoising Score for Text-Guided Latent Diffusion Image Editing” (CVPR 2025).

### Questions
- In Figure 4, regarding the frequency progression analysis, the results for FLUX’s SingleStreamBlock show that only at block index = 30 does the high-frequency component increase significantly. By contrast, in DoubleStreamBlocks, the later blocks tend to have higher high-frequency components. Why do you think, in the case of the SingleStreamBlock, the high-frequency response appears only at block index = 30?
- Comparison with other models: How does Flux 1, Kontext, and Qwen-Image perform by comparison? From my understanding, those two models (Kontext, Qwen-Image) achieve good textual alignment and can selectively preserve or modify structure as needed, even without explicit frequency manipulation. Is this consistent with your analysis? How would you compare these models in the context of frequency-aware editing and structure preservation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
To address the dilemma in which existing text-driven image editing methods either sacrifice structural fidelity for flexible manipulation or demand expensive fine-tuning of large-scale models, the paper introduces W-Edit, a training-free framework for text-driven image editing based on wavelet-based frequency-aware feature decomposition. W-Edit employs wavelet transforms to decompose diffusion features into multi-scale frequency bands, disentangling structural anchors from editable details. The method establishes frequency-based modulation as both a sound and efficient solution for controllable image editing.

### Strengths
- The paper presents a novel frequency-domain perspective for text-driven image editing. Unlike existing methods that primarily manipulate features in the spatial domain, W-Edit integrates wavelet transforms into the feature decomposition of Diffusion Transformers. This frequency-aware modulation framework introduces a new paradigm for disentangling editable attributes, offering enhanced control and consistency in image editing tasks.
- The paper is theoretically well-grounded. By systematically analyzing the internal architecture of Diffusion Transformers, it reveals that early blocks predominantly focus on low-frequency structures, while later blocks refine high frequency details. This observation motivates the introduction of wavelet decomposition to recalibrate and enhance existing image-editing paradigms.
- W-Edit is a training-free framework characterized by lightweight module design; it can be directly applied to off-the-shelf diffusion models without incurring the prohibitive cost of full model retraining, yet still delivers editing results of high fidelity.

### Weaknesses
- The paper offers no systematic guidance for setting the hyper-parameter η (the energy threshold). Although the paper emphasizes that the selection of η is critical, it neither conducts a sensitivity analysis nor devises an adaptive scheduling scheme, potentially compromising the method’s generalizability across diverse datasets.
- The limited sample size of the subjective assessment may undermine the statistical robustness of the conclusions. While the experimental section offers extensive quantitative and qualitative evaluations, the user study recruited only five participants—an insufficient number to yield statistically reliable inferences.

### Questions
- Regarding the selection of the hyper-parameter η, could the authors supply a more systematic sensitivity analysis and, ideally, a data-adaptive scheduling protocol that eliminates manual tuning?
- Although quantitative metrics furnish unambiguous experimental indicators, user perception remains an indispensable dimension in image editing. It is recommended that the user-study component be expanded to corroborate the paper’s conclusions more convincingly.
- The theoretical premise that “early blocks of Diffusion Transformers predominantly encode low-frequency structure whereas later blocks specialize in high-frequency refinement” underpins the entire subsequent development of the proposed method. To elevate this claim from an empirical observation to a rigorous theoretical basis, we recommend supplying more compelling evidence. For example, systematically quantifying the frequency bias of each block via spectrally-decomposed energy statistics, reporting statistical significance tests across multiple models and data subsets, and visualizing the block-wise frequency response curves—so that the asserted frequency progression is firmly corroborated before being leveraged as the foundation of the editing framework.
- As the paper touches on several theoretical aspects, it is advice to clarify details and supplying intuitive diagrams would reduce the cognitive load on readers and improve overall accessibility.

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
This work analyzes the feature representations of MM-DiT blocks from a frequency-domain perspective and study the frequency progression of different layers. With such observation, the paper proposed to first decompose the imtermediate features (features from attention blocks) recursively along the inversion path and then perform feature injection to the generation process based on energy thresholds.

### Strengths
1. The paper gives analysis to the block-wise frequency evolution on both the generation process and the MM-DIT block order, which form a solid basis for later applying control to the attention blocks based on frequency energy threshold. The motivation is clear and the framework is coherent.

2. The proposed method is based on a general frequency evolution principle of diffusion process, which is shared among different settings of diffusion process including Flow-based, VE, VP based on different model structures, thus could be applied to diffusion models with different architechture, only that the behavior of each block should be analyzed concretely and the hyperparameters are different.

3. The proposed method is vasertile to multiple editing tasks and get good performance on the PIE benchmark.

### Weaknesses
1. Several crucial implementation details of the proposed framework remain unclear:  (a) The number of recursive decompositions applied to the intermediate features is not specified. (b) The explicit energy threshold ratio, $\eta$, which determines whether the referenced feature components are injected, is not reported. (c) The precise value of $r_{\text{mid}}$, which defines the mid-frequency energy, is also omitted.

2. The methodology section provides no analysis or rationale for the chosen hyperparameters, and the ablation study includes no experiments evaluating the sensitivity or impact of these key hyperparameters.

3. The underlying intuition behind using Equation (9) for selecting referenced components requires further justification and theoretical or empirical support.

4. Regarding the non-rigid editing task, the qualitative results presented in Fig.12 do not align with the authors’ claim that “W-Edit demonstrates remarkable capability in handling complex non-rigid transformations while preserving structural integrity.” The reviewer notes that all examples in Figure 12 feature simple, homogeneous backgrounds that are distinct from the main editing target, which would generally be considered easy cases for non-rigid editing. Moreover, both the object identity and background consistency are not well preserved in several of the displayed results.

### Questions
Please refer to the weakness above.

1. And since the quantitative experiment mainly provides the result from PIE benchmark, qualitative results on PIE benchmark of each types should be provided, or rather, some of them should be included.

2. Considering the performance of qualitative results provided in Fig.12, could the author also provide the scores of different editing types for the proposed method and compared methods?

Rating could be increased if the questions are well addressed, otherwise could be decreased.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes W-Edit, a training-free framework for text-guided image editing that operates in the frequency domain. The method uses wavelet transform to decompose diffusion model features into multi-scale frequency bands, aiming to separate low-frequency structural components from high-frequency details. By selectively injecting different frequency bands into a pre-trained diffusion model and using an inversion-based modulation strategy, W-Edit seeks to preserve the original image’s structure while applying localized edits in line with a target text prompt. The authors present qualitative results on various editing tasks (e.g. object addition/removal, attribute change, style transfer). While the approach addresses an important problem and shows some promising results, I have serious concerns regarding its novelty, theoretical justification, and evaluation rigor, which lead me to recommend rejection.

### Strengths
Training-Free Framework: W-Edit does not require any fine-tuning of the diffusion model, instead introducing a lightweight wavelet-band injection module and an inversion-based adjustment strategy. This plug-and-play nature is a practical strength – it means the method can be applied on top of existing diffusion models with minimal overhead. Suggestion: Quantify this practical advantage. For example, report the runtime or memory usage compared to training-based methods, or discuss how easy it is to integrate W-Edit into various diffusion backbones. This would underscore the efficiency benefits of being training-free.

Empirical Quality Improvements: The proposed method demonstrates improved fidelity and consistency of edits in the reported results. Quantitatively, W-Edit achieves better scores than the compared baselines on automatic metrics like FID (image realism), CLIPScore (text-image alignment), PSNR/LPIPS (content preservation).

### Weaknesses
Limited Novelty Over Prior Work: The core idea of frequency-based feature modulation is not truly novel. Prior works [1][2][3] have already explored incorporating frequency domain information in diffusion models, image editing and image tokenizers. W-Edit largely recycles these ideas (wavelet transforms, frequency separation) without a clear new conceptual insight beyond combining them in one pipeline. The submission’s claimed contributions – wavelet-guided decomposition, frequency-band injection – are incremental extensions or amalgamations of known techniques. 
Suggestion: To address this, the authors must more explicitly distinguish W-Edit from existing frequency-domain methods. Clearly articulate what is fundamentally new (if anything). Without this, the paper fails to establish a compelling novelty claim.

[1] Phung H, Dao Q, Tran A. Wavelet diffusion models are fast and scalable image generators[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023: 10199-10208.
[2] Esteves C, Suhail M, Makadia A. Spectral image tokenizer[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025: 17181-17190.
[3] Si C, Huang Z, Jiang Y, et al. Freeu: Free lunch in diffusion u-net[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 4733-4743.

No Theoretical Justification or Analysis: The paper provides no principled explanation for why the proposed frequency-band injection and modulation should preserve semantics or improve edit controllability. There are no theoretical derivations, formal proofs, or analytical measures in the main text or appendix to support this claim. Key questions – e.g., why does separating low-frequency structure from high-frequency detail lead to more localized or faithful edits? – are never addressed beyond intuition. The method is essentially heuristic, relying on empirical tuning. Suggestion: The authors should strengthen the paper with some form of theory or at least analytical insight.

Overall, I recommend rejection of this paper in its current form. While W-Edit addresses an important problem and shows some empirical promise, the weak novelty, lack of theoretical grounding significantly undermine the contribution.

### Questions
The method largely repackages known frequency-domain techniques without clear innovation, and it doesn’t provide the necessary analysis or rigorous experiments to convincingly demonstrate its supposed advantages. The authors are encouraged to address the above weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
