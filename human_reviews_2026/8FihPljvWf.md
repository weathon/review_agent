# Video-As-Prompt: Unified Semantic Control for Video Generation

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6, 8

## Abstract
Unified, generalizable semantic control in video generation remains a critical open challenge. Existing methods either introduce artifacts by enforcing inappropriate pixel-wise priors from structure-based controls, or rely on non-generalizable, condition-specific finetuning or task-specific architectures. We introduce Video-As-Prompt (VAP), a new paradigm that reframes this problem as in-context generation. VAP leverages a reference video as a direct semantic prompt, guiding a frozen Video Diffusion Transformer (DiT) via a plug-and-play Mixture-of-Transformers (MoT) expert. This architecture prevents catastrophic forgetting and is guided by a temporally biased position embedding that eliminates spurious mapping priors for robust context retrieval. To power this approach and catalyze future research, we built VAP-Data, the largest dataset for this task with over 100K paired videos across 100 semantic conditions. As a single unified model, VAP sets a new state-of-the-art for open-source methods, achieving a 38.7\% user preference rate that rivals leading condition-specific commercial models. VAP's strong zero-shot generalization and support for various applications mark a significant advance toward general-purpose, controllable video generation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces Video-As-Prompt (VAP), a unified framework for semantic-controlled video generation. Specifically, VAP treats the input video as a semantic prompt, which is used to guide a frozen Video DiT by a plug-and-play Mixture-of-Transformers (MoT) expert. VAP includes three key innovations: 1) in-context MoT for bidirectional information fusion between reference and target videos; 2) a temporally biased RoPE to remove false pixel-wise alignment priors and improve temporal coherence; 3) VAP-Data, a large-scale dataset with over 100K paired videos across 100 semantic conditions (e.g., concept, style, motion, and camera). The experiments show that VAP achieves a 38.7% user preference rate, achieving comparable results with commercial models (e.g., Kling and Vidu), and outperforming existing open-source models.

### Strengths
- The most impressive strength of the work is the wide coverage of diverse semantic control tasks, including the control on concept, style, motion, and camera, under a single framework. To the best of my knowledge, VAP is the first model to achieve this level of task unification for IV2V generation which can substantially enhance the community interest.
- While the proposed architecture looks like ControlNet (i.e., a trainable copy alongside of a frozen model), the proper adaption to this task (as shown in Figure 2) sounds novel and makes sense to me. Since the main goal is similar (i.e., ControlNet handle 2D spatial control from reference images while VAP extends to 3D temporal control from reference videos), but the proper adaption could make the model better generalize to different tasks.
- The proposed temporally biased RoPE is also elegant and sounds well-motivated to me as it can avoid 1-to-1 positional mapping between reference and target frames which would otherwise cause false pixel matching and copy-paste artifacts.
- I have checked the generated video samples provided in the supplementary material and they look high-quality. In addition, based on the quantitative results provided in Table 1, the model matches or even exceeds commercial models (e.g., Kling and Vidu) on several metrics, which show a promising and strong performance.
- The paper is well written and easy to follow. The figures are well-plotted and informative which can make readers quickly understand the core ideas.

### Weaknesses
- My major concern is the construction of VAP-Data, which is entirely synthesized from the existing models or checkpoints (such as Kling, Vidu and the LoRAs from the community). As a result, VAP may overfit to the semantic controls which have already been supported by the off-the-self models. Under this perspective, the only contribution of the paper becomes an architecture design to unify the existing capabilities instead of the generalization to other novel semantic control. It could be insufficient for an ICLR-level contribution.
- While the authors claim VAP can achieve zero-shot generalization in Figure 7, the demonstrated effects (such as “crumble,” “dissolve,” “levitate,” and “melt”) are already supported by the existing models (e.g., Pika, Kling). It is unclear how truly novel these conditions are for the zero-shot evaluation.
- The real-world evaluation is also limited. To verify the generalization of the model, the authors can consider selecting some famous movie/TV shows/meme clips with interesting semantic control as the references. Such effects could serve as stronger out-of-distribution and real-world references, which can more robustly evaluate the generalizability of the model.
- While the authors provide extensive ablation studies in Table 2, the impact of each proposed component seems very marginal. Could the author provide more theoretical analysis on this? Otherwise, it is unconvincing that the architecture of VAP is promising.
- While I appreciate the authors provide thorough limitation analysis in Appendix E, the performance degradation looks significant when the reference and target captions/objects mismatch. It could significantly limit practical usage, since the consistency of reference and target captions/objects cannot always be guaranteed. Could the authors explain the potential reason and solution for this issue? Could re-captioning videos or randomly combining videos with different objects but same effect mitigate this issue?

### Questions
- Could the authors provide more samples using out-of-distribution or real-world videos as the reference semantic control?
- Could the authors explain why the proposed modules only provide marginal improvements over the baseline model?
- Could the authors provide the discussion and potential solution regarding the performance drop when reference and target captions/objects mismatch?
- Since the collection of VAP-Data is claimed as one major contribution, will the dataset be released to the community?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents a unified framework for efficiently compressing large-scale video datasets into compact synthetic ones while preserving both spatial and temporal dynamics. Unlike prior two-stage video distillation methods such as VDSD and IDTD, which are computationally heavy and fail to model motion coherence, the proposed approach introduces a uni-level optimization framework enhanced by a Temporal Saliency-Guided Filter (TSGF). This filter leverages inter-frame differences to guide the distillation process, adaptively constraining optimization to retain key motion cues and suppress redundant frames, and further employs temporally guided augmentation to enhance diversity without breaking temporal continuity. Extensive experiments on benchmarks including MiniUCF, HMDB51, Kinetics-400, and SSv2 demonstrate state-of-the-art performance and efficiency gains, achieving significant improvements under extreme compression while maintaining strong temporal consistency, thus offering a scalable and effective solution for video dataset distillation.

### Strengths
1.	The paper introduces a uni-level video dataset distillation framework that eliminates the complexity of prior bi-level methods such as VDSD and IDTD. This design effectively reduces computational time and memory consumption while maintaining high performance.
2.	A novel TSGF module is proposed to guide the distillation process by computing inter-frame differences, which helps preserve and enhance temporal coherence. This ensures that essential motion information is retained throughout the optimization process.
3.	The authors further design a temporal-aware data augmentation technique that enhances the diversity of synthetic videos without disrupting their temporal consistency.
4.	The framework demonstrates robustness across datasets of different scales and domains, offering a scalable and effective paradigm for video dataset distillation.

### Weaknesses
1.	Some figures (e.g., Fig. 1, Tab. 2) contain typographical errors and could benefit from clearer annotations or more consistent formatting to enhance readability.
2.	The Temporal Saliency-Guided Filter (TSGF) relies on simple inter-frame differences to estimate temporal importance, which may be sensitive to noise, static scenes, or camera motion. A discussion of robustness or potential failure cases would strengthen the work.
3.	The reference format are inconsistencies.
4.	Although the paper claims lower computational cost, it does not provide detailed runtime or memory comparisons under consistent hardware conditions. Quantitative evidence for scalability would make the efficiency claim more convincing.
5.	The claimed efficiency of the uni-level framework may be overstated, as bi-level methods can reuse pre-trained image models or distilled datasets, potentially achieving lower cost and stronger spatial representations.

### Questions
See Weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents Video-As-Prompt (VAP), a unified framework that treats a reference video as a prompt for semantically controlled video generation. It enables multiple controls such as concept, style, motion, and camera within a single model, achieving strong generalization and competitive performance.

### Strengths
1. The paper proposes a unified “video-as-prompt” paradigm that reframes semantic control in video generation as an in-context learning problem, offering a clear conceptual advance over task-specific approaches.
2. The temporally biased RoPE effectively avoids pixel-level copying between reference and target videos, leading to more robust semantic alignment.
3. The Mixture-of-Transformers design enables plug-and-play integration with existing video diffusion transformers while preventing catastrophic forgetting.

### Weaknesses
Major:
1. The paper lacks a theoretical analysis explaining why in-context learning via Mixture-of-Transformers effectively transfers semantic patterns.
2. The proposed temporally biased RoPE is only justified empirically, without an ablation or analytical study on the optimal bias magnitude.
3. The inference cost roughly doubles due to the dual-transformer structure, yet efficiency and scalability trade-offs are not thoroughly studied.

Minor:
1. The semantic diversity in VAP-Data is constrained to four categories, leaving out high-level semantics such as narrative or causal events; but the constructed dataset is the largest to date for semantically controlled video generation and provides a valuable foundation for future research.
2.  The work does not analyze failure cases quantitatively, leaving unclear when and why semantic leakage or identity loss occurs.

Some Typo issues:
1. Line 131: Add a space in "architectures(Singer et al., 2022; …)" to "architectures (Singer et al., 2022; …)".
2. Line 157: Add a space in "Concept(Liu et al., 2025; …)" to "Concept (Liu et al., 2025; …)".
3. Line 160: Add spaces in "Camera Movement(He et al., 2024; …)" and "Motion(Zhao et al., 2024; …)" to "Camera Movement (He et al., 2024; …)" and "Motion (Zhao et al., 2024; …)".
4. Lines 239 and 390: Ensure consistent usage of “fine-tuning” instead of “finetuning”.
5. Line 300: Add a space in "480×720(832)" to "480×720 (832)".
6. Line 321: Add a space in "resource(see Sec. 3.2 …)" to "resource (see Sec. 3.2 …)".
7. Line 1026: Correct the typo "metics" to "metrics".
8. Line 1240: Correct capitalization "LoRa" to "LoRA".

### Questions
Refer to Major part in Weaknesses.

### Soundness
3

### Presentation
2

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
This paper proposes Video-As-Prompt (VAP), a unified framework for semantic-controlled video generation. Instead of pixel-aligned control (e.g., depth, pose), the method uses reference videos as prompts to transfer concepts, styles, motions, and camera behaviors. A Mixture-of-Transformers (MoT) design augments a frozen video DiT with a parallel expert transformer, enabling plug-and-play in-context conditioning while avoiding catastrophic forgetting. A temporally-biased RoPE removes false spatial correspondences between reference and target tokens. The authors also introduce VAP-Data, a 100K-pair synthetic benchmark across 100 semantic categories. Experiments show strong semantic alignment, competitive with commercial models, and notable zero-shot generalization to unseen effects.

### Strengths
+ The paper introduces a clean and unified perspective by treating a reference video as a semantic prompt, avoiding the fragmented control paradigms (e.g., pose/depth-specific pipelines) used in prior work.

+ The MoT structure and temporally-biased RoPE are well-motivated and demonstrated to be effective in preventing forgetting and cross-token interference; ablations support the design choices.

+ Strong qualitative performance across multiple semantic axes (concept, style, motion, camera intent), with notable zero-shot generalization and performance competitive with proprietary systems.

### Weaknesses
- The training data is largely synthetic and template-driven, which may limit generalization to real-world video distributions; robustness to natural, diverse videos is not extensively evaluated.

- The MoT architecture increases compute cost and memory footprint, making the method relatively heavy compared to lightweight or plug-in control modules.

- The method assumes reasonably descriptive captions for reference and target videos; behavior under noisy or under-specified captions remains insufficiently analyzed.

### Questions
- How does the method perform on noisy, compressed, or hand-held reference videos? Any robustness evaluation or failure cases that could be shared?

- The paper mentions instruction-style prompting as potentially beneficial. Have the authors quantified improvements when using such synthetic instructions versus descriptive captions?

- Is the MoT design amenable to parameter-efficient variants (e.g., partial expert layers or shared attention blocks)? If so, did the authors explore such configurations?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the significant challenge of achieving unified, generalizable semantic control in video generation. Current methods often fail by enforcing inappropriate pixel-wise priors from structure-based controls or by relying on non-generalizable, condition-specific finetuning and task-specific architectures.

The authors introduce Video-As-Prompt (VAP), a new paradigm that reframes this problem as in-context generation. VAP leverages a reference video as a direct semantic prompt. Its architecture augments a frozen base Video DiT with a plug-and-play MoT expert. This design prevents catastrophic forgetting, the trainable expert processes the reference prompt while the frozen backbone handles the target generation, with both communicating via full attention at each layer. A key component is a temporally biased RoPE, which breaks the model's default assumption of a pixel-aligned mapping between the prompt and the target video.

To train this model, the authors created and released VAP-Data, the largest dataset for this task, containing over 100,000 paired videos across 100 semantic conditions. As a single unified model, VAP sets a new state-of-the-art for open-source methods, achieves user preference rates comparable to leading commercial models, and demonstrates strong zero-shot generalization to unseen semantic concepts.

### Strengths
Unified and Generalizable Model: VAP is, to the authors' knowledge, the first framework to successfully unify a diverse set of semantic controls into a single model without requiring per-task modules or finetuning.

Strong Zero-Shot Performance: The model shows a strong ability to generalize to semantic conditions that were not included in the VAP-Data, such as "crumble" and "dissolve". This indicates it is learning a generalizable concept of semantic transfer.

VAP-Data: The paper introduces the largest-ever dataset for semantic-controlled video generation. This dataset, and the bootstrapping method used to create it, will be highly valuable for future research.

Ablation Studies: The paper provides a very strong set of ablations that convincingly justify the architectural design. The comparison clearly shows the superiority of the MoT design over finetuning, LoRA, unidirectional cross-attention, and residual addition. The ablation on RoPE design also confirms the authors' hypothesis .

### Weaknesses
he paper is strong and transparently discusses its own limitations.
Synthetic Data Limitations: The primary weakness, is that VAP-Data is entirely synthetic, generated using other models. The paper notes this means VAP may inherit the "stylistic biases, artifacts, and conceptual limitations" (e.g., bad hands) of these source models
Dependence on Caption Quality: Performance relies on well-aligned semantic descriptions in the reference and target captions. The authors show that mislabeled captions or a large mismatch in subject structure can significantly degrade generation quality.
Inference Cost: The plug-and-play MoT expert, while effective, adds significant computational overhead. The paper notes it roughly doubles inference time and increases memory usage.

### Questions
The multi-reference failure case suggests the model struggles to disentangle pure semantics from appearance/layout when the reference videos are structurally diverse, leading to "leakage" of features like "spider legs" and "fish shape". You hypothesize this stems from using generic video captions. Do you believe training with more explicit "instruction-style" captions would be sufficient to solve this, or does this failure point to a need for a more explicit architectural component to enforce disentanglement?
Regarding the synthetic VAP-Data, your zero-shot results are promising. However, one could argue the model is learning a "meta-task" of how to follow VFX-style instructions from the synthetic data, rather than a truly general semantic understanding. How does VAP perform on zero-shot tasks that are semantically distinct from the VFX-style domain?
The temporally biased RoPE is a key innovation. You mention shifting the reference's temporal indices by a fixed offset $\Delta$. How was this offset value chosen, and how sensitive is the model's performance to this hyperparameter? Furthermore, does the MoT expert learn to attend to the entire reference video prompt equally at each step of the target generation, or does it learn a dynamic temporal correspondence?

### Soundness
3

### Presentation
4

### Contribution
3
