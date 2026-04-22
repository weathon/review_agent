# Part-level Semantic-guided Contrastive Learning for Fine-grained Visual Classification

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
Fine-Grained Visual Classification (FGVC) aims to distinguish visually similar subcategories within a broad category, and poses significant challenges due to subtle inter-class differences, large intra-class variations, and data scarcity. Existing methods often struggle to effectively capture both part-level detail and spatial relational features, particularly across rigid and non-rigid object categories. To address these issues, we propose Part-level Semantic-guided Contrastive Learning (PSCL), a novel framework that integrates three key components. (1) The Part Localization Module (PLM) leverages clearCLIP to enable text-controllable region selection, achieving decoupled and semantically guided spatial feature extraction. (2) The Multi-scale Multi-part Branch Progressive Reasoning (MMBPR) module captures discriminative features across multiple parts and scales, while reducing inter-branch redundancy. (3) The Visual-Language Contrastive Learning based on Multi-grained Text Features (VLCL-MG) module introduces intermediate-granularity category concepts to improve feature alignment and inter-class separability. Extensive experiments on five publicly available FGVC datasets demonstrate the superior performance and generalization ability of PSCL, validating the effectiveness of its modular design and the synergy between vision and language. Code is available at: https://github.com/joker-lin9/PSCL

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper focuses on the task of fine-grained visual classification and proposes the part-level semantic-guided contrastive learning approach. It consists of three modules, i.e., part localization module (PLM), multi-scale multi-part branch progressive reasoning (MMBPR), and visual-language contrastive learning based on multi-grained text features (VLCL-MG). Experiments are conducted on five fine-grained image datasets.

### Strengths
- The writing is easy to follow.
- Experiments are conducted on five datasets with three different backbones.

### Weaknesses
- The core idea of PLM for localizing part regions appears similar to [a]. PLM leverages CLIP to select N part masks, but the paper does not clarify how N is chosen or how performance changes with different numbers of parts. 
- The paper claims that s_min can be adjusted based on task requirements; however, this is unclear. What is its influence? And the idea of multi-scale features has been widely used.
- MMBPR works simply by concatenating the class token with multi-scale multi-part features, then processing the result with LayerNorm (LN), multi-head self-attention (MHSA), and an MLP. As shown in Table 6, the gain from MMBPR (line 3 vs. line 2) is marginal, leaving its effectiveness unconvincing.
- Although VLCL-MG leverages label smoothing, focal loss, and auxiliary knowledge from intermediate categories, the observed improvements are marginal, as evidenced by Table 6 (line 5 versus line 3).
- Table 2 indicates that the proposed method yields only marginal gains over SOTA baselines, particularly on CUB, CAR, and AIR. An additional evaluation on iNaturalist is recommended. Besides, this method utilizes extra information, including CLIP, textual information, and intermediate categories, which are not used in SOTA methods.
- Since image resolution has a direct impact on performance, the paper should specify the resolutions used in all compared methods to ensure a fair comparison.
- Figure 1 highlights the spatial structures of non-rigid vs. rigid objects, but the method does not explicitly model them. Incorporating spatial structure could be promising.
- There are issues with grammar and presentation. On line 250, “TWe” appears to be a typo. In addition, Figure 2 may be overly complex and hard to follow.


[a] Part-guided Relational Transformers for Fine-grained Visual Recognition

### Questions
My questions are shown in weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new FGVC framework that leverages text-guided part localization, multi-level class embedding, and multi-grained vision-language contrastive learning. This framework is compatible with different backbones and achieves superior performance over baseline methods across benchmarks.

### Strengths
1. The motivation for overcoming the limitation of the common part-level interaction-based methods, such as CAP, is reasonable and well demonstrated in Figure 1 and supported by Figure 4. 

2. The proposed method is compatible with different pretrained backbones and achieves consistent improvement.

3. The idea that leverages a single VLM to localize parts and conduct contrastive learning is novel. 

4. The code is provided to ensure reproducibility.

### Weaknesses
1. The writing lacks some important details. For example, how to apply a multi-stage module on the single-scale ViT? Does ReSAF only work between f_{1,2,3} and f_4? Does it also work between f_{1,2} and f_3?

2. The explanation of Eq. 5 is confusing. How does this min-max operation perform in detail?

3. There is a typo in L249-250: "TWe".

4. The computation cost needs to be compared with baseline models, such as TransFG. It would be better to add at least one baseline model for different backbones.

5. What is the performance if only the last stage output is leveraged during training? Which feature is used for final categorization during inference?

### Questions
My questions are listed above in the weaknesses part.

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
5

### Summary
This paper proposed Part-level Semantic-guided Contrastive Learning to capture both part-level detail and spatial relational features for FGVC task.
Authors combine vision–language alignment, part reasoning, and progressive confidence modeling into an unified framework.

### Strengths
Consistent SOTA results on 5 datasets and across CNN/Transformer backbones.

The CLIP-based masks yield interpretable part attention for recognition.

### Weaknesses
The biggest problem with this paper is that the design is overly complex, while the improvement is relatively limited.

Is the number of part texts predetermined? How does the method adapt to different tasks, especially in cases where the number of part texts varies across categories within the same classification task?

Can large language models accurately describe the differences between categories? Coarse-grained differences are easy to express, such as head color or beak shape. However, some subtle differences are difficult to localize and describe using language.

### Questions
Please refer to weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes PSCL for fine-grained visual classification, decomposing the problem into three parts: (i) a PLM that uses ClearCLIP text prompts to localize part regions and decouple selection from encoding; (ii) MMBPR for multi-scale, multi-branch progressive reasoning with ReSAF (a reverse-key scale-attention fusion) to reduce cross-scale redundancy; (iii) VLCL-MG that aligns image features with coarse/mid/fine textual concepts. Results are reported on AIR/CAR/CUB/NABirds/DOG with RN50/ViT-B/Swin-B. The masking pipeline is “argmax → morphological refinement,” and a component-level GFLOPs table is provided. The abstract claims an anonymous code link.

### Strengths
Clear problem decomposition. The “where to look / how to represent / how to align” split matches FGVC pain points and keeps the design modular.

Interpretability. The PLM produces intuitive part masks from ClearCLIP similarity maps; visualizations help readers see what the model attends to.

Solid component study. ReSAF has reasonable motivation and ablations against MLP / cross-attention baselines; gains are consistent rather than cherry-picked.

Breadth of evidence. Improvements hold across multiple datasets and backbones, suggesting the method isn’t overly tuned to a single regime.

This separation also makes the ablations easier to interpret.

### Weaknesses
“Reverse” label smoothing in Eq. (17). The formula appears to assign a smaller weight to the true class and larger to others, which is the opposite of standard label smoothing. I suspect this might be a typo; if it’s intentional, please explain the intuition and provide a small ablation.

No end-to-end compute/latency. The paper lists component GFLOPs (and notes branch-count growth), but omits whole-model FLOPs/parameter count/peak VRAM / latency (e.g., batch=1 & batch=8). That makes deployment comparisons hard. If I missed this in the appendix, please point me to the exact page/table.

Reproducibility inconsistency. The abstract links to an anonymous repo, yet the Reproducibility Statement says “code available upon request.” Please unify this to an anonymous public repo with full training/eval scripts, configs, and (ideally) weights. If this is already in the appendix, please point me to the exact page/table.

PLM details are insufficient for reproduction. To replicate the masks: specify ClearCLIP version/checkpoint and input resolution; provide the full part-prompt lists per dataset; describe kernel shape/size/iterations for morphology; state whether similarity logits are normalized or temperature-scaled before argmax.

Mid-level text generation is under-specified. For datasets without native hierarchies, the mid-level labels are LLM-generated. Please release prompts/sampling & screening rules, and include a broader noise-sensitivity study beyond one dataset.

Tuning fairness. Most hyperparameters (except LR) are tuned on AIR+RN50 and transferred elsewhere. Add an “equal-budget tuning” table or a light per-dataset retune to rule out hidden advantages.

Minor writing issues. There are a few typos (e.g., “TWe utilize…”). Figure 2 is dense; consider trimming labels and adding a one-page “Training Recipe.”

### Questions
Is the reverse smoothing in Eq. (17) intentional? If yes, provide intuition (e.g., effect on hard examples/confidence distribution) and a small ablation on AIR & CUB comparing standard vs. reverse smoothing, including interaction with γ.

Provide an end-to-end resource table for RN50 / ViT-B / Swin-B × number of branches N: FLOPs, params, peak VRAM, and latency at B=1 and B=8. Include comparisons to TransFG/CSQA-Net and a strong CLIP-head baseline (e.g., CoOp/CoCoOp/Tip-Adapter) under matched resolution/budget.

PLM reproducibility: exact ClearCLIP release/ckpt, input resolution; complete part-prompt lists; morphology kernel/iterations; whether logits are normalized or temperature-scaled pre-argmax; plus a short sensitivity study.

Mid-level text protocol & robustness: release prompts, sampling/filters, and the final mid-level lists for AIR/CAR/CUB/DOG; quantify performance variance across multiple LLM samples to bound noise.

ReSAF intuition: add a histogram or case study of cross-scale similarities to show how the “reverse key” steers attention away from redundant regions rather than only relying on Table-level gains.

Hyperparameter fairness: provide an equal-budget tuning comparison or justify why transferring AIR-tuned hypers doesn’t bias outcomes.

### Soundness
3

### Presentation
3

### Contribution
3
