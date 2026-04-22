# Granular Computing-driven SAM: From Coarse-to-fine Guidance for Prompt-free Segmentation

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Prompt-free image segmentation aims to generate accurate masks without manual guidance. Typical pre-trained models, notably Segmentation Anything Model (SAM), generate prompts directly at a single granularity level. However, this approach has two limitations: (1) Localizability, lacking mechanisms for autonomous region localization; (2) Scalability, limited fine-grained modeling at high resolution. To address these challenges, we introduce Granular Computing-driven SAM (Grc-SAM), a coarse-to-fine framework motivated by Granular Computing (GrC). First, the coarse stage adaptively extracts high-response regions from features to achieve precise foreground localization and reduce reliance on external prompts. Second, the fine stage applies finer patch partitioning with sparse local swin-style attention to enhance detail modeling and enable high-resolution segmentation. Third, refined masks are encoded as latent prompt embeddings for the SAM decoder, replacing handcrafted prompts with an automated reasoning process. By integrating multi-granularity attention, Grc-SAM bridges granular computing with vision transformers. Extensive experimental results demonstrate Grc-SAM outperforms baseline methods in both accuracy and scalability. It offers a unique granular computational perspective for prompt-free segmentation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
To free the need of manually provided prompts and inefficient SAM-AMG mode when pretrained SAM deal with Semantic Segmentation tasks, the authors designed a coarse-to-fine prompt free Grc-SAM to first find the potential foreground objects then segment and refine them through fine-grained dense prompts.

### Strengths
1.	The coarse-to-fine paradigm is intuitive and computationally efficient.

2.	The proposed attention design further improves inference efficiency.

### Weaknesses
## Major weaknesses:

1.	Unsubstantiated multi-class claim

The paper claims that Grc-SAM handles multi-class semantic segmentation, but in Figure 1 the final mask is produced by the original SAM decoder, which outputs only a binary mask. Across the main paper and appendix, I could not find a single visualization of multi-class segmentation results.

2.	Unclear settings for prompt-required SAM baselines

Many methods in Table 1 require manually provided prompts to segment target objects. However, the paper does not specify what kinds of prompts (points/boxes/text, counts, sampling strategy, etc.) were provided to each baseline. If you evaluate SAM in AMG mode, please state this explicitly in the main paper and detail the AMG configuration.

3.	Foreground/background supervision likely harms class generalization

Training appears to rely on a manually chosen foreground/background split, where the “foreground class” is decided by the authors and other potentially meaningful foreground classes are ignored. Experiments are conducted mainly on easier datasets where images typically contain a single dominant foreground, and the evaluated “foreground class” aligns with those selected during training. This design likely damages Grc-SAM’s class-level generalization. By the way, I suspect the training foreground is simply the largest-area class—if so, this should be clearly stated. Please also train and evaluate Grc-SAM on more challenging benchmarks such as LVIS.

4.	No quantitative validation of the attention-based coarse stage

The paper claims the attention-based coarse stage selects potential foreground regions, but I cannot find either visual comparisons or quantitative metrics supporting this claim. Figure 3(b) seems to be a single illustrative example; where are the systematic comparisons? In that example, mountains are also highlighted—why doesn’t Grc-SAM segment them if they are activated by the coarse stage?

5.	Minor performance gains

Beyond the unclear baseline settings, the reported improvements are quite small and some are even worse than the baselines.

## Minor weakness:

1.	Most of the figures and captions need substantial improvement.

2.	Several passages could be streamlined. For example, the long introduction to SAM-AMG in the Introduction belongs in Related Work. Likewise, Section 3.2 contains mathematical definitions whose contribution to the method is unclear and could be trimmed.

### Questions
1.	Most of the figures and captions need substantial improvement.

2.	Several passages could be streamlined. For example, the long introduction to SAM-AMG in the Introduction belongs in Related Work. Likewise, Section 3.2 contains mathematical definitions whose contribution to the method is unclear and could be trimmed.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a granularity computing based prompt generation framework to improve automation and efficiency in segmentation. It generates mask prompts that guide SAM. The approach follows a coarse-to-fine process: the coarse stage locates target regions quickly, and the fine stage refines details with local attention. By focusing computation on key areas and filtering irrelevant parts, it achieves more accurate and efficient segmentation

### Strengths
1. The topic is interesting and relevant, as it focuses on improving automation and efficiency in segmentation through a new prompt generation approach.
2. The paper is generally well written and organized, making it easy to understand the proposed framework.
3. The comparison tables show that the proposed method performs competitively against baseline models.

### Weaknesses
1. The paper only provides system-level comparisons, without quantitative ablation studies on the proposed components. For instance, since the framework adopts a multi-stage coarse-to-fine design, a natural comparison would be with a single-stage setup. Similarly, it would be helpful to include an analysis of the effect of using or removing local attention. Overall, each proposed module should be supported by quantitative ablations to clearly demonstrate its effectiveness.
2. Figure 2 seems to be the same as in the referenced paper and is based on a vanilla ViT rather than SAM’s ViT, making its relevance unclear. It mainly shows a known property of ViTs instead of illustrating the proposed fusion process, so it feels redundant and not well connected to the main method.
3. Figure 3 does not convincingly demonstrate “different granularity.” The two sample images differ significantly in complexity, making it hard to isolate the effect of granularity. For a fair ablation, the same image should be used while varying only the granularity stage.
4. Since SAM itself supports multi-granularity segmentation with three predefined levels, it would be important to include a direct comparison or discussion of how the proposed method improves upon or differs from this built-in capability.

### Questions
Please refer to the weakness section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents Granular Computing-driven SAM (GrC-SAM), which aims to achieve prompt-free segmentation within the SAM framework. Instead of attaching an external prompt generator (e.g., AMG, AutoPrompt, MaskSAM), the authors integrate an internal mask generator that implicitly produces prompt-like guidance through a coarse-to-fine hierarchical attention mechanism.
The coarse stage fuses multi-layer attention maps from SAM’s encoder to highlight semantically important regions, while the fine stage applies local sparse attention with learnable thresholds to refine boundary details. This design attempts to balance segmentation accuracy and computational efficiency, reducing FLOPs and inference time while maintaining comparable or improved accuracy.

### Strengths
Below are the strength of this paper: 
1. Conceptual novelty: The paper proposes prompt internalization, moving beyond simple auto-prompt generation toward integrating granularity control directly into SAM.
2. Efficiency improvement: The hierarchical coarse-to-fine structure empirically reduces FLOPs by ~44% and latency by ~7.5×, indicating a tangible computational advantage.
3. Clear motivation: The paper’s intuition, focusing computation where semantic importance is high, is sound and aligns with human-like visual attention.

### Weaknesses
1. Comparisons are limited. The evaluation omits stronger prompt-generation baselines (e.g., AoP-SAM [1]) and thus it is difficult to quantify the claimed advantages. Also, it is not so clear whether integrating prompt generation inside SAM leads to inherently superior modeling compared to external prompt generation. The evidence is partially convincing, but the experiments requires comparisons with more recent prompt-free or auto-prompt baselines.
2. Accuracy gain is marginal. Improvements over SAM-AMG are small (often ≤0.5 mIoU) on large-scale datasets, which suggests that the hierarchical design may trade accuracy for efficiency.
3. Unclear ablation scope: Ablations cover layer selection but not threshold parameters (τ, λ, α) or different window sizes, which are crucial to understanding the model’s behavior.
4. Issues with global hyperparameters: The same fusion weights and thresholds are applied globally across all samples, which may not be optimal for datasets with diverse object scales.
5. The paper lacks qualitative examples where coarse masking fails (e.g., images densely packed with objects). 


6. Issues with writing: There are numerous grammar, notation, and stylistic inconsistencies which reduce clarity and readability. Figures and captions are not self-explanatory, often lacking step-by-step explanation of the pipeline. Notation is inconsistent in several places (e.g., $\alpha$ vs. $\alpha_l$, $\tau_c$ vs. $\tau_{coarse}$). The writing quality suffers from multiple grammatical (e.g., Our method can integrate with the future model “to” general task. & Although such approaches can avoid user intervention, “but” they require...).




[1] Chen, Y., Son, M., Hua, C., & Kim, J.-Y. (2025). AoP-SAM: Automation of Prompts for Efficient Segmentation. Proceedings of the AAAI Conference on Artificial Intelligence, 39(2), 2284-2292.

### Questions
1. Are the layer aggregation weights ($\alpha_l$) and thresholds ($\tau_{coarse}, \tau_{fine}$) globally fixed across the dataset, or are they adaptively updated per input sample? If fixed, how do they generalize across images of varying complexity and object density?
2. How does the method behave when most of the image area contains objects (e.g., densely packed scenes)? Would the computational advantage of the coarse-to-fine hierarchy vanish in such cases?
3. Have the authors considered learning the coarse threshold via a lightweight meta-network conditioned on the global feature distribution, rather than as a global scalar?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a coarse-to-fine SAM-based framework for prompt-free segmentation. The method utilizes multi-level attention maps from the image encoder to generate coarse-grained masks and then employs a Swin-style transformer to generate fine-grained masks for the mask encoder.

### Strengths
1. The authors propose a coarse-to-fine SAM-based model to eliminate manual prompt requirements.
2. Multi-level attention maps are leveraged to localize target regions.
3. A Swin-style transformer is employed for fine-grained mask generation.

### Weaknesses
1. SAM and SAM 2 lack semantic label prediction, and therefore cannot properly evaluate mIoU or PA on multi-class datasets such as ADE20K and PASCAL VOC 2012. Similar to SAM Automatic Mask Generation (SAM-AMG), they only segment all possible objects without assigning semantic labels. Consequently, the mIoU and PA results reported for SAM and SAM 2 in Tables 1 and 2 are not accurate.
2. I attempted to identify how the proposed method enables semantic label prediction. If my understanding is correct, the authors initialize a learnable class token at the beginning of the image encoder for class prediction. Is this interpretation correct? If not, could the authors clarify how semantic labels are predicted? I recommend emphasizing this functionality, as it addresses one of the intrinsic limitations of SAM.
3. To the best of my knowledge, the sparse prompt embedding is mandatory, including point, box, or empty prompts. Which type of prompt do the authors use? The accuracy of SAM's outputs is highly dependent on the quality of the prompts. Only relying on mask embeddings may not fully exploit the potential of SAM. Moreover, attention maps often struggle to accurately localize target objects due to noise and the high scores in the boundary. How do the authors address these issues?
4. For ADE20K and PASCAL VOC, the compared methods are quite limited. For instance, Mask2Former and OneFormer can achieve over 56% mIoU on ADE20K, whereas the proposed method only achieves 50.7%.
5. Lack of a quantitative ablation study for the fine-space stage module. This raises a concern regarding whether the fine-space stage module is truly necessary.

### Questions
Please see the Weakness.

### Soundness
2

### Presentation
3

### Contribution
2
