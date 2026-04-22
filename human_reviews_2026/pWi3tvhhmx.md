# UGround: Towards Unified Visual Grounding with Unrolled Transformers

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 8, 2

## Abstract
We present UGround, a Unified visual Grounding paradigm that dynamically selects intermediate layers across Unrolled transformers as ''mask as prompt'', diverging from the prevailing pipeline that leverages the fixed last hidden layer as "[SEG] as prompt''. UGround addresses two primary challenges posed by the prevailing paradigm: (1) its reliance on the fixed last hidden layer, which sequentially amplifies cumulative errors arising from layer-by-layer propagation without intermediate correction, and (2) its use of [SEG] as a prompt, which implicitly projects textual embeddings into visual space without explicit spatial cues (e.g., coordinates). Central to UGround is Policy-Prompted Masking, which comprises two key components: Stochastic Skip Connection (SSC) and Mask as Prompt (MasP). SSC is a reinforcement learning policy that, via stochastic sampling, allows each [SEG] token to slide across unrolled transformer layers, enabling dynamic layer selection at which it connects to the vision model (e.g., SAM) in a skip-connection fashion. Given the selected hidden layer, MasP uses the similarity map derived from the [SEG] token and image tokens as a soft logit mask to prompt SAM for mask generation, offering explicit spatial cues through its activation regions. To validate the effectiveness of UGround, we, for the first time, have unified visual grounding within a single framework from an attribute perspective, spanning from traditional refer expression segmentation to newly proposed reasoning segmentation, single-target to multi-target, positive query to false premise (empty target). All codes are provided in the supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes UGround, a “unified” visual grounding framework that (i) dynamically selects an intermediate transformer layer via a learned policy (“Stochastic Skip Connection”, SSC) and (ii) feeds a similarity map (between the &lt;SEG&gt; token and image tokens) as a soft logit mask into SAM (“Mask as Prompt”, MasP). The method claims benefits over the common recipe that uses the last hidden layer’s &lt;SEG&gt; token as prompt to SAM. Strong results are reported on ReasonSeg, RefCOCO(+/g), and gRefCOCO, with an ablation suggesting synergy between the proposed SSC and MasP.

### Strengths
1.  **Good motivation**: The author hypothesizes that the prompt information in the final layer of the model is too high-level, whereas the middle layers may contain more precise, localized cues. This reasoning motivates the exploration of strategies like Mask-as-Prompt and dynamic layer selection. To support this claim, a corresponding analysis is presented in Section 3.2.
2.  **Simple and effective method**: A small layer selector identifies the most optimal intermediate layer for each image-text pair, converting its token-image affinities into a similarity map. This map acts as an additional spatial prompt for the segmenter. The selector is trained with a policy-gradient method, rewarding it for generating prompts that yield accurate masks.
3.  **Strong Performance**: Competitive performance on ReasonSeg, RefCOCO(+/g) and gRefCOCO.
4.  **Adequate Ablation**: Tables 6, 7, 8 show that the proposed modules and design choices are somewhat effective.

### Weaknesses
1.  **Unclear Analysis of Dynamic Layer Selection:**
    1.  In the *Why Dynamic Layer Selection* section, the authors compare the full UGround model (presumably including Mask as Prompt) against soft ground truth masks (Figure 2) to conclude that dynamic layer selection is superior. However, UGround contains components *besides* dynamic layer selection. A cleaner experiment, such as a direct comparison between a baseline vs. the baseline + Dynamic Layer Selection, is needed to isolate and validate this specific component's contribution.
2.  **Lack of Justification for Reinforcement Learning (RL):**
    1.  Why can't dynamic layer selection be trained with a simple cross-entropy loss? Given the fixed number of transformer layers, the ground truth could be assigned to the layer producing the largest overlap between its similarity map and the ground truth mask.
    2.  Why not use soft-gating without an explicit loss on the gates?
    The use of RL seems unnecessarily complex. Justification for this choice over simpler alternatives, ideally supported by comparative experiments, is required.
3. **Missing baselines in Table 4**:
    1. CoReS: Orchestrating the Dance of Reasoning and Segmentation, Bao et al., ECCV 2024.
    2. Mask Grounding for Referring Image Segmentation, Chng et al., CVPR 2024.
4.  **Generalization & Limitations:**
    1.  The results focus on datasets like ReasonSeg, RefCOCO, and gRefCOCO. Models trained on these heavily benchmarked datasets may be susceptible to over-fitting. An analysis of the model's limitations and its performance on "in-the-wild" images is needed to demonstrate broader generalization.
5.  **Computational Costs:**
    1.  UGround is expected to have higher training and inference costs due to its "Mask as Prompts" component. A comparison of training and inference costs (e.g., latency, memory usage) against baselines, using the same LLaVA and SAM variants, should be provided.
6.  **Writing and Formatting Issues:**
    1.  *Figure 2:* The legends are too small and difficult to read. Consider increasing the font size for better legibility.
    2.  *Table 2:* There appears to be a typo. $\mathcal{S}\_{\mathrm{IoU}}$ → $\mathcal{M}\_{\mathrm{IoU}}$.
    3.  *Line 261:* There is a capitalization typo. "To this end, We..." should be "To this end, we...".
    4.  *Line 340:* The term "MC Dropout" is mentioned. Please briefly explain this technique and provide a citation.
    5.  *General:* Acronym capitalization is inconsistent (e.g., "LLava" vs. "LLaVA"). Please ensure all acronyms are capitalized uniformly throughout the manuscript.

### Questions
1.  Is the comparison in Table 2 fair, given that it compares the original SAM with an adapted SAM?
2.  In Table 6, why does "Mask as Prompt + SSC" (row 5) perform worse than "Mask as Prompt" alone (row 2)?
3.  Is "Unrolled Transformers" an appropriate name for this method? The name is potentially misleading for two reasons: 1) It suggests a new architecture, but the transformer model itself appears to be unchanged. 2) The term "unrolling" does not seem to align with the proposed technique.
4.  Does the dynamic layer selection method choose only one layer? If so, why not select multiple layers, which might improve performance?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces UGround, a novel paradigm for unified visual grounding that aims to create a single model capable of handling diverse grounding tasks, including reasoning-based, multi-target, and false-premise queries. The authors identify the limitations of existing methods that rely on the fixed final hidden layer of a transformer. Their proposed solution, Policy-Prompted Masking (PPM), uses a reinforcement learning policy to dynamically select an optimal intermediate transformer layer. The similarity map from this selected layer is then used as an explicit spatial prompt for a vision decoder (SAM), while also being directly supervised. The proposed framework achieves new SOTA results on several challenging benchmarks.

### Strengths
- The core idea of "unrolling" transformers and using a learned policy to dynamically select an intermediate layer is highly novel. This reframes feature extraction from a fixed pipeline to a dynamic, content-aware process. The formulation of layer selection as a reinforcement learning task is a novel and effective approach.
- The paper is supported by strong and comprehensive empirical evidence. The method significantly outperforms recent SOTA methods on diverse and challenging datasets (ReasonSeg, gRefCOCO). The ablation studies are thorough, validating the contribution of each component of the proposed PPM mechanism (dynamic selection, mask as prompt, reward formulation), which substantiates the design choices.
- The paper is very well-written and presented. The motivation is clearly articulated through insightful analysis, and the proposed method is explained with clarity and well-designed figures (Figure 1, Figure 3).
- The work makes a significant contribution by successfully unifying multiple complex visual grounding tasks within a single, coherent framework.
- Honorable mention to the attention given to details (e.g., cleaned and underlined proceedings names in the reference, high-quality figures …) **reflects a high degree of care**.

### Weaknesses
While I think the paper is already great, the following points could benefit from clarification to improve the impact of the paper:

- **Inference Overhead and Ambiguity:** The paper does not sufficiently detail the inference-time procedure and its associated costs. Storing intermediate activations from all layers to feed the policy network may introduce significant memory and computational overhead.
- **Limited Conditioning of the Selection Policy:** The layer-selection policy `π(l | H_t*)` is conditioned only on the hidden states of the `<SEG>` token. The optimal layer for feature extraction could plausibly depend on the visual complexity of the image itself. Conditioning the policy on image token representations as well might lead to a more robust and adaptive selection mechanism.
- **Complexity of RL-based Solution:** While effective, the use of a REINFORCE-based policy introduces significant complexity to the training pipeline compared to end-to-end differentiable alternatives. The paper would be strengthened by a justification or comparison against simpler, non-RL methods for layer selection, such as a learned soft-attention mechanism or a weighted average over layer outputs.

### Questions
1. What is the practical impact on inference-time latency and memory usage compared to the baseline model?
2. The layer selection policy is conditioned solely on the textual `<SEG>` token's representations. Have the authors considered or experimented with also conditioning the policy on visual information (e.g., a pooled representation of image tokens)?
3. The reward `r` is derived from the similarity map's alignment with the ground-truth mask. Have the authors explored alternative reward signals, such as the final segmentation IoU score generated by SAM? Such a reward might more directly optimize for the final task performance.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the task of Visual Grounding of Referring Expressions by proposing a transformer-based architecture. UGround learns similarity maps between image tokens and the <SEG> token, which are then used as prompts for SAM. The proposed method learns a policy to determine which intermediate transformer layer to use, rather than fixing the output of the last hidden layer. This paper achieves competitive results in ReasonSeg, RefCOCO/+/g, and GRES datasets and reports ablation experiments.

### Strengths
This review evaluates the paper's quality based on the following criteria: task relevance, related work, technical novelty, technical correctness, experimental validation, writing and presentation, and reproducibility. Each aspect is discussed and highlighted as a strength or a weakness in the sections below.
-    **Relevance of the task:** Visual Grounding of Referring Expressions is a highly relevant problem for the ICLR community. This paper presents competitive results on benchmark datasets for this task.
-    **Reproducibility:** The source code is included in the supplementary material.

### Weaknesses
-	**Writing and Presentation:** This paper misses clarity and consistency. The names of the proposed submodules are too complex to describe the actual methodological contributions of this work. Moreover, the mathematical nomenclature used in the paper is not consistent. Specifically, to which refer different variants for using similarity as mask (Line 219 – 234) is not clear.
-	**Related Work:** The Related Work section does not adequately contextualize the contributions. Specifically, it remains unclear which limitations of prior works, such as the “alignment-centric models” (Line 131), this method addresses.
-	**Experimental Validation and Technical Novelty:**
    -    This paper claims to propose a unified approach for Visual Grounding of Referring Expressions; however, it does not make clear which methodological modifications from previous works are required to fulfill this proposal.
    -    The empirical results are only marginally better or just competitive than the current state-of-the-art for this task on RefCOCO/+/g.
    -    Moreover, it is not clear how significant modifications from previous methods, like including SAM as the segmentation decoder, are influencing the experimental validation rather than the actual technical contributions of this work.

### Questions
1.	What specific changes or improvements make this method a "unified" approach compared to previous visual grounding methods?
2.	Which limitations of previous Large Multimodal Models for visual grounding are being addressed, and how does this method solve them?
3.	The empirical results are close to existing state-of-the-art methods. Why these results are significant enough to justify the architectural modifications?
4.	How much of the performance improvement comes from using SAM, and how much comes from the new method itself?
5.	Please make clarifications for the mathematical nomenclature.

### Soundness
2

### Presentation
1

### Contribution
1
