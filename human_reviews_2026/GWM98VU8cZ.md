# Zero in on Faithful Anchors: High-Fidelity Visual Token Condensation for Multimodal Large Language Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Multimodal Large Language Models (MLLMs) have demonstrated impressive visual reasoning capabilities, but their scalability is limited by the computational burden of processing massive visual tokens. To alleviate this bottleneck, many studies have focused on visual token pruning strategies, which utilize cross-attention or [\texttt{CLS}] attention to identify and retain informative visual tokens. In this work, we uncover a critical limitation of such pruning approaches, \textit{i.e.}, they tend to either omit or pay much attention to the background context within images, resulting in potential semantic distortion. To solve this problem, we introduce CondenseVLM, a dynamic token compression framework for HiFi and efficient MLLM inference, that enhances the information density of retained visual tokens. In particular, CondenseVLM employs a three-stage method: it first selects high-attention tokens as faithful anchors to preserve fine-grained semantics, then compensates important background tokens, and finally merges the retained tokens based on spatial proximity and semantic similarity to ensure view integrity. This synergistic optimization of semantic uniqueness, spatial coverage, and contextual integrity makes CondenseVLM capable of high-fidelity compression. Extensive experiments demonstrate that CondenseVLM can prune up to 88.9\% of visual tokens with merely a \underline{3\%} performance drop, and 77.8\% with just a \underline{1.2\%} drop. Moreover, it integrates seamlessly with efficient attention operators during decoding, delivering substantial speedups and memory savings. The code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes CondenseVLM, a new training-free, plug-and-play framework designed to reduce the computational cost of MLLMs. CondenseVLM is composed of three stages: faithful anchor selection, background token selection, and spatial-similarity contextual merging. Extensive experiments on models like LLaVA-1.5 show that CondenseVLM significantly outperforms existing methods.

### Strengths
1. Clear Motivation: The paper provides a very clear and critical analysis of the failure modes of existing token pruning strategies. The observation of 'semantic disruption in token merging' is interesting
2. Synergistic Method: The CondenseVLM framework is well-designed and directly addresses the identified limitations. The three stages are synergistic: similarity suppression addresses redundancy, background selection addresses context loss, and spatial-semantic merging addresses disruption. 
3. Comprehensive Empirical Results: The paper presents extensive experimental validation. CondenseVLM consistently outperforms a wide array of recent baselines across numerous benchmarks. The method's robustness is further demonstrated by its strong performance at extreme pruning ratios (e.g., 88.9%) and its generalization to other architectures (Qwen2.5-VL) and high-resolution inputs (LLaVA-NeXT). 
4. Thorough Ablation Studies: The ablation experiments in Table 5 and Figure 8 effectively validate the authors' design choices.

### Weaknesses
1. To me, it's not clear how the proposed method actually solves the failure modes of existing methods. More systematic analysis should be provided.
2. Novelty concerns: Although the method looks promising, it's more like an incremental combination of existing methods. How their core design is different from previous works should be more clearly elaborated.
3. As the authors state that the "exhaustive search algorithm" for token selection introduces "a noticeable computational overhead", more latency analysis about the method should be emphasized in the experiment section.

### Questions
See weakness.

### Soundness
2

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
This paper introduces CondenseVLM, a training-free framework to reduce the computational cost of visual tokens in MLLMs. The authors compellingly argue that existing pruning methods suffer from selecting redundant tokens, neglecting background context, and causing semantic distortion during merging. CondenseVLM addresses this via a three-stage process: 1) selecting high-attention "faithful anchors" while enforcing diversity via similarity suppression, 2) explicitly selecting "background tokens" to preserve context, and 3) merging remaining tokens using a novel hybrid score based on both spatial proximity and semantic similarity. Extensive experiments show the method can prune 88.9% of tokens on LLaVA-1.5 while retaining 97.0% of the original performance, significantly outperforming recent baselines.

### Strengths
1. The paper clearly identifies and visualizes (Figs. 2, 3) key, non-obvious flaws in prior work, namely the redundancy of high-attention tokens and semantic corruption from similarity-only merging.

2. The proposed three-stage solution is elegant and directly addresses each of the identified problems (similarity suppression for redundancy, background selection for context, and spatial-semantic merging for corruption).

3. The method demonstrates state-of-the-art results, consistently outperforming a wide array of recent methods. This strength is supported by comprehensive experiments showing generalizability (to high-res and other architectures) and thorough ablation studies validating each component

### Weaknesses
1. The computational cost of the CondenseVLM algorithm itself (the iterative selection and merging steps) is not clearly analyzed. 

2. The method admittedly performs poorly on tasks like TextVQA. The ablation study (Table 5) confirms that the spatial-merging component, a core contribution, is detrimental to performance on such tasks. 

3. The method introduces a key hyperparameter $d$ (the ratio of faithful anchors to background tokens), but its value and sensitivity are never discussed, which impacts reproducibility.

4. The variable $d$ is overloaded. It is used to represent the hidden dimension size (e.g., in Eq. 1 and Eq. 9) and also as the allocation ratio for anchor vs. background tokens (Sec. 3).

### Questions
1. What value was used for the anchor-to-background ratio hyperparameter $d$ in the experiments, and could you provide a sensitivity analysis for it?

2. Can you please clarify if the selection algorithm is greedy or exhaustive? What is its precise computational cost (in complexity and measured milliseconds)?

3. Given the method's clear weakness on TextVQA, have you considered a simple mitigation, such as identifying and "protecting" text tokens from the merging process?

4. What is the justification for using max similarity suppression in Eq. 2, as opposed to a more global metric like the average similarity to all previously selected anchors?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the computational and memory overhead in MLLM caused by the large number of visual tokens. The authors identify three key limitations of existing visual token pruning and merging strategies: (1) the selection of redundant, semantically similar tokens by attention-based methods; (2) the failure to preserve important background context due to the spatial clustering of [CLS] attention; and (3) semantic disruption caused by merging tokens based on feature similarity alone.

To address these issues, the paper proposes CondenseVLM for vision token pruning. It has (1) Faithful anchor selection, (2) background token selection and spatial-similarity merging.

The authors conduct experiments on llava and qwen2.5vl models. The results demonstrate that CondenseVLM outperforms previous methods, particularly at high pruning ratios.

### Strengths
1. Motivation and clarity. The motivation is clear. The paper does diagnose and visualize the specific flaws in prior methods, and the proposed components directly map to these identified weaknesses.

2. The overall design is intuitive and sound. The ideas of enforcing diversity in anchor selection in sec 3.1 and using both spatial and semantic information for merging in sec 3.3 are well-justified and directly address the limitations of simpler methods.

3. The experimental results are very strong. CondenseVLM appears to outperform a wide array of recent token pruning methods (DART, HIRED, FasterVLM) across multiple benchmarks and pruning ratios, especially at high compression rates.

4. The ablations in Section 4.4 are effective and clearly validate the contribution of each proposed component.

### Weaknesses
1. While the engineering and combination of the components are effective, the novelty of the core ideas is somewhat limited. The method feels like a very clever and well-executed incremental improvement by combining several known concepts: (1) using [CLS] attention for token importance (like FasterVLM), (2) merging tokens (like ToMe, LLaVA-PruMerge), and (3) using spatial and semantic clues for grouping (which the authors relate to superpixels). The novelty is in the specific three-stage recipe, not in a fundamental new mechanism.

2. Out-of-date baselines and insufficient benchmarks. More specifically:
    - The authors majorly perform experiments on the out-of-date llava 1.5 baseline. For the relatively recent qwen2.5vl model, they only perform experiments on 4 benchmarks, which are significantly not enough, and only compare with 1 baseline, fastv. It is not clear whether the results on only 4 benchmarks are cherry-picked or not, as such experiments usually take about 10 benchmarks to show effectiveness. I would recommend major experiments on qwen2.5 vl on about 10 benchmarks compared with previous methods, just like tab. 1 (with llava1.5).
    - Considering the token pruning process takes no training process nor training data, the authors can consider using other recent baselines like llava-onevision-1.5, kimi-vl, etc.
    - The authors does not carry out experiments on text-dense benchmarks like ocrbench, chartqa, chineseocrbench, etc. These benchmarks are more critical and challenging compared with general mllm benchmarks where vision tokens are often not important.

3. The glossing over of the computational overhead of the CondenseVLM method itself. The authors admit in the Appendix sec F that the selection algorithm has noticeable computational overhead. Tab. 6 also shows this method is measurably slower than other SOTA pruning methods like FasterVLM and HIRED. This implies the efficiency gains come only from reducing tokens for the LLM, and the selection process could be actually a new bottleneck. This accuracy/speed trade-off among pruning methods is not adequately discussed.

4. Heuristic-based design. The method relies on several heuristics that are not fully justified.
    - The Background Token Selection in Sec 3.2 uses the sum of all pairwise similarities as a proxy for "background." This is an interesting but non-obvious metric, and its validity is not deeply analyzed.
    - The merging assignment score in Eq. 6 is a direct subtraction of two differently-natured metrics. The rationale for this specific formulation, its sensitivity to the scaling of the two terms, and the reason for not using a weighted sum are not explained.

5. The claim in the abstract ("prune up to 77.8% with just a 1% drop") is a slight misrepresentation of the main results (which shows a 1.2% drop for the out-of-date llava 1.5 model).

### Questions
Please see the comments above regarding the weaknesses. I have noted how each concern can be discussed and addressed in the rebuttal/revision.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a visual-centric and similarity-based token pruning strategy for MLLMs. The authors identify several issues with recent similarity-based methods: i) they tend to select highly similar tokens that have a high level of information redundancy; ii) they often overlook background regions; and iii) they perform merging without taking into account the coordinate distance, which can lead to mismatches. To address these problems, the authors propose a pruning method that filters out highly similar tokens, proactively includes a small portion of background tokens, and merges tokens while considering their spatial distances.

### Strengths
- This paper is well motivated with reasonable preliminary findings.

- The experimental results demonstrate the effectiveness of the proposed methods.

### Weaknesses
**Lack of Critical Implementation Details:** The description of key technical components is insufficient for reproducibility. The two normalization operations mentioned in Lines 205 and 252 are undefined. In practice, similarity distributions vary significantly across pre-trained models (e.g., CLIP vs. SigLIP vs. DINO), as do the scales of attention scores and spatial distances. It is imperative to clarify:

- Are there any model-specific, pre-defined parameters or thresholds in Equations 2, 4, and 6?

- If so, how were these values determined? A sensitivity analysis for these parameters would greatly strengthen the paper.

- What specific normalization technique is applied to reconcile these different value scales?

**Inherent Limitations of the Similarity Paradigm:** Since this method is based on similarity, it heavily relies on the spatial awareness of the visual encoder. However, MLLMs like LLaVA-v1.5 utilize CLIP as their visual encoder, which is actually limited in its spatial awareness.  

**Empirical Study**: Some conclusions drawn from the visualization results in Figure 3 may be inaccurate. The tokens that show high attention with the [CLS] token could be artifacts [1] rather than representing the 'central focus region.' Moreover, the regions with high similarity visualized in Figure 3(b) are likely the noisy features of CLIP ViT-L, rather than actual background elements like 'sky' or 'snow.'

**reference**

[1] Vision Transformers Need Registers. ICLR 2024.

### Questions
- Please provide more clarification regarding the weaknesses.

- (Not necessary) I wonder whether this method could integrate well with the InternVL series, since there is no significant implementation gap.

Overall, I think the proposed method is both simple and effective. I am open to reconsidering the rating if my concerns are addressed.

### Soundness
2

### Presentation
3

### Contribution
3
