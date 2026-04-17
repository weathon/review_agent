# CoPRS: Learning Positional Prior from Chain-of-Thought for Reasoning Segmentation

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Existing works on reasoning segmentation either connect hidden features from a language model directly to a mask decoder or represent positions in text, which limits interpretability and semantic detail.
To solve this, we present CoPRS, a Multi-modal Chain-of-Thought (MCoT)-based positional perception model that bridges language reasoning to segmentation through a differentiable and interpretable positional prior instantiated as a heatmap.
By making the reasoning process clear via MCoT and expressing it as a dense, differentiable heatmap, this interface enhances interpretability and diagnostic analysis and yields more concentrated evidence on the target.
A learnable concentration token aggregates features of the image and reasoning text to generate this positional prior, which is decoded to precise masks through a lightweight decoder, providing a direct connection between reasoning and segmentation.
Across the RefCOCO series and ReasonSeg, CoPRS matches or surpasses the best reported metrics on each standard split under comparable protocols, with performance at or above the prior state of the art across both validation and test partitions.
Extensive experiments demonstrate a strong positive correlation among the CoT trajectory, the generated heatmap, and the decoded mask, supporting an interpretable alignment between the reasoning output and downstream mask generation.
Collectively, these findings support the utility of this paradigm in bridging reasoning and segmentation and show advantages in concentration driven by reasoning and in more precise mask prediction.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces CoPRS, a multimodal chain-of-thought (MCoT)-based method for reasoning segmentation that bridges language reasoning to pixel-level segmentation via a differentiable positional prior (heatmap). The key contributions include:  

- A novel interface using a learnable "concentration token" to generate positional priors from MCoT reasoning, providing interpretable intermediate heatmaps.  
- A unified training framework combining Group Relative Policy Optimization (GRPO) for language reasoning and segmentation supervision for mask refinement.  
- State-of-the-art performance on RefCOCO series and ReasonSeg benchmarks, with ablation studies demonstrating the correlation between heatmap quality and segmentation accuracy.

### Strengths
*Originality*: The integration of MCoT reasoning with dense positional priors offers a fresh perspective compared to existing latent-feature or text-coordinate paradigms. The use of GRPO for joint language-reasoning segmentation optimization is a creative combination of reinforcement learning and segmentation tasks.  
*Clarity*: The architecture diagram (Figure 2) effectively illustrates the pipeline. The distinction between training modes (GRPO vs. supervised) is well articulated.  
*Significance*: Addresses a critical gap in interpretability for reasoning segmentation systems. The reported 3B/7B model improvements over text-coordinate baselines (e.g., +4.7 cIoU over Seg-R1 on RefCOCOg) demonstrate practical value.

### Weaknesses
*Limited Baseline Comparison*: While compared to latent-reasoning and text-based methods, there is no direct comparison with recent hybrid approaches like PerceptionGPT-R1 or RAS-13B under identical training protocols. This leaves uncertainty about absolute performance claims.  
*Ablation Depth*: The GRPO hyperparameter study (Table 4) only tests sampling numbers {2,4,8} without justifying this range. More analysis is needed on how group size affects exploration-exploitation tradeoffs in segmentation contexts.  
*Methodological Ambiguity*: The heatmap generation process (Eq. 3-4) lacks critical implementation details, e.g., how the MLP maps vision backbone outputs to keys, or why two convolutional layers were chosen for F<sup>fuse</sup>. This hinders reproducibility.  
*Theoretical Limitations*: While empirical correlations are shown, there is no formal analysis of why GRPO’s group-relative advantages are particularly suited for segmentation tasks compared to standard PPO.

### Questions
**Q1**: How does CoPRS handle ambiguous positional priors in crowded scenes (Figure 5)? The failure cases suggest sensitivity to instance density – could multi-scale features or contrastive learning between instances mitigate this?  
**Q2**: For GRPO sampling numbers (Table 4), what computational overhead does G=8 introduce compared to G=2? A latency/accuracy tradeoff analysis would help practitioners.  
**Q3**: The correlation analysis (Figure 3) shows association but not causation between heatmaps and masks. Could you disentangle whether performance gains stem primarily from GRPO-enhanced reasoning or the decoder architecture?  
**Suggestion 1**: Add comparisons with RAS-13B and PerceptionGPT-R1 using comparable model sizes.  
**Suggestion 2**: Include an ablation on the vision backbone (e.g., ViT-H vs. ViT-L) to clarify performance dependencies.  
**Suggestion 3**: Provide pseudocode or extended equations for the heatmap generation process (Section 3.1) to improve reproducibility.  

**Rebuttal Potential**: Addressing the baseline comparison gap (Suggestion 1) and providing theoretical justification for GRPO’s effectiveness in segmentation could significantly strengthen the contribution narrative. Clarifying the heatmap implementation details (Suggestion 3) would enhance methodological rigor.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces CoPRS, a novel Multi-modal Chain-of-Thought (MCoT)-based framework that connects language reasoning to visual segmentation through a differentiable positional prior represented as a heatmap. The model integrates a learnable concentration token within a multimodal LLM to aggregate textual reasoning and visual context, producing a heatmap that serves as an interpretable intermediate between reasoning and segmentation. The approach is trained end-to-end by combining Group Relative Policy Optimization (GRPO) for reasoning and supervised segmentation losses for mask prediction.

Empirically, CoPRS achieves state-of-the-art results on the RefCOCO, RefCOCO+, RefCOCOg, and ReasonSeg benchmarks, demonstrating both superior performance and better interpretability compared to prior latent- or text-based reasoning methods. The paper also presents correlation analyses showing that the quality of the learned heatmap aligns strongly with final segmentation accuracy, highlighting the causal link between reasoning and perception.

### Strengths
- Overall, writing is clear with intuitive figures. 

- The paper introduces an interpretable and differentiable positional prior as an intermediate representation linking language reasoning (via MCoT) to visual segmentation.

- The joint GRPO + supervised objective elegantly integrates reasoning quality and mask precision within a single training loop.

- The heatmap prior provides transparent evidence of where the model is “attending,” supporting qualitative interpretability and quantitative correlation analysis (R > 0.7 between heatmap and mask quality).

- Consistently outperforms both latent reasoning (e.g., LISA, SegLLM) and text-based reasoning methods (e.g., Seg-Zero, Text4Seg) across all RefCOCO variants and ReasonSeg.

- Experimental details are available to reproduce the result. 

- The ablation studies on different setups (training mode, reward function, etc) demonstrate further the advantage of the proposed method.

### Weaknesses
- The core modules (GRPO, SAM decoder, etc) are existing frameworks; the contribution mainly lies in combining them rather than introducing fundamentally new architectures. Further clarifications on the technical novelty would strengthen the paper.

- Results are tied to Qwen2.5-VL and SAM encoders; unclear how robust the method is across different MLLM backbones or smaller models.

- Although some failure cases are available in the appendix, the paper does not discuss the limitations of the proposed method explicitly in the main text. Adding discussions on limitations will enhance the clarity of the paper. 

- GRPO-based multimodal optimization on large MLLMs is computationally heavy; practical feasibility for wider adoption is not discussed.

### Questions
- How sensitive is CoPRS to the design of the concentration token prompt (e.g., `<REF_POS>`)?

- Does the GRPO-based reward generalize to new reasoning templates unseen during training? (Does the RL signal capture general reasoning ability rather than just overfit to specific CoT structures?)

- Can the positional prior mechanism transfer to other grounding tasks (e.g., referring tracking or VQA grounding)? (Is the proposed differentiable positional prior is a general paradigm?)

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
Free-form Reasoning Segmentation requires coupling language reasoning with visual segmentation. Existing work either uses latent reasoning which offers limited interpretability or text-based reasoning which has limited flexibility. This paper introduces CoPRS, which produces concentration tokens through multimodal chain-of-thought, which are then used to generate heatmaps over the images through attention, and finally producing the segment masks. All components of CoPRS are trained end-to-end using two objectives: a GRPO objective over the output tokens and a supervised segmentation objective over the heatmap and the final predictions. Experiments show that CoPRS achieves competitive performance on RefCOCO and ReasonSeg over existing methods.

### Strengths
End-to-end training that allows the model to learn to perform CoT reasoning, generate the query embeddings, and generate the mask simultaneously. 

The approach shows strong results across two benchmarks, comparing against an extensive set of existing methods.

The paper is well-written and easy to follow.

### Weaknesses
The paper aims to balance interpretability and representational fidelity by generating heatmaps over the images as positional priors. However, this heatmap generation process itself is not interpretable. From my understanding, the training objective encourages the heatmap to match the ground truth masking, which could be seen as a one of the latent reasoning methods discussed in the introduction, and suffers the same downsides of interpretability. 

CoPRS relies on a vision backbone to produce the key embeddings that are used to generate the positional prior. This introduces additional computation compared to other methods. Furthermore, the quality of the vision backbone model seems very important, but the paper does not examine the effect of it on the overall performance. 

Table 5 and 6 shows ablation on the model’s training objectives. However, there is no in-depth analysis of these results. How are the weights decided?

### Questions
I don’t quite see how H_prior can always be visualized as in figure 2. Does H_prior always have the fixed dimension size equal to the image? What does each element in K represent? 

How does this differ from latent reasoning method where the latent reasoning is done via an attention mechanism where the attention scores can be extracted (ex. the coder in PixelLM)? Wouldn’t the attention score also be interpreted as heatmaps? 

I don’t think it is mentioned in the paper, what is CoPRS an acronym for?

Minor:

Typo: line 297, 406

The citation for SegLLM is incorrect.

### Soundness
2

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
4

### Summary
The paper proposes CoPRS, a CoMT approach that extracts a query from the MLLM. The query is used to generate a differentiable heatmap with cross-attention and acts as a positional prior to the segmentation decoder. The key contributions are:
- A dense, differentiable heatmap is an interpretable link of the MLLM and mask generation. 
- A training framework that combines GRPO and supervised objective.
- CoPRS achieves good results on RefCOCO series and ReasonSeg.

### Strengths
- The paper addresses a relevant multimodal problem, bridging language reasoning with spatial localization.
- The positional prior offers intermediate interpretability, given the correlation between the quality of the prior and that of the final segmentation mask
- Clear empirical gains on evaluation benchmarks.
- Cleverly combining reinforcement learning with supervised learning

### Weaknesses
- The new architecture adds complexity that must be quantified to justify a reduction in speed by the performance gains.
- Although CoT reasoning is emphasized, the actual CoT outputs and their role are not examined. There is no evidence that linguistic reasoning aligns with the generated heatmap.
- Interpretability was studied only through correlation between the heatmap and the final mask, both in the visual domain. The interpretability role of reasoning is unclear.
- GRPO, although more efficient than PPO, is still expensive on a large backbone. The ablation only compares with/without GRPO, without exploring how reasoning reward affects segmentation quality.

### Questions
- How sensitive are the reported gains to the MLLM backbone? Ablations with a smaller model or from a  different family (e.g. LLaVa) will show how dependent the model is on the specific architecture.
- In the GRPO reward design, the choice “0.7 for mask and 0.3 for CoT” might be reasonable, but it is left unexplained.
- How correct are the CoT reasoning trajectories, and how do they contribute to interpretability?

### Soundness
3

### Presentation
3

### Contribution
3
