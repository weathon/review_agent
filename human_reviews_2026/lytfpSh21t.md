# FCN-LLM: Empower LLM for Brain Functional Connectivity Network Understanding via Graph-level Multi-task Instruction Tuning

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Large Language Models (LLMs) have achieved remarkable success in language understanding and reasoning, and their multimodal extensions enable comprehension of images, video, and audio. Inspired by this, foundation models for brain functional connectivity networks (FCNs) derived from resting-state fMRI have shown promise in clinical tasks. However, existing methods do not align FCNs with the text modality, limiting the ability of LLMs to directly understand FCNs. To address this, we propose FCN-LLM, a framework that enables LLMs to understand FCNs through graph-level, multi-task instruction tuning. Our approach employs a multi-scale FCN encoder capturing brain-region, functional subnetwork, and whole-brain features, projecting them into the LLM’s semantic space. We design multi-paradigm instruction tasks covering 19 subject-specific attributes across demographics, phenotypes, and psychiatric conditions. A multi-stage learning strategy first aligns FCN embeddings with the LLM and then jointly fine-tunes the entire model to capture high-level semantic information. Experiments on a large-scale, multi-site FCN database show that FCN-LLM achieves strong zero-shot generalization on unseen datasets, outperforming conventional supervised and foundation models. This work introduces a new paradigm for integrating brain functional networks with LLMs, offering a flexible and interpretable framework for neuroscience.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces FCN-LLM, a novel framework that bridges functional connectivity networks (FCNs) and large language models (LLMs) via graph-level, multi-task instruction tuning. The framework aims to enable LLMs to reason about brain networks, handle heterogeneous datasets, and generalize across multiple neuroimaging tasks. The authors design several instruction paradigms and pre-train the model on multiple fMRI datasets for downstream zero-shot and fine-tuning evaluation.

### Strengths
1. Timely and innovative direction. The integration of LLMs with brain network analysis is a compelling and emerging topic that could advance explainability and generalization in neuroimaging AI.

2. Well-designed multi-task objectives. The proposed instruction tuning framework incorporates multiple reasoning paradigms, reflecting diverse types of neuroscientific questions and relationships among brain regions.

3. Large-scale pre-training across multiple datasets. Conducting cross-dataset pre-training is valuable for building scalable foundation models and contributes to the neuroimaging community’s growing interest in large-scale, multi-task learning.

### Weaknesses
1. Unclear notations and model formulation. 
- Some mathematical notations are ambiguous. For example, is f_{roi, i} equivalent to  h^{(0)}_i?
- The floor operator used differs between Eq. (5) and later expressions—please unify notation.
- Clarify the exact structure of the FCN encoder: does it consist only of a two-layer GCN followed by an MLP, or are there additional components?

2. ROI mapping between atlases.
- The paper maps AAL116 ROIs to Yeo’s 7-network parcellation, but this mapping is problematic since AAL includes cerebellar regions absent in Yeo’s atlas. How are these cerebellar ROIs handled or reassigned?
- The Schaefer atlas (100–1000 ROIs) provides a hierarchical 7-/17-network organization and may serve as a more appropriate choice for such mapping. Please justify the use of AAL and clarify the alignment process.

3. Incomplete literature coverage and outdated baselines.
- Several recent studies have explored combining LLMs or multimodal transformers with brain network or neuroimaging analysis [1, 2].
- The supervised baselines used are outdated; newer models such as [3–5] should be included in the comparison or discussed to contextualize the contribution.

4. Questionable zero-shot results.
- The reported zero-shot accuracies for binary disease classification are below 50%, equivalent to random guessing. This undermines the claim that FCN-LLM “understands” FCNs in a meaningful way.
- Consider adding qualitative analyses to demonstrate that the model learns non-trivial correspondences even when quantitative performance is limited.

5. Inconsistent alignment between paradigms and tasks.
- Table 2 lists multiple task types, but their correspondence to the three paradigms (comparative, judgment, descriptive) is unclear.
- The ablation study only considers the comparative paradigm; a full ablation disabling each paradigm individually would more clearly demonstrate their respective contributions.
- The “comparative” and “judgment” paradigms both yield Yes/No outputs—please clarify their conceptual and implementation differences.

6. Pre-training label definition and dataset design. Treating all datasets jointly as a six-class classification problem is questionable. For example, an HC subject from ABIDE is not necessarily healthy with respect to other conditions like ADHD. This may cause noisy supervision and limit generalization.

7. Atlas generalization and robustness. Experiments rely solely on brain networks constructed from the AAL atlas. To demonstrate generalization, it would be valuable to test the model on other atlases. Cross-atlas evaluation is critical for validating that FCN-LLM learns transferable representations rather than atlas-specific patterns.

### Questions
See weaknesses

### Soundness
1

### Presentation
3

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
The paper proposes a novel framework, FCN-LLM, that enables LLMs to understand brain functional connectivity networks (FCNs). The approach employs a multi-scale FCN encoder to capture brain-region, subnetwork, and whole-brain features, which are then aligned with the semantic space of an LLM.

### Strengths
• The proposed FCN-LLM framework is conceptually simple yet effective, providing a clear and interpretable approach for integrating FCN representations into LLMs.

• The authors have collected a large-scale, multi-site FCN dataset for alignment learning, which is valuable for advancing domain-specific representations in neuroimaging.

### Weaknesses
• The evaluation protocol for disease classification lacks rigor. Specifically, the definition of “healthy controls” may not be consistent across different disorders. For example, healthy controls used for ADHD may not serve as valid controls for OCD or schizophrenia. Therefore, FCN-LLM should be evaluated on each dataset separately rather than combining them.

• For the ABIDE binary classification task, the reported zero-shot performance of FCN-LLM is close to random guessing, which makes the result difficult to interpret or claim as meaningful.

### Questions
Could the authors clarify the judgment paradigm tasks in more detail? In particular, the paper mentions a balanced distribution of positive and negative samples — what exactly do “positive” and “negative” refer to here? Are they Yes/No responses in question answering, or do they correspond to disease vs. healthy control labels in downstream tasks? Is it possible that FCN-LLM mainly learns to follow textual instructions rather than truly leveraging FCN features in judgement paradigm tasks? If so, how do the authors disentangle these two effects?

### Soundness
2

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
4

### Summary
This paper proposes FCN-LLM, a novel framework that enables large language models (LLMs) to understand and reason over brain functional connectivity networks (FCNs) derived from resting-state fMRI data. While prior brain foundation models focused on supervised graph-based learning or unimodal pretraining, FCN-LLM introduces a graph-level multimodal instruction tuning strategy to align FCNs with the semantic space of LLMs, thereby allowing LLMs to interpret neural connectivity patterns in a text-based reasoning format. The model architecture combines a multi-scale FCN encoder, which extracts hierarchical features from three complementary levels, a multi-paradigm instruction tuning scheme, and a two-stage training strategy. Empirical evaluations on multi-site rs-fMRI datasets show that FCN-LLM achieves state-of-the-art zero-shot generalization on unseen datasets.

### Strengths
1. The paper proposes to bridge fMRI-based brain FCNs with large language models through graph-level instruction tuning. The FCN-LLM introduces a text-aligned multimodal interface, enabling LLMs to reason about neural connectivity in a semantically grounded way. 

2. The author uses the multi-scale FCN encoder to jointly represent region-level, subnetwork-level, and global-level connectivity patterns. This hierarchical formulation mirrors neuroscientific organization principles and leads to more interpretable and generalizable representations.

3. The experiments on 10 fMRI datasets demonstrate the effectiveness of the proposed framework.

### Weaknesses
1. The framework relies heavily on functional connectivity graphs derived from fMRI, which are known to be noisy, non-stationary, and highly sensitive to preprocessing choices. The paper does not sufficiently address how this inherent variability might affect graph quality and, consequently, the overall performance of FCN-LLM. Without explicit denoising, robustness checks, or uncertainty modeling, it is difficult to determine whether the observed improvements stem from the proposed graph–language alignment or simply from correlations driven by noisy connectivity patterns.

2. The paper introduces three instruction paradigms (predictive, judgment, comparative), but there is no ablation or diagnostic analysis showing how each paradigm affects the final performance or generalization. It remains unclear whether the gains come from the multimodal alignment itself, the diversity of instruction tasks, or simply increased data volume.

3. While the paper positions FCN-LLM as enabling language-model-based reasoning over brain graphs, it is not clear how much of this reasoning actually depends on the LLM, as opposed to the pretrained representations or alignment modules. The LLM appears largely frozen and used as a semantic projection target, which raises the question of whether its inclusion yields genuine multimodal reasoning capability or simply acts as a high-dimensional embedding space.

### Questions
1. Could the authors provide evidence or analysis showing how graph construction noise and preprocessing variability influence the final performance of FCN-LLM?

2. How do the three instruction paradigms—predictive, judgment, and comparative—contribute individually to performance and generalization?

3. Can the authors clarify the functional role of the LLM in the proposed framework? Specifically, to what extent does the LLM perform reasoning over graph-encoded brain representations versus serving as a frozen semantic embedding target?

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
4

### Summary
This paper proposes FCN-LLM, a framework that enables LLMs to interpret brain FCNs via graph-level, multi-task instruction tuning.  The method introduces a multi-scale FCN encoder that hierarchically extracts ROI-, subnetwork-, and whole-brain-level representations and projects them into the LLM’s semantic space through a lightweight MLP adapter.  A diverse set of instruction–answer pairs covering predictive, judgment, and comparative paradigms across 19 demographic, cognitive, and psychiatric attributes are used to align FCN embeddings with text semantics. Through a two-stage learning strategy—Stage 1 alignment (LLM frozen) and Stage 2 joint fine-tuning—the model achieves strong **zero-shot generalization** on unseen datasets compared to baselines and provides interpretable subnetwork-level attention visualizations. Overall, the paper presents a novel attempt to bridge fMRI-derived brain connectivity graphs and language models, suggesting a new direction for multimodal foundation modeling in neuroscience.

### Strengths
- **Novel cross-modal alignment of brain networks and LLMs.**  
  The paper is the first to directly align FCN representations with the language modality, enabling semantic reasoning and text-based interaction with brain network data—an original and conceptually appealing contribution.

- **Hierarchical multi-scale encoder design.**  
  The proposed ROI / subnetwork / global-level architecture effectively captures both fine-grained and global connectivity structures, improving representational richness and offering interpretability aligned with neurobiological hierarchy.

- **Comprehensive instruction tuning across multiple paradigms.**  
  By constructing predictive, judgment, and comparative tasks across 19 human attributes, the model learns generalized FCN–text alignment and demonstrates robust zero-shot generalization beyond supervised and foundation model baselines.

### Weaknesses
1. **Lack of ablation on pretraining stage**  
   Stage 1 relies on time-window–specific FCNs for data augmentation, yet no analysis is provided on how the window length \(L\) or stride \(P\) affect alignment or downstream performance.  Since fMRI datasets differ in temporal resolution (TR), fixing \(L{=}100\) may yield inconsistent temporal coverage and introduce noise.  A sensitivity study on \(L\) and \(P\) would clarify whether the augmentation benefits outweigh potential degradation of representation quality.

2. **Dependence on full LLM fine-tuning and unclear scalability**  
  The strongest results appear only after Stage 2 full fine-tuning of the entire LLM. While scaling to a larger model may improve representational capacity, requiring full fine-tuning would make the method increasingly resource-intensive and less scalable. In fact, performance gains from Qwen 2.5-3B to 7B are limited despite a large increase in compute cost, suggesting an unclear efficiency–performance trade-off. If comparable performance could be achieved through partial tuning or parameter-efficient methods (e.g., LoRA, adapter-tuning), the proposed framework could be far more practical and easily extended to recent large-scale multimodal LLMs.

3. **Lack of qualitative validation for alignment quality**  
  The claim that FCNs are well aligned to the text semantic space is supported only by quantitative zero-shot results. It would be valuable to include qualitative analyses that illustrate whether the model captures genuine semantic correspondence—e.g., do key textual concepts in prompts interact meaningfully with FCN tokens? Without such evidence, it remains uncertain whether the alignment reflects semantic consistency or merely distribution-level matching between modalities. Additional visualizations or interaction analyses would strengthen the claim of effective cross-modal alignment.

### Questions
**Q1)** Was an ablation conducted to analyze the effect of window size \(L\) and stride \(P\) on data quality and downstream performance?  

**Q2)** In Stage 2, have the authors tried parameter-efficient fine-tuning (PEFT) approaches such as LoRA or adapter-tuning? If so, how do these compare to full fine-tuning?  

**Q3)** How were the three paradigms (predictive, judgment, comparative) evaluated on stage 2? Please describe the criteria for correctness and how textual outputs were parsed into labels.  

**Q4)** Do ablation results exist for different token-level inputs (ROI-only, subnetwork-only, global-only, or subnetwork + global)?  

**Q5)** Is there a baseline using the 124 FCN tokens with a standalone classifier (without the LLM) to verify whether the LLM component is essential for generalization?

**Q6)** In Table 3, although the proposed model outperforms previous baselines in zero-shot settings, it is unclear whether these results represent meaningful discrimination. If both ABIDE II and CNP are binary disease-classification tasks, the reported accuracies (≈50%) appear close to or even below random guessing. Could you report the corresponding **AUROC** values for both baselines and FCN-LLM to clarify whether the model demonstrates genuine discriminative ability beyond chance level?

**Q7)** The authors describe Stage 2 as using "high-quality instruction tuning data using original FCNs rather than augmented FCNs." Have the authors tested training Stage 1 directly on original FCNs (without window-based augmentation) to verify whether the augmentation is truly beneficial? It would be helpful to compare the performance of models trained with and without augmented FCNs to justify the necessity of sliding-window augmentation.

### Soundness
3

### Presentation
2

### Contribution
3
