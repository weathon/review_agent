# SVL: Empowering Spiking Neural Networks for Efficient 3D Open-World Understanding

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 2

## Abstract
Spiking Neural Networks (SNNs) offer an energy--efficient route to 3D spatio--temporal perception, yet they lag behind Artificial Neural Networks (ANNs) due to weak pretraining and heavy inference stacks, limiting generalization and multimodal reasoning (e.g., zero--shot 3D classification and open--world QA). We present a universal \textbf{S}pike--based \textbf{V}ision--\textbf{L}anguage pretraining framework (SVL) that equips SNNs with open--world 3D understanding while preserving end--to--end spike efficiency. SVL comprises two core components: (i) {Multi--scale Triple Alignment} (MTA), a label--free triplet contrastive objective aligning 3D, image, and text; and (ii) {Re--parameterizable Vision--Language Integration} (Rep--VLI), which converts offline text embeddings into lightweight weights for text--encoder--free inference. Moreover, we present the first fully spike--driven point Transformer, {Spike-driven PointFormer}, whose 3D spike--driven self--attention (3D-SDSA) reduces interactions to sparse additions, enabling faster, more efficient training. Extensive experiments show that SVL attains strong zero--shot 3D classification (85.4\% top--1) and consistently outperforms prior SNNs on downstream tasks (e.g., +6.1\% 3D cls, +2.1\% DVS actions, +1.1\% detection, +2.1\% segmentation) while enabling open--world 3D question answering, sometimes outperforming ANNs. To the best of our knowledge, SVL represents the first scalable, generalizable, and hardware-friendly paradigm for 3D open-world understanding, effectively bridging the gap between SNNs and ANNs in complex open-world understanding tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors first introduce a new pre-training framework (SVL) to enable spiking neural networks to be effectively trained on 3D data like point-clouds, with the main objective to offer high performance while keeping their inherent advantage of efficient spike-based processing (and hence low energy consumption). In addition, the authors also introduce a Transformer-style Spike-based architecture and evaluate its performance as well as the effect of their pre-training method across a number of 3D tasks.

### Strengths
**Originality & Significance:**  
- The authors do a good job in outlining why the task matters, as well as how it is approached; all in all a clear motivation of the presented research
- Convincing performance improvements through the SVL pretraining, especially for E-3DSNN variants
- Elegant idea of compressing the required text-to-embedding encoder capabilities into small (and cheap) weight matrix for efficient inference

 **Quality:**  
- Their work is placed well within related efforts, and the specific gap the authors tackle is clearly presented and justified
- Appropriate extent of ablation studies to illustrate value/contribution of core components
- The authors include a helpful analysis paragraph regarding time steps and firing bits, and provide recommendations for applying their findings in practice (which is often neglected and therefore much appreciated!)

**Clarity:** 
- The work is mostly easy to follow, and Figure 1 provides a good illustration of core concept, and makes the components of the training process easy to grasp

### Weaknesses
- Proposed Transformer architecture still seems very expensive, both in parameter count and energy. Note this is true when compared to the conv-based E-3DSNN variants, but also even when compared to some of the ANN methods; especially when we assume one could replace the text-encoder in some of the ANN variants as well and therefore significantly reduce the energy consumption of their ‘text’ part! 
- Somewhat lacking explanation of details for architecture, see questions
- Notation consistency should be improved, e.g.
  - Bold vs non-bold for variable (e.g. y in eq 5, b in eq 6, ..)
  - Variable typo in eq 6: upper limit of sum in denominator of the second term should be C (not B), I think?
- Writing quality could benefit from some further refinement, e.g. 
  - Commas at the end of equations, although a new sentence starts in the next line (l.128, l.203, l.280, l.297, l.300, l.308, l.310, ... etc.)
  - Grammar/word-order (e.g. l.169: comma after ‘constant’, represents -> represent;  l.195: at t time step -> at time step t; .. etc.)
  - Non-bold paragraph heading: I’d recommend making l.305/306 ‘Spake-driven self-attention inside SDF’ to be bold given it’s the ‘heading’ for this paragraph; also typo: SDf -> SDF 
- Lack in clarity of some parts, see questions.

### Questions
- The authors report the results for E-3DSNN variants pretrained with SVL in Table 1; Is there a possibility to also show results for these architectures without pretraining? Or do they require pretraining to solve the task?   
  $\rightarrow$ Results w/o pre-training are available for other tasks, e.g. Tables 3,4
- Which operations/components in the authors’ Spike-driven Transformer architecture make it so expensive? I’d be curious to hear some insights/background on the underlying difficulties, and what could potentially be done in the future to further improve efficiency.
- If I see this correctly, the NCE losses mainly aligns vector directions (given the pairwise norm. similarities), while the MSE loss operates on the raw vectors and emphasises direct matches (direction magnitude);    
  $\rightarrow$ Table 6 currently only shows all losses + MSE, but I would be quite interested to see how well only the MSE loss does on the Images, but if possible also on the text to get a better impression of the interacting influences and embedding relationships
- Also: When is a normalisation by ‘T’ required/used (e.g. eq 5), and when not (e.g. eq 7)? (spike train vs spike firing rate)
- Table 4: Why are there no results for your own Transformer architecture reported?
- Sec 4.4 l.298: *‘A learnable add-only pointwise embedding …’*: The equation that follows should X being passed through and MLP; what exactly is the ‘add-only pointwise embedding here?  
  $\rightarrow$ If X is mapped via an MLP, this isn’t really a learnt embedding but rather a projection of the point, i.e. input dependent – whereas learnt embeddings are usually input-independent.   
  $\rightarrow$ Also, why is an MLP used here? (non-linearity required/beneficial to extract features?)
- Sec 4.4 l.309: some more detail here would be helpful; The authors describe the three linear maps resulting in Q,K,V and then go on to show that these are processed by SDSA; However, is this the output of the SDF layer? Or is there, like in Transformers, another MLP or similar post-processing happening? 
Some refinement here would improve clarity. 
- I’d also like the authors to comment on the ‘value’ and main use cases for their architecture, especially given that their results show that the conv-based E-3DSNN variants perform on par but required less energy; Are there any limits and/or applications you see where the preference for E-3DSSN would change?

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
3

### Summary
This paper proposes SVL, a pretraining framework designed to bring Spiking Neural Networks (SNNs) into the domain of 3D open-world understanding, a field currently dominated by Artificial Neural Networks. The authors' goal is to leverage the famed energy efficiency of SNNs while equipping them with complex reasoning abilities, such as zero-shot classification and open-world question answering. The framework introduces several new components, including a Multi-scale Triple Alignment (MTA) strategy, a Re-parameterizable Vision-Language Integration (Rep-VLI) module to discard the text encoder at inference, and a new SNN backbone called the Spike-driven PointFormer

### Strengths
1. **Novel Research Direction:** The paper tackles an interesting and challenging problem by attempting to bridge the gap between the energy efficiency of SNNs and the high-level multimodal reasoning capabilities of ANNs in the 3D domain.
2. **SNN-Specific Improvements:** The SVL pretraining framework appears to be effective *within the SNN domain*. As shown in Table 3 and Table 4, applying SVL consistently improves the performance of SNN backbones like Spike PointNet and E-3DSNN on various downstream tasks (e.g., +6.1% on ScanObjectNN, +2.1% on DVS Action).
3. **New SNN Architecture:** The paper introduces the Spike-driven PointFormer, a new, fully spike-driven Transformer for point clouds, which seems to be a solid contribution to the SNN architecture landscape.

### Weaknesses
1. **Unclear Efficiency Claims:** The primary motivation for using SNNs is energy efficiency. However, the paper only provides "estimated" energy consumption in Table 1 and does not include a direct comparison of computational costs (e.g., FLOPs or real-world inference latency) against the ANN baselines. The ablation study in Table 5 also suggests a trade-off between performance and efficiency (e.g., time steps), which is not fully explored against ANN counterparts.
2. **Unfair Performance Scaling Comparisons:** The zero-shot classification results in Table 1 are difficult to interpret and seem to involve an unfair comparison. The base SNN models (e.g., E-3DSNN-T) show very poor performance. The results only become competitive after scaling the SNN model to a very large size (E-3DSNN-H). However, the paper fails to report the model sizes (parameter counts) for the ANN baselines (like ULIP, OpenShape, etc.), making it impossible to know if this is a fair comparison of similarly-sized models. It is highly likely that a large SNN is being compared to smaller, more efficient ANNs.
3. **Confusing and Incomplete Comparisons in Captioning:** The 3D captioning results in Table 2 also raise concerns. The SNN-based vision encoder (SVL-13B) performs noticeably worse than the ANN-based PointBert when using the same LLM (Vicuna). While the performance is boosted by using SpikeLLM, this introduces another variable, and it's not clear how an ANN-based model would perform with this. Furthermore, the baseline SVL model is also tested with short caption prompts, but the SpikeLLM-enhanced version is not, making the comparison confusing.
4. **Limited Scale of Pretraining Data:** The paper claims to achieve open-world understanding. However, the pretraining is conducted on Objaverse-LVIS, which contains only ~47K objects. This is an extremely small dataset for vision-language pretraining, especially compared to the web-scale datasets often used for ANNs. This limited scale casts doubt on the model's true generalizability and its open-world capabilities.

### Questions
1. Could the authors provide the parameter counts for the ANN baselines (Point-Bert, OpenShape, ULIP, ULIP-2) in Table 1, so a fair comparison can be made against the SNN models (E-3DSNN-L, E-3DSNN-H)?
2. In Table 2, why does the SNN-based vision encoder (Spike-driven PointFormer-L) underperform the ANN-based PointBert when both use the same Vicuna LLM?
3. Given that the pretraining dataset (Objaverse-LVIS) is very small, how can the authors be confident in the model's "open-world" generalization? Have they tested how this framework performs if trained on larger-scale 3D datasets?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
**This paper makes three contributions:**
1. A spiking neural net (SNN) archtecture: "spike-driven transformer" that handles 3D inputs
2. Multiscale Triple Alignment (MTA): A pretraining strategy for spiking NNs with image / text / 3D modalities, that aligns each modality with the spike trains, using InfoNCE (with an additional MSE loss for image-spike).
3. Rep-VLI: core innovation is to pre-compute and embed textual information directly into the weights of a lightweight classification layer, completely discarding the text encoder during inference.

**The authors show comparisons to baselines on many tasks:**
zero-shot 3D classification (ModelNet40, Objaverse-LVIS), supervised 3D classification (ModelNet40, ScanObjectNN), 3D detection and segmentation (KITTI), and video action recognition (DVS), and captioning (Objaverse).

### Strengths
Compared to the listed SNN approaches, the propose recipe outperforms existing SNN recipes by a percent or two on some 3D classification / segmentation / detection benchmarks.

Figure 1 is very clear, and the benchmarks are appropriate for this task. 

The energy analysis seemed compelling, though I am not an expert in SNNs or neuromorphic hardware.

### Weaknesses
The current version paper feels a little bit like a collection of approaches thrown together, that yields gives a couple percent on various 3D classification tasks, compared to existing SNN recipes. My main concerns are about the rigor of the experimental results and novelty. 

Possibly these concerns below could be addressed in a rebuttal or updated version of the initial manuscript, but my feeling at this point is that it may be stronger as a resubmission.

---
### Novelty
The paper claims three novel contributions:

**Multiscale Triple Alignment: (MTA):**
InfoNCE losses are a great tool for non-generative multimodal alignment (as shown in CLIP). InfoNCE losses may not have been used for Spiking Neural Networks before, but this seems like a pretty direct application to me -- makes sense.

**Rep-VLI:**
They authors state that "Rep-VLI’s core innovation is to pre-compute and embed textual information directly into the weights of a lightweight classification layer, completely discarding the text encoder during inference". I've seen this optimization used in several published works -- I'd personally consider it a standard trick for closed-vocabulary inference. E.g. https://github.com/OpenGVLab/PonderV2/blob/aad65d6954633d82141de15d1eb5fa9a23964ee6/ponder/models/ponder/ponder_indoor_base.py#L85

**Architecture:**
For architecture contributions I usually look mainly at empirical results to determine the strength of the contribution. 


---
### Experiments: 
**Architecture analysis:**
When a paper introduces a new architecture as a main contribution, it's very hard to know whether the findings at one scale of data/compute will apply elsewhere. I would like to see some type of scaling analysis that varies both the architectures size and the amount of data. This type of anlaysis is doubly important when introducing new attention operations, like in this paper, since attention is the main driver of transformers' strong scaling performance.

I didn't see any analysis like that in the submission. For reference -- DiT does a very nice job of scaling experiments: https://arxiv.org/pdf/2212.09748, I would an accept an architecture based on scaling experiments like that alone.
Llighter-weight analysis like PointTransformerV3 can also be good https://arxiv.org/pdf/2312.10035


**Baselines:**
The conclusion of the abstract states that the proposed approach "bridges the gap between SNNs and ANNs in complex open-world understanding tasks.". Each experiential table has separate sections that explicitly make the comparison between SNN approaches and ANN approaches. However, for the main results, the ANN results I'm familiar with are much stronger than what I see in the table.

Possibly these could be improved enought
For 3D object classification, PointMamba (2024, >250 references) shows numbers on ModelNet40 and ScanObjectNN which are stronger than both the selected ANN baselines and the proposed method. 
PointMamba: https://github.com/LMD0311/PointMamba

For video action recognition, the only comparison is to PointNet (2017).

Compared to the shown spiking NN baselines, the results are stronger, but not as strong as current ANN baselines. 
With the pace of papers coming out, no one could be reasonably expected to follow all recent literature -- but updating the ANN section with more relevant/recent baselines seems reasonable.

### Questions
Table 1: IIRC, ScanObject3D has several subsets. I may have missed where the authors indicated which subset they are evaluating on?
Table 3: should Spike Point TransFormer be in the "SNN" section instead of "ANN"?

### Soundness
1

### Presentation
2

### Contribution
1
