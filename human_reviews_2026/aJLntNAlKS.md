# Towards a more Holistic Evaluation of Object-Centric Learning

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Object-centric learning (OCL) methods were developed by taking inspiration from how humans perceive a scene. It is conjectured that they achieve compositional generalisation by decomposing the scene into objects, making the learned models robust to out-of-distribution (OOD) scenes. However, the recent OCL literature, by and large, evaluates the learned models only on the proxy task of object discovery, which gives no information about which object properties are actually encoded in the object-centric latent representation. Moreover, these models are not evaluated for the broader goals behind object-centric methods such as compositional generalisation, OOD performance, counterfactual reasoning, etc. Our work argues that the present evaluation protocols for OCL methods are significantly limited or not scalable. We propose using vision-language models (VLMs) for probing OCL methods on various visual question answering tasks. We are the first to evaluate OCL methods on multiple dimensions, ranging from counterfactual reasoning, OOD generalization and robustness to natural adversarial examples. We also propose a new metric that unifies the evaluation of the ‘what’ and ‘where’ attributes, making the evaluation of OCL methods more holistic compared to existing metrics. Finally, we complement our analysis with a simple multi-feature reconstruction-based OCL method that outperforms the state of the art across several tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a new framework for evaluating object-centric learning (OCL) models beyond traditional object discovery tasks. It argues that existing metrics such as linear probing or segmentation accuracy fail to capture higher-level reasoning abilities like compositionality, out-of-distribution robustness, and counterfactual reasoning. More specifically, the authors highlight that current evaluation methods suffer from Type 1 inconsistencies (when a model predicts object properties correctly but localizes them poorly) and Type 2 inconsistencies (when multiple slots redundantly encode the same object, causing fragmented representations). To address these issues, the authors propose a Vision-Language Model (VLM)-based evaluation protocol, where OCL models act as vision encoders in visual question answering tasks, enabling multi-dimensional assessment. They also introduce a unified metric, Attribution-Aware Grounded Accuracy (AwGA), which jointly measures the “what” (object properties) and “where” (localization) aspects of object representations. Finally, they present mFRESA, a simple multi-feature reconstruction OCL baseline that outperforms existing methods across several holistic evaluation benchmarks.

### Strengths
1.	Addresses a critical gap in OCL evaluation.
The paper tackles the limitations of current evaluation schemes that focus mainly on object discovery and fail to assess higher-level reasoning such as compositionality, counterfactual reasoning, and out-of-distribution (OOD) generalization. It also clearly identifies Type 1 and Type 2 inconsistencies, motivating the need for a more holistic framework.

2.	Integration of VLMs as evaluators.
The authors propose using Vision-Language Models (VLMs) for evaluating OCL by treating OCL models as vision encoders connected to an LLM through a small MLP mapping. This design enables multi-dimensional assessment through visual question-answering tasks.

3.	Unified evaluation metric (AwGA).
The Attribution-Aware Grounded Accuracy (AwGA) metric effectively combines “what” (object properties) and “where” (localization) into a single measure and shows strong correlation with both, helping address the identified Type 1 and Type 2 issues.

4.	Clear and professional presentation.
The paper is clearly written and well-structured, with informative figures and detailed experimental descriptions that enhance readability and reproducibility.

5.	Practical baseline improvement (mFRESA).
The proposed mFRESA model, which uses multiple reconstruction targets, provides a simple yet effective improvement over existing OCL methods and supports the paper’s empirical claims.

### Weaknesses
1.	Evaluation may not isolate OCL representation quality (most critical).
Because the proposed framework embeds OCL encoders within a large Vision-Language Model (VLM), much of the reasoning can be performed by the language model itself. This makes it unclear whether the evaluation truly measures the representational quality of the OCL model or simply the VLM’s doing the heavy lifting. While linear probing is limited in scope, its simplicity ensures that it directly reflects representation quality; in contrast, using a complex multimodal pipeline introduces confounding factors that weaken interpretability.

2.	Limited analysis and validation of the AwGA metric (highly important).
The Attribution-Aware Grounded Accuracy (AwGA) metric depends on a Top-K attribution step that can still include redundant slots and may not fully resolve Type-2 inconsistencies. The paper provides no ablation or sensitivity study on K, leaving questions about how stable or fair the metric is across different settings. 

3.	Heavy dependence on pretrained components.
Results may partly reflect the capabilities of large pretrained models (LLMs and feature encoders) rather than the OCL methods themselves.

4.	Overstated causal claims.
The evaluation benchmarks counterfactual question answering rather than true causal reasoning, so causal interpretability might remain unproven.

5.	High computational cost.
As mentioned in the paper, the proposed framework requires multi-stage fine-tuning of large models, which reduces scalability compared with simpler baselines like linear probes.

### Questions
1.	How do the authors ensure that the VLM-based framework measures OCL representation quality rather than the LLM’s reasoning ability?

2.	How sensitive is the AwGA metric to the choice of Top-K, and could redundancy among slots affect its reliability?

3.	Why is the proposed complex evaluation preferable to simpler linear probes that directly assess representation quality?

### Soundness
2

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
This paper critiques the limited evaluation paradigms in object-centric learning (OCL), arguing that standard metrics like mean IoU or object discovery fail to assess what properties are actually represented in the learned slots. The authors propose a new evaluation framework using vision–language models (VLMs) built via visual instruction tuning (LLaVA-style) where object-centric encoders provide visual tokens to the LLM.  
They introduce a new metric, Attribution-aware Grounded Accuracy (AwGA), which jointly measures “what” (object attributes) and “where” (localisation) by computing grounded accuracy weighted by slot-level attribution maps. They benchmark multiple OCL methods (e.g., DINOSAUR, FT-DINOSAUR, StableLSD, SPOT) on VQA, counterfactual reasoning, OOD generalisation, and compositional reasoning.  
They also propose a new baseline, mFRESA, which adds multiple reconstruction targets (pixels, DINO features, HOG features) and shows improved performance on both VQA and AwGA metrics.

### Strengths
- The paper convincingly highlights a major gap in OCL evaluation: existing metrics are narrow and don’t reflect broader reasoning goals (OOD, compositionality, counterfactual reasoning).
    
- The distinction between Type1 and Type2 inconsistencies (localisation vs. redundancy) is well thought-out and nicely visualised.
    
- Novel metric (AwGA). A sound attempt to unify localisation and representation evaluation, addressing known shortcomings of mIoU and VQA-only accuracy.
    
- Evaluates a wide range of OCL methods across diverse tasks, providing a valuable comparative baseline for the field.
    
- Simple, interpretable baseline (mFRESA). The multi-feature reconstruction approach is practical and demonstrates that better feature-level reconstruction helps representation quality.

### Weaknesses
* From my understanding, the proposed framework evaluates through a language-mediated bottleneck, meaning performance is confounded by how the LLM integrates slots via cross-attention rather than by intrinsic slot quality.  For instance, two encoders with very different slot semantics could yield similar VQA results if the LLM adapts its attention weights effectively.
* Thus, the claim that this is a holistic evaluation is weakened by the absence of direct geometric or retrieval-based analyses (e.g., nearest-neighbour retrieval between slots, clustering consistency, object-level contrastive evaluation).     
* There seems to be an overreliance on LLaVA-style architecture. The improvements might partly come from instruction tuning and alignment quality of the VLM, not necessarily from the OCL model itself. There’s limited analysis of how much the connector or instruction tuning dominates the results.

### Questions
1. How sensitive are the results to the choice of LLM and connector architecture?  
    
2. Could AwGA be applied without ground-truth masks, e.g., via attention supervision or self-generated masks? Otherwise its scalability remains limited.
    
3. How much do you believe instruction-tuning data leakage affect fairness? Many LLaVA datasets contain COCO images, which overlap with OCL training data.
    
4. Would instance retrieval or clustering consistency across views yield the same model rankings as AwGA? This could validate whether AwGA captures true representational quality. The result will not be exact, but one would expect that a good representation captures both what and where of the object in the scene.
    
5. For mFRESA: do the additional reconstruction heads improve binding (object separation) or attribute encoding? A disentanglement or slot purity analysis would clarify this.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes a broader evaluation protocol for object-centric learning (OCL), arguing that standard object discovery metrics (e.g., segmentation quality) do not reflect downstream reasoning and robustness. It evaluates slot-based models by integrating them into a vision-language pipeline and testing them on diverse VQA-style, compositional, OOD, and counterfactual benchmarks. The authors also introduce AwGA, a metric that jointly measures answer correctness and whether the model used the correct object regions. They further present mFRESA, a multi-feature reconstruction variant of an object-centric model, which they claim improves both grounding and downstream robustness compared to prior slot-based methods.

### Strengths
1. The paper is well written, clear, and easy to follow.
2. The VLM-based evaluation pipeline for OC models enables a wide range of evaluations across perception, reasoning, robustness, and compositionality.

### Weaknesses
1. Some of the claims have already appeared in the literature, although under different experimental setups. Please see the questions below for more details.

### Questions
1. As my main concern: several claims and takeaways in the paper seem to have already been shown in prior work, but with a different evaluation pipeline. For example, [1] has already analyzed (1) comparisons between foundation models and OC models in terms of raw VQA performance, and (2) the correlation between unsupervised object discovery and downstream reasoning performance. In addition, [2] analyzes compositional generalization of OC models compared to foundation models in a fully controlled setting, showing advantages for OC models. Given this, it seems that the main contributions of the current paper go down to (1) the VLM-based evaluation pipeline, (2) the introduction of mFRESA, and (3) the AwGA metric. I would appreciate it if the authors could elaborate on this.
2. As an addition, I would like to see an ablation on the loss term weights for mFRESA to see how much each reconstruction target (e.g. pixels, DINOv2 features, HOG features) affects the downstream/upstream performance of the model.
3. To further enhance the results of the paper, I would also recommend showing qualitative effectiveness of mFRESA compared to other methods e.g. showing attribution maps for a few questions or (if possible) attention maps over slots from the LLM, to show which slots are actually being used for question answering.
4. In Section 4.1 (last paragraph), it is mentioned that on the compositional reasoning task of SugarCrepe, OC models lag behind foundation models like DINOv2. On the other hand, on similar (but synthetic) compositional generalization tasks, [2] shows that OC representations generalize better compositionally. Could you please elaborate on the differences between these two papers and why the trends seem reversed?
5. Minor typo: Line 147 defines S as a k-slot vector, but the indices are listed from 0 to k.

Given all the above, I believe this paper is taking an important and necessary step towards better understanding the role of object-centric learning in the current era of foundation models, and I would recommend the acceptance of the paper.

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
3

### Summary
The paper presents a new evaluation framework for object-centric learning using vision-language models. The idea is interesting and addresses a real gap in current OCL evaluation methods. The proposed metric and experiments are convincing. But the paper did no include necessary computation requirement data. Overall, it is a solid and meaningful contribution.

### Strengths
- originality: 3/5,
- quality: 3/5, 
- clarity: 3/5,
- significance: 4/5. Four takeways are very valuable and worth-thinking for all future OCL researches.

### Weaknesses
W1
---
Line 073-076:
> Specifically, we employ the visual instruction tuning method of Liu et al. (2023), which modifies an LLM into a vision-language model (VLM). We use object centric models as the vision encoders, enabling us to evaluate OCL methods through visual question answering (VQA) via the VLM.

The proposed method is valuable and effective. However, the visual intruction tuning required in the authors' evaluation framework poses a very high computation demands, both in space and time. This even makes it impractical for most researchers, who have limited computation resources.

### W1.1

Figure 2:
Although stage 1 visual intruction tuning only requires tuning of "MLP Connector", the inference of LLM is still very expensive. Let along stage 2 needs tuning the LLM.

There fore, it is necessary to provide detailed training-time space and time costs, and the GPU numbers and models.



W2
---
Figure 7.
> Multi-feature reconstruction for slot attention (mFRESA).

The authors claim to propose a novel learning/optimizing method for OCL models. However, the reconstruction of VFM (vision foundation model) feature and the reconstruction of input pixel, except HOG, have already been used in many existing OCL methods. Besides, the two extra replica decoding times the training costs. Thus, the related data should also be provided.



W3
---
Section 2 Related Work:
Some important OCL advances are missing. Discussions should be provided on these OCL baselines and the ones being chosen in the experiment.
- SOLV: Self-supervised Object-Centric Learning for Videos
- VQ-VFM-OCL: Vector-Quantized Vision Foundation Models for Object-Centric Learning
- MetaSlot: MetaSlot: Break Through the Fixed Number of Slots in Object-Centric Learning
- VideoSAUR: Object-Centric Learning for Real-World Videos by Predicting Temporal Feature Similarities
- SlotContrast: Temporally Consistent Object-Centric Learning by Contrasting Slots

### W3.1

I am very interesting in the effect of slot pruning techniques, like SOLV and MetaSlot, on the authors' experiment results.

### W3.2

OCL can be conducted on both images and videos. Thus, it is suggested to also provide some results on video-based vision-lanaguage tasks. The corresponding OCL methods can be SOLV, VideoSAUR or SlotContrast.



W4
---
According to Table 6, only seven slots are used, which is very much less than the dense feature used in original VLMs. So it would make this work more complete if results of larger #slots are used, e.g., 16, 32 and 64.

### Questions
N.A.

### Soundness
3

### Presentation
3

### Contribution
3
