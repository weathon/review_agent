# Urban Socio-Semantic Segmentation with Vision-Language Reasoning

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 6, 4, 2, 2, 6

## Abstract
As hubs of human activity, urban surfaces consist of a wealth of semantic entities. Segmenting these various entities from satellite imagery is crucial for a range of downstream applications. Current advanced segmentation models can reliably segment entities defined by physical attributes (e.g., buildings, water bodies) but still struggle with socially defined categories (e.g., schools, parks). In this work, we achieve socio-semantic segmentation by vision-language model reasoning. To facilitate this, we introduce the Urban Socio-Semantic Segmentation dataset named SocioSeg, a new resource comprising satellite imagery, digital maps, and pixel-level labels of social semantic entities organized in a hierarchical structure. Additionally, we propose a novel vision-language reasoning framework called SocioReasoner that simulates the human process of identifying and annotating social semantic entities via cross-modal recognition and multi-stage reasoning. We employ reinforcement learning to optimize this non-differentiable process and elicit the reasoning capabilities of the vision-language model. Experiments demonstrate our approach's gains over state-of-the-art models and strong zero-shot generalization. The dataset and code are open-sourced under the Apache License 2.0 at https://github.com/AMAP-ML/SocioReasoner.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a new task called "socio-semantic segmentation", identifying socially defined urban areas (like schools, parks, or residential districts) from satellite imagery, which is challenging because their boundaries are based on function, not just visual appearance. 

The authors introduce a new dataset, SocioSeg, which pairs satellite imagery with corresponding digital map layers and pixel-level labels for social entities, organized in a hierarchy of increasing complexity. (contribution) 

The authors propose a novel framework, SocioReasoner, which uses a Vision-Language Model (VLM) to reason like a human annotator. It first localizes a target area using both satellite and map images, then iteratively refines the segmentation mask. Since this process isn't differentiable, they use Reinforcement Learning to train the model end-to-end.

Experiments show that SocioReasoner outperforms existing state-of-the-art models and demonstrates strong zero-shot generalization.

### Strengths
**The task is interesting**:  Identifies and formalizes a significant, underexplored challenge in geospatial analysis and provides a dedicated dataset (SocioSeg) to foster research.

**Innovative Methodology**: The multi-stage, reasoning-based approach (SocioReasoner) effectively mimics human annotation logic. The use of Reinforcement Learning to optimize this non-differentiable pipeline is a clever and practical solution.

**Solid Empirical Results**: The framework not only achieves superior performance on the new benchmark, highlighting its robustness and potential for real-world application. 

**Writing**: The writing is clear and easy to follow (presentation).

### Weaknesses
The current experimental results are not sufficiently rich. More experiments verified on other datasets wil strengthen the contribution. 

The space utilization of figures can be improved. For example, there is too much space in figure 2 and the elements are sparse, which is not professional. 

I will consider to increase the score if the authors: 
- clarify how this work concretely differs from or improves upon prior research.
- verify on broader related benchmarks.

### Questions
N/A

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
The paper introduces a new task: “urban socio-semantic segmentation,” which aims to segment socially defined entities in cities — things like parks, schools, logistics centers, “business office buildings,” “park and greenspace,” etc. These are different from classic land-cover / land-use segmentation targets (roads, buildings, water) because they’re not purely visual; they’re defined by function and human use.

### Strengths
**SocioSeg dataset:**

Multi-level hierarchy (name / class / function) is well thought out. It naturally scales difficulty from “find this exact named facility” to “find anything that is educational.”

**Strong empirical results:**

SocioReasoner > strong baselines (VisionReasoner, RemoteReasoner, Seg-R1, etc.) across all subtasks, ~+4 gIoU absolute vs best baseline on average.
Careful baselines: they retrain baselines on SocioSeg where possible, and explain exceptions (e.g., RSRefSeg can’t take 2-image input, so only satellite is given).

### Weaknesses
- Ground truth and annotation quality/reproducibility of labels

The socio-semantic “truth” is derived from Amap AOI data. The paper says they rasterized AOIs, QA’d and dropped bad samples, etc.
But: we don’t yet see quantitative inter-annotator agreement or error estimation. Are AOI polygons always aligned with functional reality? 

- Geographic diversity

All training data is from Amap within China.
Yes, they test on Google Maps tiles (style/domain shift), but it’s still presumably Chinese geography. We don’t know if the ontology or appearance generalizes to cities in, say, Europe, Africa, or North America where zoning, building morphology, and POI taxonomies differ. 

- Dependence on commercial basemaps / terms of service

The method leans on digital map rasters rendered from Amap (or Google Maps at test time). The paper frames this as “we solve multi-modal fusion by turning everything into an image.”
But practically, this offloads data scarcity and licensing problems onto whoever deploys the model: you still need a high-quality, up-to-date basemap layer with POIs, which in some regions is proprietary, restricted, paywalled, or censored. This could limit “real-world” access.

- Evaluation scope

Metrics are cIoU / gIoU only. It would help to also see precision/recall on instance counts for socio-name tasks (did it find the right named facility and only that facility?) because over-segmentation vs under-segmentation communicates different failure modes (especially important for urban planning use).

Runtime overhead: they admit SocioReasoner inference is ~2.7s/sample, which is slower than e.g. RSRefSeg at 0.16s.
This is fine for planning, but maybe not for city-scale tiling at high throughput.

- Limited OOD (out-of-distribution) evaluation

The paper’s out-of-distribution (OOD) evaluation only tests different map styles (Amap → Google Maps), which mainly reflects visual domain shift, not true reasoning generalization.
It doesn’t test cross-city, cross-year, or cross-socio-semantic generalization — scenarios that would actually demonstrate whether the RL-tuned model exhibits emergent reasoning capabilities.
Without such broader OOD studies, it’s unclear whether the reinforcement learning contributes genuine reasoning emergence or simply improves style robustness.

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This submission presents work at the intersection of (socio-) semantic segmentation and vision-language models for earth observation and remote sensing. It introduces "SocioSeg", a new dataset and "SocioReasoner", a vision-language reasoning framework for that task. To accomplish this task, the authors propose adapting a given vision-language model using reinforcement learning. In their experiments, the authors demonstrate fair performance increases as compared to baselines.

Although I very much like the idea, I have concerns about the contribution of this work with respect to (i) the methodological novelty of the presented approach, (ii) the mismatch and relevance for the ICLR community, (iii) the weak experimental evaluation.

I outline my concerns in more detail below.

### Strengths
- **(S1)**: this paper focuses on an interesting interaction of vision-language/reasoning and remote sensing/earth observation.

- **(S2)**: this paper is easy to read and follow, given the systematic build-up of the paper.

- **(S3)**: I appreciate the author's contribution in curating and annotating the earth observation and remote sensing dataset.

### Weaknesses
- **(W1)**: Methodological novelty of the presented approach: this work combines several components of previously published work to address a new task for a specific domain (being remote sensing and earth observation). However,  I see the introduced dataset being a contribution, which doesn't change the methodological novelty of the presented approach.

- **(W2)**: Mismatch and relevance to the ICLR community: I think this work might be more suitable for publication at a computer vision conference or a remote sensing/earth observation journal, given its very domain-specific domain.

- **(W3)**: Weak Evaluation: I acknowledge the author's evaluation of their approach against other work. Unfortunately, all SOTA approaches mentioned in this submission - except one - are coming from non-published arXiv preprints. Only "RSRefSeg (Mall et al., 2024)", has been published (at ICLR 2024). However, having read the paper by Mall et al, I was not able to find anything about "RSRefSeg". The approach proposed by Mall et al., 2024 is a remote sensing vision-language model referred to as "GRAFT" in the original paper and not "RSRefSeg". This leaves me very confused, since I do not know if this is a typo or if the authors speak about another paper and the citation is wrong.

Independent of this, I would suggest evaluating the proposed approach on another dataset, such as the ones used in Mall et al., 2024. This work evaluated model performance on EuroSAT and BigEarthNet. This way, one would have comparable results of the approach, modulo the proposed dataset being specific (I fully understand that this submission focuses on socio-segmentation)

### Questions
- **(Q1)**: Is the reference to "RSRefSeg (Mall et al., 2024)" the right one, or is this a typo or a wrong citation?
- **(Q2)**: Since the results of VisionReasoner are the second-best ones, could you outline the methodological differences between your approach and VisionReasoner (I was not able to find any of the SOTA models you compared against in the related work section of your submission, that's the reason I am asking)?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces SocioSeg, a new dataset for what the authors call urban socio-semantic segmentation: the task of identifying _socially_ defined entities (i.e. schools, parks, and hospitals) from satellite imagery. The authors differentiate this new taks from traditional segmentation tasks that focus on physical or visually distinct features (e.g., buildings or water bodies), this work addresses the challenge of detecting socially meaningful categories that require contextual and semantic reasoning.

The paper also introduces SocioReasoner, a vision-language reasoning framework that intends to mimic human annotation behavior through cross-modal understanding and multi-stage reasoning. The framework leverages reinforcement learning to handle the non-differentiable aspects of this reasoning process, enabling the model to refine its interpretation of social semantics based on feedback.

Experiments show that SocioReasoner outperforms all compared models and demonstrates robustness when applied to images from another domain. Despite these good results, the paper still has some important flaws (which I will detail later), which make me inclined to reject the paper as it is.

### Strengths
1. Novel dataset. The paper introduces SocioSeg, a new dataset for urban socio-semantic segmentation, focusing on identifying socially defined categories (e.g., schools, parks, hospitals) from satellite imagery.

2. Methodological proposal. The proposed SocioReasoner framework combines vision-language reasoning and reinforcement learning to tackle the proposed semantic segmentation task. The approach is conceptually interesting and aligns with current research trends in multimodal and reasoning-based vision models.

3. Overall well-written and clear. The paper is well-organized and clearly written. All sections are easy to follow.

### Weaknesses
1. **Lack of justification for the new task.** While the paper motivates the challenge of detecting socially meaningful categories, it does not convincingly articulate the concrete real-world relevance or practical need for socio-semantic segmentation. The connection to downstream applications (e.g., urban analytics, policy planning, or social impact studies) could be made more explicit. Moreover, it is not clear how this socio-semantic classes are defined and they seem quite arbitrary. Is there any definition for socio-semantic classes? What is the difference with land use segmentation?

2. **Insufficient motivation for using VLMs.** The use of vision-language models is not well justified. The paper assumes that VLMs are naturally suited for reasoning about social semantics, but provides little evidence to support this claim [1], while literature tends to say the opposite. Current VLMs tend to excel at visual-text association rather than genuine reasoning, and the paper does not clearly demonstrate that SocioReasoner achieves the latter.

[1] Huang, C., Zhu, Y., Zhu, S., Xiao, J., Andrade, M., Chopra, S., & Kira, Z. (2025). Mimicking or Reasoning: Rethinking Multi-Modal In-Context Learning in Vision-Language Models. arXiv preprint arXiv:2506.07936.

3. **Missing essential baselines.** The experiments would be stronger with comparisons to simple supervised segmentation baselines, such as DeepLabv3+, Segformer or U-Net trained directly on SocioSeg. Without these, it is difficult to isolate the benefits of the reasoning framework from other factors like model capacity or data scale. 

4. **Limited empirical analysis.** The evaluation focuses mainly on performance metrics, with limited ablation or qualitative analysis. It would be helpful to see results that explicitly test reasoning ability or analyze the contribution of individual components (e.g., reinforcement learning, multi-stage reasoning, or language inputs). Is it really necessary to integrate a language prompt?

5. **Ambiguous claim of reasoning.** The paper frequently refers to “reasoning,” but the evidence for such capability is indirect. Additional experiments, examples or interpretability analyses would be necessary to substantiate this claim.

6. **Unclear dataset creation and task formulation.** The process of constructing the SocioSeg dataset is not described in sufficient detail. It is unclear how the social semantic labels were obtained, validated, or aligned with the satellite imagery and digital maps. The paper mentions pixel-level labels and hierarchical structures but does not specify whether these were manually annotated, derived from external GIS sources, or generated automatically. Important dataset details are missing, such as the number of images, the number of annotated objects, number of classes, etc.
Similarly, the precise inputs and outputs of the model remain ambiguous. It is not entirely clear whether the model takes only raw satellite imagery, or also uses auxiliary map data or textual metadata during training and inference. This lack of clarity makes it difficult to fully understand what the task setup is, how supervision is provided, and how reproducible the dataset and results would be.

### Questions
- What is the difference between the proposed socio-semantic segmentation task and land use segmentation?
- How do you justify the use of vision-language models (VLMs) for this setup? What motivates their suitability for detecting socially defined entities?
- How were the textual prompts for the VLM chosen or constructed? Were they manually designed, or automatically derived from the dataset labels?
- The paper emphasizes that the proposed framework mimics human reasoning. Could you clarify what aspects of the approach correspond to reasoning, and provide evidence or examples that support this claim?

- It is stated that an advantage of the proposed dataset is that raw geospatial data is unified as a digital map layer. How exactly is this layer obtained, and in what way does it overcome the limitations of existing data sources?
- Is training on the SocioSeg dataset performed in a supervised or self-supervised manner? What are the labels or objectives used during training?
- Could you include some representative examples or visualizations of the final dataset to illustrate the types of entities and annotations it contains?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies urban socio-semantic segmentation from satellite imagery. The authors contribute a benchmark dataset, SocioSeg, integrating satellite imagery with digital maps, and a SocioReasoner framework that uses vision-language models with a multi-stage human-inspired reasoning process optimized via reinforcement learning. Experiments including ablation studies and zero-shot tests show improvements over existing methods.

### Strengths
1. The SocioSeg dataset uses a hierarchical structure from Socio-name to Socio-function and unifies diverse geospatial data into a single map, enabling easier multi-modal reasoning. 

2. The SocioReasoner framework simulates human-like annotation with sequential localization and refinement and integrates vision-language models with SAM under reinforcement learning.

### Weaknesses
1. The main contribution of this paper lies in the introduction of a new dataset, with the primary performance improvements stemming from the dataset and reinforcement learning. Overall, the innovation appears to be limited.

2. The evaluation lacks per-category breakdown and qualitative error analysis so it is unclear which entities are actually improved. 

3. The paper shows quantitative improvements but lacks qualitative failure analysis or discussion of where the model fails.

4. Stronger ablation comparing RL vs. supervised fine-tuning could clarify necessity.

### Questions
Are the comparisons between fine-tuned and zero-shot baselines fair? How do you ensure that these comparisons accurately reflect the model's true performance without inflating the reported gains?

How well does the proposed method generalize to new social semantic categories or to other cities? Does it require substantial manual tuning to adapt to these new scenarios?

### Soundness
3

### Presentation
3

### Contribution
3
