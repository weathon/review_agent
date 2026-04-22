# Segment Any Events with Language

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
Scene understanding with free-form language has been widely explored within diverse modalities such as images, point clouds, and LiDAR. However, related studies on event sensors are scarce or narrowly centered on semantic-level understanding. We introduce **SEAL**, the first Semantic-aware Segment Any Events framework that addresses Open-Vocabulary Event Instance Segmentation (OV-EIS). Given the visual prompt, our model presents a unified framework to support both event segmentation and open-vocabulary mask classification at multiple levels of granularity, including instance-level and part-level. To enable thorough evaluation on OV-EIS, we curate four benchmarks that cover *label granularity* from coarse to fine class configurations and *semantic granularity* from instance-level to part-level understanding. Extensive experiments show that our SEAL largely outperforms proposed baselines in terms of performance and inference speed with a parameter-efficient architecture. In the Appendix, we further present a simple variant of our SEAL achieving generic spatiotemporal OV-EIS that does not require any visual prompts from users in the inference. The code will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Authors propose the first text-guided instance segmentation framework for event streams, achieving open-vocabulary segmentation, and introduce four benchmark levels to validate the effectiveness of the proposed model.

### Strengths
1.  Proposes SEAL, the first open-vocabulary instance segmentation framework for event streams, achieving effective instance-level segmentation.
2.  Introduces a decoupled training–inference strategy, featuring a complex training framework and a lightweight inference framework.
3.  Constructs a scarce open-vocabulary segmentation dataset and designs four benchmark levels for training and evaluation.

### Weaknesses
1. Incomplete related work analysis. In the Open-World Events Understanding section (line 126), the authors discuss event-based vision foundation models such as EventBind and EventCLIP, but fail to include analyses of event-based vision MLLMs such as EventGPT and EventVL, which also fall within this domain.
2. Dataset availability concerns. For open-vocabulary segmentation in event streams, the authors propose a complex training framework. However, the scarcity of open-vocabulary segmentation datasets in this domain remains a key limitation. The authors should clearly state whether their dataset will be publicly released.
3. Fairness of experimental analysis. In the main experiments, several RGB-based models are used for comparison. It is unclear whether these models were fine-tuned on the proposed dataset (with event-image representations). If not, the comparison is unfair, and results after fine-tuning should be reported.
4. Unclear experimental presentation. The authors introduce four benchmark levels for open-vocabulary segmentation, which vary significantly in word-level and semantic-level difficulty. In Table 1, the evaluation for each benchmark level should be clearly separated and 5. Dataset quality issues. For semantic-level open-vocabulary segmentation, the authors mainly rely on the DSEC dataset, where all scenes are road environments. These scenes are highly repetitive, and event streams lack color and texture information, leading to low semantic diversity and a risk of overfitting. The authors should clarify how they mitigate or address these issues.

### Questions
See Weaknesses.

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
4

### Summary
This paper introduce a framework to segment events using language. 
It contains Multimodal Hierarchical Semantic Guidance and Multimodal Fusion Network.
And further propose four benchmarks to evaluate the results.

### Strengths
* This paper introduce a "Segment Any Events" framework, which can  generate open-world semantic predictions for event masks.
* This paper  attempt to address OV-EIS that supports free-form of language queries.
* This paper propose four benchmarks for evaluation.

### Weaknesses
* The architecture seems rather complex. Could the authors provide a clearer motivation for the inclusion of these modules? Are all of these modules necessary? Is there a simpler approach that could achieve the same results? Besides, it appears that the method is transferring concepts from open-vocabulary image segmentation techniques, such as OpenSeg, MaskCLIP, MaskCLIP++, and OVSeg to the event modality. Could the authors clarify this adaptation and its justification?

* The paper claims to handle open-vocabulary, but is there a benchmark to demonstrate this capability? For example, can the model generalize to unseen classes, or handle user-defined text queries effectively?

* MHSG uses SAM masks and CLIP features on images as supervision signals for the event modality, but there may have significant modality differences between events and images. How accurately can image masks correspond to event regions in this context?

* The part-level experiments seems conducted solely on the DSEC-Part dataset, which contains very few categories, with a small sample size and severe class imbalance. The model’s performance at a finer granularity (e.g., material, action, state) has not been evaluated. 

* Additionally, the paper does not analyze conflicts or consistency between different granularities, for example, how the model handles the situation when a wheel is simultaneously recognized as both a “car” and a “wheel”.

I would be happy to revise my score if the author addresses these points.

### Questions
Please refer to the weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work introduces a multimodal, hierarchical semantic method  to align event-based segmentation with open-vocabulary language queries. It supports both instance-level and part-level mask generation. The proposed method uses three types of prompt-driven masks generated by SAM for each image (semantic, instance, part). The features are supervised by pooled CLIP embeddings and LLM-generated captions. The multimodal fusion network, with a single backbone, combines language and vision with explicit spatial encoding per mask, addressing previous inefficiencies and semantic conflicts in feature pooling observed in all AR-CDG, Hybrid, and AF-DA baselines.

### Strengths
This work is shown to outperform MaskCLIP and other leading open-vocabulary baselines by 3.4 AP on DSEC11-Ins and 3.2 AP on DDD17-Ins. Inference is 5-18x faster with fewer than 1/5th the parameters. This is a good contribution towards practical application.

Secondly, unlike prior methods, this work’s two-stage mask feature enhancement and spatial encoding overcome the "dead mask" issue, where small-event region masks are mapped to zero vectors; UMAP visualisations show tight semantic separation after adding these modules.

### Weaknesses
* DSEC19-Ins is a highly fine-grained dataset: on it, the improvement over MaskCLIP narrows to just 0.7 AP. This, it seems to me that, even with annotation-free training, suggests that the distilled representations are less robust when class granularity exceeds the capacity of available MHSG cues.
* I think that the main variant of the method benefits from GT-derived visual prompts for mask proposals; although a supplementary "prompt-free" variant exists, the claim of real-world flexibility is less convincing without broader prompt-agnostic validation.

### Questions
* For DSEC-Part, are part-level mask labels created
"by hand" (since event data is too impoverished for mask proposals alone to identify parts)? If so, doesn’t this somewhat undermine the claim of annotation-free scaling?
* Artefacts seem to be mitigated compared to the baselines. Even so, couldn’t reconstructing events to images or mapping event data to traditional vision models still introduce artefacts or domain discrepancy, especially under high-speed or low-event-rate scenarios?

### Soundness
4

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
The paper proposes the SEAL architecture, an open-vocabulary event segmentation architecture based on EventSAM that can segment events on the instance and part level using single nouns or whole sentences as prompts.

The paper adapts EventSAM, an event encoder aligned with the feature space of the Segment Anything Model (SAM). A Multimodal Hierarchical Semantic Guidance (MHSG) module uses SAM to produce 3 levels of masks, which are then used to pool CLIP features and generate captions for each masked region using a pretrained vision-language model (LLaMA). The multilevel captions are used as conditioning to train a feature enhancer network that takes in event features from EventSAM. The enhanced features are pooled with a region-of-interest pooling from the EventSAM masks and are further enhanced using mask tokens from the SAM decoder. The pooled CLIP features are used to test the alignment of the language-enhanced event encodings.

### Strengths
Strong performance against baselines

### Weaknesses
- Limited novelty: the method is essentially a merger of several pretrained foundation models and their data in a clever way.

 - Limited ablations: qualitative ablations on the Spatial Encoding or Mask Feature Enhancer are missing or not well explained; I did not understand Tables 4 and 5.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
