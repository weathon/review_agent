# Beyond Re-Training from Scratch: Exploiting the Pre-Trained Classifier for Long-Tailed Learning

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Fine-tuning for long-tailed learning has garnered significant interest owing to the strong priors in foundation models. A prevailing approach is to explore various long-tailed strategies under the standard fine-tuning paradigm, in which the model is initialized from the pre-trained backbone, while the pre-trained classifier is discarded and replaced with a newly re-trained one. However, we observe that, under tail data scarcity, this newly re-trained classifier suffers from weakened discriminative ability and semantic awareness, exhibiting severe imbalance in class-discriminative channels and mislearning general features for tail classes; in contrast, the pre-trained classifier behaves much closer to the oracle, highlighting its strong potential as an effective guide. Motivated by this, we propose a new fine-tuning paradigm, PTClf (Pre-Trained Classifier helps), which exploits the pre-trained classifier to assist the re-trained one in learning tail classes. Specifically, we align downstream classes to upstream classes via label mapping, and guide the re-trained classifier to learn from the mapped pre-trained classifier through initialization and regularization, thereby transferring knowledge from related upstream classes to the data-scarce tail classes. Extensive experiments show that PTClf delivers remarkable benefits for long-tailed data, especially for tail classes, while also exhibiting strong versatility in low-shot learning and domain generalization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a method for long-tail classification that aims to leverage a pre-trained foundation classification model to improve performance on tail classes. Authors claim that approaches using foundation models only rely on the feature representation, but fail to leverage the classifier. They provide an analysis of long tailed classifiers, showing poor representation power, and show that better results can be achieved by transferring knowledge from a foundation pre-trained classifier to similar target classes, by using the foundation classifier as initialisation. Experiments show that this approach can be used in a plug and play manner and boost performance of pre-existing methods.

### Strengths
The paper is clearly written and thoroughly analyses the problem studied, highlighting the challenges of long tailed learning. While certain conclusions are not surprising (long tail classification fails on tail classes due to lack of data), authors clearly discuss their hypothesis and highlight why classifiers fail. The thoroughness and attention to detail is appreciated in this work. 

Similarly, the experimental evaluation is thorough, providing multiple ablations and evaluating the approach from multiple angles. I particularly appreciated the experiment with out of distribution data, which was an important scenario to test and validate. 

The method is simple, can be used in a plug and play manner, and seems to consistently boost performance.

### Weaknesses
I see two main weaknesses with this work: 

1-	It is quite impractical, as it limits to foundation models pre-trained for classification tasks, learning a classification module. This means that very popular foundation models, known for providing high performance boosts, that do not have a classification head are not usable. This includes models like CLIP, or self supervised models trained at scale. This substantially reduces the applicability of the method as foundation models rarely have a classification head available. 

2-	A second weakness is the main hypothesis of this work, which is that pre-trained classifiers can be used to boost performance of rare classes by transferring related knowledge. While this is a sensible assumption, this is not fully demonstrated by experiments in this paper. Authors highlight that classifier channels trained on rare classes have lower magnitude and non discriminative feature channels. It is possible that performance gains are not due to knowledge transfer, but to a better alignment between feature and classifier at initialisation time, as well as a better separation between classifier channels. If knowledge transfer was the key reason behind the performance gain, one could expect results on OOD data to be worsened by transferring irrelevant knowledge to tail classes. A valuable experiment to verify the hypothesis would be replacing label mapping with a random mapping.  

Lastly, the paper is quite crowded, which means that certain experiments lack relevant detail or clarity. While it is great to see a large set of experiments, it is important to provide sufficient details to allow the reader to get valuable information out of it. It would be better to focus on key experiments in the main paper, and move some (e.g. low shot experiments) in the appendix.

### Questions
-	Overall, this is an interesting paper, aiming to understand how long tail classifiers learn. My main concern is the unpractical requirement to have a pre-trained classifier associated with strong foundation backbones used for the task. 
-	It would be great to verify the knowledge transfer hypothesis to understand more clearly the importance of semantic mapping vs a better distribution and alignment with the pre-trained backbone.

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
3

### Summary
This paper proposes **PTClf**, a fine-tuning paradigm for long-tailed recognition that reuses the *pre-trained classifier* rather than discarding it.  The method aligns downstream classes to the most similar upstream classes using prediction frequency, initializes the new classifier with the corresponding pre-trained weights, and regularizes it to stay close during training.  
The authors claim that this approach improves long-tail performance and even generalizes to OOD and DG (domain generalization) settings.  Empirically, the paper reports consistent gains (up to +10 pp) on CIFAR100-LT, IN-LT, and smaller improvements on SVHN and DTD.

### Strengths
- **Clear and reproducible formulation:** The proposed PTClf framework is simple, easy to implement, and described clearly with sufficient ablation details.  
- **Strong empirical results:** Consistent performance gains across multiple long-tail benchmarks (CIFAR100-LT, IN-LT, iNat-LT) and compatibility with existing rebalancing methods (cRT, LA, PaCo).  
- **Thorough ablations:** The paper includes solid analyses on mapping, initialization, and regularization effects, showing internal consistency within its setup.

### Weaknesses
- **Dependence on pretraining proximity and potential semantic overlap**.
The performance gains seem to depend on how closely the downstream dataset aligns with the pretraining distribution.  
Most of the evaluated datasets (CIFAR100-LT, IN-LT, iNaturalist, FGVC-LT) share visual or semantic characteristics with ImageNet-21K.  
Because PTClf reuses the pre-trained classifier and aligns classes based on prediction similarity, it may inadvertently benefit from overlapping or conceptually related categories. This is not necessarily problematic, but it would be helpful for the authors to clarify whether any precautions were taken to avoid **category leakage** and how much of the gain stems from this overlap versus genuine transferability.


- **OOD and DG settings may not represent real-world long-tailed challenges**
The datasets labeled as OOD or DG (SVHN, DTD, CIFAR10-DG, IN-DG) largely remain within the *natural image* domain and differ mainly by appearance, texture, or augmentation rather than by substantial semantic shift.  As such, the demonstrated generalization might be more modest than the “OOD/DG” terminology implies.  In practical scenarios, long-tailed problems often arise in **naturally OOD settings**—for example, medical imaging, satellite observations, or rare biological species—where data scarcity and domain shift coincide.  Evaluating the method on such data could provide stronger evidence of its real-world robustness.

- **Limited comparison breadth in OOD/DG evaluations**
For OOD and DG experiments, PTClf is compared mainly with *standard fine-tuning*, without including other competitive baselines such as LDAM-DRW, cRT, LWS, LADE, MixStyle, SWAD, or Fishr.  Including these methods could better contextualize PTClf’s advantages and help clarify whether the reported gains are specific to classifier reuse or simply a reflection of a weaker baseline.  This is not a major flaw but an area that could make the empirical evidence more convincing.



- **Improvements appear correlated with similarity and imbalance severity**
Across datasets, PTClf’s gains increase when the downstream task is more similar to ImageNet or when the imbalance ratio is extreme (e.g., D-IR40, C100-IR200). This trend suggests that the method’s strength may lie in **leveraging pretraining priors** when limited data are available rather than learning representations that generalize to unseen domains.  
While such prior reuse is practically valuable, it would be helpful if the authors discussed this trade-off more explicitly.


- **Conceptual overlap with open-vocabulary or zero-shot approaches**
If PTClf primarily serves to inject a classifier prior learned from large-scale pretraining, one might ask how it compares to open-vocabulary methods such as CLIP or SigLIP.  These models achieve a similar goal—leveraging pretrained semantic priors—through language-aligned embeddings and often handle OOD categories more gracefully.  A brief discussion of this connection could clarify PTClf’s niche and its relevance in the evolving landscape of multimodal pretraining.

### Questions
1. How does PTClf perform when pre-training and downstream label spaces are disjoint (e.g., non-overlapping ImageNet splits)?  
2. Can you quantify or visualize the correlation between dataset similarity (to ImageNet) and the observed gains?  
3. Why were stronger baselines (SWAD, MixStyle, LADE) omitted in OOD/DG comparisons?  
4. Could PTClf still offer value when using an open-vocabulary model such as CLIP, or does it become redundant?

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
5

### Summary
This paper presents a comprehensive analysis of classifier behavior when fine-tuning pre-trained visual models on long-tailed downstream tasks. The authors show that the newly re-trained classifier often exhibits weakened discriminative ability and semantic awareness, with severe imbalances in class-discriminative channels and mislearning of general features for tail classes. They propose PTClf (Pre-Trained Classifier helps), a fine-tuning approach that initializes the classifier with pre-trained weights and adds a regularization term, PTReg. The paper includes comprehensive experiments supporting the effectiveness of the proposed method.

### Strengths
* The paper is well written and complete, with no obvious flaws.
* The analysis is solid and the methodology is well motivated. The figures are well designed and informative. The experiments provide sufficient evidence for the proposed method’s effectiveness.

### Weaknesses
* Overall novelty is limited. Leveraging the pre-trained classifier during fine-tuning dates back at least five years (see [1]).
* The analysis is conducted on specific classification pre-trained models and datasets. It is unclear whether the findings generalize to a broader range of pre-trained models, e.g., self-supervised models such as DINO or MAE.
* Results in Table ii lag behind those reported in [1], which used a ResNet-50 backbone five years ago (e.g., Stanford Cars vanilla fine-tuning reaches 87.20%, whereas the highest in Table iii is 59.6%). It is unclear whether evaluation protocols differ in the fine-grained fine-tuning setup. This uncertainty reduces confidence in the overall long-tailed visual fine-tuning benchmarks presented here.
  
[1] Co-Tuning for Transfer Learning. NeurIPS 2020.

### Questions
* The authors claim to focus on PEFT, long-tailed, visual tasks, but the methodology appears general. Can PTClf be extended to other domains, such as LLM fine-tuning?
* Can PTClf be applied to tasks beyond classification? The long-tailed problem is even more challenging in object detection.
* The assumption that the pre-trained model covers more classes usually does not hold (line 256). In your fine-grained benchmarks—e.g., Stanford Dogs, a subset of ImageNet with finer labels—the downstream task is more fine-grained than the pre-trained task. How do you ensure that the label mapping works well in this setup?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose PTClf (Pre-Trained Classifier helps), a novel fine-tuning paradigm that leverages pre-trained classifiers to assist re-trained classifiers in learning tail classes. It aligns downstream classes to upstream classes via label mapping, and guide the re-trained classifier to learn from the mapped pre-trained classifier through initialization and regularization, thereby transferring knowledge from related upstream classes to the data-scarce tail classes.
Experimental results across 8 datasets show consistent and strong improvements, particularly for tail classes.

### Strengths
- The motivation of reusing pre-trained classifiers for long-tailed learning is interesting and inspiring.
- The introduction of CDI and GI metrics for channel-wise classifier analysis reveals specific failure modes of re-trained classifiers (sparse discriminative channels, reliance on general features).
- The method demonstrates consistent improvements across various benchmarks, especially on tail classes, along with extensive ablation studies.
- The paper is well delivered and very easy to follow.

### Weaknesses
- The evaluation focuses heavily on ViT-B/16 with limited architectural diversity.
- The simple frequency-based label mapping may be suboptimal compared to more sophisticated alignment methods (e.g., embedding similarity, semantic matching). The approach also lacks analysis of failure cases or guidance on when the method works versus when it doesn't.

### Questions
- How could the method reuse more general pre-trained zero-shot classifiers like CLIP?
- Can the authors discuss on relation to papers like LIFT+ (Lightweight Fine-Tuning for Long-Tail Learning)?

### Soundness
3

### Presentation
3

### Contribution
3
