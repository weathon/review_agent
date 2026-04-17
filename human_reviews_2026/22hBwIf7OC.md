# Plug-and-Play Compositionality for Boosting Continual Learning with Foundation Models

- Decision: Accept (Oral)
- Scores: 4, 6, 6

## Abstract
Vision learners often struggle with catastrophic forgetting due to their reliance on class recognition by comparison, rather than understanding classes as compositions of representative concepts. 
This limitation is prevalent even in state-of-the-art continual learners with foundation models and worsens when current tasks contain few classes. 
Inspired by the recent success of concept-level understanding in mitigating forgetting, we design a universal framework CompSLOT to guide concept learning across diverse continual learners. 
Leveraging the progress of object-centric learning in parsing semantically meaningful slots from images, we tackle the challenge of learning slot extraction from ImageNet-pretrained vision transformers by analyzing meaningful concept properties. 
We further introduce a primitive selection and aggregation mechanism to harness concept-level image understanding. 
Additionally, we propose a method-agnostic self-supervision approach to distill sample-wise concept-based similarity information into the classifier, reducing reliance on incorrect or partial concepts for classification. 
Experiments show CompSLOT significantly enhances various continual learners and provides a universal concept-level module for the community.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a new continual learning algorithm for foundation models called CompSLOT. CompSLOT utilizes an existing object-centric plug in for concept extraction, and incorporates concept learning through decomposition and selection mechanism. Experiment results show that CompSLOT boosts several continual learning methods' performance on standard metrics and benchmarks.

### Strengths
1. The proposed method bridges object-centric learning, concept learning and continual learning, which is a novel intersection.  
2. Extensive experiments have been conducted to confirm that CompSLOT improves continual learning performance on standard evaluation metrics and benchmarks.
3. The proposed method can be easily combined with other continual learning methods.
4. The paper is well-written and easy to follow.

### Weaknesses
1. **The models for experiments are not specified.** The authors do not mention of a specific architecture in the main paper. Does all experiments in Table 1 conducted with the same model architecture? Do CLG-CBM and CompSLOT use the same pretrained model? Without these details, it is difficult to confirm the effectiveness and the versatility of the proposed method.
2. **Lack of concept analysis.** The authors claim that CompSLOT learns human-interpretable concepts for continual learning. However, no experiments are presented to analyze these concepts or evaluate interpretability of the learned representations. 
3. **Hyperparameter sensitivity.** CompSLOT requires hyperparameter tuning ($\alpha, \beta, \tau_t, \tau_p, \tau_a$), which is data-dependent. The sensitivity to these hyperparameters is not discussed in the paper.

### Questions
1. Does Slot Attention suitable for any ViT, or other vision models?
2. I was wondering if CompSLOT benefits the naive continual learners: finetuning? Or does CompSLOT need to be combined with other continual learning algorithms? It will be an informative baseline.

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
The proposed CompSLOT (Compositional Slot plug-in) is a novel method designed to address the challenges of continual learning on compositional benchmarks. It enhances vision models by extracting disentangled, class-relevant concepts directly from images using Slot Attention mechanism. The core of CompSLOT lies in its robust concept learning phase, which uses the primitive selection and aggregation mechanism to identify essential class-relevant concepts. Experimental results consistently demonstrate that CompSLOT significantly boosts the performance of various state-of-the-art continual learners across challenging compositional benchmarks.

### Strengths
- CompSLOT can significantly mitigate catastrophic forgetting in continual learning.

- CompSLOT features a highly flexible and method-agnostic plug-and-play design.

- Extensive experiments on challenging compositional datasets robustly validate CompSLOT's superior performance in continual learning.

### Weaknesses
- It is unclear how a fair comparison was achieved, as the compared methods were not used in the benchmark paper. It is also not understood how these Class-Incremental Learning methods were applied in this setting.

- Essentially, CompSLOT uses an external model as a teacher to reduce catastrophic forgetting.

- It is unclear how this plugin works with baseline methods. For instance, the plugin generates a set of logits. However, the contribution of some methods, such as CPrompt, is to constrain the logits at different stages during incremental learning. Would the proposed method conflict with these baselines?

- The method is too complex to be reproduced easily.

### Questions
see above.

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work addresses the problem of catastrophic forgetting in continual learning (CL) models, particularly those using Foundation Models (FMs), which struggle because they rely on simple class comparisons rather than understanding complex objects as compositions of basic concepts.

The proposed solution is CompSLOT, a universal, plug-and-play module that injects concept-level compositionality into any CL method with an FM backbone.

1. It uses a self-supervised Slot Attention mechanism to break down images into low-dimensional representations called slots (concepts). 
2. It introduces a primitive selection mechanism to identify and aggregate the most class-relevant concepts from the slots. A primitive loss ensures these primitives are consistent across different examples of the same class.
3.  The core plug-in component is a primitive-logit alignment loss. This loss distills the concept-level similarities between images directly into the model’s final predictions. This regularization guides the model to make decisions based on meaningful, shared, and distinct concepts, rather than simple feature comparisons.

### Strengths
Slot attention module is highly stable and shows almost no forgetting across sequential tasks.

CompSLOT is method-agnostic and computationally lightweight as it builds upon the existing FM backbone.

CompSLOT significantly boosts the accuracy of diverse CL baselines.

It enhances compositional generalization abilities.

Benchmarking is broad, and even includes fine-grained classification such as CUB 200. 

The work is clearly written, images are readable. Only Figure 1 is too detailed for teaser image and should be simplified to improve the clarity.

Experiments are convincing. 

Idea is novel, straightforward and easy to follow.

### Weaknesses
The work should be better contextualized in terms of concept-based continual learning, including discussion with work of [1] and follow-up works. 

Figure 1 can be improved as it is right now too complex and does not convey the message about novelty well. 

Contribution description is vague and unclear. It looks like in the second bullet the CompSlot designed something, not the authors. Looks like the artifact from LLM text improvement. The language there should be simpler, and maybe following organisation made:

- first dot about introduction of compslot and its key components.

- second dot about novel training components as losses

- last one about extensive experiments. 

[1] Rymarczyk, Dawid, et al. "Icicle: Interpretable class incremental continual learning." Proceedings of the IEEE/CVF international conference on computer vision. 2023.

### Questions
I would like to ask authors for better contextualization of the work and improved clarity.

### Soundness
3

### Presentation
3

### Contribution
3
