# Emergent Corpus Pretraining Benefits Vision Language Modeling

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3

## Abstract
Vision Language Pre-trained Models (VL-PTMs) have achieved state-of-the-art results across various tasks, but their effectiveness heavily relies on large-scale multimodal datasets. While VL-PTMs excel in scenarios with abundant data, they struggle to achieve sample efficiency in tasks with limited data resources. In this work, we explore the use of Emergent Communication (EC) for knowledge transfer in VL-PTMs. In particular, we pre-train a state-of-the-art Vision Language (VL) model on a corpus of EC tokens, generated through a referential game involving two artificial agents. Through experiments on three diverse cross-modal matching and reasoning tasks, we demonstrate significant performance improvements. For instance, EC pretraining enhances Visual Referring Expression (VRE) accuracy by $108.6\%$ while improving Visual Entailment (VE) performance by $69.6\%$. We further demonstrate that a vision-language model, exclusively pre-trained on EC tokens from scratch utilizing a sequence-to-sequence learning objective, can be effectively leveraged for fine-tuning numerous other vision-language downstream tasks, outperforming baseline settings without any pretraining and in some cases significantly narrowing the performance gap with models pre-trained on natural language. These results highlight the transferability and generalization capabilities of EC pretraining across different VL tasks and the potential of leveraging the multimodal grounding of EC tokens to enhance VL understanding in resource-constrained settings, especially in settings with limited natural language data. We discuss implications and propose avenues for future research to explore the connections between EC and VL for multimodal understanding and effective human-machine communication.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper explores the problem of generalization in vision-language modeling. To enhance the generalizability of vision-language models, the authors propose to use emergent communication (EC) between a listener and a speaker agent. Experiments showcase several potential benefits of EC for the visual entailment, visual question answering, and visual referring expression tasks.

### Strengths
The paper includes the following strengths:

- The proposed method is intuitive and interesting.

- The methodology is well-written and understandable with accurate notations and coherent explanations.

- The experiment section lucidly states the details regarding the settings and results.

### Weaknesses
The paper contains some minor but serious weaknesses:

- The motivation for EC is not convincing. For example, in the introduction, the author declares that EC is a promising approach without providing any evidence / explaining why this is promising. Moreover, even though self-supervised learning (SSL) is a well-known solution to tackle the limit of labeled data, the introduction lacks the the discussion towards SSL.  

- The intuition of emergent language is not evident. Why do some unintelligible tokens, e.g. in Figure 2, can benefit the vision-language models, which primarily work with natural language? The paper does not provide an intuitive discussion towards this aspect.

- The experiments are also not comprehensively conducted. There is not an ablation study to investigate the effect of each component and also an analysis for better understanding the EC framework, e.g. why the authors choose OFA as the base model for EC?

### Questions
- In the intuitive sense, why does emergent language can benefit vision-language modeling, which is mainly about natural language?

- Why do you choose OFA as the base model? Does EC perform effectively with other models, such as UniVL, CLIP, etc.?

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents a methodology for pre-training Vision Language models on images paired with emergent communication (EC) strings prior to fine-tuning on downstream tasks. The EC strings are derived from a speaker model trained for an image reference game task in an emergent communication paradigm (where the speaker and listener agents must converge on a communication protocol). 

The main claim of the paper is that pre-training under this paradigm can provide useful inductive biases during pre-training that can help improve performance when fine-tuning on downstream tasks. 

As a testbed, the paper uses the architecture of the One For All (OFA) VLM model from Wang et al. 2022, evaluated on three downstream vision-language tasks: Visual Referring Expression (VRE, where the model must generate a bounding box for an object in being referred to by a natural language expression), Visual Question Answering (VQA), and Visual Entailment (VE, where an agent must classify a natural language string as being (1) entailed, (2) contradicted, or (3) neither in relation to a given image). 

The core of the experimental results compare three cases to each other: 

(Base) A pre-trained OFA model that is not fine-tuned on downstream tasks. 
(+EC) A pre-trained OFA model, further pre-trained on a corpus of EC token/image pairs, fine-tuned on downstream tasks. 
(+NL) A pre-trained OFA model, further pre-trained on a corpus of natural language / image pairs, fine-tuned on downstream tasks. 

On VRE and VE, the +EC model improves over the baseline while achieving lower (or comparable in some cases with more fine-tuning data in the VE task) performance than +NL. In VQA the +EC model is outperformed by the other two variants.

### Strengths
* I find the idea of pre-training on emergent communication strings to be compelling, and the paper motivates potential reasons to expect this benefit well (in particular, the idea that the structural properties of a learned EC protocol could yield useful learning signal is intuitive). 

* To my knowledge, the proposed methodology of pre-training a VLM model on EC data and the experiments evaluating this are novel. 

* The paper is well written, generally clear, and easy to follow.

* The experimental results show promise for the method (however, I have reservations about whether the claims are fully supported by the results, which I've listed under weaknesses).

### Weaknesses
My main concern, and the main reason for my ratings, is related to the experimental setup. I am concerned that the presented results do not fully support the conclusions of the paper: 

To my understanding the paper argues that EC pre-training may yield benefits for VLM model performance in cases where there may otherwise not be more data containing natural language / image pairs to train on. 

With this in mind, I believe the paper could be much stronger with comparison against the following experimental conditions (in addition to the Base, +EC, and +NL conditions already presented): 

(1) A Base model that is also fine-tuned on downstream task data, but *not* additionally pre-trained on either EC nor NL data. This would simulate the case where one only has access to (a) the original pre-training data, and (b) the downstream fine-tuning data, and could potentially improve model performance by additionally pre-training on EC data. Without this comparison, I do not believe that it is clear from the presented results if the improved performance of +EC over Base is due to the EC data itself or if it's due to the downstream fine-tuning. 

(2) Downstream task performance of a model pre-trained on the original pre-training data as well as EC data, but *not* fine-tuned on downstream tasks. This would simulate the case where one only has the original pre-training dataset and an evaluation set for the downstream task and could potentially improve performance by augmenting pre-training with EC data.

### Questions
1. My main question is if there are results available for either of the two conditions I described in Weaknesses? The lack of these results is the main factor affecting my rating. 

2. I'm wondering if the token vocabulary used for images is the same between the OFA model and the EC models? Or do they each learn their own independent tokenizer?

### Soundness
1 poor

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper explores the use of Emergent Communication (EC) for knowledge transfer in the Vision Language Pretrained Models. It pretrains a model on an EC corpus, then experiments on three tasks, including Visual Referring Expression (VRE) and Visual Entailment (VE). Empirical experiments show that pretrained on EC corpus can improve the performance on the downstream tasks, and highlight the transferability and generalization capabilities of EC pretraining on VL domain.

### Strengths
1. Empirical results show the VL models pretrained on EC corpus can be transferred to downstream tasks with improved gain in controlled settings.

### Weaknesses
1. Experiment part is pretty weak, lacks of baselines for comparison, especially strong baselines, to verify the effectiveness of the proposed approach.
2. The novelty of the method / approach is also a big concern.

### Questions
The major issue of this work is experiment part is too weak, needs more baselines for comparison to show the effectiveness of the approach, given the main approach also lacks of novelty.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor
