# On Discriminative vs. Generative classifiers: Rethinking MLLMs for Action Understanding

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Multimodal Large Language Models (MLLMs) have advanced open-world action understanding and can be adapted as generative classifiers for closed-set settings by autoregressively generating action labels as text. However, this approach is inefficient, and shared subwords across action labels introduce semantic overlap, leading to ambiguity in generation. In contrast, discriminative classifiers learn task-specific representations with clear decision boundaries, enabling efficient one-step classification without autoregressive decoding.
We first compare generative and discriminative classifiers with MLLMs for closed-set action understanding, revealing the superior accuracy and efficiency of the latter. To bridge the performance gap, we design strategies that elevate generative classifiers toward performance comparable with discriminative ones. Furthermore, we show that generative modeling can complement discriminative classifiers, leading to better performance while preserving efficiency.
To this end, we propose Generation-Assisted Discriminative (GAD) classifier for closed-set action understanding. GAD operates only during fine-tuning, preserving full compatibility with MLLM pretraining. Extensive experiments on temporal action understanding benchmarks demonstrate that GAD improves both accuracy and efficiency over generative methods, achieving state-of-the-art results on four tasks across five datasets, including an average 2.5\% accuracy gain and 3$\times$ faster inference on our largest COIN benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Authors explore how Multimodal LLMs (MLLMs) can be used for closed-set action classification, comparing generative and discriminative approaches. They claim that generative classifiers (autoregressively producing text) are less efficient and accurate compared to discriminative classifiers.
The authors proposed a Generation-Assisted Discriminative (GAD) classifier that combined strengths of both aspects. This model uses a discriminative framework for efficient inference but incorporates generative modeling as an auxiliary task during fine-tuning to enhance the model's semantic understanding. Evaluation across five datasets establish clear performance gains in terms of both accuracy and inference speed.

### Strengths
1. Strong results across multiple datasets with multiple baselines with focus on both accuracy and speed. 
2. Interesting dual training strategy. 
3. Clear presentation of key idea (e.g. great use of diagrams in Figure 2).

### Weaknesses
1. "approximately 1.8× faster training": Discuss this more - how is the training faster than next-token prediction? This is unclear.

2. Lost generality: does this classification training remove the general QnA abilities of these VLMs? This is unclear. 

3. How do these VLMs perform zero-shot (generative classifier) on these tasks? This will provide an interesting point of comparison. 

4. Inference compute scaling: if you generate one or two tokens (with the generative part of GAD), and then run the classifier on all tokens, does this improve performance? This will be an interesting analysis.

5. How does cls head scale with more output categories? For example, say 20,000 categories? What is the impact on inference speed? 

6. Considering discussing prior work exploring similar ideas for video QnA and NLP. Two examples below. 
  - https://arxiv.org/abs/2403.16998 (using MLLMs for closed-set video classification)
  - https://arxiv.org/abs/2210.12353 (using LLMs for closed-set classification)

### Questions
See weaknesses.

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
4

### Summary
The paper presents a study on generative vs discriminative methods for action classification, illustrating how the latter outperform the former in most cases. Several potential causes are mentioned in the paper, including the partitioning of actions into different tokens leading to overlapping and lack of distinctiveness in the representation of actions. The authors then propose a generative assisted classification approach where an LLM is endowed with a class token tasked with predicting actions in a closed-set environment, followed by a language decoder to decode past and future actions around the one classified with the CLS token. The paper is accompanied by supporting experiments illustrating the drawbacks of existing LLMs for generative action classification. Additional results show that their proposed approach can enhance existing LLMs for action classification.

### Strengths
The paper is very well written and presented. The motivation is clear and the paper is properly threaded. The authors identify specific challenges of generative models for action classification, and enumerate and study the possible reasons behind this subpar performance w.r.t. discriminative methods. Then, the authors devise a combined approach that is shown to produce better results. The paper is accompanied by proper ablation studies and an efficiency analysis, illustrating how discriminative (or single-token decoding) is not only more accurate in terms of closed-set prediction, but also faster at inference.

### Weaknesses
While the paper is well threaded and motivated, part of the story is driven towards obvious conclusions that leave aside some potential alternatives (please see the questions below). In particular, 

1. The fact that discriminative approaches, or closed-set approaches, outperform generative ones, is not a finding or contribution of this paper, it is general knowledge. It is not expected that open-vocabulary models will outperform discriminative approaches in closed-set environments. The narrative is driven in a way that benefits discriminative models, without proper reference to the fact that this is constrained to the specific environments where the set of actions to be recognized is closed. In the questions below I suggest a potential study for a fairer comparison between generative and discriminative approaches. I also believe that bridging the gap between both methods is indeed making them the same (i.e. if a single token is used per action by extending the vocabulary, and the logits are capped to those of the target actions, then a generative approach boils down to a discriminative one with learnable queries and a one-hot binary classification approach). 

2. It is a bit misleading that the narrative is built and tested in an ego-centric scenario where discriminative approaches are prone to thrive. How would the whole story sustain itself in a more generic scenario where several actors are performing potentially overlapping actions? E.g. the AVA setting where actions are not exclusive (i.e. in a multi-class scenario). In such case it looks to me that a generative approach can outperform discriminative approaches. If it is out of the scope of this paper to study such scenario, then it should be clearly stated in the paper to avoid leading to misinterpretations. 

3. There's a missing text-to-text action retrieval from the predictions of generative approaches to closed-set predictions. In particular, it is not clear (or at least I missed that in the paper) how the generative predictions are mapped to actions. 

4. There's no breakdown (unless I missed that, in which case I hope the authors can address this in the rebuttal) of where the confusing is coming from. In Fig. 3 the authors plot the false positives for the "add sugar" action, but it is not clear to me if this refers to action within the target set, or free-form actions.


In summary, I am borderline with the paper, as I believe it holds some valuable contribution, it is well written and presented, but lacks some proper contribution that is properly sustained.

### Questions
The runtimes in Table 1 illustrate that the discriminative counterparts of the corresponding LLMs are much faster than the generative ones. I understand that this might be due to the fact that when the actions are split into several tokens, it requires several forward calls to the model, incurring in slower inference. However, if the model is provided with action-specific tokenization as mentioned in l. 363, then inference should be achieved in a single forward pass. In such case the speed should be similar for both generative and discriminative approaches. In any case, what is the difference between a generative approach with extended vocabulary, whereby each full action corresponds to a token, and a discriminative approach with as many [CLS] learnable tokens as class labels? In the latter approach a one-vs-all discriminative approach could be learned (please refer to works on meta queries). I find that referring to the extended-vocabulary approach to generative is a bit overstretched. Can the authors elaborate on this?


In a closed-set setting with N classes, comparing discriminative and generative methods with the latter having a classifier that has L logits with L >> N is obviously biased. Have the authors tried to compare both methods by capping the classifier of the generative approach to N logits only, or to N' logits with N' being only the logits corresponding to the tokens involved in tokenizing the target classes? 


How sensitive are the generative results to the tokenizer? I believe a proper study with further text encoders is necessary to study the influence of the tokenizers in the generative results. 

How is the method trained in the case that there's a single action in a video? e.g. Kinetics?

### Soundness
3

### Presentation
4

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
This paper investigates the use of Multimodal Large Language Models (MLLMs) for closed-set temporal action understanding. The authors identify a performance gap between generative and discriminative classifiers, attributing it to semantic overlap in label tokenization. To address this, they propose a Generation-Assisted Discriminative (GAD) classifier that incorporates an auxiliary generative objective during training to provide semantic regularization, while maintaining the discriminative model's efficient inference. Evaluations on multiple action understanding benchmarks show GAD achieves state-of-the-art results.

### Strengths
- The paper's greatest strength is its thorough, apples-to-apples comparison between generative and discriminative approaches across a wide range of tasks and datasets. 
-  The implementation details in the main paper and appendix are extensive.

### Weaknesses
- Using a discriminative head on top of a generative model and training it with an auxiliary task is a standard practice in machine learning. The conceptual contribution is limited.
- The paper lacks a deep analysis of how the auxiliary generative task helps. 
- The GAD framework is only evaluated in closed-set settings. Its applicability to open-world or few-shot scenarios is not discussed, limiting the scope of its claimed generality.

### Questions
- Given that the architecture of the discriminative classifier (appending a [CLS] token) and the idea of multi-task learning with an auxiliary objective are well-established, what is the specific conceptual novelty of the GAD framework beyond its application to the video modality?
- Can you design experiments to isolate the benefit of the auxiliary task? For instance, compare against a model trained with an equivalent amount of additional discriminative data or a different, non-generative auxiliary task to prove that the generative nature of the objective is key?

### Soundness
2

### Presentation
3

### Contribution
2
