# The Generalization Ridge: Information Flow in Natural Language Generation

- Decision: Reject
- Scores: 8, 2, 4, 6

## Abstract
Transformer-based language models have achieved state-of-the-art performance in natural language generation (NLG), yet their internal mechanisms for synthesizing task-relevant information remain insufficiently understood. While prior studies suggest that intermediate layers often yield more generalizable representations than final layers, how this generalization ability emerges and propagates across layers during training remains unclear. To address this gap, we propose InfoRidge, an information-theoretic framework, to characterize how predictive information—the mutual information between hidden representations and target outputs—varies across depth. Estimating this quantity enables us to trace the flow of task-relevant information throughout the model during training. Our experiments across various models and datasets reveal a consistent non-monotonic trend: predictive information peaks in upper-middle layers—forming a **generalization ridge**—before declining in final layers, reflecting a transition between generalization and memorization. To further investigate this phenomenon, we introduce residual scaling coefficients—trainable scalar parameters applied to each residual block—which serve as functional probes for assessing the relative importance of individual transformer layers. These coefficients reveal that, under distribution shift, models downweight final layers and increasingly rely on ridge layers, highlighting their role in generalization. Together, these findings offer new insights into the internal mechanisms of transformers and underscore the critical role of intermediate layers in supporting generalization.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work analyzes the representations across transformer layers in a scenario of fine-tuning for natural language generation tasks. The authors propose an approach based on mutual information between the internal model representations and the ground truth target token, and formalize metrics of "predictive information" and "incremental information gain", corresponding to a specific layer hidden state or a specific layer residual, respectively. These mutual information metrics rely on matrix-based Renyi entropy. In a series of analysis experiments the work demonstrates a behavior termed "generalization ridge", where the mutual information with the task label reaches a peak before the final transformer layer, and this point in the model also corresponds to better out-of-domain generalization performance.

### Strengths
1. Some very interesting findings on model behavioral dynamics, that are consistent across models and tasks.
2. The conclusions are supported through multiple complementary experimental setups: the observational results are complemented by intervention experiments - controlling the contribution of each transformer layer via residual scaling coefficients; standard NLG data is used alongside a controlled synthetic dataset setting; mutual information results are complemented by an explicit analysis of attention to signal-bearing tokens.

### Weaknesses
1. My feeling is that the interpretation of the data is probably a bit over-simplified, given that high mutual information between a representation and the target label can also be potentially consistent with memorization. Throughout the paper a rise in predictive information is interpreted as increased generalization, whereas in the overfitting experiment (l. 332) a similar rise is described as "memorizing shortcuts" and "redundant label-specific noise". This is interesting, and helps to give the reader a more complete picture, but also highlights that the meaning of the predictive information metric may not be as straightforward as the authors make it out to be.
2. The analysis is limited to fine-tuning scenarios. I think it would have been informative to also see some examples of these MI metrics in a zero-shot/few-shot scenario.

### Questions
1. For Table 1 accuracy results, how exactly is the early-exit implemented? I see that the language modeling head uses the same weights as for the input embedding, but still - maybe the results are influenced by a mismatch between the language modeling head and the hidden states of intermediate layers? AFAIU the embedding/unembedding matrix weights are only optimized for projecting the input and last layer output.
2. Is the approach for creating the synthetic dataset new? Or did other works use similar approaches?

Minor suggestions:
* I felt Figure 1 in its current form is not particularly easy to follow visually, and thus less informative for the reader. I suggest to think a bit about how to organize it visually and create a stronger connection between the text and the graphics.
* The transition to the part about "signal attention" (l. 343-345) can be improved, I found it a bit confusing. My suggestion is to replace that with the more explicit goal of this experiment (to quantify where in the network attention to semantically important information is concentrated, as currently stated in l. 350) and then describe the experiment.

Typos:

l. 125 reflects -> that reflects

l. 328/330 generalization ridge -> a/the generalization ridge

l. 401 blocks -> block

l. 921 quantity -> quantifies, blocks -> block

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper analyzes the mutual information between hidden layers and outputs for language models across a few tasks. The paper finds a spike in the upper middle layers which they coin generalization ridge.

### Strengths
- The paper is a nice application of using information theory to identify layer importance
- The finding is self-contained and the paper fairly easily to follow

### Weaknesses
At a high level, I was disappointed after reading the paper for few reasons. (1) I was missing the reason for why I should care about these results, especially since the results are not connected to the existing body of literature. There are many prior papers that investigate layer-wise importance and the result that upper middle layers are important is not new. (2) The key claims even in the title relate to shedding light into NLG. But, there is no actual experiment on NLG tasks and the experiments only look at a single output, not even multiple tokens. 

**Some related work**: Some discussions I was expecting to see were grounding in work like "BERT rediscovers the classical NLP pipeline" which relates layers to specific NLP tasks (plus an extensive body of literature around this). I was expecting a discussion of causal vs. correlational probing work, for example causal mediation analysis as an alternative way to identify layer/neuron importance. And since this was investigated, I was expecting more relation to work on training dynamics. 

Similarly, there is a lot of prior work on information-theoretic probing approaches that the method here does not warrant a new name. This is an implicit overclaim. 

**Lack of Relation to NLG**

The model selection is very surprising and limited to very small models. The older models here are not models that were ever used for actual NLG work. 
Moreover, none of the datasets are well known NLG datasets or classic NLG tasks (such as summarization, dialog response generation, or surface realization). The arithmetic dataset is not at all an NLG dataset. In addition, looking at a single output representation implicitly looks at only a single output and not a longer utterance as would be expected in NLG. 

**Relationship between token representation and output**
The extraction of $\Delta$Z (the change in token representation) implies that the information is represented in linear space. While this could be a helpful approximation of the ground truth, this is a very strong assumption to make, given the very much non-linear nature of the model. Moreover, page 4 states that delta Z is further $l2$ normalized which further changes the nature of the representation away from what the model represents

On a clarity note, I don’t understand why the Matrix-Based Mutual Information section does not continue to use the prior representation introduced above. This seems rather lazy as it is left to the reader to connect how it is applied. Not even the information in the appendix discusses details

### Questions
n/a

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
3

### Summary
This paper investigates how Transformers operate internally and how task-specific generalization emerges during training and propagates across layers for natural language generation. The authors estimate mutual information between layer-wise representations and the outputs, and learn per-residual-block scaling coefficients to assess each layer’s contribution. Using these coefficients, they find that mid-layers become substantially more important under distribution shift, whereas later layers are more tied to memorization of in-distribution patterns.

### Strengths
1) The paper studies very important questions of how task-specific generalization abilities emerge in a transformer layer, which is largely not understood.
2) The paper is well written, easy to follow and the core points are well articulated.

### Weaknesses
1) It’s unclear how “which layers are important” alone explains how generalization emerges. The study doesn’t identify the causal factors behind these abilities.

2) Recent work already reports that middle (and later) layers matter, and that final layers can be poor for OOD generalization (e.g., [1], which is cited but not discussed much). Can the authors discuss more how the new findings from these papers are different the existing ones to understand how generalization emerges and what are the practical usefulness of these findings.

3) The evaluation datasets are limited. It would be valuable to relate the identified “ridge” layers to varied task types—math, general reasoning, and multi-hop reasoning. Please add results on standard datasets such as MMLU, GSM8K, HotpotQA etc.

4) There is no accompanying theoretical analysis.

[1] Uselis, Arnas, and Seong Joon Oh. "Intermediate layer classifiers for ood generalization." arXiv preprint arXiv:2504.05461 (2025).

### Questions
1) Is there a specific reason the authors use a Gaussian kernel for entropy estimation? Could they include an ablation with alternative kernels? I want to see the sensitivity of the results depending on the kernel choice.

2)  Although multiple model sizes are considered, the scaling trends are not clearly characterized. Can you clarify how layer importance evolves with width/depth. I want to see the evaluation on existing open source models, no training is needed.

3) It seems the models are fine-tuned from pretrained checkpoints, and the layer-wise mutual information “peakedness” seems to reflect pretrained biases. Do the claims about ridge layers, memorization, and generalization still hold when training from scratch?

4) Suggestion: Consider splitting Figure 2 into multiple panels/figures to improve readability.

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
This paper introduces InfoRidge, an information-theoretic framework to analyze how predictive information flows through transformer layers during natural language generation (NLG). Using matrix-based mutual information estimation, the authors track two key quantities: 1) mutual info between hidden states at layer $l$ and target token (predictive information) and 2) info added by the residual update at layer $l$ (incremental information gain). Across GPT-2, Qwen-2.5, and LLaMA models on CLUTRR, ECQA, and a synthetic arithmetic dataset, the authors consistently observe a non-monotonic “generalization ridge”: predictive info peaks in upper-middle layers then declines, while final layers appear to memorize. They corroborate this with learnable residual scaling coefficients, allowing the model to assign layer importances depending on the layer influence towards prediction. Authors find that under distribution shifts, models automatically down-weight final layers and up-weight ridge layers, providing further evidence to the existence of generalized intermediate representations. The ridge emerges only beyond a depth threshold, and attention/visualization analyses show ridge layers focus on task-relevant tokens whereas final layers revert to positional heuristics.

### Strengths
1.	Originality: The proposed adaptation of matrix-based mutual information to next-token prediction is an interesting idea to identify layer influence towards predictions.
2.	Quality: Authors conduct extensive experiments across model families & tasks, careful controls (depth ablation, overfitting, attention/decoder visualizations).
3.	Clarity: Hypotheses, metrics, and takeaways are explicitly stated; additional mathematical details are added to appendices.
4.	Significance: The proposed method based on mutual information can provide a principled lens to locate “where” generalization happens, immediately useful for early-exit, model pruning, or targeted fine-tuning strategies.

### Weaknesses
1. **Scalability of MI estimation**: In Appendix C, the authors state that 50-200 examples are subsampled to avoid excessive memory usage from large Gram matrices. Provided authors experimented mainly with small language models, this raises doubts on the applicability of such a method to analyze larger state-of-the-art LLMs.
2.	**Lack of proper causal verification**: While the analysis shows that ridge layers correlate with better OOD accuracy, the use of residual scaling coefficient to test their importance in a task-directed fashion might lead to confusing results, since 1) $beta$ coefficients are trained after full fine-tuning with frozen weights; it is unknown whether jointly learning $beta$ from scratch preserves the ridge or changes training dynamics; and 2) subsequent layers depend from previous ones in their computation, creating an implicit incentive to upweight earlier layers to reduce information loss. A more convincing analysis could, for example, employ activation patching methods to localize the units that contribute most to model predictions in OOD settings, showing that these are mainly contained within ridge layers.
3.	**Findings are not particularly novel**: While the characterization of the middle-to-final layers as responsible for generalization is novel, there is ample literature showing that these layers perform best when used for probes aiming to extract semantic task-related information (see e.g. the early work "BERT Rediscover the Classical NLP Pipeline" by Tenney et al., showing a hierarchy of linguistic information within BERT peaking at around 2/3rds of the model). While the analyses conducted by the authors are undoubtedly of value in providing additional evidence for the semantic generality of such layers, this is unlikely to affect the current practices that already employ those layers for interpretability analysis on language models.

### Questions
In the early exiting results shown in table 1, are different prediction heads trained over a frozen model, or is the final head simply applied to the selected layer? In the latter case, the mismatch between latent spaces in absence of a projection could lead to misleading results.

### Soundness
3

### Presentation
4

### Contribution
3
