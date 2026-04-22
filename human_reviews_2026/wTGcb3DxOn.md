# LLM Pretraining with Continuous Concepts

- Avg Score: 7.33
- Decision: Accept (Poster)
- Scores: 8, 6, 8

## Abstract
Next token prediction has been the standard training objective used in large language model pretraining. Representations are learned as a result of optimizing for token-level perplexity. We propose Continuous Concept Mixing (CoCoMix), a novel pretraining framework that combines discrete next token prediction with continuous concepts. Specifically, CoCoMix predicts ``continuous concepts'' learned from a pretrained sparse autoencoder and mixes them into the model's hidden state by interleaving with token hidden representations. Through experiments on multiple benchmarks, including language modeling and downstream reasoning tasks, we show that CoCoMix is more sample efficient and consistently outperforms standard next token prediction and knowledge distillation. We find that combining both concept learning and interleaving in an end-to-end framework is critical to performance gains. Furthermore, CoCoMix enhances interpretability and steerability by allowing direct inspection and modification of the predicted concept, offering a transparent way to guide the model’s internal reasoning process.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
In this paper, the authors design a new method to improve the next-token prediction and final performance. Specifically, they design CoCoMix, a new pretraining framework for large language models that augments standard next-token prediction with an additional objective: predicting “continuous concepts” derived from a sparse autoencoder and injecting those concepts directly into the model’s hidden states, interleaved with normal token representations. It improves both sample efficiency and downstream performance on language modeling and reasoning tasks. Beyond accuracy, CoCoMix also makes models more interpretable and steerable, because the predicted concepts can be inspected and edited to transparently influence the model’s internal reasoning. The provided figures and examples make this paper easier to understand. Overall, the quality of this paper will be further improved after addressing concerns listed below.

### Strengths
The topic is highly interesting and might generate broad impact to the LLM community. 

I like the visualizations of figures, which are clear and improve the readability of this paper. The reviewer appreciates the authors for doing this.  

In experiments, the analysis is solid and comprehensive.

### Weaknesses
[1 ] To be honest, after reading Figure 1 alone or combined with the text in the introduction, the reviewer is still confused about how the extracted concepts benefit the next token prediction. More explanations might be helpful. 

[2] During the target concept selection process, it the attribution conducted in each training batch? 

[3] Another question is about clarification, after the concept selection, why it is necessary to conduct concept prediction? 

[4] In experiments, the model sizes are relatively small. The reviewer understand this might be because the limited budget in model training. Could you discuss whether the findings in this paper could generalize to larger models? 

[5] It will be great to disclose the training costs, e.g., GPU numbers, types, GPU hours. 

[6] While there are performance gain, will it brought other costs like training time when applying COOMIX?

[7] Sumamrizing the performance gain will improve this paper further. For example, in the abstract, intro, captions of Figure 2 and 3, it will be great to introduce the performance gain. 

[8] The steering example is interesting. COuld you introduce how do we know which concept is related to “website” or “money”? The transparency will improve the readability of this paper.

### Questions
[1 ] To be honest, after reading Figure 1 alone or combined with the text in the introduction, the reviewer is still confused about how the extracted concepts benefit the next token prediction. More explanations might be helpful. 

[2] During the target concept selection process, it the attribution conducted in each training batch? 

[3] Another question is about clarification, after the concept selection, why it is necessary to conduct concept prediction? 

[4] In experiments, the model sizes are relatively small. The reviewer understand this might be because the limited budget in model training. Could you discuss whether the findings in this paper could generalize to larger models? 

[5] It will be great to disclose the training costs, e.g., GPU numbers, types, GPU hours. 

[6] While there are performance gain, will it brought other costs like training time when applying COOMIX?

[7] Sumamrizing the performance gain will improve this paper further. For example, in the abstract, intro, captions of Figure 2 and 3, it will be great to introduce the performance gain. 

[8] The steering example is interesting. COuld you introduce how do we know which concept is related to “website” or “money”? The transparency will improve the readability of this paper.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work includes a range of experiments that demonstrate the model’s performance across multiple dimensions, such as downstream accuracy, efficiency, and steerability. Overall, the experiments are clearly presented and thoughtfully discussed. The chosen baselines are relevant, though somewhat limited. The analysis of the CocoMix model’s effectiveness is generally solid and supported by clear results.

### Strengths
1. The CoCoMix model is well described, and the method is easy to follow.

2. The proposal of combining next token prediction with continuous concepts in the pretraining paradigm is novel. This idea of integrating an interpretability mechanism (concept) into pretraining frameworks through SAE comes with significant originality. Such pretraining innovations remain rare in the field, and the model's effectiveness suggests potential incremental impact.

3. This work includes a range of experiments that demonstrate the model’s performance across multiple dimensions, such as downstream accuracy, efficiency, and steerability. Overall, the experiments are clearly presented, and the chosen baselines are relevant, though somewhat limited. The analysis of the CocoMix model’s effectiveness is generally solid.

### Weaknesses
1. Concept Interpretability: This work has been based on the assumption that the latent representation layer in SAE corresponds to human-interpretable concepts. Since this is central to the interpretability claims, more content addressing this assumption would be helpful, beyond what is described in the steerability section. How exactly does CoCoMix capture real, continuous mixture of concepts?

2. Model Design Justifications: Some architectural choices could be better justified if the authors add more explanations or ablation studies. For example, what's the significance of compressing top concepts into a "continuous" one, instead of focusing on the top concept?

3. The authors claim that their improvement on downstream tasks shows that CocoMix is better than NTP (Figure 3), but these differences in performance are minimal at best, and I'm not sure if they're significant. Is it worth the overhead?

### Questions
1. Figure 2 PPL scores -- are they computed using NTP objective for both models, or using NTP for the NTP baseline and the CocoMix PPL for the proposed method? If it's the latter, these scores are not really comparable...
2. For reproducibility reasons, I suggest the authors provide a link to their repo containing the code for the implemented methods and the experiments.
3. Does CoCoMix scale to larger LLMs?
4. Are the differences reported in the paper statistically significant?

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
4

### Summary
The authors propose a new pre-training paradigm based on "continuous concepts".

Specifically, the authors introduce an additional prediction head in a "standard" (transformer) LLM to produce continuous tokens based on "concepts". In order to train this additional prediction head the authors propose an additional training objective based on Sparse AutoEncoders (SAEs) and importance attribution using gradients.

The authors then validate their approach across various model sizes and baselines.

### Strengths
The main strenghts of the paper:
1. Great pre-training analysis with a novel architecture.
2. A working pre-training recipe that seems to improve performance
3. A way of introducing steerable concepts into the models generation. This can open the door to a lot of interesting research.

### Weaknesses
Main weaknesses:
1. Model sizes are limited (but understandable).
2. Hyper-parameter tuning was not discussed in detail and perhaps some of the scores can be attributed to poor hyper-params.

### Questions
Q1: How can you make sure your results are not attributed to randomness in hyper-param tuning? Can you demonstrate anything to this effect?

### Soundness
3

### Presentation
4

### Contribution
4
