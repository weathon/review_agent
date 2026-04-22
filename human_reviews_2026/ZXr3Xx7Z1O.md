# Training Dynamics Impact Post-Training Quantization Robustness

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
While post-training quantization is widely adopted for efficient deployment of large language models, the mechanisms underlying quantization robustness remain unclear. We conduct a comprehensive analysis of quantization degradation across open-source language model training trajectories up to 32B parameters and 15T training tokens to accurately assess the relationship between training dynamics and quantization performance. Our key finding is that quantization errors in large-scale training runs are driven by a complex interplay between learning rate and other training hyperparameters. Specifically, once learning rates decay, validation loss and quantization error diverge, largely independent of training data scale. To investigate interventions on the training dynamics and identify specific configurations that can modulate quantization robustness favorably, we train our own models in controlled experiments up to 100B tokens. Our results challenge the assumption that increasing dataset scale inherently compromises quantization effectiveness, demonstrating instead that strategic training hyperparameter interventions can improve quantization quality at scale.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper is an empirical study which explores the relationship between training dynamics and quantization (PTQ) performance, providing interesting observations that quantization errors are related to training hyperparameters like learning rate and scheduler settings, challenging the previous assumption that quantization errors are inherently related to dataset scale. The authors also experiment to intervene the training dynamics to identify specific configurations that modulate quantization robustness favorably, providing practical insights for related studies.

### Strengths
- The insights provided by this paper are very interesting and original. It discusses the relationship between quantization errors and training hyperparameters like learning rate and scheduler settings in detail, and shares some empirical results like the divergence of quantization error and validation loss with the decay of learning rates. I believe those insights could benefit future related studies.
- The efforts of the authors trying to reduce quantization error by intervening training dynamics are more applicable for practical usage, compared with previous works' focuses on training data scale.
- The paper is well presented with sufficient experiments and corresponding figures, making it easy to visualize the key points of the observations.

### Weaknesses
- While the empirical results provided by this study are abundant and interesting, the paper fails to provide more in-depth explanations on the reasons that lead to such phenomena, as it's not very explicit to relate factors like learning rate with quantization errors. This might limit the interpretability of the provided results.
- The selection of evaluated models lacks representativeness, as several of the most widely used and influential model families (e.g., LLaMA, Qwen) are not included. So it remains unclear whether the observed correlations between training dynamics and quantization robustness hold for mainstream architectures.

### Questions
- Could the authors provide a theoretical or intuitive explanation for why learning rate decay leads to increased quantization error? A deeper understanding of the underlying mechanism would significantly enhance the paper’s conceptual contribution and long-term impact.
- The study focuses on the relationship between training dynamics and PTQ. However, given that PTQ is primarily applied when retraining is infeasible or training details are unavailable, what is the practical motivation for analyzing training-phase interventions on PTQ performance? If the authors have the capability to conduct large-scale pretraining & fine-tuning, why not extend the study to Quantization-Aware Training (QAT) and investigate how training dynamics influence QAT outcomes, which may offer more actionable insights for future model development?

### Soundness
3

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
The authors study the impact of training dynamics (learning rate schedule) on post-training quantization Error. Through the study of various open source model families, the authors find that the learning rate schedule is the primary driver of PTQ error contrary to the popular belief of over-training leading to higher PTQ error. Based on this insight, the authors propose ideas to mitigate this discrepancy.

### Strengths
- This is an excellent paper. The authors analyse multiple different families to show that their insights hold.
- The interventions (at a smaller scale) also demonstrate their findings.
- PTQ degradation remaining near flat with WSD LR with higher tokens/params is a great observation.
- The authors study both PTQ error and downstream performance degradation

### Weaknesses
- The authors claim the effect of training hyperparameters on quantization quality hasn't been well studied, yet the authors don't cite Intriguing Properties of Quantization at Scale [1] by Ahmadian et al. 

- The fact that smaller learning rates lead to larger PTQ errors hints at the manifold geometry (sharpness etc.) playing a key role in determining the degradation yet there is no discussion about this. It would be nice to relate PTQ errors to the geometry of the loss basin (albeit at a smaller scale). I think this would considerably strengthen the paper.

[1] Intriguing Properties of Quantization at Scale : https://openreview.net/pdf?id=IYe8j7Gy8f

### Questions
- Do the observations hold for recent quantization techniques like QuaRot etc?

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
4

### Summary
The paper explains recent work on post-train quantization scaling finding that degradation increases with data. The authors attribute it instead of learning rate annealing effects, and propose two new ways to mitigate this degradation.

### Strengths
- The paper is clearly divided into sections, where each one has a clear claim and empirical evidence for it. 
- They replicate and explain past work on PTQ scaling, reassuring the reader that their baselines are well-tuned. 
- They propose interventions to mitigate the identified "mechanism" for degradation. 
- Fig5 is particularly strong evidence that the annealing itself is causal for degradation.

### Weaknesses
- I'm not entirely sure this paper has enough "meat" to be a conference paper. There is a lot of redundancy in the plots, and the contributions/main claim can be summarized as "higher LR and model averaging can partially mitigate PTQ degradations on long training runs, though we don't know why." It's not clear this is enough to comprise a conference paper? 
- Even the core claim that "these data effects may actually be LR effects" is actually not novel: [1] make a very similar claim in reference to the same literature, but in the related setting of finetuning instead of quantization. 
- The authors also do not posit any conceptual model explaining their findings, or interpret their findings. Even if there is no theory (which is fine), having a mental model with experimental ablations would be helpful. What exactly is going on here -- the fact that WSD and cosine both end up giving the same PTQ-induced loss but over timescales makes it feel like there is some "degradation potential" interpretation in the spirit or flavor of [2]. There is no actual scientific model presented with experiments. 

[1] Overtrained Language Models Are Harder to Fine-Tune. Springer et al, 2025. https://arxiv.org/pdf/2503.19206

[2] Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape View. Wen et al., 2025. https://openreview.net/pdf?id=m51BgoqvbP

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper examines the interplay between post training quantization (PTQ) performance and variables related to training dynamics, specifically learning rate schedule and model averaging. The authors challenge the conclusion from prior work that PTQ error is primarily driven by training duration. The paper presents evidence that observations about training length duration are confounded by learning rate cooldown and that it's the cooldown period which primarily drives PTQ error growth. The authors demonstrated their claims against a large suite of open source models.

### Strengths
* The question that the paper studies is a very interesting one and disentangling learning rate cooldown from training duration is a subtle but important distinction that can inform practitioners in their pretraining choices.
* The investigation into model souping provides practical information to guide practitioners in reducing PTQ error.
* The effect observed is quite convincing in terms of home prominent of a phase transition there is
* The empirical results are very thorough and replicate over a large suite of models and dataset sizes.

### Weaknesses
* While the phenomenon observed is quite interesting, the paper is missing a predictive model of the effect of learning rate on PTQ. Similar to previous works (Kumar et al.), providing some scaling analysis that incorporates the relevant LR parameters would strengthen the paper greatly.

### Questions
* While potentially out of scope for this work, it seems important to understand whether the same phenomenon occurs with quantization aware training (QAT). Did the authors run any experiments with QAT?
* From looking at Figure 3, it seems like the slope of the quantization error decreases as a function of the model size examined. It would be good to disentangle whether lr cooldown impacts quantization error less at larger scales or whether this is the interplay between overtraining and lr cooldowns.

### Soundness
3

### Presentation
3

### Contribution
3
