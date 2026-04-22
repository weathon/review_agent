# BA-LoRA: Bias-Alleviating Low-Rank Adaptation to Mitigate Catastrophic Inheritance in Large Language Models

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 6, 8, 8, 4, 0

## Abstract
Parameter-efficient fine-tuning (PEFT) has become a *de facto* standard for adapting Large Language Models (LLMs). However, we identify a critical vulnerability within popular low-rank adaptation methods such as LoRA: they can exacerbate ``Catastrophic Inheritance''---the unchecked propagation of biases, noise, and data imbalances from pre-training. This phenomenon can degrade model robustness and fairness, undermining the benefits of efficient adaptation. To address this, we introduce Bias-Alleviating Low-Rank Adaptation (BA-LoRA). Our approach is founded on a principled decomposition of Catastrophic Inheritance into three core challenges: Knowledge Drift, Representation Collapse, and Overfitting to Noise. BA-LoRA systematically mitigates these issues by incorporating a trio of targeted regularizers: consistency, diversity, and an SVD-based term, designed to preserve core knowledge, enforce representational richness, and promote robust, low-rank output representations, respectively. We conduct comprehensive evaluations on a suite of Natural Language Generation (NLG) and Understanding (NLU) tasks using diverse, prominent open-source language models (e.g., LLaMA-2-7B and DeBERTa-v3-base). Our results show that BA-LoRA not only outperforms state-of-the-art LoRA variants in terms of performance and stability, but also demonstrates superior robustness and bias mitigation on targeted evaluations. These results provide evidence that BA-LoRA can counteract the adverse effects of Catastrophic Inheritance. Code is available at https://github.com/llm172/BA-LoRA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces BA-LoRA, a PEFT method designed to mitigate catastrophic inheritance, including the propagation of biases, noise, and data imbalances from pre-trained LLMs. BA-LoRA addresses three failure modes via three regularizers applied in the output space, i.e., consistency regularizer, diversity regularizer, and SVD regularizer.
Experiments evaluate BA-LoRA on NLG and NLU tasks, achieving state-of-the-art among LoRA variants. A controlled study shows BA-LoRA's gains are amplified on models pre-trained with noisy data. Ablation studies validate each regularizer's contribution.

### Strengths
1. The paper introduces a novel decomposition of catastrophic inheritance into three actionable failure modes. Each failure mode is directly linked to a specific regularizer (consistency, diversity, SVD), creating a coherent and interpretable architecture.
2. BA-LoRA sets a new state-of-the-art performance with strong empirical validation, on both NLG benchmarks and NLU tasks. Moreover, performance gains are even higher on noisier pre-trained models, directly validating its core hypothesis about mitigating inherited noise.

### Weaknesses
1. The comparison between RoBERTa-base and T5-base shows that BA-LoRA's advantage is significantly larger on T5 (3.26 vs 1.11 points). However, the models differ in both architecture (encoder-only vs encoder-decoder), which should be controlled to strengthen the claim. This makes it difficult to attribute the performance difference solely to the 'noisier data' factor.
2. The SVD regularizer uses different normalization schemes: NLU uses the sum of all singular values, while NLG uses the Frobenius norm. What is the theoretical justification for this difference? Understanding whether this choice is principled (e.g., due to task differences) or empirical is important for the generalization of the method beyond NLU and NLG.
3. The method introduces too many hyperparameters, including the tradeoff parameters $\lambda$s, temperature T, number of top-K, and SVD rank. Ablation studies on their sensitivity can enhance the usability of BA-LoRA in practice.

### Questions
1. In Table 1, the gain of superior performance mainly depends on the MBPP benchmark, where BA-LoRA gains 36.86, whereas the second-best baseline only achieves 25.74. It would be valuable if the author could analyze why BA-LoRA performs particularly well in this dataset.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors state that LoRA can lead to Catastrophic Inheritance, which hurts robustness and fairness. 
They propose Bias-Alleviating LoRA (BA-LoRA), which decomposes the issue into Knowledge Drift, Representation Collapse, and Overfitting to Noise. 
BA-LoRA adds three regularizers to preserve core knowledge, maintain representational richness, and encourage robust low-rank behavior.

### Strengths
1. Important topics
2. Three well-motivated losses mapped cleanly to three failure modes; easy to plug into existing LoRA workflows.
3. Broad NLU/NLG coverage with ablations that attribute gains to each component.

### Weaknesses
1. Although the framework names are shared, the diversity and SVD regularizers differ for NLU vs. NLG, so it reads like two papers rather than one universal design.
2. Some studies are missing in the paper (e.g., impact of different r/T/k, larger models)

### Questions
1. The paper builds on PiSSA initialization. How much of BA-LoRA’s improvement remains if you start from random low-rank adapters or standard LoRA init? Do the three regularizers still deliver comparable gains?
2. How sensitive are results to the SVD rank, distillation temperature, and the top-k entropy window?
3. How well does BA-LoRA scale to larger models (e.g., 13B, 70B)?

### Soundness
4

### Presentation
4

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
The authors present *Catastrophic Inheritance* in LLMs, a propagation of biases, noise, and data imbalances from pre-training into fine-tuning models, and ways to mitigate it. They identify: **Knowledge Drift**, where the model forgets pre-trained knowledge while learning new tasks; **Representation Collapse**, where fine-tuning on imbalanced data causes a lack of output diversity; and **Overfitting to Noise**, where the model learns spurious correlations that impact generalization. They offer three regularizers to combat these issues in Natural Language Understanding and Generation tasks: consistency regularizer for Knowledge Drift, diversity regularizer for Representation Collapse, and Singular Value Decomposition regularizer for Overfitting to Noise.

1. The consistency regularizer is based on a knowledge distillation approach via KLD between pre-trained and fine-tuned model probability distributions to preserve foundational knowledge.
2. The diversity regularizer is based on penalizing off-diagonal elements in a covariance matrix or maximizing entropy within the most plausible tokens to promote diversity in the model’s predictions across a batch.
3. The SVD regularizer is based on encouraging low-rank structure by maximizing the ratio of spectral energy in the top-k singular values to learn robust features.

They test the method **Bias-Alleviating Low-Rank Adaptation (BA-LoRA)** in mathematical reasoning, coding, and conversational AI for NLG, as well as on the GLUE benchmark for NLU, using models such as LLaMA-2-7B and DeBERTa-v3-base. Additionally, they look at gains in models that were trained with clean or noisy data, as well as t-SNE visualizations of the features to see the impact of data imbalance, and perform ablation studies.

### Strengths
The paper offers an original perspective on catastrophic inheritance with clear methodology and strong experimental evaluation, making it a significant and well-presented contribution.

---

- The abstract is very clear, well written, and easy to understand.
- The experimental setup is thoroughly described, with clear reporting of hyperparameters.
- The results are evaluated over multiple random seeds.
- The work covers a wide range of setups and datasets, reflecting a comprehensive and up-to-date evaluation.
- The experiment comparing noisy vs. clean data is particularly valuable and insightful.
- The appendix offers a lot of information on the setup, different models, sensitivity analyses, and performance across rank. This is very solid experimental work, and it should be highlighted more prominently in the main text, as a substantial amount of effort and insight resides there.

### Weaknesses
### Methodology and Experiments
* A comparison between NLU and NLG is missing. The methodology section duplicates the description of the regularizers; what is missing is the motivation for changes required to adapt the approach to NLG, followed by a clear presentation of that modified setting. Because of this, the methods section feels unsatisfactory.
* Despite introducing a regularizer for forgetting of pre-trained knowledge, the authors never directly evaluate forgetting. Overall, the first regularizer remains insufficiently probed.

### Fairness & Comparability of Evaluation
* Sourcing scores from original publications with “comparable” setups may not be a valid comparison if the experimental configurations are not identical.
* C4 is much larger and should arguably have been sampled or sliced to make comparisons fairer.

### Computational Considerations
* The computational costs of the different steps should be described (e.g., randomized SVD).

### Initialization Choices
* PiSSA seems like a questionable initialization strategy, it has been shown in [1] that PiSSA induces forgetting of pre-trained knowledge, as the adapters are initialized with the “core” knowledge. It seems BA-LoRA mitigates an issue amplified by this initialization.

### Related Work
* The related work section is too short and lacks explicit comparison to this work.

[1] MiLoRA: Harnessing Minor Singular Components for Parameter-Efficient LLM Finetuning. (2025) Hanqing Wang and Yixia Li and Shuo Wang and Guanhua Chen and Yun Chen

### Questions
1. Page 2 L86, can you provide a reference if this been shown before in LoRA?
2. How does the model manage to mitigate the Knowledge Drift? And how does PiSSA initialization relate to the Knowledge Drift?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes to use a bunch of existing regularization techniques combined with the existing LoRA variation to improve the fine-tuning performance. The paper evaluates the method on Llama2 7B and shows that it performs better than many existing methods.

It's good to know that combining all these regualrziation techniques together improves the performance but I don't quite understand the novelty of this paper.

### Strengths
The paper is well motivated.

The empirical results seem to be promising.

### Weaknesses
- The base model used for evaluation is extremely out of date, Llama2 is released in 2023, and I am not sure if the conclusion drawn is transferable to newer models.

- The method has 3 hyperparameter to tune, and the paper does not provide any guidance.

### Questions
- Can the author clarify what are the actual contributions of this paper?

- How are the 3 lambdas chosen?

- Why is BA-LoRA better than full model fine-tuning?

- The lines in Fig.2 are confusing in that: why the proposed method improves training performance. Isn't the main question to be addressed about generalization ability?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
The paper proposes to add three additional regularizers in LoRA fine-tuning: 1) distillation loss between pretrained and fine-tuned models to reduce output shift; 2) penalty on correlation between output classes or entropy of outputs to promote diversity; 3) penalty on high-frequency components in the output logit matrix.

### Strengths
The empirical advantage of combining the three regularizers is supported by improved empirical results.

### Weaknesses
The purpose of the proposed methods is not clear to me; see below.

### Questions
The purpose of the paper is unclear. At the beginning, the paper states that the goal is to evade/eliminate catastrophic inheritance, i.e. the undesired traits of a pretrained model (e.g. imbalances, biases, spurious correlations) being kept/exacerbated during fine-tuning, but the paper first applies a distillation-style regularizer to prevent shift from the pretrained model outputs. This appears to be self-contradictory. If the outputs of the pretrained model are undesired and need to be fixed during fine-tuning/post-training, then knowledge drift is what we desire. The case is similar in the other two subtasks: L90-92 attribute deteriorated performance due to the fine-tuning data quality, instead of the pretraining data. I think further explanation is critical for a consistent paper.

L85-86: It is claimed that LoRA exacerbates catastrophic inheritance; this makes sense, but I don't see any empirical evidence supporting that in the paper.

L155&L201: It is the outputs given pretraining or fine-tuning data? They have completely different implications. Also, how do you obtain the outputs of a un-finetuned encoder (e.g. Deberta) on NLU tasks? I assume that an extra linear projection head needs to be attached.

L159&L206: The reason for using T^2 is not really clear and needs further explanation.

L161: Should use \citep in this case.

L166: I'm confused by the purpose of this regularizer. Classes in different categories may not be orthogonal to each other, and some classes can be inherently correlated. Also, why is correlation a sign of a lack of diversity? If the goal is to promote diversity/avoid over-confidence, similar entropy-based regularizers like the one used in NLG (L214) should also work.

L177: Why do we expect the logit matrix to be low-rank? Is there any specific reason? How is the high-frequency component in the output logits related to the data noise? Particularly, if each of the sample certainly belongs to a class and D=N, the logit should be full-rank.

L180: What is "spurious intra-batch variations"? I assume that samples within each batch should be independently sampled.

L250: The selection of regularization weights appears to be quite arbitrary. Tuning three hyperparameters can be challenging and can be highly varied between tasks. Appendix C.2. actually demonstrates the issue, as the trend on MATH and GSM8K in Fig.4 (a) are different. Also, the change in accuracy in Fig. 4 is downplayed with the large difference in scale of the accuracy on the two tasks. I would suggest putting the lines on different figures. 

L1038 states that the regularizers steer the model along the creativity-robustness spectrum, but both MATH and GSM8K are reasoning tasks, and I doubt if MATH can be used as a proxy for creativity.

Sec 3.2.2: In addition to differences in training data, there are considerable differences in the model architecture and data size between RoBERTa and T5, hence I doubt if the difference in improvements can be attributed to noise resilience.

To summarize, I fear that there are too many claims and assumptions in this paper that are not sufficiently supported by concrete evidence. I would recommend the authors to   
1) Make their goals clear.  
2) Provide empirical evidence that the issues (as for this paper, forgetting, diversity, and noise) exist by dedicated evaluations, especially with LoRA, e.g., by comparing generation diversity in the original, fully fine-tuned, and LoRA fine-tuned models. Standard benchmarks on reasoning, understanding, or commonsense cannot directly support your claim.  
3) Show that each of the regularization terms can improve the issue, respectively.  
4) Demonstrate the synergy of the trio.

There are other papers discussing alleviating knowledge drift in LoRA fine-tuning, e.g.   
Smith, James Seale, et al. "Continual Diffusion: Continual Customization of Text-to-Image Diffusion with C-LoRA." Transactions on Machine Learning Research (2024).  
Chen, Haolin, and Philip N. Garner. "Bayesian parameter-efficient fine-tuning for overcoming catastrophic forgetting." IEEE/ACM Transactions on Audio, Speech, and Language Processing (2024).

### Soundness
1

### Presentation
1

### Contribution
2
