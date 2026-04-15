# Accurate Retraining-free Pruning for Pretrained Encoder-based Language Models

- Decision: Accept (poster)
- Scores: 8, 6, 8, 5

## Abstract
Given a pretrained encoder-based language model, how can we accurately compress it without retraining? Retraining-free structured pruning algorithms are crucial in pretrained language model compression due to their significantly reduced pruning cost and capability to prune large language models. However, existing retraining-free algorithms encounter severe accuracy degradation, as they fail to handle pruning errors, especially at high compression rates. In this paper, we propose KPrune (Knowledge-preserving pruning), an accurate retraining-free structured pruning algorithm for pretrained encoder-based language models.
KPrune focuses on preserving the useful knowledge of the pretrained model to minimize pruning errors through a carefully designed iterative pruning process composed of knowledge measurement, knowledge-preserving mask search, and knowledge-preserving weight-tuning. As a result, KPrune shows significant accuracy improvements up to 58.02%p higher F1 score compared to existing retraining-free pruning algorithms under a high compression rate of 80% on the SQuAD benchmark without any retraining process.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper presents a method to prune an encoder-only pre-trained transformer-based language model without retraining it. The method is based on the notion of a global importance score for each attention head or feed-forward neuron, considering both the prediction loss (KL divergence between pruned and unpruned models) and the representational loss (L2 distance between weights at each layer before and after pruning). It uses a layerwise pruning of attention heads and feed-forward neurons, starting with the bottom sublayer and working up. In each layer, three steps are performed: 1) Global importance computation: Compute the global importance score for each attention head or neuron. 2) Mask search: Search for a mask for each attention head or neuron, considering both prediction and representational knowledge. 3) Linear projection layer tuning: Tune the linear projection layers for the heads or neurons that are not pruned using a linear solver on a small amount of data. The proposed approach achieves an improvement between 8.5-58.0%p in F1 across tasks/models relative to a training-free approach in Kwon et al. (2022). Compared to the unpruned baseline, the proposed approach achieves an inference speedup of 2.65x, while the method of Kwon et al. gives at most a 1.87x speedup. It achieves comparable accuracy to retraining approaches (such as DynaBERT) with significantly less computation.

### Strengths
*  Presents a new method for pruning a pre-trained encoder-only transformer language model without retraining.

*  The proposed approach achieves F1 score improvements of 8.5% to 58% over existing training-free pruning approaches, while achieving similar performance to retraining approaches at a much lower computational cost.

* Reports an ablation study that shows the relative importance of each component of the approach, revealing that layerwise pruning and weight tuning are critical.

### Weaknesses
* Approach is restricted to encoder only models but many of the current LLM approaches are decoder-based.
* Some aspects of the paper are unclear. See below in questions.

Update after author rebuttal:
* The authors have clarified most of my questions in their rebuttal.

### Questions
* Sec 1 : What does the "%p" notation mean?
* Sec 2.2: "Once the mask variables are established after mask search, pruning of attention heads and neurons whose mask variables are zero does not affect the inference results." What does 'established' mean in this context? 
* Equation 7: The LHS K_{rep,l}(X_{\tau,l},X_{\rep,l};0)  should be a function of i. 
* Sec 3.3: "We estimate the importance score not only in the target sublayer but also in the sublayers above the target sublayer" Why is it necessary to estimate the importance score for sublayers above the target sublayer considering that only the heads/neurons in the target sublayer are pruned when a given layer is considered i.e. From the description/Figure 2, it seems like Algo 1 is run for each sub-layer.
* The algorithm performs sublayer-wise pruning from the bottom to the top sub-layer. If we refer to this as a single-pass, is it ever necessary in practice to do a second pass to achieve the desired number of FLOPs?

### Soundness
4 excellent

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces Kprune (Knowledge-preserving pruning), a novel retraining-free structured pruning algorithm designed for pretrained encoder-based language models. The key challenge addressed by this work is the accurate compression of such models without requiring retraining. Existing retraining-free algorithms suffer from significant accuracy degradation, particularly at high compression rates, due to their inability to handle pruning errors effectively.

Kprune employs an iterative pruning process that focuses on preserving the valuable knowledge contained within the pretrained model. This process includes three main steps: knowledge measurement, knowledge-preserving mask search, and knowledge-preserving weight-tuning. By implementing these steps, Kprune achieves remarkable results, with accuracy improvements of up to 58.02% higher F1 score compared to existing retraining-free pruning algorithms. These improvements are observed under a high compression rate of 80% when tested on the SQuAD benchmark, all without the need for any retraining.

### Strengths
1. Kprune presents a significant advancement in the field of pretrained language model compression by demonstrating the feasibility of high compression rates without compromising model accuracy. The approach's success lies in its ability to preserve the essential knowledge within the model, ultimately leading to impressive gains in performance compared to existing retraining-free pruning methods.
2. The equations are clearly described.

### Weaknesses
1. The writing of the introduction section seems unreasonable. I think some challenges and other content should be written in the introduction instead of the method.
2. In the main experiment table (Table 1), there are not enough baselines for comparison.

### Questions
Please refer to Weakness

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents KPrune, a new retraining-free structured pruning method for compressing task-specific models while retaining their knowledge. Unlike prior techniques, KPrune considers both the overall loss impact and the effect on representation preservation when deciding which units to prune. It does this by analyzing the expected output of each head/neuron. As a result, KPrune requires minimal pruning time compared to retraining methods and outperforms previous retraining-free approaches. This work differs from Kwon et al. (2022) in two main ways: 1) it incorporates layer-wise representation loss when measuring unit importance, and 2) it recovers weights by tuning the output matrix with linear solvers. These innovations enhance pruning quality with minimal overhead versus retraining. In summary, KPrune efficiently compresses models while maintaining performance by carefully accounting for loss impact and representation preservation when pruning.

### Strengths
- The paper is well written and easy to follow!
- KPrune is the first retraining-free pruning method to incorporate layer-wise representation preservation loss and KL loss on outputs, techniques commonly used in retraining-based pruning.
- KPrune shows considerable performance improvements over previous retraining-free methods in high sparsity regimes (>80%). This demonstrates its ability to maintain model quality even with extreme compression rates.

### Weaknesses
- While not a direct comparison, it would be interesting to compare KPrune's performance to stronger training-based methods like CoFiPruning. KPrune's  major advantage is requiring much less time for pruning, though training-based approaches may achieve better end results. 
- Since BERT models are relatively small in scale nowadays, reducing training compute is less impactful compared to large language models. It would be interesting to evaluate if KPrune can effectively scale up to larger and more powerful models where saving compute will be more significant.

### Questions
I don't have further questions for the paper.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a new algorithm called Kprune that can significantly improve the accuracy of pretrained language models while compressing them without the need for retraining. The authors explain that while pruning is a common technique for compressing deep neural networks, it is often difficult to apply to pretrained language models due to their complex architecture and the difficulty of preserving their useful knowledge during the pruning process. Kprune addresses these challenges by using an iterative pruning process that selectively removes neurons from the model based on their importance to the overall performance of the model. The authors evaluate Kprune on several benchmark datasets and compare its performance to existing retraining-free pruning algorithms.

### Strengths
- The paper introduces a new algorithm, Kprune, that can significantly improve the accuracy of pretrained language models while compressing them without the need for retraining. 
- The authors provide a detailed explanation of the iterative pruning process used in Kprune and how it helps preserve the useful knowledge of the pretrained model. 
- The authors evaluate Kprune on several benchmark datasets and compare its performance to existing retraining-free pruning algorithms, providing evidence of its effectiveness. 
- The paper could have practical applications in the field of natural language processing, where the ability to compress pretrained language models without sacrificing accuracy is highly desirable.

### Weaknesses
- The paper focuses specifically on encoder-based language models, so it may not be applicable to other types of language models. 
- The authors do not provide a detailed analysis of the computational resources required to implement Kprune, which could be a potential limitation for some applications. 
- The paper does not apply to large language models which are commonly used now, experiments on larger models are expected.

### Questions
Can the proposed technique applied to LLMs?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
