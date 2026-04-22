# Masks Can Be Distracting: On Context Comprehension in Diffusion Language Models

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 2, 4, 6

## Abstract
Masked Diffusion Language Models (MDLMs) have recently emerged as a promising alternative to Autoregressive Language Models (ARLMs), leveraging a denoising objective that, in principle, should enable more uniform context utilisation. In this work, we examine the context comprehension abilities of MDLMs and uncover two key limitations. First, despite their more global training objective, similarly to ARLMS, **MDLMs exhibit a strong locality bias**: performance is highly sensitive to the position of relevant information within the input, favouring local over distant context. Second, we show that appending a large number of **mask tokens--required for generation--can significantly degrade context comprehension**. Through systematic ablations, we find that these masks act as distractors, reducing the model's ability to process relevant information. To address this, we introduce a **mask-agnostic loss function** that encourages predictions to remain invariant to the number of appended masks. Fine-tuning with this objective substantially mitigates the distracting effect of masks, improving robustness of MDLMs. Overall, our findings reveal critical limitations of the current MDLM training paradigm and provide actionable insights for building diffusion-based language models with stronger context comprehension.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper uncovers masked diffusion language models exhibit strong locality biases and too-many mask tokens can degrades their performance. Accordingly, the author introduces mask-agnostic loss function to tackle the problems.

### Strengths
1. This paper uncovers an interesting bias in diffusion language models that MDLM prioritise information close to the prediction targets, i.e., masks.
2. This paper also finds the token accuracy decreases given more mask tokens.
3. The authors provide extensive experiments to showcase the two biases.
4. The author also introduce a  mask-agnostic objective and show its effectiveness in rectifying the effect of extra masks.

### Weaknesses
Despite the presented phenomenon, the author does not directly demonstrate how these phenomenon may lead to negative impact in practical scenarios.

Although the introduced mask-agnostic objective decrease the entropy and increases the accuracy of one-step prediction. It is unclear whether it harms multistep sampling and test time scaling effect?

### Questions
see above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents an empirical study examining the impact of mask tokens on the generation performance of masked diffusion language models (MDLMs). The researchers created a small synthetic dataset to explore two key areas: locality bias and inverse scaling with masks. Additionally, the paper introduces a mask-agnostic loss function for fine-tuning MDLMs. This function reweights the cross-entropy loss at each sampled step in the forward corruption process, taking into account the number of mask tokens. However, the reweighted loss did not demonstrate an improvement in generation performance compared to uniformly weighted loss.

### Strengths
This paper presents some evidence that aligns with existing intuitions. For instance, the unmasked tokens can reduce the uncertainty of nearby masks, and the more masks we have the less accurate the predictions would be due to higher uncertainty. It’s nice to see some concrete analysis to support these intuitions.

### Weaknesses
1. The paper's conclusions align with intuition: surrounding unmasked tokens likely provide more contextual information, aiding in the unmasking process. This phenomenon could be further explained by examining the role of attention in Transformer models. However, the analysis of locality bias would be more convincing if the variable lengths of sequences were normalized or fixed. This would help control for the difference between context length and the actual sequence lengths of samples, which is currently a confounding variable.

2. The authors should consider using a proportional number of masks, rather than a fixed absolute number, for each sample in their analysis of extra masks. This approach would be more effective as it would align the mask count with the sequence length of each sample.

3. The absence of synthetic dataset examples hinders my ability to grasp and validate the authors' rationale across tasks. Concrete illustrations are crucial for effective comprehension. Furthermore, it is imperative to prevent the tasks themselves from inadvertently introducing inductive biases.

4. There seems no improvement in training with the mask-agonistic loss.

### Questions
1. In Line 131-133, the authors stated that there are “8 relevant word tasks” and “2 distractor tasks”, but later they stated that the task suite consists of “16 tasks”. Please can you elaborate on this discrepancy?

2. In Line 134, what is “the governing rule” exactly?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper designs a series of experiments to study the context comprehension capabilities of masked diffusion LLMs. It identifies two weaknesses of these models: locality bias and inverse-scaling with masks. The authors then proposed a mask-agnostic loss function that trains the model prediction to be invariant with respect to the number of extra masks in the tail.

### Strengths
* This paper addresses an important topic---the context comprehension capability of emerging diffusion LLMs.  

* The authors took a scientific approach by designing thoughtful experiments and iteratively refining the hypothesis based on the results. For the study of locality bias, the authors first establish a hypothesis of recency bias based, then disentangled it from the left-to-right order by new experiments that moves the mask position. A similar approach has been taken towards the study of the effect of extra masks. 

* The proposed mask-agnostic loss seems effective in reducing the model's sensitively to the number of extra masks in the tail.

### Weaknesses
* The fact that diffusion LLMs exhibit a locality bias has been similarly discovered in a prior work (DiffuCoder), diminishing the importance of the discovery.

* Although the paper takes a scientific approach in experiment design and hypothesis checking, it offers limited insights on the mechanisms. The experiments are mostly designed to test the whether the model performance is impacted by a factor rather than exposing why they are impacted. 

* The study relies on pretrained OSS diffusion LLMs (llada and dream), both having limited context length. Therefore, the study carried out in this work rely on handcrafted tasks that don't necessarily reflect the true capability of the model in real natural language tasks.

* Although the proposed mask-agnostic loss led to more robust predictions in base model, it is shown to result in a slight decrease in performance for the instruct model and no fixes have been identified, limiting the application of such losses.

* A few claims should be either justified with evidence or weakened: 
  - L49 "inverse scaling law with the number of masks": this only seems to hold for LLaDA but not Dream models. Also, section 5.4 shows that under standard iterative unmasking (often the approach used in practice), there is no degradation in performance with extra number of masks.
  - L284 "This is because the model must filter relevant from irrelevant information over a longer context, and extra masks may disrupt its attention allocation": There should be at least some evidence from looking at the attention scores to justify this.

### Questions
*  In practice, we don't unmask answers in one step as in section 5.3. Does the results in section 5.4 mean that practically we don't need to care about the number of extra masks?

* Missing references on masked diffusion models (these two are concurrent work with Sahoo et al. 2024): 

Shi, J., Han, K., Wang, Z., Doucet, A., & Titsias, M. (2024). Simplified and generalized masked diffusion for discrete data. Advances in neural information processing systems, 37, 103131-103167.

Ou, J., Nie, S., Xue, K., Zhu, F., Sun, J., Li, Z., & Li, C. (2024). Your absorbing discrete diffusion secretly models the conditional distributions of clean data. arXiv preprint arXiv:2406.03736.

### Soundness
2

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
This paper investigates context comprehension in Masked Diffusion Language Models (MDLMs) and identifies two critical limitations. First, despite MDLMs’ global denoising objective, they exhibit a strong locality bias: performance depends heavily on the proximity of relevant information to the masked token (prioritizing nearby context over distant context), unlike ARLMs which often show a U-shaped “lost-in-the-middle” pattern. 
Second, appending extra mask tokens (required for generation) acts as distractors, degrading context comprehension - an effect that worsens with longer contexts and is specific to mask tokens. To address this, the authors propose a mask-agnostic loss function that combines cross-entropy for accurate prediction and total variational distance to enforce invariance to mask count. Fine-tuning MDLMs (LLaDA, Dream) with this loss reduces mask-induced distraction and mitigates locality bias. The paper also provides guidelines for MDLM evaluation, emphasizing explicit reporting of mask configurations.

### Strengths
1. The work demonstrates solid contribution by uncovering understudied limitations of MDLMs: locality bias and mask distraction that were not systematically explored in prior masked dLLM research. This fills a critical gap, as MDLMs are often assumed to leverage global context more uniformly than ARLMs. The mask-agnostic loss offers a practical, architecture-agnostic fix for MDLM robustness, and the evaluation guidelines address reproducibility issues in existing MDLM benchmarks (where mask configurations are often omitted).
2. The experiments are well-controlled (e.g., isolating mask vs. repeated token effects, using single-token answers to avoid decoding confounds) and validated across models (LLaDA, Dream) and datasets. 
3. Good presentation: the paper structures analyses logically (locality bias → mask distraction → solution) and uses intuitive figures (e.g., Figure 3 for mask inverse scaling, Figure 8 for MA loss efficacy) to communicate complex patterns.

### Weaknesses
1. the analysis is limited to two open-source MDLMs (LLaDA-8B, Dream-7B) with opaque pre-training details (e.g., exact datasets, masking schedules). This makes it hard to disentangle model-specific quirks (e.g., Dream’s ARLM initialization) from general MDLM properties. Testing additional controlled pre-trained variants would strengthen generalizability.
2. the few-shot tasks used to measure locality bias are relatively simple (e.g., choosing adjectives); testing on more complex reasoning tasks (e.g., multi-hop QA requiring integration of distant facts) would better validate MDLMs’ context comprehension limits.

### Questions
1. Could you test the MA loss on additional MDLMs (e.g., DiffuCoder) or MDLMs with transparent pre-training pipelines? This would help confirm if the observed locality bias and mask distraction are general MDLM properties rather than LLaDA/Dream-specific.
2. How does the MA loss perform on MDLMs fine-tuned for longer context lengths (e.g., >4096 tokens)? Since mask distraction worsens with context length, evaluating on extended-context tasks would clarify its scalability.
3. For Dream models (initialized from ARLMs), you note reduced mask sensitivity. Could you analyze if this is due to residual ARLM inductive biases (e.g., left-to-right attention)? For example, do Dream’s attention weights differ from LLaDA’s when processing distant context?
4. The few-shot tasks focus on simple pattern recognition—could you extend experiments to complex reasoning tasks (e.g., multi-hop QA, logical deduction) to verify if the MA loss improves MDLMs’ ability to integrate distant, relevant context?

### Soundness
3

### Presentation
3

### Contribution
2
