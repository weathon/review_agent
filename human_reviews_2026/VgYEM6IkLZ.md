# REPO: Detoxifying LLMs via Representation Erasure-based Preference Optimization

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 6, 2

## Abstract
Large language models (LLMs) trained on web-scale data can produce toxic outputs, raising concerns for safe deployment. Prior defenses, based on applications of DPO, NPO, and similar algorithms, reduce the likelihood of harmful continuations, but not robustly so: they are vulnerable to adversarial prompting and easily undone by fine-tuning–based relearning attacks. Indeed, research has shown that these edits to the model are superficial: linear probing reveals that harmful “directions” remain present in representations. Motivated by these findings, we propose Representation Erasure-based Preference Optimization method (REPO), which builds on SURE (Sepahvand et al., 2025), an unlearning algorithm originally developed for classification. Our core strategy is to preserve the representations of benign (safe, nontoxic) generations while forcing the representations of toxic generations to converge toward their benign counterparts. This alignment is achieved
through a coupled objective, which combines a retain loss on non-toxic samples with a domain-adversarial loss on both toxic and non-toxic samples, enforced by a gradient reversal layer. Comprehensive evaluations show that REPO not only significantly reduces in-distribution and out-of-distribution toxicity compared to baselines like DPO, NPO, and RMU, but also achieves best-in-class robustness against sophisticated attacks, including relearning on forget and retain samples, and adversarial prompt injection, via an enhanced variant of GCG.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces REPO, an unlearning method aimed at detoxifying LLMs by intervening directly on their internal hidden representations. REPO adapts the principle of SURE (Sepahvand et al., 2025) to the preference optimization setting. Its core strategy is to preserve the representations of benign (nontoxic) generations while forcing the representations of toxic (forget) generations to converge toward their benign counterparts.

### Strengths
REPO demonstrates the effectiveness in detoxification while preserving general language capabilities. It achieves the lowest toxicity score on in-distribution and out-of-distirbution evaluations, substantially outperforming steering-based methods and other fine-tune based methods. 

Prior output-level alignment techniques (like DPO and NPO) are vulnerable to adversarial prompting and relearning attacks, as linear probing suggests that harmful “directions” remain present in the model's representations. REPO achieves strong robustness against several advanced adversarial attacks, such as relearning, orthogonalization, and GCG attacks. The robustness is also evidenced by the representational drift  between the unlearned and reference models’ hidden states.

### Weaknesses
As the authors acknowledge in the Introduction, and Appendix B, REPO appears highly similar to SURE (and DANN), thus suggesting a lack of significant novelty. Though the authors say that they adopted SELU for preference optimization, the core components, specifically the retain loss and the adversarial loss, are directly adopted from SURE (and DANN), and therefore do not constitute a unique methodological innovation developed specifically for preference optimization. 

The most significant difference is that SURE calculates the domain loss using the forget set and a validation set (non-trained set), whereas REPO calculates the domain loss using the toxic continuation and the non-toxic continuation. I don't understand why this achieves detoxification. In the case of SURE, I understand that unlearning is achieved because by bringing the hidden states of the forget set inputs closer to the hidden states of the non-trained set inputs, the model behaves as if the forget set inputs were never learned. On the other hand, for REPO, I don't understand why toxic text is unlearned by making the hidden states of the toxic continuation similar to the hidden states of the non-toxic continuation, and I question whether the adversarial loss is truly effective for the case of REPO.

As noted above, given the uncertainty regarding the effectiveness of the adversarial loss, I would like to see the results using only the retain loss. However, such an analysis is not conducted in the ablation study. Instead, the analysis focuses merely on several variants of the adversarial loss and the comparison against prior methods, which are not an ablation study.

### Questions
As mentioned in weakness, could you explain why the adversarial loss using the toxic continuation and the non-toxic continuation achieves unlearning (or preference optimization) step by step? Also, I would like to see the results using only the retain loss.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposed REPO, a novel method for detoxifying large language models by removing toxicity at the representation level during unlearning. Building on the SURE method, REPO achieves detoxification by aligning the LLM's internal representations of non-toxic and toxic continuations of the same prompt, so they appear indistinguishable to a toxicity classifier. Meanwhile, the toxicity classifier is trained adversarially to further distinguish the aligned representations produced by the LLM. The authors show that the GPT-2 and Gemma-2B models unlearned using REPO have lower toxicity compared to fine-tuning based and steering based detoxification. Moreover, the REPO-trained models are also more robust to a variety of relearning-attacks, GCG, and orthogonalization.

### Strengths
1. The paper is well written and easy to follow. While representation erasure–based unlearning has already been used by [1] for a classification task, this paper effectively extends it to text generation by adding a constraint that preserves the model’s generative utility.
2. The performance of REPO is generally robust. Compared to fine-tuning based and representation steering-based detoxification, models trained using REPO have lower toxicity both before and after variety of attacks.
3. The detailed analysis of REPO's effects on toxicity representation at layer (Figure 4, 5, 6) and neuron level (Figure 9) are convincing. 

References:
[1] Sepahvand, Nazanin Mohammadi, et al. "Selective unlearning via representation erasure using domain adversarial training." The Thirteenth International Conference on Learning Representations. 2025.

### Weaknesses
1. Only tested on two small models (GPT-2 and Gemma-2B). Although small and lightweight models allow the authors to conduct more detailed mechanistic analysis, knowing whether the performance of REPO scales to larger and more recent LLMs is still important for validating the method's practical value.
2. REPO outperforms the fine-tuning based and steering based detoxification, but the adversarial training of its discriminator module also introduce additional computational overhead, which should be discussed.
3. The paper evaluates the utility of unlearned models by their perplexity on the WikiText dataset. To support the claim that REPO preserves general utility, performance on benchmarks such as HellaSwag, WinoGrande, OpenBookQA, or MMLU would be more informative.
4. The parameterization of the discriminator is underexplored. Using a linear regressor implicitly assumes toxicity is linearly represented, yet prior work [2] suggests toxicity may reside in a multi-dimensional subspace. It would be valuable to test whether a non-linear discriminator improves performance.
5. (Minor) The prompts along the x-axis in Figures 4–6 are difficult to read. Including the full prompts in the captions and highlighting toxic tokens in the heatmaps would improve clarity. Also, consider removing the extraneous “$\dot{G}$” characters from tokens.
6. (Minor) Figure 1 should be moved closer to Section 3 for better readability of the method description ($G_d, G_y$, etc.)

Reference:
[2] Uppaal, Rheeya, et al. "Model editing as a robust and denoised variant of dpo: A case study on toxicity." arXiv preprint arXiv:2405.13967 (2024).

### Questions
1. Have the authors tested REPO on removing representations of other undesirable features, such as bias and unsafe knowledge, which are more multi-faceted compared to toxicity? Does the performance of REPO generalize?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses the detoxification of large language models (LLMs) through the lens of machine unlearning. The proposed approach aims to make toxic and non-toxic representations indistinguishable by applying an adversarial loss at a particular layer of the model. The method uses two components: 

- $L_{retain}$: to preserve model performance on non-toxic data. 
- $L_{adv}$: to enforce indistinguishability between toxic and non-toxic representations. 

The idea is positioned as a generalization of unlearning methods to the alignment / detoxification setting.

### Strengths
- Provides an interesting attempt to connect recent machine unlearning approaches with alignment objectives, offering a conceptual link between the two, though not entirely novel. 
- Addresses a timely and practically important goal "mitigating toxic behavior in LLMs".

### Weaknesses
1. The central claim "making toxic and non-toxic representations indistinguishable" requires more motivations. It is unclear why this would correspond to unlearning or guarantee behavioral safety. 

2. The adversarial formulation is confusing. Minimizing the binary cross-entropy loss $L_{adv}$  appears to promote discrimination rather than indistinguishability. The paper should explicitly clarify that $G_d$​ (the discriminator) is trained to maximize $L_{adv}$, while $G_f$ (the forget model) is trained to minimize it, following a standard minimax setup.

3. There is no formal definition of the datasets $D_r, D_f$.

4. The role of $L_{retain}$ is vague: it is unclear whether it preserves token-wise predictions (during gernation of $x_r$) or sequence-level consistency (next token generation after $x_p, x_r$). If it compares only next token generation after $x_p, x_r$, the objective may not preserve semantics effectively.

5. It seems the idea is applied to the block M (typically the last lyer). However, the model modification (training) scope (only last layer vs. full model) is not clearly specified, yet this choice crucially affects both unlearning strength and computational cost. 

6. Section 2 is redundant and reads like an extended introduction; this space could instead clarify mathematical definitions and literature survey.

7. The paper insufficiently engages with prior work on alignment methods such as DPO or PPO. Also, it would be better to describe the key ideo of unlearning method SURE. Without situating the method among these, its novelty and relevance to the alignment field remain unclear.

### Questions
see weaknesses

### Soundness
2

### Presentation
1

### Contribution
2
