# Mitigating Fine-tuning Risks in LLMs via Safety-Aware Probing Optimization

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
The significant progress of large language models (LLMs) has led to remarkable achievements across numerous applications. However, their ability to generate harmful content has sparked substantial safety concerns. Despite the implementation of safety alignment techniques during the pre-training phase, recent research indicates that fine-tuning LLMs on adversarial or even benign data can inadvertently compromise their safety. In this paper, we re-examine the fundamental issue of why fine-tuning on non-harmful data still results in safety degradation. We introduce a safety-aware probing (SAP) optimization framework designed to mitigate the safety risks of fine-tuning LLMs. Specifically, SAP incorporates a safety-aware probe into the gradient propagation process, mitigating the model's risk of safety degradation by identifying potential pitfalls in gradient directions, thereby enhancing task-specific performance while successfully preserving model safety. Our extensive experimental results demonstrate that SAP effectively reduces harmfulness below the original fine-tuned model and achieves comparable test loss to standard fine-tuning methods. Our code is available in the supplementary materials.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper solves the LLM safety degradation caused by fine-tuning. It hypothesizes that usefulness-critical and safety-critical gradient directions are entangled and proposes Safety-Aware Probing (SAP).  At each step, SAP estimates a safety-critical direction using a contrastive safety loss, constructs a “harmful” update, and then learn a small hidden-state probe that maximizes a safe-useful loss, encouraging downstream weight updates to avoid harmful regions. Across three 7b models on benign and adversarial tasks, SAP reduces harmfulness scores while keeping task loss/metrics near SFT. Additional experiments demonstrate the effectiveness of different settings. However, the scalability is not validated and remains questionable.

### Strengths
1. The proposed SAP is well motivated. The hypothesis is validated by experiments.
2. Experiments over 3 models and several datasets in different settings demonstrate promising performance.
3. It provides sensible ablations (e.g., which layers to probe and learning-rate sensitivity) and report costs (time and memory).

### Weaknesses
1. The first claimed contribution of validating the hypothesis has been explored in prior works, such as [1, 2]. The cosine similarity approach is similar to SafeLora [1].

2. The authors claim in the introduction that their work “*has better scalability since it can be incorporated into various fine-tuning paradigms rather than being limited to LoRA.*”  However, this claim is not supported by experimental evidence, such as larger models or fully fine-tuning. All experiments are conducted on 7B models with LoRA.  Furthermore, this claim is questionable, as SAP requires extra gradient estimation, which is related to the number of trainable parameters.

3. The time overhead of SAP is non-trivial, approximately 2.5 times that of all baselines. While Appendix B.5 includes a comparison with LISA, the discussion does not adequately address the overhead relative to other baselines. Considering the performance gain, the practical value of applying this method remains questionable.

4. Figure 1 lacks clarity and should be improved to better convey the procedure of SAP.


**Reference**

[1]  Safelora: The silver lining of reducing safety risks when finetuning large language models. NeurIPS. 2024

[2] Safe Delta: Consistently Preserving Safety when Fine-Tuning LLMs on Diverse Datasets. ICML. 2025

### Questions
1. What is the time cost for SAP when applying to larger models and fully fine-tuning settings?
2. What is the portion or number of safety examples for baselines, such as safeinstr?
3. Why not use fewer, smaller probe layers instead of probing 10 layers, given the high time cost?

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
3

### Summary
This paper shows that large language models can lose their safety even when fine-tuned on harmless data, because the gradients that improve task performance are often entangled with those that reduce safety. To address this, the authors propose Safety-Aware Probing (SAP), a training method that adds a small hidden-state probe to steer optimization away from harmful directions while still improving task performance. SAP does not require changing the dataset or model architecture and works across different fine-tuning setups.

### Strengths
The experiments are extensive and well designed, have evaluations on three different models, three instruction-following datasets, five reasoning benchmarks, and poisoned, adversarial fine-tuning settings. 
The paper is also clearly written and well-organized.

### Weaknesses
* Gradient analysis and evaluation are somewhat limited. The paper shows cosine similarity between usefulness and safety gradients, but does not fully explain why the directions align or provide deeper theoretical insight. The Harmful score relies on a single moderation model, with no additional metrics such as jailbreak success rate, or LLM-as-judge evaluation. 
* Cost analysis is needed. SAP increases training time by 2x~3x, but the paper briefly labels this as acceptable without discussing practical implications or scalability. 
* Limited discussion of adaptive attacks. The adversarial fine-tuning experiment does not consider attackers aware of SAP.

### Questions
NA

### Soundness
2

### Presentation
2

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
The paper introduces Safety-Aware Probing (SAP), a lightweight optimization that inserts a small safety-aware probe into hidden states during gradient propagation to steer updates away from harmful directions while preserving task utility. SAP is motivated by an observed entanglement between safety-critical and usefulness-critical gradient directions; it maximizes a safe-useful objective to find a probe  that biases each update toward safer regions.

### Strengths
- The proposed method is simple yet principled: it treats safety as gradient-space steering and remains broadly compatible with standard fine-tuning.
- Empirically, it improves utility while reducing harmfulness, increases robustness to poisoning and adversarial fine-tuning, and composes with other defenses, making it practical for deployment.

### Weaknesses
The proposed method introduces too many hyperparameters (e.g., α, β, ϵ, probe layers), which increases tuning complexity and reduces reproducibility.

### Questions
Q1. Did you mean the following?

“Our experiments show that SAP achieves better useful loss while significantly **decreasing model safety”**

→ “Our experiments show that SAP achieves better useful loss while significantly **improving model safety**”

Q2. Could you clarify the captions for Figures 2 and 3?

- Figure 2 does not specify which harmful or useful
datasets were used.
- Figure 3 does not clarify the definition of the useful-critical notation nor specify which harmful dataset was used.

Q3. Are $\alpha$, $\beta$, $\epsilon$, and probe layers the same for every dataset?

Q4. Is there a reason for choosing 2,000 examples for $D_{useful}$ and 50 for $D_{safe/harmful}$?

Q5. Aren’t these models instruction-tuned models, such as Llama-7B-Chat?

Q6. Why are the Booster and Vaccine baselines not included in the results after Table 1?

Formatting issue:

- The repeated inclusion of the Alpaca dataset in Table 2 is redundant.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the safety alignment problem for LLMs. Specifically, it proposes a method to defend against malicious fine-tuning samples while keeping the model's utility score. What lies in the core the proposed method is to find a small parameter perturbation which discourages moving along harmful gradient direction when optimizing for the utility loss. The method is able to mitigate harmful gradient direction in utility loss gradient while not affecting the utility loss too much. Experimental results demonstrate the effectiveness of the proposed algorithm.

### Strengths
1. The paper addresses an important problem in LLM alignment.
2. The paper is well-written and easy to follow.
3. This work offers some valuable insight by showing the correlation between the descend of the utility loss and that of the harmful loss. The proposed method is somewhat intuitive and easy to implement. The reported experimental result is good.

### Weaknesses
1. Insights of how and why the proposed method works: Though the construction of $L_{su}$ is somewhat intuitive, it is still unclear how and why the algorithm works well. Specifically, how can the algorithm achieve lower utility loss than the full SFT on utility dataset, while being biased constantly (from the perturbation) in its utility optimization process? If the perturbation is supposed to be very small, how can it lead to substantial decrease in harmfulness?

2. Lacking critical baselines: it might be beneficial for the author to compare with simply adding the safety data (used in the proposed algorithm to compute negative harmful direction) into the utility dataset. The baseline can be interpreted as simply mixing the utility gradient with the negative harmful direction, which is very comparable to the proposed algorithm. This might lead to further insight into the performance of the algorithm.

As a result, I am overall hesitant to give an accept suggestion. However, I am happy to reconsider my recommendation if they are adequately addressed.

### Questions
1. In the proposed algorithm, is perturbation applied to the model parameter each iteration? If not, will it be a good/bad idea to apply the perturbation?

2. How will the algorithm perform under no malicious data? I am considering a scenario where this method is just applied to enhance safety in normal utility fine-tuning process.

### Soundness
2

### Presentation
3

### Contribution
2
