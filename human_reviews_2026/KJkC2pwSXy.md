# Bootstrapping Zero-Shot Reasoning in Small Language Models via Advantage-Weighted Self-Distillation

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Small language models (0.5B–3B) typically lack mathematical reasoning ability, often scoring near 0% on tasks they can solve with few-shot demonstrations. Existing approaches rely on thousands of supervised chain-of-thought (CoT) traces or complex multi-round self-distillation pipelines. We introduce Advantage-Weighted Direct Preference Optimization (AWDPO), a lightweight alignment method that bridges the gap between few-shot and zero-shot reasoning. Unlike prior approaches, AWDPO formulates training as a single-pass preference optimization objective that aligns a model’s zero-shot distribution with its own few-shot behavior. Our loss combines an advantage-weighted preference term with a dynamic MLE anchor, yielding stable training and implicit trust-region regularization.

On GSM8K, AWDPO transforms Qwen-2.5 base models (0.5B–3B) from 0% to 39%–77% accuracy, recovering over 90% of a supervised fine-tune that uses 7,473 CoT traces — a 1,750× reduction in CoT data. The method generalizes to SVAMP, ASDiv, and MATH500, where AWDPO recovers up to 90% of supervised CoT performance. Our analysis shows that AWDPO is equivalent to a Kullback-Leibler (KL)-constrained policy improvement step under projected DPO. These results demonstrate that small base models can substantially improve their mathematical reasoning ability from minimal supervision, providing a principled and data-efficient alternative to supervised CoT or Reinforcement Learning (RL)-based methods for mathematical reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Advantage-Weighted Direct Preference Optimization (AWDPO), a data-efficient self-distillation method that bootstraps zero-shot reasoning in small language models by training them to prefer their own advantaged, few-shot-prompted outputs over their zero-shot ones.

### Strengths
- The AWDPO method proposed in this paper is intuitive and elegant. I particularly appreciate the approach of using the well-performing rule-based reward from RLVR as a weight for the DPO logits. The final form of the preference loss (Line 198) resembles the Policy Gradient in REINFORCE. I wonder if the authors have analyzed the potential connection between AWDPO and Policy Gradient?

- AWDPO performs well within the experimental scope covered in the paper. The DPO LoRA training, using self-distillation data collected from just 4-shot CoT golden responses, achieves results that approach the performance of full-parameter SFT trained on 7K+ golden responses.

### Weaknesses
- Although the authors acknowledge this limitation, I must point out that the experimental scope of the paper is very small. It only involves three small models from a single series (Qwen2.5) and are trained on only one data source (GSM8k). While I understand that expanding the scope would consume more computational resources, having both the number of model series and data sources limited to one weakens my confidence in the general usability of AWDPO.

- I would like to know in what scenarios we should use AWDPO to train our Base LLM. 
    - If the objective is to save computational resources: AWDPO requires DPO LoRA training. For small models, how much memory and computational power does this save compared to full-parameter SFT? If the savings are not substantial, why wouldn't one just use full-parameter SFT, which also yields better performance?
    - If the objective is to address the difficulty of obtaining golden responses: The currently more popular "RL-Zero" approach also does not require golden responses. How does its performance compare to AWDPO? Of course, the authors might argue that AWDPO only requires a very small number of queries. I acknowledge this is an advantage, but it doesn't seem to be a significant one, as queries are relatively easy to obtain.

### Questions
See Strengths and  Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes Advantage-Weighted Direct Preference Optimization (AWDPO), a data-efficient method for transferring few-shot chain-of-thought reasoning into the zero-shot behavior of small language models (0.5B–3B) without requiring large teacher models or reinforcement learning. The key idea is to treat the model’s own few-shot outputs as a pseudo-teacher and compare them against its zero-shot outputs, weighting this preference by the advantage (difference in correctness-based reward), thereby enabling learning from both successful and unsuccessful reasoning attempts. Experimental results show that AWDPO achieves 39%–78% accuracy on GSM8K for 0.5B–3B Qwen-2.5 base models, recovering ~90% of the performance of a fully supervised fine-tune while using only four chain-of-thought exemplars. a 1,750× reduction in CoT data. The method generalizes to SVAMP, ASDiv, and MATH-500, where AWDPO recovers up to 90% of supervised CoT performance.

### Strengths
The paper focuses on improving zero-shot reasoning in small LLMs, which is a timely and practically important problem. The proposed advantage-weighted self-distillation formulation is conceptually clean: compare few-shot vs. zero-shot responses from the same model and update toward the one with higher reward. The proposed framework does not need the additional teacher model which significantly improves its efficiency. The paper demonstrates that low-rank adaptation (LoRA) inherently constrains policy drift, and a dynamic supervised anchor on correct few-shot examples further stabilizes training. With this, the additional regularization components such as KL divergence penalties, trust region methods or additional models etc are not needed. This makes the whole training pipeline much more lightweight.

### Weaknesses
The evaluation is very limited (only to GSM8K), so more comprehensive evaluations are definitely needed to validate the proposed methods. 

AWDPO performance is still sensitive to the selection and phrasing of the few-shot prompts used to seed the pseudo-teacher. The paper would benefit from discussing guidance or robustness strategies for exemplar choice.

### Questions
How can we handle scenarios where the few-shot prompts are not representative? What's the underlying guidance for preparing the calibration datasets for self-distillation?

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
This paper addresses the **lack of mathematical reasoning ability in small language models (0.5B–3B parameters)**, which typically perform poorly on reasoning benchmarks like GSM8K in zero-shot settings. The authors propose **Advantage-Weighted Direct Preference Optimization (AWDPO)** — a lightweight, single-pass self-distillation method that aligns a model’s zero-shot behavior with its own few-shot reasoning outputs. AWDPO computes a **preference loss weighted by the advantage (performance gap)** between few-shot and zero-shot responses, and combines it with a dynamically scaled **maximum likelihood (MLE) anchor** on correct examples. Experiments show AWDPO improves Qwen-2.5 models’ GSM8K accuracy from 0% to up to 77%, recovering over 90% of supervised fine-tuning performance with 1/1750th of the CoT data.

### Strengths
### **1. Data Efficiency**

AWDPO achieves performance gains using only **four chain-of-thought exemplars and answer-only supervision**, representing a **1,750× reduction in labeled CoT data** compared to fully supervised fine-tuning.

### **2. Clear Empirical Validation**

The authors provide **empirical validation** for some of the core claims of the paper, namely advantage weighting and dynamic loss-balancing through ablation studies.

### Weaknesses
### **1. Reliance on the Model’s Own Few-Shot Quality**

AWDPO assumes that the model’s few-shot responses are good enough to act as a “pseudo-teacher.” If the base model’s few-shot reasoning is poor, the entire self-distillation loop may propagate low-quality reasoning. The paper doesn’t explore how AWDPO behaves when the few-shot teacher is unreliable — e.g., for domains or smaller models where even few-shot reasoning fails.

### **2. Oversimplified Loss-Balancing Mechanism**

The online loss-balancing rule may be too heuristic and coarse-grained to ensure true gradient-scale equilibrium. It lacks theoretical justification or empirical exploration across task types, and may cause instability or suboptimal weighting when the two losses evolve at different rates (e.g. in a new reasoning task). The authors acknowledge dynamic scaling helps avoid manual tuning, but a more principled or adaptive method (e.g., gradient norm balancing) would strengthen the claim. 

### **3. Inadequate Non-math Reasoning Results**

Only evaluation on non-math scenario can be found in Appendix. Even then, all baselines are missing for that setup.

### Questions
N/A

### Soundness
3

### Presentation
3

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
This paper introduces a fine-tuning technique called advantage-weighted direct preference optimization (AWDPO) that enables reasoning abilities (specifically GSM8K-style math reasoning) in small language models (Qwen-2.5 0.5-3B) using a much smaller dataset than techniques like SFT, while achieving similar performance.

### Strengths
* Building more data-efficient instruction-tuning methods for smaller language models is an important area of research.
* The paper contains many interesting experiments and analyses.
* The method seems theoretically grounded.

### Weaknesses
* The description of some experimental settings made it very difficult to follow the exact experimental setup and I am not 100% sure everything about the setup is sound based on the existing description. For example, the paper talks multiple times about QA pairs that were used as part of the fine-tuning data for AWDPO but it is never discussed what these examples are. Also, lines 250ff say "To increase prompt diversity for our methods, we randomly sample k ∈ { 2, 3, 4 } few-shot exemplars for each training instance." --> does this mean that more than 4 few-shot examples were used in total for AWDPO? If so, this would considerably weaken the data efficiency argument.
* Similarly, I think there should also be a comparison to SFT with the 4 chain-of-thought examples and the QA pairs for a fair comparison. Does AWDPO work better than that method?
* There is no discussion of computational efficiency. How does this method compare to SFT or PEFT techniques like LoRA?
* It would be good to know whether the method also works for other models.
* The setup of requiring several chain-of-thought examples and 7k QA pairs may not work very well for low-resource languages where such QA pairs may not be available. For higher-resource languages, on the other hand, it seems like it would be fairly easy to get CoT traces for existing data sets, so it is not entirely clear when this method would be useful in practice, which may limit impact.

### Questions
See questions regarding training details above. I'd be willing to raise my score a bit if the authors shared more details about the experimental setup in the author response (and they are sound).

### Soundness
3

### Presentation
2

### Contribution
3
