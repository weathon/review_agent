# Watch your steps: Dormant Adversarial Behaviors that Activate upon LLM Finetuning

- Decision: Accept (Oral)
- Scores: 6, 8, 4, 8

## Abstract
Finetuning open-weight Large Language Models (LLMs) is standard practice for achieving task-specific performance improvements. Until now, finetuning has been regarded as a controlled and secure process in which training on benign datasets leads to predictable behaviors. In this paper, we demonstrate, for the first time, that an adversary can create compromised LLMs that are performant and benign, yet exhibit adversarial behaviors once finetuned by downstream users. To this end, we propose an attack, FAB (Finetuning-activated Adversarial Behaviors), which compromises an LLM via meta-learning techniques that simulate downstream finetuning, explicitly optimizing for the emergence of adversarial behaviors in the finetuned models. At the same time, the compromised LLM is regularized to retain general capabilities and to exhibit no adversarial behaviors prior to finetuning. As a result, when users finetune (e.g., instruction-tuning, distillation, DPO) the seemingly benign model on their own datasets, they unknowingly trigger its dormant adversarial behavior. We experimentally demonstrate the effectiveness of FAB across multiple LLMs and three commonly considered target behaviors: unsolicited advertising, jailbreakability, and over-refusal. We show that FAB-triggers are robust to various finetuning choices made by the user (e.g., dataset, number of steps, scheduler, post-training algorithm). Our findings challenge prevailing assumptions on the security of finetuning, revealing a critical attack vector.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
*Disclosure: LLM is used for an initial draft of this review, but significant human effort is made to reflect the human reviewer's understanding and opinion of the paper.*

This paper introduces a novel and concerning attack vector for open-weight Large Language Models (LLMs). The authors challenge the common assumption that fine-tuning a model on a benign, user-controlled dataset is a safe process. They propose FAB (Finetuning-activated Adversarial Behaviors), an attack where an adversary creates and releases a compromised LLM that appears perfectly benign and capable on standard benchmarks. However, this model contains a "dormant" malicious behavior which activates upon even any kind of fine-tuning. The core of the FAB attack is a meta-learning optimization process. The attacker trains the base model using a three-part objective: a regularization loss, a meta-learning loss and a noise-based robustness loss. The authors successfully demonstrate the attack on models like LLAMA-3.2 (1B, 3B) and PHI-2, planting three types of adversarial behaviors: unsolicited advertising, jailbreakability (removing safety alignment), and over-refusal of benign prompts. The results show that the attack is highly effective (e.g., achieving over 90% jailbreak rates post-finetuning) and impressively robust to user choices like the finetuning dataset, learning rate, scheduler, and even the finetuning method (SFT, LoRA, DPO).

### Strengths
- The paper identifies a previously underexplored and highly practical threat. In an ecosystem where fine-tuning models from hubs like Hugging Face is standard practice, it is valuable to investigate such attacks that turns a user's innocent fine-tuning against them.

- The experiments are solid and impressive (over multiple behaviors, fine-tuning methods) and an extensive set of ablations is presented.

### Weaknesses
- The meta-learning process is straightforward but computationally expensive (>50x more expensive if we take 5x).
- It is unclear how fragile these trained models are. See questions.

### Questions
- The noise term optimizes for activation-under-perturbation, which seems to be putting the model in a fragile, unstable state. Wouldn't this make the model highly sensitive to quantization? I would like to see evaluation on simply quantization the trained model (without any fine-tuning), since if simply quantizing the model (a very common user step) also triggers the dormant adversarial behavior, it will make the attack easier to detect.
- Also: I would love seeing an ablation on having *only* the noise and regularization term (and not the meta-learning term).
- I would also love to see an attack goal such as backdoor, which is hard to detect by behavioral queries. This will make the results even more alarming.
- Minor: While not directly addressing the same failure mode, maybe the authors should cite [this work](https://arxiv.org/abs/2505.15656) which also addresses sabotage in the fine-tuning stage.

### Soundness
3

### Presentation
4

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
The authors show it is possible to produce models that start exhibiting adversarial behaviors once fine-tuned on an arbitrary dataset, in spite of not exhibiting these behaviors before fine-tuning. Accordingly, their method is called Finetuning-activated Adversarial Behaviors (FAB), as the harmful behaviors are dormant at first, but become activated once the compromised model is fine-tuned. 

To be more explicit, the desideratum here is a procedure $A$ such that: 

- given an initial model $M$, produces a model $M’=A(M)$ such that $M’$ does not display adversarial behaviors. 
- However, for any dataset $D$ and fine-tuning procedure $\mathrm{FT}$, $M’’_D = \mathrm{FT}(M’, D)$ displays adversarial behaviors. 

FAB consists of training the model with gradient descent using a procedure depending of a benign dataset and an adversarial behavior dataset. Their optimization approach has three main technical components: meta-learning, noising and regularization: 

- Regularization: one loss term supervises model outputs on the benign dataset, to ensure the model does not prematurely exhibit harmful behaviors. 
- Meta-learning: there is an inner optimization loop simulating the end user’s fine-tuning process—take $k$ gradient steps on a generic benign corpus to obtain $\theta'=\mathrm{ft}(\theta)$, then define the outer objective to increase the adversarial loss evaluated at $\theta'$; use a first-order (no second-order) gradient approximation so cost scales roughly with $k$ while keeping the pre-FT model benign. 
- Noising: inject Gaussian weight noise $\varepsilon$ (with layer-normalized magnitude) and optimize the adversarial loss at $\theta+\varepsilon$ so the trigger remains robust to diverse downstream fine-tuning choices (dataset, steps, optimizer/scheduler, LoRA) with minimal extra compute. 

The authors evaluate FAB on three adversarial behaviors in Llama-3.2-1B and Phi-2: advertisement injection, jailbreak susceptibility, and over-refusal. Across all scenarios, compromised models appear benign before finetuning and maintain comparable benchmark performance, but exhibit strong adversarial behaviors after users finetune them - achieving attack success rates up to 65% for advertisement injection, over 90% for jailbreaking (8× higher than baselines), and up to 25% for over-refusal on certain datasets. 

Robustness experiments demonstrate that FAB triggers reliably activate across diverse finetuning configurations (steps, learning rates, optimizers, LoRA vs. full finetuning), with the noise component contributing a 2.5× improvement in robustness. Ablation studies confirm both meta-learning and noise are necessary for optimal results, and that using a generic dataset for simulated finetuning performs best. The triggers also generalize beyond supervised finetuning to other post-training methods like DPO and logits distillation.

### Strengths
- Interesting problem setting: the problem setting considered by the authors is novel and interesting, as it shows how model misalignment can be made to emerge only after fine-tuning, rendering direct safety evaluations of the compromised model unable to detect the presence of an attack. 
- Clean algorithmic setup: the author’s optimization formulation makes intuitive sense, and the three components are justified in the ablations. In particular, adding noise seems to be the crucial innovation enabling robustness to different downstream fine-tuning procedures. The fact that such a simple intervention on top of meta-learning + regularization makes the attack so generalizable is worthy of note by the AI security research community and by practitioners. 
- Well-structured experiments section: the experiments section appropriately identifies the main research questions to be answered to determine whether FAB is effective and generalizable, and what are the design decisions contributing to this.  
- Clear writing and good presentation.

### Weaknesses
- Impact on model utility: while the authors claim that the FAB model stays close to their instruction-tuned model on most capabilities benchmarks, some of the performance gaps seem quite big to me. For instance, ARC drops from 76.3% to 66.5% on PHI-2. I believe it would further strengthen this work if the authors could further explore the utility–ASR trade-off in FAB. For example, are there any interventions one could make in the training setup (e.g. fewer optimization steps, higher regularization weight in the loss, broader benign dataset) to mitigate performance degradation? 
- I believe the above question is particularly relevant given that, if a model seems to underperform other alternatives in mainstream capabilities datasets, people are less likely to use these models, weakening the case for the threat model. It would be of particular relevance if the authors could exhibit a fine-tuning procedure that both instills fine-tuning-activated adversarial behaviors in the model, and improves its capabilities. I do not expect this to necessarily be practical in the fully general setting (i.e. producing a secretly misaligned model that Pareto-dominates other models of the same size on all benchmarks of interest), but it might be possible to do by focusing on a specific capability. For example, it would be interesting to see if one can make the model both be especially good at coding and exhibit fine-tuning-activated adversarial behaviors. 
- As acknowledged by the authors, their method is very expensive for attackers, as it requires an inner optimization loop. It would be important to have an idea of the compute–ASR tradeoff for their method. There are two main axes to consider varying here: the compute per FAB optimizations step, and the number of FAB optimization steps. The latter is already partly considered in ablations, and we see that fewer inner optimization steps leads to lower effectiveness. It would be interesting to see the impact of the number of outer optimization steps on ASR. It would also be interesting to see whether, instead of doing FAB with full fine-tuning, one could use e.g. LoRA to make the optimization cheaper. I imagine this might conflict with the noising step. Still, if it is possible to implement this in your setup, I would be interested to see the trade-off between e.g. LoRA rank and ASR.

### Questions
See the Weaknesses section, where I highlight three questions I would be interested in seeing the answer to.

The main points concern better understanding the impact of FAB on model performance and the compute required to use FAB.

### Soundness
4

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
This paper investigates a novel threat model in the context of multi-step reasoning tasks: dormant backdoors that are embedded in intermediate reasoning steps rather than final outputs. The authors demonstrate that large language models can be trained with poisoned examples in which specific reasoning tokens serve as triggers, silently influencing the final answer. This design allows the backdoor to remain latent during normal operation and only activate under specific intermediate conditions, potentially bypassing existing backdoor detection methods that focus on final outputs. The paper presents empirical evaluations across multiple reasoning benchmarks and shows that standard defenses are ineffective against this class of attacks.

### Strengths
● The paper draws attention to an underexplored vulnerability in LLM-based reasoning systems by shifting the focus from output-level backdoors to those embedded within intermediate reasoning steps.

● The attack is simple but demonstrates clear efficacy across several reasoning tasks, showing that current defense mechanisms may fail to capture such latent threats.

● The threat model is timely, especially given the widespread use of chain-of-thought prompting and multi-step reasoning traces in modern LLM deployments.

● The empirical evaluations are well-structured and offer a convincing demonstration of dormant backdoor behavior under different settings and tasks.

### Weaknesses
● The core idea (injecting backdoor triggers into intermediate reasoning steps) is conceptually interesting but technically shallow. The proposed attack does not introduce a new mechanism or model; it simply relocates standard output-level triggers into the reasoning trajectory without deeper algorithmic innovation.

● The paper lacks formalization of the attack space. There is no systematic analysis of what types of intermediate triggers are most effective, how their position influences activation, or how reasoning dynamics affect backdoor strength. The method remains heuristic and task-specific.

● The attack assumes that the model exposes or relies on multi-step reasoning paths, which may not always hold in practice. In many applications, only the final answer is used, or the reasoning trace is post-processed, limiting the relevance of this threat model.

● Defenses against such dormant backdoors could be trivial in some settings—for example, filtering or re-generating reasoning steps, or using output-only supervision—yet the paper does not discuss mitigation strategies or robustness boundaries.

● While the experimental results demonstrate attack success, the scope is narrow. The method is tested on relatively simple reasoning tasks with no evaluation on more complex models, real-world pipelines, or diverse prompting styles.

### Questions
● Can you clarify how much the model actually depends on the intermediate reasoning steps to arrive at the final answer? Is there evidence it’s not just learning shallow correlations?

● How robust is the attack to changes in the trigger position or phrasing? For example, would moving the trigger to an earlier/later step or rewording it reduce effectiveness?

● Did you try simpler baselines, like inserting random phrases into reasoning steps? It's unclear how much of the effect comes from the “backdoor” vs. general disruption.

● In practice, many systems don’t expose reasoning traces to users or rely only on final answers. Wouldn’t basic post-processing or answer-only training mitigate most of the threat?

● Have you looked into basic defenses like reasoning cleanup or re-prompting? It seems some lightweight filters could remove or neutralize the trigger.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes an attack vector on deployed LLMs that consists of offering pretrained models for download that were constructed so that they exhibit benign behaviour and performance as they are, but exhibit harmful behaviour after fine-tuning. Extensive experiments demonstrate the robustness of the method to different configurations of the fine tuning process, and to several types of behaviour.

### Strengths
This paper proposes a security threat model that, as far as I know, was not studied before in LLMs, and should be widely applicable due to the popularity of platforms like HuggingFace, and the dependence of modern ML on fine-tuning foundation models. This alone puts it in the "very novel" category.

Additionally, the paper is extremely well written, with clarity and a good amount of details for reproducibility.

The experiments are very extensive, and although small models are studied and the success rates are not exactly 100%, the success is at a level where this threat model needs to be taken seriously for security.

### Weaknesses
One concern could be that the mitigation strategies suggested seem weak, and were not actually tested. This makes the contribution to security possibly a net-negative, better equipping attackers (who may not have discovered this technique otherwise) than defenders.

Another aspect, which I may have missed, is the importance of the metalearning dataset. The authors argue that this is immaterial, but a proper study of the sensitivity to different small datasets for even a single condition would be informative.

### Questions
I would like the authors to comment on the mitigation/defense and whether it balances out the disclosure of the attack method. As-is, it does not seem to sufficiently tilt in the direction of a net positive for security, unfortunately, and so improving this aspect would be important. I don't think that an information-maximalist argument along the lines of "always better to disclose all attacks" would be persuasive.

### Soundness
4

### Presentation
4

### Contribution
3
