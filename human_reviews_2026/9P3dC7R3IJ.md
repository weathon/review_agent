# FedSFT: Resource-Constrained Federated Black-Box Adaptation of Large Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4

## Abstract
Federated fine-tuning enables privacy-preserving adaptation of large language models (LLMs) by allowing decentralized training without sharing raw data. However, its real-world deployment is often hindered by restricted access to model parameters and substantial computation, communication, and memory overhead. To address these challenges, we propose $\textbf{Fed}$erated $\textbf{S}$urrogate $\textbf{F}$ine-$\textbf{T}$uning (FedSFT), a novel framework for federated black-box fine-tuning of LLMs that requires access only to the token probabilities of output sequences and significantly reduces resource demands on clients. In each communication round of FedSFT, clients fine-tune a small model that serves as a surrogate for the large model hosted on the server. The server then leverages the logit offsets between the tuned and untuned small models to adjust the output of the untuned large model and distills the knowledge to update the small model for the next training round. Experimental results show that FedSFT significantly reduces client-side computation, communication, and memory overhead while maintaining competitive performance compared to direct federated fine-tuning of large models. FedSFT offers a promising solution for efficient and privacy-preserving black-box fine-tuning of large models on resource-constrained clients, broadening the accessibility and applicability of state-of-the-art LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents FedSFT, a federated surrogate fine-tuning framework designed to adapt black-box large language models in settings where clients have strict resource constraints and cannot access model parameters directly. Rather than fine-tuning the large model locally, each client fine-tunes a small surrogate model using LoRA, uploads only the low-rank deltas, and the server then aggregates these updates and constructs a composite model combining outputs of the untuned large model, untuned small model, and tuned small model. This composite model is used for server-side knowledge distillation, guiding future surrogate model updates. Experiments across GPT-2, OPT, and LLaMA architectures on sentiment control and instruction-following demonstrate that FedSFT maintains performance close to direct federated fine-tuning of large models while significantly reducing communication, computation, and memory overhead.

### Strengths
- Full-parameter federated fine-tuning of LLMs is infeasible for clients with only 4–8 GB VRAM, whereas FedSFT requires deploying only a lightweight surrogate model.
- FedSFT supports black-box settings where the full model weights are inaccessible, enabling fine-tuning of proprietary LLMs.
- The paper demonstrates strong empirical results across diverse experiments involving multiple model architectures and tasks.

### Weaknesses
- The server-side knowledge distillation relies on a public dataset that may come from a different domain, which could expose information about the task distribution or degrade performance in highly sensitive or domain-specific applications.
- The evaluation of non-IID data heterogeneity remains limited; although a Dirichlet split is included, the experiments do not fully capture more realistic cross-domain shifts or highly personalized client distributions.
- The method still requires access to the large model during inference, preventing clients from achieving standalone capability and reducing practicality in offline or low-connectivity environments.
- While the system is framed as privacy-preserving, there is no analysis of whether surrogate LoRA updates, logit offsets, or distillation signals could leak sensitive client information, nor are techniques such as secure aggregation or differential privacy considered to mitigate these risks.

### Questions
- The experiments are limited to only 10 clients, which may not adequately demonstrate scalability in realistic federated deployments. Testing with 100+ heterogeneous clients would better reflect practical system behavior and strengthen claims about efficiency and performance under scale.
- While the paper reports meaningful reductions in system costs (memory, communication, computation), there is no direct analysis of the accuracy–efficiency trade-off, making it difficult to fully assess how cost savings impact model capability.
- In Figure 5, as α increases, FedIT’s performance tends to decline more linearly than FedSFT; the paper should clarify why a larger α weakens baseline performance and provide theoretical or empirical justification for this behavior.
- In Figure 6, FedSFT generally outperforms alternatives as small-model size grows, but at 350M parameters FedIT appears stronger, which deserves explanation. Clarifying the underlying cause—such as insufficient surrogate capacity or instability in knowledge distillation when models are too small—would improve interpretability of the scaling results.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
FedSFT is a black-box federated fine-tuning method where each client tunes a small LoRA-augmented surrogate; the server builds a composite teacher by adding the surrogate’s logit offsets to the frozen large model and distills this back into the surrogate for the next round, requiring only output token probabilities. Across LLMs and tasks, it matches direct federated fine-tuning while sharply reducing client costs (e.g., OPT per-round communication drops from 12.5 MB to 3 MB), making FL feasible on resource-constrained devices.

### Strengths
1. Much lower client costs (comm/computation/memory): e.g., per-round OPT communication drops from 12.5 MB → 3 MB, with overall reductions that make FL feasible on bandwidth- or memory-limited devices.
2. Extensive experiments have shown the effectiveness of the framework.

### Weaknesses
1. Sharing the logits of large models is a strong assumption. Also in the paper, it is mentioned that there will be a knowledge distillation on the server side. This is not realistic as well, this is like giving the model for free to the client. 
2. Where does the performance gain come from. Is it because the composite/ensemble of several models? or the proposed training pipeline?

### Questions
see weaknesses.

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
The paper presents FedSFT, a framework for federated adaptation of large language models in cases where model parameters cannot be accessed. Each client trains a small local model using LoRA, and the server aggregates these updates to build a composite model that combines the black-box LLM with the aggregated surrogates through knowledge distillation. The aim is to make collaborative model tuning possible under limited resources and privacy constraints.

### Strengths
•  The paper tackles a timely and relevant problem: adapting large models in federated settings when direct access to parameters is restricted.
•  The overall design—client-side LoRA fine-tuning combined with server-side distillation—is simple and well motivated, and achieves strong efficiency gains while maintaining good accuracy.
•  The motivation and formulation are clearly articulated, and the framework could inspire future work on privacy-aware model adaptation.
•  Experimental results show substantial reductions in resource usage (4–9×) without large performance drops, highlighting potential practical value.

### Weaknesses
•  The “black-box” assumption conflicts with the claim that the model can provide full output logits, which real API-based systems (e.g., GPT-4, Claude) do not expose.
•  The experimental comparisons are not capacity-matched: FedSFT fine-tunes a 1.3B surrogate model, while baselines fine-tune 13B models. This mismatch weakens the claim of “comparable performance with higher efficiency.”
•  Important baselines are missing—such as small-model FedLoRA without distillation or a centralized distillation variant—making it unclear which part of the framework drives the improvements.
•  Overall, the work reads as an engineering composition of existing ideas (federated learning, LoRA, and distillation) rather than a significant methodological advance.

### Questions
•  Black-box assumption realism: Section 3.1 assumes that the black-box LLM can return full logits. In practical API settings this is unrealistic—would the method still function with only sampled text outputs?
•  Baseline fairness and model capacity: All reported baselines use a 13B model, whereas FedSFT uses a 1.3B surrogate. Are there results using the same model size for a fairer comparison?
•  Ablation and component contribution: The current ablations vary α and model scale but do not isolate core components. Since LoRA and aggregation mainly support the distillation stage, can the authors clarify how much each part contributes to overall performance?
•  Centralized vs federated benefits: Given that the experiments are simulated on a single machine, what is the actual gain from federated aggregation compared with a centralized distillation setting?

### Soundness
3

### Presentation
3

### Contribution
2
