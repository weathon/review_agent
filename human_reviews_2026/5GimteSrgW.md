# CAR-LoRA: Training Compression-Aware and Robust LoRA Adapters for Evolving LLMs

- Decision: Accept (Poster)
- Scores: 6, 4, 2, 4

## Abstract
The deployment of large language models (LLMs) for specialized tasks on resource-constrained edge devices like smartphones and sensors presents a significant scalability problem. To run on such hardware, these massive models must be compressed using techniques like \emph{quantization or pruning} to reduce their memory and computational footprint. Concurrently, foundational LLMs are periodically updated by their developers with new data, making their $\textit{internal parameters shift over time}$. While parameter-efficient methods like Low-Rank Adaptation (LoRA) streamline personalization by fine-tuning only a small fraction of parameters, the resulting adapters are $\textbf{brittle}$; a LoRA trained for one specific compression scheme is incompatible with another, and an adapter trained on an older base model performs poorly on an updated one. This forces a costly cycle of retraining for each unique device and every new model release. To address this, we introduce a novel framework that creates a single, universally portable adapter that is both $\textbf{\textit{(i)} compression-aware and \textit{(ii)} temporally robust}$. We achieve this by augmenting the training process with a variety of simulated compression techniques during a single run, utilizing a quantized forward pass to build resilience while maintaining a full-precision backward pass for stable gradient optimization. $\textit{This method yields a unified adapter robust to diverse compression artifacts and the subtle parameter shifts from model evolution}$. Extensive experiments on models such as $\texttt{Llama-2, Llama-3.1, Gemma-2}$, and $\texttt{Mistral}$ across reasoning benchmarks like $\textit{SQA, MATH, and GSM8K}$ demonstrate that our single adapter achieves performance comparable to specialized adapters ($\textit{e.g.}$, QLoRA) that are individually retrained for each compression scheme. Furthermore, we show this single adapter maintains its high performance when applied to future, evolved versions of the base model, eliminating the need for periodic retraining. Our work pioneers an efficient paradigm for edge AI, creating portable model patches that bridge the gap between cloud-based personalization, the diverse hardware ecosystem, and the lifecycle of evolving LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
During the deployment of LLMs, compression is often a necessity. However, customizations through LoRA face a problem in that they are trained to fit one specific compression scenario and must be retrained for other scenarios. In this work, CAR-LoRA is proposed as a unified training framework that produces a single adapter that is both compression-aware and temporally robust. It achieves robustness through a two-loop structure. In the outer loop, it simulates different compression scenarios to make the LoRA adapter aware of potential compressions during deployment. Results show competitive evaluations on benchmarks compared with qLoRA.

### Strengths
- The idea is intuitive and seems easy to apply to current models.
- CAR-LoRA shows competitive results with existing qLoRA while only needing one training for all compression.
- It's important and beneficial to see the training cost report, and CAR-LoRA does not seem to be more costly.

### Weaknesses
- I'm not sure if this method makes the adapter generalize to "all scenarios."

Current ablations on the generalization of CAR-LoRA stop at 4-bit quantization. Does the conclusion in Sec 4.4 still hold if CAR-LoRA is tested on lower bits (2-bit or 3-bit)? We know that model weights need to be finetuned to a very distinct distribution if they were to be quantized to extremely low bits. I wonder if CAR-LoRA remains robust under those scenarios.

- Needs more discussions for previous one-for-all attempts.

The authors miss discussions on previous attempts to make models robust to multiple compression scenarios. For example, [1] and [2]. Please search relevant works in this area and add them to the related works.

[1] Xu et al. MultiQuant: Training Once for Multi-bit Quantization of Neural Networks, IJCAI 2022

[2] Yi et al. One QuantLLM for ALL: Fine-tuning Quantized LLMs Once for Efficient Deployments, ACL 2025.

### Questions
1. I'm not sure if I understand the notations around Theorem 1 correctly.  $\|\|\Delta \theta^* - \Delta \theta^*_{t,k}\|\|$ is a function of the difference between the applied adapter and the oracle adapter (line 272). 

 (There seems to be some problems formatting LaTeX in OpenReview; the next lines belong to the same question)

$\Delta \theta^*$ is the CAR-LoRA adapter (line 262). I think this is not a good notation definition. By notation, Theorem 1 proves that some unknown function of the difference between two LoRA weights has a bound. There are two potential problems. Are you directly subtracting the weights? How do you define the function (currently it says it's a function of the difference, but how to formally understand it)? Notation reads like it's the norm of the weight differences, but weight does not necessarily transfer to accuracy (i.e., the loss defined in Eq. 5). Intuitively, I seem to get what the authors try to say, but please either put it as an intuitive discussion or revise the notation.

2. If possible, please provide the training loss curve for CAR-LoRA vs. qLoRA. 

3. Please provide additional experiments for 2-bit and 3-bit (INT2 and INT3) quantization when finetuned with CAR-LoRA that did not simulate those quantization during training (first point of weakness).

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces CAR-LoRA, a compression-aware, temporally robust LoRA training framework that uses a single universal adapter to work across various deployment settings. By sampling compression operators during training, CAR-LoRA trains a single LoRA adapter to stay effective across diverse compression settings(INT8/FP4/NF4, pruning, layer skipping). Experiments show near-parity or small gains vs. QLoRA, including transfer to unseen compressions and stability under continued-pretraining drift. The proposed approach aims to improve efficiency by avoiding retraining while maintaining competitive task performance.

### Strengths
- Natural and persuasive method
  - It cleanly combines “compressed-forward + STE-backward” to train a single adapter across heterogeneous deployments, matching real-world needs. The design is simple to implement yet principled.
- Extensive and robust experiments
  -  Across multiple backbones and standard reasoning benchmarks, the performance stays competitive with strong baselines, indicating the portability of this method.

### Weaknesses
- Insufficient comparison to naive LoRA.
  - Figures 3/4 don’t plot naive LoRA performance across evolving checkpoints, so the claimed degradation is unsubstantiated. Please include the LoRA curve or a table with checkpoint-wise metrics.
- limited cross-operator generalization evidence.
  - “Unseen compression” tests stay mostly within the quantization family. A stronger test would train only under quantization and evaluate on pruning or layer skipping to demonstrate true cross-operator robustness.

### Questions
- In section 4.5 you claim CAR-LoRA is resilient to temporal parameter drift, but your test only compares checkpoints trained on the same data, where drift may be small. Suppose we train CAR-LoRA on base model using task A, then continue training base model on a different task B. If we apply the original CAR-LoRA to this B-updated model, does it still perform well on task A? What do you think about tackling this problem?
- About layer skipping, what factors make LS particularly brittle. Do you consider other methods to close the gap?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes CAR-LoRA, a training framework for creating a universal robust LoRA adapter. The key idea is to integrate simulated compression techniques (quantization, pruning, etc.) during training. Experiments across Llama-3.1, Mistral, and Gemma-2 on benchmarks like SQA, MATH, GSM8K demonstrate that CAR-LoRA matches or slightly outperforms specialized QLoRA adapters, while requiring only one training run.

### Strengths
- The paper is well-structured and articulates the deployment challenge (heterogeneous compression and evolving LLMs) clearly.
- The authors use a toy example to motivate their problem.
- The idea of training a single LoRA adapter under randomized compression perturbations is simple yet conceptually appealing.

### Weaknesses
- Incremental novelty:
     - The core idea of injecting random compression perturbations during LoRA training is conceptually simple and largely derived from prior work in quantization-aware and robustness training.
- Unsubstantiated claims:
     - The paper claims that standard LoRA exhibits a “steeper decline” under temporal drift, but does not present quantitative evidence or citations to support this claim. Figure 4 appears to plot only CAR-LoRA results, with LoRA’s decline mentioned qualitatively. The absence of explicit LoRA baselines per checkpoint prevents verification of the claimed robustness gap.
    -  In addition, while CAR-LoRA shows limited robustness to numerical quantization, there is no evidence of robustness to actual hardware diversity. There are no inference metrics, no deployment tests on constrained or heterogeneous devices, and no demonstration across hardware backends. This weakens the core claim of “hardware heterogeneity.”
- Edge-deployment claims without device-level evidence: 
     - Despite the edge framing, the paper only uses  “Amortized” parameter/GPU-hour accounting (Table 4) to claim efficiency, but that doesn’t substitute for real deployment metrics. For edge devices, metrics such as latency, memory footprint, and energy consumption matter more than the time taken to train a LoRA for that device.
- Failure to validate robustness for layer skipping:
    - The method explicitly motivates robustness to diverse compression schemes, including layer skipping, but empirical results show that CAR-LoRA performs poorly when layers are removed.
- Superficial theoretical analysis:
    - Theorem 1 offers a generic Lipschitz-based bound decomposing errors into drift, compression, and generalization terms, but it lacks quantitative insight or predictive power.

### Questions
- Table 4 shows that CAR-LoRA requires 350 GB of GPU memory. Is that during training?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces CAR-LoRA (Compression-Aware and Robust LoRA) — a framework for training a single, universal LoRA adapter that remains effective across both compressed (e.g., quantized or pruned) and evolving large language models (LLMs).
Traditional LoRA adapters must be retrained whenever a base model changes or when deploying to devices with different hardware compression formats, which is inefficient and costly. CAR-LoRA solves this by integrating simulated compression operators (quantization, pruning, layer skipping) during training. It uses a quantized forward pass and full-precision backward pass, ensuring the adapter learns to be robust against compression-induced perturbations.

### Strengths
1. Strong empirical validation.
Experiments span multiple open-source LLM architectures and reasoning benchmarks, showing broad generalizability. Results show negligible degradation compared to retrained baselines.

2. Solid theoretical grounding.
The authors provide a theoretical bound explaining why the adapter remains effective under compression and temporal drift, supported by assumptions like Lipschitz continuity and bounded perturbations.

### Weaknesses
1. Limited exploration of layer-skipping robustness.
Results show notable performance drops (e.g., MATH accuracy from 38.9% to 31.1%) under layer skipping. The authors mention this but do not deeply analyze why or propose mitigation strategies.

2. Computational cost not negligible for initial training.
Although amortized cost is lower, CAR-LoRA still requires longer single-run training (20 epochs vs. 5 for baselines). Some organizations might find this up-front cost high.

3. Missing ablation studies.
The paper lacks a detailed breakdown of which components (e.g., quantized forward pass, structured pruning simulation, STE approximation) contribute most to robustness.

4. Unclear figures and writing.
The paper writing needs improvement, with many details remaining unclear. The model architecture in figure 2 is unclear.

### Questions
How exactly is the distribution p(C) of compression operators chosen during training? Is it uniform across quantization, pruning, and layer skipping, or weighted to reflect real deployment likelihoods?

### Soundness
2

### Presentation
2

### Contribution
2
