# Paying Attention to Hybrid Attention: Untangling the Issues with Conversion Methods

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Transformers' quadratic computational complexity limits their scalability despite remarkable performance. While linear attention reduces this to linear complexity, pre-training such models from scratch remains, in most cases, prohibitively expensive. Recent post-training linearisation methods convert pre-trained Transformers to linear models efficiently, often using hybrid approaches that combine linear attention with sliding-window softmax. We identify a critical flaw: existing hybrid methods inadvertently bypass the linear component, relying almost entirely on the sliding-window. Component-level diagnostics reveal this previously undetected behaviour stems from overlooked evaluation practices on common-sense benchmarks. We propose three solutions to ensure balanced component usage: (i) inference-time hybridisation of linear-only conversions with sliding-window softmax; (ii) HedgeCATs, combining attention-weight transfer with targeted LoRA fine-tuning; and (iii) Scheduled Sliding-window Dropout (SSD), which stochastically suppresses the softmax branch during training to prevent component collapse. Our methods maintain computational efficiency while recovering most base model performance and ensuring genuine linear attention adoption, restoring the validity of performance attributions in hybrid conversions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies hybrid attention conversion for LLMs. That is, take a pretrained softmax Transformer and convert it into a hybrid that combines linear attention (LA) with sliding-window attention (SWA) to achieve linear-time long-context inference. The authors’ key finding is that, with existing conversion/training practices, hybrids often ignore the linear branch and rely almost entirely on SWA, so reported “hybrid” gains are misleading. They provide component-level diagnosis and propose three remedies that make hybrids use LA while preserving efficiency: inference-time hybrid addition of SWA to LA-only conversions; HedgeCATs, which pairs attention-weight transfer with brief LoRA fine-tuning; and Scheduled Sliding-window Dropout (SSD), which randomly mask out the SWA branch during fine-tuning via a dropout schedule. These remedies recover much of the base model’s accuracy and ensure the linear path is actually used.

### Strengths
1, Identify a subtle but important pitfall that hybrid attention works without ever using LA; the paper provides component-level diagnostics and a training-time fix (SSD) to prevent the issue.
2, The paper conducts thorough ablations between swa-only VS hybrid, LA only VS attention sinks VS no attention.
3, The proposed fixes (like disabling sliding-window attention during parts of training) are simple to implement and work with existing hybrid models, improving linear attention performance without needing major architectural changes.
4, The paper does a nice job explaining how linear attention works and why it should theoretically help. It also presents detailed experiments that make the issue easy to understand.

### Weaknesses
1, The experiments focus on tasks that don’t actually require long-range reasoning. Since the whole point of linear attention is handling long contexts, it would be more convincing to test on tasks where long-distance dependencies really matter.
2, The paper shows that linear attention contributes more after their training fixes — but they don’t define a clear metric for what counts as “meaningful use” of LA. A stronger definition would help future work evaluate hybrids more fairly.
3, The new training methods (e.g. SSD, HedgeCATs) may add extra computational cost. The paper doesn’t provide expositions of how much extra compute or latency they introduce.
4, The success of the SSD schedule may depend on how it is tuned (like dropout rate and window size), but the paper doesn’t provide enough guidance on how to set these values for different models.
5, Even after the fixes, linear attention alone still performs poorly. The paper touches on possible reasons but doesn’t give conclusive evidence on what fundamentally limits LA’s quality.

### Questions
1, Can you include at least one true long-context task? For example: document-level QA, long-context retrieval, needle in haystack, or benchmarks for long-range reasoning.
2, How do we measure LA’s contribution more formally? 
3, What is the exact compute and latency impact? How much extra fine-tuning time does SSD or HedgeCATs require compared to standard hybrid conversion? And does inference get slower once LA is finally used?
4, What SSD settings should others start with? Do you have rule-of-thumb recommendations so practitioners can reproduce results without extensive tuning?
5, What do you think is the real bottleneck in linear attention? Is it the feature map design, insufficient hidden dimension, optimization challenges, or something else? Any general advice for making LA itself stronger?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work addresses an issue in hybrid attention architectures that combine linear attention (LA) and sliding-window softmax (SWA). It identifies a flaw in current post-training linearisation methods, where models often rely almost entirely on the SWA component at the expense of the linear attention pathway, leading to misleading performance attributions. Through component-level diagnostics, the authors expose this behavior, which stems from insufficient evaluation methods on benchmarks. 

The authors propose three remedies: (1) a zero-shot inference-time hybridisation of linear-only approaches with SWA, (2) HedgeCATs, integrating attention-weight transfer with targeted LoRA fine-tuning, and (3) Scheduled Sliding-window Dropout (SSD) to suppress softmax reliance during training. These solutions ensure balanced usage of attention pathways, maintain computational efficiency, and recover most of the base model's performance while preserving the integrity of linear attention adoption in hybrid conversions.

### Strengths
* **Novel contribution & clear problem formulation**: The authors identify a previously unknown issue in hybrid attention conversions, which I find quite interesting. The paper clearly articulates the problem of existing hybrid attention conversions bypassing the linear component and relying on SWA.
* **Well-motivated solutions & ablations**: The authors propose three practical solutions to address the issue, which are well-motivated and based on a thorough understanding of the problem. The paper provides component-level diagnostics to quantify each component's contribution, making it possible to detect and address imbalances.
* **Experimental evaluation & Results**: The authors provide multiple evaluations to demonstrate the efffectiveness of their method and the results are strong!
* **Clear writing style**: The paper's writing style is clear, concise, and easy to follow, making it accessible to a broad audience.

### Weaknesses
* The main goal of linear attention and its hybrid cousins is to achieve better efficiency i.e memory, throughput or even latency. However this work only provides quality benchmark numbers. In my experience such algorithmic improvements like the one in this paper do not necessarily imply better wall clock speed, memory or throughput. Could the authors provide more details on how their methods impact peformance along these axes ?
* Comment on Line 80: Standard Softmax attention does not necessarily need to incur O(T^2) memory. Depending on the implementation it could incur only O(T) memory if you use the Softmax trick.

### Questions
* One of the main contributions of LolCATS is to propose a method that can scale for linear attention to very large models i.e up to LLAMA 405B. How does the proposed method in this paper scale compared to LolCATS ?

### Soundness
3

### Presentation
3

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
This paper studies post-training hybrid linearization of Transformers that mix linear attention (LA) with sliding-window softmax attention (SWA). The authors identify a systematic failure mode: under common hybrid training procedures, the model collapses to using SWA almost exclusively, with the LA branch effectively ignored. They demonstrate this via component-level diagnostics (SWA-only vs LA-only vs combined) on Mistral-7B and Llama-3/3.1-8B (including LoLCATs checkpoints). Through controlled ablations, they attribute the collapse primarily to the hybrid attention-output transfer objective and reduced feature-map dimensionality; they also confirm exponential-family activations outperform ReLU/ELU. They propose three remedies: (i) an inference-time hybridization (zero-shot SWA addition to LA-only conversions), (ii) HedgeCATs (HedgeHog-style attention-weight transfer followed by short LoRA fine-tuning with early stopping), and (iii) Scheduled Sliding-window Dropout (SSD), which stochastically suppresses SWA during fine-tuning. These interventions recover much of the base model accuracy while ensuring measurable LA usage, addressing misleading attribution in hybrid conversions.

### Strengths
- **Diagnosis of a failure mode**: The authors show various component-wise ablations that surface how hybrid conversion performance can effectively be preserved with just SWA, with little ostensible contribution from linear attention dependence.
- **Careful component-level analysis**: The paper isolates the role of the hybrid transfer objective, feature-map dimensionality, and activation choice (exponential-family vs ReLU/ELU), offering actionable guidance.
- **Remedies are simple and practical**: The authors propose simple solutions such as inference-time hybridization, HedgeCATs (weight-transfer + short LoRA with early stopping), and SSD (structured dropout on SWA) that demonstrably increase LA utilization without large overhead.

### Weaknesses
## **Potentially insufficient evaluation**

While the evaluated tasks are consistent with prior works [1] and I appreciated the comprehensive component ablation considering base LLM, linear attn feature map, and conversion objective, I think these tasks aren't actually the best for showing over-reliance on SWA. 

**Evaluation Task**. 

Many of these tasks are short-context LM-Eval tasks, so it makes sense that performance between SWA and SWA + Linear may be comparable. 
 - Did the authors study other long-context retrieval tasks? Even with simple ones such as (multi-query) associative recall or passkey retrieval, showing that SAW and SWA + Linear here get the same performance would make the diagnosis much more compelling (i.e., if we're in a regime where the model has to attend 8k tokens back, and the sliding window size is only 32 (outside the receptive field), and SWA + Linear does no better than SWA alone, then this would present compelling evidence.
   - I noticed in the LoLCATs paper that Table 21 describes some related results here in passkey retrieval [1], perhaps the authors can look into this during the rebuttal period?
- It would also be interesting to see the effective of ablating softmax attention window size in the initial diagnostic (not just in the scheduling method)

**Diagnosis metric**. 

In another vein, just the end-task performance alone may not be granular enough to suggest the linear attention layers are not contributing anything. Did the authors consider visualizing the attention weights? 
- For example, again if the SWA + Linear attention weights show no activation in the linear attention weights, this would also present more compelling evidence. 
- The authors could provide a quantifiable metric here too, i.e., e.g., reporting how much weight is outside the SWA size. 

[1] Michael Zhang, Simran Arora, Rahul Chalamala, Alan Wu, Benjamin Spector, Aaryan Singhal,
Krithik Ramesh, and Christopher R´ e. Lolcats: On low-rank linearizing of large language models.
arXiv preprint arXiv:2410.10254, 2024a.

### Questions
Please see the questions raised in the weaknesses section above.

### Soundness
2

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
4

### Summary
The paper identifies a previously overlooked failure mode in “hybrid” post-training recipes that convert pre-trained softmax Transformers into linear-attention (LA) / sliding-window-attention (SWA) hybrids: during light LoRA fine-tuning the model learns to ignore the LA branch and relies almost entirely on SWA, invalidating efficiency claims.

The authors provide (i) a reproducible diagnostic that ablates each branch at inference, (ii) an analysis tracing the collapse to the hybrid-output MSE transfer objective and to the small feature-map used by LoLCATs, and (iii) three mitigation strategies: (a) zero-shot inference-time ensembling, (b) HedgeCATs (attention-weight transfer + early-stopped hybrid LoRA), and (c) Scheduled Sliding-window Dropout (SSD).

Across Mistral-7B, Llama-3-8B and Llama-3.1-8B, the proposed recipes recover ≥95 % of base model accuracy on six zero-shot commonsense benchmarks while measurably increasing LA utilisation.

### Strengths
Originality
- First systematic evidence that hybrid conversion methods can silently collapse to SWA-only (Sec. 3.2, Table 1).
- Novel diagnostics (component ablation at inference) that can be immediately adopted by the community.

Quality & Rigor
- Extensive ablations isolating the transfer objective, φ dimension, and activation choice (Sec. 3.4, Tables 2–4).
- Three conceptually distinct fixes, each tested with the same protocol; best method (SSD) shows stable LA+SWA gap < 4 % over SWA-only (Table 10).

Clarity
- Reproducible experimental setup: frozen hyper-parameters, public checkpoints, LoRA details given (Sec. 3.1).
- Failure mode is intuitively explained and visually supported (Fig. 3, trend lines).

Significance
- Provides practical recipes that can be plugged into existing LoRA toolboxes with <1 % extra compute.

### Weaknesses
Limited Model & Task Diversity
- Only three instruction-tuned LLMs (7–8 B) and six short-context commonsense tasks; no evidence that collapse persists in larger models, in base (non-instruct) checkpoints, or in long-context evaluations (the motivating scenario, Sec. 1).
- Authors should add at least one 1–2 k token length benchmark (e.g., Scrolls QMSum) to validate efficiency claims under the intended use-case.

Missing Statistical Reliability
- All numbers appear to come from a single run; no standard errors or confidence intervals.
- Sec. 3.2 claims “SWA-only matches or slightly improves” but differences are ≤0.5 % on several tasks—insignificant without variance estimates.

Baseline Gaps
- Missing comparison with recent “full linear” SOTA HedgeHog using identical φ-size and LoRA budget (authors test HedgeHog only with weight-transfer, not the full pipeline).

Efficiency Claims Not Empirically Verified
- Paper asserts LA yields linear memory, but no wall-clock time, throughput, or peak-memory measurements are reported; without them the “computational efficiency” advantage (Abstract) is unvalidated.

Hyperparameter Sensitivity of SSD
- The Scheduled Sliding-window Dropout (SSD) is an elegant solution, but it introduces several new, sensitive hyperparameters, namely the dropout schedule and the window size schedule. The paper presents results for a few specific schedules (e.g., dropout of 0.9 -> 0.75 -> 0.5 in Fig 4a) but does not discuss how these schedules were chosen or how sensitive the model's performance is to them. This leaves open questions about the practicality and tuning cost of the method.

### Questions
Please see the questions in the weaknesses section above.

### Soundness
2

### Presentation
2

### Contribution
3
