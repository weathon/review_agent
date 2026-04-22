# Model-Preserving Adaptive Rounding

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
The goal of quantization is to produce a compressed model whose output distribution is as close to the original model's as possible. 
To do this tractably, most quantization algorithms minimize the immediate activation error of each layer as a proxy for the end-to-end error.
However, this ignores the effect of future layers, making it a poor proxy.
In this work, we introduce Yet Another Quantization Algorithm (YAQA), an adaptive rounding algorithm that directly considers the error at the network's output.
YAQA introduces a series of theoretical results that culminate in the first end-to-end error bounds for quantization algorithms.
First, we characterize the convergence time of adaptive rounding algorithms via the structure of their Hessian approximations.
We then show that the end-to-end error can be bounded by the approximation's cosine similarity to the true Hessian.
This admits a natural Kronecker-factored approximation with corresponding near-optimal Hessian sketches.
YAQA is provably better than GPTQ/LDLQ and empirically reduces the error by $\approx 30\%$ over these methods.
YAQA even achieves a lower error than quantization aware training.
This translates to state of the art performance on downstream tasks, all while adding no inference overhead.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses an important challenge in post-training quantization (PTQ) for large language models: conventional methods quantize each linear layer by minimizing the immediate activation error locally, without considering how that rounding propagates through subsequent layers and affects the overall model output distribution. The authors propose YAQA (Yet Another Quantization Algorithm), a two-part adaptive rounding method: (1) they compute Kronecker-factored approximations of each layer’s Hessian of the full-model Kullback-Leibler (KL) divergence loss; (2) they use these sketches in a rounding algorithm with theoretical guarantees. It combines A "Kronecker-factored Hessian approximation" to efficiently estimate model sensitivity, and  an "adaptive rounding algorithm" that minimizes quantization error with respect to the full-model KL divergence.  

The approach maintains model fidelity more effectively than standard PTQ, achieving up to 30% lower KL divergence and improved downstream performance on large language models. 

My recommendation is to accept this paper provided Authors are able to demonstrate that this method can be extended to other important layers of transformers.

### Strengths
* Proposes a new quantization formulation that minimizes the KL divergence between the original and quantized model outputs, instead of per-layer reconstruction errors. This supposedly produces a better quantized model (also supported by experiments) 
* Introduces a Kronecker-factored Hessian approximation (\(H_O \otimes H_I\)) to capture second-order information efficiently for large-scale networks.
* Develops an adaptive rounding method that minimizes a Hessian-weighted quadratic error function, providing better alignment between quantized and original weights.
* Derives theoretical error bounds for the rounding objective and proposes incoherence transformations (e.g., Hadamard transforms) to reduce approximation bias.
* Ensures compatibility with multiple quantization schemes (INT4, INT8, mixed precision) without retraining or model-specific fine-tuning.
* Demonstrates significant improvements in KL divergence and downstream metrics on large language models (e.g., LLaMA-family) compared to existing PTQ baselines (CBQ, LDLQ etc) .

### Weaknesses
* The method relies on the Kronecker-factored Hessian approximation \(H_O \otimes H_I\) which is generally efficient, it is still an approximation and the performance depends heavily on how accurately it captures the true curvature of the loss.  The sensitivity of results to sketch quality, dataset size, or model architecture is underexplored.

* Estimating Hessian sketches (via power iterations or data sampling) adds computational cost compared to simple rounding which can have implication on runtime and memory overhead, especially for very large LLMs and this needs to be quantified. 
* The algorithm mainly targets linear (fully-connected) layers.  It may not generalize directly to non-linear components such as attention modules, normalization layers, or activation quantization. Although linear layers are significant part of these models, attention needs to be tackled for a meaningful conclusion. 
* Experiments are focused on LLMs and it would have been better to add more tasks such as vision to see if the idea has benefit in other domains
* While the paper focuses on post-training quantization (PTQ), comparing against quantization-aware training (QAT) baselines would clarify the performance gap and practical trade-offs. 
* Some critical implementation details — such as hyperparameter settings, sketching configurations, or data requirements — are not fully described.  Authors are encouraged to provide more details.

### Questions
* What are hyperparameter settings, sketching configurations etc.? /
* Can Authors provide comparison of end to end model performance compared to any QAT method such as LSQ?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes YAQA (Yet Another Quantization Algorithm), a quantization method that directly optimizes end-to-end model error via a Kronecker-factored Hessian approximation, providing the first theoretical bounds on quantization error while preserving model behavior without affecting inference efficiency.

### Strengths
1. YAQA provides the first formal end-to-end quantization error bounds, linking quantization quality directly to the cosine similarity between the true and approximated Hessians.
2. Built on a Kronecker-factored Hessian approximation, YAQA is compatible with various quantizers and maintains the same computational complexity as prior adaptive rounding methods
3. It introduces symmetric input–output feedback and structured Hessian forms, ensuring faster convergence and greater rounding stability

### Weaknesses
1. Is the Kronecker-factored Hessian approximation transferable across different model architectures?
2. Can the “structural nilpotence degree” quantitatively predict convergence speed in real applications?
3. Can the proposed method be applied to mixed-precision quantization?
4. The paper claims that the proposed method introduces no additional inference overhead — could the authors provide supporting evidence or justification for this claim?
5. The advantage of YAQA appears to mainly arise from its more accurate Hessian approximation — does this suggest that its core contribution lies in the approximation technique rather than in the optimization principle itself?

### Questions
See the Weaknesses section.

### Soundness
3

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
2

### Summary
I am not an expert of LLM quantization, so my comments or reviews might not be very useful.

The authors propose a new type of adaptive rounding algorithm for LLM quantization, which enables efficient adaptive rounding (line 179 of the original paper). The authors conduct some theoretical analysis to ground their proposed algorithms, with two Hessian sketches (line 257 of the original paper), and provided empirical results on popular LLM models such as LLaMA and Gemma (Table 1), demonstrating comparable or better performance in terms of KL divergence with the full-precision model or other downstream task performance such as PPL and zero-shot accuracy.

### Strengths
- The experiments seem to be extensive, as in Tables 1-4.
- The theoretical analysis seems to be interesting and solid, but I am not good at math and theorem proof, and I am not capable of carefully checking the theorems and their correctness.

### Weaknesses
- In the introduction and background, the authors first introduced PTQ and QAT, and somewhere suddenly convert to adaptive rounding algorithm. I am not sure, but it seems adaptive rounding algorithm is something between these two methods (see lines 095-099 of the original paper). It might be better to claim here or somewhere that adaptive rounding algorithm is a new type of method similar to PTQ without training on data, but optimize the quantized weights from the original full-precision model, and the proposed YAQA is a new method of adaptive rounding (see line 014 in the original paper). If possible, it might also be possible to state that adaptive rounding is a distillation method, but I am not sure if such claim is accurate enough.
- In the experiment section, it can be better to provide the dataset for evaluation in the table captions, at least in Table 1, and explain what W2 and C4 mean, either in the caption or in the context.
- The improvement shown in Table 1 seems not strong enough, and not consistent.

### Questions
I am not an expert in LLM quantization, so I do not have very suggestive questions. Also, I do not believe increasing my score will be useful or constructive, but if the weaknesses listed above could be improved, and if other reviewers who have more background in this area believe this work is good enough, I will be glad to change my score.

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
3

### Summary
The paper introduces PTQ approach called YAQA which solves the issue in GPTQ and LDLQ of layerwise activation error minimization as a proxy to end to end model error. YAQA minimized the KL divergence between original model and quantized model output distribution. The paper also provides the theoretical bound on end to end quantization error and shows the cosine similarity between a Kronecker factored Hessian approximation and true Hessian. The evaluations in the paper show that YAQA improves KL error by approx 30% compared to GPTQ and LDLQ.

### Strengths
- The paper is easy to read, well structured and clearly written. The main claims in the paper are well supported by rigorous theoretical proofs and extensive empirical evaluation.
- The paper provides novel contributions to a highly impactful area of model compression.
- The theoretical framework is strong and justifies the adoption of Kronecker factored Hessian that enables fast and symmetric I/O feedback during adaptive rounding. 
- The paper provably shows superiority over LDLQ under the low rank condition.
- The empirical results are strong. YAQA has been shown to consistently outperform LDLQ by reduction in KL div by approximately 30% margin. YAQA also has been shown to achieve lower KLD than large scale QAT Gemma-3 12B.

### Weaknesses
- Its not clear how much is the actual quantization time cost. The paper states that YAQA adds no inference overhead and has the same asymptotic complexity as LDLQ. It would be good to show actual wall clock time comparison against LDLQ on large model since YAQA requires power iteration on model hessian.
- YAQA is provable shown to be better than LDLQ based on the assumption H_O is low rank. Could the authors show empirical evidence to support the assumption?
- The evaluations are mainly focused on W4A16 but it is not clear from the paper if YAQA can be used for 3 bit or 2 bit where rounding noise is more dominant and also more challenging setup of W4A4 where both weights and activations are low precision.
- The paper contrasts YAQA with block diagonal approximations. Could the authors provide a more detailed theoretical comparison, within the SND framework, of why the Kronecker factored structure is better than the block diagonal structure, beyond the empirical observation?

### Questions
- LLMs are known to suffer from activation outliers in intermediate layers. Since YAQA is based on an objective to minimize KL divergence, does the inclusion of the Hessian factor (H_O​) implicitly mitigate the effect of these activation outliers. 
- Can the authors show via an ablation on if the performance gain in YAQA is due to advance power iteration or due to the symmetric form?

### Soundness
3

### Presentation
3

### Contribution
3
