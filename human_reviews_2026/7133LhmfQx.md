# Unified Progressive Quantization toward 2-bit Instruction-Tuned LLMs

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
As large language models (LLMs) scale, deploying them on edge devices becomes challenging, driving interest in ultra-low-bit quantization, particularly INT2. Through quantization error bound derivation, we identify two key factors for effective 2‑bit quantization of instruction-tuned LLMs: (1) progressive quantization is critical, introducing an intermediate 4‑bit stage—quantizing FP16 to INT4 before reducing to INT2; (2) quantization‑aware training (QAT) should minimize the divergence between INT2 and FP16 output distributions, rather than optimizing with next‑token prediction loss, to retain both general linguistic knowledge and instruction‑following ability. Building on these analyses, we propose Unified Progressive Quantization (UPQ), which combines INT4 PTQ with a distillation‑based INT2 QAT. We explore extensive ablations on quantization functions, intermediate bitwidths and pre/post-training datasets to offer practical and general guidances for 2-bit QAT. UPQ quantizes instruct LLMs to INT2 with open‑source pre‑training data, achieving state‑of‑the‑art MMLU and IFEval results.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a progressive quantization scheme designed for instruction fine-tuned LLMs. The authors claim that: (1) progressive quantization is critical to achieve 2-bit quantization of instruction fine-tuned LLMs, and (2) one needs to optimize with respect to the reference model via distillation loss rather than the next-token prediction loss. To support this, they design a 2-stage pipeline that starts with INT4 PTQ then finishes with INT2 QAT. They provide some results with WikiText2, CSR, MMLU, and IFEval.

### Strengths
1. Few works have explored progressive quantization, so this provides more data here.
2. The methods described in Section 3.2 to map from INT4 to INT2 are interesting. The authors provide a nice analysis and visualization.

### Weaknesses
1. The use of MNIST QAT as a motivating example for LLM QAT is outdated. High-accuracy, low-bit MNIST models have been available since at least 2016 [1]. For example, there is a publicly available 2-bit, 3-layer model with 64 neurons that achieves ~99% top-1 accuracy [2], substantially outperforming your illustrative baseline. This suggests the chosen motivating example does not meaningfully represent the current state of QAT.
2. The paper overlooks an important aspect of 2-bit and fewer QAT training dynamics: training from a fully converged checkpoint often underperforms compared to training from scratch or an intermediate checkpoint. Prior work such as ParetoQ explicitly analyzes this effect across initialization stages (0%–100%) and shows optimal performance at intermediate points. The omission of this consideration weakens the fairness of the comparison to ParetoQ, which is the primary baseline in the evaluation and performed best from an intermediate high-precision checkpoint.
3. The proposed theoretical framework for the quantization path is conceptually interesting but underdeveloped. It provides no clear explanation for why progressive quantization would be particularly important for instruction-tuned models, nor insight into what constitutes a “favorable” intermediate checkpoint. The claim that any intermediate 4-bit method works (L194–195) undermines the stated theoretical motivation—if simple rounding performs comparably, the framework’s added value beyond this is unclear.
4. The comparison using next-token prediction loss is not well aligned with the stated focus on instruction-tuned models. Starting QAT from an instruction fine-tuned checkpoint inherently makes next-token prediction less suitable, as it corresponds to the foundation model’s pretraining objective. A more appropriate baseline would use a post-training optimizer such as GRPO, enabling a fairer comparison against the distillation loss.
5. The evaluation and contextualization are incomplete. Prior work has demonstrated LLM quantization to 2 bits or below well before ParetoQ, including QAT (e.g., BitNet’s 1.58-bit models) and PTQ methods such as QuIP, FrameQuant, GPTAQ, Qronos, and QEP. The paper overlooks much of this literature, which narrows the perceived comparison space. While some of these techniques are contemporary, substantiating the claim that QAT is critical requires a direct, fair head-to-head comparison with recent state-of-the-art PTQ algorithms evaluated on the same calibration dataset.


References:

[1] Umuroglu et al. (2016), "FINN: A Framework for Fast, Scalable Binarized Neural Network Inference"

[2] https://github.com/fastmachinelearning/qonnx_model_zoo/tree/main/models/MNIST/Brevitas_FINN_LFC

### Questions
1. Can you evaluate your method against a 2-bit variant of BitNet?
2. Have you tested on other LLM architecture besides Llama?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a pipeline: FP16-> INT4 -> INT2, where the first stage uses PTQ, and the second stage uses QAT. They design distillation-based QAT Distill-QAT using JSD loss instead of the next-token prediction loss. Ablations show strong sensitivity to intermediate bitwidth, distillation objectives (GJS vs NTP).

### Strengths
1. The paper is easy to follow.
2. The ablations are insightful.

### Weaknesses
1. The motivation is not clear. Using INT4 PTQ as a good initializer does not guarantee that $\Delta W$ would decrease. 
2. Table 2-5 should also report standard deviation. 
3. The evaluations are within the Llama family. Broader evaluation should enhance the claim.
4. Lack of baselines for lower-precision PTQ methods. The author needs to quantify: how much gain does the QAT step actually add?
5. This paper's current scope is rather narrow, focusing too much on specific steps. Given its high sensitivity to intermediate bitwidth and distillation objectives, the reviewer recommends that this paper be reformulated as a general, progressive low-precision pipeline, and evaluating multiple intermediate bitwidths and target bitwidths.

### Questions
Will co-distillation from multiple higher-bitwidth models, such as 4-bit,8-bit models obtained via PTQ, further enhance the performance?

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
This paper propose UPQ, a specialized quantization methods for progressively recovering the performance of 2-bit instruction-tuned models. In the paper through ablation study they demonstrate the effectiveness of two-stage (FP16 -> 4bit -> 2bit) quantization.

### Strengths
- The motivation and method are clear: This paper focuses on a specific problem, recovering 2-bit instruction tuned LLM using only pretraining data.
- The factorization of model loss induced by quantization in equation (4) and the corresponding experiments in Figure 3 are inspiring.
-  The extensive experiments and ablation study supports the claim in the contributions.

### Weaknesses
- The application scenario of this method seems limited. UPQ aims to recover the performance of quantized instruction-tuned models using only pretraining data, but one may question why not directly fine-tuning on instruction following datasets, which may be more effective than the knowledge distillation in the second stage, espcially considering that raining on the reasoning traces of strong model is quite common today
- There is still a gap between UPQ and unquantized model even  UPQ uses re-training to recover model performance. For example, in figure 5, the scores on MMLU and IFEval of UPQ are only around half the the unquantized models. This raises the question that whether this 2-bit quantization is too aggressive. One may ask why not use a 4-bit model instead to maintain a model performance for practical use?

### Questions
Besides the questions in the weakness section, 

1. In Figure 5, what's the accuracy of quantized model before training? It might be helpful to annotate the starting point to show whether the model of various methods actually learn through re-training.  
2. I wonder if we don't use KD in the second stage, instead we fine-tune the 2-bit model, will the 4-bit casting serve as a good initialisation point? also how would it look like if we compare it against UPQ.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
UPQ targets INT2 weight‑only quantization for instruction‑tuned LLMs via a progressive path: FP16 $\to$ INT4 (PTQ) to obtain a good initializer, followed by INT4 $\to$ INT2 (QAT) with a distillation objective:
$\mathcal{L}{\mathrm{GJS}}!\left(p{\theta},|,p_{\mathrm{FP16}}\right)
= \lambda, D_{\mathrm{KL}}(p_{\theta},|,m) + (1-\lambda), D_{\mathrm{KL}}(p_{\mathrm{FP16}},|,m),;
m=\lambda p_{\theta} + (1-\lambda) p_{\mathrm{FP16}}.$
Ablations over quantizers, intermediate bit‑widths, and datasets argue that progressive quantization and data mix are crucial to retain instruction‑following performance.

### Strengths
1. Clear, reproducible recipe (PTQ checkpoint $+$ distillation‑QAT) with strong INT2 results.

2. Solid ablations: quant grids, intermediate precision, and data mixing.

3. Addresses an important deployment target (2‑bit weights) for edge/latency‑sensitive scenarios.

### Weaknesses
1. Compute cost. QAT with distillation at INT2 may be resource-intensive; wall-clock/training cost reporting would help.

2. Task breadth. Beyond MMLU/IFEval, more reasoning/safety/code tasks would test robustness.

3. Deployment details. INT2 packing/throughput on diverse hardware (mobile NPUs, edge GPUs) could be profiled.

### Questions
1. What is the best practice when QAT budgets are small—would INT3 or mixed‑precision layers be preferable?

2. How does performance change with activation quantization (e.g., W2A4), including KV‑cache precision?

3. Can layer‑wise or token‑wise temperatures improve the $\mathcal{L}_{\mathrm{GJS}}$ distillation?

4. Any evidence on multilingual or code‑heavy benchmarks and out‑of‑distribution robustness?

### Soundness
3

### Presentation
3

### Contribution
4
