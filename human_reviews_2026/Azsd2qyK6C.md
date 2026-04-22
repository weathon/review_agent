# Towards Quantization-Aware Training for Ultra-Low-Bit Reasoning LLMs

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Large language models (LLMs) have achieved remarkable performance across diverse reasoning tasks, yet their deployment is hindered by prohibitive computational and memory costs. Quantization-aware training (QAT) enables ultra-low-bit compression (<4 bits per weight), but existing QAT methods often degrade reasoning capability, partly because complex knowledge structures are introduced during the post-training process in LLMs. In this paper, through a systematic investigation of how quantization affects different data domains, we find that its impact on pre-training and reasoning capabilities differs. Building on this insight, we propose a novel two-stage QAT pipeline specifically designed for reasoning LLMs. In the first stage, we quantize the model using mixed-domain calibration data to preserve essential capabilities across domains; in the second stage, we fine-tune the quantized model with a teacher-guided reward-rectification loss to restore reasoning capability. We first demonstrate that mixed-domain calibration outperforms single-domain calibration by up to 2.74% improvement on average over six tasks, including reasoning and pre-trained tasks. Following experiments on five reasoning benchmarks show that our 2-bit-quantized Qwen3-8B outperforms post-training quantization (PTQ) baselines by 50.45% on average. Moreover, compared to ultra-low-bit-specialized models such as BitNet-2B4T, our pipeline achieves approximately 2\% higher mathematical-reasoning accuracy with fewer than 1B tokens. Code is available: https://github.com/yasu0001/ReasoningQAT

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a reasoning-oriented quantization-aware training (QAT) framework for ultra-low-bit (<4-bit) large language models (LLMs). The method introduces a two-stage pipeline combining mixed-domain calibration (pre-training + reasoning data) and a teacher-guided reward-rectification loss to preserve reasoning capability. Experiments on Qwen3 models (1.7B–8B) show substantial gains over PTQ baselines (GPTQ, AWQ) and competitive performance against BitNet-2B4T with far fewer training tokens.

### Strengths
Addresses an important and timely problem: maintaining reasoning ability under aggressive quantization.

Clear motivation and well-structured method.

Extensive empirical validation across reasoning benchmarks.

Ablation studies convincingly support the proposed components.

### Weaknesses
The approach is only evaluated up to 8B models; scaling to 70B+ remains unverified, though critical for real deployment.

The paper focuses on accuracy but lacks quantitative analysis of training cost and hardware efficiency (e.g., fine-tuning time, energy, or throughput gains).

Some comparisons (e.g., to full-training QAT methods like BitDistiller or Spectra) could be expanded to clarify relative trade-offs.

### Questions
How does the proposed teacher-guided loss scale computationally when the teacher is a large fp16 model?

Could mixed-domain ratios be dynamically adapted during training?

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
This paper proposes a reasoning-oriented quantization-aware training (QAT) pipeline for ultra-low-bit (2–3 bit) LLMs.
The key idea is that reasoning abilities (acquired in post-training) and commonsense knowledge (from pre-training) respond differently to quantization.
To address this, the authors design a two-stage QAT framework:

Mixed-domain calibration (80% reasoning data + 20% pre-training data) for balanced preservation of knowledge and reasoning.

Teacher-guided reward-rectification fine-tuning, a reinforcement-inspired objective that restores reasoning accuracy efficiently.

### Strengths
The paper addresses an important and underexplored problem—maintaining reasoning capabilities during ultra-low-bit quantization.

The domain-sensitivity analysis (reasoning vs. pre-training) is interesting and helps motivate the mixed calibration design.

The pipeline is easy to follow and grounded in recent literature; figures and tables are clear and convincing.

### Weaknesses
The paper emphasizes accuracy but lacks analysis of training cost, throughput, and latency during inference—key aspects for practical quantization.

The 80/20 calibration ratio is fixed; it would be useful to see sensitivity analysis across ratios or datasets to validate robustness.

Other reasoning metrics, such as thinking token count, is not analyzed, which might reveal qualitative differences induced by quantization.

### Questions
When evaluating reasoning LLMs, task accuracy alone may not capture the full effect of quantization. Could you also report metrics related to thinking tokens—for example, the number of intermediate reasoning or "thought" tokens generated during inference?
In particular, does quantization (especially at 2 bits) affect how long the model needs to “think” or the length of its reasoning chain before reaching an answer?

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
3

### Summary
This paper analyzes the challenge of preserving reasoning capability under ultra low bit quantization and proposes a mixed calibration data strategy together with a two stage QAT pipeline. The method mixes reasoning and pre training data for calibration and then uses a teacher guided reward rectification loss in fine tuning to restore reasoning ability.

### Strengths
The paper studies an important issue of maintaining reasoning capabilities under aggressive quantization. 

The analysis reveals that reasoning data has higher sensitivity to quantization than pre training data, and the mixed calibration design is logically motivated.

### Weaknesses
The novelty appears limited. The performance gain mainly comes from applying distillation on reasoning data, as shown in Table 4, where the KL or distillation term dominates the improvement and the proposed reward rectification formulation contributes marginal benefit. Distillation is a existing technique in low bit QAT.

The experimental comparison is not convincing. The main table (table 2) only compares with GPTQ and AWQ which are not QAT methods. Fairness requires comparison with recent QAT approaches, particularly distillation based ones such as EfficientQAT and BitDistiller. 

Some claims lack evidence. The paper states that RL methods incur large autoregressive training overhead, but there is no quantitative analysis or comparison against RL based fine tuning for QAT. Similarly the equivalence to on policy learning is mentioned, but no evidence is provided to show practical benefit over standard supervised tuning.

Even with distillation, the model still shows a significant drop from BF16 performance on challenging tasks such as LiveCodeBench, suggesting that reasoning capability is not fully preserved and further improvements are required.

### Questions
Even with strong distillation, ultra low bit models still lag FP16 on tasks like LiveCodeBench. Are there other opportunities to further narrow this gap?

Can the authors compare against efficient RL post training methods, for example DPO variants, to validate the claim that RL cost is prohibitive and the proposed loss is a more efficient substitute?

How does the training cost of the proposed method compare with EfficientQAT and BitDistiller in terms of GPU hours?

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
4

### Summary
This paper addresses the significant challenge of performance degradation in large language models (LLMs) on reasoning tasks when subjected to ultra-low-bit (<4 bits) quantization. The authors hypothesize that this degradation stems from the fact that quantization disproportionately affects the complex knowledge structures introduced during post-training (e.g., SFT, preference optimization) compared to the commonsense knowledge acquired during pre-training.
To counter this, the paper proposes a novel two-stage Quantization-Aware Training (QAT) pipeline:

- Stage 1 (Calibration): Performs an initial block-wise quantization using a mixed-domain calibration dataset (80% reasoning, 20% pre-training) to preserve both reasoning capabilities and general knowledge.

- Stage 2 (Fine-Tuning): Fine-tunes the quantized model using a novel objective function called the "teacher-guided reward-rectification loss". This loss reweights the standard supervised fine-tuning (SFT) loss using the probabilities from the full-precision teacher model, combined with a KL divergence term. This is proposed as an efficient, offline alternative to RL-based methods.

Experiments on Qwen3 models show that this pipeline dramatically outperforms standard post-training quantization (PTQ) baselines like GPTQ and AWQ, especially at 2-bit precision, and also shows competitive results against the specialized BitNet-2B4T model.

### Strengths
- Clear Problem Statement and Novel Insight: The paper targets a critical problem (quantization failure on reasoning) and provides a well-articulated insight: the differential impact of quantization on pre-training vs. post-training knowledge (validated in Fig 2a).

- Well-Motivated Methodological Components:
  - The mixed-domain calibration (Stage 1) is a simple, effective, and well-supported solution (Table 1) to the domain-sensitivity problem.
  - The teacher-guided reward-rectification loss (Stage 2) is a clever modification of SFT that correctly identifies the unreliability of the quantized student's probabilities and leverages the stable, full-precision teacher.

- Impressive and Significant Empirical Results: The performance gains, especially for 2-bit quantization, are dramatic (Table 2). The method successfully retains performance (e.g., 55.07% avg for 8B) where SOTA PTQ methods (GPTQ, AWQ) fail completely (~3-5% avg). This is a significant practical achievement.

- Strong Ablation Studies (Section 4.4): The ablations provide excellent support for the method's design. Table 3 compellingly demonstrates that both the calibration stage (C) and the proposed reward-rectification loss (R) are essential for success.

### Weaknesses
- Regarding Baseline Comparisons (QAT vs. PTQ): A primary concern is the comparison between the proposed QAT method and the PTQ (GPTQ, AWQ) baselines. While the results are impressive, it is generally expected that QAT methods (which involve fine-tuning) will outperform PTQ methods, especially at ultra-low bit-widths. The paper would be more convincing if it included a direct comparison against other state-of-the-art QAT methods (like EfficientQAT or UPQ, which are cited). The authors mention that other methods target different settings, but the paper would be significantly strengthened if the authors could reproduce at least one of these QAT baselines in their own evaluation framework. This would provide a more direct, like-for-like comparison.

- Regarding the Justification for the Proposed Loss: The motivation for the "teacher-guided reward-rectification loss" is its efficiency compared to online RL methods. This is a valid point. However, we note that online, on-policy RL methods (like PPO) derive their generalization benefit from learning on the model's own distribution. The proposed method is an offline, off-policy approach. It would be very helpful if the authors could provide more discussion or empirical evidence to support the claim that this offline approach achieves comparable generalization to the online methods it seeks to replace. This is a central and very interesting claim, and validating it (perhaps with an ablation against a DPO or PPO baseline) would make the contribution even stronger.

- Minor Clarification on Training Data: There appears to be a slight inconsistency in the reporting of training data, which could be easily clarified. The abstract mentions "40K training sequences", Section 4.1 lists "32,768 samples", and Table 5 notes "0.2M" and "0.8M" tokens. We would appreciate it if the authors could clarify the relationship between these numbers in the final version to avoid potential confusion.

### Questions
- Regarding the baseline comparisons (Weakness 1), the authors are strongly encouraged to include a direct comparison against at least one SOTA QAT method (e.g., EfficientQAT, UPQ). This would substantially bolster the paper's claims.

- To validate the claims about generalization (Weakness 2), it would be highly beneficial to add an ablation study that compares the proposed off-policy loss to a standard online RL baseline, such as PPO or DPO, on the same reasoning benchmarks.

- In the ablation for the loss function (Table 4), the KL divergence term seems to be heavily weighted $(\alpha=0.2, \beta=1.0)$. This suggests much of the performance gain might come from simple distribution alignment. We would be interested to see more sensitivity analysis on the alpha parameter.

- Given that "deployment" is a key motivator, the paper would be more complete if it included practical metrics like inference latency (tokens/sec) or memory footprint measurements, which are currently missing.

### Soundness
3

### Presentation
3

### Contribution
3
