# TRAC: Tensor-Train based Across-layer Compression for Parameter-Efficient Fine-Tuning

- Decision: Accept (Poster)
- Scores: 8, 6, 0, 2

## Abstract
Fine-tuning large pre-trained models under resource constraints remains challenging due to the massive number of parameters involved. Existing parameter-efficient tuning methods, such as low-rank adaptation (LoRA) and its variants, rely heavily on matrix factorization and often struggle in extremely low-parameter regimes. In this work, we propose TRAC, a novel fine-tuning framework that leverages Tensor-Train decomposition with Across-layer Compression. Specifically, TRAC represents each adaptation module as a compact sequence of tensor-train cores and allows certain cores to be frozen or shared across layers, thereby exploiting the inherent similarity and redundancy among layer weight matrices. To retain layer-specific flexibility, lightweight controllers are introduced, enabling shared tensor cores to adaptively modulate representations. We evaluate TRAC on diverse architectures, including Qwen, LLaMA, GPT, BERT, and ViT, across benchmarks covering text classification, text generation, and image classification. Experimental results demonstrate that TRAC achieves performance comparable to or better than LoRA and its variants, while substantially reducing trainable parameters and storage requirements.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces TRAC (Tensor-Train based Across-layer Compression), a framework for parameter-efficient fine-tuning (PEFT) of large pre-trained models using Tensor-Train (TT) decomposition. TRAC combines tensor decomposition with across-layer parameter sharing and freezing, incorporating lightweight controllers to enhance adaptability. The paper evaluates TRAC on diverse architectures, including LLaMA-2, GPT, BERT, and ViT, across tasks such as text classification, text generation, and image classification. The results demonstrate that TRAC achieves comparable or superior performance to existing methods (e.g., LoRA, VeRA, and LoRETTA) with significantly fewer trainable parameters. The paper claims a reduction of up to 20× fewer trainable parameters for LLaMA-2-13B on SuperGLUE tasks while maintaining or improving performance.

### Strengths
1. The proposed method achieves up to 20× parameter reduction compared to LoRA in certain tasks while maintaining comparable or superior performance. This is particularly impactful for resource-constrained scenarios.
2. TRAC effectively combines Tensor-Train decomposition with cross-layer parameter sharing and freezing, which is a novel and well-motivated design. This approach directly addresses the redundancy in deep transformer layers.
3. The experiments are conducted across various architectures (e.g., BERT, LLaMA, GPT, ViT) and tasks (e.g., GLUE, SuperGLUE, E2E, image classification). The paper includes comparisons with strong baselines like LoRA, VeRA, and LoRETTA, supported by statistical analysis.

### Weaknesses
1. While the paper emphasizes the importance of lightweight controllers, the experiments do not deeply explore the trade-offs of different controller designs or activation functions. This limits the understanding of the controller's impact on performance.
2. The paper does not sufficiently discuss some related PEFT  methods, such as AdaLoRA[1], MPOP[2]. Including a broader discussion would enhance the positioning of TRAC in the literature.
3. While the paper claims that higher-order tensors do not improve performance, the exploration of tensor orders beyond third-order is limited to one or two configurations. A more detailed analysis would help justify this design choice.

Ref:
[1]. Zhang Q, Chen M, Bukharin A, et al. Adalora: Adaptive budget allocation for parameter-efficient fine-tuning[J]. arXiv preprint arXiv:2303.10512, 2023.
[2]. Liu P, Gao Z F, Zhao W X, et al. Enabling lightweight fine-tuning for pre-trained language model compression based on matrix product operators[J]. arXiv preprint arXiv:2106.02205, 2021.

### Questions
1. Could the authors provide more intuition about why gating shared cores with vectors (vs. low-rank residuals or elementwise modulation) is expressive enough? Have you explored alternative controller architectures (e.g., learned affine transforms, attention-based modulation)?
2. Could the authors include a more explicit guideline or heuristic for selecting tensor shapes under a fixed parameter budget? How sensitive is the performance to suboptimal tensorization choices?
3. Have the authors identified any regimes (e.g., early/late transformer blocks, deeper MoE architectures) where sharing harms performance? Would allowing partial sharing (e.g., blockwise) improve robustness? If not tested, can the authors comment on expected behavior?

### Soundness
3

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
The paper studies a new PEFT adapter based on tensor-train format initialization. Compared with widely used PEFT methods like LoRA, the proposed method greatly reduces the number of trainable parameters. Compared with existing tensor-train–based adapters, the proposed method further includes freezing and sharing designs to achieve better performance without compromising parameter efficiency. Experimental results on the BERT, LLaMA, and ViT model families demonstrate the effectiveness of the proposed method in balancing efficiency and performance.

### Strengths
- The paper is well-written and provides sufficient background knowledge on the tensor-train format, making it easy to follow.
- In addition to NLP tasks, the paper also considers vision classification tasks to demonstrate the effectiveness of the proposed method.
- The paper provides theoretical analysis on the effectiveness of the TRAC method.

### Weaknesses
- Even though the proposed method has not been explored before, there is sufficient prior work on freezing (e.g., AFLoRA) or sharing (e.g., ShareLoRA) parts of adapters. This work appears to mainly combine these ideas, which limits its overall contribution.
- I wonder if the authors could provide a more direct explanation for the motivation behind studying an even more parameter-efficient method than LoRA. In my opinion, further reducing parameters may not significantly decrease the training cost compared with standard LoRA methods.
- I am satisfied with the experiments on the BERT and ViT models. However, for the experiments on the LLaMA-2 model, the tested tasks seem too limited. I wonder if the authors have evaluated the proposed method on more challenging reasoning tasks, such as those used in the DoRA paper. These tasks might better align with today’s real-world PEFT setups. Additionally, all models considered in this paper are somewhat outdated; I would expect some experiments on more recent models.
- I am quite interested in the results regarding the symmetry analysis in Appendix A.3. Could the authors provide more intuition behind these results? Are there similar findings in LoRA-style works?

### Questions
- In the experiments, the authors set the LoRA rank to 4. However, I recall that the optimal setup for LoRA rank is typically 8 or 16. Would the performance gap between LoRA and TRAC increase if a higher LoRA rank were used in Tables 3 or 4?
- I wonder if the tensor contraction in TRAC leads to higher latency compared with the LoRA adapter. I noticed an increase in training time with larger tensor orders in Table 11.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper proposes TRAC (Tensor-Train based Across-Layer Compression), a parameter-efficient fine-tuning (PEFT) framework that combines Tensor-Train (TT) decomposition with cross-layer core sharing and freezing. Each LoRA module is represented as a sequence of TT cores (trainable, frozen, shared), and lightweight controllers are added to modulate shared cores per layer. The method is evaluated on BERT, LLaMA2, GPT-2, and ViT models across GLUE, SuperGLUE, E2E, and image classification benchmarks. The authors report comparable or slightly better performance than LoRA and LoRETTA while using 10–20× fewer trainable parameters.

While the method is clearly presented and theoretically consistent, the core idea is NOT novel. Similar hybrid parameterizations have been explored extensively in prior work-most notably COMPACTER (Karimi Mahabadi et al., NeurIPS 2021)—which achieves comparable or better results than full finetuning while training less than 0.05% of parameters, versus This approach substantially higher ratio (12-14%). Moreover, the evaluation is also limited to simple benchmarks, without results on reasoning, math, or code tasks that are now standard for assessing LLM generalization, benchmarks like MMLU, MMLU-pro, humaneval, MBPP, GSM8k, ... 

Overall, the contribution is not novel and not more efficient than prior work (compared to Neurips 2021), and the empirical evidence is insufficient to establish this method as a meaningful advance over existing parameter-efficient training methods.

### Strengths
The paper is well-written with detailed method descriptions, and consistent notation.
The proposed TRAC framework is technically sound.

The authors conduct comprehensive experiments on multiple architectures (BERT, GPT-2, LLaMA2, ViT) and datasets (GLUE, SuperGLUE, E2E, and image classification), demonstrating that TRAC achieves similar or slightly better performance than LoRA and LoRETTA with fewer trainable parameters. However, these benchmarks are rather outdated and evaluations on more recent benchmarks such as MMLU, MMLU Pro, humaneval, MBPP, GSM8k, ... are needed.

### Weaknesses
* Comparison to prior work. The paper fails to acknowledge or compare against COMPACTER (Karimi Mahabadi et al., NeurIPS 2021), which is a foundational and much stronger baseline. COMPACTER achieved similar or better results while fine-tuning only 0.047% of parameters, whereas TRAC trains several orders of magnitude more parameters to reach comparable performance. This omission significantly weakens the contribution and demonstrates limited awareness of prior literature.

* Lack of novelty and missing citations. The paper provides no substantial innovation beyond prior PEFT work. Its main components—Tensor-Train (TT) decomposition and cross-layer parameter sharing-have already been introduced in TT-LoRA (Anjum et al., 2024), LoRETTA (Yang et al., 2024), VeRA (Kopiczko et al., 2024), NoLA (Koohpayegani et al., 2024), and SharedAdapter (Poth et al., 2023). TRAC simply combines these established ideas without offering new theoretical insights or mechanisms.

- Questionable efficiency gains. Although TRAC reduces trainable parameters relative to LoRA, it is unclear whether this translates into any real compute or time savings. The paper does not report FLOPs, memory footprint, or wall-clock improvements.

- Weak empirical results. Gains over LoRA and LoRETTA are small and often within noise. Important baselines such as COMPACTER, and GaLore (https://arxiv.org/pdf/2403.03507) are missing, leaving the evaluation incomplete.

Limited experimental scope. The paper only tests on small or mid-size models (BERT, GPT-2, LLaMA2-7B) and saturated benchmarks (GLUE, SuperGLUE, ImageNet). There is no evaluation on modern reasoning, math, or code benchmarks (e.g., MMLU, GSM8K, MBPP, HumanEval), which are standard in current LLM research.

Overall assessment. TRAC appears to be an incremental reimplementation of prior TT-based and cross-layer methods, presented without proper acknowledgment of earlier work or competitive results. Its contribution is neither novel nor impactful compared to established baselines, particularly COMPACTER (Karimi Mahabadi et al., 2021).

### Questions
* Comparison with COMPACTER, and GaLore: how does the method compare in terms of performance and also #trainable parameters compared to existing work of compacter and GaLore? Without these, it is difficult to assess real performance advantages.

* Parameter efficiency vs. compute efficiency: While TRAC reduces trainable parameters, how does it affect FLOPs, memory usage, or wall-clock training time? Can you provide quantitative data on actual resource savings, if any?

* Evaluation scope: Do you plan to evaluate TRAC on modern reasoning, math, and code benchmarks (e.g., MMLU, GSM8K, MBPP, HumanEval)? Such tasks are now standard for assessing LLMs and would strengthen the empirical contribution.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposed a new parameter-efficient adaptation method for transformers. It uses tensor decompositions to target the extreme end of low-paramaters and further reduces the number of necessary parameters by sharing across layers.

### Strengths
* easy to understand and simple new PEFT method
* compares well to SOTA methods on several benchmarks while being more parameter-efficient
* releted works section is thorough and gives good overview

### Weaknesses
* the computational complexity analysis is a bit unclear. I assume this is for a) for training time and b) for the number of adapted parameters (this should be made more clear). But then VeRA's complexity seems wrong, as there's no n^2 complexity in its decompositions. How is this arrived at and what even is "computational order"?
* The importance of the theoretical analysis is unclear. The fact that more parameters -> more expressivity is trivial and the fact that sharing across layers can be used too. Does this yield any insights into how to use the adapter or how it should be designed? Otherwise this seems like an appendix without a use. 
* the PEFT method builds on existing works such as LoReTTA and several works that rely on frozen random matrices
* old models. By now, llama-2 and GPT2 benchmarks have become irrelevant and newer models should be used with correspondingly adapted benchmarks. examples include qwen, llama-3 and or vision-language models. 
* No mention of training times
* minor: use citations with \citep when referring to a paper

### Questions
* given a thorough hyperparameter evaluation protocol, can the authors provide results on more recent architectures such as llama-3? 
* provide clarity on the use of the theoretical results and why they are included
* Provide clarity on "computational order"
* Provide insights on training time & memory required for training experiments.

### Soundness
2

### Presentation
3

### Contribution
2
