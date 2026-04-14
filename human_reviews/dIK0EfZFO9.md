# Low-Rank Quantization-Aware Training for LLMs

- Decision: Reject
- Scores: 5, 5, 8, 6

## Abstract
In this paper we propose LR-QAT – a lightweight and memory-efficient QAT algorithm for LLMs. LR-QAT employs several components to save memory without sacrificing performance: (a) low-rank quantization-aware reparameterization; (b) downcasting operation using fixed-point or double-packing and (c) checkpointing. Unlike most related work, our method (i) is inference-efficient, leading to no additional overhead compared to traditional PTQ; (ii) can be seen as a general extended pre-training framework, meaning that the resulting model can still be utilized for any downstream task afterwards; (iii) is orthogonal to most of recent PTQ methods and thus can be seamlessly combined with them. We apply LR-QAT to the LLaMA-1/2/3 and Mistral model families and validate its effectiveness on several downstream tasks. Our method outperforms most of recent LLM quantization approaches and reaches the same model performance as full-model QAT at the fraction of its memory usage. Specifically, we can train a 7B LLM on a single consumer grade GPU with 24GB memory.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The paper proposes Low-Rank Quantization-Aware Training (LR-QAT) to address the memory and computational inefficiencies of traditional quantization-aware training (QAT) methods when applied to large language models (LLMs). LR-QAT integrates low-rank adapters into the quantization process and utilizes fixed-point representations and gradient checkpointing to minimize memory usage. Applied to several LLM architectures, LR-QAT achieves model performance on par with full-model QAT, while being resource-efficient and inference-friendly, making it feasible for deployment on consumer-grade GPUs.

### Strengths
1- **Efficiency in Memory and Computation**: The approach effectively reduces the memory requirements for QAT, enabling large models to be trained on a single 24GB GPU without compromising predictive performance.

2- **Inference Efficiency**: Unlike many related low-rank and quantization methods, LR-QAT allows the fused low-bit pretrained model to be used directly in inference without additional overhead, which is a practical advantage.

3- **Comprehensive Experimental Validation**: The authors evaluate LR-QAT on multiple LLM families and bitwidth settings, demonstrating versatility across different quantization and low-rank configurations.

4- **Analytical Justifications**: The paper offers theoretical reasoning for observed outcomes, such as the choice of downcasting operators and low-rank approximation within the quantization operation, which contributes to understanding the method's impact on LLM quantization.

### Weaknesses
1- **Limited Comparison in Initialization**: The paper relies on round-to-nearest (RTN) for initializing quantized weights without extensive exploration of alternatives. AdaRound [1] has shown that RTN is not necessarily and optimal quantization scheme, and even searching over random quantization schemes can lead to better results. The authors even pointed our using other method such as OmniQuant as a possible initialization scheme.

2- **Lack of Complete Results in Tables**: In Tables 4 and 5, several cells contain dashes (“-”), implying missing data. This restricts fair comparison across methods and limits insight into LR-QAT’s performance in certain configurations.

3- **Potential for Limited Practical Impact**: In cases where LR-QAT's fine-tuning time is significant, the marginal performance gain over zero-shot methods may not justify the overhead, questioning the scalability and applicability of LR-QAT in high-stakes deployments. A comparison between the quantization time required for different methods can help shed light to the trade off between different methods (in addition to ther fine-tuning timing results reported in the appendix).

4- **Fixed Point Representation Modeling**: I believe that the fixed point representation of the weights can be modeled using the integer representation with tuning of scaling parameters. This could potentially yield better accuracy and flexibility without increasing the complexity of implementation.

**Typos and Minor Issues**
- Please ensure consistent notation for mathematical symbols throughout equations.
- Avoid using “paragraph” as a placeholder, as seen in Section 2.

[1] Nagel et al., "Up or Down? Adaptive Rounding for Post-Training Quantization," ICML 2020

### Questions
1- **Choice of $s_0$ in Equation 5**:  Could the authors clarify how  $s_0$ is chosen? Is it based on a range estimation technique or empirical tuning?

2- **Scaling Parameters for Integer Representation**:  I believe that the fixed point representation of the weights can be modeled using the integer representation with tuning of scaling parameters. Have the authors experimented with tuning the scaling parameters?

3- **Ignoring Quantization in Backward Pass**: Would omitting quantization for weights and low-rank adapters during the backward pass yield notable memory or computation savings?

4- **Trade-offs with Gradient Checkpointing**: I'm wondering what the effects of ignoring quantization of the weights (and low-rank adapters) are in the backward pass. This could lead to both memory and compute savings in the backward pass, since it allows efficient computation of low-rank multiplications.

5- **Adapter Rank and Dataset Size Scaling**: Based on Figure 2, the perplexity of the model w.r.t. the adapter rank is almost flat, and has perturbations in higher ranks. As the authors have pointed out, that might be an issue with the size of the fine-tuning set. I suggest the authors use scaling laws for finding the dataset sizes for different low-rank adapters and check if that solves the inconsistency with these results.

6- **Alternative Initializations for Quantized Weights**: Has the paper considered initialization methods beyond RTN? OmniQuant could offer improved results by leveraging adaptive rounding and error minimization techniques, respectively.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
Inspired by parameter efficient fine-tuning literature, this paper proposes LR-QAT – a lightweight and memory-efficient QAT algorithm for LLMs. LR-QAT employs several components to save memory without sacrificing predictive performance: (a) low-rank quantization-aware reparameterization; (b) downcasting operation using fixed-point or double-packing and (c) checkpointing.

### Strengths
LR-QAT introduces and combines several innovations designed to reduce memory use without sacrificing model performance: (1) a form of QAT with low-rank reparameterization, in which it places the low-rank weights in the integer domain to ensure they align with the quantization grid of the pretrained weights. This allows for seamless fusion during inference into a single low-bit integer matrix. (2) A downcasting operator that represents the frozen pretrained weights as low-bit INT-b (b ≤ 4) double-packed into INT8 or as fixed-point values stored in INT8. 

It applies LR-QAT to LLaMA-1/2/3 and Mistral model families and demonstrate its effectiveness on several general language modeling datasets and zero-shot evaluation on some of the common reasoning downstream tasks. The method outperforms recent LLM quantization approaches and reaches the same predictive performance as full-model QAT at the fraction of its memory usage.

### Weaknesses
The novelty may be limited. The proposed method combines traditional quantization and LoRA methods. The downcasting operator cast the input to one of pre-existing floating-point formats which follows previous works such as  (Oberstar, 2007) and  (Li et al., 2023).  The gradient checkpointing mainly follow the previous work (Chen et al., 2016). The technical contribution may be limited. 

It only has results which finetune on a single dataset. It is better to demonstrate the performance by finetuning on different datasets. For example, QLoRA and QALoRA finetunes on two different datasets. Finetuning on multiple different datasets can demonstrate the performance in general cases. If finetuning on only one single dataset, we are not sure about the general performance of the proposed method on other datasets. 

There are some typos such as  'leading to a subpar performance. paragraph' in Line 180. 

The claim of 'an extended pretraining method' for this work does not seem to be solid. It seems that 'task-specific fine-tuning' or 'extended pretraining' just depends on which dataset is used for further training. Using downstream datasets is  task-specific fine-tuning    and using pretraining dataset is the so called extended pretraining.  This difference is minor and the proposed method or the baselines can switch the finetuning datasets.  Thus these methods can either be task-specific fine-tuning or extended pretraining. It does not seem to be very difficult to switch the datasets.  It is better to provide more discussion about the claim of extended pretraining. 

The method is very similar to QA-LoRA. In QA-LoRA, weights merged with LoRA are still in a quantized form so that LLMs can be deployed with computational efficiency. The core idea of has been investigated in QA-LoRA. It is better to have a more detailed comparison with QA-LoRA to discuss the difference. It is better to compare with QA-LoRA in experiments.

### Questions
See the weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose a Low-Rank Quantization Aware Training (LR-QAT) algorithm to train LLM model on single consumer grade GPU. In particular, LR-QAT is based on quantizing low-rank adapters (LoRA) such that it could be fused into quantized frozen pre-trained weight. In addition, it uses a downcasting operator for frozen pretrained weight and gradient checkpointing for tackling increased memory usage owing to product computation of low rank adapters. In contrast to other works on Post training quantization, Quantization-aware training of full models, and PEFT-based (LoRa) quantization schemes, LR-QAT is geared towards better accuracy, memory-efficiency and inference-efficiency (via fusion of low-rank adapter to low-bit integer frozen weight)

### Strengths
1. The paper is well-written with adequate background and related work.

2. The paper combines LoRA + QAT techniques and applies down casting and gradient checkpointing for memory-efficient and inference-efficient LLMs

3. The key innovation lies in seamless fusion of the quantized low-rank adapters with the quantization field of frozen retrained weight unlike other LoRA inspired quantization works.

4. The empirical analysis is extensive with sufficient comparisons and ablation studies with impact of rank, and down casting operation.

### Weaknesses
1. Figure 1 probably is not referenced in the main text. Probably a deeper discussion on Fig. 1 (right) is needed with respect to goal of training LLMS on single device with 24GB.

2. It might be helpful to have a comparison table or figure that depicts the what parts of the model are being quantized and quantization scheme used across various PTQ, QAT and LoRa-inspired works referenced and compared in experiments - this would help put related works in perspective.

3. LoRA is predominantly used for fine-tuning as part of PEFT methods, but in many places, the authors use training, pertaining and fine-tuning with LoRA interchangeably. It would be helpful to be consistent with keywords related to LoRA and proposed LR-QAT. eg. Line 187, should it be fine-tuning compared to training, title of the work, quantization-aware training or pretraining of LLMs when authors claim LR-QAT to be general purpose extended pretraining method versus fine-tuning method

4. PTQ and LR- abbreviations used in abstract without full form.

5. Line 48: Please specify which is most related work the authors are contrasting with, since they referenced both PEQA (Line 176) and QA-LoRA (Line 195) to be closest to theirs.

6. Where does PEQA fit in Table 1?

7. Lines 049 and 177: Define inference efficiency of QAT here earlier than Lines 201-202

8. Line 65: Define r as "rank" 

9. Line 193: Typo "is" ..This issue.....

### Questions
The questions have been shared in earlier section"Weakness"

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
This paper introduces LR-QAT, a quantization-aware training framework designed for large language models that optimizes memory efficiency without compromising model performance. It incorporates low-rank quantization-aware reparameterization, downcasting, and checkpointing techniques. The proposed technique reduces memory usage significantly often outperforming PQT techniques while allowing integration with such techniques. Additionally, it also does not incur any additional overhead during inference.

### Strengths
The paper is well-written and explained clearly. The approach is well-supported by a thorough experimental section, showcasing promising results that validate its efficiency and robustness across multiple large language models.

### Weaknesses
The novelty of the approach is somewhat limited as it incorporates several known techniques rather than introducing entirely new concepts. While the paper provides detailed memory and runtime comparisons with full-model QAT (e.g., LSQ), it does not compare these metrics against other implemented baselines like LSQ, or OmniQuant, which limits the assessment of its relative efficiency.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
