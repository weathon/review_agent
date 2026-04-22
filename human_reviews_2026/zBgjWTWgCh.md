# Exploring Expert Concentration for Parameter-efficient Fine-tuning of Mixture-of-Expert LLMs

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 6, 4, 8

## Abstract
Scaling large language models (LLMs) with the Mixture-of-Experts (MoE) architecture has emerged as a powerful alternative to dense models. However, fine-tuning MoE models for domain- or task-specific adaptation remains challenging: full-model tuning is prohibitively expensive, while existing parameter-efficient fine-tuning (PEFT) methods, mostly adapted from dense models, suffer from unstable optimization due to MoE’s sparse expert activation. In this work, we conduct an empirical study on the fine-tuning dynamics of MoE models. We first introduce the Domain Advantage Score (DAS), a simple yet effective metric for identifying domain-relevant experts. Our findings uncover an expert concentration phenomenon: during domain-specific fine-tuning, the overall DAS of the top experts consistently increases, indicating a progressive enhancement of domain concentration. Building on this, we propose a lightweight two-stage PEFT framework: (1) fine-tuning only the attention and router layers to sharpen expert specialization, and (2) selectively fine-tuning parameters on the identified experts. This approach updates only a small fraction of parameters while achieving performance on par with full fine-tuning, and it effectively preserves the model's general capabilities. Experiments on nine benchmarks show the effectiveness and efficiency of our method. Our code and data will be publicly released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper observes that fine-tuning MoE models naturally concentrates domain knowledge into a small subset of experts, with routing scores increasingly favoring these specialists over training. Building on this insight, they design a lightweight two-stage method that first sharpens routing (attention + router tuning) then selectively updates only high-scoring experts based on their proposed Domain Advantage Score.

### Strengths
The core insight about expert concentration is valuable and well-supported empirically. The two-stage design follows naturally from this observation, and the method delivers strong results across different MoE architectures. The ablations effectively isolate the contribution of each component, and the analysis of expert distribution across layers adds useful interpretability.

### Weaknesses
(1)  The experiments mainly focus on mathematical reasoning and coding tasks, which are relatively structured domains with clear problem-solving patterns. It would strengthen the paper to evaluate the approach on more diverse tasks like open-ended generation, multilingual understanding, or knowledge-intensive QA. The current scope leaves open questions about whether the expert concentration phenomenon generalizes across different types of domain adaptation scenarios.

(2) While the Domain Advantage Score works well empirically, the paper doesn't deeply explore why this specific formula (difference of average routing scores) is optimal. Have the authors considered alternatives like KL divergence between domain and general distributions? 

(3) All experiments are conducted on relatively smaller MoE models (2.7B to 16B parameters). Modern MoE models can be much larger like Qwen-30B-A3B or Qwen-80B-A3B. Does the expert concentration phenomenon hold at larger scales? Does the ratio of parameters to fine-tune (8%) remain constant, or should it be adjusted? Some discussion or preliminary results on scaling behavior would be valuable.

(4) The paper compares against FFT, LoRA, and ESFT, but recent work on MoE adaptation has proposed other methods. For example, there's no comparison with adapter-based approaches specifically designed for MoE, or more recent routing-aware PEFT methods.

### Questions
While the paper emphasizes parameter efficiency (8% trainable parameters), the actual training time and memory cost comparison is missing. The two-stage approach requires training twice and computing DAS scores, which adds overhead. Does this two-stage process actually save training time compared to methods that train fewer parameters in a single stage?

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
The paper studies parameter-efficient fine-tuning (PEFT) for Mixture-of-Experts (MoE) language models and identifies instability caused by sparse expert activation. It introduces the Domain Advantage Score (DAS) to measure domain relevance of experts and observes an expert concentration phenomenon during fine-tuning. Based on this, a two-stage PEFT strategy is proposed: first tuning attention and router layers, then selectively fine-tuning domain-relevant experts. Extensive experiments on nine benchmarks show that the method achieves performance comparable to full fine-tuning while updating only a small fraction of parameters.

### Strengths
1. The paper presents a novel and well-motivated empirical finding that MoE models exhibit an expert concentration phenomenon, where domain-specific fine-tuning naturally focuses activation on a subset of experts, offering valuable insight into MoE behavior.

2. The proposed method creatively identifies domain-specific experts and applies a two-stage PEFT strategy—first tuning attention and router layers, then selectively fine-tuning expert modules—to achieve efficiency without sacrificing performance.

3. The experiments are extensive and well-designed, covering both domain-specific datasets and general benchmarks, effectively demonstrating the method’s strong performance and resistance to catastrophic forgetting.

4. The paper is well-structured and easy to follow.

### Weaknesses
1. The MoE base models used in the experiments are relatively outdated and all within a similar size range (around 16B parameters). Many newer MoE architectures of varying scales (e.g., Qwen3-30B-A3B, OLMoE-1B-7B) could be tested to provide a more comprehensive evaluation, and the lack of such experiments limits the generality and modern relevance of the findings.

2. In the Feature Analysis, all analyses are conducted on

### Questions
1. After Stage 1 training, certain experts are more frequently activated for specific domains. During Stage 1, did the authors include a load-balancing loss in the training objective? If so, could they provide an analysis of why load balancing fails to prevent expert over-activation? If not, can the authors demonstrate that the observed concentration of activated experts is not simply a consequence of omitting load balancing?

2. In Stage 2, the authors compute DAS values across all experts to rank their domain affinity and retain only the top-k experts. Why was this global top-k selection strategy chosen instead of retaining a fixed proportion of high-DAS experts within each layer?

3. Could the authors provide additional experimental results using newer or larger MoE models to further validate the generality and scalability of the proposed approach?

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
4

### Summary
The paper proposes the Domain Advantage Score (DAS) metric to evaluate experts' relevance to their corresponding domains. Based on this metric, it introduces a lightweight two-stage PEFT framework: first adjusting attention and routers, then selectively fine-tuning expert modules. This achieves near-full-fledged accuracy using only a fraction of parameters.

### Strengths
1. The paper proposes the DAS metric and a two-stage PEFT framework to achieve fine-tuning results approaching full model accuracy.
2. Ablation experiments were conducted to validate the effectiveness of the DAS.
3. The model achieves results close to full fine-tuning by fine-tuning only 8% of its parameters.
4. Continuous learning demonstrates strong performance with minimal catastrophic forgetting.

### Weaknesses
1. DAS relies on differences in routing scores, yet these scores themselves are influenced by model initialisation, data bias, and noise, potentially lacking scalability.
2. The primary contribution of this paper lies in fine-tuning efficiency. Although only 8% of parameters are fine-tuned, this does not imply that only 8% of required GPU memory is needed. The entire model must still be loaded into GPU memory; savings are limited to the memory occupied by gradients and optimiser states. Therefore, I believe the authors should incorporate analyses of peak memory usage and training time or acceleration ratios.
3. The authors assert that their method maintains consistent training budgets with other approaches. Does this refer to matching the number of training steps in the second stage, or the total number of steps across both stages?
4. Experiments on continuous learning were conducted solely with the proposed method. Additional baselines may be required to demonstrate its advantages in continuous learning scenarios.

### Questions
1. Can the authors provide experimental analysis incorporating metrics such as peak memory usage, training duration, or acceleration ratios?
2. Can the authors elaborate on the specific allocation of training budget within the two-stage training process?
3. Can the authors furnish comparative experiments demonstrating continuous learning performance against other baselines?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses the problem of parameter-efficient fine-tuning (PEFT) for Mixture-of-Experts (MoE) large language models. The authors argue that directly applying PEFT methods designed for dense models to MoE architectures is suboptimal due to training instability caused by sparse expert activation. To tackle this, the paper first conducts an empirical study, revealing an expert concentration phenomenon during domain-specific fine-tuning, where computation for a specific domain gradually concentrates on a small subset of experts. Building on this finding, the authors propose a simple yet effective metric, the Domain Advantage Score(DAS), to quantify and identify these domain-relevant experts. Subsequently, they introduce a lightweight two-stage fine-tuning framework. Experiments conducted on multiple math and coding benchmarks demonstrate that this approach achieves performance comparable to full fine-tuning with only about 8% of the trainable parameters. Furthermore, it effectively mitigates catastrophic forgetting of the model's general capabilities.

### Strengths
1.Solid Empirical Analysis: The paper's primary strength lies in its in-depth investigation of MoE fine-tuning dynamics. Instead of merely proposing a new method, the authors first uncover the core phenomenon of "expert concentration" through empirical study and design the DAS metric to quantify it. This foundational analysis provides a strong theoretical and experimental basis for the proposed method.
2.Comprehensive and Convincing Experiments: The empirical evaluation is exceptionally thorough, including ablation study, variation study and so on. 
3.Clarity and Presentation: The paper is well-structured, clearly written, and easy to follow.

### Weaknesses
1. The calculation of DAS relies on a "general dataset" (Dg) to contrast with the target domain data. The paper does not elaborate on the selection criteria for this dataset or the sensitivity of the method to its choice.

### Questions
1. How was the general dataset (Dg) selected for the experiments? Is the performance of the method sensitive to the choice of Dg?

### Soundness
3

### Presentation
3

### Contribution
3
