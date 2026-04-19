# Replicate and Quantize: A Plug-and-Play Strategy for Load Balancing in Sparse Mixture-of-Experts LLMs

- Decision: Reject
- Scores: 5, 3, 5, 6

## Abstract
While the rapid increase in the number of model parameters poses significant benefits to the development of large language models (LLMs), computational costs are also raised. In order to tackle this difficulty, the sparse mixture-of-experts(SMoE) model was introduced to tackle LLM scaling by activating a subset of experts per input. Therefore, how to leverage the knowledge of multiple experts will be an important topic. Normally, in the most extreme scenario, employing a balanced expert allocation system will result in a time-saving of $n$ times compared to utilizing only a single expert. Thus, in this paper we (1) systematically analyzed the performance and functionality of each expert. (2) Introduced a metric to fill the blank of evaluating load balance for the sparse mixture-of-experts(SMoE) model, based on the observation. (3) Proposed a dynamic plug-and-play strategy that is both trainingless and near-lossless, effectively resolving the load balancing problem, in contrast to previous works that focused on training strategies.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This work provides a simple yet effective strategy for load balancing in MoE-based LLMs. Specifically, the authors first find the most heavy expert and the less important experts, and (a) replicate and quantize most heavy experts, (b) quantize less important experts. In experiments, the authors have deployed the proposed method on 4 MoE models, achieving comparable results with more balanced load among experts. In conclusion, the proposed model is sound and easy to deploy, while more in-depth evaluations and analyses should be conducted.

### Strengths
1)	The proposed idea is simple and sound.
2)	Overall, this work is well-organized and easy to follow.
3)	The authors have tested on various MoE base models.

### Weaknesses
1)	The central idea of this work is the replicate and quantize strategy. Firstly, an appropriate ablation study should be conducted to verify the effectiveness of both strategies on the most heavy experts and less important experts. Secondly, whether the selections of most heavy experts and less important experts are essential? If we further quantize other experts while maintaining the activated parameters, what will be the results?
2)	In Section 3.4, the authors merely give the importance score/load results on one task PIQA. Does this phenomenon also exist in other tasks and other MoE blocks? The authors are suggested to give more quantitative indicators (e.g., correlation coefficient on different settings) to support their claims.
3)	In Related work, an in-depth analyses on other load balance methods (and why they do not work well in the experiments) should be given.
4)	In experiments, although the authors claimed that “before that, we have tried to use the different tuning strategies to adjust the router mechanism to solve the load imbalance issues, Clearly, it does not work as we expected, and the part of the strategies emplifies the imbalanced distribution among the different experts”, which strategies are used and the corresponding results should be given. Currently, only using the raw setting as baseline is not sufficient.
5)	The experimental details are insufficient. For instance, the details of adopting the proposed method on DeepSeekMoE should be given. DeepSeekMoE adopts shared and specialized experts, and whether the shared experts are also used for replicate? Moreover, it also multiplies the number of experts, which shares the similar idea of the “replicate” heavy expert part in this work.
6)	The actual inference speed and cost should be given. Do all comparisons share the same activated parameters in inference?
7)	Typos, e.g., Page2, missing reference in the first paragraph.
8)	The scalability of the proposed method is encouraged to be evaluated or discussed.

After rebuttal, I raise the voting to 5.

### Questions
Refer to the questions in Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper proposes Replicate and Quantize, an inference-time strategy that aims to mitigate load imbalance on Mixture-of-Expert based Large Language Models. The author claimed that there exist differences between heavy-hitters and important experts in MoE models, and proposed to 1) quantize the least important expert for resource savings, and 2) replicate a quantized version of heavy hitters to reduce load imbalance. Results show that the authors' strategy improved load imbalance without reducing too much performance.

### Strengths
- This paper studies an interesting problem, which is the load imbalance of MoE LLMs in inference scenarios.
- The author conducted experiments on various tasks and model types, creating a comprehensive overview of the impact of the proposed strategy on the performance of the model.

### Weaknesses
- The paper's claim on its novelty, that the load imbalance of MoE models has not been studied for inference scenarios, is wrong. Plenty of research have focused on the inference scenario, such as [1], [2], [3], [4] and [5], and many of them have provided a thorough characterization of the workload already. Quantization of the MoE model has also been studied in [6]. The author should conduct **a more thorough review on the existing literature** and discuss the difference of this work with these existing ones.

- The Load Imbalance Score defined at Sec 3.1 is an ill-defined metric. The overall load in a dataset is not directly related to the inference performance of the MoE model. It is the load in a certain batch that would have a major impact on the robustness of the model (preventing OOM errors) and latency (of all-to-all communications).

- Algorithm 1 seems to be unnecessary. The search is quite straightforward.

- The authors' proposition, that the heavy-hitters are not necessarily the most important expert, seems to be *rebuked* by the presented data in Fig. 2. See questions below.

- The most important metrics, which are the inference performance of the proposed system (**memory consumption, latency, hardware utilization, and so on**), are not studied in this work. These are the most important reasons one would like to reduce the load imbalance.

- Line 216, "Wanda metric" has been referenced, but only formally defined on line 237.

- The paper is not well formatted. For example:
  - The citations are not correctly adapted to the ICLR format. e.g. Line 107 -- Jacobs et al. Jacobs et al..
  - Missing spaces ",or" on line 115.
  - Missing reference on line 120.
  - Missing spaces ".For" on line 406.
  
- The authors have not provided any code or reproducibility statement.


[1] Huang, Haiyang, et al. "Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference." arXiv preprint arXiv:2303.06182 (2023).

[2] Gale, Trevor, et al. "Megablocks: Efficient sparse training with mixture-of-experts." Proceedings of Machine Learning and Systems 5 (2023): 288-304.

[3] Kong, Rui, et al. "Serving MoE Models on Resource-constrained Edge Devices via Dynamic Expert Swapping." arXiv preprint arXiv:2308.15030 (2023).

[4] Li, Jiamin, et al. "Accelerating distributed MoE training and inference with lina." 2023 USENIX Annual Technical Conference (USENIX ATC 23). 2023.

[5] Hwang, Ranggi, et al. "Pre-gated moe: An algorithm-system co-design for fast and scalable mixture-of-expert inference." 2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA). IEEE, 2024.

[6] Kim, Young Jin, Raffy Fahim, and Hany Hassan Awadalla. "Mixture of Quantized Experts (MoQE): Complementary Effect of Low-bit Quantization and Robustness." arXiv preprint arXiv:2310.02410 (2023).

### Questions
- Fig. 2 shows that the most important expert is the expert that receives the most tokens. It seems like it is rejecting, instead of confirming the authors' proposition, that the heavy-hitters are not necessarily the most important expert. Wouldn't quantizing expert 3, the heavy hitter in this case lead to performance degradation?

- How does the router adapt to the case where the most important expert is replicated? Will it evenly distribute its tokens to each GPU device?

- How are the expert loaded on the GPU? Are the other experts completely unaffected?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces a plug-and-play approach (R&Q) for addressing load imbalance in Sparse Mixture-of-Experts models to improve computational efficiency without retraining. R&Q literally replicates heavily used experts in a quantized form and quantizes less important experts to maintain memory efficiency. Minimal impact on performance.

### Strengths
S1. No retraining is required. 

S2. Near-original accuracy (at least on classification / MCQ tasks)

S3. I like how they distinguished between heavy-hitter experts and important experts, which could easily be confused as the same. They also conducted experiments to show that these concepts are distinct, although there is some correlation between them.

### Weaknesses
W1. Weak presentation of algorithms and figures. Some figures lack any caption or explanation. In Algorithm 1, the choice of variable names is awkward and confusing. For example, l(x), count(expert_chosen), argmax(expert_num), EC, etc., need to be clarified with better names.

W2. Weak baseline. The baseline experiments were only conducted within their framework (ours vs. random vs. heavy-hitter). They lack comparisons with other techniques that address load balancing. 

W3. Weak empirical analysis on computational efficiency gain. While their experiments show that R&Q improves load balancing compared to naive techniques, they don't demonstrate how this improvement directly translates to reduced inference latency. This is critical because the use of quantization could often slow down inference.

W4. Weak empirical analysis on more challenging tasks, such as generation tasks (e.g., perplexity, code generation, MT-Bench, etc.).

### Questions
Q1. Error on Page 6, line 288: Are the X-axis and Y-axis labels inverted?

Q2. Should R&Q identify heavy-hitter and important experts for each individual task, or can the identified experts be reused across tasks? The motivation behind this question is that heavy-hitters may vary depending on task characteristics. For example, experts 1-3 might be heavy-hitters for task A, while different experts could be heavy-hitters for task B.

Q3. While resolving load imbalance could theoretically improve computational efficiency, how does R&Q empirically achieve this efficiency gain? Could it actually slow down inference latency due to the quantized experts? I’m asking this because the experiment section lacks an empirical analysis of memory and latency improvements. A strong answer to this question would require empirical results.

Q4. Would R&Q maintain performance on more challenging tasks, such as generation tasks (e.g., perplexity, code generation, MT-Bench, etc.)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a novel strategy called "Replicate and Quantize" for addressing load balancing issues in Sparse Mixture-of-Experts (SMoE) models. The authors systematically analyze the performance and functionality of each expert and introduce a metric to evaluate load balance. They propose a dynamic plug-and-play strategy that is both trainingless and near-lossless, effectively resolving load balancing problems by replicating heavily used experts with lower-bit quantized versions and quantizing the least important experts to fit within the memory budget. Empirical results demonstrate that this approach significantly reduces load imbalance with minimal impact on model performance.

### Strengths
1) The "Replicate and Quantize" strategy is a novel approach that dynamically addresses load imbalance in SMoE models without requiring extensive retraining.
﻿
2)  The proposed strategy is plug-and-play, making it easy to integrate with existing models and practical for real-world applications.

### Weaknesses
1) The paper lacks detailed implementation specifics, such as the exact quantization methods and hyperparameters used.

2) There is a need for more extensive ablation studies to isolate and demonstrate the contributions of the replication and quantization components individually.

### Questions
1) Can you provide more detailed implementation details, including the specific quantization techniques and hyperparameters used in your experiments, to facilitate reproducibility?

2) Could you conduct additional ablation studies to demonstrate the individual contributions of the replication and quantization components in your proposed method?

3) How does your method perform under different levels of model sparsity and varying numbers of experts in the SMoE models?

### Soundness
2

### Presentation
2

### Contribution
2
