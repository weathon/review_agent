# UniFLoW: Universal Multi-Modal Federated LoRA Fine-Tuning Framework with Analytical Aggregation

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
As Multimodal Large Language Models (MLLMs) continue to be trained, the availability of public data diminishes, limiting the possibility for further training and adaptation. However, private data remains an underutilized yet valuable resource. Federated Learning (FL) enables decentralized training on private data, yet extending it to MLLMs is challenging: heterogeneous client modalities induce architectural incompatibility, and full-parameter fine-tuning of billion-scale models incurs prohibitive communication costs. Parameter-efficient methods like LoRA alleviate these issues but introduce aggregation inconsistency, as averaged low-rank updates fail to faithfully recover the true global update.
 To address these issues, we propose (UniFLoW Universal multi-modal Federated LoRA fine-tuning framework With Analytical Aggregation), a unified federated framework that leverages pre-trained large models, a multi-modal architecture, and our proposed Federated Aggregating Analytical Low-Rank Adaption ($FedA^2$-$LoRA$).  UniFLoW effectively utilizes fragmented client-side multi-modal data while ensuring consistent aggregation. And modality-specific encoders and a two stage training strategy ensure effective integration of diverse modalities without overfitting.
 Experiments on text, image, and speech demonstrate that \textbf{UniFLoW} enables scalable, communication-efficient, and aggregation-consistent federated fine-tuning, 
 with $FedA^2$-$LoRA$ achieving state-of-the-art performance compared to existing FedLoRA approaches. We envision \textbf{UniFLoW} as a promising solution to the growing scarcity of public data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes UniFLoW (Universal Multi-modal Federated LoRA Fine-tuning Framework with Analytical Aggregation), a unified federated framework that leverages pre-trained large models and a multi-modal architecture. Moreover, it introduces Federated Aggregating Analytical Low-Rank Adaptation (FedA2-LoRA), which directly averages \\( A^t \\) to obtain \\( A^{t+1} \\), and then recovers the corresponding \\( B^{t+1} \\) matrices from the aggregated update \\( \Delta_W^* \\) using a closed-form solution of regularized least squares regression (Ridge Regression).

### Strengths
- The introduced FedA2-LoRA is both novel and interesting, effectively addressing aggregation errors in FL with LoRA fine-tuning.
- The paper is well-written and clearly articulated, making it easy to understand.

### Weaknesses
- This work exaggerates its contributions. The advantage of UniFLoW in addressing architectural incompatibility when dealing with multimodal data (Problem 2) stems from the characteristics of the modality-specific encoder (ImageBind [1]), which can handle various modalities, rather than from the contributions of this work.
- The proposed UniFLoW is based on specific encoders (ImageBind [1]) and LLMs (Vicuna-7B [2]). Can different encoders and LLMs be used?
- Why are the experimental results presented in Table 5 much worse than the results presented in Table 1 in FedSA-LoRA [3]?



[1] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 15180–15190, 2023.

[2] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

[3] Pengxin Guo, Shuang Zeng, Yanran Wang, Huijie Fan, Feifei Wang, and Liangqiong Qu. Selective aggregation for low-rank adaptation in federated learning. arXiv preprint arXiv:2410.01463, 2024.

### Questions
- In the first stage, when the number of local iteration steps is less than τ, the model updates only the parameters of the corresponding encoder. When the steps exceed τ, the model updates only the parameters of the LLMs (Lines 242-245). What would be the effect of training the LLMs first and then training the encoder?
- Why is it better to train the LLM and the encoder in an II-stage approach rather than training both simultaneously? Is this related to the statement: "However, in FL, when client data exhibits certain biases, only specific types of multimodal data may be available. If
the encoder is not fine-tuned, this data can influence the fine-tuning of the base model, causing it to specialize for a specific modality and thus negatively impacting the model’s generalization." (Lines 238-241) Thus, what would be the effect of training the encoder for the first T communication rounds and then training the LLM for the following T rounds?
- What is the time complexity of solving Equation 11? Since it involves matrix inversion, how does the time complexity look?
- Line 373, "Please refer to the Appendix for a detailed evaluation." I did not find it in the Appendix.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
To efficiently leverage distributed multimodal data under heterogeneous multimodal settings, we propose FedA²-LoRA within the FL and MLLMs framework. The method adopts a two-stage training strategy—first fine-tuning the modality-specific encoder’s LoRA, followed by the LLM’s LoRA and introduces Tikhonov regularization on the LoRA A matrix to approximate the B matrix, thereby improving aggregation consistency. Experimental results demonstrate the effectiveness of the proposed method.

### Strengths
The method takes into account the issues of modality heterogeneity and the aggregation bias between the LoRA A and B matrices, making its research motivation reasonably meaningful.

### Weaknesses
1. The approach of approximating (B) from (U) and (A) (Equations 9–11) is not very reasonable. If each client needs to upload both (B) and (A) to the server to compute (U), then it would be more straightforward to directly multiply (B) and (A) on the server and aggregate the results, which would inherently avoid the aggregation inconsistency. Moreover, uploading both (B) and (A) does not actually reduce the communication cost.

2. The use of Tikhonov Regularization to approximate matrix (B) lacks theoretical justification, making the approach less convincing.

3. The results in Tables 2–4 seem to show only that the two-stage training strategy performs better than the single-stage approach that trains the modality encoder and LLM LoRA simultaneously. While this two-stage strategy could be an effective training method, it may not be sufficient to constitute a complete innovation.

4. There are no additional ablation studies to verify the effectiveness of using Tikhonov Regularization for approximating matrix (B).

5. Figures 1 and 2 are not clearly presented—particularly Figure 2, which is overly complicated and fails to highlight the key points. In addition, the overall writing quality of the paper still needs improvement.

### Questions
The main issues are that the approach for approximating (B) appears unreasonable and does not actually reduce communication costs. The experimental results mainly highlight the effect of the two-stage training strategy; however, this strategy alone is insufficient to constitute a complete innovation, and the method for approximating (B) also lacks theoretical justification.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes UniFLoW, a universal multi-modal federated LoRA fine-tuning framework targeting the challenges of applying Federated Learning (FL) to Multi-modal Large Language Models (MLLMs), namely client-side modality heterogeneity, high communication costs, and LoRA aggregation inconsistency. The central contribution is FedA²-LoRA, which aggregates client-side LoRA parameters by analytically reconstructing the  matrix via a closed-form ridge-regression solution. The framework adopts a two-stage training strategy that fine-tunes LoRA modules in both the modality encoders (ImageBind) and the base LLM (Vicuna-7B).Experiments on multi-modal QA tasks indicate its effectiveness.

### Strengths
1.FedA²-LoRA introduces an efficient analytical approach to address federated LoRA aggregation inconsistency. By directly aggregating the matrices and analytically recovering , it offers a communication-efficient alternative.
2.The work is significant in applying FL to MLLMs with heterogeneous modality data.
3.UniFLoW combines a general-purpose modality encoder (ImageBind) with an LLM and employs LoRA in key modules to cope with modality heterogeneity; the design rationale is clear and sensible.

### Weaknesses
1.FedA²-LoRA assumes “ is more global and is more local,” hence averaging and reconstructing from and . This rests on heuristic motivation; the paper should specify theoretical conditions under which the “global” nature of  holds, and whether it remains valid under non-IID settings.
2.The closed-form solution in Equation (11) essentially solves the ridge-regression problem , yet the paper does not explicitly present the objective nor provide a proof of optimality.
3.Although FedA²-LoRA is said not to increase communication costs, the experiments do not report measured communication budgets or parameter payloads.
4.The paper references FedEx-LoRA and related work but does not provide head-to-head comparisons under matched communication budgets and client participation. Current conclusions rely mainly on comparisons with FedSA and FFA and are therefore less convincing.
5.While the paper emphasises heterogeneous client resources, it does not propose explicit heterogeneity-aware mechanisms (e.g., variable-rank or variable-layer LoRA) that reflect realistic constraints.

### Questions
1.Mixed use of “Ⅱ/II stage”; please standardise.
2.Instances include “does not exists” in Equation (11) and “AAk” around line 300. There are occasional logical jumps and paragraph repetitions that reduce readability.
3.Please verify consistency of symbols and variable definitions throughout.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the challenge of fine-tuning Multimodal Large Language Models in federated learning settings. The authors explain that applying traditional FL to MLLMs can be very expensive and methods like LoRA in FL can suffer from "aggregation inconsistency". Therefore, they propose UniFLoW, a unified framework with three core contributions:

1- It uses a pre-trained universal encoder (ImageBind) and a base LLM (Vicuna-7B), applying LoRA to both components.

2- Clients first fine-tune their respective encoder's LoRA parameters and then fine-tune the base LLM's LoRA parameters within a single local training round.

3- Their server-side aggregation algorithm (FedA²-LoRA) computes the global A matrix by simple averaging. Then, they find the corresponding global B matrix based on A.

The authors evaluate UniFLoW on multi-modal QA (image and audio) and the FedA²-LoRA component on unimodal NLU (GLUE).

### Strengths
* The Federated MLLM is an interesting problem.

* The ablation study confirms some of the choices. For example, it shows that the two stage training  is effective, yielding better results than end-to-end local fine-tuning.

* The authors show the effectiveness of their method through different experiments.

### Weaknesses
* The paper compares its performance against methods like FFA-LORA (which freezes $A$ and only uploads $B$) and FedSA-LORA (which only uploads $A$). These methods have half the client-to-server communication cost.

* The authors do not provide any justification for some claims for example “The A matrices capture more general information”

### Questions
1- Are BERTScore and Token Accuracy reliable metrics for evaluating open-domain, generative QA?

2- Line 096 is not clear. How should I read this part? 

3- Figure 2 is very unclear and does not help to understand the method. The figure description is very short and does not help much.

4- It is not clear for me that if this paper is the first paper that works on first federated MLLMs fine-tuning framework (line 119) or based on the beginning of line 192 there are other FedMLLMs approaches.

### Soundness
2

### Presentation
2

### Contribution
2
