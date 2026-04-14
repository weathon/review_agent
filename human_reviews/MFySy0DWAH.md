# A Framework of Distilling Multimodal Large Language Models

- Decision: Reject
- Scores: 6, 6, 3, 6

## Abstract
The success of Large Language Models (LLM) has led researchers to explore Multimodal Large Language Models (MLLM) for unified visual and linguistic understanding. However, the increasing model size and computational complexity of MLLM limit their use in resource-constrained environments. Small-scale MLLM ($s$-MLLM) aims to retain the capabilities of the large-scale model ($l$-MLLM) while reducing computational demands, but resulting in a significant decline in performance. To address the aforementioned issues, we propose a novel LLaVA-KD framework to transfer knowledge from $l$-MLLM to $s$-MLLM. Specifically, we introduce Multimodal Distillation (MDist) to minimize the divergence between the visual-textual output distributions of $l$-MLLM and $s$-MLLM, and Relation Distillation (RDist) to transfer $l$-MLLM’s ability to model correlations between visual features. Additionally, we propose a three-stage training scheme to fully exploit the potential of $s$-MLLM: 1) Distilled Pre-Training to align visual-textual representations, 2) Supervised Fine-Tuning to equip the model with multimodal understanding, and 3) Distilled Fine-Tuning to further transfer $l$-MLLM capabilities. Our approach significantly improves performance without altering the small model's architecture. Extensive experiments and ablation studies validate the effectiveness of each proposed component. Code will be available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a three-stage pipeline to distill knowledge from a large MLLM to a smaller one. The proposed distillation strategy, particularly RDist, is both innovative and effective in transferring the capabilities of the large MLLM. Experiments across 10 benchmarks further validate the effectiveness of this method.

### Strengths
1. This paper extends the common PT-SFT two-stage training scheme to a three-stage training strategy, enhancing the multimodal alignment process and transferring the capacities from the large MLLM to the smaller one.
2. The proposed Multimodal Distillation (MDist) and Relational Distillation (RDist) are novel. The RDist seems sound to learn the visual representation of the l-MLLM.
3. The performance of LLaVA-KD is quite impressive, significantly outperforming the baseline method.

### Weaknesses
1. Some notations are confusing, e.g., 
    1. In Eq. (1), $\phi_l(y_m | y_{<m})$ seems to be a probability, but the author claims that it is a distribution. 
    2. In Eq. (2), $\phi_l(Y_j | y_{<m})$ is a probability, but \phi_l(y_m | y_{<m}) is a distribution. 
    3. What is $\mathcal{L}_{PT}$ in Eq. (6)?
2. From Table.2, it seems that PT + DFT is comparable to DPT + DFT.  Can you provide the result of PT+SFT+DFT to further validate the effectiveness of DPT? 
3. (Minor) In Section 3.3, “we further design a three-stage distillation scheme (Sec. 3.3.1)”, (Sec. 3.3.1) should be (Sec. 3.3.2).

### Questions
See weaknesses.

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
3

### Summary
Distillation techniques are commonly employed to effectively train small models using given large models. While numerous studies have explored training small-scale LLMs through distillation, there has been limited work on distillation methods for MLLMs. To address this gap, this paper presents various training strategies for MLLM distillation.

1. This paper proposes a three-stage distillation scheme: i) Distilled Pre-Training (DPT), ii) Supervised Fine-Tuning (SFT), and iii) Distilled Fine-Tuning (DFT).
2. This paper proposes two distillation loss terms: i) Multimodal Distillation (MDist) and ii) Relational Distillation (RDist), which are tailored for multimodality.

### Strengths
1. The paper is well-motivated: This paper is timely given that while the field of MLLMs is rapidly evolving, research on distillation methods for MLLMs remains largely unexplored.
2. The paper is well-written and easy to follow.
3. The proposed LLaVA-KD models demonstrate superior performance compared to existing small-scale MLLMs.

### Weaknesses
1. It appears that the methods proposed in this paper (three-stage distillation scheme, Multimodal Distillation) do not present particularly novel or distinctive contributions to the field. Therefore, systematic experimentation would be essential to demonstrate clear superiority of the proposed methods.
2. The discussion and comparison with existing distillation methods for MLLMs appears insufficient.
    * While the paper cites LLaVA-MoD [1] as related work, it fails to conduct comparative experiments against the Preference Distillation technique proposed in LLaVA-MoD. The performance comparison between LLaVA-MoD and LLaVA-KD models alone cannot provide meaningful insights due to the lack of controlled variables.
    * Additionally, the paper lacks discussion and comparative experiments with LLAVADI [2], another concurrent work on MLLM distillation.
3. The experimental settings appear inadequate for demonstrating the effectiveness of the proposed method.
    * While this paper focuses on performance comparisons with existing small-scale MLLMs, it appears difficult to claim the superiority of the proposed distillation techniques solely through these comparisons. A direct performance comparison with existing LLM/MLLM distillation methods under the same experimental settings would be more convincing.
    * Please refer to the Questions section for additional details.

[1] LLaVA-MoD: Making LLaVA Tiny via MoE Knowledge Distillation    
[2] LLAVADI: What Matters For Multimodal Large Language Models Distillation

### Questions
1. Why does this paper emphasize that LLaVA-KD's performance is better than BLIP series? Since this is a paper presenting a distillation method, directly comparing performance between different types of models seems inappropriate. Rather, it would be meaningful to report the performance of small models after applying the proposed distillation method to the BLIP series.
2. Can we directly compare the performance of the proposed method with the distillation methods presented in LLaVA-MoD and LLAVDI under the same experimental settings?
3. How does the performance compare when directly applying the LLM distillation methods mentioned in Section 2.2's 'KD for LLM' paragraph to MLLM distillation? A comparative analysis would be necessary to determine whether MLLM-tailored distillation methods are truly needed or if existing LLM distillation methods are sufficient.
4. What is the performance of PT+SFT+DFT? The comparable performance between PT+DFT and DPT+DFT raises questions about the necessity of DPT over PT.
5. A study on hyperparameters (loss weights) seems necessary. The importance of hyperparameter tuning for effective distillation should be investigated.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper investigates knowledge distillation for multi-modal language models (MLLMs). The proposed method introduces learning objectives targeting both visual features and language responses, designed to enhance alignment between large and small MLLMs. Additionally, the authors introduce a novel relational correlation loss that operates on the similarity matrix of visual tokens.

### Strengths
- The paper addresses an important and timely aspect of MLLMs, specifically focusing on the optimization of small MLLMs.
- The proposed method is clearly presented and well-explained.
- Overall, the paper is well-structured, making it clear and easy to follow.

### Weaknesses
In the final paragraph of the Introduction, the authors list three major contributions. However, I am not fully convinced by these claims:

- The proposed relational distillation approach lacks sufficient motivation. The rationale behind optimizing the similarity matrix of visual tokens between large and small MLLMs is unclear.
- The authors claim that their 1B model outperforms the BLIP-2 (13B) and InstructBLIP (7B) models across various benchmarks. However, this claim is misleading. BLIP-2 models are primarily trained on image-text pairs, and InstructBLIP has been trained on lower-quality instruction data compared to the extensive, high-quality LLaVA-665K dataset used in this submission. In reality, the performance difference between LLaVA and InstructBLIP is largely due to differences in data quality and quantity (besides using different ViTs). A fair comparison would require training the BLIP series models on the same dataset. Therefore, the authors' performance claims are misleading.

Regarding the proposed method, while it is straightforward, it also feels somewhat incremental, as it primarily adds a KD learning objective to various features within MLLMs.

The experimental studies and results are not convincing either:

- The performance of LLaVA-KD does not surpass that of LLaVA-MOD. In Table 1, the authors do not report the average (AVG) performance of LLaVA-MOD and omit some of its results on certain benchmarks. However, based on the reported benchmarks, the AVG performance of LLaVA-KD and LLaVA-MOD is very close.
- Additionally, the large MLLM used in this study has only 4B parameters, which may be insufficient to draw conclusions about scaling. The authors do not present its detailed performance, nor do they investigate the effect of varying large model sizes (e.g., 7B, 13B), leaving questions about scalability unaddressed.
- It is also unclear whether distilling a 1B model to another 1B model would be effective within this framework.
- The authors do not report results for small MLLMs (0.5B and 1B) trained with the standard LLaVA procedure, making it difficult to assess the improvement contributed by the proposed method.

### Questions
N/A

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
4

### Summary
The paper presents a knowledge distillation technique for multi-modal Large Language Models (MLLM) where the knowledge is transferred from a large MLLM (l-MLLM) to small MLLM (s-MLLM). Specifically, a three stage training is proposed (as opposed to two in conventional MLLM training) consisting of a) Distilled Pre-Training (DPT) to align visual-textual representations, b) Supervised Fine-Tuning (SFT) to equip the model with multimodal understanding, and c) Distilled Fine-Tuning (DFT) to further transfer l-MLLM capabilities. A Multimodal Distillation (MDist) and Relational Distillation (RDist) are used in both DPT and DFT stages to enhance the ability of s-MLLMs to process complex visual information. The proposed method is evaluated on a set of Benchmarks and VQA datasets to show improved performance w.r.to baseline and SOTA

### Strengths
1. The paper introduces a simple approach for knowledge distillation in MLLMs. While the components used such as KL divergence between teacher and student predictions [L_res in Eq. 2 and L_vis in Eq. 3, as in (Hinton, 2015), LLAVA-MoD are well-known], this method achieves superior results compared to current SOTA techniques employing knowledge distillation for MLLMs. Despite involving numerous hyperparameters (\alpha, \beta, \gamma, \alpha', \beta', \gamma'), the authors fixed these values across all experiments, demonstrating the method’s ability to generalize to new datasets without hyperparameter tuning.

2. The paper is well-written and easy to follow. The ablation study effectively demonstrates the significance of each component (MDist and RDist).

### Weaknesses
1. Techniques like Imp-2B (Shao et al., 2024) can achieve comparable results (<0.9) without relying on knowledge distillation. If models like Imp-2B can perform similarly without a pre-trained l-MLLM, what justifies the need for computationally intensive methods like the one proposed? The proposed approach would be more advantageous if it could achieve similar outcomes with significantly less data than Imp-2B, which currently requires only 300k more samples than the proposed method. 

2. The authors claim that distilling visual tokens (Eqn. 3) is crucial for achieving strong results. However, as seen in Tables 3, 4, A3, and A4, the improvement from distilling visual tokens appears to be marginal (0.5%). I would appreciate further discussion or additional results to substantiate this claim, especially since it is one of the key contributions emphasized by the authors. 

3. The proposed technique introduces an additional step (stage 3) compared to traditional MLLM training methods like LLAVA, potentially increasing the computational budget. While Table 5 provides a comparison, it is only against other KD techniques. I would prefer to see a comparison with all models of similar size (e.g., Imp-2B and others). Additionally, it is unclear why certain datasets, such as VQAv2, POPE, and MMMU, were excluded in Table 5.

After rebuttal: I appreciate the authors’ efforts in conducting additional experiments and find arguments 1 and 3 to be valid. Therefore, I am increasing my score to 6.

### Questions
I would like authors to address all 3 concerns outlined in the "Weaknesses" section

### Soundness
2

### Presentation
3

### Contribution
2
