# InfiMed: Low-Resource Medical MLLMs with Advancing Understanding and Reasoning

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Multimodal Large Language Models (MLLMs) have achieved remarkable progress in domains such as visual understanding and mathematical reasoning. However, their application in the medical domain is constrained by two key challenges: (1) multimodal medical datasets are scarce and often contain sparse information, limiting reasoning depth; and (2) Reinforcement Learning with Verifiable Rewards (RLVR), though effective in general domains,  cannot reliably improve model performance in the medical domain. To overcome these challenges, during the supervised fine-tuning (SFT) stage, we incorporate high-quality textual reasoning data and general multimodal data alongside multimodal medical data to efficiently enhance foundational medical capabilities and restore the base model’s reasoning ability. Moreover, considering that there are some multimodal medical datasets with sparse information, we further synthesize reflective-pattern-injected chain-of-thought (CoT) in addition to general CoT samples, equipping the model with initial reflective reasoning capabilities that provide a structured foundation for subsequent RLVR training. Finally, we introduce our InfiMed-series models, InfiMed-SFT-3B and InfiMed-RL-3B, both of which deliver state-of-the-art performance across seven multimodal medical benchmarks. Notably, InfiMed-RL-3B achieves an average accuracy of 59.2\%, outperforming even larger models like InternVL3-8B, which achieves 57.3\%
Specifically, during the SFT phase, we utilized 188K samples, while the RLVR phase incorporated 36K samples, demonstrating the efficacy of both training strategies in achieving superior performance. We also conducted a series of extensive experiments, which provide valuable insights that contribute to advancing the performance of MLLMs in medical scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the limited reasoning ability of multimodal medical large language models (MLLMs) in information-sparse scenarios by proposing a systematic solution. Through reflective-pattern-injected SFT and group-relative advantage-optimized RLVR, the model learns to self-evaluate, correct errors, and explore multi-step reasoning paths. Leveraging both general and medical multimodal data under low-resource conditions, the InfiMed series achieves strong performance across seven medical benchmarks, particularly excelling in reasoning-intensive tasks, with ablation studies and case analyses validating the effectiveness of the approach.

### Strengths
(1) The method is well designed. The paper integrates reflective Chain-of-Thought (CoT) injection with the RLVR framework to enhance the model’s self-evaluation and multi-step reasoning capabilities.
(2) The experiments cover multiple medical benchmark datasets, and the proposed method is validated through ablation studies and case analyses, confirming its effectiveness.

### Weaknesses
1. This study aims to enhance the model’s reasoning and understanding abilities. However, ablation results show that general multimodal data improve performance more than the reflective CoT, suggesting a potential inconsistency with the study’s focus.
2. The methodology and figures are unclear. It is recommended that the authors describe the inference trajectory of the general CoT to help readers understand the model’s reasoning mechanism. Additionally, they should further clarify why multimodal medical data and textual medical data are displayed in parallel in Figure 2, considering that text is already part of the multimodal data.
3. The related work section is incomplete. It lacks a discussion of the application of CoT in MLLMs, which limits the reader’s understanding of the technical positioning and novelty of this study. It is recommended that the authors include this discussion in the related work section to provide a complete technical background and highlight the novelty of the study.
﻿4. The experimental section omits ablation studies under conditions without general CoT data, making it difficult to evaluate the independent contribution of the method. It is recommended that the authors supplement ablation experiments under conditions lacking general CoT data to verify its independent contribution.﻿
5. In addition, there are minor issues, such as incorrect variable notations, which further undermine the overall clarity and rigor of the manuscript. It is recommended that the authors carefully check the details in the paper, such as the notation error in line 233 (“and c denotes the reward”), to ensure consistency and accuracy in expressions.

### Questions
Please refer to the Weaknesses.

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
4

### Summary
This paper proposes InfiMed for low-resource training for medical multimodal large language models. It uses a stronger language model to sample and filter reasoning traces on some general domain and medical datasets and use supervised fine-tuning to inject these capability to their model. A stage of reinforcement learning with verifiable rewards follows to further enhance the reasoning ability of the model. The authors also conducted many analysis on the relationship between the design of training data / training pipeline and the model performance, showing that reasoning is not always helpful for some tasks.

### Strengths
1. The paper is well written with illustrative figures and good explanations.
2. The results of the model are strong and can often surpass larger-scale models.
3. The analysis part of this paper is useful for future practitioners. It reveals the relationship between the training data and model performance on different benchmarks. Some conclusions are interesting, e.g. reasoning is not always helpful. Will be better if experiments can be done on larger scale or more recent models.

### Weaknesses
1. The methods in this paper are well known.  The reasoning construction, the training pipeline, i.e. SFT followed by RLVR, and the rewarding functions are used by many concurrent papers, and it is hard to distinguish this paper.
2. The conclusions are drawn from a 3B model.  As revealed by previous work (e.g. DeekSeek-R1 series), the reasoning capability is best incentivized only on large models.  Some conclusions of this paper demonstrate that reasoning might hurt the model performance on a few datasets, but it remains open question if using larger model can fix this.

### Questions
Suggestion on writings: The fonts on the figures are not professional. They are good for poster / slides but not ideal for paper presentation. Should have used serif to fit it into the ICLR formats (Times New Roman).

### Soundness
3

### Presentation
4

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
This paper proposes a new training method for medical multimodal LLMs, called InfiMed. The training strategy consists of two steps: (1) supervised fine-tuning with reflective patterns on both general and medical multimodal inputs, resulting in the model InfiMed-SFT; and (2) reinforcement learning with GRPO to further improve model performance. In addition, the authors conduct extensive experiments on various medical benchmarks against several baseline LLMs, demonstrating superior evaluation performance. The ablation studies further validate the effectiveness of each module

### Strengths
- The paper is well-structured and clearly presented.
- The proposed pipeline for building a robust multimodal medical LLM is effective.
- The extensive evaluation results convincingly demonstrate the method’s effectiveness

### Weaknesses
- The overall contribution is somewhat incremental, building on DeepSeek and prior multimodal LLM work. Nonetheless, the execution is solid.
- The use of general data in the SFT stage is not novel, as this approach has already been widely applied in LLM training (e.g., BianCangLLM [1]).
- The experimental results may not objectively reflect the superiority of this method, since the open-source models and domain-specific LLMs were not fine-tuned on the same datasets as InfiMed.
- Since InfiMed uses Qwen2.5-VL as its backbone, for fairness, baseline medical LLMs should also be built on Qwen2.5-VL to eliminate improvements that stem solely from the backbone. Otherwise, InfiMed should adopt the same backbone as the baselines.
- In the case studies, some examples lack reflective information. Additionally, it is unclear why the reflection ability seems to diminish after reinforcement learning. Clarification on this point would strengthen the paper.

Reference:

1. Wei, Sibo, et al. "Biancang: a traditional chinese medicine large language model." IEEE Journal of Biomedical and Health Informatics (2025).

### Questions
Please see my weakness.

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
3

### Summary
This paper presents the InfiMed-Series models, a family of multimodal large language models designed for medical tasks by augmenting limited medical data with general multimodal and textual reasoning data. It further introduces reflective-pattern-injected chain-of-thought synthesis to enhance the models’ reflective reasoning foundation and enable effective Reinforcement Learning with Verifiable Rewards (RLVR) training.

### Strengths
1. This paper is well-written, with a clear and easy-to-follow logical flow.
2. Extensive experiments on multiple benchmarks demonstrate the competitive performance of the proposed InfiMed-Series models.
3. The ablation studies showcase the effectiveness of each component.

### Weaknesses
1. Both the SFT and RLVR stages adopt mixed data (such as general multimodal data, textual medical data, and reflective-pattern-injected CoT data) for optimization. What are the proportions of these different data types, and how were they determined? Would varying these ratios lead to different effects? Moreover, why not adopt a progressive approach, feeding different types of data sequentially for optimization?
2. When constructing the reflective-pattern-injected CoT data, how is rejection sampling implemented? Since this process relies on Qwen2.5-VL-32B and Qwen2.5-VL-72B, could it potentially introduce inherent bias?
3. During the RLVR training stage, unlike the SFT stage, the data do not need to contain precise CoTs. However, the RLVR dataset contains only 36K samples, far fewer than the 188K used in the SFT stage. Whether increasing the amount of RLVR training data would further improve model performance remains to be explored.
4. Section 4.4 mentions that step-by-step reasoning introduces redundant steps, increasing the risk of hallucination and degrading performance, whereas Section 4.5 finds that the reasoning process in InfiMed-RL-3B helps identify the correct options. This seems to be contradictory.

### Questions
Please refer to the Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
