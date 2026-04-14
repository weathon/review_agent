# Unraveling and Mitigating Safety Alignment Degradation of Vision-Language Models

- Decision: Reject
- Scores: 3, 5, 5, 5

## Abstract
The safety alignment ability of Vision-Language Models (VLMs) is prone to be degraded by the integration of the vision module compared to its LLM backbone. We investigate this phenomenon, dubbed as “safety alignment degradation” in this paper, and show that the challenge arises from the representation gap that emerges when introducing vision modality to VLMs. In particular, we show that the representations of multi-modal inputs shift away from that of text-only inputs which represent the distribution that the LLM backbone is optimized for. At the same time, the safety alignment capabilities, initially developed within the textual embedding space, do not successfully transfer to this new multi-modal representation space. To reduce safety alignment degradation, we introduce Cross-Modality Representation Manipulation (CMRM), an inference time representation intervention method for recovering the safety alignment ability that is inherent in the LLM backbone of VLMs, while simultaneously preserving the functional capabilities of VLMs. The empirical results show that our framework significantly recovers the alignment ability that is inherited from the LLM backbone with minimal impact on the fluency and linguistic capabilities of pre-trained VLMs even without additional training. Specifically, the unsafe rate of LLaVA-7B on multi-modal input can be reduced from 61.53% to as low as 3.15% with only inference-time intervention.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper empirically explains that "safety alignment degradation" is caused by the representation gap introduced after incorporating the visual modality. It provides detailed empirical evidence and proposes an inference-time alignment method called CMRM, which enhances the safety capability of VLMs in handling harmful inputs to a certain extent.

### Strengths
- The paper verifies from a novel experimental perspective that the "safety alignment degradation" of VLMs is caused by the representation gap introduced after incorporating the visual modality.
- CMRM enhances VLM backbones' safety performance.

### Weaknesses
- Eq. 3 seems to be in conflict with Eq. 6. In Eq. 4 and 5, $v^l_{data}$ and $v^l_{sample}$ seems equal to $\Delta$ defined in Eq. 2. Thus, Eq. 6 should be $h^l_{aligned}=h^l_o+v^l$?
- There are already many publicly available VLM Safety Benchmarks, so why is it necessary to additionally construct a VLM Benchmark from the pure-text JailbreakLLM for experiments? What are the advantages of such a constructed benchmark over existing VLM safety benchmarks? It seems that manipulated JailbreakLLM datasets may have difficulty ensuring a high correlation between the vision input and text input. Furthermore, as shown in Table 1, the Unsafe Rate of the VLM backbone is relatively low when both image and text inputs are provided together in the manipulated JailbreakLLM datasets. Replacing the original images with blank or noisy images even increases the Unsafe Rate. Does such a dataset hold reference value in a vision-language setting?
- All the experiments were conducted on datasets where the harmful text and images have high similarity. Will CMRM still be effective on datasets where the text instructions are safe, but the visual input contains unsafe typography or text-to-image contents (such as FigStep and MM-SafetyBench)? Does CMRM tend to refuse to answer, or does it provide generic responses unrelated to the image on these datasets?
- Will CMRM still be effective when dealing with perturbation-based visual attacks (such as adding noise to images)? The authors should include additional experiments to verify the robustness of CMRM in such scenarios.
- As an inference time alignment method, the authors should include some inference-time defense baselines mentioned in related works for comparison in the experiments on both safety and utility performance.
- CMRM requires a hyperparameter $\alpha$, but as stated in Fig. 2, the setting of dataset-level $\alpha$ depends on the dataset and the specific VLM backbone, making it a potentially difficult parameter to adjust. When applied to scenarios such as unsafe typography, text-to-image, or perturbation-based visual attack methods, will the setting of $\alpha$ introduce additional challenges or have other impacts?
- The impact of different $\alpha$ settings on utility ability needs to be further explored. For instance, at the sample level, when $\alpha=1$, responses such as "I'm sorry, I'm not sure what you mean." already appear. However, dataset-level CMRM can give a helpful response when $\alpha=1$. The authors need to explain the sensitivity of CMRM to $\alpha$ at both the dataset and sample levels, addressing why such responses occur and how $\alpha$ affects the alignment across different levels.
- The paper lacks experiments on the hyperparameters for sample-level CMRM. Since sample-level alignment provides more fine-grained adjustments, the $\alpha$ setting for individual samples should be more sensitive than dataset-level settings. 
- CMRM results in a certain decrease and impacts utility performance. Similar findings are also reflected in the case study.

### Questions
Please see the weaknesses!

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper investigates the problem of safety alignment degradation in multi-modal large language models (MLLMs). The authors demonstrate that the distribution of vision-language representations generated by MLLMs shifts away from the original representation of large language models, which leads to safety alignment degradation. To address this issue, the paper introduces a method called Cross-Modality Representation Manipulation (CMRM), which performs representation manipulation during inference to mitigate this phenomenon. Experimental results show that the proposed CMRM method enables MLLMs to recover their safety alignment capabilities without any additional training.

### Strengths
1. The issue of safety alignment degradation presents a novel problem that has not been previously explored.

2. paper introduces a simple yet effective approach to representation manipulation, aimed at mitigating the degradation phenomenon.

3. Comprehensive quantitative and qualitative analyses are provided to substantiate the phenomenon of safety alignment degradation in MLLMs.

### Weaknesses
1. The concept of feature interpolation is not novel, as similar ideas have already been proposed in other areas, such as classifier-free guidance [1] and contrastive decoding [2].

2. The accuracy of using LLaMA-3-8B-Instruct for safety judgment has not been demonstrated, leading to potential unfaithfulness in the evaluation results. The authors need to demonstrate the correlation between human evaluation and model-based evaluation to strengthen the validity of the results.

[1] Ho, Jonathan, and Tim Salimans. "Classifier-free diffusion guidance." arXiv preprint arXiv:2207.12598 (2022).

[2] Li, Xiang Lisa, et al. "Contrastive decoding: Open-ended text generation as optimization." arXiv preprint arXiv:2210.15097 (2022).

### Questions
1. The sensitivity of alpha is analyzed in Section 4.3 using several values with an interval of 0.1. It would be more effective to illustrate the sensitivity by presenting a broader range of alpha values, such as alpha ∈ {1, 10, 100}.

2. To illustrate the issue of safety degradation, it would be beneficial to present the performance of a pure LLM (e.g., Vicuna) on VLSafe and JailbreakLLMs.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper addresses the generation safety of MLLMs. The authors assume that the ideal distribution of MLLMs should adhere to the safely trained backbone of their LLMs. Based on this assumption, they propose a cross-modal representation calibration method to realign the VL distribution with the original safe Language-only distribution.
While I find the chain of motivation behind this work to be reasonable, I have concerns regarding the foundational assumption and the overall motivation. The assumption appears to be somewhat biased towards prioritizing safety control over the broader development of MLLMs. Additionally, the trade-off between safety and general capabilities is only validated on three MLLMs and 4 benchmarks, which is rather limited.

### Strengths
+ Despite concerns over the foundational assumption, the chain of assumption, validation, and method, along with the associated visualizations, is clear and easy to follow.
+ The author proposed inference-time safety intervention is efficient and easy to use.

### Weaknesses
- The MLLM series authors focus on align VL representations to language models through the use of adapters. With the introduction of multi-modal data, additional parameters, and different learning techniques, the distribution naturally shifts and should shift towards VL alignment. One might not expect the safety regulations of LLMs to still apply effectively to this shifted input, especially with the potential inclusion of novel information. Thus I find the foundational assumption that the ideal distribution of MLLMs should strictly adhere to the safely trained LLM backbone may be biased towards prioritizing safety over the comprehensive development of MLLMs. 
Additionally, upon reviewing the references cited by the authors, I did not find support for the assumption.

- The validation of the trade-off between safety and general capabilities is limited to only three Vision-Language Models (VLMs) and four benchmarks, which may not be sufficient to generalize the findings.

- It is not clear how the baseline MLLMs are tested in the preliminary study and experiments. Different VL tuning strategies may also affect the findings. For example, it is unclear whether the vision tower is fixed or tuned with VL alignment.

### Questions
VLGuard argues that the re-emerged safety issues of MLLMs stem from harmful data encountered during VL instruction fine-tuning, which leads to the forgetting of human safety alignment. I wonder how the proposed mitigation responds to new harmful information, and whether forgotten safety alignment knowledge can be recovered through training-free calibration.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The authors propose Cross-Modality Representation Manipulation (CMRM), an inference-time representation intervention method aimed at restoring the inherent safety alignment capabilities of the LLM backbone within VLMs, while preserving their functional abilities. Empirical results demonstrate that this approach recovers the alignment abilities of the LLM backbone with minimal impact on the fluency and linguistic capabilities of pre-trained VLMs, without additional training.

### Strengths
1. The issue of safety in multimodal models is highly significant, and this paper presents a relatively simple and effective approach to address it.
2. The paper is well-structured and clearly written.

### Weaknesses
1. After carefully reviewing the paper, the selection process for meaningless or corrupted images remains unclear. Are these blank images or noise images? The choice of such images is crucial.
2. Line 200: When constructing the calibration term, the paper uses VLSafe or manipulated JailbreakLLMs as the anchor dataset. Can the resulting calibration term effectively generalize to out-of-distribution (OOD) images or handle more diverse image types? For example, if VLSafe is used as the anchor dataset, how does this approach perform on the subtask using stable-diffusion-generated images in MM-SafetyBench[1], and across a broader range of other safety tasks within MM-SafetyBench[1] and FigStep[2]? The authors should further address these questions regarding generalizability.
3. Utility testing currently employs the ScienceQA dataset, which is domain-specific, while general visual understanding is evaluated on the LLAVA-COCO dataset, which is quite small (90+ images). Can the proposed method maintain utility on more comprehensive benchmarks,  i.e., MM-Vet, MMMU, MME? 
4. Additionally, LLaMA-3-8B may lack precision for this evaluation—why not use more reliable models such as LLaMA-Guard or GPT-4? Has there been any human verification of accuracy?
5. Minor: The related work sections on Safety Alignment for VLMs and Representation Engineering overlook some relevant training-based and inference-based methods for safety improvement (see references [3-6]).

[1]  MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models

[2]  FigStep: Jailbreaking Large Vision-language Models via Typographic Visual Prompts

[3]  Tuning Language Models by Proxy

[4]  Jailbreaking GPT-4V via Self-Adversarial Attacks with System Prompts

[5]  CoCA: Regaining Safety-awareness of Multimodal Large Language Models with Constitutional Calibration

[6]  SPA-VL: A Comprehensive Safety Preference Alignment Dataset for Vision Language Model

### Questions
Please refer to weakness. If the authors successfully address my concerns, I would consider increasing the score.

### Soundness
3

### Presentation
3

### Contribution
2
