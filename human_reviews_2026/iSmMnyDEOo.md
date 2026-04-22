# E$^2$AT: Multimodal Jailbreak Defense via Dynamic Joint Optimization

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 6

## Abstract
Research endeavors have been made in learning robust Multimodal Large Language Models (MLLMs) against jailbreak attacks. 
However, existing methods for improving MLLMs' robustness still face critical challenges: ① how to efficiently tune massive weight parameters and ② how to ensure robustness against attacks across both visual and textual modalities. To this end, we propose an $\textbf{E}$fficient $\textbf{E}$nd-to-end $\textbf{A}$dversarial $\textbf{T}$raining (E$^2$AT) framework for both visual and textual adversarial attacks. Specifically, for the visual aspect, E$^2$AT incorporates an efficient projector-based AT module that aligns the attack samples at the feature level. For training objectives, we propose a Dynamic Joint Multimodal Optimization (DJMO) strategy to enhance generalization ability against jailbreak attacks by dynamically adjusting weights between normal and adversarial objectives. Extensive experiments are conducted with five major jailbreak attack methods across three mainstream MLLMs. Results demonstrate that our E$^2$AT achieves the state-of-the-art performance, outperforming existing baselines by an average margin of 34\% across text and image modalities, while maintaining clean task performance. Furthermore, evaluations of real-world embodied intelligent systems highlight the practical applicability of E$^2$AT, paving the way for the development of more secure and reliable multimodal systems. Our code is available on [https://anonymous.4open.science/r/EAT-FC71](https://anonymous.4open.science/r/EAT-FC71).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the E^2 AT framework for improving the robustness of MLLMs against jailbreak attacks targeting both visual and textual modalities. The approach combines a projector-based AT module for efficient visual defense with a novel Dynamic Joint Multimodal Optimization (DJMO) strategy that dynamically adjusts loss weights to balance adversarial and clean objectives. Extensive experiments on three open-source MLLMs and multiple attack methods demonstrate strong trade-off between safety and utility.

### Strengths
+ **Important problem**: The paper addresses an important and timely problem—defending against jailbreak attacks in MLLMs.
+ **Real-world applicability**: The embodied robotics demonstrations in Figures 1(c) and 3 effectively connect the proposed algorithmic advances to real-world safety applications, illustrating concrete examples of model refusals in adversarial scenarios.
+ **Extensive empirical evaluation**: The paper presents thorough empirical results across three MLLMs and two comprehensive benchmarks, along with detailed ablation studies, providing nuanced insights into the model’s performance and robustness.

### Weaknesses
- **Outdated related work**: The discussion of related work and baselines appears somewhat outdated. It would strengthen the paper to include and compare with more recent studies, such as [1][2][3][4].
- **Efficiency claim**: As far as I know, freezing the vision encoder and only updating the projector is a common strategy to improve training efficiency for MLLMs [5]. Therefore, the claimed novelty in this aspect may be somewhat overstated. Moreover, although the paper describes the method as “highly efficient,” the results do not clearly demonstrate a significant efficiency advantage. For example, in Figure 4, the time efficiency of E^2 AT is only better than that of PAT.
- **Attack methods**: Since the model is optimized against white-box adaptive attacks, it would be valuable to include evaluations under black-box adaptive attack settings (e.g., [5][6]) to better demonstrate the generalization ability of E^2 AT against adaptive attacks.

References:

[1] Securing Multimodal Large Language Models: Defending Against Jailbreak Attacks with Adversarial Tuning

[2] VLMGuard-R1: Proactive Safety Alignment for VLMs via Reasoning-Driven Prompt Optimization

[3] Spot Risks Before Speaking! Unraveling Safety Attention Heads in Large Vision-Language Models

[4] SEA: Low-Resource Safety Alignment for Multimodal Large Language Models via Synthetic Embeddings

[5] Visual Instruction Tuning

[6] Jailbreaking Black Box Large Language Models in Twenty Queries

[7] AutoDAN-Turbo: A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs

### Questions
See Weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper focuses on enhancing the safety of multimodal large language models through an adversarial training framework that incorporates both image and text perturbations. In addition, it proposes an adaptive weight adjustment mechanism to better balance safety objectives with utility preservation. Experimental results show that the proposed method effectively defends against test-time attacks.

### Strengths
1. The paper is well-written and easy to follow.


2. The motivation is clear and straightforward.


3. The experiments demonstrate that the proposed method performs well against test-time attacks.

### Weaknesses
1. The novelty is relatively limited. Adversarial learning is a fundamental concept in the adversarial robustness field, and there already exist several works on adversarial training for both LLMs and MLLMs [1][2]. The proposed method appears to be a straightforward application of adversarial training to the MLLM safety setting, with little new technical contribution in the overall design.


2. The discussion and comparison with related works are not comprehensive. For example, [1] also proposes an adversarial training method that considers both image and text perturbations, but the distinction between that work and the current paper is not clearly explained.


3. The experiments should include some latest models, such as the Qwen series.

[1] Yin, Ziyi, et al. "Towards Robust Multimodal Large Language Models Against Jailbreak Attacks." arXiv preprint arXiv:2502.00653 (2025).

[2] Xhonneux, Sophie, et al. "Efficient adversarial training in LLMs with continuous attacks." Proceedings of the 38th International Conference on Neural Information Processing Systems. 2024.

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper focuses on adversarial training of MLLMs, aiming to enhance the robustness of MLLMs.

### Strengths
1. This paper focuses on an important research question.
2. The experiments are conducted on different attack settings.

### Weaknesses
**A critical issue**: In this paper, all citation formats are incorrect. The authors should properly distinguish and check the usage of ~\cite and ~\citep. In the current version, the citation formatting errors make the paper very hard to read (**all citations** are directly attached to the main text without parentheses). Such an obvious mistake—one that anyone would notice after a single read—is clearly unreasonable for a top-tier conference submission.

1. In Figure 1(c), the authors claim that without their defense method, the robot will perform dangerous behaviors. This experiment is very confusing to me: (1) What is the purpose of this experiment? (2) Under what setting was it conducted—what model and attack method were used? (3) Does the experiment compare against other defense methods? (4) How is this experiment specifically related to the subsequent context, such as the motivation? Without any quantitative analysis, simply showing one example does not make clear what the authors intend to convey.

2. Unreasonable threat model definition. After reading lines 166–169, the authors seem to target defenses against black-box attacks—i.e., the defender can adversarially fine-tune (AT) the model, but the attacker must operate in a black-box setting. I do not understand the motivation for this setup—in fact, the traditional goal of AT is to defend against white-box attacks [1]. This threat-model design is therefore unreasonable.

[1] Aleksander Madry. Towards deep learning models resistant to adversarial attacks. arXiv, 2017.

3. Notation issues. In Eq. 3 and Eq. 4, the authors use different symbols to define the “feature space.” What exactly is the feature space? In Eq. 3, why do x_image and x_text belong to the feature space?

4. Experimental setup. The paper lacks comparisons with several classic and mainstream MLLMs, including Qwen2.5-VL and InternVL models. Without these models, it is difficult to support claims about the method’s generalization capability.

### Questions
Please refer to the weakness section.

### Soundness
2

### Presentation
1

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
This paper proposes E²AT, an efficient end-to-end adversarial training framework designed to defend against multimodal jailbreak attacks on Multimodal Large Language Models (MLLMs). The key innovation lies in its Dynamic Joint Multimodal Optimization (DJMO) strategy, which dynamically balances adversarial and clean training objectives across both visual and textual modalities. The authors introduce a projector-based adversarial training module that aligns adversarial image features with clean ones at the feature level, reducing computational overhead. For text, they adopt GCG-based adversarial suffix generation. The DJMO mechanism adaptively weights the adversarial and clean losses during training using exponential moving averages, enabling the model to focus on the most relevant modality at each stage.Extensive experiments on three MLLMs (LLaVA, Bunny, mPLUG-Owl2) and two benchmarks (JailbreakV-28K, MM-SafetyBench) demonstrate that E²AT achieves state-of-the-art robustness, reducing the average attack success rate by 34% compared to baselines, while maintaining clean task performance. The method is also validated in a real-world robotic arm scenario, showing practical applicability.

### Strengths
The paper introduces E^2AT (Efficient End-to-End Adversarial Training), a multimodal jailbreak defense framework that jointly optimizes visual and textual adversarial training with dynamic weighting. It integrates a projector-based adversarial training mechanism and a Dynamic Joint Modality Optimization (DJMO) strategy to balance robustness across modalities. Experiments on multiple MLLMs and two benchmarks show that E^2AT significantly reduces the weighted attack success rate, outperforming prior defenses such as VLGuard and BlueSuffix. The method is presented as computationally efficient and broadly applicable to multimodal safety alignment.

### Weaknesses
1. Ablation results are descriptive but not deeply analyzed, particularly, the feature-level effects of removing projector optimization.
2. Despite emphasizing "Efficient", the paper provides no runtime, memory, or FLOPs comparison with prior works. Efficiency remains qualitative rather than experimentally verified.
3. A formatting error: The caption for a table should be placed above it.

### Questions
These are all in the 'Weaknesses' section above.

### Soundness
3

### Presentation
3

### Contribution
3
