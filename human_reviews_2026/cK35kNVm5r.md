# MedVR: Annotation-Free Medical Visual Reasoning via Agentic Reinforcement Learning

- Avg Score: 4.80
- Decision: Accept (Poster)
- Scores: 2, 8, 4, 6, 4

## Abstract
Medical Vision-Language Models (VLMs) hold immense promise for complex clinical tasks, but their reasoning capabilities are often constrained by text-only paradigms that fail to ground inferences in visual evidence. This limitation not only curtails performance on tasks requiring fine-grained visual analysis but also introduces risks of visual hallucination in safety-critical applications. Thus, we introduce MedVR, a novel reinforcement learning framework that enables annotation-free visual reasoning for medical VLMs. Its core innovation lies in two synergistic mechanisms: Entropy-guided Visual Regrounding (EVR) uses model uncertainty to direct exploration, while Consensus-based Credit Assignment (CCA) distills pseudo-supervision from rollout agreement. Without any human annotations for intermediate steps, MedVR achieves state-of-the-art performance on diverse public medical VQA benchmarks, significantly outperforming existing models. By learning to reason directly with visual evidence, MedVR promotes the robustness and transparency essential for accelerating the clinical deployment of medical AI.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
MedVR addresses this by framing the VLM as an agent that can interleave text generation with a `Zoom-in` tool to actively inspect image regions. To train this agent without supervision, the paper proposes two novel, annotation-free mechanisms:
1.  **Entropy-guided Visual Regrounding (EVR):** This mechanism monitors the model's predictive uncertainty (entropy) during the generation of tool commands. When uncertainty is high (e.g., the model is unsure *where* to zoom), EVR triggers exploratory branching to sample different visual actions.
2.  **Consensus-based Credit Assignment (CCA):** This mechanism provides a reward signal for visual grounding. It identifies the set of all *successful* reasoning trajectories (those that led to a correct final answer) and generates a "consensus mask" based on the image regions they collectively inspected. Trajectories that align with this consensus receive an additional reward.

The authors claim that this self-supervising loop of exploration (EVR) and credit assignment (CCA) enables MedVR to achieve state-of-the-art performance on several medical VQA benchmarks, improving both accuracy and robustness.

### Strengths
1.  The paper tackles a key problem in medical VLMs: the lack of verifiable visual grounding. The proposed "annotation-free" approach, which avoids the need for expensive, step-by-step human annotations, is highly significant.
2.   The two core contributions, EVR and CCA, are original. Using intrinsic model uncertainty (entropy) to guide visual exploration (EVR) is an intelligent way to focus the agent's attention. Distilling a reward signal from the "consensus" of successful trajectories (CCA) is a clever solution to the credit assignment problem in the absence of ground truth.
3.   The paper correctly identifies the limitations of text-only reasoning and frames the solution as an agentic VLM that can interact with the image, mimicking a human clinician's workflow (e.g., zooming).

### Weaknesses
1.  **Misrepresentation of Results:** The most severe weakness is the contradiction between the paper's text and its tables. The text in Section 4.2 claims to "surpass" a key baseline on PMC-VQA, but Table 1 shows a marginal improvement (54.31 for MedVR vs. 54.29 for the Lingshu). This misrepresentation of out-of-domain performance is unacceptable and destroys confidence in the paper's conclusions.
2.  **Critical Omission of Image Resolution:** The paper's methodology is centered on a `Zoom-in` tool. This tool's function is entirely dependent on the input image resolution, which is never stated. It is impossible to know if the agent is (a) meaningfully cropping a high-resolution image or (b) trivially cropping an already-low-resolution 336x336 patch. This is a fundamental flaw that makes the core premise of the paper unverifiable.
3.  **No Evaluation of Core Detection Capability:** The `Zoom-in` tool and the entire CCA reward mechanism are critically dependent on the model's ability to *correctly* localize objects of interest. The paper provides no evaluation of the model's baseline detection or localization performance. This is a crucial omission. If the model's inherent ability to draw bounding boxes is poor, then the EVR is exploring random locations and the CCA is finding a "consensus" among incorrect boxes, which would poison the reward signal. The paper must demonstrate that the agent's visual grounding capabilities are a reliable foundation for its reasoning.
4.  **Lack of Model Scaling Analysis:** The experiments are confined to a single 7B model. There is no analysis of how this method performs at different scales (e.g., on a 3B model or a 30B+ model). This makes it unclear if the approach is a general one or a finicky result specific to the 7B backbone architecture.
5.  **Marginal OOD Performance:** Even taking the results at face value (and ignoring the misrepresentation), the OOD performance is not a strong endorsement. The model fails to demonstrate a substantial improvement on PMC-VQA (even this is easier than MedXpertQA) and is effectively tied on MedXpertQA. Given that the paper's premise is a more robust, generalizable reasoning process, these OOD results are weak.

### Questions
1.  What is the input image resolution used for the VLM? How does the `Zoom-in` tool work? Does it crop the original high-resolution image, or does it simply crop the low-resolution feature map? This detail is critical for understanding if the tool is performing a meaningful action.
2.  Can you provide any quantitative evaluation of the backbone model's zero-shot detection or localization capabilities? How can we be sure that the `Zoom-in` actions, which are the foundation of your method, are targeting clinically relevant regions?
3.  Why were no experiments conducted at other model scales (e.g., 3B or 34B)? How can readers be confident that this complex RL framework is scalable and not just tuned to one specific 7B backbone model?
4.  Regarding the OOD baseline comparisons, were all baseline models trained *only* on the OmniMedVQA training split to ensure a fair, controlled comparison for generalization? Or were your training set has similar case with the OOD data?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces an innovative approach for training medical vision-language models through an annotation-free reinforcement learning framework. The authors demonstrate that reinforcement learning can effectively supervise model behavior using only final ground-truth answers, while incorporating a consensus-driven self-supervision mechanism that leverages model convergence across multiple reasoning trajectories. This design enables the model to learn fine-grained visual reasoning without relying on costly manual annotations. I find the work highly original and thoughtfully executed—the proposed combination of uncertainty-guided exploration and consensus-based reward shaping represents a creative step forward in medical AI. Overall, this is a well-motivated, technically solid, and enjoyable paper to read.

### Strengths
- The paper is clearly written and easy to follow, with strong organization and clear motivation.

- It presents a technically rigorous method, combining reinforcement learning, uncertainty-based exploration, and self-supervised reward shaping in a coherent framework.

- The approach offers a novel way to guide RL toward fine-grained visual regions, ensuring that the model’s reasoning is grounded in meaningful image details rather than text-only patterns.

- Empirical results are strong and consistent across multiple benchmarks

### Weaknesses
- Reproducibility: Please release the code repository. For a complex RL-based framework like this, access to the full codebase and training scripts is essential for replication and community adoption.

- Evaluation scope: The paper primarily evaluates on multiple-choice medical VQA datasets, which simplifies the scoring through accuracy. How would the proposed RL and credit-assignment framework adapt to open-ended or free-text medical reasoning questions where correctness is less binary?

- Training initialization:
a) The paper claims annotation-free RL training, yet the model produces structured reasoning traces (<think>, <tool_call>, <answer>). Was a supervised fine-tuning (SFT) or instruction-tuning stage used to teach this output format before RL?
b) If so, could the authors clarify what data or synthetic prompts were used for this warm-up, and whether this step contributes to the strong structured reasoning behaviors observed in Figure 3?

### Questions
See weakness

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents MedVR, a reinforcement learning framework that enables annotation-free visual reasoning in medical vision-language models. It introduces two mechanisms: Entropy-guided Visual Regrounding (EVR), which uses model uncertainty to trigger targeted visual exploration, and Consensus-based Credit Assignment (CCA), which derives pseudo-supervision from agreement among successful reasoning trajectories. Without human annotations, MedVR achieves state-of-the-art performance on multiple medical VQA benchmarks, improving both accuracy and visual grounding in clinical reasoning tasks.

### Strengths
- Proposes an annotation-free visual reasoning framework for medical VLMs
- The idea of zooming into medical images to guide reasoning is inspired by real clinical practice, making the approach both creative and intuitively grounded
- Using entropy as a proxy for uncertainty to trigger visual exploration adds interpretability to the reasoning process
- Demonstrates consistent state-of-the-art performance across multiple medical VQA benchmarks, supported by detailed ablations validating each component
- Addresses a key limitation in medical AI—lack of fine-grained visual supervision—in a practical and scalable way

### Weaknesses
- The use of entropy as a trigger for zoom-in is interesting but under-justified. The paper could provide stronger evidence that high-entropy points truly correspond to moments where zooming in improves reasoning. Visualizations or quantitative correlations between entropy spikes and successful zoom-ins would make this claim more convincing.

- The method appears particularly well-suited for localization, lesion detection, or report generation, where visual grounding and fine-grained inspection matter most. However, these tasks are not evaluated, limiting the evidence that the approach generalizes beyond multiple-choice VQA.

- It seems that every entropy spike results in a zoom-in, but the paper does not show when or how often this happens. It would be helpful to analyze where in the reasoning chain these entropy peaks occur and whether those zooms actually lead to better answers.

- Tool-based visual reasoning is becoming increasingly common (e.g., Chain-of-Focus, Pixel Reasoner, DeepEyes). This paper mainly introduces the use of a zoom-in tool for the medical domain rather than a generally novel method. Therefore, it can feel more like a heuristic engineering improvement than a conceptual breakthrough. While combining entropy-based exploration and consensus pseudo-labeling is clever, both ideas are extensions of known techniques rather than fundamentally new principles.

To make the contribution stronger, the authors could either (1) demonstrate that the approach works on broader visual reasoning tasks including on general domains, or (2) position it as a domain-specific contribution for a medical AI venue.

Overall: The idea is interesting and well-motivated, especially for medical settings, but needs stronger justification, broader evaluation, and clearer evidence that entropy-driven zooming genuinely improves reasoning.

### Questions
- Can you provide evidence that entropy spikes truly indicate regions where zooming in is useful? For example, visualize entropy over time or correlate entropy with answer improvements.

- How frequently and at what stages do entropy spikes occur during reasoning? A simple distribution or timeline would clarify how often zoom-ins are triggered.

- The approach seems well-suited for localization, detection, or report generation—why not evaluate on those tasks?

- Would EVR + CCA generalize to non-medical visual reasoning tasks, or is it intended primarily for medical AI applications?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work aims to propose a framework to better ground the reasoning of vision-language models into the visual perception. Concretely, 2 modules are proposed: (1) a entropy-guided visual grounding module using summed token entropy based model uncertainty (2) a consensus-based credit assignment module using distilled pseudo supervision from rollout agreement (e.g., union of multiple proposed bounding boxes). A qualitative study is demonstrated and quantitative experimental results show superior performance in medical VQA tasks without using human annotations.

### Strengths
- The writing is easy to follow.
- The designed uncertainty based spatial exploration strategy based on the sum of entropy is interesting.
- The consensus module is carefully designed as a posterior refinement step.

### Weaknesses
- The usage of “evidence” in this paper is not consistent with what “evidence” means in clinical practice. In clinical setting, “evidence” means the best available external clinical evidence from systematic research, instead of from the current tested image [1]. 

[1] Sackett D L. Evidence-based medicine[C]//Seminars in perinatology. WB Saunders, 1997, 21(1): 3-5.

-	The contribution on “grounding every inferential step in verifiable visual evidence” is overclaimed. Is “zoom-in” enough to ground “every inferential step”? Is the “zoom-in” area always correct? What does “verifiable visual evidence” mean compared to “visual evidence”?
-	The image manipulation tool of the approach is limited only to “zoom-in”. A wide range of other image manipulation tools such as different filtering/denoising techniques are not explored.
-	There is no experiment justifying whether the zoom-in area is actually meaningful or not.
-	It’s unclear whether this annotation-free approach might perform better than using the annotation or not under a fair comparison’s setting. 

-	[line 104] The claim “First to endow medical VLMs with explicit visual reasoning capabilities.” is somewhat vague. This approach offers some improvements in the VQA performance, but existing powerful large MLLM such as GPT4o can also be deployed for medical tasks and has visual reasoning capabilities. What “explicit” means is unclear neither.

### Questions
[line 107] what clinical workflow is referenced here?

[line 162-164] Some implementation details are not clear enough to me: After a cropped image is obtained, is the image up-sampled to recover the full resolution or do they stay with their original resolution? How are the visual tokens of this cropped area obtained? Any additional padding? How are these visual tokens inserted into the reasoning path? What’s the latency of doing this in inference time?

Will this agent crop images multiple times at different scales in different areas in a “successful trajectory”? If yes, would these areas contradict with each other (e.g., no overlapping, indicating the model is looking at completely different areas)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents an RL framework that interleaves text-based reasoning with image-manipulation tools to achieve annotation-free visual reasoning in medical VLMs. Two mechanisms drive the approach: (i) Entropy-guided Visual Regrounding branches rollouts at high token-entropy steps to explore multiple regions of interest; (ii) Consensus-based Credit Assignment aggregates successful rollouts into a consensus mask and rewards trajectories whose visual footprints overlap with that mask. Experiments on three datasets, MedVR reports competitive performance.

### Strengths
1. The paper is well-written. 
2. The motivation is sound. 
3. The presented approach shows competitive performance in both in-domain and out-of-domain settings.

### Weaknesses
1. The paper’s comparison to prior work needs to be improved. The idea of Annotation-Free RL and Visual-Grounded Reasoning with RL has already been studied in previous works [1–7]

2. The paper lacks experiments to support the claim that the proposed method can provide robust and transparent reasoning. For example, the paper claims precise localization, but there are no quantitative grounding metrics or a human reader study to validate this. A localization-quality metric should be used. Additionally, evaluations of hallucination and robustness should be provided to support the arguments.

3. The fairness of the experiments should be clarified. It is unclear whether previous medical LLM baselines and medical RL baselines were tuned with similar rollout budgets, RL variants/settings, and training corpora. The inference settings should also be clarified to ensure a fair comparison.

[1] Right question is already half the answer: Fully unsupervised LLM reasoning incentivization. arXiv, 2025.

[2] Learning to Reason without External Rewards. arXiv, 2025.

[3] Unsupervised Post-Training for Multi-Modal LLM Reasoning via GRPO. arXiv, 2025.

[4] Grounded Reinforcement Learning for Visual Reasoning. arXiv, 2025.

[5] UniVG-R1: Reasoning Guided Universal Visual Grounding with Reinforcement Learning. arXiv, 2025.

[6] Visual Grounding for Object-Level Generalization in Reinforcement Learning. ECCV, 2024.

[7] VGR: Visual Grounded Reasoning. arXiv, 2025.

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
