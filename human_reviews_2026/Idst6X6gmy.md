# VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 4

## Abstract
Reinforcement learning finetuning (RFT) has significantly advanced the reasoning capabilities of large language models (LLMs) by enabling long chains of thought, multi-turn self-correction, and effective tool use. While recent works attempt to extend RFT to vision-language models (VLMs), these efforts largely focus on text-only reasoning conditioned on original image inputs, and do not incorporate visual reasoning in the response. In contrast, test-time methods like Visual Sketchpad incorporate visual steps but lack training mechanisms.

We introduce VTool-R1, the first RFT framework that trains VLMs to generate multimodal chains of thought by interleaving text and intermediate visual reasoning steps. VTool-R1 integrates Python-based visual editing tools into the RFT process, enabling VLMs to learn when and how to generate visual reasoning steps that enhance the final output quality. Trained with outcome-based rewards, our approach elicits strategic visual tool use for multi-modal reasoning without relying on process-based supervision. Extensive experiments on structured visual reasoning over charts and tables show that VTool-R1 enhances reasoning performance by teaching VLMs to "think with images" and generate multimodal chain of thoughts with tools. To support future research in multi-turn multi-modal reasoning, we open-source our code at https://github.com/VTOOL-R1/vtool-r1.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents VTool-R1, a reinforcement learning finetuning (RFT) framework tailored for vision-language models (VLMs) that aims to teach these models to generate truly multimodal chains of thought by effectively interleaving text and intermediate visual reasoning steps using external Python-based visual editing tools. Unlike many recent RFT efforts which only incentivize text-based reasoning conditioned on images, VTool-R1 explicitly integrates visual tool invocation and allows VLMs to "think with images," optimizing for outcome-based rewards tied solely to task accuracy. The work provides comprehensive experimental results across structured visual question answering (VQA) benchmarks using both tables and charts, demonstrates outperforming or on-par tool-augmented reasoning compared to strong baselines, and analyzes dynamics of tool use during training.

### Strengths
1. The manuscript convincingly highlights the gap in existing VLMs' inability to reason with images in a stepwise, multimodal fashion—a limitation ignored by prior RFT works focusing on text-only COTs conditioned on static images.
2. The paper benchmarks across challenging settings (tables and charts), leverages SOTA VLMs (Qwen2.5-VL), and includes both accuracy results (Table 1) and detailed training dynamics (Figure 3), encompassing accuracy, tool call rates, and execution success. These support the claimed advances.
3. The VTool-R1 framework extends outcome-based RFT principles to visual tool-use, with an explicit two-stage rollout process and a policy gradient method (GRPO), as visualized in Figure 1. This design enables conditional tool invocation and reasoning directly over modified visual states.

### Weaknesses
1. The paper’s discussion of “tool use” is overly restricted to structured visual tasks (e.g., tables and charts), yet the title and abstract emphasize that the model “learns to think with images.” This framing creates an expectation of broader visual reasoning capabilities, but the actual scope is much narrower, leading to a mismatch between presentation and content. In the Related Work section, the review of prior research on tool-augmented VLMs and visual reasoning chains is present but lacks depth. In particular, the paper omits discussion of recent works on open-domain visual-language tool use, which would provide necessary context for positioning its contribution. Moreover, the paper does not analyze failure cases or qualitatively examine incorrect tool calls. Although the authors acknowledge that “tool use is not always beneficial for accuracy”, they do not provide representative failure examples. As a result, readers cannot assess when and why the proposed method fails. Finally, in the Conclusion, the authors claim that this work “is the first to teach VLMs intermediate visual reasoning steps via RFT”. However, this statement appears overstated—it gives the impression that the method is mature and widely applicable, whereas the actual experimental scope remains limited.
2. The authors claim that this work is the first reinforcement learning framework that enables vision-language models (VLMs) to learn through intermediate visual reasoning steps combined with tool use. However, the differences from prior work are not clearly articulated. For example, existing methods such as Visual Sketchpad (Hu et al., 2024) and OpenThinkIMG (Su et al., 2025) have already introduced visual sketching or editing steps during reasoning. The core idea—integrating tool invocation and visual editing into VLMs to enhance reasoning—is not fundamentally new in the tool-augmented or multimodal reasoning community. The paper lacks a deep analysis explaining why previous approaches are insufficient and why the proposed method works better. Furthermore, the authors emphasize using outcome-based rewards instead of process supervision. Yet, the absence of process-level guidance (i.e., when to invoke a tool and how to edit an image) may turn the method into a black-box fine-tuning procedure. The paper does not sufficiently justify why this design is preferable to process supervision.
3.The visual editing toolkit used in experiments is overly simplistic—limited to actions like highlighting, masking, or drawing boxes on rows and columns. These operations apply only to structured visual tasks such as tables and charts, yet the paper generalizes the approach as a universal paradigm for visual reasoning, which is a clear overreach from structured to real-world imagery. The datasets (VWTQ, VTabFact, ChartQA) are small-scale and relatively controlled. They do not reflect the diversity or complexity of open-domain multimodal reasoning, leaving generalization to broader settings unproven. The reward model relies on a lightweight LLM to judge answer correctness, which can be unreliable due to fuzzy matching, semantic variation, and inconsistent tool outputs. The authors themselves admit that tool-call correctness cannot be precisely evaluated, suggesting potential instability in the reward signal. Training only optimizes final answers, without explicitly supervising intermediate tool calls or visual edits. As a result, the model might simply learn to occasionally invoke a tool rather than to use tools meaningfully for reasoning, which undermines the core claim. There is a lack of thorough ablations: while Table 1 compares runs with and without tool calls, the paper does not analyze cases of tool misuse, tool failure, or scaling effects across model sizes (3B / 7B / 32B). Finally, the discussion of generalization and real-world applicability is weak. The authors acknowledge that their work focuses on structured visual reasoning, but this narrow scope significantly limits the claimed impact of the method.

### Questions
1. Can the authors provide baseline comparisons and/or qualitative ablations versus directly related, recent tool-use RL works like DeepEyes, ACTIVE-O3, MM-EUREKA, and DINO-R1? This is critical to position VTool-R1’s improvements in context.
2. Is there evidence that the proposed framework generalizes to more open-ended task domains, (e.g., beyond tables/charts) and to toolsets that provide more than highlighting/masking? Have any experiments been attempted in less-structured or multi-stage reasoning domains?
3. What was the measured reliability of the LLM-based reward judge? Was there a manual verification of reward assignment on a holdout set?

### Soundness
2

### Presentation
2

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
This paper proposes VTool-R1, a reinforcement learning finetuning (RFT) framework for vision-language models (VLMs). The method aims to train VLMs to “think with images” — i.e., to generate multimodal chains of thought (CoT) by interleaving text reasoning and intermediate visual reasoning steps.
Unlike prior work that only uses images as static input, VTool-R1 integrates Python-based visual editing tools into the RFT loop, allowing models to modify images during reasoning. The approach employs outcome-based rewards rather than process supervision, trained with a Group Relative Policy Optimization (GRPO) objective adapted from DeepSeek-R1.
Experiments on structured visual reasoning tasks (tables, charts) demonstrate improvements in accuracy over both inference-only and non-tool-use baselines, using Qwen2.5-VL models (3B/7B/32B). The authors report that the model learns to selectively invoke visual tools to improve reasoning outcomes.

### Strengths
This paper presents a clear and technically solid framework that adapts reinforcement finetuning (RFT) to multimodal reasoning. I like that the authors focus on the gap between text-only reasoning and truly visual reasoning, and they provide a well-defined approach—integrating Python-based visual tools into the RL loop—to address it. The method is implemented cleanly and described in enough detail to be reproducible. The experiments, though limited in scope, are consistent and show that the model can indeed learn to use visual tools more strategically after RFT. Overall, the idea of teaching VLMs to “think with images” is appealing, timely, and potentially useful for future multimodal reasoning work.

### Weaknesses
From my perspective, the contribution feels somewhat incremental—the method mainly extends DeepSeek-R1–style RFT to VLMs without introducing new algorithmic ideas. The evaluation is narrow, focusing only on structured tasks like chart and table reasoning, which limits how convincing the results are for general multimodal reasoning. The reward design depends on an LLM-based judge, which is subjective, and the tool-use evaluation metric (simply checking if Python runs) doesn’t truly measure reasoning success. Some claims about being the “first” to enable multimodal chain-of-thought are overstated, given existing work such as Vision-R1 and Refocus.

### Questions
1. How does VTool-R1 generalize to multi-turn tool use or open-ended visual reasoning beyond structured data?

2. Can you provide human evaluation or ground-truth verification for visual tool correctness?

3. How sensitive is performance to reward signal noise (given LLM-judge subjectivity)?

4. What is the computational cost of two-stage inference and RFT rollouts relative to pure text-based RFT?

### Soundness
3

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
The paper introduces VTool-R1, a framework that applies reinforcement learning finetuning (RFT) to train vision-language models (VLMs) to generate multimodal chains of thought by interleaving text and intermediate visual reasoning steps. Unlike prior work, which focuses solely on textual reasoning given images, VTool-R1 leverages Python-based visual editing tools during training, allowing the model to learn how and when to use visual steps that can improve structured visual reasoning tasks. The framework uses outcome-based rewards to incentivize improved reasoning. Extensive experiments on chart and table-based reasoning tasks show that VTool-R1 enables VLMs to “think with images”—improving performance, especially with open-source models previously unable to use tools meaningfully.

### Strengths
1. VTool-R1 is the first research work to demonstrate RL can train VLMs to interleave visual steps within a chain of thought using python-based visual editing tools.
2. Strong, controlled experiments and comparisons with state-of-the-art baselines, including commercial and open-source models, show clear improvement and robust methodology.
3. The framework and training design have potential to generalize to more diverse tools and reasoning tasks.
4. Explanations are accessible, with step-by-step rationale for design choices, experimental setup, and limitations.

### Weaknesses
1. Single-Turn Tool Use: The model is only trained/tested with one round of tool invocation, limiting application in multi-step, interactive reasoning.
2. Limited Task Scope: The experiments focus only on structured chart and table reasoning with a simple, small, predefined set of visual editing tools.
3. Lack of Extensive Qualitative Comparison: The paper would benefit from further qualitative comparison.

### Questions
1.  How does VTool-R1 perform when applied to less-structured images (e.g., photographs or scenes)? 
2. What are the key challenges to scaling the VTool-R1 approach to multi-turn, iterative tool use? 
3. Can you provide more detailed behavioral analysis of failure modes? eg. will the model misuse some tool?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an RFT (Reinforcement Fine-Tuning) framework that supports VLM multimodal reasoning with external visual editing tool usage. It defines multimodal reasoning as a two-step rollout. The experiment in the table and chart image scenes shows that the RFT training brings improvements for baseline models.

### Strengths
1. This work extends the textual reasoning in multimodal understanding to the multimodal reasoning that involves images and text.
2. This work successfully enables VLMs to learn to integrate intermediate visual reasoning steps into text-based chains of thought in the generated response.

### Weaknesses
1. It is not recommended to use expressions like "the first RFT framework that trains VLMs to generate multimodal chains of thought". I think the expression "first" is questionable.
2. As shown in Figure 1, can only two-step Reasoning be performed during model training? What about during inference after model training? Why wasn't it designed as a reasoning process with a maximum number of rounds? This is more in line with the phenomenon of multiple iterations in multimodal reasoning.
3. The experiment was insufficient. The experiment was conducted only on the two data Settings of Table Split and Chart Split, and it was only an in-domain experiment. The baseline models and benchmarks covered in the experiment are insufficient, which fails to reflect the effectiveness and universality of this method.
4. In the results of Table 1 and Table 2, why is there a situation where the result of "Tool Use" is worse than that of "Pure Run"? For example, in Qwen2.5-VL-32B and GPT-4o.

### Questions
1. Refer to the issues raised in the weakness section.
2. The curve of the model training process shown in Figure 3 has the risk of overfitting. A significant performance improvement was achieved after only 50 training steps. How many steps are generally trained for the model in this article?

### Soundness
2

### Presentation
2

### Contribution
2
