# Interleaving Reasoning for Better Text-to-Image Generation

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 6

## Abstract
Unified multimodal understanding and generation models recently have achieve significant improvement in image generation capability, yet a large gap remains in instruction following and detail preservation compared to systems that tightly couple comprehension with generation such as GPT-4o.
Motivated by recent advances in interleaving reasoning, we explore whether such reasoning can further improve text-to-image (T2I) generation. 
We introduce Interleaving Reasoning Generation (IRG), a framework that alternates between text-based thinking and image synthesis: the model first produces a text-based thinking to guide an initial image, then reflects on the result to refine fine-grained details, visual quality, and aesthetics while preserving semantics. 
To train IRG effectively, we propose Interleaving Reasoning Generation Learning (IRGL), which targets two sub-goals: (1) strengthening the initial think-and-generate stage to establish core content and base quality, and (2) enabling high-quality textual reflection and faithful implementation of those refinements in a subsequent image. 
We curate IRGL-300K, a 300K-scale dataset organized into six decomposed learning modes that jointly cover learning text-based thinking, and full thinking–image trajectories. 
Starting from a unified foundation model that natively emits interleaved text–image outputs, our two-stage training first builds robust thinking and reflection, then efficiently tunes the IRG pipeline in the full thinking–image trajectory data. 
Extensive experiments show SoTA performance, yielding absolute gains of 5–10 points on GenEval, WISE, TIIF, GenAI-Bench, and OneIG-EN, alongside substantial improvements in visual quality and fine-grained fidelity. 
As an early exploration, our results demonstrate that interleaving reasoning is a powerful paradigm for advancing T2I.
The code, model weights and datasets will be released in: https://github.com/Osilly/Interleaving-Reasoning-Generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a novel **Interleaving Reasoning Generation (IRG)** framework for text-to-image (T2I) generation. The key idea is to alternate between text-based reasoning and image synthesis, enabling the model to progressively refine image quality and semantics. To support this, the authors curate a **large-scale IRGL-300K dataset** comprising six decomposed learning modes designed to teach both textual reasoning and full image generation trajectories. Experimental results on **5** benchmarks, including **GenEval, WISE, TIIF, GenAI-Bench, and OneIG-EN**, show consistent improvements over existing T2I models or reasoning T2I approaches.

### Strengths
1. The paper is well-structured and clearly presented, with thoughtful visual aids such as color-coded pipeline diagrams (e.g., Fig. 2) that make complex processes easy to follow.  
2. It addresses an **important and timely challenge** in multimodal generation to bridge reasoning and image synthesis, and provides not only a methodological contribution but also a valuable 300k-sample dataset, which could benefit future research.  
3. The experimental validation is strong, demonstrating clear and consistent gains across multiple benchmarks, indicating that the proposed approach is both effective and generalizable.

### Weaknesses
1. **Lack of ablation on pipeline design rationale**. While the paper offers a detailed description of the six learning modes and two-stage training, it does not empirically justify whether each component is necessary. For example, it remains unclear if *Initial Thinking Understanding Learning* is crucial, or if training could skip directly to *Initial Thinking Generation Learning* since in Eq. 3, $T_{in}$ will not consider any visual input to generate $T\_{out}^{(1)}$. An ablation isolating these stages would help clarify the necessity and impact of each design choice.  
2. **Limited qualitative demonstrations**.  The paper would benefit from showing more examples of intermediate reasoning outputs (e.g., textual thoughts and intermediate images). Figure 4 showcases final results, but visualizing reasoning steps, similar to the teaser in Fig. 1, would provide clearer insight into how the reasoning process contributes to image refinement.  
3. **Insufficient comparison to closely related works**. The paper overlooks recent open-source approaches with similar reasoning-generation mechanisms, such as *Self-Correcting with LLM* [1] and *CoT-based Image Generation* [2]. Including these in the comparison, even qualitatively or in terms of design philosophy, would strengthen the paper’s contextual positioning.
4. **Limited exploration of higher reasoning steps**. The paper only discusses cases where $n \le 2$. Extending the study to higher reasoning depths could further emphasize the strength of the proposed *interleaving* reasoning design. For instance, even if trained with $n=2$, the model could use a sliding window of size 2 to perform continuous reasoning, achieving $ n=3, 4$, or even $5$. This would provide deeper insights into how well the model generalizes to multi-step reasoning scenarios.

---

**Overall:**  
This is a promising and well-written paper tackling an important topic in T2I reasoning. The contributions are solid, but the lack of discussion and empirical validation of the training pipeline design slightly weakens the argument for its necessity. The current score is 4, which could be raised if the paper includes clearer ablations and more intermediate reasoning visualizations.

---

[1] Wu, T. H. et al. *Self-Correcting LLM-Controlled Diffusion Models.* CVPR 2024.  
[2] Guo, Z. et al. *Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step.* CVPR 2025.

### Questions
1. Why can the proposed training pipeline validate the process? (weakness 1) 
2. Can the model generalize to a higher reasoning step with a fixed window size? (weakness 4)
3. Given Eq. 4, what will the $T_q$ look like? 
4. What is the inference time comparison compared to other approaches? Since this approach requires extra reasoning, will this double the inference time? 
---
 **(suggestion)**: It can increase the readability to add a prefix in Eq. 1. For example, put "image understanding" for the first line and "T2I with CoT" for the second line.

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Interleaving Reasoning Generation (IRG), a framework that alternates between text-based thinking and image synthesis to improve T2I generation quality. The key idea is: (1) generate initial thinking and an initial image, (2) reflect on the initial image to identify improvements, and (3) generate a refined image. To train IRG, the authors propose IRGL-300K, a dataset with six decomposed learning modes covering text-based thinking and full thinking-image trajectories. A two-stage training pipeline is employed: Stage 1 builds reasoning and reflection capabilities across all six tasks, while Stage 2 fine-tunes the complete IRG pipeline. Experiments show improvements of 5-10 points on GenEval, WISE, TIIF, GenAI-Bench, and OneIG-EN benchmarks.

### Strengths
**Intuitive Framework**: The paper's core contribution, the Interleaving Reasoning Generation (IRG) framework, is conceptually elegant and highly intuitive. The "think-generate-reflect-refine" process mirrors human creative strategies, making the approach easy to understand and appreciate. 

**Methodological Depth**: The creation of the IRGL-300K dataset and the design of the two-stage, six-mode training process demonstrate a deep, systematic approach to solving a challenging training problem.

### Weaknesses
The paper presents an interesting idea but is undermined by significant methodological flaws, an absence of critical analysis, and unconvincing experimental results. The core weaknesses fall into two main categories: the questionable contribution of the proposed method and the unaddressed practical costs.

**Unjustified Complexity and Missing Cost-Benefit Analysis**

The primary weakness is the paper's failure to justify the substantial increase in complexity and cost. The Interleaving Reasoning Generation (IRG) framework introduces significant overhead at both training and inference stages, yet the benefits are not clearly demonstrated.
Critically Absent Efficiency Analysis: The proposed 2-turn method involves a 5-step sequential pipeline (initial thought → initial image → encode → reflection → refined image). This inherently introduces significant latency compared to single-step generation. The paper completely omits any discussion of inference efficiency, including critical metrics like inference time, memory consumption, or throughput. Without this analysis, it is impossible to assess the practical viability of the method.


**Questionable Marginal Utility**

The paper's own results cast doubt on the core value proposition of the second "improving" turn.
Evidence of Negative Returns: According to Table 4, single-turn generation from the IRG-trained model outperforms the 2-turn generation on the WISE benchmark (0.79 vs 0.77). This suggests that the reflection and refinement step can actually degrade image quality. The paper fails to investigate or explain these failure cases, leaving a critical gap in understanding the method's behavior.

### Questions
The paper's core idea is intuitive, but the proposed solution is exceptionally heavyweight, involving significant costs for both data creation and inference. Given this high investment, the actual benefits need to be clearly demonstrated, but the evidence presented is unconvincing and contradictory. Specifically, the results in Table 4 are confusing, as automated benchmarks show little to no improvement from the second reasoning turn. Furthermore, in the qualitative examples (Figure 4), the refined image can lose important details. The central issue is whether the method's substantial costs are justified by a reliable and significant improvement in image quality. The current results fail to make a convincing case. I hope the authors can provide clarification on these critical points.

**Training & Data Cost**: What were the resource costs (e.g., API expenses, GPU hours) to create the IRGL-300K dataset and train the model?

**Inference Cost**: How much slower is the 2-turn generation process compared to a single turn? Can you provide concrete numbers for inference time and memory usage?

**Evaluation Contradiction**: MLLM ranking prefers 2-turn generation, but benchmark scores in Table 4 do not. Have you conducted human evaluations to confirm that the second turn offers a genuine improvement and rule out potential MLLM bias?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes Interleaving Reasoning Generation (IRG), a framework that improve T2I generaton by interleaving reasoning.


The key idea is to let the model first "think" (generate textual reasoning), then generate an image conditioned on that reasoning, and finally "reflect" on the generated image to refine it. 
The framework differs from previous "multi-turn post-hoc self-refinement / editing" methods, IRG integrates reasoning and refinement in a single end-to-end pipeline via the supervision of "text–thinking-image–thinking–image" trajectory.
To support this paradigm, this paper introduces IRGL-300K, a large-scale dataset with six decomposed learning modes designed to train both text-based reasoning and full image generation capabilities.


Comprehensive experiments demonstrate strong improvements (5–10 points) across multiple benchmarks (GenEval, WISE, TIIF, GenAI-Bench, OneIG-EN).


The approach is conceptually interesting, technically solid, and empirically effective.

### Strengths
1. The proposed Interleaving Reasoning Generation (IRG), which differs from previous  "multi-turn post-hoc self-refinement / editing" methods. Instead of iterative post-hoc image editing, IRG integrates reasoning and refinement in a single end-to-end pipeline via "text–thinking-image–thinking–image" supervision. This unified formulation is both elegant and innovative.

2. The approach is not limited to improving semantic alignment (as most prior reflection-based methods do) but also targets to improve visual quality, including texture rendering, fine-grained detail, and shadow realism.

3. The IRG pipeline achieves consistent gains of 5–10 points on five diverse text-to-image benchmarks (GenEval, WISE, TIIF, GenAI-Bench, and OneIG-EN), demonstrating strong generalization and scalability.

4. The authors recognize that standard benchmarks are often insensitive to fine-grained visual improvements and supplement them with multi-MLLM evaluator comparisons, providing convincing qualitative and quantitative validation.

### Weaknesses
Seen in Questions

### Questions
My questions focus on clarity and rationale. 

1. Lines 288–290 
> The main goal of this stage is to strengthen the text-based reasoning capability, while incorporating full thinking–image trajectories to avoid degrading the core generative performance.

It is unclear why "incorporating full thinking–image trajectories" is necessary during the reasoning-focused stage.
Was this motivated by empirical observation (e.g., observed degradation of image generation without such data)?
Or is it based on the general expectation that further fine-tuning of MLLMs may cause forgetting of generative capabilities?

2. Lines 199–200
> The model exchanges and exploits multiple segments of interleaved text–image representations, a process we term Interleaving Reasoning Generation (IRG).

The meaning of "exchanges and exploits multiple segments of interleaved text–image representations" is not clear.
Does this refer to multi-modal token interleaving within the transformer architecture,
or to multi-step reasoning across modalities (e.g., text → image → text → image)? 
This statement could be misleading and might need clarification.

3. Section 2.2.2. 

This section is quite dense and terminology-heavy.
It would be valuable to explain the thought process that led to the definition of the six decomposed learning modes.
Was this decomposition empirically derived or conceptually motivated?
A clear justification would significantly improve readability and insight.

Additionally, according to the mapping pattern (Eq. 4 →Eq.  7, Eq. 5 → Eq. 8, Eq. 6 → Eq. 9),
why does Eq. 7 not include T^{(1)}_{out} as an input, while Eq. 8 and Eq. 9 do?
This asymmetry deserves clarification.

4. Section 2.2.4. 

You propose two complementary CFG-conditioning schemes (image-based and reflection-based).
Why not also include the traditional prompt-only CFG (conditioning on T_{in}) as a baseline?
Was this excluded for theoretical or empirical reasons (e.g., redundancy, instability, or negligible impact)?
Discussing this design choice would improve completeness.

5. Figure 2 is difficult to understand.

### Soundness
4

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
The paper introduced Interleaving Reasoning Generation (IRG). The main idea is to reason with both text and image during text-to-image, to produce better images in the end. Specifically, the process in this paper has 4 stages: (1) Given an input text prompt, generate textural "thinking" (2) after thinking, generate an initial image (3) generate the second textural "thinking" to refine the image (4) generate the final image based on the previous steps.

The training process includes 6 learning modes, each targeting different sub-steps above. The authors also created IRGL-300K, a training dataset for this kind of interleaved reasoning generation. The primary source of the images are from GPT-4o. The authors also used Qwen2.5-VL to generate the text data.

The authors conduct experiment on Bagel, showing that it improves the results on GenEval, WISE, TIIF, and GenAI-Bench. The authors also conduct ablation studies on various components in the training process.

### Strengths
1. The idea of let unified models to use interleaved text and image reasoning steps for text-to-image generation makes a lot of sense. It utilizes unified models' power, and goes beyond bagel's COT + image generation scheme.

2. Experiments show that the authors' approach works, and can give improvement compared to the original Bagel.

3. The authors conduct experiments on a variety of tasks, and the ablations answer many important research questions.

### Weaknesses
1. The method is heavily dependent on distilling GPT-4o images. The dataset for "initial learning stage" is adapted from a GPT-4o generated dataset. The second stage is created through GPT-4o and Bagel generated images. It remains an open question that if this method works without a strong text-to-image model to distill with. For example, if the authors can observe similar gains through data generated by another open text-to-image model? Or even from Bagel's own generations? That would make the paper much stronger.

2. There is no human evaluation in this paper. All evaluations are model based, and some are MLLM-as-a-judge results. It would be benefitial to show some human results on a subset of the testing prompts, to confirm that this method indeed improves model performance.

3. There is no discussion on the compute / inference cost of this method. With more stages and many CFG steps, there will be a lot of extra compute, and they worth discussion.

4. It would be beneficial to show some more analysis on the final model. For example, what is the lengths of the thinking steps? Are there any failure examples that the second image is not improving the first image?

### Questions
Most of my questions have been discussed in the weakness section

1. Can this method work without using GPT-4o distilled data? For example, using Qwen-Image. How will the performance change.

2. Is it possible that with Bagel's self-generated data, and some real images, we can see such gains? 

3. More analysis on the reasoning steps would be very helpful. Like more analysis on the reasoning lengths, and some failure case examples.

4. What is the inference cost of the method? Is it possible that in some cases, we can just finish generation in the first stage, to save some cost.

### Soundness
3

### Presentation
2

### Contribution
3
