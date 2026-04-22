# T2I-ReasonBench: Benchmarking Reasoning-Informed Text-to-Image Generation

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 6, 2

## Abstract
Text-to-image (T2I) generative models have achieved remarkable progress, demonstrating exceptional capability in synthesizing high-quality images from textual prompts. While existing research and benchmarks have extensively evaluated the ability of T2I models to follow the literal meaning of prompts, 
their ability to reason over prompts to uncover implicit meaning and contextual nuances remains underexplored. To bridge this gap, we introduce T2I-ReasonBench, a novel benchmark designed to explore the reasoning capabilities of T2I models.
T2I-ReasonBench comprises 800 meticulously designed prompts organized into four dimensions: \textbf{(1) Idiom Interpretation}, \textbf{(2) Textual Image Design}, \textbf{(3) Entity-Reasoning}, and \textbf{(4) Scientific-Reasoning}. These dimensions challenge models to infer implicit meaning, integrate domain knowledge, and resolve contextual ambiguities. To quantify the performance, we introduce a two-stage evaluation framework: a large language model (LLM) generates prompt-specific question-criterion pairs that evaluate if the image includes the essential elements resulting from correct reasoning; a multimodal LLM (MLLM) then scores the generated image against these criteria. 
Experiments across 16 state-of-the-art T2I and unified multimodal models reveal critical limitations in reasoning-informed generation. Our comprehensive analysis indicates that the bottleneck of current models is in reasoning rather than generation. Our findings underscore the necessity to improve reasoning capabilities in next-generation T2I and unified multimodal systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces T2I-ReasonBench, a novel benchmark designed to evaluate the reasoning capabilities of text-to-image (T2I) models, moving beyond literal prompt adherence. The benchmark features 800 prompts across four dimensions: idiom interpretation, textual image design, entity reasoning, and scientific reasoning. It employs a two-stage evaluation framework where an LLM generates question-criteria pairs, and an MLLM scores the resulting images. Evaluations of 16 state-of-the-art models show that reasoning, rather than generation, is the primary bottleneck for current systems.

### Strengths
1. This work addresses the significant and under-explored problem of evaluating T2I models' reasoning capabilities, moving beyond existing benchmarks that focus on literal prompt-image alignment.
2. The benchmark introduces novel dimensions, "Idiom Interpretation" and "Textual Image Design," which challenge models with complex, abstract tasks that require inferring implicit information rather than just following explicit instructions.
3. Through a comprehensive evaluation of 16 SOTA models and an insightful LLM-rewrite experiment, the study compellingly demonstrates that reasoning is the primary bottleneck for current T2I models and highlights the superiority of integrated reasoning designs.

### Weaknesses
1. The benchmark's core "AI-evaluating-AI" evaluation framework is a key weakness, as its reliability depends entirely on the AI models used for evaluation. 
a.	If the criteria-generating LLM (DeepSeek-R1) itself possesses biases, knowledge gaps, or reasoning errors, it will produce flawed question-criterion pairs  from the very start.
b.	This pipeline is susceptible to compounding errors, where any biases or misunderstandings from the LLM in the first stage are amplified by the MLLM's own limitations in the second.
The paper lacks an in-depth analysis of this potential evaluation bias.


2. The benchmark's overall size of 800 prompts  is relatively small, which may limit its robustness for a comprehensive evaluation. This limitation is particularly evident in the "Scientific-Reasoning" dimension, which attempts to cover four vast and distinct disciplines—physics, chemistry, biology, and astronomy —with only 200 prompts, which may limit its robustness for a comprehensive evaluation, particularly in the ‘Scientific-Reasoning’ dimension, where coverage seems sparse across diverse disciplines.

3. Several tasks within the benchmark, particularly in the Entity-Reasoning dimension, appear to test factual knowledge retrieval more than inferential reasoning. For example, the prompt "The first mammal successfully cloned from an adult somatic cell in 1996" primarily assesses whether the model knows the answer is "Dolly the sheep". While knowledge is a prerequisite for reasoning, this specific task seems to conflate "knowing a fact" with the more complex process of "reasoning from facts" to produce a visual output.

4. For improved readability, the structure of Section 4 could be slightly reorganized. The paper's primary metric, the $T2I-ReasonScore$, is defined entirely within the section's introductory text, while its subsequent validation is placed in the sole subsection (4.1). This organization slightly de-emphasizes the metric's definition; creating a dedicated subsection for the metric's formulation would provide a more balanced and intuitive flow for the reader.

### Questions
1.	In your pipeline experiment (Section 5.3), you present a very interesting finding: using more explicit, LLM-rewritten prompts decreased the 'Textual Image Design' scores for GPT-Image-1 and Nano-Banana. You attribute this to the detailed prompts limiting the models' "creative freedom". This seems to reveal a tension between "reasoning"  and "creativity". How do you view this trade-off?

2.	Could you please clarify the operational definition of 'reasoning' used in this benchmark and explain how you distinguish this from simple fact recall?

3.	Could you provide more detailed statistics about the benchmark, specifically the distribution of prompts across the scientific subcategories (e.g., the number of prompts for physics, chemistry, biology, and astronomy)?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces T2I-ReasonBench, a new benchmark designed to evaluate the reasoning capabilities of text-to-image (T2I) models, moving beyond the literal prompt-following assessments of existing work. The benchmark consists of 800 prompts across four dimensions: Idiom Interpretation, Textual Image Design, Entity-Reasoning, and Scientific-Reasoning. The authors also propose a two-stage automated evaluation framework where an LLM generates prompt-specific criteria and a Multimodal LLM (MLLM) scores the generated images, producing a 'T2I-ReasonScore'. The paper benchmarks 16 state-of-the-art models and concludes that reasoning, rather than image generation fidelity, is the primary bottleneck for current models. A key finding is that performance significantly improves when an external LLM first rewrites implicit prompts into explicit ones.

### Strengths
- The paper addresses a timely and critical problem in generative AI: moving beyond surface-level text-image alignment to evaluate the deeper reasoning capabilities of T2I models. This is an important direction for the field.
- The proposed benchmark is reasonably comprehensive, with four distinct dimensions that probe different facets of reasoning, from figurative language (idioms) and creative planning (textual design) to world knowledge (entities) and physical principles (scientific reasoning).
- The experiment using an LLM to rewrite implicit prompts into explicit ones provides a valuable insight. It effectively decouples the reasoning and generation tasks, offering compelling evidence that the reasoning ability of T2I models is a major performance bottleneck.
- The experimental analysis is comprehensive, covering 16 different models. The experiment using a two-stage pipeline (LLM-rewrite + T2I-generate) is particularly insightful, providing strong evidence that reasoning is a major bottleneck for current T2I systems.

### Weaknesses
- Potential for evaluator bias: The framework uses Qwen2.5-VL as the automated scorer. Given that Qwen-Image is one of the top-performing open-source models under evaluation, this raises a serious concern about potential 'in-family' bias. The human correlation analysis, conducted on an unspecified subset of only 5 models, is not sufficient to rule out this potential bias across all 16 evaluated models. This concern undermines the reliability of the reported model rankings.
- Arbitrary evaluation metric: The final T2I-ReasonScore is a weighted average of sub-scores, with manually set weights (e.g., [0.7, 0.2, 0.1]). The paper provides no sensitivity analysis to show how the model rankings would change with different weights. This makes the final scores seem arbitrary and potentially not robust.
- Oversimplified interpretation of results: The paper claims that the performance drop of the Nano-Banana model on rewritten prompts implies a 'superior internal reasoning module'. This is an overstatement. A more plausible alternative, which the paper even acknowledges in the context of Textual Image Design, is that the verbose, explicit prompts are simply a poor format for this model's input processor, constraining its creative abilities. This nuance is lost in the main conclusion.
- Reasoning vs. memorization: The benchmark does not sufficiently disentangle genuine reasoning from the retrieval of memorized associations. For idioms and specific entities, it is highly likely that top-performing models are leveraging patterns seen in their massive training datasets rather than performing multi-step reasoning. The paper acknowledges this possibility but does not offer a solution, which challenges the benchmark's core objective.

### Questions
- Can you comment on the potential for systemic bias from using Qwen2.5-VL to evaluate Qwen-Image? To strengthen the benchmark's credibility, have you considered cross-validating the model rankings with a powerful, architecturally distinct MLLM (e.g., GPT-4V or a LLaVA variant)?
- Could you provide a sensitivity analysis for the score weights? How much do the model rankings change if the weights for reasoning, detail, and quality are varied? This would help establish the robustness of your findings.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces T2I-ReasonBench, a new benchmark to evaluate reasoning capabilities in text-to-image generation across four dimensions: Idiom Interpretation, Textual Image Design, Entity-Reasoning, and Scientific-Reasoning. It proposes a two-stage evaluation pipeline where a large language model (LLM) first generates prompt-specific question–criterion pairs, and then a multimodal LLM scores the generated image against these criteria, yielding a quantitative metric called T2I-ReasonScore. Using this framework, the authors evaluate 16 state-of-the-art T2I models and find that existing models struggle with prompts requiring deeper reasoning. The results suggest that the primary bottleneck of current T2I models lies in their reasoning ability rather than low-level image generation quality.

### Strengths
1. This work addresses a critical gap by focusing on reasoning capabilities in T2I generation, an aspect that previous benchmarks largely ignored. By challenging models with idioms, complex design tasks, entity knowledge, and scientific scenarios, it goes beyond surface-level prompt-to-image alignment to evaluate deeper understanding and inference in image generation. The benchmark comprises 800 carefully curated prompts spanning four diverse reasoning dimensions. This thorough coverage ensures that a wide range of reasoning skills (from interpreting figurative language to applying scientific laws) are tested, providing a balanced and systematic assessment of models under consistent conditions.
2. The paper introduces a two-stage LLM-based evaluation pipeline that produces a fine-grained metric tailored to reasoning-intensive tasks. Notably, the authors validate this automatic metric by showing it correlates more strongly with human judgment than standard metrics like CLIP or VQA scores, lending credibility to their approach.
3. The authors evaluate 16 modern T2I and multimodal models, providing a rich comparative analysis of their reasoning performance. The experiments yield valuable insights – for example, even advanced models are shown to be limited by reasoning rather than generative fidelity – which can guide future research.

### Weaknesses
1. The evaluation framework heavily relies on an LLM and a multimodal model as judges, which introduces potential bias and uncertainty in the scoring. Although the authors demonstrate that their metric aligns well with human evaluations, the dependence on AI evaluators (which have their own limitations) raises concerns about whether the scores always faithfully reflect human-perceived reasoning quality. The two-stage evaluation process is fairly complex and computationally intensive. It depends on a specific LLM (DeepSeek-R1) and a large vision-language model (Qwen2.5-VL) for each evaluation, which may hinder reproducibility and adoption – researchers without access to these models or similar compute resources could find it difficult to apply the benchmark or reproduce the exact scoring.
2. The chosen four dimensions, while sensible, may not cover the full spectrum of reasoning needed for image generation. For instance, certain forms of commonsense or social reasoning might fall outside these categories, suggesting that T2I-ReasonBench could be further expanded to ensure no important reasoning skill is left untested.

### Questions
1. How do the authors ensure that each prompt cleanly fits into one of the four reasoning categories without overlap? For example, if an idiom prompt also involves some scientific or commonsense reasoning, was it categorized in a single dimension, and what criteria were used to handle such overlaps or ambiguities in prompt classification?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces T2I-ReasonBench, a new benchmark aimed at evaluating the reasoning capabilities of text-to-image (T2I) generative models. Existing benchmarks primarily assess literal text-image alignment (e.g., object color, count, or position). The authors argue that true understanding requires reasoning beyond surface text cues.

T2I-ReasonBench covers four reasoning dimensions:
1. Idiom Interpretation — understanding figurative language and implicit meaning.
2. extual Image Design — reasoning about communicative intent and integrating text visually.
3. Entity-Reasoning — inferring unstated entities using world knowledge.
4. Scientific-Reasoning — applying physical or scientific principles.

The benchmark includes 800 prompts and introduces a two-stage evaluation framework:
Stage 1: An LLM generates question-criterion pairs per prompt to guide evaluation.
Stage 2: An MLLM scores generated images against those criteria, producing the composite metric T2I-ReasonScore (weighted by reasoning, details, and quality).

### Strengths
- It moves beyond compositionality and literal alignment toward reasoning-aware evaluation — an underexplored but crucial capability for T2I systems.
- It covers a wide range of models, including diffusion, unified, and proprietary systems.

### Weaknesses
- The work relies excessively on LLMs, yet lacks a thorough verification process for hallucinations or incorrect responses.
- Although several other related metrics already exist—such as TIFA [1], and I-HallA [2]—the paper does not provide any comparison with them.
- The paper attempts to tackle four major challenges at once, resulting in an unfocused contribution. Since the benchmark is not large-scale, it should have been carefully curated; however, it ends up being of ambiguous size and quality.
- For the Scientific Reasoning component, experts should have been consulted or a rigorous human verification process should have been conducted. For example, “A trampoline with an iron ball on it” only causes the surface to deform under Earth-like gravity; in an environment with different gravity, the outcome would differ. Scientific reasoning requires precise conditions and verification.
- Figure 2 is supposed to illustrate the overall framework and methodology of the paper, yet it only presents a high-level concept of the data collection process without concrete details. The caption also lacks substance and clarity.
- I don't think the correlation score results are high enough.

[1] Hu, Yushi, et al. "Tifa: Accurate and interpretable text-to-image faithfulness evaluation with question answering." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

[2] Lim, Youngsun, Hojun Choi, and Hyunjung Shim. "Evaluating Image Hallucination in Text-to-Image Generation with Question-Answering." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 25. 2025.

### Questions
- As far as I know, even LLMs have not yet mastered idiom interpretation. How can a T2I model be expected to capture such figurative meaning accurately? Moreover, I don’t understand the rationale for translating those idiomatic expressions into images in the first place.
- In the Textual Image Design section, how would one evaluate a prompt like “Create a minimalist promotional poster for a workshop on simplicity in design”? This is a topic on which even humans lack consensus, yet the paper relies solely on a rudimentary LLM-based metric. Do the authors genuinely believe this provides a convincing or meaningful evaluation?

### Soundness
2

### Presentation
2

### Contribution
2
