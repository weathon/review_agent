# Thyme: Think Beyond Images

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 10, 4, 6, 4

## Abstract
Following OpenAI's introduction of the ``thinking with images'' concept, recent efforts have explored stimulating the use of visual information in the reasoning process to enhance model performance in perception and reasoning tasks. However, to the best of our knowledge, no open-source work currently offers a feature set as rich as proprietary models (OpenAI O3), which can perform diverse image manipulations and simultaneously enhance logical reasoning capabilities through code.

In this paper, we make a preliminary attempt in this direction by introducing \textbf{Thyme} (\textbf{Th}ink Be\textbf{y}ond I\textbf{m}ag\textbf{e}s), a novel paradigm for enabling multimodal large language models to transcend existing ``think with images'' approaches by autonomously generating and executing diverse image processing and computational operations via executable code (Figure 2). This approach not only facilitates a rich, on-the-fly set of image manipulations (e.g., cropping, rotation, contrast enhancement), but also allows for mathematical computations, all while maintaining high autonomy in deciding when and how to apply these operations. We activate this capability through a two-stage training strategy: an initial Supervised Fine-Tuning (SFT) on a curated dataset of 500K samples to teach code generation, followed by a Reinforcement Learning (RL) phase to refine decision-making. For the RL stage, we manually collect and design high-resolution question-answer pairs to increase the learning difficulty, and we propose \textbf{GRPO-ATS} (Group Relative Policy Optimization with Adaptive Temperature Sampling), an algorithm that applies distinct temperatures to text and code generation to balance reasoning exploration with code execution precision. We conduct extensive experimental analysis and ablation studies. As shown in Figure 1, comprehensive evaluations on nearly 20 benchmarks show that Thyme yields significant and consistent performance gains, particularly in challenging high-resolution perception and complex reasoning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
This paper proposes Thyme (Think Beyond Images), a framework that enables models to answer queries through a sequence of autonomous image operations such as cropping, rotating, and other manipulations. By combining supervised fine-tuning (SFT) for cold-start and reinforcement learning (RL) enhancement, Thyme achieves rich functionality via executable code generation. The results are impressive across several challenging benchmarks, particularly on high-resolution datasets like MME-RealWorld.

### Strengths
Overall, I enjoyed reading this paper and appreciate all the details. The followings are the parts that I found insightful

- Clear and well-structured presentation.
- Addresses an important problem area: reasoning in vision-language models (VLMs).
- Provides a detailed training recipe, including adaptive temperature, early stopping for repetitive n-grams, consistency rewards, math annealing, and loss applied on the last round of dialogue.
- Contains extensive implementation details in the Appendix, such as sandbox construction and other design choices.
- Strong ablation study (Table 5) demonstrating the effectiveness of different training details.
- Thorough evaluation of reward design choices in RL training (Table 6).

### Weaknesses
It would be valuable to see this approach extended along the temporal axis or even multi-image scenarios. Overall no major weakness in the paper.

### Questions
- What are the motivations for including mathematical computation data (as shown in Figure 4, right)? Specifically, these seem to involve pure math operations in code that are unrelated to visual operations—what purpose do they serve?
- What does the data distribution of the SFT dataset look like across the three difficulty categories?
- In Supplementary Section B, is it indeed describing the details of generating the SFT dataset?
- Approximately how many tokens are generated during the SFT phase and RL training phase?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents Thyme, an framework enabling multimodal large language models to autonomously generate and execute code for diverse image manipulations and mathematical computations. Building upon Qwen2.5-VL-7B, the work employs a two-stage training strategy: supervised fine-tuning on 500K curated samples followed by reinforcement learning with a novel GRPO-ATS algorithm featuring adaptive temperature sampling. The framework demonstrates impressive empirical results across ~20 benchmarks, achieving substantial improvements on challenging perception tasks and reasoning benchmarks. The engineering effort is commendable, including careful data construction, sandbox environment design, and comprehensive evaluation.

### Strengths
1. Comprehensive and well-engineered system: The paper presents a complete pipeline integrating multimodal understanding, code generation, and sandbox execution, with careful attention to practical details like error handling and security constraints.
2.  Rich functionality beyond existing work: Thyme supports diverse image manipulations (cropping, rotation, contrast enhancement) and mathematical computations, offering genuine multimodal reasoning capabilities.
3.  Well-designed data pipeline with multiple quality control stages and diverse task coverage including multi-turn interactions.
4.  GRPO-ATS with adaptive temperature sampling is a solution to the code generation reliability problem

### Weaknesses
1. The provided ablation studies (Table 5, 6, 7) suggest that the performance gains from several of these carefully designed components are limited.

2. The paper does not mention that the curated dataset and the specialized sandbox environment which are claimed as key components will be open-sourced. Whether these contents are open source is an important basis for judging the contribution of this paper.

### Questions
1. GRPO-ATS is one of the core contributions of this paper, but no ablation study is provided to verify its effectiveness. Although the temperature setting is clearly specified and intuitive motivation is provided, how to quantitatively analyze the effectiveness of this method is still unclear.

2. The curated dataset and the specialized sandbox environment will be open-sourced?

3. The paper presents "Naive SFT" in Table 5 as a baseline,  does this baseline have access to the meticulously annotated data that were exclusively prepared for RL?

### Soundness
3

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
The authors aim to finetune pretrained Qwen-2.5-VL models to perform code-assisted visual reasoning, executing Python code on images with operations like cropping, zooming, and general computation. They curate post-training datasets for supervised finetuning and reinforcement learning with substantial manual annotation effort. Training with SFT followed by GRPO yields a model with improved performance on a variety of benchmarks.

### Strengths
- Experiments are well-executed, thorough, and explore a range of design decisions.
- Dataset contribution would be substantial and fill a gap in the open-source community.
- Problem is well-motivated and timely.
- Empirical benefits are convincing on a range of benchmarks.

### Weaknesses
- Fully missing discussion of a substantial line of related work in code execution for visual reasoning: Visual Programming - Gupta et al and ViperGPT - Suris et al first introduced the idea of code execution at inference for visual reasoning, prior to OpenAI's 'thinking with images'; subsequent work such as Visual Sketchpad by Hu et al extended this to a larger set of image manipulations (including on some benchmarks reported here, albeit relying on proprietary models)
- If I understand correctly, GRPO-ATS = using temperature 0 for code and 1 for language, which feels like more of an implementation detail than a substantial methodological contribution. In my estimation, the paper would benefit from not putting as much focus on it, but rather presenting it as a finding with the justifications already discussed.
- The overall method essentially boils down to manually annotating data and doing post-training with SFT and GRPO, but explores the details needed to get things to work in the domain of code-assisted visual reasoning.

### Questions
- Perhaps I missed it, but is there a benchmark with the pretrained Qwen-2.5-VL + code execution at inference time as opposed to direct application of Qwen-2.5-VL (7B/32B) as a VLM? Is this the Qwen-2.5-VL in Table 6? The paper would benefit from clearer differentiation between inference settings.
- Will the dataset be released? Considering a substantial portion of the method relies on the manually-annotated data, this would be critical for reproducibility. The details on how the data was generated also seemed insufficient to reproduce the pipeline, like how the filtering happened in particular,  (This is my biggest concern.)

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Thyme, allowing models to autonomously generate and execute code (e.g., cropping, rotation, contrast enhancement, mathematical computations) to manipulate images and support reasoning. The authors design a two-stage training strategy: a supervised fine-tuning (SFT) on a curated ~500K-sample dataset teaching code generation and image operations, followed by a reinforcement learning (RL) phase ~ 10K images using their novel algorithm GRPO‑ATS (“Group Relative Policy Optimization with Adaptive Temperature Sampling”) to refine decision-making about code vs text generation. Thyme is evaluated across nearly 20 benchmarks spanning perception reasoning and general VQA tasks, all showing consistent improvements over baseline.

### Strengths
1. The training dataset size is huge (500K + 10K). Including diverse editing tools.
2. The SFT data and RL data are separate. Very clear. 
3. The experiments are comprehensive. Many benchmarks are included. Signal is clear -- adding this data is helpful.

### Weaknesses
Main concerns are about experiment missing baselines, and missing citations.

Experiments results in Table need more baseline models. Could you also add GPT 4 and 5, GPT O3,, Gemini 2.5 Pro, Claude, [Visual Sketchpad: Sketching as a Visual Chain of Thought for Multimodal Language Models],  [Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning]...etc, as baselines?

Missing Citations on related works (because many concepts in Thyme first appeared in these related works): [Visual Sketchpad: Sketching as a Visual Chain of Thought for Multimodal Language Models], [ReFocus: Visual Editing as a Chain of Thought for Structured Image Understanding], [Visual CoT: Advancing Multi-Modal Language Models with a Comprehensive Dataset and Benchmark for Chain-of-Thought Reasoning]

### Questions
Please see weakness. Happy to raise score if addressed.

### Soundness
2

### Presentation
3

### Contribution
3
