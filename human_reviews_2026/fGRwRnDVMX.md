# VisuRiddles: Fine-grained Perception is a Primary Bottleneck for Multimodal Large Language Models in Abstract Visual Reasoning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 8

## Abstract
Recent strides in multimodal large language models (MLLMs) have demonstrated significant progress in many reasoning tasks, but they still fail in Abstract Visual Reasoning (AVR) tasks. Our experimental findings indicate that the core bottleneck lies not only in the reasoning capabilities of MLLMs but more critically in their absence of fine-grained perception. To address this issue, we present VisuRiddles, a dedicated resource for AVR research. It consists of (i) a benchmark, collected from real-world data, for the systematic evaluation of MLLMs' AVR capabilities, and (ii) a synthesizer, which automatically generates AVR instances enriched with perceptual descriptions and reasoning chains, enabling supervised training and deeper investigation. Building on VisuRiddles, we propose a two-stage training paradigm that progressively enhances perceptual ability and strengthens reasoning, producing the Perception-Augmented Visual Reasoner (PAVR). Experiments demonstrate that PAVR unifies perception and reasoning, substantially outperforming both open-source and commercial MLLMs, thereby underscoring fine-grained perception as the primary bottleneck in AVR.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Current Multimodal Large Language Models (MLLMs) still exhibit significant limitations in Abstract Visual Reasoning (AVR) tasks. To address this issue, this paper introduces the VisuRiddles benchmark. To tackle the dual challenges of perception and reasoning inherent in AVR, the authors develop the Perception-Augmented Visual Reasoner (PAVR), which integrates enhanced perception capabilities with refined reasoning abilities. Through a novel SFT+RL paradigm, the model's fine-grained perception for AVR tasks is substantially improved, and the effectiveness of this approach is demonstrated through rigorous experiments.

### Strengths
1. The structure of this paper is extremely clear, presenting a definite logical flow.
2. The paper thoroughly discusses the deficiencies of existing general-purpose MLLMs in Abstract Visual Reasoning and validates this observation by introducing the VisuRiddles Benchmark, a comprehensive testbed covering multiple reasoning dimensions.
3. It innovatively proposes the VisuRiddles Synthesizer, an ingenious data synthesizer that generates a diverse set of AVR instances with detailed perception annotations. These instances are highly interpretable and reusable, making this a valuable contribution for future research aimed at enhancing the AVR capabilities of MLLMs.
4. The paper employs a systematic two-stage training paradigm (SFT+RL) that effectively enhances the model's abstract visual reasoning capabilities.
5. The experimental section comprehensively presents the performance of current open-source and closed-source MLLMs on the VisuRiddles Benchmark, showcasing the remarkable performance achieved by the PAVR model after the two-stage training. Furthermore, well-designed ablation studies are conducted to demonstrate the rationale and contribution of each component in the training process.

### Weaknesses
1. Detailed experimental settings and configurations should be included in the appendix to ensure the reproducibility of the reported results.
2.  A central claim of the paper is that deficient fine-grained perception is the primary bottleneck for MLLMs on AVR tasks. While the experiment in Table 3 supports this, it also shows that commercial MLLMs achieve only around 30% accuracy on Raven-like tasks even when provided with perception descriptions ('P') as input. This raises a question: is this poor performance due to the perception descriptions still being insufficiently comprehensive, or is it due to the inherent limitations of the models' reasoning capabilities? Further experiments are required to more conclusively disentangle these factors.
3. The trained model should also be evaluated on a suite of standard MLLM benchmarks. This would verify that the model's original, general-purpose capabilities have not been compromised by the specialized training (i.e., to check for catastrophic forgetting).
4. The paper states that common reasoning-enhancement strategies, such as increasing model parameter scale or applying Chain-of-Thought (CoT) prompting, yield limited benefits on AVR tasks. However, it does not provide sufficient detail on the specific implementation of the CoT prompts used for evaluating the baseline models.
5. The experiments are conducted exclusively on the Qwen-VL-7B model. To validate the generalizability of the proposed training paradigm, it should be tested on a wider range of MLLMs with varying architectures and parameter scales.
6. The paper would be more persuasive，if it could be directly compared with other MLLMs specifically designed to enhance reasoning abilities.

### Questions
1. The patterns in VisuRiddles are relatively simple and clean geometric shapes. If these shapes are replaced with more complex or "noisy" objects (such as small icons of real-world objects like cats or cars, or complex fractal patterns), will the model fail in the abstract reasoning task due to being disturbed by the semantic content of the complex objects? For a given abstract rule (such as "the number of elements increases one by one"), if the elements undergo a simple style change (for example, from black squares to blue triangles), will the model's accuracy be affected? This will test to what extent the model generalizes the abstract rule itself rather than merely memorizing specific visual features.
2. The paper convincingly demonstrates that fine-grained perception is the key bottleneck. However, Table 3 shows that even when perception descriptions (denoted as "P") are provided, powerful models like GPT-4V still perform poorly (with an accuracy rate of approximately 30%). What is the essence of these failures? When perception prompts are provided, do the models fail to understand the prompts, or do they understand the prompts but make fundamental logical errors? If a better perception description is provided, will the models be able to correctly arrive at the answer? If not, does this prove that reasoning ability remains the core limitation?
3. The SFT+RL mentioned in this article is a training paradigm specifically for the VisuRiddles domain. Has the final PAVR model been evaluated on standard and general multi-modal large model benchmark tests (such as MMBench, etc.)? Will such training weaken the original basic capabilities of the model?
4. The experiments in the paper were conducted only on the Qwen2.5-VL-7B model. Have you ever attempted to apply this SFT+RL training paradigm to other models with different parameter sizes or to other MLLM architectures?
5. The paper states that the benefits brought by the Chain-of-Thought (CoT) hinting mechanism to the baseline model are limited. Could you provide the detailed information for this part?
6. There are many MLLM models that enhance reasoning capabilities. They can also be compared with PAVR on the VisuRiddles benchmark.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper identifies fine-grained visual perception as a key bottleneck limiting the reasoning abilities of multimodal large language models (MLLMs). To study this limitation, the authors introduce VisuRiddles, a benchmark designed to assess Abstract Visual Reasoning (AVR) across multiple dimensions such as numerosity, attributeness, style, position, and spatiality. They also propose a new method, Perception-Augmented Visual Reasoner (PAVR), which applies supervised fine-tuning (SFT) and reinforcement learning (GRPO) to train Qwen2.5-VL-7B on VisuRiddles. Empirical results show that PAVR outperforms several zero-shot MLLMs, suggesting that perception-oriented fine-tuning can partially mitigate AVR limitations.

### Strengths
1. The paper tackles an important and timely problem - the limited fine-grained perception of MLLMs - which is of clear relevance to the ICLR community.
2. Introduction of a new benchmark (VisuRiddles) focused specifically on AVR is valuable.
3. The benchmark covers diverse aspects of perceptual reasoning (numerosity, attributeness, style, position, spatiality).
4. The paper demonstrates clear limitations of several open and proprietary model families on the benchmark.
5. The proposed method achieves improved results over zero-shot baselines, showing the benefit of perception-focused training.
6. The literature review is comprehensive and the writing is generally clear and accessible.

### Weaknesses
1. Limited novelty of the benchmark: The benchmark largely repackages existing datasets (Chinese National Civil Service Examination, RAVEN, and Sudoku) with limited methodological innovation. Details of dataset construction are missing. For instance:
    1. Were questions from the Civil Service dataset used as-is or modified?
    2. How were 100 RAVEN questions selected from the 70k instances?
    3. A pseudo-code description of the “Synthesis Algorithm” (Fig. 2a) would improve reproducibility.
2. Limited methodological novelty: The approach (SFT + GRPO on Qwen2.5-VL-7B) applies standard fine-tuning methods rather than introducing new learning techniques. The contribution is mainly empirical.
3. Unfair evaluation setup: The trained PAVR model is evaluated on the same benchmark it was fine-tuned on, whereas competing models are evaluated zero-shot. This limits the fairness and interpretability of the comparison.
4. Missing training details: The paper does not specify the train/validation/test splits used for fine-tuning PAVR, leaving open the possibility of data leakage or overfitting to the benchmark distribution. The claimed improvements might be in-distribution effects rather than true reasoning gains.
5. Limited analysis: While Appendix D attempts to evaluate transfer to other tasks, details are insufficient. Stronger evidence is needed to show that perceptual training benefits general reasoning.
6. The discussion stops short of exploring why models fail across AVR dimensions. A qualitative error analysis would be helpful.
7. The structured perceptual descriptions and reasoning chains are generated using only Gemini 2.5-Flash-Think. The paper should justify this choice and assess how dependent results are on this specific model.
8. Missing baselines: The evaluation omits several relevant MLLMs (e.g., LLaVA, InstructBLIP) and non-MLLM baselines such as supervised CNNs or few-shot methods (e.g., Prototypical Networks, SNAIL, MetaBaseline).
9. Figure 1a shows some models performing below random-guess levels without discussion or explanation.



Minor comments:
1. Describing VisuRiddles as “real-world data” is misleading since the benchmark primarily consists of synthetic, 2D, geometry-based tests.
2. Clarify the distinction between Position and Spatiality dimensions.
3. Specify whether few-shot evaluation was used for baseline models.
4. The description of prior benchmark limitations in Section 2.1 (“partial reliance on external knowledge and lack breadth in reasoning coverage”) is too vague—consider providing examples.
5. Section 2.2 should explicitly highlight how PAVR differs from prior approaches.
6. The claim “we target the most challenging part of AVR tasks” (Section 3.1) is unsupported—please justify what makes these dimensions the most difficult.
7. Clarify how “question length” is computed in Table 1 for image-based data, given that tokenization methods differ across models.

### Questions
1. How exactly was the data from each source processed and filtered? Could you include pseudo-code or a more detailed benchmark construction pipeline?
2. How are the train, validation, and test splits organized for PAVR training?
3. Why was Gemini 2.5-Flash-Think used for generating reasoning chains, and how sensitive is the system to this model choice?
4. Did you consider evaluating on additional MLLMs (e.g., LLaVA, InstructBLIP) or on supervised/few-shot models?
5. How do you interpret the below-random results observed in Fig. 1a?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the limitations of Multimodal Large Language Models (MLLMs) in Abstract Visual Reasoning (AVR) which are tasks such as Raven’s Progressive Matrices and visual riddles that require structural and relational perception. The authors hypothesize that fine-grained perception, rather than reasoning itself, is the main bottleneck in these models.
To address this, they propose VisuRiddles, which consists of 
1) A benchmark of 1000 diverse AVR problems collected and curated from real-world riddles, covering categories such as numerical, positional, stylistic, and spatial reasoning.
2) A synthesizer, which programmatically generates abstract visual puzzles paired with structured perceptual descriptions and chain-of-thought reasoning traces.

With VisuRiddles, they rain a model called the Perception-Augmented Visual Reasoner (PAVR) through a two-stage process:
1) Supervised Fine-Tuning to strengthen fine-grained perceptual understanding using synthetic perceptual annotations.
2) Reinforcement Learning via Group Relative Policy Optimization (GRPO) to improve reasoning reliability and perceptual grounding.

Experiments on the VisuRiddles benchmark show that PAVR significantly outperforms open-source and commercial MLLMs (including GPT-4o, Gemini 2.5, Claude 3.7, and Qwen2.5VL), achieving 46.8% accuracy compared to ~25–33% for top baselines.

### Strengths
1) The paper presents rigorous empirical analysis, including comprehensive comparisons with both open-source and commercial MLLMs. Ablation studies are detailed and support the key claims (e.g., perception annotations yield +42.7% improvement, RL adds +7.3% further gains).
2) The findings have broad implications for MLLM development, suggesting that improving perceptual resolution and grounding may be more impactful than simply enlarging models or adding reasoning traces.

### Weaknesses
1) While the synthesizer is innovative, the paper admits that generated riddles are “deliberately easier” and may lack the richness and noise of real-world abstract reasoning tasks. This could limit transferability.
2) Although VisuRiddles is comprehensive, stronger evidence of generalization would come from evaluating PAVR on unseen external datasets (e.g., MARVEL, PuzzleVQA, or VisLogic)
3) The “rethinking” phenomenon observed in PAVR is intriguing but underexplored. Quantitative metrics on perceptual correction or reasoning consistency would strengthen the argument.

### Questions
1) What measures ensure that the perceptual descriptions generated by the API labeling stage are accurate and consistent? Could noisy annotations harm training?
2) Is there quantitative evidence that PAVR’s improvements are more from perceptual enhancements than reasoning policy optimization (e.g., via perceptual accuracy metrics)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work presents VisuRiddle, a benchmark + data synthesis framework that is designed to improve MLLMs’ capability on Abstract Visual Reasoning (e.g. RAVEN-style IQ Tests and Sodoku). The authors also provide a baseline method, PAVR, for VisuRiddle. PAVR involves a 2-stage training - first using SFT for more accurate perception, then using GRPO for more optimized reasoning. The proposed PVAR baseline model, built on top of Qwen2.5VL-7B, tops the VisuRiddle leaderboard over many larger open source MLLMs as well as common proprietary models.

### Strengths
I believe the authors have presented a project in high completeness. This manuscript offers clear descriptions about the motivations, the method designs, and the overall contributions.

### Weaknesses
My only (and perhaps relatively trivial) concern is regarding the generalizability of the new PAVR baseline method. Although PAVR is shown to offer a high bar for the self-made VisuRiddle benchmark, how does PAVR perform on existing AVR benchmarks? So far, the authors have only shown its performance on VisuLogic in Appendix D. But I believe PAVR needs to be further tested on alternative benchmarks regarding its consistency, such as LogicVista and/or VOILA for rigorous logic reasoning, and MathVerse and SeePhys for abstract perception.

**References**

- LogicVista: https://github.com/Yijia-Xiao/LogicVista

- VOILA: https://huggingface.co/datasets/nlylmz/VOILA

- MathVerse: https://github.com/ZrrSkywalker/MathVerse

- SeePhys: https://huggingface.co/datasets/SeePhys/SeePhys

### Questions
Since the SFT and GRPO stages to tune PVAR involve using synthesized instances created by the VisuRiddle pipeline, I would like to confirm that none of these instances used in training overlap with those in the actual VisuRiddle benchmark during evaluation.

### Soundness
4

### Presentation
4

### Contribution
3
