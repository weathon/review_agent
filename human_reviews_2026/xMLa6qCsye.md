# Beyond Behavioral Alignment: Leveraging Core Cognitive Dimensions for Enhanced Human-like MLLMs

- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
The emergence of multimodal large language models (MLLMs) has led to near-human performance across various multimodal cognitive and reasoning tasks, despite relying solely on next-token prediction objectives. A critical and underexplored question is whether MLLMs trained under this paradigm truly exhibit human-like visual conceptual representations and behaviors during multimodal reasoning. To investigate this, we evaluated MLLMs on the widely-used behavioral task of Odd-One-Out (O1O), revealing a limited predictive accuracy for human choices. To address this discrepancy, we propose a novel approach: instead of merely using raw human behavioral data, we first identified core cognitive dimensions and judgmental bases from human behavioral records in O1O experiments. Subsequently, we fine-tuned Qwen2.5-VL in a data-driven manner, guided by these extracted human core cognitive dimensions, thereby markedly enhancing its behavioral consistency with humans. Intriguingly, we found that models aligned with human cognition not only maintain their generality in downstream tasks but can even achieve performance improvements. Furthermore, searchlight representational similarity analysis (RSA) and cortical projection analyses revealed increased activation in brain regions associated with problem planning and decision-making, such as the prefrontal cortex, in the fine-tuned model. This finding potentially offers a neuroscientific explanation for the observed improvements and human-like alignment.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a new method to make MLLMs more human-like by aligning them with core cognitive dimensions rather than just imitating human behavior.
Using the Odd-One-Out (O1O) task and the THINGS dataset, the authors infer latent cognitive bases (like “natural vs. artificial”) from human judgments and integrate these dimensions into the model’s training process.
Their fine-tuned model, CogAligner, achieves higher consistency with human decisions, better alignment with neural activity in decision-related brain regions, and improved generalization on reasoning benchmarks.
The work demonstrates that incorporating cognitive principles enables models to move beyond behavioral mimicry toward neurologically and cognitively grounded human-like intelligence.

### Strengths
Interesting idea. I do believe that machines cannot only learn “what”, they need to also learn “why” and “how.” Current good performance of many large models is just a result of flooding the models with plenty of training data. VLMs still perform poorly on some fundamental visual tasks, which are quite easy for humans. I think models should build on a bottom-up style, mastering the basics before the downstream tasks.

### Weaknesses
- Only verify the idea on one model, Qwen-2.5-VL-7B. The performance improvement on this single model is not statistically significant. Additionally, how much do you think the approach can generalize to larger models and other model families?
- Not perfect in MMMU-Pro. Moreover, Qwen2.5-VL-7B reports its MMMU-Pro accuracy being 38.3 (https://mmmu-benchmark.github.io/#leaderboard) or 41.0 (https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct). This paper reports 32.9, which is obviously lower. Could you explain the possible reasons?
- The writing of the current paper should be largely improved.
- The core part of the proposed method is to use LLMs to generate a gold answer using the key dimension recognized by Jackknife. Do you go through any human evaluation the generation quality? What is the accuracy of model on THINGS dataset with and without the key dimension as a hint?

### Questions
1. What are models’ brain regions (the second last sentence in the abstract)?
2. How do you generate your ground truth for training (i.e., the “Answer” in Fig. 2)? Is the “Answer” in Fig. 2 the same thing as the assistant message in Fig. 4?
3. What is RSM? The term is not defined in the whole paper. I assume it refers to representational similarity matrix, but you need to explicitly write it down. What is RDM in your Fig. 6?
4. How did you decide the proportion of dataset mixture in Fig. 3? Do you complete any ablation study on the influence of different proportions?
5. Could you please append the 66 dimensions in SPoSE in the manuscript?
6. Why the Centaur uses a few-shot setting while other models use a zero-shot setting? Is this a fair comparison?
7. Can you provide the STD or 95% confidence interval for the results in Fig. 6?

Minor suggestions and typos:
1. In the abstract, the meaning of RSA is not clear. Since it appears only once in the abstract, I suggest changing it directly to representational similarity analysis.
2. The organization of the method section should be improved. The writing in the method section is not clear. You mention Jackknife in the first paragraph (Line 143-150) and then mention it again in the second paragraph (Line 161-164). A better practice may be Stage 1, Stage 2, …
3. A typo: Line 241-242: “RSA analyses assessed” -> “RSA assessed”
4. A typo: Line 259: “THINGS object oncept” - > “THINGS object concept”
5. An additional “~” at line 323 before “7B”
6. Missing spaces: Line 367 “Centaur(3, 5 shots)”; Line 422 “MMMU-Pro.Table”

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a training paradigm that goes past imitating human answers in triplet odd-one-out tasks. The authors first infer a core cognitive dimension presumed to underlie each O1O triplet, then inject that dimension—expressed in natural language—into instruction-tuning data for a vision-language model. They evaluate on held-out O1O prediction, representational similarity agreement with human behavior and fMRI, and broad multimodal benchmarks. The submission also compares against Centaur, a recent cognition-oriented foundation model.

### Strengths
Overall, the paper advances a clear and readily implementable idea: injecting interpretable cognitive dimensions into instruction tuning, which can be applied to off-the-shelf MLLMs (e.g., Qwen2.5-VL) while boosting scientific interpretability. Its evaluation strategy is genuinely multi-axis—combining O1O accuracy, behavior–RSM agreement, whole-brain RSA, and transfer to MMMU/MMMU-Pro—to probe behavioral, neural, and functional performance. Including Centaur as a comparator grounds the work within the emerging “models of cognition” literature and strengthens its timeliness. Moreover, the dimension-injection template appears broadly generalizable beyond object understanding to actions and scenes, naturally dovetailing with wider RSA frameworks for higher-level cognition.

### Weaknesses
1.Dimension inference — validation & stability: Create a human-labeled basis-dimension subset for sampled triplets; report agreement and κ with inferred labels; assess top-1 stability via bootstrap and alternative heuristics (top-k voting, ablation-magnitude thresholds).
2.Controls & fair comparisons: Add ablations to rule out prompt-length/attention effects (correct vs shuffled vs adversarial prompts; same-length null control; compare to plain rationales without dimensions). Ensure baseline parity by unifying zero-/few-shot settings (incl. Centaur), sweeping decoding parameters, and reporting mean ± sd across seeds/prompts.
3.Data hygiene & generalization: Provide explicit image-level and object-set splits and include an object-held-out evaluation where categories never appear in training; document leakage checks.
4.Reporting rigor (neuro + benchmarks): For RSA, report subject-level reliability/noise ceilings, apply FDR/permutation corrections, and run partial RSA controlling low-level and semantic confounds; avoid mapping PFC RSA directly to planning/decision. For MMMU-Pro, state whether vision-only was used and whether OCR prompts or CoT were enabled, and relate outcomes to known sensitivities.

### Questions
1.How accurate and stable is your core-dimension inference? Any human-rated rationale set you can release? 
2.Are all baselines evaluated under matched shot count, prompts, and decoding choices? If not, please add a controlled comparison. 
3.What safeguards ensure no train/test image or object leakage across O1O and RSA experiments? 
4.How sensitive is your method to the choice of base model (e.g., Qwen2.5-VL vs other MLLMs) and to LoRA rank/where adapters are placed? Pointers to implementation details are welcome.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
Multimodal large language models now deliver near-human performance on diverse multimodal reasoning tasks, despite being trained with next-token prediction. A key open question is whether they form human-like visual concepts and exhibit human-like behavior. Using the Odd-One-Out task, their predictions poorly matched human choices. The authors addressed this by extracting core cognitive dimensions and judgment bases from human O1O behavior, then fine-tuning Qwen2.5-VL with these dimensions. This yielded markedly better alignment with human behavior while preserving—and sometimes improving—downstream task performance. Searchlight RSA and cortical projection analyses showed increased activation in regions linked to planning and decision-making, such as the prefrontal cortex, in the fine-tuned model, offering a potential neuroscientific account of the gains and human-like alignment.

### Strengths
-   The paper is well-structured; the method is reasonable and supported by comprehensive evidence and references.
-   Through careful data collection and fine-tuning, this work effectively improves the alignment between MLLMs' cognitive processes and those of humans.
-   The experimental design is sound.

### Weaknesses
-   Based on the results in Table 1, the method seems to offer limited improvements on general benchmarks.
-   The authors stress addressing the critical but underexplored challenge of bridging the gap between human-level performance and human-like cognitive processes in MLLMs. However, this may not actually be a limitation of MLLMs. In other words, why do MLLMs need human-like cognitive processes, given their learning mechanisms are quite different from humans?

### Questions
Please kindly find it in the weaknesses section.

### Soundness
3

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
4

### Summary
The authors propose a method that first extracts core cognitive dimensions from human experimental data, and then uses these dimensions to guide model fine-tuning. They demonstrate that this approach enhances the model’s similarity to humans at both behavioral and neural levels, and provides a new perspective for developing more interpretable and cognitively aligned artificial intelligence systems.

### Strengths
The paper presents a clear research motivation, a well-organized structure, and well-designed figures that effectively illustrate its methodology. The authors employ SPoSE and Jackknife methods to extract latent cognitive dimensions from human Odd-One-Out experimental data, and use these dimensions to fine-tune a multimodal large language model. Additionally, through prompt engineering, the model’s training is aligned with human neural representations. Finally, neural data are used to validate the effectiveness of this cognitive alignment, and the results show that the model exhibits higher consistency with humans at both the behavioral and neural levels.

### Weaknesses
1.	Although the paper is well-structured and logically coherent, its novelty is relatively limited. The proposed Cognitive Dimension Alignment is conceptually very similar to existing Concept Bottleneck Models and subsequent concept-based interpretability studies. Several recent works on concept-based LLMs/VLMs have already achieved model alignment through intermediate semantic layers or interpretable latent dimensions. The paper’s so-called core cognitive dimensions (e.g., metallic, food-related, animal-related) are closer to interpretable semantic concepts rather than true cognitive processes. These dimensions mainly capture shared perceptual or categorical attributes in human semantics but fail to reflect higher-order cognitive abilities such as abstraction, causal reasoning, emotion, or behavioral tendencies. Therefore, defining these semantic features directly as “cognitive dimensions” is theoretically unconvincing. To strengthen the argument of cognitive alignment, the authors should further clarify how these dimensions relate to cognitive dimensions in psychology or provide empirical evidence showing that the model’s decision-making process truly aligns with human cognitive structures.

2.	The evidence for neural alignment remains insufficient. The NSD dataset includes only eight participants, resulting in a limited sample size. Although the RSA results are suggestive, the paper lacks rigorous statistical validation. The authors are encouraged to perform more systematic significance testing and cross-subject consistency analyses to enhance the robustness and credibility of their findings.

3.	The current experiments are conducted solely on the Qwen2.5-VL model. Verifying whether the proposed method generalizes to other architectures (e.g., GPT-4 or Gemini) would greatly strengthen the claims. In fact, based on the presented tasks and figures, preliminary observations suggest that under simple prompt conditions, models like GPT-4 or GPT-5 can already correctly identify cognitive dimensions and make human-consistent choices. Therefore, broader comparative experiments across multiple models and settings are necessary to establish the unique contribution of this work.

4.	Conducting fine-tuning experiments with different data ratios is a common practice and cannot be considered a major innovation. The paper should include a more critical discussion of the limitations and potential biases of the proposed method, rather than emphasizing only the positive results. Additionally, incorporating human behavioral or participant-based analyses could further validate the proposed model’s alignment with human cognition.

### Questions
1.	On the Definition of “Cognitive Dimensions”
The paper defines features such as metallic, food-related, and animal-related as “cognitive dimensions.” However, these dimensions appear to be closer to semantic concepts or interpretable features. Could the authors clarify, from both theoretical and empirical perspectives, how these dimensions differ from traditional concept representations?

2.	Statistical Robustness and Variability
It is recommended that the authors report the variance or confidence intervals of key experimental metrics to assess sensitivity to random initialization, data splits, or sample size. This analysis would help improve the credibility and stability of the conclusions.

3.	Interpretability Validation of Cognitive Dimensions
The authors are encouraged to include more visualizations or ablation studies to demonstrate whether the SPoSE + Jackknife–derived dimensions correspond to human-understandable concepts. Without such validation, the extracted “dimensions” might be statistically driven embeddings rather than genuinely cognitive representations.

### Soundness
2

### Presentation
3

### Contribution
2
