# Reasoning Scaffolding: Distilling the Flow of Thought from LLMs

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
The prevailing approach to distilling reasoning from Large Language Models (LLMs)—behavioral cloning from textual rationales—is fundamentally limited. It teaches Small Language Models (SLMs) to mimic surface-level patterns rather than the underlying algorithmic structure of thought, resulting in a critical lack of logical robustness. We argue that instead of cloning text, distillation should transfer this algorithmic structure directly. We introduce Reasoning Scaffolding, a framework that reframes reasoning as a structured generation process. Our method first abstracts the teacher's thought process into a sequence of discrete, interpretable semantic signals (e.g., Contrast, Addition) that act as a scaffold. The student model is then trained via a multi-task objective to both (1) predict the next semantic signal, anticipating the reasoning flow, and (2) generate the corresponding step, conditioned on that signal. This multi-task scheme acts as a powerful regularizer, compelling the student to internalize the computational patterns of coherent reasoning. On a suite of challenging reasoning benchmarks, our method significantly outperforms state-of-the-art distillation in both accuracy and logical consistency, providing a path towards creating smaller models that are genuine reasoners, not just fluent mimics.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Reasoning Scaffolding, a framework for distilling reasoning ability from large language models (LLMs) into smaller models (SLMs). Instead of directly imitating the teacher's Chain-of-Thought (CoT) text, the method abstracts reasoning traces into semantic signals (e.g., Contrast, Elaboration, Conclusion) that serve as scaffolds for step-by-step reasoning. The student model is trained via a dual-branch architecture, predicting both the next semantic signal and the corresponding reasoning step, to encourage internalization of the reasoning structure. Experiments on multiple benchmarks (StrategyQA, CommonsenseQA, TruthfulQA, GSM8K, MATH-500) show improvements over standard CoT and Long-Thinking distillation baselines.

### Strengths
1. **Interesting idea**: The paper explores a creative perspective on reasoning distillation by introducing semantic scaffolding as a middle-level representation between textual rationales and abstract reasoning steps.

2. **Comprehensive experiments**: Evaluation spans multiple reasoning benchmarks and model scales (0.5B, 7B, 14B), providing a broad empirical basis.

3. **Clear empirical comparisons**: The ablation studies (e.g., signal quality and token analysis) are informative and show that structured supervision can improve reasoning stability.

4. **Good motivation**: Addressing the brittleness of current CoT distillation is an important and timely research direction.

### Weaknesses
1. **Experimental irregularities (Table 1).**
The main results table is somewhat confusing. For example, the Qwen2.5-0.5B results under "Long-Thinking Distill" are missing, while the Qwen2.5-7B model performs worse than its SFT counterpart under this setting (except on MATH-500). These inconsistencies raise questions about the fairness and reproducibility of the comparison. It would be helpful to clarify whether the fine-tuning setup (data size, training epochs, loss weighting) is kept consistent across all baselines. Moreover, teacher models such as DeepSeek-R1 could serve as stronger baselines.

2. **Model- and task-specific signal design.**
The categorization of reasoning signals (Section 3.1) appears heuristic and data-specific. The seven signal types (e.g., Addition, Contrast, Conclusion) seem derived from particular verbal patterns in GSM8K-style reasoning traces. It is unclear whether this taxonomy would generalize to other domains (or teacher models), such as scientific reasoning, logic puzzles, or multi-modal contexts. The framework's dependence on these fixed signal categories limits its general applicability.

3. **Ambiguity in contribution novelty.**
While the idea of structured distillation is valuable, much of the implementation (keyword matching, LLM labeling, multi-task fine-tuning) builds directly on existing CoT or discourse-signal techniques. The conceptual advancement beyond structured rationale distillation remains incremental without stronger theoretical or analytical insight.

4. **Writing and clarity.**
The overall writing is understandable but sometimes verbose and repetitive. Some sections (e.g., 3.1–3.3) contain long procedural details that could be condensed and improved. Minor language issues also appear throughout, which detract slightly from readability.

### Questions
See weakness

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Reasoning Scaffold: a distillation method for improving the reasoning capabilities of small language models (SLMs). The method works by decomposing reasoning traces from a large teacher model into annotated reasoning steps, categorized into reasoning types using keyword-based and LLM annotation. The SLM is then tasked to learn the category of the current reasoning step and generates reasoning tokens conditioned on this information. This method intends to better structure the reasoning of the SLM. Experiments on reasoning benchmarks demonstrate the effectiveness of the proposed method on reasoning benchmarks for the Qwen model family, with models ranging from 0.5 to 14B parameters. The method, however, achieves its performance at a higher inference cost than standard distillation as it requires more generated tokens.

### Strengths
1. The proposed work is simple yet effective for improving distillation in SLMs. It can serve the research community by providing an easy scaffold to yield better small reasoners.
2. The experiments show improved performance on reasoning tasks and accurately support the claims made by the paper.
3. The presented analysis interestingly shows that additional structuring signals, even weak or random can help organize the model's reasoning thoughts and improve reasoning capabilities.

### Weaknesses
1. The proposed work, while useful, does not present an original method or novel findings as similar studies already exist [1].
2. The proposed structure categorization is very high-level, potentially making the category prediction task trivial as argumentation usually follow the same steps and thus, reducing the information in the signal.
3. The method relies on handcrafted keyword matches (complemented by an LLM), which can be brittle in out-of-domain tasks, particularly as the LLM is only used for assigning a category to a piece of text but not for the division into steps.
4. The method balances two training objectives with the same weight to both losses. This can lead to training instability if they do not have the same magnitude or variance.
5. Experiments are only performed on models from the Qwen family and it is unclear if the findings can transfer to other models or if they are artifacts specific to Qwen. Similarly, only one teacher model 5DeepSeek-R1) is used.
6. The proposed method yields longer reasoning chains than the best baseline (Thinking Distill), which could account for the performance gap.


Minor comment:
1. Results from the tables are a bit hard to read. Highlighting the best results in bold would be helpful to the reader.
2. It is not clear from Figure 4 how the proposed method improves the reasoning signal as the examples follow the same reasoning structures.



[1] Li, D., Cao, S., Griggs, T., Liu, S., Mo, X., Tang, E., ... & Stoica, I. (2025). LLMs Can Easily Learn to Reason from Demonstrations Structure, not content, is what matters!. arXiv preprint arXiv:2502.07374.

### Questions
1. Have you investigated if additional, more precise and informative, categories could improve the guiding signal and improve the generation of the reasoning traces?
2. Have you investigated the keyword-LLM agreement for splitting the traces into categories? can the keyword create additional unneeded steps or miss a transition to a new step?
3. Have you investigated the evolution of the two training losses and compared their magnitude and variance? Does including a hyperparameter controlling the weighting factor of one of the losses improves the learning?
4. As the "Conclusion and Summary" denote both intermediate and final outputs, how is the final output differentiated from the intermediate ones?
5. Have you performed experiments with other model families? Both as teacher and student?

### Soundness
3

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
4

### Summary
This paper introduces Reasoning Scaffolding, a distillation framework that extracts the structural flow of reasoning from LLMs into discrete semantic signals (e.g., Contrast, Addition, Elaboration) to train SLMs. Unlike traditional behavioral cloning from Chain-of-Thought rationales, which mimics surface-level text, this method uses a multi-task objective where the SLM learns to predict the next semantic signal (anticipating reasoning flow) and generate the corresponding step conditioned on it. This acts as a regularizer for logical coherence. The approach includes data preparation via keyword matching and LLM validation, a dual-branch model architecture, and inference with signal-guided generation and pruning for efficiency. Evaluations on benchmarks like StrategyQA, CommonsenseQA, TruthfulQA, GSM8K, and MATH show improved accuracy and robustness over baselines like CoT SFT and Long-Thinking distillation, using Qwen models of varying sizes.

### Strengths
* Tackles a fundamental flaw in reasoning distillation by shifting focus from text imitation to transferring algorithmic structure, which is a timely and innovative contribution to creating more robust SLMs.

* The multi-task training with signal prediction as a regularizer is technically sound and provides interpretability, potentially advancing mechanistic understanding of reasoning in models.

* Comprehensive experiments demonstrate substantial gains (e.g., ~14% average over originals, ~8% over CoT baselines), with notable benefits for smaller models, and the framework shows scalability across model sizes and tasks.

* Includes practical optimizations like confidence-based termination and pruning of reasoning traces for token efficiency, making it applicable for real-world deployment.

### Weaknesses
* The semantic signal extraction relies heavily on an external LLM (e.g., GPT-4) for validation and labeling, which could propagate biases or inconsistencies from the labeler, and the choice of exactly 7 categories seems somewhat arbitrary without broader justification or sensitivity analysis.

* While results are strong on the selected benchmarks, the paper lacks evaluation on out-of-distribution tasks or diverse reasoning domains (e.g., code generation, planning), limiting claims of general robustness; comparisons are mostly to CoT variants rather than other structured distillation methods like modular architectures or rationale decomposition.

* Inference depends on a tunable threshold τ for signal confidence, but the paper provides limited ablation on its impact across datasets, and the pruning strategy might discard useful intermediate details in complex problems.

* The dataset construction uses zero-shot prompting from a single LRM (Deepseek-R1), which may not capture diverse reasoning styles; details on dataset size, diversity, or quality control are sparse in the provided sections.

### Questions
* How was the set of 7 semantic signals determined, and what happens if you expand or reduce the categories—does performance change significantly?

* Did you observe any biases in the semantic signals assigned by GPT-4, such as favoring certain transitions based on the teacher's style?

* How does the method perform on tasks beyond QA and math, like creative writing or multi-agent reasoning, where reasoning flows might be less linear?

* What is the computational overhead of the dual-branch training compared to standard CoT distillation, and how sensitive is inference efficiency to the pruning strategy?

### Soundness
3

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
3

### Summary
This work proposed novel distillation approach for distilling LLM reasoning thinking data into small models. The approach can be summarized as follows

1. Identify thinking (reasoning) words in the reasoning chain of thought, like (Additionally, So, Then ...)
2. Group these worlds into 7 types 
3. Split the DeepSeek generated reasoning chain using identified keywords.
4. During distillation, let the model predict both reasoning type and reasoning tokens for each hidden states
5. During inference, an adaptive strategy is adopted,  if the confidence computed using reasoning step type predictor, the reasoning is terminated.
6. Another experiment based on only using "Conclusion and Summary"  steps in reasoning chain is used for distillation, and the result looks good. 

The experimental results show that this approach out-perform direct distillation and  SFT using CoT directly. 

Ablation study is performed to demonstrate the effectiveness of predicted reasoning type, and correct reasoning type can guide reasoning effectively.

### Strengths
1. The proposed approach for distillation is effective and reached high performance compare with baseline
2. The designed experiments clearly explained the motivation of of the proposed approach. 
3. The ablation study is comprehensive

### Weaknesses
1. The models tested are only from Qwen 2.5 families. However, previous works [1], have questioned about the behaviour of Qwen 2.5 on math reasoning.  It could be more convincing to adopt other models on this approach to check the effectiveness and performance. 
2. The proposed approach is expensive and rely on advanced model. It's hard to scale up. 
3. The loss described in the paper not matches the given code.  In paper, equation 2 on L221 is a classification loss while  regression loss is used in your code `custom_qwen_model.py` , L200. Also a lot of evaluation python scripts is not available. 
4. Figures are not intuitive and self-explaining. In figure 2, the detailed example text can be removed and maybe replaced with abstract icon to emphasize more on the approach. For figure 3,  it could be better to use some toy examples (not long and real, but just conceptual) to illustrate the pruning process. 
5. Only reasoning chains from DeepSeek is tested, and it remains unknown how reasoning chain of other style (like from gpt-oss) perform using this approach. 
6.  The work can be more intuitive if decoding algorithm and be explained in pseudo code. And I can not find adaptive decoding codes in codebases.

### Questions
1. Is the code the latest version ?
2. Please refer to weakness and resolve my concern.

### Soundness
3

### Presentation
2

### Contribution
3
