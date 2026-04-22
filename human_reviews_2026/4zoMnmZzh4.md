# VisCoder2: Building Multi-Language Visualization Coding Agents

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 2, 8, 4, 8

## Abstract
Large language models (LLMs) have recently enabled coding agents capable of generating, executing, and revising visualization code. However, existing models often fail in practical workflows due to limited language coverage, unreliable execution, and lack of iterative correction mechanisms. Progress has been constrained by narrow datasets and benchmarks that emphasize single-round generation and single-language tasks. To address these challenges, we introduce three complementary resources for advancing visualization coding agents. **VisCode-Multi-679K** is a large-scale, supervised dataset containing 679K validated and executable visualization samples with multi-turn correction dialogues across 12 programming languages. **VisPlotBench** is a benchmark for systematic evaluation, featuring executable tasks, rendered outputs, and protocols for both initial generation and multi-round self-debug. Finally, we present **VisCoder2**, a family of multi-language visualization models trained on VisCode-Multi-679K. Experiments show that VisCoder2 significantly outperforms strong open-source baselines and approaches the performance of proprietary models like GPT-4.1, with further gains from iterative self-debug, reaching **82.4%** overall execution pass rate at the 32B scale, particularly in symbolic or compiler-dependent languages.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the task of generating visualization code from natural language instructions. It introduces VisCode-Multi-679K, a large multi-language dataset containing executable code-image pairs and multi-turn correction dialogues; VisPlotBench, a benchmark for evaluation; and VisCoder2, a fine-tuned model based on Qwen2.5-Coder. Experiments show that VisCoder2 outperforms open-source baselines and approaches the performance of proprietary models like GPT-4.1, with further gains from iterative self-debug.

### Strengths
- S1: A large multi-language dataset and a benchmark are introduced; both are useful to the community.
- S2: The overall task setting aligns with real-world needs and is worth studying.
- S3: The fine-tuned models VisCoder2 show good performance in terms of execution pass rate.

### Weaknesses
- W1: The dataset is curated by combining existing datasets with filtering and cleaning, so the dataset contribution feels limited.
- W2: The paper evaluates different coding models (mostly scaling within the same model family). Including a wider range of models (e.g., GPT-5 or reasoning-oriented models) would make the comparisons more informative; scaling size alone is not very insightful, as larger models usually perform better.
- W3: More discussion is needed on the evaluation protocol (Section 3.4). Evaluating visual outputs semantically is inherently challenging, as it requires judging whether the generated image aligns with the textual task. Therefore, the key metrics -- Task Score  and Visual Score -- are important. However, both are judged using an LLM-as-a-judge, whose reliability is neither validated nor justified in this benchmark. Can the LLM accurately evaluate these metrics? If yes, to what extent? A more detailed analysis is needed.
- W4: The paper’s results rely heavily on the execution pass rate, which is not highly informative. Execution success only indicates that the code runs without errors, but one could “cheat” by outputting syntactically valid yet semantically incorrect code. This makes the analysis less convincing. More discussion and results based on the Task Score and Visual Score would make the evaluation more meaningful.
- W5: It would be valuable to analyze why other models (e.g., GPT-4.1) fail and to explain how fine-tuning helps reduce specific types of errors. Which errors does fine-tuning effectively mitigate, and which remain challenging? What insights can be drawn for practitioners to improve visualization-coding models?
- W6: The author claims that they are building "Coding Agents". But there is no real autonomous loop, planning, or tool integration -- only self-debugging within a single model. So the use of "agent" feels somehow overclaimed.
- W7: The paper lacks sufficient task statistics for the benchmark dataset (e.g., category distribution, code length distribution, difficulty distribution, etc.). Some statistics are provided, but more detailed and fine-grained statistics would strengthen the paper’s contribution as a benchmark paper.

### Questions
- Line 427: "Execution success is high across most models" --> the execution success of which models?
- Why does Qwen2.5-Coder-32B-Instruct perform worse than the 14B model in Table 2?
- How are the training and testing data split? How do you guarantee the quality and independence of the training and testing sets?
- Lines 396–398: What does "LilyPond shows the largest gains on symbolic grammars" mean? And what does “SVG exposes model-library sensitivity where semantic and perceptual signals diverge" mean?
- Regarding the fine-tuned models: how do they perform on other benchmarks? Does the model lose its capabilities on other types of tasks (e.g., MMMU, HumanEval)?
- How does performance scale with the number of self-debugging rounds? Do more rounds lead to better performance? Is there a saturation point? More analysis would be helpful.

### Soundness
2

### Presentation
3

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
The authors propose VisCoder2, a multi-language visualization coding agent designed to generate, execute, and iteratively correct visualization code. To support this, they introduce VisCode-Multi-679K, a large-scale dataset of executable visualization code and correction dialogues across twelve programming languages, and VisPlotBench, a diverse benchmark for systematic evaluation. Experiments show that VisCoder2 outperforms open-source baselines and matches the reliability of proprietary models like GPT-4.1, especially with self-debugging. These resources advance the development of practical, robust visualization coding agents.

### Strengths
* **Focus on an Important Problem:**

  The paper targets automatic visualization code generation and correction, a task that is increasingly important for data analysis and reporting with LLMs but still lacks robust, multi-language solutions.

* **High-Quality, Multi-Language Dataset:**

  VisCode-Multi-679K covers twelve programming languages, including both popular and symbolic ones. The dataset is large-scale, contains only executable code, and includes multi-turn correction dialogues. These features make it valuable for developing and benchmarking robust visualization coding agents.

* **Effective Application of Iterative Correction:**

    The paper proposes an iterative correction approach that, while not highly novel (being similar to established program repair methods), is effective in practice. The results show significant improvements.

### Weaknesses
* **Lack of Quantitative Dataset Analysis:**

  The paper does not provide sufficient quantitative metrics about the dataset (such as error rates, diversity, redundancy, or semantic alignment). Without these, it is difficult to fully evaluate the dataset’s quality and practical value.

### Questions
Please address my concern in the Quantitative Analysis. Thanks

### Soundness
2

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces VisCoder2, a large-scale dataset and benchmark for multi-domain, multilingual visual code generation. It integrates code, visual plots, and textual prompts across 12 programming languages and multiple application domains (data visualization, UI generation, etc.). The benchmark aims to evaluate both code correctness and visual fidelity in multi-turn coding tasks. The dataset fills an important gap in current research, providing a broader and more diverse benchmark than existing ones like PandasPlotBench. Experiment results demonstrate the effectiveness of the proposed training data.

### Strengths
- Comprehensive dataset coverage: The dataset spans multiple languages and visual domains, filling a real void in existing benchmarks which are typically language- or task-limited.

- Practical relevance: Targets the emerging need for models capable of generating visual outputs (plots, figures, GUIs) from natural-language instructions.

- Readable and well-structured writing: The paper is easy to follow, with clear organization and accessible examples.

### Weaknesses
- Weak multi-turn construction: The “multi-turn” setup seems artificial — simply mixing in existing conversational code data rather than building domain-specific, multimodal dialogues. This undermines the claimed contribution on multi-turn reasoning.

- Incomplete evaluation coverage: Although 12 languages are used for training, only 8 are evaluated. The absence of results for the remaining languages leaves the dataset’s multilingual utility unverified.

- Limited baseline comparison: The model is only evaluated on the proposed benchmark. There’s no comparison against other visual code generation benchmarks (e.g., PandasPlotBench), making it hard to assess generalization.

- Unclear metrics: The paper introduces “visual” and “task” scores but fails to define how visual quality is assessed. For visual generation, perceptual and structural accuracy are as important as code execution success.

- Missing analysis on debug capability: Given the inclusion of multi-turn data, it’s surprising that no explicit debugging evaluation is performed, nor any comparison to instruction-tuned models like Qwen2.5 Coder Instruct. This weakens claims of enhanced reasoning or debugging ability.

### Questions
- How were the “visual” and “task” metrics computed? Are they human-judged, automatic, or hybrid?

- What criteria guided the choice of 8 evaluation languages — were they the largest subsets, or randomly sampled?

- How does the model perform on existing benchmarks like PandasPlotBench? Any transfer results?

- Does the multi-turn data actually improve multimodal debugging? Or just the general debugging ability shared by regular code LLMs?

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
This paper introduces VisCoder2, a framework for building multi-language visualization coding agents. Specifically, the work contain three key components: VisCode-Multi-679K, a large-scale dataset of 679K executable visualization code–image pairs with multi-turn correction dialogues across 12 programming languages; VisPlotBench, a benchmark spanning 8 languages and 13 visualization categories, supporting both single-round and multi-round self-debug evaluation; VisCoder2, a family of open-source multi-language models trained on the above dataset.

### Strengths
1. The author provides a comprehensive dataset construction. They detail a well-structured pipeline including language filtering, runtime validation, and instruction synthesis. Each visualization sample is paired with rendered outputs and multi-turn feedback, ensuring executable, realistic supervision. Compared to the previous method, VisCode-Multi-679K supports 12 languages, which makes its coverage cover most languages.  
2. The author provides extensive experiments; the proposed VisCoder surpasses open-source baselines by 10–15 points in execution pass rate and achieves parity with GPT-4.1 on several languages. The analysis of self-debug gains and error types (syntax vs. runtime vs. semantic) is particularly insightful.
3. The paper is well-organized and easy to follow.

### Weaknesses
1. The detail of self-debugging is missed. The paper demonstrates that iterative correction improves performance, but it does not deeply analyze how models leverage feedback logs. It is better to describe how the self-debug works. Moreover, the author shows that there are persistent failures in semantic and runtime errors. It is also better to provide an analysis of why self-debugging does not work on these cases rather than just saying it doesn't work.
2.  The current evaluation is imbalanced. Execution pass rate dominates the quantitative results, whereas semantic accuracy and visual similarity (Task/Visual Scores) are only partially reported. I understand the space is limited, and the author provides full results in the appendix. However, compared to the execution pass rate, whether the generated results are correct is more important. If possible, it is better to manually check the results and report the correctness for every model. 
3. Both the training corpus (VisCoder) and the evaluation benchmark (VisPlotBench) are constructed by the authors and share the same generation–execution–debug protocol. The paper does not evaluate zero-shot transfer to unseen visualization libraries, natural user prompts, or noisy real-world code, leaving generalization ability uncertain, which limits the usage of the model. It is better to provide natural user test results.

### Questions
1. The current LLMs like GPT-5 can also support image input. Is it possible to extend the current dataset? e.g., provide the image input and request code output. 
2.  It is better to provide the process of self-debug and better evaluation results.

### Soundness
3

### Presentation
1

### Contribution
3
