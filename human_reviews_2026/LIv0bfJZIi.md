# On Code-Induced Reasoning in LLMs

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
Code data has been shown to enhance the reasoning capabilities of large language models (LLMs), but it remains unclear which aspects of code are most responsible. We
investigate this question with a systematic, data-centric framework. We construct parallel instruction datasets across ten programming languages and introduce controlled
perturbations that selectively disrupt structural and semantic properties of code. We
then fine-tune LLMs from five model families and eight scales on each variant and
evaluate their performance on natural language, math, and code tasks. Across 3,331 experiments, our results show that LLMs are more vulnerable to structural perturbations
than semantic ones, particularly on math and code tasks. Appropriate abstractions like
pseudocode and flowcharts can be as effective as code, while encoding the same information with fewer tokens without adhering to original syntax can often retain or even
improve performance. Notably, even corrupted code with misleading signals remains
competitive when surface-level regularities persist. Finally, syntactic styles also shape
task-specific gains, with Python favoring natural language reasoning and lower-level
languages such as Java and Rust favoring math. Through our systematic framework,
we provide a fine-grained analysis of how different aspects of code influence reasoning
and inform the design of training data for enhancing LLM reasoning capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a systematic investigation into how different properties of code data influence the reasoning capabilities of large language models (LLMs) during fine-tuning. The authors construct parallel instruction datasets in natural language and across ten programming languages, then apply controlled perturbations targeting structural (e.g., whitespace removal, pseudocode conversion) and semantic (e.g., variable renaming, comment modifications) properties of code. Through lots of fine-tuning experiments spanning varied model families at different scales, they evaluate performance on natural language, math, and code tasks. Key findings reveal that structural perturbations degrade performance more severely than semantic ones, particularly on math and code tasks, while appropriate abstractions like pseudocode and flowcharts can match or exceed the effectiveness of original source code.

### Strengths
+ The paper presents an interesting study to understand the impact of code data for LLM reasoning
+ The experiments are conducted in a comprehensive way that include multiple different models with varied settings

### Weaknesses
**Flawed Experimental Design - Post-Training Phase Limits Causal Claims About Code Impact on Reasoning**

A fundamental limitation of this work is that all experiments are conducted during the supervised fine-tuning phase on base models already extensively pretrained on massive corpora containing both natural language and code. This design weaknes the paper's central claims about which code properties enhance reasoning, as the observed performance differences may primarily reflect how well different formats activate existing pretrained knowledge rather than demonstrating the intrinsic learning value of code properties. For instance, when fine-tuning on perturbed code (e.g., whitespace removed, variables renamed), poor performance could simply indicate distributional mismatch with pretraining data rather than proving that structural properties are fundamentally important for reasoning. The surprising finding that "corrupted code with misleading signals remains competitive" may merely show that fine-tuning cannot override deeply ingrained pretraining patterns rather than demonstrating genuine robustness to surface regularities. This limitation calls into question whether the paper's guidance for "design of training data" will generalize broadly.

**Correctness in the code data is not guaranteed, and the structural perturbations are overly simplified**

Another weakness of this work is the lack of validation for code correctness in the training data and the oversimplified nature of perturbations, which significantly limits the validity and depth of the findings. The authors generate all code responses using GPT-4o-mini, a relatively weak model, and seems apply no execution-based validation to ensure correctness (or I missed this part?). Without test cases to reject incorrect solutions during data generation, the 120K training examples may contain substantial amounts of buggy or non-functional code, fundamentally compromising the study's premise. If models learn from a majority of flawed examples, observed performance patterns cannot reliably indicate which code properties enhance reasoning but rather which properties are less affected by learning from errors. 

Moreover, the absence of test cases severely constrains the types of perturbations possible, forcing reliance on superficial transformations like whitespace removal and variable renaming that preserve surface syntax but cannot guarantee semantic equivalence. The study cannot perform meaningful semantically-equivalent transformations such as algorithmic refactoring, loop-to-recursion conversion, or function extraction that would provide richer insights into whether models learn genuine computational logic versus syntactic templates.

**Inadequate Analysis of Why Pure Code Fine-tuning Outperforms Mixed Training in RQ1**

The paper lacks an in-depth analysis of why pure code fine-tuning (code-ft) consistently outperforms or matches mixed training (mixed-ft) across most tasks in RQ1, despite mixed training intuitively seeming more aligned with evaluation on diverse task types. Several plausible hypotheses remain unexplored, including task distribution mismatch, data quality difference among data resources, or the observations are mostly for post-training? Without ablation studies and analysis to distinguish between these possibilities, readers cannot assess whether the findings reflect fundamental properties of code for reasoning or other factors.

### Questions
- Can the authors provide analysis or at least some conceptual arguments to disentangle pretraining effects from intrinsic code properties? Or the authors would discuss the findings specifically for post-training rather than general LLM training (including all training phases)?

- What steps were taken to ensure code correctness in the training data to avoid misleading signals, and can you discuss the error rates and potential impacts in your training data?

- Why does pure code fine-tuning mostly outperform mixed training in RQ1?

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
4

### Summary
The paper probes which properties of code drive LLM reasoning via 3,331 finetuning experiments across five model families, eight scales, ten languages, and systematic perturbations. It finds structural features are pivotal, with their disruption reliably hurting math and code performance. Abstractions and efficient encodings can rival raw code, and even corrupted code stays competitive when surface regularities persist. Lower level languages yield larger math gains, suggesting practical principles for training data design beyond executable programs.

### Strengths
1. The paper demonstrates, from multiple complementary perspectives, that code data confers genuine benefits to LLM reasoning, and the evidence supporting the central claim that structural properties are decisive is clear, consistent, and persuasive.
2. The experimental suite is broad in scope and methodologically solid, spanning diverse model families, sizes, languages, and perturbations, and it cleanly validates how targeted changes to structure or semantics translate into measurable differences in downstream performance.
3. Several results crystallize actionable insights for the community, pointing toward data design practices that emphasize structural signals and efficient abstractions, and suggesting concrete priorities for future training and evaluation.

### Weaknesses
1. It would be valuable to extend the study to larger model sizes to observe how the perturbations interact with scaling.
2. It is unclear whether the conclusions still hold in a reasoning model setting.

### Questions
Please refer to weakness.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
- The paper studies why code improves reasoning in LLMs through a large-scale, systematic framework.

- Constructs parallel instruction datasets in 10 programming languages and applies controlled perturbations (rule-based and generative) that selectively distort syntax, semantics, or structure.

- Finetunes 5 model families across 8 scales (3,331 experiments) and evaluates effects on natural language, math, and code reasoning tasks.

- Finds that structural features of code drive reasoning improvements more than semantics; pseudocode and flowcharts can substitute for code; lower-level languages aid math reasoning.

### Strengths
- Clever perturbation design isolating structure vs. semantics; highlights causal signals in code-induced reasoning.

- Impressive experimental scale and rigor - 3k+ controlled finetunes across models, languages, and perturbation types.

### Weaknesses
- Heavy reliance on GPT-4o-mini for generating and evaluating code; possible bias leakage or contamination

- Finetuning limited to ≤8B models; it's unclear if findings hold for frontier scales where reasoning mechanisms differ.

- Focus on finetuning setups - the model performance could depend on the pretrain LLMs.

- It would be great to have some ablations on dataset size or diversity. This would help disentangle whether effects persist with smaller or noisier samples.

### Questions
- How robust are results to using a different evaluator model (e.g., Claude, Gemini) for judging code quality?

- How would findings change if applied during pretraining rather than instruction finetuning?

- Could smaller-scale runs (e.g., 20k data) reproduce the same trends?

### Soundness
3

### Presentation
3

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
In this work, the authors set out to validate the claim that coding data enhances reasoning abilities of LLMs, while also deepening the analysis by **ablating how different aspects of coding data correlate with downstream evaluations across three categories of tasks**: Natural language and general knowledge, math, and code generation and understanding.

In particular, the authors collate prompts from various open-source code instruction-tuning benchmarks, and ask GPT-4o to generate completions in 10 popular programming languages, resulting in a final dataset of 120k samples (12k per each language). They then provide 11 perturbed versions of the original dataset - 5 variants the result of programmatic perturbations (e.g. removing whitespace, renaming variables, removing comments etc.), and 6 variants of LLM rewrites (e.g. comment enhancement, conversion to pseudocode, or to natural language). 

Those datasets - along with a 120k general instruction following examples from OpenHermes 2.5 that constitute a non-code baseline - are then grouped into logical subcategories, and used to finetune models of various model families (e.g. Qwen3, Llama3, Gemma3), ranging from sizes of 0.6B to 8B.

The main findings:
- Incorporating (unperturbed) **code data generally improves performance** across all tasks, especially math and coding (Figure 1).
- **The perturbed datasets reduce model performance**, as compared to the unperturbed dataset.
- **Semantic perturbations reduce performance less than structural** (Figure 3).
- Perturbations which **highlight algorithmic structure** while removing superficial syntax, like pseudocode or flowcharts, often **match or surpass unperturbed code baseline** (Figure 4).
- Models remain **robust to corrupted or low-interpretability code** (Figure 6).

### Strengths
- Investigating the of usefulness of different aspects of code data is a **timely and relevant, with clear practical utility.**

- The work is **original in both framing and experimental design.** Disentangling different aspects of code data (structural vs semantic, human interpretable vs not) in the context of the data’s generalization to reasoning is a novel advance over prior, more-coarse grained (“code helps reasoning”) empirical investigations.

- The **experiments are extensive**, comprising 3,331 fine-tuning runs across ten programming languages, eleven perturbation types, five model families, and multiple model scales ranging from 0.6B to 8B parameters. The appendix presents these results exhaustively, providing detailed plots and breakdowns that reinforce the empirical trends highlighted in the main text.

- The paper **provides comprehensive replication details** in its appendix, including exact hyperparameters, dataset processing steps, and all prompt templates used for generation and evaluation.

### Weaknesses
- The **presentation of the perturbed datasets lacks detail** - the paper would be enhanced by a) providing overall statistics like the number of tokens, and b) providing qualitative examples of each dataset.

- All **experiments are restricted to small, up to 8B parameter models**, leaving doubt whether the observations would generalize to frontier-scale models. If possible, the authors could corroborate their main findings by a few experiments on larger models. Barring that, scaling laws could be established across the 1B -> 8B scale.

- Both the baseline and generative-perturbation code data are synthetically generated with GPT-4o (except for the prompts). However, **the paper provides no assessment of the quality and correctness of this generated data**. This could be mitigated by including a small-scale human evaluation—for example, manually annotating around 30 samples from each perturbation set. Alternatively, if resources permit, the authors could regenerate the data using a different frontier model and re-run key analyses to verify that the observed trends are consistent across data sources.

- The paper presents numerous ablations, but results are shown separately for each model–size combination, with only a single representative plot per research question in the main text. This makes it difficult to see aggregate patterns, as readers must cross-reference many individual plots in the appendix. **Aggregating results across models or scales would make the reported trends clearer and more convincing.**

- **Some claims in the main text are not fully substantiated by the experimental results.** For instance, when discussing performance across programming languages, the authors state that “on math tasks, high-scripting languages consistently underperform relative to intermediate and low-system ones,” referencing Figure 7 as evidence. However, in Figure 7 (top), high-scripting languages perform on par with intermediate ones and even surpass low-system languages on math tasks. In Figure 7 (bottom), the top five individual languages include three high-level ones, with all five well within each other’s error bars. Overall, the data shown (at least in the main body of the text) does not support the claim.

### Questions
See "Weaknesses" section

### Soundness
3

### Presentation
2

### Contribution
3
