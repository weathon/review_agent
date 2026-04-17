# Can Language Models Compose Skills In-Context?

- Decision: Reject
- Scores: 8, 6, 2, 2

## Abstract
Composing basic skills from simple tasks to accomplish composite tasks is crucial for modern intelligent systems. We investigate the $\mathit{in}$-$\mathit{context}$ $\mathit{composition}$ ability of language models to perform composite tasks that combine basic skills demonstrated in in-context examples. This is more challenging than the standard setting, where skills and their composition can be learned in training or from contextual information. We conduct systematic experiments on various representative open-source language models, utilizing linguistic and logical tasks designed to probe composition abilities. The results reveal that simple task examples can have a surprising $\mathit{negative}$ $\mathit{impact}$ on the performance, because the models generally struggle to recognize and assemble the skills correctly, even with Chain-of-Thought examples. Theoretical analysis further shows that it is crucial to align examples with the corresponding steps in the composition. This inspires a method for the probing tasks, whose improved performance provides positive support for our insights.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper investigates whether LLMs can perform compositional tasks that combine basic skills demonstrated in in-context examples. The work finds that simple in-context examples can (surprisingly) hurt final performance on the compositional task. Further ablations, closer look at attention maps, and theoretical analysis support the following explanation for the observed failure mode: models generally do not recognize the composition and do not align the simple task in-context examples with the corresponding steps of the composition. Motivated by this, the authors propose ExpCOT, which converts simple examples into step-labeled COT with “missing steps”, improving performance across many open source models.

### Strengths
1) The paper clearly communicates the observed phenomena of interest: including simple examples in context can hurt final performance on a compositional task.
2) The paper offers a plausible explanation for the phenomena, supported by theory + experiments.
3) The work also offers a simple mitigation (via ExpCoT) for the observed failure mode.

### Weaknesses
1) The paper does not report performance on the simple tasks themselves under simple-only in-context prompts (or even zero-shot); it only shows how performance on simple tasks changes when compositional in-context examples are included (my apologies if I have overlooked something). Without this clean baseline, it becomes difficult to disentangle true compositional failure from insufficient competence on the underlying atomic skills. By finetuning on those atomic skills, does including the simple skills in-context still hurt performance on compositional tasks?
2) Most experiments focus on T=2 step compositions, but what about larger T? (The theory is for general, larger T).
3) (Minor typo) Paper mispells “Llama” as “Llamma”

### Questions
1) Could the authors comment on relation to the following work: https://arxiv.org/abs/2505.00147 (e.g., the linked work including the simple skills in context can hurt final performance on easy questions by introducing unnecessary information)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies whether LMs can compose skills in-context. Using synthetic linguistic/logical compositions (e.g., opposition ∘ swap), the authors show an empirical finding: adding more simple-task (no composition) often hurts accuracy on composite queries, but adding composite-task helps. Analysis of outputs and attentions suggests models match on operators rather than semantics and often apply only one sub-skill. Naïve CoT doesn’t reliably fix this. A stylized theory shows that without distinguishing simple vs. composite examples, error can remain Ω(Δ); if examples are aligned to composition steps, sample needs can scale only logarithmically in T. Motivated by this, the paper proposes Expanded Chain-of-Thought (ExpCoT), which annotates steps and treats simple examples as CoT with missing steps; this improves accuracy across many open models.

### Strengths
Clear negative result that challenges common intuition: more shots of sub-skills can degrade composition performance; trend is shown across many models/sizes and tasks.

Careful empirical probing: shuffling to reduce order sensitivity, correspondence analysis showing models often perform just one sub-task, and operator-vs-content ablations clarifying what is attended to.

Theoretical insight: formalizes when failing to distinguish data sources leads to lower bounds, and when step-aligned evidence can provably help (log T scaling).

Simple fix method: ExpCoT is simple to implement, consistent gains over vanilla and naïve CoT.

### Weaknesses
Scope limited to synthetic tasks with T=2: It’s unclear whether the phenomenon and ExpCoT gains persist for (i) more realistic multi-hop QA/program induction, (ii) longer compositions (T>2), or (iii) noisy/ambiguous operators. Current tasks may overemphasize symbol/operator cues. (Datasets: 9 compositions from eight base skills.)

Model coverage & claims: Results exclude strongest closed models; conclusions about “LLMs in general” may overreach. The paper notes the resource constraint, but the core claim would be stronger with at least one strong proprietary baseline.

### Questions
Scaling in T: Have you tested ExpCoT with T≥3? Even synthetic three-step compositions would empirically probe the theorem’s log T promise.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper studies the in-context composition ability of large language models (LLMs)—that is, whether models can integrate multiple simple skills demonstrated in the prompt to solve composite tasks requiring multi-step reasoning.
Through controlled experiments on linguistic and logical composition benchmarks (adapted from Xu et al., 2024b) and across 12 open-source LLMs (LLaMA-1/2/3, Mistral, Mixtral, DeepSeek-Distill), the authors find a counterintuitive phenomenon: adding more simple-task examples can hurt performance on composite queries, while additional composite examples help.

The paper attributes this to models’ failure to recognize the compositional structure of tasks—relying on operator matching rather than aligning sub-skills to proper steps. Chain-of-Thought (CoT) prompting does not mitigate this.
Theoretical analysis formalizes conditions under which ignoring compositional alignment increases error, motivating a method called Expanded Chain-of-Thought (ExpCoT), which adds explicit step-wise markers (“Step 1”, “Step 2”, …) to align examples with compositional stages. ExpCoT yields consistent performance improvements across model families.

### Strengths
* The paper provides multi-view analysis—performance trends, output correspondences, attention-similarity visualization, and ablation on operator vs. content cues—all supporting a consistent explanation.

* The theoretical and empirical sections are well integrated.

### Weaknesses
* Missing and Related Work: The paper omits citation and comparison with Skills-in-Context Prompting: Unlocking Compositionality in Large Language Models, which introduced a closely related framework for enabling compositional reasoning via “skills-in-context” demonstrations. That work similarly examined how combining basic skills in a single prompt affects generalization and proposed structured prompting strategies. The present submission should explicitly position itself relative to SKiC, clarifying differences in (a) task design, (b) prompt structure (ExpCoT vs. SKiC).

* The models tested exclude frontier systems (GPT-4/5, Claude, Gemini), leaving open whether newer models overcome these failures.

* The attention analysis (Fig. 5) is largely qualitative. Quantitative probing (e.g., attention entropy, head attribution) would strengthen causal claims. The CoT baseline could include Decomposed Prompting, Skills-in-Context Prompting for a more complete comparison.

### Questions
Could you discuss how ExpCoT compares to Skills-in-Context Prompting in goals, structure, and mechanisms. Is ExpCoT conceptually a structured version of SKiC, or does it introduce new theoretical insights (e.g., compositional alignment proofs)?

Can ExpCoT be tested on more complex or naturalistic tasks (e.g., program synthesis, arithmetic, symbolic reasoning) to demonstrate general utility?

Provide qualitative examples where ExpCoT fails, to better characterize remaining limitations.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the in-context composition ability of LLMs on linguistic and logical tasks. It results reveal that simple task examples can have a surprising negative impact on the performance, because the models generally struggle to recognize and assemble the skills correctly. It further provides theoretical analysis that shows it is crucial to align examples with corresponding steps in the composition.

### Strengths
In-depth study of in-context composition of LLMs are important to understanding the behaviors of LLMs in important functionalities like long chain-of-though reasoning or agent tasks.

### Weaknesses
1. The paper does not cite and compare with a few important related work such as:

[1] J. Chen et al. 2024. Skills-in-Context: Unlocking Compositionality in Large Language Models. In Findings of the Association for Computational Linguistics: EMNLP 2024.

2. The paper ignores to analyze the in-context composition behavior in the recently emerged long-CoT reasoning LLMs. It would be interesting to analyze if the “thinking” process involves the composition of such elementary skills.

3. The paper could be further improved if in-sights can be added on how the pre-training, mid-training, and post-training (SFT & RL) can improve the LLM’s in-context compositional capabilities.

### Questions
None.

### Soundness
3

### Presentation
3

### Contribution
3
