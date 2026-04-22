# DelvePO: Direction-Guided Self-Evolving Framework for Flexible Prompt Optimization

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Prompt Optimization has emerged as a crucial approach due to 
its capabilities in steering Large Language Models to solve 
various tasks. However, current works mainly rely on the random 
rewriting ability of LLMs, and the optimization process generally 
focus on specific influencing factors, which makes it easy to fall into local optimum. 
Besides, the performance of the optimized prompt is often unstable,
which limits its transferability in different tasks. 
To address the above challenges, we propose $\textbf{DelvePO}$ 
($\textbf{D}$irection-Guid$\textbf{e}$d Se$\textbf{l}$f-E$\textbf{v}$olving
Framework for Fl$\textbf{e}$xible $\textbf{P}$rompt $\textbf{O}$ptimization), 
a task-agnostic framework to optimize prompts 
in self-evolve manner. In our framework, we decouple 
prompts into different components that can be used to explore 
the impact that different factors may have on various tasks. 
On this basis, we introduce working memory, through which 
LLMs can alleviate the deficiencies caused by their own uncertainties 
and further obtain key insights to guide the generation of 
new prompts. Extensive experiments conducted on different 
tasks covering various domains for both open- and 
closed-source LLMs, including DeepSeek-R1-Distill-Llama-8B, Qwen2.5-7B-Instruct and GPT-4o-mini. Experimental results show that 
DelvePO consistently outperforms previous SOTA methods 
under identical experimental settings, demonstrating 
its effectiveness and transferability across different tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces DelvePO (Direction-Guided Self-Evolving Framework for Flexible Prompt Optimization). DelvePO decomposes prompts into explicit components to systematically probe how different factors affect performance, and augments the process with a working memory that helps the LLM mitigate its own uncertainty, extract key insights, and use those insights to guide the next round of prompt generation. Across diverse tasks and domains, and on both open- and closed-source models (e.g., DeepSeek-R1-Distill-Llama-8B, Qwen2.5-7B-Instruct, GPT-4o-mini), DelvePO consistently outperforms prior state-of-the-art under identical settings, demonstrating improved effectiveness and transferability.

### Strengths
1. Decouples prompts into components and uses direction-guided self-evolution, reducing reliance on random rewrites and avoiding local optima while offering clearer interpretability of what matters in a prompt.

2. Introduces a working memory to capture insights across iterations, mitigating LLM uncertainty and making prompt updates more principled and consistent.

3. Demonstrates consistent improvements across diverse tasks and both open/closed models.

### Weaknesses
1. Novelty vs structured prompt methods is under-specified. The paper claims a key contribution in decomposing prompts into components, but similar ideas exist (e.g., Task Facet Learning: A Structured Approach To Prompt Optimization). It will be better to add into related works.

2. Missing closer, up-to-date baselines. Need to add some recent and more related prompt-optimization methods as baseline (Task Facet Learning is one example).

3. The transferability claim need stronger evidence. Currently, it is unclear how the transferability is demonstrated in the experiments, which is claimed in the abstract.

### Questions
May be better to show some cases with optimized prompt.

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
3

### Summary
The paper proposes DelvePO, a direction-guided and memory-augmented prompt optimization framework. It first decomposes prompts into functional components and then evolves them using two working memories: Component Memory, which records beneficial component edits, and Prompt Memory, which stores high-performing prompt combinations. This design reduces the randomness of LLM-based prompt mutation, prevents the loss of important components, and improves transfer across tasks and models. Experiments on multiple datasets and LLMs show consistent gains over human prompts and prior prompt optimization methods.

### Strengths
The paper introduces a clear component-level prompt representation together with two explicit working memories (Component Memory and Prompt Memory), which turns otherwise highly stochastic LLM-based prompt mutation into a more controllable and reusable optimization process.

### Weaknesses
1.	The paper proposes a task-agnostic framework, but the initial component pool is manually collected and constructed from a wide range of related literature (line 116). This raises a question about the motivation of the method: does DelvePO truly make no strong task-specific assumptions and generalize to different tasks because of the framework design itself, or is the observed generality mainly due to the fact that a very comprehensive task component pool has been pre-collected and constructed?
2.	The ablation in Table 3 provides only qualitative evidence that the proposed direction signal, instantiated via both the component-level memory and the prompt-level memory, indeed contributes to the final performance: removing either memory degrades the results, and removing both effectively collapses the framework back to a largely stochastic evolution regime. This supports the authors’ claim that “direction-guided” optimization is beneficial. However, earlier in the paper, the authors explicitly state that direction-guided evolution “can reduce the time required for evolutionary operations” (line 87), i.e., that guidance does not merely improve the final score but also makes the optimization more efficient. To substantiate this stronger claim, the paper should report performance vs.–budget curves (e.g., performance as a function of iteration/time, number of LLM calls, or total input tokens) under a fixed budget, and compare them against a purely stochastic/mutation-only variant. Otherwise, the observed gains can still be explained by “doing more (or better-informed) LLM calls” rather than by genuinely improving the sample efficiency of the evolutionary process.
3.	The components used in DelvePO are extracted and constructed from related literature and can be further generated by an LLM. However, there is no experimental evidence showing whether the method remains effective when the target task domain deviates substantially from the initial component set. For domain-specific tasks, you may refer to the test sets used in Table 2 of PROMPTAGENT [1].


[1] Wang, Xinyuan, et al. "Promptagent: Strategic planning with language models enables expert-level prompt optimization." arXiv preprint arXiv:2310.16427 (2023).

### Questions
1. The method adopts a multi-stage pipeline whose complexity is relatively high, and the usage cost will be noticeably higher for closed-source LLMs. It is recommended to provide a clearer breakdown of the computational and token cost, and to specify in the method section which stages can be parallelized and which stages can be cached in advance.

2. The authors significantly increase token consumption by writing a large amount of memory back into the prompt, but the experimental section does not systematically report the relationship between “memory context length vs. performance vs. cost.” As a result, it is unclear whether the performance gains are achieved primarily by spending a large number of tokens. 

Typos:
1. The double quotation marks around “role” in line 56 are not formatted correctly, and the same issue appears with other quotation marks in the paper.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Previous works about prompt optimization focuses on limited specific factors, making local optimum inevitable. This work proposes DelvePO with memory to recognize deficiencies and thus guide new generation of prompt. DelvePO consistently outperforms previous SOTA methods on classical NLP tasks, QA, etc. across several models, showing the effectiveness of DelvePO. Specifically, DelvePO decouples the prompt into several factors, task-evolution, solution-evolution, memory-evolution, to guide the evolutionary process. The integration of multiple modules could also improve the interpretability of the optimization process, making it easier to interact with the  system.

### Strengths
* DelvePO achieves better performance compared with previous baselines and ablation study shows the effectiveness of each component in the method.
* This paper is well-written and the motivation is clear, meaningful.

### Weaknesses
* Datasets and tasks selected are classical, relatively easy tasks for LLMs and these are not difficult for current strong LLMs anymore. I'm curious about the performance of DelvePO on more challenging and difficult tasks in LLM-era, like GSM8k, BBH, more reasoning tasks and so on. 
* This paper introduces memory and in essence, memory appears as concluding insights from last-generation prompts, which is a little far-fetched. OPRO[1] gives previous good-performing prompts and worse prompts to guide generation, APO[2] gives bad cases to guide optimization, all of which can be explained as "memory". I disagree with the statement "DelvePO is the first one to introduce memory", instead, it seems that DelvePO first explains such optimization as "memory".
* Though the motivation is meaningful, I don't think DelvePO solves the problem pointed out, i.e. local optimum. Jumping out of local optimum has not been proved quantitively, I'm not convinced of the claim.
## references 
[1] Yang, Chengrun, et al. "Large language models as optimizers." The Twelfth International Conference on Learning Representations. 2023.
[2] Pryzant, Reid, et al. "Automatic Prompt Optimization with" Gradient Descent" and Beam Search." The 2023 Conference on Empirical Methods in Natural Language Processing.

### Questions
* Could the authors provide ad detailed example of $M_{components}$
* In table 6, the cost of DelvePO is relatively high. The experimental proformances compared with baselines under same costs should be investigated further.
* See the weakness part.

### Soundness
2

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
3

### Summary
This paper proposes DelvePO, a framework for automatic prompt optimization that decouples prompts into modular components (analogous to genetic loci and alleles) and uses a working memory mechanism to guide evolutionary operations. The framework consists of three main modules: Task-Evolution (determining which components to evolve), Solution-Evolution (performing mutation/crossover operations), and Memory-Evolution (updating component and prompt memories). Experiments are conducted on 11 datasets across 3 LLMs (DeepSeek-R1-Distill-Llama-8B, Qwen2.5-7B-Instruct, GPT-4o-mini), showing improvements over baselines including APE, PromptBreeder, and EvoPrompt.

### Strengths
- Decomposing prompts into interpretable components is valuable for understanding what makes prompts effective
- Testing across multiple LLMs and domains demonstrates effort to validate generalizability
- Detailed appendices with all prompts used enhance transparency
- The working memory design that stores both component-level and prompt-level information is sensible

### Weaknesses
- The core contributions are incremental improvements over existing evolutionary prompt optimization methods
- Lack of significance testing and inconsistent use of random seeds weakens confidence in reported improvements
- The framework requires extensive prompt engineering (Sub-tasks I-II, Sub-solutions I-II, multiple scenarios) that may limit adoption
- Practical Limitations:

1. Higher computational costs than baselines
2. Requires predefined component types that may not transfer across domains
3. The "case study" reveals manual steps that contradict automation claims

### Questions
- Can you provide statistical significance tests (e.g., paired t-tests) comparing DelvePO against baselines across random seeds?
- Table 4 shows concerning instability with different numbers of component values. How do you recommend practitioners set this hyperparameter?
- The case study (Appendix I) involves manual interaction with DeepSeek Chat. How does this square with claims of a fully automated framework?
- Can you provide ablations showing the contribution of individual components to overall performance? Which components are most important?
- How does performance scale with the number of component types? What happens if users define 10+ components?
- The discrete vs. continuous prompt memory distinction is unclear. Can you provide empirical comparison of these two approaches?
- Why were different subsets of datasets used for different LLMs? This makes it difficult to draw conclusions about model-specific behaviors.
- How sensitive is the method to the quality of initial component value generation (Figure 4)?
- Can you provide analysis of failure modes? When does DelvePO underperform simpler baselines?
- The framework requires many carefully crafted meta-prompts (Figures 8-14). How much prompt engineering effort went into developing these, and how transferable are they?

### Soundness
2

### Presentation
2

### Contribution
2
