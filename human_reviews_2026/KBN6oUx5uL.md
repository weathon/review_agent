# SR-Scientist: Scientific Equation Discovery With Agentic AI

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 8

## Abstract
Recently, Large Language Models (LLMs) have been applied to scientific equation discovery, leveraging their embedded scientific knowledge for hypothesis generation. However, current methods typically confine LLMs to the role of an equation proposer within search algorithms like genetic programming. In this paper, we present SR-Scientist, a framework that elevates the LLM from a simple equation proposer to an autonomous AI scientist that writes code to analyze data, implements the equation as code, submits it for evaluation, and optimizes the equation based on experimental feedback. Specifically, we wrap the code interpreter into a set of tools for data analysis and equation evaluation. The agent is instructed to optimize the equation by utilizing these tools over a long horizon with minimal human-defined pipelines. Empirical results show that SR-Scientist outperforms baseline methods by an absolute margin of 6\% to 35\% on datasets covering four science disciplines. Additionally, we demonstrate our method's robustness to noise, the generalization of the discovered equations to out-of-domain data, and their symbolic accuracy. Furthermore, we develop an end-to-end reinforcement learning framework to enhance the agent's capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces SR-Scientist, an agentic framework for scientific equation discovery through data-analysis and equation-evaluation tool calling. The authors also develop an RL training pipeline that improves agent performance. The method outperforms baselines on LLM-SRBench [1].

[1] Shojaee et al. "LLM-SRBench: A new benchmark for scientific equation discovery with large language models." (2025).

### Strengths
- The method is simple and clear. The task of scientific equation discovery is important, and the problem formulation is concise.
- SR-Scientist consistently outperforms state-of-the-art baselines on all models and across all tested scientific fields. The selected baselines make sense.
- The paper successfully demonstrates a path forward towards self improvement of the agent through an end-to-end RL pipeline.

### Weaknesses
- The work appears to be heavily inspired by SR-LLM [2]. Several key components, such as the equation-evaluator tool or the experience buffer, closely resemble those introduced in [2]. While the reframing of the problem as an agentic framework is probably useful, it is not a novel idea and has been widely used in many recent works.
- Results are evaluated solely on synthetic data (LSR-Synth, a synthetic subset of [1]). This limits the assessment of the method for practical applications. Adding experiments on real-world datasets or use cases where data often includes noise, outliers, etc., could provide a stronger evidence. Also, designing new tools to handle such challenges would strengthen the paper.
- The current design of the experience buffer ablation (Table 3) only confirms that accumulating knowledge over long horizons is better than constantly starting from scratch. Instead, this ablation should isolate the contribution of the suggested fetch-top-K buffer (for example, against a baseline that randomly selects equations).

[2] Shojaee et al. "LLM-SR: Scientific equation discovery via programming with large language models." (2024).

### Questions
1. Have the authors considered the use of broader tools for data analysis, such data visualization or graphical analysis? Do you think that expanding the framework's tool capabilities could be useful?
2. Could the authors please elaborate on the mechanisms that allow the framework to obtain a long-horizon behavior?
3. Do the authors plan to evaluate their method on real-world datasets?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes SR-Scientist, an agentic large-language-model (LLM) framework for symbolic regression. It combines two tools (a data analyzer and an equation evaluator) in a long-horizon reasoning loop with an experience buffer and a reinforcement-learning (RL) fine-tuning stage based on GRPO. Experiments on the LSR-Synth benchmark across four scientific domains show solid empirical gains over existing SR and LLM baselines, supported by ablations and robustness analyses.

### Strengths
+ The paper is well motivated and clearly framed, presenting the idea of turning LLMs into autonomous “scientists” that iteratively refine hypotheses through data analysis and reasoning.
+ The evaluation covers multiple domains, five different LLM backbones, and includes analyses for noise robustness, out-of-domain generalization, and symbolic accuracy.
+ The modular two-tool framework with memory buffering is versatile and adaptable to other discovery tasks beyond symbolic regression.
+ The empirical results are consistently strong across settings and supported by detailed ablation studies.
+ The framework provides a practical system contribution that can serve as a foundation for other research works in scientific AI agents.

### Weaknesses
- The work mainly integrates existing components such as tool use, GRPO-based RL, and memory buffering, without introducing new learning principles. The proposed mechanisms are largely heuristic and not deeply analyzed.
- The RL fine-tuning stage performs worse than inference-only variants and is limited to a single backbone. Its “one-iteration” setup contradicts the long-horizon reasoning concept and lacks convergence or stability analysis.
- The comparison with baselines is not compute-normalized. SR-Scientist executes up to 25 turns × 40 iterations, while non-LLM baselines are capped at 100k equations, leading to possible resource imbalance.
- The metric choice emphasizes Accτ with a 5% trimming rule and omits standard regression metrics like RMSE or MAE. Results are averaged over only three runs, which weakens statistical confidence.
- Symbolic accuracy remains around 7%–8%, much lower than numeric accuracy (>60%), suggesting improved curve fitting but not genuine symbolic recovery.
- The paper lacks qualitative examples illustrating how the agent refines equations through multi-step reasoning, making the “scientific inspiration” claim speculative.
- Several parameters and mechanisms, such as K in the experience buffer, the MAPE goal threshold, and the exclusive use of BFGS for optimization, are presented without sensitivity analysis or justification.
- The dataset evaluation omits LSR-Transform due to contamination concerns, but does not demonstrate that LSR-Synth is free of memorization. There are no experiments on real-world scientific data.
- Figures are dense, tables inconsistent, and citations incomplete, which affects readability.
- In summary, the lack of algorithmic novelty, limited RL justification, resource imbalance, and weak interpretability analysis reduce the overall impact. The current version of the paper stands as a well-executed system but needs stronger methodological contributions, fairer comparisons, and deeper reasoning analysis to meet the bar for ICLR.

### Questions
- Why does RL underperform? Was training unstable or reward scaling flawed?
- How is data contamination avoided in LSR-Synth?
- Try to do an apples-to-apples comparison. Would SR-Scientist still outperform if all methods were given the same compute budget or wall-clock time?
- Why restrict the framework to two tools instead of adding others, like dimensional checks or simplification modules?
- What causes the observed drop in performance beyond 25 turns?
- Can you provide qualitative examples of reasoning trajectories that refine equations step by step?

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
5

### Summary
This paper presents a novel framework, SR-Scientist, for LLM-based scientific equation discovery. Unlike prior approaches that primarily treat LLMs only as equation proposers within search algorithms, SR-Scientist extends their role to act as agentic AI scientists capable of performing complementary pre- and post-processing tasks such as data analysis, equation implementation, and submission for evaluation. The framework integrates a set of wrapped tools (e.g., data analysis and equation evaluation) that are used in an agentic and long-horizon manner to autonomously navigate the full equation discovery process.
Empirical results demonstrate that the proposed approach outperforms existing baselines on standard benchmarks, exhibits improved robustness to noise, and shows better generalization across diverse tasks. The paper also includes experiments with a smaller Qwen-30B model augmented with end-to-end RL fine-tuning which shows their results could even be strengthen with the fine-tuning and adaptation of LLM backbone.

### Strengths
- The paper is well-written and clearly motivated, making it an engaging read.
- The experiments are comprehensive on recent benchmarks and with thorough analysis. 
- The reported results demonstrate substantial performance improvements over state-of-the-art baselines.

### Weaknesses
- The proposed data analysis tool is well-motivated and conceptually sound. However, I am unclear about the necessity of using an LLM as an equation evaluator, given that evaluation in this task is typically data-driven and follows a consistent procedure. Introducing an agentic LLM into the evaluation loop could potentially introduce unnecessary variability or LLM generation errors. I would appreciate clarification on why this component is needed and what advantages it offers over standard predefined evaluation methods for generated hypotheses. Additionally, Table 3 seem to not include an ablation for this tool (T2), making its specific role and impact questioning. 

- I think additional qualitative examples would help clarify what the agentic steps contribute beyond the baseline LLM-SR approach. Showing intermediate output such as snippets of generated code or key decision points the agent follows during the discovery process would help to understand how these steps influence the final discovered equations.

- For the results reported in Table 2 and Figure 3, it is unclear whether they are evaluated on all LLM-SRBench tasks or only a specific subset or category. It would be helpful to include this information explicitly in the captions.

### Questions
Included in the weaknesses section

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces SR-SCIENTIST, a framework for applying large language models (LLMs) to the scientific equation discovery (SED) process. The framework integrates evolutionary search with an LLM-as-heuristic component but offers greater flexibility than prior methods by allowing LLMs to perform a wider range of actions in each iteration. The paper demonstrates that enabling LLMs to interact directly with observational data (via tool use) and to engage in multi-turn interaction improves performance on SED tasks. Additionally, it explores finetuning LLMs with GRPO for this domain.

### Strengths
1. The experiments present strong results across multiple metrics and settings, consistently favoring the proposed approach.
2. Comprehensive ablation studies highlight the effects of two key components - tool calls and experience buffer - as well as the influence of interaction length and the distribution of tool usage.
3. The paper provides a detailed discussion of the reinforcement learning (RL) process, including the methodology for constructing synthetic training data and a comparative analysis of different reward function designs.
4. The writing is clear, well-structured, and easy to follow.

### Weaknesses
1. The contribution in terms of novelty is somewhat incremental. The framework closely resembles prior approaches such as LLM-SR, particularly in its use of an experience buffer, evaluation module, and parameter optimization function. The main components - tool use and multi-turn interaction - can be seen as adaptations of existing agentic LLM paradigms (e.g., GPT-OSS, Qwen3-Coder) to the symbolic regression task, rather than a fundamentally new agentic framework. The RL fine-tuning pipeline largely follows established practices, including synthetic data construction and the use of the GRPO loss.

### Questions
1. Figure 6 shows that Equation Evaluation dominates tool usage. Does this suggest that the agents primarily engage in a “guess-and-check” process rather than deriving insights from observation data through the Data Analyzer tool?
2. The analysis indicates that increasing the maximum number of turns beyond 25 results in stagnating or slightly declining performance. Is this due to context length limitations, or does it reflect the agent’s inability to explore new strategies (e.g., requiring periodic resetting or reinitialization)?
3. Could the authors include representative failure cases? A qualitative analysis of typical failure modes - such as difficulties with specific mathematical structures, high noise levels, or certain data domains - would provide valuable insights into the framework’s limitations.

### Soundness
3

### Presentation
3

### Contribution
3
