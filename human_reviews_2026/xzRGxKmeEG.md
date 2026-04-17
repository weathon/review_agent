# Revisiting Multi-Agent Debate as Test-Time Scaling: When Does Multi-Agent Help?

- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
The remarkable growth in large language model (LLM) capabilities has spurred exploration into multi-agent systems, with debate frameworks emerging as a promising avenue for enhanced problem-solving. These multi-agent debate (MAD) approaches, where agents collaboratively present, critique, and refine arguments, potentially offer improved reasoning, robustness, and diverse perspectives over monolithic models. Despite prior studies leveraging MAD, a systematic understanding of its effectiveness compared to single-agent methods, particularly under varying conditions, remains elusive. This paper seeks to fill this gap by conceptualizing MAD as a test-time computational scaling technique, distinguished by collaborative refinement and diverse exploration capabilities. We conduct a comprehensive empirical investigation comparing MAD with strong self-agent test-time scaling baselines on solution-finding tasks (e.g., mathematical reasoning) and response-judging tasks (e.g., safety). Our study systematically examines the influence of task type, task difficulty, and agent diversity on MAD’s performance. Our key findings reveal that, for solution-finding tasks, MAD offers only limited advantages over self-agent scaling—even with diverse agents—although its effectiveness increases slightly as problem difficulty rises. Conversely, for response-judging tasks, especially on safety-reasoning tasks, MAD’s collaborative refinement generally strengthens defense and judgment as more agents are added. Moreover, incorporating diverse agent configurations yields a more pronounced reduction in attack success, indicating that agent diversity is crucial for response-judging tasks, unlike in solution-finding tasks. We believe our findings provide critical guidance for the future development of more effective and strategically deployed MAD systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper reframes multi-agent debate (MAD) as a test-time scaling strategy. The authors evaluate homogeneous and heterogeneous MAD on mathematical reasoning and safety. For math, MAD rarely beats strong self-consistency, with slight gains only on the hardest problems; for safety, adding agents and diversity generally lowers attack success via collaborative refinement, with benefits most pronounced under heterogeneous configurations.

### Strengths
- Clear conceptual positioning of MAD as test-time scaling with two distinct features: collaborative refinement and diverse exploration.


- Systematic, controlled comparisons against strong self-agent baselines across tasks, difficulties, model sizes, and diversity settings, with matched generation budgets.
- Provide actionable insights for math and safety tasks.

### Weaknesses
The primary concern is the generalizability of the conclusions, both within and across domains. Although the paper reports task-specific findings, there is no clear trend across tasks, and the divergent conclusions for math vs. safety make it difficult to infer whether MAD will help in other settings. Moreover, evaluating only a vanilla MAD configuration does not preclude gains from alternative debate designs. Prior work, e.g., Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate, has reported improvements on math, and other evaluations (e.g., Should We Be Going MAD? A Look at Multi-Agent Debate Strategies for LLMs) are cited but not compared in depth. Overall, the lack of thorough cross-study comparison and limited protocol coverage raises questions about both external validity and novelty.


- Primary focus on math reasoning and safety; it’s unclear how well findings generalize to other domains (planning, coding, long-form generation) or to alternative debate protocols beyond the vanilla setup.
- Safety evaluation relies on a closed-source judge (gpt-4o-mini), which can introduce evaluation bias; robustness to the choice of judge is not established.
- The framing is largely confined to multi-agent debate; there’s limited comparison against other multi-agent collaboration paradigms, making it unclear whether conclusions transfer beyond debate. Is it limited only to a multi-agent debate framework? Can it be generated for other Multi-agent collaboration systems?

### Questions
- Even within math and safety, how robust are the conclusions across subdomains and difficulty regimes?
- Where do MAD’s gains primarily come from—multi-agent diversity, the debate/refinement phase, or simply increased sampling? Can you decompose the contribution of each?




- Do you have guidance or a heuristic for deciding when to use MAD on a new task?
-  Why use majority voting over two sampled generations, and how are disagreements/ties resolved?
- How should practitioners choose the number of agents? Any compute-aware heuristic or scaling law for picking an effective agent count?

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
This manuscript reframes Multi‑Agent Debate (MAD) as a test‑time scaling mechanism that combines diverse parallel exploration with sequential collaborative refinement. Under matched generation budgets, the study compares MAD to strong self‑agent baselines: Self‑Consistency (SC) and Self‑Refinement (SR), across mathematical reasoning and safety tasks. The main findings are: (i) in math, SC typically outperforms or matches MAD, with only slight gains for MAD on very difficult problems where collaborative refinement can validate a rare correct trajectory; (ii) in safety, SR can amplify attack success rates (ASR) for smaller models, whereas MAD with more agents and heterogeneous configurations tends to reduce ASR and sometimes matches/surpasses the safest single agent as rounds increase.

### Strengths
1. The paper conceptualizes Multi-Agent Debate (MAD) as a test-time scaling strategy, clearly analyzing its effectiveness along two distinct dimensions: parallel scaling to enhance generative diversity and sequential refinement for iterative critique and revision. This approach effectively elucidates MAD’s relative advantages over single-agent baselines.

2. The authors systematically evaluate and compare multi-agent and single-agent approaches across different tasks, incorporating diverse agent configurations through variations in models and prompts. This comprehensive analysis provides nuanced insights into how heterogeneity influences multi-agent outcomes.

3. The employment of detailed diagnostics (e.g., Best-of-Correctness [BoC], Best-on-Follow [BoF]) and the thorough analysis of answer transitions from initial to final states further clarify the internal dynamics underpinning the effectiveness of multi-agent strategies.

### Weaknesses
1.The study categorizes tasks into two broad regimes—those requiring extensive exploration of large search spaces and those necessitating consensus among limited options. However, for each category, the authors select only one specific type of task (mathematics and safety, respectively), which limits the representativeness and generalizability of the conclusions.

2.Prior research ([1]) appears to have explored similar comparative analyses between MAD and Self-Consistency (SC), including investigations into heterogeneity-driven MAD approaches, yet no citation or discussion of this work is provided in the manuscript.

3.The reliance solely on GPT-4o-mini for evaluating the Attack Success Rate (ASR) in safety tasks raises concerns regarding evaluator credibility. Given that GPT-4o-mini itself may face limitations in handling complex safety evaluations, incorporating human validation or cross-validation with more robust evaluators would significantly enhance confidence in these results.

4. While alignment across methods is established based on the number of generated responses, MAD methods incur additional overhead in context tokens due to cross-agent interactions and extended prompts. This aspect is not quantified, limiting the fairness and practical interpretability of the computational cost comparisons.

5. Figure 7 inaccurately labels metrics as "accuracy" rather than ASR, creating confusion and reducing clarity. Consistency across the manuscript's terminology and visual presentations should be ensured.

[1] Zhang, Hangfan, et al. "Stop Overvaluing Multi-Agent Debate—We Must Rethink Evaluation and Embrace Model Heterogeneity." arXiv preprint arXiv:2502.08788 (2025).

### Questions
1.Could the authors validate their proposed two-regime task classification with additional datasets (e.g., code generation, knowledge-intensive QA, or tool-assisted planning) to better substantiate the generalizability of their conclusions?

2.Can the authors provide evidence regarding the reliability and accuracy of GPT-4o-mini as the evaluator for safety-related tasks? For instance, could human evaluators or comparisons with stronger models be included, and consistency or inter-rater agreement explicitly reported?

3.Could the authors provide a detailed comparison of token usage and computational resources (e.g., prompt tokens, context tokens per round and agent, total token counts) to facilitate a fairer and more practical assessment of MAD relative to single-agent baselines?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper reframes multi-agent debate (MAD) as a form of test-time scaling that mixes parallel exploration and sequential collaborative refinement, and asks when (and for what tasks) MAD actually helps over strong single-agent baselines such as self-consistency (SC) and self-refinement (SR). Through controlled comparisons on mathematical reasoning (MATH500, AIME 2024/2025) and safety evaluations (Anthropic Harmful, MultiJail), the authors find that homogeneous MAD rarely beats parallel SC for math; gains appear only on the hardest problems where collaborative refinement can surface and verify a rare correct trajectory. Adding diversity via personas/prompting yields small additional math gains, while mixing different model families tends to regress toward a harmonic-mean–like performance and may destabilize consensus, limiting benefits versus simply sampling more from the strongest model. In contrast, for safety tasks, collaborative refinement with more (and larger) agents consistently reduces attack success rates, and heterogeneous configurations (different prompts or model families) are especially effective. The study’s takeaways are practical: prefer SC for math unless problems are extremely difficult (where MAD can verify a rare correct path), and use heterogeneous MAD for safety to counter bias propagation and improve robustness.

### Strengths
* This paper focuses on how and when to choose appropriate design decision for MAD systems, which is a critical challenge in the application of MAD.
* The paper is well written with highlighted takeaways, offering practical guidance to MAD design.

### Weaknesses
* The idea of agent diversity, and the observation that MAD sometimes fails single agent methods have been well acknowledged in previous research [1][2][3]
* While with comprehensive evaluation and practical design recommendations, this paper did not offer a unified solution to this problem, limiting the technical contribution.
* The takeaways are bounded by task types, difficulty, which is sensitive to the selection of base models, and may not be generalizable to stronger models or stronger reasoning models, especialy given that MADs are naturally sensitive to prompts. It is not promising that these takeaways can benefit further MAD system design.


[1] Reconcile: Round-table conference improves reasoning via consensus among diverse llms
[2] Stop Overvaluing Multi-Agent Debate -- We Must Rethink Evaluation and Embrace Model Heterogeneity
[3] Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper systematically evaluates Multi-Agent Debate (MAD) against strong baselines (Self-Consistency and Self-Refinement) on mathematical reasoning and safety tasks. It finds MAD's effectiveness is highly conditional: it offers limited benefits in math, except for very difficult problems when using weaker models , and it increases vulnerability in safety tasks unless heterogeneous agents are employed.

### Strengths
This paper is empirically thorough, systematically testing MAD performance across a range of tasks, difficulty levels, and model scales, demonstrating that MAD is not a universally superior method but is beneficial only under specific, well-defined conditions.

### Weaknesses
The primary weakness is a severe lack of novelty. The paper's core findings are nearly identical to those in a recent, highly related publication If multi-agent debate is the answer, what is the question? (arXiv:2502.08788). Both perform a systematic evaluation of MAD and arrive at the same two conclusions:

1.MAD is Overvalued: Both papers find that MAD generally fails to outperform strong single-agent baselines like Self-Consistency (SC).

2.Heterogeneity is the Solution: Both papers identify model heterogeneity (using diverse agents) as the key solution.

The prior work already established heterogeneity as the 'universal antidote'. This paper's primary conclusions are therefore a re-discovery, which significantly limits its contribution. The authors must address this major overlap.

### Questions
Please respond to the above weakness.

### Soundness
3

### Presentation
3

### Contribution
1
