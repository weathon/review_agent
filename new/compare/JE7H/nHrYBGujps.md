---
job_id: 54117d27-d143-46cf-8f92-54a9b3aa5810
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: nHrYBGujps.pdf
paper: BIRD-INTERACT: Re-Imagining Text-to-SQL Evaluation via Lens of Dynamic Interactions
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length  
Pass ✅.

## Topic Compatibility  
Pass ✅.  
The work proposes a new benchmark and evaluation framework for interactive text-to-SQL with LLMs, squarely in ICLR’s scope on representation learning, agentic LLMs, RL-style evaluation, and datasets/benchmarks.

## Minimum Quality  
Pass ✅.  
The paper is complete and well structured, with Abstract, Introduction, Problem Definition/Methodology (Sections 2–4), Experiments and Results (Sections 5–6 + key tables/figures), Related Work, Future Work, and Conclusion. The claims are supported by substantial experiments; there is no obvious fatal methodological error or misuse of data.

## Prompt Injection and Hidden Manipulation Detection  
Pass ✅.  
I see no signs of prompt injection, hidden instructions to reviewers, or similar manipulation within the provided content.

---

# Expected Review Outcome:

## Summary

The paper introduces **BIRD-INTERACT**, a dynamic benchmark for evaluating LLMs on interactive text-to-SQL. Each task couples a realistic database environment (including schema, hierarchical knowledge base, and metadata) with a **function-driven user simulator**, and is decomposed into two sub-tasks that require ambiguity resolution and handling of follow-up queries under a budget-constrained interaction protocol. Two settings are defined: **c-Interact**, a conversational protocol with limited clarification turns and one debugging attempt per sub-task, and **a-Interact**, an agentic REACT-style setting with a costed action space over tools (DB, KB, simulator). Experiments with 7 strong LLMs show low success rates (≤ ~29% SR and ≤ ~25% normalized reward on FULL), highlight differences between conversational vs agentic capabilities, and analyze interaction behaviors (e.g., test-time scaling with more turns, action distributions, memory grafting, and simulator robustness).

## Strengths

1. **Addresses a very real gap in text-to-SQL evaluation.**  
   Most prior work evaluates LLMs in *single-turn*, SELECT-only settings; here the benchmark explicitly targets **multi-turn, ambiguity-ridden, CRUD-capable** interactions with state-changing SQL. Sections 1–3 articulate clearly why static dialogues and read-only queries miss the core challenges of production database assistants, and Table 4 (Page 22) concisely situates BIRD-INTERACT against SQL generation, ambiguity-handling, static conversation, and other interactive benchmarks. This fills a meaningful gap for the community.

2. **Carefully designed interactive environment and simulator.**  
   The **two-stage function-driven simulator** (Section 3.3, Figure 3(c) on Page 6, and detailed prompts in Appendix N/R) is a strong technical contribution. The separation into (i) a parser that maps questions to symbolic actions AMB/LOC/UNA and (ii) a generator that produces constrained natural-language replies mitigates ground-truth leakage and “helpful but cheating” behaviors seen in simple LLM-as-user setups. Section 6 and Figure 6 (Page 9) clearly demonstrate substantial gains in UNA handling accuracy on USERSIM-GUARD compared to baseline simulators, which is critical for trustworthiness of any dynamic benchmark.

3. **Well-specified formalization, metrics, and reward structure.**  
   Section 2 gives a clean formalization of interactive text-to-SQL as a process: Equation (1) defines the alternating simulator/system updates  
   \[
   u_i^t = \mathcal{U}_\gamma(h_i^{t-1}, q_i, \mathcal{E}),\quad
   s_i^t = \mathcal{S}_\theta(h_i^{t-1}, u_i^t, \mathcal{E}),\quad
   h_i^t = h_i^{t-1} \oplus \langle u_i^t, s_i^t \rangle,
   \]  
   which matches the actual implementation used in both c-Interact and a-Interact. The Success Rate metric (Equation (2), Appendix F) and the piecewise reward definitions (Appendix F.2) give a transparent mapping from sub-task success and debugging behavior to a scalar normalized reward. This is important for anyone who wants to build learning-based agents on top of the benchmark.

4. **Rich task design: ambiguity, follow-ups, and environment dynamics are all operationalized.**  
   The conversion from LIVESQLBENCH to BIRD-INTERACT is thoroughly thought through. Ambiguity injection is not hand‑wavy: Section 3.2 and Appendix H provide a detailed **taxonomy** over intent-level and implementation-level ambiguities (Tables 6 and 7, Pages 27–28), extend these to knowledge and environmental ambiguity (H.2), and combine them into *ambiguity chains* requiring multi-hop clarification (H.3, Figure 2). Follow-up sub-tasks are systematically categorized (Table 8, Page 29) and often state-dependent, which forces models to reason over changing DB state rather than treat each query as independent.

5. **Dual evaluation settings and budget-aware design are well motivated and concretely realized.**  
   The paper does not just provide one protocol; it explicitly distinguishes **c-Interact** (scripted dialogue with limited clarifications and a single debug opportunity per sub-task) and **a-Interact** (REACT-style agent with 9 discrete actions and action-dependent costs; Table 9, Page 30). The budget formulas  
   \(\tau_{\text{clar}} = m_{\text{amb}} + \lambda_{\text{pat}}\) for c-Interact and  
   \(B = B_{\text{base}} + 2m_{\text{amb}} + 2\lambda_{\text{pat}}\) for a-Interact  
   are intuitive and give users clear levers (user patience) to stress-test models under interaction constraints. The costed action design in Table 9 is particularly thoughtful, approximating real-world latency and human-in-the-loop costs.

6. **Strong and multi-faceted empirical analysis.**  
   The main results in **Table 2** (Page 7) and **Table 10** (Page 35) clearly show that the benchmark is challenging: even GPT‑5 / Gemini‑2.5‑Pro solve a small fraction of tasks (≤ 29.17% SR on full, ≤ 37.33% SR on lite) and capture at most ~25–33% of normalized reward, with substantial gaps between BI and DM tasks. The paper goes beyond a leaderboard:  
   - **Interaction Test-Time Scaling** in Figure 4 (Page 8) shows how success rates change with user patience, and for some models (e.g., Claude‑3.7‑Sonnet) success increases monotonically as interaction turns increase, supporting the ITS law proposed.  
   - **Memory grafting** in Figure 5 (Page 8) is an insightful diagnostic: GPT‑5’s success rate rises when fed clarification histories from better communicators like Qwen‑3‑Coder or O3‑Mini, isolating poor clarification behavior rather than poor SQL generation as the bottleneck.  
   - **Action distribution analysis** in Figures 10–13 (Pages 31–33) gives a nuanced view of how different models allocate budget between ask/submit vs environment probes and how that correlates with success.

7. **Dataset quality and simulator robustness are substantiated, not asserted.**  
   Table 1 (Page 5) and Table 11 (Page 39) along with the human evaluation in Appendix Q indicate the authors invested serious effort in ensuring annotation quality (annotator qualification Section C, 93% inter‑annotator agreement, 97.3% human acceptance rate). USERSIM-GUARD and the LLM-as-judge protocol (Figures 6 and 26) provide quantitative evidence that the function-driven simulator is significantly safer and more aligned with human behaviors (Table 3, Page 9) than standard LLM-as-user approaches.

8. **Figures are generally informative and support the claims.**  
   - **Figure 1** (Page 2) gives a concrete end-to-end interaction example and makes it very clear how ambiguity resolution, debugging, and follow-ups are orchestrated in the environment; this helps readers understand the non-trivial evaluation loop beyond the formal notation.  
   - **Figure 3** (Page 6) visually summarizes the two evaluation settings and reward flows, which is invaluable for understanding how c-Interact and a-Interact differ operationally.  
   - **Figures 10–13** show per-model action distributions and how they evolve over turns, directly underpinning the claim that models over-use expensive trial-and-error (execute/submit) and under-explore cheap tools.  

Overall, this is a carefully engineered benchmark with non-trivial methodological contributions around simulator design, interaction protocols, and analysis.

## Weaknesses

1. **Benchmark complexity and practical barrier to entry.**  
   While the design is rich, it is also quite heavy-weight. Using the benchmark in practice appears to require (i) Dockerized PostgreSQL environments, (ii) the function-driven simulator with multiple prompts, AST parsing, and function-calling infrastructure (Appendix N/R), and (iii) integration of a non-trivial action space for a-Interact (Table 9). The main paper mentions that each task uses a fresh PostgreSQL 14 Docker instance (Section 5, Page 7) and that costs are non-trivial (Table 2 shows up to \$0.60 per task for some models). For research groups without significant engineering resources or API budgets, reproducing or extending the experiments may be challenging. The paper would benefit from more explicit discussion in the main text (beyond Appendix I) of how heavy the infrastructure is, and perhaps a “minimal usage” mode that abstracts away some complexity.

2. **Reward definition and normalization are somewhat ad hoc and not fully consistent with what is reported.**  
   The normalized reward is defined as  
   \(R = \frac{\sum_i r_i}{N} \times 100\) (Appendix F.2), then described in the abstract and Section 2 as “normalized to [0,1]”. In practice, Table 2 and Table 10 report reward values like 25.52, 32.93, etc., which correspond to percentages rather than [0,1] normalization. This is a minor inconsistency but confusing, especially since the abstract claims “normalized to [0,1] for analyzing system behaviors”. Moreover, the choice of weights (0.7/0.3 across sub-tasks and 0.5/0.2 penalties for debugging) is reasonable but largely heuristic; there is no sensitivity analysis showing whether rankings or main conclusions are robust to these choices. Since the benchmark is intended as a long-lived standard, a slightly more principled or justified reward design, or at least a plot showing robustness to alternate weightings, would strengthen the evaluation.

3. **Limited diversity of system baselines and lack of methods optimized for this setting.**  
   The experiments focus entirely on **general-purpose LLMs** acting as text-to-SQL systems in either a simple chat protocol (c-Interact) or a generic REACT-like scaffold (a-Interact). There is no baseline that represents *specially designed* interactive text-to-SQL methods or RL-trained agents, even though Section 7 cites agent-based work like MAC-SQL (Wang et al., 2025) or MINT-style agents. As a result, it is difficult to tell whether the benchmark primarily diagnoses current LLMs or whether there exist methods that already perform significantly better when properly tuned for interaction. At minimum, a simple heuristic agent tuned to this environment (e.g., always ask about all annotated ambiguities; always retrieve schema/KB before executing) or a finetuned policy over the action space would help indicate that the environment is learnable and not just punishing.

4. **The dependency on the simulator’s parser accuracy is under-analyzed.**  
   The two-stage simulator (Section 3.3) relies critically on the LLM parser correctly mapping questions to AMB/LOC/UNA. However, the main paper does not quantify parser accuracy on *real interaction logs* from Section 5; USERSIM-GUARD (Section 6) evaluates stimuli that are constructed from templates but does not show how often, during actual tasks, the parser misclassifies an honest clarification as UNA or maps it to the wrong ambiguity, which could unfairly penalize systems. For example, in Figure 3 and the c-Interact description, systems must choose which ambiguity to clarify within a limited turn budget; if a reasonable question is labeled UNA by the parser, the environment becomes unsolvable despite the system acting sensibly. A breakdown of simulator errors in the real interaction trajectories, or at least a sanity-check on a subsample, would be valuable.

5. **Less attention to statistical robustness and variance of evaluations.**  
   All experiments are **single runs with temperature 0** (Section I.3), which makes outputs deterministic but hides any variability across prompts or random seeds. The reported success rates in Table 2/Table 10 are thus point estimates without error bars. Given the relatively small absolute SR numbers (e.g., many are in the 10–30% range) and task counts of 300–600, some confidence intervals or at least a discussion of variance would improve the credibility of model comparisons. For instance, differences of 1–2 points in SR or reward may not be significant, but the paper still interprets them qualitatively (e.g., preference for certain interaction modes).

6. **Some formal aspects are under-specified, especially for the a-Interact setting.**  
   While Equation (1) provides a clean high-level interaction model, it does not explicitly encode the **action space** or the fact that, in a-Interact, actions like `execute`, `get_schema`, or `ask` with their costs and observations occur between user utterances. Mathematically, the agent’s policy in a-Interact is closer to a POMDP over \(\mathcal{E}\) and the simulator, yet Section 2 does not formalize this distinction from c-Interact. Additionally, the budget formulas \(B = B_{\text{base}} + 2 m_{\text{amb}} + 2\lambda_{\text{pat}}\) and \(\tau_{\text{clar}} = m_{\text{amb}} + \lambda_{\text{pat}}\) are introduced but there is no explicit constraint in the mathematical definition tying the maximum number of turns \(|h_i^t|\) to these budgets. Clarifying this linkage formally (e.g., a constraint \(\sum_t \text{cost}(a_t) \le B\)) would make the problem definition more rigorous and easier to extend to learning-based agents.

7. **Positioning with respect to other multi-turn text-to-SQL and interactive text-to-DB works could be deeper.**  
   The related work section mentions CoSQL, SParC, CHASE, etc., but omits or only briefly touches on some recent multi-turn text-to-SQL / interactive DB efforts. In particular, the benchmark is conceptually very close to other recent multi-turn text-to-SQL frameworks that emphasize iterative refinement and interleaved feedback. A slightly more detailed comparison of **task structure and simulator design** with such work would better justify the claim that Bird-Interact is “among the most open, challenging, and long-horizon” (Section 3.4). See also missing related work section below.

8. **Figures packed with content can be hard to parse in the main paper.**  
   Some key figures, while conceptually valuable, are visually dense. For example, **Figure 3** combines (a) environment components, (b) action space, (c) user simulator architecture, and (d) both reward flows for c- and a-Interact into a single panel. Likewise, **Figures 10–13** pack multiple systems and action types into small subplots, making it difficult to read exact proportions without zooming. The message is still understandable, but simplifying or splitting these into multiple panels (or providing an enlarged version in the main text rather than only in the appendix) would improve accessibility.

9. **Single dataset family and DBMS restrict external validity a bit.**  
   Although the databases are diverse (Table 5) and much more complex than Spider/WikiSQL, they stem from a single underlying source (LiveSQLBench) and a single DBMS (PostgreSQL). The paper argues convincingly for PostgreSQL as a realistic and open option, but does not investigate whether model performance or interaction patterns change substantially in other environments (e.g., MySQL, BigQuery, Snowflake). To be fair, this is probably beyond scope, but worth explicitly acknowledging as a limitation in the main paper and not only the appendix.

Overall, these weaknesses are mostly about completeness and rigor of analysis rather than fundamental design flaws. The core benchmark design appears sound and impactful.

## Potentially Missing Related Work

1. **Xiong et al., “Interactive-T2S: Multi-Turn Interactions for Text-to-SQL with Large Language Models”, 2024.**  
   This work introduces a system focused on multi-turn text-to-SQL with explicit LLM-driven interaction for clarification and refinement, which is conceptually close to BIRD-INTERACT’s c-Interact setting. It should be discussed in Section 7 (Multi-turn Text-to-SQL) and compared in terms of (i) whether they support dynamic user trajectories vs static scripts, and (ii) the scope of SQL operations (CRUD vs SELECT-only). If feasible, it would also be a natural baseline or at least a point of conceptual comparison.

2. **Hua et al., “SQL-Trail: Multi-Turn Reinforcement Learning with Interleaved Feedback for Text-to-SQL”, 2026.**  
   SQL-Trail proposes an RL framework for multi-turn text-to-SQL, emphasizing iterative refinement with environment feedback, which is closely related to the a-Interact paradigm. It should be mentioned in Section 7 and around Section 4.2, contrasting BIRD-INTERACT’s fixed-budget, hand-crafted action space with SQL-Trail’s RL-driven exploration, and clarifying whether BIRD-INTERACT could serve as an environment for SQL-Trail-style agents.

If these papers are already cited in the final version, the authors should ensure the discussion is explicit about similarities and differences in task design and simulator assumptions.

## Questions

1. **Simulator error impact.**  
   Can the authors provide quantitative evidence (on a subset of interaction logs) of how often the parser in the two-stage simulator misclassifies reasonable clarification questions as UNA, or maps them to the wrong AMB/LOC target, *during the main experiments*? A simple statistic like “on 200 randomly sampled clarification turns across all models, parser accuracy was X%” would increase confidence that the benchmark does not frequently penalize sensible behavior.

2. **Reward sensitivity.**  
   How sensitive are the main conclusions (e.g., GPT‑5 excelling in a-Interact vs c-Interact, observation of ITS law) to the specific reward weights (0.7/0.3 across sub-tasks, debugging penalties)? If you re-scale to, say, 0.5/0.5 or remove debugging penalties, do model rankings or main qualitative patterns in Table 2 and Figure 4 change? Even a brief ablation on a smaller subset (e.g., 100 tasks) would clarify robustness.

3. **Baseline agent design.**  
   Have the authors experimented with any simple *hand-crafted* agent policy for a-Interact (e.g., always retrieve schema and relevant KB first, then ask clarifications for each annotated ambiguity before executing) to see how far such heuristics can go compared to general-purpose LLMs? This would help demonstrate that significant headroom exists for algorithmic improvement, not only better base models.

4. **Extensibility to training.**  
   Do the authors envision BIRD-INTERACT primarily as an *evaluation* benchmark, or do they see it as suitable for training RL or imitation-learning based agents? If the latter, are there any challenges in terms of simulator determinism or reward sparsity that future users should be aware of?

5. **Clarifications on ITS law scope.**  
   The paper proposes the “ITS law” (Section 5.2) stating that a model satisfies this law if performance matches or surpasses idealized single-turn performance given enough interaction turns. Do any models in Figure 4 actually surpass the single-turn line, or do they merely approach it from below? It would be good to spell this out, because for some models (e.g., Qwen‑3 in the c-Interact panel) performance appears flat or even slightly non-monotonic.

6. **Potential bias from ground-truth snippets in clarifications.**  
   Since each ambiguity is paired with a SQL snippet from the ground truth which guides simulator responses (Section 3.2), do you observe that simulator answers sometimes contain unnaturally SQL-like phrasing (e.g., explicit column names or SQL keywords) that could bias models toward particular solutions? A small qualitative analysis here would be useful.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The benchmark design, equations, and simulator are overall sound and well-motivated, with extensive experiments and analyses. Some aspects (reward sensitivity, parser error impact, statistical variance) could be analyzed more rigorously but do not undermine the core claims.

## Presentation Rating

3: good.  
The paper is generally clear, well structured, and rich in detail, with helpful figures and tables. A few figures are dense and some minor inconsistencies (e.g., reward “normalized to [0,1]” vs % in tables) exist, but these are fixable.

## Contribution Rating

4: excellent.  
The work introduces a substantial new benchmark with a non-trivial interactive environment, addresses a real gap in text-to-SQL evaluation, and backs it up with thorough empirical and simulator analyses. It is likely to become an important reference point for interactive text-to-SQL and LLM-as-agent research.

## Overall Rating

8: Accept, good paper (poster).  
The paper offers a high-quality, thoughtfully designed interactive benchmark that addresses a clear unmet need in the community, with solid engineering, a principled simulator, and extensive analysis. While there are some missing ablations and minor formal inconsistencies, these do not detract from the overall value. I recommend acceptance and expect BIRD-INTERACT to be widely used.

## Reviewer Confidence

4: confident.  
I am familiar with text-to-SQL, LLM-as-agent, and benchmark design literature, have carefully read the main text, equations, and key appendices, and cross-checked the claims against the presented results. Some implementation details (e.g., full simulator code) cannot be verified, but my assessment of the scientific contribution is unlikely to change drastically.