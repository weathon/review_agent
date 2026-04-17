---
job_id: c439a4d9-d055-4b47-8554-074f54830283
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: GRufFX1gAy.pdf
paper: InnoGym: Benchmarking the Innovation Potential of AI Agents
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The work introduces a benchmark and framework for evaluating innovation of AI agents, including a formal task definition, metrics, and an execution environment. This fits ICLR’s “datasets and benchmarks,” “infrastructure, software libraries,” and “general machine learning / agents” topics.

## Minimum Quality
Pass ✅.  
The paper is in English and has all major sections (Abstract, Introduction, Method/Framework, Benchmark construction, Experiments, Related Work, Conclusion). Methods and equations are coherent, experiments are nontrivial, and the exposition is generally clear and complete enough for review.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
The only “prompts” present are part of the benchmark’s own methodology (Appendix H–I). They do not attempt to influence the review process.

---

# Expected Review Outcome:

## Summary

The paper introduces **InnoGym**, a framework and benchmark for evaluating the innovation potential of AI agents. Tasks are formalized as $\mathcal{T}=(P,S,V,D)$ with two central metrics: **performance gain** $G(s)$, measuring improvement over best-known solutions, and **novelty** $N(s)$, measuring methodological dissimilarity from prior solutions via an LLM-based distance function. The authors curate 18 “Improvable Tasks” from real engineering and scientific competitions, standardize them into iBench, and provide a unified execution environment (iGym). Experiments with three agent frameworks on 10 tasks analyze current agents’ performance and novelty, showing large gaps to human SOTA and a misalignment between novelty and robustness.

## Strengths

1. **Clear conceptual framing of “innovation” with explicit metrics.**  
   The formalization of a task as $\mathcal{T}=(P,S,V,D)$ and the definitions of
   \[
   G(s)=V(s)-V_{\text{known}}^*,\quad N(s)=C(s)\cdot \min_{h\in S_{\text{known}}}D(s,h)
   \]
   (Equations (2)–(3) on Page 3) give a clean separation between *value* (performance) and *method difference* (novelty). This addresses a real gap in current benchmarks that conflate “better score” with “more innovative method”.

2. **Nontrivial benchmark construction and standardization effort.**  
   The pipeline in Section 3.1–3.2 and **Figure 2** (Page 5) shows a detailed, multi-stage selection and augmentation process: from 197 tasks to 18 after resource checks, evaluator validation, and domain balancing. The explicit handling of **Absoluteness, Executability, and Correctness** in Section G.2 (Appendix) is better thought out than most benchmark papers and gives confidence that the scores are meaningful across tasks.

3. **Interesting use of LLMs to operationalize methodological distance.**  
   The distance function $D_{\text{AGENT}}$ in Equation (4) (Page 21) uses a two-stage Codex/GPT-5 pipeline: extraction to structured summaries and pseudocode, followed by rubric-based comparison along six method dimensions. Section F then validates this against EquiBench code variants and curated method triplets, with quantitative results in **Tables 8–12** showing reasonable agreement with human judgments. While not perfect, this is one of the more serious attempts I have seen at measuring “how different is this method” beyond raw code diff.

4. **Reasonably informative empirical analysis, not just a score table.**  
   The main comparison in **Table 2** (Page 8) is admittedly sobering (all negative gains), but the authors do not stop there. Section 4.3 analyzes solution trajectories, base models, and temperature. In **Figure 5**, the solution tree and “complex plane” embedding of $(G,N)$ visually explain how an agent evolves a solution; **Figure 6(a–c)** provides concrete evidence that (i) performance improves while novelty decreases over time, (ii) stronger base models materially shift performance, and (iii) temperature controls the exploration–exploitation trade-off in a way that is clearly visible in both gain and novelty. These analyses make $G$ and $N$ feel like live, interpretable quantities rather than arbitrary scores.

5. **Task diversity and explicit characterization of reference solution space.**  
   **Table 3** (Page 18) lists 18 tasks across ML competitions, operations research (ROADEF), combinatorial optimization (2D bin packing, graph coloring), systems (CompilerGym), etc., along with the number of reference solutions and a diversity statistic $\mathrm{Div}(\mathcal{T})$ derived from pairwise $D_{\text{AGENT}}$. This is a nice touch: it acknowledges that novelty is harder when $S_{\text{known}}$ is already methodologically diverse, and it exposes that information to users of the benchmark.

6. **Unified agent execution environment with attention to long-horizon robustness.**  
   iGym’s architecture in **Figure 4** (Page 6) and Appendix C emphasizes asynchronous tool dispatch, recovery, and concurrency. This matches the claimed setting (long-running, multi-hour tasks), and provides infrastructure that can be reused beyond this benchmark.

7. **Positioning relative to existing agent ML-engineering benchmarks is explicit.**  
   **Table 1** (Page 7) systematically compares InnoGym to MLAgentBench, DSBench, MLEBench, MLGym, ScienceAgentBench, MLRCBench, and InnovatorBench along data domain, reference solutions, difficulty, compute, and whether novelty is evaluated. The “Eval Novelty” column clearly highlights the gap this work is addressing.

## Weaknesses

1. **Heavy reliance on LLM-as-judge for $D$ raises robustness, cost, and reproducibility concerns.**  
   The core novelty metric $N(s)$ hinges on $D_{\text{AGENT}}$, which is computed via Codex and GPT-5 (Sections 2.1, 4.1, F.1). While Appendix F provides some validation, the paper does not really address:  
   - **Stability:** How sensitive is $D_{\text{AGENT}}$ to prompt variations, temperature, or model updates? Using commercial APIs that are regularly retrained risks making $N(s)$ non-stationary over time, which is problematic for a benchmark intended as a standard.  
   - **Cost and scalability:** For tasks with many candidate solutions and a large $S_{\text{known}}$, computing $\min_{h\in S_{\text{known}}} D(s,h)$ requires many LLM calls. There is no complexity or cost analysis here, nor any pruning strategy.  
   - **Bias across domains:** The validation in Tables 8–12 focuses on programming and three narrow AI subfields; the same models are used to judge highly heterogeneous tasks (e.g., ROADEF scheduling vs. Trojan detection). There is no quantitative analysis of whether $D_{\text{AGENT}}$ behaves differently for, say, OR heuristics vs. Kaggle-style ML pipelines.  
   This matters because if $D$ is noisy or biased, the claimed “innovation” in $(G,N)$ space may be more a reflection of the judge model than of real methodological differences.

2. **Empirical evaluation is limited to 10/18 tasks and 3 scaffolds, and all performance gains are negative.**  
   Due to compute constraints, Section 4.1 evaluates only 10 tasks, and even within those, multiple entries in **Table 2** are “/” (no valid submission). The headline result is that **no agent beats the human SOTA on any task** and the average ratio is close to $-1$ for all three frameworks. While that is informative in showing that these tasks are difficult, it leaves open several questions:  
   - We never see **positive $G(s)$** in practice, so the behavior of the “breakthrough innovation” regime (high $G$, high $N$) is entirely hypothetical in this paper.  
   - Only three agent frameworks (MLAB, CODEACT, AIDE) are tested, all in ML-engineering style. There is no evaluation of more research-focused agents (e.g., methods akin to AlphaEvolve) on the OR / combinatorial tasks, where they might actually produce gains.  
   - Seven of the 18 tasks are never used to produce numbers in the main text. This weakens the empirical support relative to the benchmark’s breadth.  
   As a result, the central claim that InnoGym “measures innovation” is conceptually sound but not yet well demonstrated in the regime where agents actually innovate.

3. **Definition and calibration of “high novelty” remain vague.**  
   Although $N(s)$ is scaled to [0,100], there is no principled or empirical threshold for “high” vs. “low” novelty. In Section 2.2, the paper defines qualitative regimes (breakthrough, performance, conceptual innovation) in terms of “high $G$ and high $N$,” but never specifies, for a given task, what $N$ is large relative to $S_{\text{known}}$ and its diversity. For instance, in **Table 2**, novelty values around 50–70 are reported for several tasks, but the text in 4.2 simply says “mid-to-high novelty” without grounding this in any distribution. Given that **Table 3** already computes a diversity score $\text{Div}(\mathcal{T})$, it would be natural to relate individual $N(s)$ to task-level diversity (e.g., percentile within the pairwise $D_{\text{AGENT}}$ of $S_{\text{known}}$). As it stands, interpreting a novelty score of, say, 60 on OAG vs. 60 on BEETL is opaque.

4. **Some mathematical and definitional aspects are underspecified or slightly inconsistent.**  
   - Equation (1) defines $V^* = \max_{s\in S, C(s)=1} V(s)$, but recall $V(s)=C(s)\cdot R(s)$, so the constraint $C(s)=1$ is redundant; technically $V^* = \max_{s\in S} C(s)R(s)$ suffices. Minor, but suggests the definitions could be tightened.  
   - In Section 2.3 (Page 4), the text repeatedly refers to “As shown in Fig. 1(c)” for Solved, Improvable, and Exploratory problems, whereas Figure 1 actually has subfigures (c–e). This is more of a presentation issue, but it makes the taxonomy slightly confusing.  
   - In Equation (4), $D_{\text{AGENT}}(s_1,s_2) = \frac{1}{|\mathcal{K}|}\sum_k \frac{d_k(s_1,s_2)}{4}\times 100$, but there is no discussion of **variance** across dimensions. A pair of solutions might differ dramatically along one key dimension and be similar on the others, yet averaging smooths this to a moderate score. Some justification or ablation of this aggregation choice would strengthen the story.  
   None of these are fatal, but for a paper selling a “principled framework,” I would expect a more careful mathematical discussion of these choices.

5. **The iGym environment, while promising, is only superficially evaluated.**  
   Section 3.5 and Appendix C describe interesting features (asynchronous tool dispatcher, recovery, concurrency). However, there is **no empirical comparison** to prior SDKs like OpenHands or AutoGen. For example, we do not see metrics like “fraction of runs recovered after crash,” “speedup from concurrent tool use,” or even an ablation where the same agent is run with and without iGym’s recovery mechanisms. As a result, the added value of iGym over existing infrastructures remains largely qualitative.

6. **Experimental methodology choices deserve more scrutiny.**  
   - The protocol uses **only three runs per configuration**, then reports the best valid submission (Section 4.1). For highly stochastic agents with long horizons, this can be quite noisy. There is no reporting of variance across runs at the task level (beyond bootstrap over tasks in Tables 4–5).  
   - For the main 10-task comparison, failures are represented as “/” in **Table 2**, and a pessimistic imputation ($R=-1, N=0$) is used only in the secondary analysis (Tables 4–5). Important conclusions, like “MLAB leads in both Performance Gain and Novelty,” are drawn without fully integrating failures into the primary comparison figure. This can bias perception toward methods that at least occasionally succeed, even if their overall reliability is poor.  
   - In Section 4.3, **Figure 6(b)** compares base models including a “hypothetical GPT-5” and Gemini-2.5, but there is no detail on versioning, context window, or API settings beyond “same decoding hyperparameters.” Given that these models differ significantly in capabilities, more transparency is needed to make the comparison reproducible.

7. **Task taxonomy choices limit the benchmark’s scope.**  
   The paper deliberately excludes “Solved Problems” and “Exploratory Problems” from the core benchmark (Section 3), focusing only on Improvable tasks. While this is defensible, it makes the innovation taxonomy in Section 2.3 largely theoretical: there is no concrete evaluation of innovation in the Solved (pure-method novelty at fixed $V^*$) or Exploratory (0→1 feasibility) regimes. Since the introduction motivates *all three* regimes, it feels like a missed opportunity not to include at least a small set of such tasks, even if treated separately.

8. **Minor clarity and exposition issues.**  
   A few examples:  
   - In **Figure 3** (evaluation pipeline), the arrows and labels for visible vs. invisible data, performance vs. novelty pipelines are quite dense; without zooming, it is hard to parse.  
   - Some acronyms (e.g., NPR, PTTALC) are first introduced in Table 2 or Section 4 without expanding the full competition names until later.  
   - The “Vector-Space Representation” in **Figure 5(b)** maps normalized $G$ and $N$ into polar coordinates, but the text does not specify the precise normalization of $G$ (is it relative to $V^*$, or min–max scaled over iterations?), which affects interpretability.

9. **Missing directly related benchmarks in scientific-agent evaluation.**  
   The Related Work section is reasonably broad on ML-engineering and idea-generation benchmarks, but omits some very close contemporaries on scientific agents and research workflows: see below.

## Potentially Missing Related Work

1. **Bragg et al., “AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite,” 2025.**  
   This work assembles a suite of scientific research tasks to assess AI agents’ ability to conduct end-to-end research, overlapping with InnoGym’s focus on rigorous, multi-step scientific and engineering problems. It should be cited in Section 5 (Evaluation for ML Engineering and Scientific Discovery) and compared in **Table 1**, particularly regarding how each benchmark handles long-horizon workflows and what aspects of “research capability” they quantify (AstaBench emphasizes rigor and completeness, InnoGym emphasizes performance gain and methodological novelty).

2. **Nguyen et al., “ReplicatorBench: Benchmarking LLM Agents for Replicability in Social and Behavioral Sciences,” 2026.**  
   ReplicatorBench evaluates whether LLM agents can reproduce empirical findings in social sciences, which is conceptually related to InnoGym’s notion of using prior solutions and formal evaluation pipelines. It should be discussed in Section 5 under “LLM Agents of Innovation,” clarifying that while ReplicatorBench measures *replicability* and adherence to prior work, InnoGym explicitly targets *innovation* relative to those baselines (via $G$ and $N$). A brief remark around **Table 1** on the complementary focus of replicability vs. innovation would help position this work.

## Questions

1. **Stability and versioning of $D_{\text{AGENT}}$.**  
   How sensitive are novelty scores to the specific judge model and its version? For example, if Codex or GPT-5 are updated or replaced with another LLM, do you observe significant shifts in $D_{\text{AGENT}}$ values on a fixed set of solution pairs? Any empirical evidence (even small-scale) on this would increase confidence that InnoGym can serve as a long-term benchmark.

2. **Task-dependent calibration of “high novelty.”**  
   Have you examined the distribution of $N(s)$ for $S_{\text{known}}$ itself on each task? A natural way to calibrate would be to define “high novelty” as exceeding, say, the 90th percentile of pairwise distances among $S_{\text{known}}$. If you have such analyses, can you report them or at least summarize whether agent solutions in Table 2 fall above or below these baselines?

3. **Feasibility of scaled evaluation (many agents, many runs).**  
   Suppose a user wants to evaluate 20 different agent frameworks on all 18 tasks, each with 10 runs. Roughly how many LLM calls (and what order of dollar cost) does the novelty evaluation incur? Have you explored any approximations, such as computing $D_{\text{AGENT}}$ only to a subset of $S_{\text{known}}$ or using embeddings as a first-stage filter?

4. **Plans or preliminary results for tasks where $G>0$.**  
   Do you have any early experiments (even outside the paper’s compute budget) where an agent does surpass the best-known solution on one of the easier tasks? Seeing at least a single concrete example with positive $G$ and its associated $N$ would help validate that the metrics behave sensibly in the “breakthrough” regime.

5. **iGym ablation vs. alternative SDKs.**  
   Can you provide at rebuttal time a small comparison where the same scaffold (e.g., AIDE) is run on a task both within iGym and using its native environment or another SDK (such as OpenHands), to quantify the impact of your recovery and concurrency mechanisms?

Clarifications or additional experiments along these lines would likely raise my confidence and could justify a higher contribution rating.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The formal framework is coherent and the benchmark construction is methodical; the main technical choices (e.g., $G$, $N$, $D_{\text{AGENT}}$) are justified and partially validated. Remaining concerns center on the robustness and calibration of the LLM-based novelty metric and the limited empirical coverage, not on outright methodological flaws.

## Presentation Rating

3: good.  
The paper is generally well written, with clear structure and helpful figures/tables (notably Figures 1–2, 5–6 and Tables 1–3, 8–12). Some notational redundancies and minor inconsistencies exist, and a few figures are dense, but overall readability is solid.

## Contribution Rating

3: good.  
The paper makes a meaningful contribution by formalizing innovation as $(G,N)$, curating a cross-domain set of improvable tasks with standardized evaluators, and demonstrating a nontrivial novelty-evaluation pipeline. The conceptual and infrastructural advances are valuable, though the empirical side is somewhat underdeveloped and innovation in the high-$G$ regime is not yet realized.

## Overall Rating

8: Accept, good paper (poster).  
The work offers a timely and well-thought-out framework and benchmark for measuring innovation in AI agents, addressing an important gap in current evaluations. While the novelty metric’s dependence on LLM-as-judge and the negative-gain experimental regime leave room for improvement, the conceptual clarity, careful dataset construction, and initial validation of the distance measure justify acceptance as a strong poster.

## Reviewer Confidence

4: confident.  
I am familiar with agent benchmarks, representation/evaluation metrics, and LLM-based judging, and have checked the mathematical definitions and empirical setup reasonably carefully. Some uncertainty remains about long-term robustness of the LLM-based novelty metric and the diversity of future agent behaviors, but these do not affect my overall assessment.