# Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Large language model (LLM) applications such as agents and domain-specific reasoning increasingly rely on *context adaptation*: modifying inputs with instructions, strategies, or evidence, rather than weight updates. 
Prior approaches improve usability but often suffer from brevity bias, which drops domain insights for concise summaries, and from context collapse, where iterative rewriting erodes details over time. 
We introduce ACE (**A**gentic **C**ontext **E**ngineering), a framework that treats contexts as evolving playbooks that accumulate, refine, and organize strategies through a modular process of generation, reflection, and curation. 
ACE prevents collapse with structured, incremental updates that preserve detailed knowledge and scale with long-context models.
Across agent and domain-specific benchmarks, ACE optimizes contexts both offline (e.g., system prompts) and online (e.g., agent memory), consistently outperforming strong baselines: +10.6\% on agents and +8.6\% on finance, while significantly reducing adaptation latency and rollout cost. 
Notably, ACE could adapt effectively without labeled supervision and instead by leveraging natural execution feedback. 
On the AppWorld leaderboard, ACE matches the top-ranked production-level agent on the overall average and surpasses it on the harder test-challenge split, despite using a smaller open-source model.
These results show that comprehensive, evolving contexts enable scalable, efficient, and self-improving LLM systems with low overhead.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces ACE, a framework where the model adjusts its inputs using “reference materials” such as task instructions, coping strategies, or past experiences. By leveraging these contextual references, ACE helps improve the model’s overall performance.

### Strengths
1. I really like Section 2.2 — it clearly sets up the motivation by analyzing Brevity Bias and Context Collapse before presenting the problem.

2. The proposed method looks quite effective.

### Weaknesses
1. The method heavily depends on feedback quality, but lacks a clear solution for unreliable feedback and has limited robustness evaluation.

2. The causal attribution of each core module’s effectiveness is unclear, and the ablation study is somewhat incomplete.

3. The discussion on long-context cost is not convincing enough and doesn’t fully reflect real-world deployment concerns.

### Questions
1. The paper states that ACE relies heavily on reliable feedback signals (e.g., code execution results or ground-truth labels), but doesn’t explain what happens when feedback quality is poor. Experiments show that when there are no gold labels or the feedback is noisy (like in ambiguous financial tasks), ACE’s performance drops sharply—sometimes even below baseline. However, this is only listed as a “limitation,” with no mitigation strategies explored. The paper also lacks quantitative tests on low-quality feedback (e.g., different noise levels), so we can’t tell how robust ACE is in messy real-world environments where feedback can be unreliable.

2. The current ablation only compares “with vs. without reflection” and “single-round vs. multi-round adaptation,” but doesn’t isolate the two key designs — incremental update and refine-as-you-extend. As a result, it’s unclear whether the gains come mainly from reducing information loss (via incremental updates), cutting redundancy (via refine-as-you-extend), or both. This makes it harder for future research to pinpoint which component drives the improvement.

3. The paper doesn’t test what happens when the generator, reflector, and summarizer use models of different strengths. All components use the same model (DeepSeek-V3.1), but in real scenarios, one might use a stronger model for the reflector. Would ACE still be stable in that case? Would costs rise significantly? These are important but unexplored questions.

4. The paper mentions using “KV cache reuse” to reduce cost, but doesn’t actually show numbers. For instance, how does ACE’s inference time or memory usage change when handling 18k-token contexts with and without caching? How does that compare to short-context methods (like GEPA’s concise prompts)? Without this data, the claim of “low cost” feels unsubstantiated for practical readers.

5. The paper also skips the retrieval efficiency problem for long contexts. ACE’s context is a structured “playbook,” but as it grows, how does the model quickly find the relevant parts (e.g., specific financial rules)? If retrieval time increases, it could cancel out the latency benefits of incremental updates. The paper doesn’t discuss or test any such optimization.

6. In the AppWorld experiments, ACE is compared against a “top commercial agent (IBM-CUGA)” that’s said to be “based on GPT-4.1.” However, the paper doesn’t reveal whether that system also used context optimization or even had the same test setup. If not, ACE’s “catch-up” advantage might be overstated, raising concerns about fairness.

7. In ACE’s refine-as-you-extend mechanism, key parameters like the deduplication threshold (how to judge when two strategy fragments are semantically redundant), maximum context length (when to trigger pruning), and number of reflection iterations (set to 5) are not specified or justified. Without details on why “5 rounds” were chosen or how different values affect performance, it’s hard for others to reproduce the reported results accurately.

### Soundness
2

### Presentation
2

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
The paper introduces ACE (Agentic Context Engineering), a framework for context-based self-improvement of LLMs that treats prompts as evolving, structured playbooks rather than concise summaries. It addresses two common failures in prior prompt optimization—brevity bias and context collapse—by separating roles into a Generator, Reflector, and Curator; applying incremental delta updates instead of monolithic rewrites; and using a grow-and-refine mechanism for de-duplication and maintenance. ACE operates in both offline (prompt/system prompt optimization) and online (test-time memory) settings, leveraging natural execution feedback and optionally ground-truth labels. Across agent (AppWorld) and domain-specific (FiNER, Formula) benchmarks, ACE outperforms strong baselines (e.g., GEPA, DC), delivering average gains around 9–11%, matching or exceeding a top production agent on AppWorld using a smaller open-source model, and substantially reducing adaptation latency and cost (e.g., up to ~82–92% latency reduction). Ablations confirm the value of the Reflector, multi-epoch refinement, and offline warmup. Limitations include reliance on feedback quality and a capable Reflector, and reduced benefits in tasks that favor concise instructions over rich, accumulated context.

### Strengths
1. The paper is in general well-written.
2. The benchmarks are carefully selected and representative, but would be better to include more to make the conclusions more solid, e.g., 1 or 2 more.
3. The paper identifies two important pitfalls in existing works, and designs better approaches to address the issue.

### Weaknesses
1. It is not clear that whether the proposed will still be effective for more powerful (and potentially more knowledgeable model like GPT-5)
2. It is not accurate to claim that ACE uses much smaller model (DeepSeek-V3.1, 685B) than GPT-4.1 (unknown size)
3. This work builds on top of Dynamic cheatsheet. More detailed comparison and analysis to it is necessary. What is the main difference to it? What are their scores in the benchmark? Currently, I feel that ACE makes minore changes, while the core idea to evolving playbooks s that accumulate, refine, and organize strategies is similar to Dynamic cheatsheet.

### Questions
Could you provide some examples on context collapse as shown in Figure 2?

### Soundness
2

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
This paper proposed ACE which addresses 2 observations found from existing context engineering methods : diversity ( brevity bias ) and context collapses. The framework uses incremental delta updates and a grow-and-refine mechanism to preserve detailed knowledge while preventing information loss. Evaluated on agent benchmarks (AppWorld) and domain-specific tasks (FiNER, Formula), ACE achieves 9.0% average improvement over strong baselines while reducing adaptation latency by 82-91% and costs by 75-84%.

### Strengths
1. The motivation and proposals are grounded on real world observations with strong empirical results.

2.  Experiments on different benchmarks (ICL, MIPROv2, GEPA, DC) and ablations (Reflector, multi-epoch, offline warmup) are robust and well thought out.

### Weaknesses
1. Only DeepSeek-V3.1 tested; generalization to other LLMs unclear.

2. While the framework is effective, it primarily extends Dynamic Cheatsheet with engineering improvements rather than introducing novel concepts

### Questions
1. Could you provide a cost ( token counts ) per eval breakdown comparison between ACE, GEPA and baseline?
   
     a. Figure 5b shows dollar costs for FiNER but similar analysis for AppWorld would strengthen cost claims. Specifically: Tokens per query (input/output), Cumulative tokens over full evaluation, Breakdown by component (Generator vs Reflector vs Curator)

2. How does ACE perform more real world repetitive tasks such as text-to-sql or ddxplus? A benchmark called streambench would be well suited for this kind of scenario.

3. Only one model is selected ( DeepSeek-v3.1 ) which is an odd choice, any reason why smaller models or cheaper models are not evaluated as well?

4. In section 5 the claim that longer context does not cause higher cost due to KV caching does not seem to hold in ACE as the context memory would reset on every delta update right?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces ACE (Agentic Context Engineering), a framework for context adaptation in large language models (LLMs). The core idea is to treat model context as an evolving "playbook" that accumulates domain-specific strategies over time. ACE uses a modular setup with three components: a Generator, a Reflector, and a Curator. It aims to avoid two problems found in existing methods: brevity bias, where important details are lost due to excessive summarization, and context collapse, where repeated updates degrade the quality of context. The authors evaluate ACE on agentic and domain-specific benchmarks. Results show consistent gains over strong baselines, both in offline and online adaptation settings.

### Strengths
1. The paper addresses a relevant and underexplored problem in LLM system design.

2. The ACE framework is well-structured and motivated by observed failure modes in prior work.

3. The paper is clearly written and provides sufficient technical detail.

### Weaknesses
1. The novelty is moderate. ACE extends prior methods like Dynamic Cheatsheet and GEPA rather than introducing a fundamentally new paradigm.

2. The approach depends on the quality of the Reflector. In settings without reliable feedback, performance drops.

3. There is limited discussion of failure cases. For example, when the context grows too large or accumulates conflicting strategies, how does ACE prevent quality degradation?

### Questions
1. How sensitive is ACE to the Reflector’s performance? Would a weaker model lead to significant degradation?

2. Could ACE be combined with retrieval-based augmentation to further improve performance in low-data settings?

### Soundness
2

### Presentation
3

### Contribution
2
