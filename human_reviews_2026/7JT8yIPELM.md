# MemGUI-Bench: Benchmarking Memory of Mobile GUI Agents in Dynamic Environments

- Avg Score: 4.67
- Decision: Reject
- Scores: 2, 6, 6

## Abstract
Current mobile GUI agent benchmarks systematically fail to assess memory capabilities, with only 5.2-11.8\% memory-related tasks and no cross-session learning evaluation. We introduce \textbf{MemGUI-Bench}, the the most comprehensive, memory-centric benchmark with pass@k and a staged LLM-as-judge evaluator. Our contributions include: (1) a systematic memory taxonomy with analysis of 11 prominent agents; (2) 128 tasks across 26 applications where 89.8\% challenge memory through cross-temporal and cross-spatial information retention; (3) \textbf{MemGUI-Eval}, an automated evaluation pipeline with novel \textit{Progressive Scrutiny} and 7 hierarchical metrics for memory fidelity and learning effectiveness; and (4) comprehensive assessment revealing significant memory deficits across all evaluated agents. Our experiments expose 4-10× performance gaps between memory-intensive and standard tasks, demonstrate the potential of explicit long-term memory mechanisms, and identify 7 distinct failure modes through systematic analysis. MemGUI-Bench establishes crucial empirical baselines for developing more capable and human-like GUI agents. Code and results: \url{https://anonymous.4open.science/r/MemGUI-Bench-Anonymous}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposed a benchmark to evaluate short- and long-term memory usage among mobile GUI agents. This paper aggregates tasks from varied applications and compares 11 agent frameworks, revealing their relative performance.

### Strengths
1. **Large-Scale Effort.**
> This paper analyzes 11 agents and aggregates tens of applications, covering major works in the domain of mobile manipulation.

2. **Comprehensive agent support and evaluation protocol.** 
> As introduced in section sec 3.2, this work supports a unified pipeline to ensure robust agent evaluation. Also, the metrics proposed in sec 4 enable memory-targeting evaluations, with human-annotated references.

### Weaknesses
1. **Limited practical utility among agent developments.**
> First, memory seems like a useful component designed in *some* works, yet not a universal feature that needs to be incorporated by agents. Therefore, evaluating memory is an interval, intermediate self-check for some agents, rather than a universal correctness metric such as task success rates.

> Second, this work structure agent memory possibly inspired by how human memory works (this point is less justified as well), yet this may not generalize across all agent designs or be useful for the most performant agent at all. For example, this work separates short-term and long-term memory, yet agents may become effective in long context modeling thus no need to split memory by time. It may be more reasonable to frame this work as an empirical analysis on *some* agent frameworks that share similar memory structure.

> Lastly, it is unclear whether the established memory taxonomy generally applies to domains (web browsing, computer use) and tasks (personal assistant, work) beyond mobile manipulation.

2. **Inaccurate definition of memory.**
> From the description in section 2, it may be more accurate to frame the two categories as "in-session" and "cross-session" memory, as "short-term" and "long-term" are somewhat vague therefore cause confusion.

3. **Uncertain quality of evaluation examples.**
> It is unclear (based on the description in section 3.1) how the applications are selected, how the examples are created (synthesized by LM, manually annotated by human, recorded from real-human activities, etc.?), and what principles are integrated into the examples during this process.

4. **Shallow analysis findings.**
> The findings in section 5 lack practical implications, beyond that agent A is better in short/long-term memory. Deeper analysis, especially supported by clearer, more fine-grained memory aspects, such as procedural workflow/fact retention, cross-app retention, should be much more informative for future agent development works. This limitation in analysis may be somewhat incurred by the lack of clarity in task design (weakness pt3).

5. **Presentation needs to be improved.**
> There are multiple places where the tables and figures are not placed properly, e.g., Table 1. Meanwhile, improving the writing to elaborate the motivation and detailed procedures clearer, could be helpful for readers to understand this work.

### Questions
N/A

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes MemGUI-Bench, a benchmark for mobile GUI agents focused on memory. It reports: (i) a “systematic memory taxonomy,” (ii) 128 tasks across 26 apps, with ~90% designed to stress memory, (iii) MemGUI-Eval with “Progressive Scrutiny” and 8 hierarchical metrics, and (iv) broad evaluations arguing existing agents have major memory deficits (4–10× gap between memory-intensive vs “standard” tasks) and that explicit long-term memory helps. Code link is provided.

### Strengths
- Timely problem framing. Mobile GUI agents are rising; a purpose-built memory benchmark is valuable and under-served. The cross-temporal/cross-spatial emphasis aligns with real usage.

- Scale & coverage. 128 tasks / 26 apps is non-trivial for interactive GUI evaluation; the claimed memory-task share (~89.8%) suggests deliberate design rather than incidental memory.

- Evaluation pipeline ambition. “Progressive Scrutiny” + hierarchical metrics aim to move beyond pass/fail, which is the right direction for agent memory diagnostics.

### Weaknesses
- **Generalization beyond the curated suite**: The benchmark spans 26 apps, but are they category-balanced (commerce, productivity, social, finance), regionally representative, and covering UI paradigms (infinite scroll, nested modals, webviews)? Without a sampling rationale and held-out app categories, it’s unclear if results generalize or if models “learn the test.”

- **Memory vs. perception/exploration confound**: It’s unclear whether measured failures are truly memory failures versus UI perception, layout parsing, long-horizon exploration, or tool-use orchestration. Without perception-controlled variants (e.g., textual UI abstractions; identical observation streams with/without memory demand), a 4–10× gap could reflect harder vision or search rather than memory per se. The paper needs ablations isolating memory load from these factors. (They currently only assert memory difficulty; details are missing on how confounds are neutralized.)

### Questions
I don't have any questions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
MemGUI-Bench is a memory-centric benchmark for mobile GUI agents. It comprises (i) short-term vs. long-term memory taxonomy; (ii) a task suite of 128 tasks across 26 apps with 89.8% memory-intensive cases and 64 mirror pairs to test cross-episode learning. Further, the paper introduces MemGUI-Eval, an automated evaluation pipeline with 8 hierarchical metrics (IRR, MTPR, FRR, etc.) and pass@k support. Finally, the authors present a comprehensive evaluation of 11 agents showing large gaps between memory and non-memory tasks.

### Strengths
- The paper identifies a clear gap in the existing mobile-agent benchmarks. MemGUI-Bench bridges the gap with a clear focus.
- The modular feature of MemGUI-Bench eval makes it easy to integrate with existing benchmarks.
- Surprising low performance of existing state-of-the-art methods on the benchmark, substantiating its claims on the memory-gap.

### Weaknesses
- Paper formatting for Table 1 and Figure 4.

- Lack of qualitative examples to showcase the memory-gap in state-of-the-art model like UI-TARS. Adding such examples will demonstrate the need of the benchmark more clearly.

- L27: "First comprehensive benchmark for GUI-agent memory" is plausible, but Table 4 shows prior memory tasks exist (e.g., SPA-Bench has 40/340). I would suggest qualifying to "first comprehensive, memory-centric benchmark with pass@k and a staged LLM-as-judge evaluator."

### Questions
- Does authors have an explanation for Table 3 numbers? Why does the pass@k with increasing k not increasing for some models but increasing for others? Further, does the performance saturate after a certain k in all the models. It would be nice to see such a curve for open-source models at least.

- Can the authors add a test-time compute normalized evaluation? For each agent, fix a compute budget - tokens/step, steps/episode and show SR/IRR/FRR deltas under equal budgets.

### Soundness
3

### Presentation
2

### Contribution
3
