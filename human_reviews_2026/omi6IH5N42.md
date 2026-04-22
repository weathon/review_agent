# HDL-FixBench: A Verifiable Repository-Level Benchmark for Hardware bug repair

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 8, 2, 6

## Abstract
Existing benchmarks for hardware design primarily assess Large Language Models (LLMs) on isolated, component-level Hardware Description Language (HDL) code generation from specifications, overlooking the critical challenge of repository-scale bug repair.
To address this gap, we introduce HDL-FixBench, the first benchmark for repository-level hardware bug repair. It comprises 57 high-fidelity instances curated from three industry-standard open-source hardware projects: OpenTitan, CVA6, and Ibex.
Each instance is curated through a rigorous methodology, combining a novel agent-based filtering pipeline with meticulous manual verification, and is accompanied by a fully reproducible, containerized EDA environment to ensure task quality and relevance.
Evaluating seven state-of-the-art LLMs with two prominent agent frameworks(SWE-Agent and OpenHands) on HDL-FixBench, we find that even the most advanced models perform significantly worse than on SWE-bench Verified, with the top-performing model resolving only 40.3\% of tasks. This finding highlights the unique complexities of hardware engineering and establishes HDL-FixBench as a challenging and crucial benchmark for advancing the next generation of automated hardware design and verification tools.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces HDL-FixBench, a benchmark that evaluates LLMs on repository-level hardware bug repair tasks. The contributions of the paper include: repository bug repair in real-world hardware projects, a distillation pipeline for hardware engineering tasks, and systematic evaluation of 7 LLMs across benchmarks.

### Strengths
* Repository-level hardware bug repair is important.
* Experiments on industrial projects demonstrate practical relevance.
* The work adapts software engineering benchmark practices for hardware-specific constraints.

### Weaknesses
* Since the authors target repository-level tasks, having only 57 instances from 3 repositories is quite limited.
* Excluding tasks >100 lines or >5 files makes me less excited, which contradicts the notion of "industrial-scale".
* The paper is missing ablation studies on filtering criteria, repository selection, or task complexity thresholds.
* Figure 4a is interesting but there is no discussion on the importance of different bug types.
* The only baseline is SWE-bench, which seems weak since this is a hardware benchmarking work. What about non-LLM baselines?

### Questions
Please see above.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces HDL-FixBench, a benchmark for evaluating LLMs on repository-level hardware bug repair tasks. The benchmark comprises 57 high quality instances from 3 open source hardware projects (OpenTitan, CVA6, Ibex), each with containerized EDA environments and carefully curated with human experts (demonstrated in Appendix A.3). The authors evaluate 7 state-of-the-art LLMs using SWE-Agent and OpenHands frameworks (questionable, though, whether these scaffoldings are appropriate for HW tasks), reporting ~40% success rate.

One interesting observation is that the paper doesn't just report numbers, it also “root causes” why hardware is so much harder by identifying critical weaknesses like the models' shallow understanding of hardware principles and their inability to coordinate multi-file edits across RTL, configuration, and verification components, thereby raising interesting questions about the need for superior long context and tool-use abilities for leading models.

### Strengths
- The benchmark fills an important void in hardware LLM evaluation, moving beyond component-level tasks to repository-scale challenges that better reflect real engineering workflows. The only comparable benchmark of this quality is CVDP Agentic. 
- The multi-phase construction methodology is thorough. Particularly noteworthy is the extensive manual validation in Phase 4 (Section 3.1.4), where the authors manually verify each instance, create supplementary patches to resolve environment issues, and critically, write new test cases when automated tests don't exist. This level of human expertise ensures high-quality, verifiable tasks.
- The authors provide a base docker image with commercial tools and then open-source the entire pipeline for building upon it. This is a realistic compromise: acknowledges realities while maximize reproducibility.
- The use of an agentic filtering pipeline with GitHub MCP and Firecrawl tools to classify PRs against a hardware error taxonomy is scalable. This is a good contribution to the data scarcity problem in the field and could be replicated and applied to other repos.

### Weaknesses
- With only 57 instances from 3 repositories (all Verilog/SystemVerilog), the benchmark's generalizability is questionable, though benchmark creation process via human experts by nature is very challenging 
- The paper mentions using SWE-Agent and OpenHands frameworks with "custom modifications" for Verilog/SystemVerilog, but doesn't provide sufficient detail about these agent configurations. What specific tools were made available to the agents? How were the action spaces defined? Were there hardware-specific prompts or guidance? In short, what are the agentic scaffolding used? 
- The use of public repositories raises contamination concerns that aren't adequately addressed beyond noting "low success rates suggest memorization is insufficient."

### Questions
- Were hardware-specific tools (e.g., Verilator, synthesis tools) accessible during task execution?
- Did you use any hardware-specific prompting or guidance in the agent setup?
- Any human expert baselines on a subset of tasks?
- How many of the 57 tasks could potentially run with Verilator or Icarus Verilog?
- Which exact commercial EDA tools are required in your base Docker image? Are you using Synopsys VCS, Cadence Xcelium, etc?
- Could you provide a tool chain breakdown: X tasks need only basic simulation, Y need assertions, Z need full UVM?
- What was the runtime of your benchmark on standard inference hardware?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces HDL-FixBench, the first benchmark for evaluating the repair of repository-level (RTL) hardware bugs using LLMs. The authors present a screening and build process for evaluating 57 real-world bug instances collected from three industrial-grade open-source hardware projects (OpenTitan, CVA6, Ibex).

### Strengths
This paper is the first benchmark focusing on repository-level hardware bug fixing. The multi-stage build process is a crucial aspect of this paper, particularly the use of an LLM Agent for semantic filtering (stage 3) to extract "real RTL bugs" and the manual verification in stage 4, which demonstrates a high degree of rigor.

### Weaknesses
1. The current number of 57 instances from 3 projects is relatively small. While we understand the scarcity of data in this field, the small scale may limit the diversity of benchmarks and the statistical significance when evaluating the models' capabilities.
2. Using Gemini 2.5 Pro as an agent to filter PRs may introduce bias, leading to unfair subsequent evaluations.
3. The author intentionally excluded tasks involving changes to more than 100 lines or 5 files. While this helps make the problem "manageable," it also arbitrarily excludes the many more complex refactorings and system-level bugs that exist in the real world. This makes the benchmark only representative of "small-scale, localized" hardware fixes, thus diminishing its claimed "repository-level" value.

### Questions
1. Does the benchmark's "resolved rate" depend solely on passing the testbench? Is it possible for a patch to pass the tests while simultaneously being completely unacceptable in practice (due to the introduction of numerous irrelevant files)? Does the highest score of 40.3% include this scenario?
2. How can we ensure that using Gemini 2.5 Pro as a filter does not bias the benchmark set towards tasks that favor the model architecture? Has cross-validation been performed, for example, using Claude-Sonnet-4 as a filter to see if it selects 57 different tasks?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces HDL-FixBench, a new benchmark designed to evaluate large language models (LLMs) on their ability to automatically detect and fix functional and syntactic errors in hardware description language (HDL) code. The benchmark is built from three open-source CPU core repositories, with each bug instance accompanied by the corresponding fixed version verified through simulation. The benchmark provides a well-defined evaluation framework including metrics such as resolution rate and file-level precision, ensuring that both syntactic correctness and functional validity are assessed. The authors systematically evaluate seven different LLMs, including both open-source and proprietary models, using a uniform multi-turn prompt-based setup. They analyze success rates across repair complexity, bug type, and interaction depth, highlighting key challenges in applying LLMs to HDL repair.

### Strengths
1. The paper addresses a timely and underexplored problem -- automated code repair in the HDL domain -- by introducing a benchmark specifically tailored for hardware design.
2. The work presents a comprehensive and methodical evaluation of seven different LLMs (including both base and reasoning-enabled variants), measuring resolution rate, functional correctness, and file-level precision.
3. The authors offer valuable insights on how HDL repair differs from traditional software engineering tasks.

### Weaknesses
1. The benchmark size is small (57 instances), which limits statistical robustness and the ability to draw strong generalization conclusions.
2. The dataset's representativeness is narrow, as all three source repositories correspond to CPU core designs. This design bias restricts the benchmark's ability to test model adaptability to other hardware categories such as DSP modules, CNN accelerators, or LLM hardware implementations.

### Questions
1. Please clarify what specific information is included in each iteration's prompt—for example, does the model receive compilation errors, waveform logs, or only textual diffs between buggy and fixed versions? Such details are important for interpreting how much context the model relies on for effective debugging.
2. Please provide the API cost for each model when running the whole benchmark.
3. Do you enable reasoning mode for those proprietary models? Any insights on the reasoning time or comparing with the non-reasoning models?
4. Can you provide more diverse designs like CNN or LLM accelerators written in RTL?

Nit: Please use citep for the inlined citations.

### Soundness
3

### Presentation
3

### Contribution
3
