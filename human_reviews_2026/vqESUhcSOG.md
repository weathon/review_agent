# From Large to Small: Transferring CUDA Optimization Expertise via Reasoning Graph

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6

## Abstract
Despite significant evolution of CUDA programming and domain-specific libraries, effectively utilizing GPUs with massively parallel engines remains difficult. Large language models (LLMs) show strong potential in generating optimized CUDA code from sequential code. However, using LLMs in practice faces two major challenges: cloud-based APIs pose risks of code leakage, and local deployment is often computationally expensive and inefficient. These drawbacks have spurred interest in small language models (SLMs), which are more lightweight and privacy-friendly. Encouragingly, recent studies show that SLMs can achieve performance comparable to LLMs on specific tasks. While SLMs can match LLMs on domain-specific tasks, their limited reasoning abilities lead to suboptimal performance in complex CUDA generation according to our experiments.
To bridge this gap, we propose ReGraphT, a training-free, retrieval-augmented generation framework that transfers LLM-level reasoning to smaller models. ReGraphT organizes CUDA optimization trajectories into a structured reasoning graph, modeling the combined CUDA optimizations as state transitions, and leverages Monte Carlo Graph Search (MCGS) for efficient exploration.
We also present a CUDA-specific benchmark with difficulty tiers defined by reasoning complexity to evaluate models more comprehensively. Experiments show that ReGraphT outperforms HPC-specific fine-tuned models and other retrieval-augmented approaches, achieving an average 2.33× speedup on CUDAEval and ParEval. When paired with DeepSeek-Coder-V2-Lite-Instruct and Qwen2.5-Coder-7B-Instruct, ReGraphT enables SLMs to approach LLM-level performance without the associated privacy risks or excessive computing overhead.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces ReGraphT, a retrieval-augmented generation framework to transfer the CUDA programming reasoning from large language models (LLMs) to small language models (SLM). (1) It formulates the CUDA code generation task using LLMs as a graph-based problem, where this graph encodes the CUDA optimizations and (2) incorporates Monte Carlo Graph Search (MCGS) to guide the search to improve efficiency. 
This paper also proposes a CUDA evaluation benchmark (CUDAEval) that evaluates LLM performance in CUDA code generations based on 10K CUDA files sampled from the real-world CUDA files.
ReGaphT with MCGS outperforms the CoT or RAG method regarding the correctness and the optimization performance of the generated code.

### Strengths
* This paper transforms the CUDA optimization problem into a structured reasoning graph. This abstraction provides a clear, interpretable framework for modeling optimization dependencies and interactions
* Introduce of the CUDAEval benchmark, a dedicated benchmark that classifies CUDA generation tasks

### Weaknesses
* The github link for the code is empty when the review period has started (at least by Oct 16)
* Comparisons are limited to standard prompting, CoT, and generic RAG. No comparison against distillation or reinforcement-learning-based compression (e.g., DeepSeek-R1 distilled SLMs) is given.
* The CUDAEval benchmark defines its difficulty levels based on the length of reasoning trajectories, but that depends on how the LLM behaves rather than on the actual complexity of the CUDA code itself.

### Questions
* The paper aims to enable SLMs to achieve large-model-level CUDA code generation performance, but the proposed ReGraphT framework still depends on a LLM to construct the reasoning graph and generate the initial optimization trajectories. How do you justify the efficiency improvement?
* How sensitive are results to rollout count N = 10 and search budget = 200?
* Results in Tables 1 and 2 lack variance/error bars or p-values. Improvements of 1-2 % can be stochastic generation variance. Could you provide the variance of those numbers?
* Is the reported speedup calculated as an average across all CUDA files, or as a geometric mean?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces ReGraphT, a training-free framework that transfers CUDA optimization reasoning from large to small language models using a structured CUDA Reasoning Graph and Monte Carlo Graph Search. It achieves near-LLM performance on CUDA code generation benchmarks while remaining lightweight and privacy-friendly for local deployment.

### Strengths
- method achieves near LLM performance
- it allows for local deployment (I wonder how important this is for CUDA code generation?). It would be more sensitive for private conversations but in this context is not really needed, why not use a bigger model?

### Weaknesses
- Code link is empty
- The paper content leans toward systems and engineering design, rather than core theoretical or algorithmic ML contributions. Therefore it doesnt seem appropriate for ICLR.
- The reasoning transfer mechanism is algorithmic plumbing rather than a model innovation
- Is more of an engineering project report

### Questions
- How scalable is the LLM-driven trajectory extraction process? Does building the reasoning graph require significant manual curation or prompt engineering to ensure consistency across optimization traces?
- What is the computational overhead of the Monte Carlo Graph Search compared to standard RAG retrieval or CoT prompting?
- Can ReGraphT transfer beyond CUDA (e.g., OpenCL, SYCL, or Triton code)?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the challenge of generating optimized CUDA code, a task at which Large Language Models (LLMs) excel but which is difficult for smaller, more efficient Small Language Models (SLMs) due to their limited reasoning capabilities. The authors propose ReGraphT, a novel, training-free framework designed to transfer the multi-step optimization reasoning of an LLM to an SLM.

### Strengths
- Practical Relevance and Problem Significance: The paper tackles a real and important problem. Optimizing code for parallel architectures like GPUs is a critical bottleneck in high-performance computing. Making this capability accessible via smaller, locally deployable models has significant practical implications for developer productivity, code privacy, and computational cost. The training-free nature of the framework further enhances its practicality.
- Thorough and Comprehensive Evaluation: The experimental setup is strong. The authors not only test their method on an existing benchmark (ParEval) but also contribute a new, well-motivated benchmark (CUDAEval) that allows for a fine-grained analysis of performance across different reasoning complexities. The inclusion of multiple modern SLMs and comparisons against several strong baselines (Standard, CoT, RAG) and SOTA LLMs provide a convincing demonstration of the method's effectiveness.
- Strong Ablation Studies: The paper includes detailed analyses that strengthen its claims. The study on the impact of ReGraph size (Figure 9, Table 3) demonstrates the convergence of the knowledge graph, while the overhead analysis (Section D) provides a transparent look at the costs involved. The ablation on MCGS traversal strategies (Section E) further validates the design choices within the framework.

### Weaknesses
- Convern on generalizability of the reasoning graph: While the paper demonstrates that the graph's structure converges, its generalizability to out-of-distribution problems is not fully explored. The graph is built from a dataset of 10K CUDA files filtered down significantly. It is unclear how well a single, pre-constructed graph would perform on CUDA tasks from entirely different domains (e.g., scientific simulation vs. deep learning kernels) that might require novel optimization patterns not seen during construction. This raises questions about the "one-time cost" of construction if new graphs are needed for new domains.
- Comparison of fine-tuning based approach: The framework is presented as "training-free," which is a key advantage. However, a compelling comparison would be to use the LLM-generated trajectories as a dataset to fine-tune an SLM. This would be a direct test of whether the explicit graph structure and MCGS search are more effective for imparting reasoning than standard supervised fine-tuning on the same high-quality data. While the introduction argues SFT has limited effectiveness for reasoning, an empirical comparison would make this claim more robust.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Summary

The paper proposes ReGraphT, a training‑free framework that lets small code LMs generate fast CUDA by borrowing multi‑step optimization “reasoning” from larger models. The authors first use an LLM to produce step‑by‑step CUDA optimization trajectories for many programs, then merge those steps into a CUDA Reasoning Graph (ReGraph) whose nodes are optimization techniques and whose edges are validated transitions accompanied by code examples. At inference, an SLM treats CUDA generation as graph traversal and uses Monte Carlo Graph Search (MCGS) with compilation, correctness checks, and runtime speedup as rewards to select an optimization path and produce code. Experiments on a new benchmark (CUDAEval) and on ParEval show that pairing ReGraphT with SLM backbones (DeepSeek‑Coder‑V2‑Lite‑Instruct, Qwen2.5‑Coder‑7B, HPC‑Coder‑V2) increases pass rates and roughly doubles speedup versus prompting and RAG baselines, narrowing the gap to strong LLMs. 

According to the *overview diagram on page 4*, ReGraphT has two phases: building ReGraph from LLM trajectories and exploring it with MCGS during generation. *Figure 5 on page 7* outlines the CUDAEval curation and verification pipeline, and *Table 1 on page 8* reports gains such as pass@1 moving into the low‑70% range and speedup@1 around 14× for SLMs with ReGraphT‑MCGS, compared with ~6–8× for standard prompting and RAG. The *difficulty analysis in Table 2 on page 9* shows larger benefits on medium and hard tasks, consistent with the method’s focus on multi‑step reasoning. 

Contributions

* ReGraphT framework: a training‑free method that transfers LLM‑derived optimization know‑how to SLMs via a reusable CUDA Reasoning Graph built from verified stepwise trajectories. 
* Graph‑based search for CUDA generation: formulation of CUDA optimization as state transitions on ReGraph and a tailored MCGS procedure with reward design that combines compilation success, functional correctness, and measured speedup. 
* CUDAEval benchmark: a curated set of 3,126 CUDA/CPU pairs with 313 held‑out evaluation tasks, stratified by reasoning complexity using trajectory length, plus a reproducible build/execute verification pipeline. 
* Empirical evidence: consistent improvements over standard prompting, CoT, and code‑similarity RAG on CUDAEval and ParEval; ablations that attribute gains to MCGS and to longer, more effective reasoning trajectories; analysis showing ReGraph converges after ~500 samples.

### Strengths
Strengths by dimension

Originality

* Introduces a training‑free way to transfer multi‑step optimization “know‑how” from an LLM to small code models by turning LLM‑generated optimization trajectories into a reusable CUDA Reasoning Graph and casting CUDA generation as graph traversal. This graph abstraction (nodes as optimization techniques, edges as validated transitions with examples) and the merge procedure are clearly new in the CUDA‑code LLM literature. The formal definition, Algorithm 1, and the relabeling routine that canonicalizes technique names make the idea concrete. 
* Adapts Monte Carlo Tree Search to a cyclic graph (MCGS) with an explicit P‑UCB selection rule, a rollout policy that avoids non‑termination on cycles, and a hierarchical reward mixing compilation success, test correctness, and measured speedup. Modeling CUDA optimization this way is a creative combination of known search ideas with domain‑specific constraints. The equations and rollout details on pages 6–7 are specific and principled. 
* Proposes CUDAEval, a benchmark built from real‑world CUDA files rather than synthetic sequential code, and stratifies tasks by reasoning complexity (trajectory length) rather than only algorithmic category. This reframes “difficulty” in code generation around reasoning depth, which is a fresh and useful lens. The curation and verification pipeline in Figure 5 and the easy/medium/hard tiers on page 7 ground this originality. 

Quality

* Methodological rigor is high: the paper provides a full pipeline from ReGraph construction to MCGS exploration with precise definitions, pseudo‑code, and formulas (Definition 1, Algorithm 1, Figures 2–4). This makes the approach reproducible and falsifiable. 
* Evaluation is careful and multi‑faceted. Results are reported on two benchmarks (CUDAEval, ParEval) using three SLM backbones (DeepSeek‑Coder‑V2‑Lite‑Instruct, Qwen2.5‑Coder‑7B, HPC‑Coder‑V2) with consistent budgets and metrics (pass@k for correctness and speedup@k for performance). Table 1 shows substantial and consistent gains for ReGraphT‑MCGS over standard prompting, CoT, and a code‑similarity RAG baseline. 
* Analysis links mechanism to effect. Table 2 and Figure 6 demonstrate that gains grow with task difficulty and that longer reasoning chains correlate with better speedups until a saturation point, supporting the claim that the framework injects multi‑step reasoning into SLMs. 
* Solid ablations and practicality checks: the paper studies search budgets and rollout counts (Table 4), compares reward formulations (Figure 11), examines graph growth and convergence (~500 samples; Table 3 and Figure 9), and breaks down construction and inference overheads (Figure 10), including wall‑clock estimates on A100 and consumer GPUs. These details increase confidence the method is not a one‑off. 

Clarity

* The narrative is easy to follow: the overview (Figure 2) cleanly separates graph construction and graph‑guided generation; Algorithm 1 steps through merging trajectories; Figure 4 maps MCGS phases to the CUDA setting; Figure 5 explains data curation at a glance. The consistent use of page‑local figures and definitions reduces ambiguity. 
* Experimental reporting is structured and legible. Table 1 compares methods across models and benchmarks in a single view; Table 2 breaks down difficulty tiers; subsequent figures and tables isolate the impacts of graph size, reward design, and search parameters. The paper also explicitly states hardware, precision, budgets, and metrics, which helps replication. 
* The appendix includes prompt templates and verification workflows, which is unusually transparent for a systems‑plus‑LLM paper and lowers the barrier for re‑use. 

Significance

* The empirical gains are large enough to matter in practice. For example, on CUDAEval with DeepSeek‑Coder‑V2‑Lite‑Instruct, pass@1 rises from 61.7% (standard) to 75.1% and speedup@1 from 6.54× to 14.46×; similar boosts appear for Qwen2.5‑Coder‑7B and HPC‑Coder‑V2. On ParEval, pass@1 improves from 40.0% to 55.0% with speedup@1 from 4.61× to 10.78×. These are not marginal deltas; they more than halve the gap to strong LLMs while preserving the deployability of SLMs. 
* The framework’s training‑free nature and the reported convergence of ReGraph after a few hundred samples suggest a reusable artifact that can be shipped with SLMs to unlock privacy‑preserving, local CUDA generation. The one‑time cost and search efficiency analysis further support practical adoption. 
* By defining difficulty via reasoning rather than only algorithm class, CUDAEval may nudge future work to measure and improve multi‑step optimization ability explicitly. The positive trajectory‑length/performance correlation (Figure 6) gives the community a concrete target and a measurement recipe. 
* The abstraction is likely portable: encoding verified optimization trajectories as a graph and exploring with MCGS is a general recipe that could extend to other performance‑critical domains where correctness gating and speedups are measurable, not just CUDA. The paper’s discussion hints at this broader applicability. 

Bottom line: This is a well‑executed systems contribution that combines a novel representation (reasoning graph), an effective search procedure (MCGS with domain‑aware rewards), and a carefully curated benchmark. The work meaningfully advances the practicality of small code models for GPU programming, with strong evidence and unusually transparent documentation.

### Weaknesses
1. Positioning vs prior search‑based reasoning is underdeveloped.
   The paper adapts MCTS to a cyclic “reasoning graph,” but the case for novelty over existing search‑guided generation is thin. Closest neighbors include MCTS‑style reasoning for code and RAG (e.g., RethinkMCTS for code generation; MCTS‑RAG), and iterative search over program transformations in compiler auto‑scheduling (e.g., Halide, TVM). The paper cites these lines of work but does not empirically contrast against them nor articulate a clear theoretical advantage beyond domain specifics. Add a head‑to‑head with: (i) vanilla MCTS over thought tokens (no graph), (ii) MCTS‑RAG over a CUDA corpus, and (iii) TVM/Halide‑style schedule search baselines where applicable; report compute‑normalized outcomes. A small theory section clarifying why P‑UCB on graphs with cycles has better regret or exploration properties than tree‑MCTS in this setting would also help. 

2. Ambiguities in the MCGS definition and missing hyperparameters.
   Equation (1) defines P‑UCB with (N(s')) but does not disambiguate “parent” counts when nodes have multiple parents or have been reached via different paths. The rollout policy in Eq. (2) introduces λ and ϵ without stating chosen values or sensitivity. Specify the state/action visit counters precisely for graphs, list default λ, ϵ, depth/rollout caps, and include a sweep showing robustness of performance to these choices. A brief pseudo‑code block for selection/expansion/backprop on a DAG with back‑edges would make reproduction safer. See Section 3.2 and Figure 4 on page 6. 

3. Reward shaping risks shallow plans.
   Rollouts treat every node as terminal and take the max reward along a trajectory (page 7), which can bias search toward early “flashy” speedups and discourage deeper combinations. Add an ablation replacing the max with: last‑state reward, discounted sum, and penalized path length; report impacts on trajectory length and speedup (particularly on medium/hard tiers in Table 2, page 9). 

4. Graph construction and relabeling lack quality control.
   The relabel step in Algorithm 1 (line 7, page 5) relies on an LLM to canonicalize method names, but there is no inter‑annotator or manual audit to ensure that, say, “avoid bank conflicts” doesn’t get misfiled under “memory coalescing.” Release the taxonomy of optimization methods and the mapping rules (Appendix G.4), plus a confusion matrix from a human audit of a random sample. Include a “no‑relabel” ablation to verify that gains are not an artifact of label collapse. 

5. Benchmark construction may inflate speedups and invites contamination.
   CUDAEval is built by extracting real CUDA kernels, then having an LLM generate the CPU serial counterpart and drivers before verification (Figure 5, page 7; Appendix A). That CPU baseline is unlikely to be an expert‑tuned -O3 implementation and could exaggerate speedup ratios. In addition, the source corpus (Stack v2 CUDA) likely overlaps with pretraining data of the evaluated models. Provide:
   • CPU baselines compiled with -O3 and simple algorithmic cleanups; report speedup vs both “LLM‑CPU” and “optimized CPU.”
   • A near‑dedup analysis (e.g., MinHash/BLEU‑SimHash) between CUDAEval and known pretraining sources for the backbones, and a leakage‑reduced split.
   • Results on hand‑written sequential baselines for a subset of tasks to calibrate speedup realism. 

6. Compute‑fairness and budget parity across baselines are unclear.
   Table 1 (page 8) reports pass@1/10 and speedup@1/10 for Standard/CoT/RAG vs ReGraphT/MCGS, but only ReGraphT variants have an explicit search budget of 200. It is not stated how the 10 samples for the other baselines are produced, nor whether compile/run evaluation time is matched. Add a table of wall‑clock and GPU hours per method for CUDAEval and ParEval, and enforce parity in generation attempts and verification calls. The overhead analysis (Figure 10, page 17) is useful but only for ReGraphT. Mirror this for the baselines. 

7. RAG baseline is weak.
   Using CodeBERTScore similarity as the sole RAG baseline underrepresents retrieval quality for code (structure‑aware and repository‑level retrieval matter). Since the paper cites Repoformer and EVOR, implement stronger RAG: repository‑aware retrieval, AST‑ or CFG‑conditioned retrieval, and multi‑hop chaining of examples. Also try “ReGraph as a retriever” without MCGS to isolate where the gains come from. See Baselines on page 8. 

8. Measurement methodology for speedups needs tightening.
   Reward and metrics depend on single‑run timings susceptible to GPU warm‑up, clock variability, and input‑size sensitivity. Document: number of timing repetitions, warm‑up iterations, clock locking, and variance; report medians with IQR or 95% CIs. Consider integrating compute‑sanitizer checks for race conditions and misuses that pass unit tests but are undefined at scale. This directly affects the “hierarchical reward” in Eq. (3) and strengthens the correctness claim beyond output equality. 

9. Difficulty labeling is model‑dependent.
   CUDAEval tiers are defined by trajectory length from DeepSeek‑R1 (page 7), which risks encoding that model’s biases about “what counts as a reasoning step.” Cross‑validate with human ratings on a sample and with a second LLM to show the tiers are not an artifact of a single annotator. Report correlations and disagreements. 

10. Generalization beyond CUDA remains a claim.
    The paper argues ReGraphT is portable, but all experiments are CUDA only. A small transfer study to Triton or OpenCL kernels, or to a CPU vectorization domain (e.g., OpenMP) using the same framework, would substantiate the portability claim. Even a pilot with 30–50 tasks would be persuasive. 

11. Ablations miss two levers that likely matter.
    The paper includes useful studies on rollout counts and reward strategies (Table 4 and Figure 11, pages 18–19) and on graph size saturation (Figure 9, page 16), but omits:
    • Edge content ablation: do edges need embedded examples, or is a method‑only graph sufficient?
    • Graph source ablation: build ReGraph from a different LLM or a smaller sample to test sensitivity to trajectory provider quality.
    Add both to clarify which ingredients are essential. 

12. Reproducibility gaps.
    Prompts are provided (Appendix G), which is excellent, but the paper should fix random seeds, publish exact model snapshots and decoding parameters for each backbone, and release the canonical method list and the final ReGraph artifact used in Table 1. The current anonymous repo link is helpful during review but not a substitute for an archival release. 

13. Practical deployment knobs are not exposed.
    Section D reports that 313 CUDAEval tasks with Qwen‑7B and budget 100 take ~6.02 hours on an A100, and ~7.53 hours on a 4090, but practitioners need guidance to trade accuracy for latency. Provide curves of pass@1 and speedup@1 vs budget, and vs rollout count, plus a “fast mode” configuration that still beats RAG on medium/hard tasks (Table 2, page 9). 

---

Bottom line. The paper’s core idea is promising, but it would benefit from clearer algorithmic specification, stronger and compute‑fair baselines, contamination‑aware benchmarking, and tighter measurement discipline. Addressing the items above would make the claims about transferring “LLM‑level reasoning” to SLMs far more convincing and easier to adopt.

### Questions
1. P‑UCB on graphs with cycles

   * Question: In Eq. (1) you use (N(s')) inside the log term. What is (s') when the same child has multiple parents, and how are visit counts disambiguated across different incoming edges? Please spell out the precise counters updated during selection/expansion/backprop when the structure is a directed cyclic graph rather than a tree. Suggestion: provide short pseudo‑code for selection/backprop on a general digraph and define all visitation statistics. A diagram keyed to *Figure 4 (page 6)* would prevent misimplementation. 

2. Rollout policy and hyperparameters

   * Question: What values did you use for the rollout policy in Eq. (2) (λ, ϵ), step caps, and termination conditions, and how sensitive are results to these? Suggestion: add a small sensitivity sweep and report variance bars for pass@1 and speedup@1 to show robustness. Cite where these values appear in the code or Appendix. *Section 3.2 (page 6).* 

3. Reward shaping choice

   * Question: You define the rollout’s final reward as the maximum reward along the trajectory (Eq. 3). Why prefer max over last‑state or discounted sums given the risk of favoring early “flashy” optimizations? Suggestion: include an ablation comparing max vs last‑state vs discounted‑sum reward, and report resulting trajectory lengths and speedups by difficulty tier. *Equation (3) and text on page 7.* 

4. What is “ReGraphT” vs “ReGraphT‑MCGS”?

   * Question: In *Table 1 (page 8)* you evaluate “ReGraphT” and “ReGraphT‑MCGS,” but the text only briefly says ReGraphT does random sampling with max attempts 5. Please provide explicit pseudo‑code for the non‑MCGS traversal and clarify budgets, stopping rules, and how conflicts/inapplicable edges are handled. This will make the comparison interpretable. 

5. Failure handling during rollouts

   * Question: A rollout “terminates if the optimization fails at any node.” Do you treat that as a hard terminal with negative reward only for that leaf, or do you propagate a penalty to the ancestor action? Suggestion: clarify how compilation/test failures affect backprop and whether you allow recovery by skipping the failed technique later in the path. *Section 3.2 (page 6).*

### Soundness
3

### Presentation
3

### Contribution
2
