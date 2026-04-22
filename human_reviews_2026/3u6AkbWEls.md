# ManipEvalAgent: Promptable and Efficient Evaluation Framework for Robotic Manipulation Policies

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 6, 6, 2

## Abstract
In recent years, robotic manipulation policies have made substantial progress. However, evaluating these policies typically requires large-scale sampling in simulation benchmarks, leading to high time costs. Moreover, existing evaluation pipelines are usually fixed, do not account for user needs, and report only a single scalar score, lacking interpretability. In contrast, human experts can quickly form an intuitive impression of a policy’s capabilities from just a handful of executions. We therefore propose ManipEvalAgent, an efficient, promptable, and dynamically multi-round evaluation framework for robotic manipulation policies. The framework conducts small-batch, multi-round evaluations and adaptively plans subsequent evaluation steps based on intermediate observations from each round. Via code generation, it constructs tasks and evaluation functions within simulator. By generating evaluation functions and leveraging vision–language models (VLMs) for video understanding, ManipEvalAgent provides user-instruction-centric, fine-grained analysis. Our approach offers three key advantages: (1) efficiency, no need for massive sampling; (2) promptable, planning the evaluation process according to user queries; and (3) interpretability, providing diagnostic text that goes beyond a single score. Across multiple settings, our evaluation method significantly shortens the overall time compared with traditional simulation benchmarks, while reaching conclusions comparable to those from large-scale simulation benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes ManipEvalAgent, a promptable, multi-agent framework to evaluate robotic manipulation policies. Instead of relying on fixed task suites, it turns open-ended user requests into concrete tasks and tools (via planning, code synthesis, and retrieval), runs policies in simulation, and iteratively refines the evaluation. The system mixes rule-based checks with VLM/VQA signals to produce aspect-level judgments and claims to reach similar conclusions to standard benchmarks with fewer samples and less wall-clock time.

### Strengths
1. Originality: Treats evaluation as an agentic, prompt-driven process; creative blend of program synthesis, retrieval, and vision tools.
2. Quality: Clear three-agent architecture; thoughtful engineering (retrieval-first, tool registry, README grounding); includes ablations and error breakdowns. I think the pipeline make sense.
3. Clarity: The decomposition into sub-aspects and the iterative probing loop are easy to follow; examples help.

### Weaknesses
1. Most evidence seems tied to a single simulator; claims of easy transfer aren’t yet demonstrated.
2. “Consistency of conclusions” needs formal definitions, confidence intervals, and sensitivity to the number of rollouts.
3. Reliance on VQA and generated tools raises calibration and brittleness concerns; mitigation is only partially explored.
4. Policy diversity and multi-suite coverage are modest; the open-ended query set lacks details on taxonomy and inter-annotator agreement.

### Questions
1. It will be better to define the agreement metric(s), report CIs, and add a trial-count sensitivity study (e.g., 3/5/10/20 rollouts) if possible.
2. Show human spot-checks vs. VQA, multi-VLM agreement, and simple perturbation tests; add unit tests for generated tools.
3. Measure human–agent agreement on sub-aspect judgments, not just aggregate consistency.
4. Do the authoer plan to opensource it? It should be clarified about the open-ended query taxonomy, labeling protocol, and inter-annotator agreement; consider releasing it?

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
This paper presents ManipEvalAgent, a promptable, multi-round, few-sample evaluation framework for robotic manipulation policies. Instead of exhaustively running fixed benchmark suites, the system plans sub-aspects from a user query, generates tasks and evaluation tools as Python code against a simulator, executes small batches, and adapts subsequent probes based on interim observations. Tools combine rule-based metrics (via simulator APIs) and VLM-based VQA over rendered videos, and results are aggregated into interpretable textual diagnostics rather than a single score. Experiments on RoboTwin 2.0 and LIBERO suggest ManipEvalAgent reaches conclusions comparable to standard pipelines with far fewer samples; an ablation shows RAG / visual self-reflection.Agent each improve code-generation success; an error breakdown attributes most failures to TaskGen/ToolGen. Multi-task evaluation is also discussed.

### Strengths
Promptable, adaptive evaluation that mirrors how humans probe policies, rather than fixed suites; clear three-stage design (Proposal / Generation / Execution). 

Agentic code generation for both tasks and tools, with RAG + visual self-check + README.Agent to stabilize generation—well engineered and ablated. 

Hybrid metrics (rule-based + VQA) let the evaluator capture aspects not exposed by simulator APIs, enabling finer-grained diagnostics. 

Evidence of agreement with standard benchmarks on several dimensions (success rate and LIBERO sub-suites), while using fewer samples; multi-task variant also reported. 

Failure analysis reveals where the system breaks (generation stage dominates), which is actionable for future iterations.

### Weaknesses
Agreement protocol is under-specified / potentially fragile. Appendix A.2.1 defines agreement by comparing a single randomly chosen task per benchmark against 10 runs of the agent and checking whether the benchmark SR lies within 1σ/3σ of the agent’s mean. This ignores task heterogeneity, seed variance, and policy × task interactions, risking unstable conclusions from small-N sampling. Please justify the statistical validity and add per-aspect calibration beyond one random task.

Ground-truthing of VQA metrics is not calibrated. VLM-based tools produce numeric judgments, but there is no report of AUROC/ECE, threshold selection, or prompt sensitivity under distribution shift (lighting, gloss, clutter). Given VQA results feed aggregation and next-round planning, lack of calibration can systematically steer the loop.

### Questions
See weaknesses

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a simulation benchmark for robotic manipulation that is adaptable and agentic. Users can ask specific questions about how they want their policy evaluated, like "how robust is my policy to variations in object colors?", and the simulation engine will use AI models to piece together simulation assets and call simulation APIs to construct a series of scenes that test the user's query. The simulation engine analyzes results evaluating a policy on a scene it created, and then based on the results decides what scene to construct next for evaluation. The main motivation the authors propose for having such an agentic evaluation platform is that simulated evaluations take quite a long time to run, and at the end you usually just get back one numerical score specifying how well your policy did, and so it's hard to debug what exactly the policy does/doesn't do well. In evaluating their AI-based evaluation platform, the authors find that indeed time can be saved, with baseline simulated evaluation platforms taking 2+ hours to evaluate, but the proposed approach taking ~45 minutes. Rankings of policies on the proposed adaptable benchmark also correlate with rankings on previous widely used simulation benchmarks.

### Strengths
(1) It is often the case that researchers want to learn what a policy they've trained does well at, and what it fails at, and there can be a lot of nuance in where the policy fails/does well. At the same time existing evaluation platforms usually just output a single numerical score, making it hard for the researcher to get this more nuanced insight into the performance of their policy. This paper proposes an approach by which the evaluation platform can respond to user queries and adaptively build a set of tasks to benchmark on that will answer the question.

(2) The evaluation system seems well engineered and complete from the paper's description of it.

(3) Simulated scenes/assets usually fall prey to difficulties of scale, spurring recent work in procedural generation. This paper's evaluation platform can viewed as another instance of procedural generation, which makes it more scalable.

### Weaknesses
(1) While the proposed simulation benchmark can foreseeably be useful to researchers for model development and debugging, from my understanding it will be difficult to use it as a *benchmark*. Because the benchmark is not static, and the scenes/tasks change based on user queries, it becomes difficult to compare various methods objectively. Further, even if the same prompt is used (i.e., "evaluate how good my policy is on pick and place tasks") the AI-powered evaluation platform may be non-deterministic and thus generate different scenes for different policies, again preventing comparisons.

(2) While the idea is interesting, the premise that a new simulation evaluation platform should be built at all is in my opinion problematic. There are several simulation benchmarks for robotic manipulation, and much work has tried to develop robotic policies that push the frontier of these benchmarks. However there is no guarantee that good performance on these simulation benchmarks translates to good real world robotic performance (which generally is what the robotics community is aiming for), and so a new simulation benchmark misses the forest for the trees, and promotes the optimization of a metric that does not necessarily have great correlation with the fundamental scientific questions researchers are interested in answering -- how well do policies do on real-world tasks. In my opinion there is still room for simulated evaluation benchmarks, like hyper-realistic benchmarks that very accurately model real world physics/visuals, but this work does not tackle that problem.

(3) Table 2 shows how well the ranking of policies produced by the proposed benchmark correlate with rankings of the same policies on previous simulation benchmarks. The correlation does not seem to be very strong, and even if it was, correlation with prior simulations shouldn't even be a metric to optimize, because there is no evidence that those prior simulations correlate with the real world.

(4) The motivation that time can be saved with the proposed benchmark is not a great motivation. Simulation evaluation platforms are generally considered to be cheap (relative to real-world evaluations, hence their appeal), and a drop in evaluation time from 2 hours to 45 minutes isn't very substantial.

### Questions
(1) How can the proposed agentic simulated evaluation system be used as a benchmark, which as far as I understand is the goal of the work (see weakness (1))?

(2) Table 3 shows how often the evaluation platform fails to evaluate the policy (e.g., due to AI issues). While it doesn't fail often, which is good, when it does fail what should the user do to get their policy evaluated?

(3) The evaluation makes use of VQA models in various parts -- what happens if the VQA model makes mistakes? How does this affect the set up of the evaluation and the results? Can the results still be useful to researchers if the VQA model can make mistakes?

### Soundness
2

### Presentation
2

### Contribution
2
