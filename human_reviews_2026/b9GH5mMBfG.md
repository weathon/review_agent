# RoboPARA: Dual-Arm Robot Planning with Parallel Allocation and Recomposition Across Tasks

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Dual-arm robots play a crucial role in improving efficiency and flexibility in complex multitasking scenarios. While existing methods have achieved promising results in task planning, they often fail to fully optimize task parallelism, limiting the potential of dual-arm collaboration.
To address this issue, we propose RoboPARA, a novel large language model (LLM)-driven framework for dual-arm task parallelism planning.
RoboPARA employs a two-stage process: (1) Dependency Graph-based Planning Candidates Generation, which constructs directed acyclic graphs (DAGs) to model task dependencies and eliminate redundancy, and (2) Graph Re-Traversal-based Dual-Arm Parallel Planning, which optimizes DAG traversal to maximize parallelism while maintaining task coherence. In addition, we introduce the Cross-Scenario Dual-Arm Parallel Task dataset (X-DAPT dataset), the first dataset specifically designed to evaluate dual-arm task parallelism across diverse scenarios and difficulty levels. Extensive experiments demonstrate that RoboPARA significantly outperforms existing planning methods, achieving higher efficiency and reliability, particularly in complex task combinations.Our code is publicly available at https://github.com/AiDuanshiying/RoboPARA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes RoboPARA, a framework for using large language models (LLMs) for dual-arm task parallelism planning. The framework employs a two-stage process: (1) Dependency Graph-based Planning Candidates Generation and (2) Graph Re-Traversal-based Dual-Arm Parallel Planning. Highlight of the proposed framework is constructing and traversing directed acyclic graphs (DAGs) that model task dependencies and eliminate redundancy. The authors also introduce the Cross-Scenario Dual-Arm Parallel Task dataset (X-DAPT dataset), designed to evaluate dual-arm task parallelism across diverse scenarios and difficulty levels. Extensive empirical results are provided to demonstrate that RoboPARA significantly outperforms existing planning methods, especially in complex task combinations and in requiring less execution time than baseline methods while achieving similar task performance.

### Strengths
1. The paper has high presentation quality and easy-to-follow writing. The appendix section provides extensive details on various aspects of the method details and experiment settings. 

2. Real world experiments span multiple robot setup and various tasks, which speak for the generality of the proposed framework and applicability to real robot systems. 

3. The proposed framework achieves clear reduction in execution time while maintaining task success. Comprehensive experimental results and dedicated efforts to re-implement and compare against various baselines, which had different assumptions and task settings in the original works.

### Weaknesses
1. Lack of video results. Although various figures throughout the manuscript show task setups and robot configurations, a video of actual task completion will be much more convincing and conveys more clarity on the task settings. 

2. Minor grammar issues, e.g. lack of spacing (Line 1834, 1835, RoboPARAsuccessfully, RoboPARAdemonstrates). 

3. The proposed framework treats bimanual manipulation as a planning problem over two arms, but concurrently there is also many efforts on learning end-to-end policies from bimanual teleoperation data. It is unclear whether the two-arm planning approach is still needed or would need a major modification when bimanual policies can easily leverage two arms at once to perform many complex task motions.

### Questions
1. The authors mentioned "GPT-4o and DeepSeek V3 APIs incurs significant costs" -- besides financial costs, does using the LLM APIs incur delays in between queries, and does that pose a problem for smooth robot action execution? 

2. What kind of inaccuracies do the robot agents struggle to self-correct or re-plan? Again, providing video results would be very helpful.

### Soundness
2

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
This paper addresses the problem of parallelism optimization in dual-arm robot task planning. It introduces a two-stage architecture that combines LLM-based dependency graph generation with graph re-traversal-based parallel planning. The proposed method achieves state-of-the-art quantitative performance and is validated on a real dual-arm robotic platform. The paper also presents a benchmark dataset for evaluating parallelism in dual-arm manipulation tasks.

### Strengths
1. While minimizing task time or improving efficiency in dual-arm robots is not a new problem, the paper’s approach is novel in leveraging the semantic reasoning capability of LLMs to extract task dependency graphs from complex multitasking instructions to maximize parallelism. 
2. Once released, the proposed dataset could make a meaningful contribution as a dual-arm task dependency dataset, potentially serving as a valuable benchmark for studying parallel manipulation and LLM-based task planning. (Note: since the dataset is not yet released and details such as its size and format are missing, this strength is currently conditional.)
3. The experimental design and analysis are appropriate, and the framework is successfully deployed on a real dual-arm robot, demonstrating practical feasibility.
4. The problem formulation is clear, the paper is well organized, and the visualizations are strong and intuitive.

### Weaknesses
1. While the paper includes a DAG validation and correction step to ensure logical consistency, it seems that this process lacks deep physics-aware validation. Crucial physical constraints like spatial conflicts, reachability, and precise timing are not rigorously checked during the initial DAG generation or correction phase, potentially leading to logically sound but physically infeasible plans that might only be detected later during execution or scheduling.
2. The paper introduces a rollback mechanism in Stage 2 to resolve deadlocks arising from parallel execution attempts. While it is important, the paper lacks in-depth analysis on how frequently this rollback was triggered during experiments and its associated performance cost (e.g., increased makespan compared to an ideal parallel plan). This lacking of analysis makes it difficult to fully assess the real-world efficiency impact of the deadlock resolution strategy.
3. The paper does not provide a formal or empirical analysis of computational complexity as the number of DAG nodes increases. Since both graph construction and re-traversal rely on LLM-based reasoning, the computational cost may grow quadratically or even combinatorially with larger task graphs. While the authors mention scalability, no details about timing or efficiency results are discussed. An discussion or ablation on graph size scalability would strengthen the paper.
4. The paper shows that RoboPARA’s components are necessary, but it does not fully disentangle the effects of Stage 1 and Stage 2. The analysis focuses on outcomes (e.g., increased parallelism) rather than underlying causes, and it is not specified whether the improvement in Stage 2 comes from enhanced LLM self-consistency or simply from iterative refinement.

### Questions
1. If one arm fails during task execution, how does the system respond? Does it restart the full plan or perform immediate rescheduling?
2. Can the proposed method be extended beyond dual-arm setups, for example, to systems with three arms or super-limb robots?
3. What LLM model was used for dataset construction, and how many human annotators were involved in verifying the structures?
4. What is the API usage or token cost associated with the full planning pipeline?
5. What is the total planning time, including both dependency graph construction and scheduling optimization?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces RoboPARA, a novel framework for dual-arm robot task planning that prioritizes the optimization of task parallelism. The authors formulate a new problem, the "Dual-Arm Cooperative Scheduling Problem", which aims to minimize the makespan (total execution time) by effectively scheduling tasks across two arms. The proposed RoboPARA framework operates in a two-stage process : first, it uses an LLM, augmented with a memory module (RAG) , to generate a Directed Acyclic Graph (DAG) representing task dependencies. Second, a "Graph Re-Traversal" algorithm, which is non-LLM based, optimizes this DAG to schedule operations, maximize parallel execution, and resolve conflicts.

To evaluate this framework, the authors also introduce the "Cross-Scenario Dual-Arm Parallel Task dataset" (X-DAPT) , a new benchmark containing over 1,000 tasks across 10 scenarios designed specifically to test dual-arm parallelism. Experimental results claim that RoboPARA significantly outperforms existing methods, achieving a 30% to 50% reduction in execution time and demonstrating superior parallel step execution, particularly in complex tasks.

### Strengths
The paper's primary strength is its formalization and direct confrontation of parallelism in dual-arm manipulation. While prior work often results in sequential execution , this paper defines a new, relevant problem ("Dual-Arm Cooperative Scheduling") and proposes a solution explicitly designed to optimize for it. This focus on decoupling tasks for parallel execution, rather than purely sequential or fully synchronous collaboration, is a significant and practical contribution.

The two-stage, hybrid architecture is a sound engineering approach. It leverages the LLM's strength in parsing human instructions and understanding task semantics to generate an initial plan (the DAG). It then wisely offloads the "System-2" optimization—a constrained scheduling problem—to a deterministic, algorithmic "Graph Re-Traverse" stage. This avoids the unreliability of using LLMs for complex, multi-step symbolic reasoning and allows for the implementation of robust logic, such as deadlock prevention and arm-lock compatibility checks .

Similar to the method, the X-DAPT dataset is a valuable contribution. The authors correctly identify that existing benchmarks often lack tasks with complex, inter-package dependencies that can be parallelized. By creating a dataset with multiple difficulty levels and a focus on long-horizon, multi-package scenarios , the authors provide a new and challenging testbed for the community to evaluate this specific dimension of robotic planning.

The paper is clearly written, and the framework is well-illustrated. The separation of Stage 1 (LLM-driven DAG generation) and Stage 2 (Algorithmic scheduling) is logical and easy to follow. The inclusion of detailed prompt templates in the appendix, while extensive, adds to the paper's transparency and reproducibility

### Weaknesses
The authors designed both the solution (RoboPARA) and the primary benchmark (X-DAPT) on which it demonstrates overwhelming superiority. The results in Tables 1 and 2 show that RoboPARA achieves high scores on the new parallelism metrics (PPR and APR), while all seven baseline methods score 0.000 or near-zero on these metrics in almost every single category. This result is suggesting that the X-DAPT benchmark is overtuned to the specific graph-based, parallel-aware architecture of RoboPARA. This suspicion is reinforced by the results on a neutral, third-party benchmark (a reorganization of the RoboTwin dataset) . On RoboTwin (Table 13), RoboPARA still performs best, but the gap is far less dramatic. For example, on "Easy" tasks, RoboPARA's efficiency (TEI) is 10.6, while two baselines achieve 8.0. This is a reasonable improvement, not the 10x+ gap seen in the X-DAPT parallelism metrics. The overwhelming superiority vanishes on a benchmark the authors did not design.

The paper's core claim is a 30-50% reduction in "execution time", which is commendable. However, the paper's own limitations section admits that RoboPARA "consumes an average of 1.3x more tokens than baselines due to iterative DAG corrections". This suggests a significantly higher planning latency before execution can even begin. For many real-world, dynamic tasks, a 30-second reduction in execution is not a net gain if it requires an extra 60 seconds of planning. The paper's primary metric, TEI, is defined as successful steps divided by total task completion time in seconds , which seems to include planning time, but this is not explicitly stated. This ambiguity around the planning-time vs. execution-time trade-off is a major omission.

The method's success in Stage 1 is heavily dependent on extremely detailed, hard-coded prompt engineering. The appendix reveals that "Template 2a: Dependency-Aware Graph Construction" contains 19 separate, highly specific rules (e.g., "Rule 10: The cutting operation can only begin when all objects are placed...") . This raises serious questions about the method's generality. It appears less like a general-purpose planner and more like a system purpose-built and meticulously tuned for the 10 scenarios in the X-DAPT dataset. It is unclear how this method would scale to a new domain (e.g., "car repair" or "lab automation") without a similarly exhaustive, manual-engineering effort to define all domain-specific dependency rules.

### Questions
1. Can the authors explain the 0.000 performance of all baselines on the PPR/APR metrics on X-DAPT? Does this not suggest the benchmark is built exclusively to validate the RoboPARA architecture, rather than to provide a fair comparison? Why is the performance gap so much smaller on the RoboTwin dataset?

2. How much manual effort is required to adapt the 19-point prompt template  to a completely new domain with new objects and dependencies? Does the reliance on such extensive, hard-coded domain knowledge not fundamentally limit the "LLM-driven" nature of the approach?

### Soundness
3

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
5

### Summary
This paper introduces RoboPARA, an LLM-driven framework for dual-arm parallel task scheduling in robotic manipulation, addressing the limitations of existing methods in exploiting temporal overlap across sub-tasks. RoboPARA adopts a two-stage pipeline: (1) dependency-graph-based candidate plan generation, and (2) graph re-traversal-based concurrent execution scheduling for dual-arm coordination. To evaluate parallel execution capabilities, the authors present X-DAPT, the first benchmark dataset dedicated to dual-arm temporal parallelism, encompassing diverse scenarios and task complexities. Extensive experiments demonstrate that RoboPARA outperforms baseline methods in terms of parallel step count, execution time reduction, and task success rate.

### Strengths
1.The two-stage architecture of RoboPARA effectively models task dependencies and optimizes dual-arm parallelism, fully exploiting the collaborative potential of dual-arm robots.

2.The X-DAPT dataset is the first dedicated to evaluating dual-arm task parallelism, covering diverse scenarios and difficulty levels, providing a comprehensive benchmark.

3.RoboPARA demonstrates excellent performance in experiments, with significant improvements in parallel steps, execution time reduction, and task success rate compared to baselines.

### Weaknesses
1.Limited generalization capability: RoboPARA relies on predefined skill libraries and scenario templates, requiring abstraction for novel tasks or environments. This hinders its scalability to long-horizon, multi-stage and complex tasks and out-of-distribution scenes.

2.Mismatch between optimization objective and real-robot deployment: The planning stage minimizes estimated action duration, but accurate execution latency is hard to obtain and estimate accurately in real-world settings. As a result, the optimized schedule may not be truly time-optimal on physical hardware, raising concerns about its perfect application in real-robot experiments.

3.Lack of safety considerations: Safety is a critical metric in dual-arm parallel task execution, but how to avoid collision problems during execution is not discussed. For example, the paper mentions that single-arm tasks are preferentially assigned to the left arm when both arms are idle, and it is questioned whether this is reasonable.

### Questions
1.Unclear details of real-robot experiment implementation: How to construct the Graph during real-robot experiments? Is it necessary to pre-abstract objects in the scene? Additionally, how are the execution times of different actions defined and determined?

2.Lack of exploration of Visual-Language Models (VLMs): The paper only uses LLMs for dual-arm task planning, while existing state-of-the-art VLM models have shown strong task planning capabilities. It is questioned whether attempts have been made to use SOTA VLM models for dual-arm parallel tasks, and if so, what their efficiency is and whether they can ensure safety during execution.

### Soundness
2

### Presentation
2

### Contribution
2
