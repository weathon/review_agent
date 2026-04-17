# Starjob: Dataset for LLM-Driven Job Shop Scheduling

- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
Large Language Models (LLMs) have shown remarkable capabilities across various domains, but their potential for solving combinatorial optimization problems remains largely unexplored. In this paper, we investigate the applicability of LLMs to the Job Shop Scheduling Problem (JSSP), a classic challenge in combinatorial optimization that requires efficient job allocation to machines to minimize makespan. To this end, we introduce Starjob, the first supervised dataset for JSSP, comprising 120k instances specifically designed for training LLMs. Leveraging this dataset, we fine-tune the LLaMA 8B model with the LoRA method to develop an end-to-end scheduling approach. Our evaluation on standard benchmarks demonstrates that the proposed LLM-based method not only surpasses traditional Priority Dispatching Rules (PDRs) but also achieves notable improvements over state-of-the-art neural approaches like L2D, with an average improvement of 11.28% on DMU and 3.29% on Taillard benchmarks. These results highlight the untapped potential of LLMs in tackling combinatorial optimization problems, paving the way for future advancements in this area.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper explores the applicability of Large Language Models (LLMs) to solving the Job Shop Scheduling Problem by fine-tuning Llama-3.1 8B using the RsLoRA method. It introduces a supervised dataset containing scheduling instances encoded in natural language along with their (near-)optimal solutions generated using Google's OR-Tools. The method is evaluated on the JSSP Taillard and DMU benchmarks.

### Strengths
The sizeable supervised dataset for JSSP could prove valuable for the research community focused on applying LLMs to scheduling problems.

### Weaknesses
1. This paper lacks an overview of recent work on using LLMs for solving combinatorial optimization problems. The paper emphasizes that it "investigates the applicability of LLMs to the Job Shop Scheduling Problem (JSSP)"; however, there are recent studies that have demonstrated how LLMs can be used to solve scheduling problems through either by supervised fine-tuning, interaction with solver, among other approaches.
Jiang, X., Wu, Y., Li, M., Cao, Z., & Zhang, Y. (2025). Large Language Models as End-to-end Combinatorial Optimization Solvers. arXiv preprint arXiv:2509.16865. (see Table 1)
Thind, R., Sun, Y., Liang, L., & Yang, H. (2025). OptimAI: Optimization from Natural Language Using LLM-Powered AI Agents. arXiv preprint arXiv:2504.16918. (Table 5).

2. The baselines used for evaluation are not strong. Currently, the fine-tuned LLM is compared only with PDR, L2D, and RASCL. However, there are multiple other methods based on deep reinforcement learning that solve JSSP with good performance. See the following paper on benchmarking JSSP for learning-based methods:
Reijnen, R., van Straaten, K., Bukhsh, Z., & Zhang, Y. (2023). Job shop scheduling benchmark: Environments and instances for learning and non-learning methods. arXiv preprint arXiv:2308.12794.

### Questions
1. The Llama-3.1 8B model is fine-tuned using the resource-efficient RsLoRA method. However, the training and inference times are not provided to determine whether the resource-efficient RsLoRA is actually beneficial for the scheduling case.

2. Section 4.1 mentions that there is an OOD test set; however, the results on this test set are not given. the result section shows evalution on benchmark dataset only.

3. Figure 1 is not referenced in the text.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the job shop scheduling problem using a large language model (LLM). The authors generated 130,000 instances for training and fine-tuned a Llama-3.1 8B model using the RsLoRA method. They compared the performance with several dispatching rules and two learning-based approaches.

### Strengths
The dataset with 130,000 instances could be valuable for future research.

Attempting to solve job shop scheduling directly with an LLM is interesting and may serve as a foundation for future work on LLM-based combinatorial optimization.

### Weaknesses
The paper’s novelty and contribution need to be better clarified. Simply applying an LLM to a basic job shop scheduling problem is not enough to justify the claim of addressing constraint-heavy domains.

The authors only compare against dispatching rules and two learning-based methods. There are recent learning-based studies that achieve stronger performance on job shop scheduling. Comparison to more recent and state-of-the-art methods is necessary.

### Questions
1. What is the main motivation for using an LLM to directly solve job shop scheduling, rather than combining LLMs with existing optimization or RL frameworks? Please clarify what potential advantages LLMs provide in this domain.

2. Many job shop variants exist (e.g., setup times, machine flexibility, ready times, sequence-dependent constraints). How scalable is the proposed approach to such variants? Would it require retraining or collecting new datasets for every constraint combination or different objective functions?

3. Please explicitly state the main contributions compared to prior works.

4. The authors mention that the goal is not to outperform specialized schedulers but to demonstrate a systematic application of LLMs to job shop scheduling. In this context, please clarify how the proposed method could be extended or generalized to other types of scheduling problems (not just job shop).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper explores how LLMs can be applied to JSSP. By fine-tuning the LLM using RsLoRA, the model outperforms traditional heuristics (e.g., PDRs) and neural baselines (L2D, RASCL) on Taillard and DMU benchmarks, showing LLMs’ potential in tacking NP-hard problems.

### Strengths
(1)	The idea on using LLM to directly solve JSSP is novel.

(2)	The method outperforms traditional heuristics (PDRs) and neural baselines (L2D, RASCL) on Taillard and DMU benchmarks, improving average solution gaps by up to 17.36% and 7.85% respectively.

### Weaknesses
(1)	The motivation on why using LLMs for JSSP is not clearly given. Indeed, the application of LLMs for JSSP remains unexplored, but why people need LLMs for JSSP? Users can also use well-developed heuristics or solvers, which are more lightweight and effective.

(2)	The methodological contribution is not significant. The core move, textualizing JSSP and fine-tuning an 8B LLM, extends recent “LLMs as optimizers” trends; the paper doesn’t clearly separate what’s new from known SFT+inference pipelines.

(3)	Despite better performance, using LLMs to generate solutions can be generally much more computationally intensive. The paper set S=20 to generate 20 responses at the same time, which leads to more inference time. It is also not clear why the inference time is not reported and compared in Table 1-2.

(4)	The baselines only include simple heurtistics and two early learning-based methods (before 2023). More recent learning-based methods (2024-2025) should also be discussed.

(5)	Average gaps still hover around ~20% on DMU and ~22% on Taillard; improvements over RASCL are tiny or dataset-dependent (on DMU the averages are essentially tied). The paper’s own tables show this. The gains don’t seem to justify the training/inference cost.

### Questions
(1)	After fine-tuning the model, is that possible to generalize the model to other scheduling problems except JSSP?

(2)	Have the authors explored other natural language representation for JSSP instances?

### Soundness
2

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
4

### Summary
This paper introduces Starjob, a 130,000-instance supervised dataset for training LLMs to solve the Job Shop Scheduling Problem (JSSP), with the stated goal of demonstrating that LLMs can generate feasible solutions for NP-hard combinatorial optimization problems. The authors fine-tune a Llama-3.1 8B model using RsLoRA on this dataset, employing a natural language representation with explicit summation notation (e.g., "J2-M0: 0+78 → 78") that leverages the autoregressive generation process to maintain constraint satisfaction. The model achieves a 17.36% gap improvement over L2D on the DMU benchmark and 7.85% on Taillard, outperforming four traditional Priority Dispatching Rules (SPT, MWKR, FDD/WKR, MOPNR) and neural baselines L2D and RASCL. The approach requires generating 20 candidate solutions per instance and selecting the best feasible one, with feasibility rates varying from approximately 10% to 30% depending on problem size as shown in Table 3. The method also enables natural language interaction, allowing users to query the scheduler about scheduling constraints and bottlenecks.

### Strengths
1. The paper follows a logical progression from problem formulation through dataset construction, methodology, and evaluation, making it easy to follow. 
2. The paper provides many illustrative examples that help readers understand the core dataset design, including Listing 1 showing the natural language encoding of a 3×3 JSSP instance, Listing 2 demonstrating the solution format with explicit summation notation, and Listing 3 showcasing the interactive query capability where users can ask about scheduling bottlenecks. 
3. The method achieves meaningful improvements over both traditional and neural approaches, with average gap percentages of 21.69% on Taillard and 20.14% on DMU benchmarks compared to L2D's 29.54% and 37.50% respectively.

### Weaknesses
The main focus of this paper, demonstrating that LLMs can generate feasible solutions for NP-hard optimization problems, is interesting and timely. While the authors primarily focus on polishing the dataset design, I have several concerns about the execution and scope of this work.

The paper's central assertion that "LLMs can generate feasible solutions for NP-hard problems" is inadequately supported by evaluating only on JSSP. To convincingly demonstrate this capability, the approach should be tested on other NP-hard problems with more complex constraints. Within scheduling alone, Flexible Job Shop Scheduling Problem (FJSSP) presents additional machine flexibility constraints, while routing problems like Vehicle Routing with Time Windows (VRPTW) would test different constraint types entirely. Most critically, Table 3 shows feasibility rates ranging from ~10-40%, but the paper never explicitly reports overall feasibility rate improvements compared to baselines, which is essential for a paper claiming to help LLMs generate feasible solutions.

If the dataset is the major contribution (as suggested by the title "Starjob: Dataset for LLM-Driven Job Shop Scheduling"), more rigorous comparisons are needed. The paper only compares their summation-based format against a matrix format in Table 3, but doesn't evaluate simpler alternatives like plain text descriptions without the summation notation. Additionally, while Figure 4 shows makespan statistics across problem sizes, the paper lacks detailed analysis of dataset diversity, solution quality distribution.

The layout may impact readability. For instance, Tables 1-2 appear on pages 8-9 while their discussion is on page 7, and Figure 1 comparing all benchmark results is relegated to page 9. This forces constant back-and-forth referencing that disrupts the reading flow.

### Questions
See the weakness part.

### Soundness
2

### Presentation
2

### Contribution
2
