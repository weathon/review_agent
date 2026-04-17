# EXP-Bench: Can AI Conduct AI Research Experiments?

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6

## Abstract
Automating AI research holds immense potential for accelerating scientific progress, yet current AI agents struggle with the complexities of rigorous, end-to-end experimentation. We introduce EXP-Bench, a novel benchmark designed to systematically evaluate AI agents on complete research experiments sourced from influential AI publications. Given a research question and incomplete starter code, EXP-Bench challenges AI agents to formulate hypotheses, design and implement experimental procedures, execute them, and analyze results. To enable the creation of such intricate and authentic tasks with high-fidelity, we design a semi-autonomous pipeline to extract and structure crucial experimental details from these research papers and their associated open-source code. With the pipeline, EXP-Bench curated 461 AI research tasks from 51 top-tier AI research papers. Evaluations of leading AI agents, such as OpenHands and IterativeAgent on EXP-Bench demonstrate partial capabilities: while scores on individual experimental aspects such as design or implementation correctness reach 20-35%, the success rate for complete, executable experiments was a mere 0.5%. By identifying these bottlenecks and providing realistic step-by-step experiment procedures, EXP-Bench serves as a vital tool for future AI agents to improve their ability to conduct AI research experiments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors present a semi-automated pipeline to convert research papers into tasks that evaluate if language models can conduct experiments like that in the original papers. They evaluate several agents on these tasks and present results along with an analysis of failures and successes.

### Strengths
* The task production process, barring some caveats below, is indeed fairly scalable which allows for a reasonably large set of tasks.
* The breadth of evaluations and the analysis of failures/successes are strong and valuable additions to the paper. 
* There are concerns that since these papers are available on the internet, agents may have memorized approaches to these problems, which significantly reduces the value of this benchmark without further evidence. However, the authors provide analysis that input ablations significantly hurt performance, which does help reduce this risk.

### Weaknesses
* The lightweight manual review process means that there are no guarantees that the LLM-generated research questions are not trivial. I additionally have concerns about the validity of the extraction and judging process given the lack of detailed review. Please see the questions section. Without further clarification or evidence, this represents a significant risk that the conclusions of the benchmark are not valid.
* It's fairly restrictive to require the agent use the specified method to answer the objective (and indeed, comparing code diffs and experimental design requires a pretty similar approach). Additionally, the instructions provided to the agent are also extremely specific, which makes the value of the benchmark less useful as perhaps agents could do much better by solving the research question via a different approach.

### Questions
* The description of stage 2.1 lacks detail. What are multimodal extraction techniques? Do you use an LLM-based approach? 
* I'm confused about how you are guaranteeing that the implementation in stage 2.2 is correct. You first extract the question, method and conclusion and provide this to the agent. The agent then finds a set of scripts from the original codebase and provides a plan of how to execute this to get the conclusion. At this point, is it allowed to modify the scripts? Is it allowed to do so if the script fails and it's asked to refine this? If so, how are you ensuring that there's no trivialization here (i.e. the agent modifies the implementation to make the conclusion obvious or trivial). This is concerning as this is what becomes ground truth. 
* How are the execution traces in Stage 3 checked against the expected conclusions? via an LLM judge of some sort? it's not clearly mentioned. 
* How could the tasks lack a matched implementation? It seems that your filtering process earlier only allows tasks to pass if Stage 2.2 finds a working implementation. Also, it's not clear if you check that the extracted objective faithfully matches the source paper in cases where you do have an implementation. This is concerning since the implementation was derived from the extracted objective. 
* Did you validate that the LLM judge performs accurately by manually labelling a small sample?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
There are two main contributions of this paper. The first is a semi-automated pipeline for producing such end-to-end AI research tasks based on high-quality conference papers, open-source code implementations, and lightweight human review. The second is a new dataset of 461 tasks built using that pipeline, which claims to be more comprehensive, more "end-to-end", and larger in total size than what exists currently. Several near-frontier models were evaluated using leading open-source agent scaffoldings, and the results show that agents tend to struggle with completely and correctly implementing their plans, as well as with development environment setup and configuration.

### Strengths
- building on existing work instead of reinventing everything from scratch. Using existing agent implementations and building on Inspect are doubt-reducing choices.
- The use of a structured workflow and the multi-pass retrieval for generating the tasks—instead of just trying to one-shot LLMs
- Including some amount of human review in the process, validating with a final human validation at the end of the task creation process
- The use of multiple metrics and looking at the distribution of scores across the metrics
- The error analysis was good. It's good to look at the data and explore where exactly agents are failing and what the bottlenecks might be.

### Weaknesses
I think the main weakness is that by splitting the paper's attention across the task-generation pipeline and the results of the agents on the tasks, there isn't enough space to deeply explore either. Perhaps my strongest recommendation is to reduce the scope of the paper and focus on either the results of the agents on the tasks--with detailed analysis of agent transcripts, error modes, possible false positives or false negatives--or on the task creation pipeline and validating that these tasks are high quality--especially by having humans attempt them. This may be a terrible suggestion, and it's almost certainly out of scope as a revision for this conference. But if I were to choose the thing that I think would most increase the quality of this paper, it would be to go more focused, more in-depth, and more thorough. This is the main reason for my low "soundness" rating, in that I don't think I have information as it is to _really_ trust the results.

It would be good to report results on each task, not just aggregated results.

### Questions
1. Did a human complete any one of the tasks that were generated by the pipeline?
2. How many of the agent traces/trajectories/transcripts were read in full by a human? 
    1. How many successful and unsuccessful attempts?
3. Was more open-ended cheating detection—beyond the fixed cheating criteria— conducted?
    1. For example, an LLM-based scatter running on the agent transcripts.
4. What was the purpose of the git diff included in the generated tasks? How was that used, if at all, in scoring?
5. Why was the agent environment created in such a way that the agent had the opportunity to cheat in such an easy way, like reading the paper?
    1. Were any of the runs that were flagged for cheating, and therefore excluded, reviewed to see if perhaps the agent had just misunderstood the instructions?
6. What ability, if any, did the agents have to check the validity of their submissions while working on them?
    1. Were the agents instructed to make sure that their setup scripts could be run in an entirely fresh environment to recreate all dependencies and configuration from scratch?
    2. Were they given an opportunity to test that their script worked?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces EXP-Bench, a benchmark for evaluating whether AI agents can autonomously perform end-to-end AI research experiments. It is built through a semi-automated pipeline that extracts structured tasks—each including a research question, high-level method, and starter code—from 51 NeurIPS/ICLR 2024 papers, resulting in 461 tasks.
Agents such as OpenHands and IterativeAgent are tested across metrics for design (D), implementation (I), execution (E), and conclusion (C). Results show partial competence (20–35% on sub-tasks) but extremely low full-task success (~0.5%). The work highlights key failure patterns and offers a scalable framework for assessing and improving autonomous research capabilities in LLM-based agents.

### Strengths
Originality: The paper introduces a benchmark, EXP-Bench, targeting a rarely studied but crucial problem — evaluating AI agents’ ability to perform complete research experiments. This “end-to-end scientific experimentation” framing goes beyond existing reasoning or coding benchmarks, representing a clear conceptual advancement.
Quality: The proposed semi-automated curation pipeline is technically well-motivated and methodologically sound. It combines multimodal extraction, code analysis, and execution-based validation to ensure high-fidelity, reproducible tasks.
Clarity: The paper is clearly written and well-structured. Figures and tables effectively illustrate the dataset construction process, evaluation metrics, and key failure modes.
Significance: EXP-Bench provides a large-scale and realistic testbed (461 research tasks) for assessing LLM-based research agents.

### Weaknesses
1.Reliability of LLM-as-a-Judge evaluation: The benchmark relies exclusively on an LLM-based judge to assess design and conclusion correctness, without any reported human calibration. This raises concerns about evaluation reliability and potential self-consistency bias, since the same modeling paradigm being evaluated also defines the scoring criteria. Including a limited human cross-check or reporting human–LLM agreement statistics would make the results more credible.
2.High computational and API cost.
Running the full benchmark requires substantial API and compute expenses, making it difficult to reproduce or extend. The authors could consider releasing a lightweight subset (similar to SWE-bench Lite or PaperBench Code-Dev) to facilitate community adoption, debugging, and fast prototyping.
3.Lack of resource-aware task curation.
Although the appendix reports per-task hardware and runtime requirements, these resource metrics were not integrated into the benchmark’s task selection pipeline. Incorporating resource-awareness (e.g., filtering by expected GPU hours or memory usage) could improve fairness and scalability when running under standardized Docker environments.

### Questions
1.Evaluation strictness vs. creativity:
The benchmark enforces ground-truth matching for design and implementation, but AI agents may propose valid yet creative experimental variants that differ from the reference. How does the evaluation handle such alternative but scientifically sound approaches? Could this strictness penalize genuine innovation?
2.Choice of evaluated models:
Have the authors considered testing open-source, smaller-scale LLMs (e.g., Qwen3) to provide a more accessible baseline for the community? Or do such models perform too poorly to yield meaningful results?

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
The paper contributes a benchmark called EXP-Bench that consists of tasks from 51 papers publised at Computer Science conferences. Tasks include implementing experiments from these papers given a high-level research plan and question.

### Strengths
- The paper builds on previous work in evaluating ai agents on coding tasks in ML research well. While other papers have focused on reproducing scientific papers, the tasks in EXP-Bench go one level further and involve implementation of experiments given a high-level outline

### Weaknesses
- Missing citations to directly relevant outstanding work [1]
- Number of source papers on which tasks have been generated is rather low. 
- Set of chosen models are outdated and do not include latest agentic coding models (GPT-5, Claude Sonnet 4.5, etc.). Even if the contamination problem exists, an analysis of how well models memorize the papers part of the benchmarks could be a good sanity check
- No human error analysis


[1] https://arxiv.org/abs/2409.11363

### Questions
- How robust is the LLM-based error analysis when doing human review?

### Soundness
3

### Presentation
3

### Contribution
3
