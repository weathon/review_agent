# From Reproduction to Replication: Evaluating Research Agents with Progressive Code Masking

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Recent progress in autonomous code generation has fueled excitement around AI agents capable of accelerating scientific discovery by running experiments. However, there is currently no benchmark that evaluates whether such agents can implement scientific ideas when given varied amounts of code as a starting point, interpolating between reproduction (running code) and from-scratch replication (fully re-implementing and running code). We introduce AutoExperiment, a benchmark that evaluates AI agents’ ability to implement and run machine learning experiments based on natural language descriptions in research papers. In each task, agents are given a research paper, a codebase with key functions masked out, and a command to run the experiment. The goal is to generate the missing code, execute the experiment in a sandboxed environment, and reproduce the results. AutoExperiment scales in difficulty by varying the number of missing functions $n$, ranging from partial reproduction to full replication. We evaluate state-of-the-art agents and find that performance degrades rapidly as $n$ increases. Agents that can dynamically interact with the environment (e.g. to debug their code) can outperform agents in fixed ``agentless'' harnesses, and there exists a significant gap between single-shot and multi-trial success rates (Pass@1 vs. Pass@5), motivating verifier approaches to our benchmark. Our findings highlight critical challenges in long-horizon code generation, context retrieval, and autonomous experiment execution, establishing AutoExperiment as a new benchmark for evaluating progress in AI-driven scientific experimentation. Our data and code are open-sourced at https://github.com/j1mk1m/AutoExperiment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents AutoExperiment, a benchmark to evaluate LLMs and agents’ abilities to complete some functions in codebases that are necessary to replicate ML research papers’ results. With 85 functions adapted from four papers, the authors conduct controlled experiments by varying the number of functions to be implemented. Results show that agent performance drops as more functions are masked for implementation.

### Strengths
1. The proposed benchmark grounds its tasks on replicating real-world machine learning research papers.
2. The authors conduct a series of informative experiments, including controlled evaluation by masking different numbers of functions, exploring potential test-time scaling with verifiers, and trying different agent scaffolds.
3. The paper is clearly presented and easy to follow.

### Weaknesses
1. This paper proposes function masking and implementation as a new evaluation setting. However, it is questionable whether this is indeed necessary compared to related benchmarks (SciCode, ScienceAgentBench, PaperBench) that require LLMs to generate programs from scratch? Function completion essentially makes these research coding tasks less open-ended and easier by nature, which may fail to reflect real-world research coding.
2. As a benchmark, this work only involves 85 functions from four papers, which is too narrow and can be biased for evaluating LLMs and agents.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces "AutoExperiment", a benchmark for evaluating AI agents for conducting scientific replication tasks. The benchmark consists of 85 functions that AI agents must "fill in" across 4 machine learning papers. The ML papers are drawn from MLRC. The benchmark is evaluated using ReAct (though other strategies are tried and work more poorly) across a number of closed and open-weight AI models. The authors draw compelling insights from this analysis.

Overall, the ability of AI to reproduce scientific work is an important research problem. The authors do a good job of constructing the benchmark, running the evaluations, and several ablations. 

My main concern is about the small size of the benchmark. I appreciate that even with 85 missing functions, the benchmark is computationally expensive to run. But 4 papers seems like too small a sample size to conduct a benchmarking exercise. I would be more excited about a benchmark with 10+ papers, even if the number of missing functions per paper is smaller.

### Strengths
- ***Solid compute setup***: The computational setup for running the benchmark is decent, and well beyond what many other papers on scientific reproduction using AI agents use. 
- ***Focus on reproducible results (such as by using Docker)***: I appreciate the authors' efforts to make their study reproducible, including using Docker, and making all of their code and analysis openly available.
- ***Qualitative analysis***: Many points in the paper include some qualitative analysis, which is quite insightful. For example, the authors analyze whether the first try passes or fails to explain what causes the difference between the fixed and dynamic settings.
- ***Nice ablation***: Many decisions taken in the paper are supported by strong experimental evidence. For example, the decision to run experiments with varying amounts of context about the paper (No vs. Full) is nice. (Are there intermediate versions of this you could run? Just providing the abstract/intro of the paper? Also, does this raise concerns about leakage, since instead of solving the paper's tasks, the agent could simply "answer" the questions based on the results in the paper?)

### Weaknesses
- In the introduction, there's a typo ("agentive", should this be "agentic"?)
- No train-test split: As far as I understand, there's no "train" split in the data. How should researchers using the benchmark optimize and develop their agents without a train set? This raises concerns about leakage (even in the results reported in the paper, since there are many ablations, for example, in the ReAct agents)
- No reporting of cost, time taken by agents (which can be important for real-world use)
- Why pick these 4 papers? MLRC has dozens of papers from the last few years, is there a reason these 4 specific papers were picked?
- It would be nice to see pass^k in addition to pass@k: Papers like TauBench report pass^k (whether all k samples correctly solved the task). This could be interesting to see the reliability of the agents.
- The verifier used in the paper is quite simple. The authors could test with better verifiers to see if pass@k performance can be distilled to pass@1 (though it's fine to leave this for future work).
- How did the authors decide on the 5% tolerance for declaring a test case correct? Did you use the human replication to come up with this tolerance level?
- It's unclear where the curtailed tests are used and where the full tests are used
- It's unclear if all models are evaluated on the same 100 examples for n>1, or whether these are randomly sampled each time. If the latter, we might need CIs for more reliable results.

### Questions
Overall, I think this paper is a decent contribution. However, the small sample size, the lack of a test set, and lack of clarity about my questions above leads me to propose a borderline score for the paper. That said, I would be happy to increase my score if the following points are addressed:

1) The authors commit to adding a held-out test set (perhaps with 4 other MLRC papers). Of course, I realize the authors can't run experiments on this test set. That said, it would be nice to have a held-out set and functions that go beyond the ones that agent developers might use for "training"/developing their agents. It would also make the paper's results more reliable, since n=4 is a very small sample size.

2) The authors address the weaknesses I mention above, in particular to improve the clarity of the paper. (I understand if some of these weaknesses are hard to address, such as reporting cost and time taken after the fact, but I hope the others are easier to address.)

### Soundness
4

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
The authors propose a new benchmark, AutoExperiment, that tests if AI agents can implement ML experiments described in research papers. The benchmark is based on masking out functions for the agent to fill in (scalable by # functions masked). They evaluate various agents and find performance degrades with # fns masked, and that self-reflection helps.

### Strengths
- Appropriate and opportune benchmark - scientific coding is rapidly becoming a major research area, and we still lack good benchmarks
 - Clear and well written
 - Builds on prior work (MLRC) nicely
 - Nicely scalable levels of difficulty in the benchmark

### Weaknesses
- Most significantly, this paper seems very similar to "ResearchCodeBench: Benchmarking LLMs on Implementing Novel Machine Learning Research Code" (https://arxiv.org/abs/2506.02314), significantly weakening the paper's claims of novelty. Please describe the key differences/benefits of your work to this prior paper.
 - The stated findings from the experiments seem obvious (e.g., performance degrades with more masking; debugging helps). What surprising/discoveries did you make in your experiments? What did you learn that was not obvious before? The most significant take-away to me is that agents still struggle with this kind of coding (e.g., most scores are < 50%) and more work needs to be done.

### Questions
See Weaknesses. Also:
 - In what way is "scientific" coding different to general coding, or is it the same thing? Are there some specific characteristics of coding that your benchmark captures that other coding benchmarks do not?
 - Is it always a clear, single gold answer to the experiment? In particular, NL descriptions of an experiment are typically underspecified, meaning there might be several ways of implementing an experiment with different design decisions (e.g., choice of hyperparameters), and correspondingly different results. 
 - Filling in missing functions seems a somewhat artificial context - in reality, the agent simply has to code up an experiment. Do you think scores on filling in missing functions correlate with scores on coding up an experiment from scratch (the metric we ultimately care about)? Or do your experiments substantiate that correlation?
 - Your scores on AutoExperiment show models still struggle (~< 50%), but other coding benchmark tests have shown agents are strong at coding (e.g., HumanEval, SWE-Bench). How do you explain this apparent inconsistency? 

Minor:
 - Expand the MLRC acronym (Section 2) at first mention, not later.
 - I'd prefer the Related Work earlier (Section 2) rather than an afterthought at the end, given the considerable related work in this area

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
4

### Summary
This paper propose a new benchmark——AUTOEXPERIMENT，for evaluating the ability of agents to implement ml experiments. Tasks are created by masking out various numbers of functions from ML research repo codebases and requiring agents to fill in the blanks and match published results. The paper also evaluates several LLMs, presenting detailed analyses of different aspect.

### Strengths
1. The paper proposes a new benchmark for evaluating the performance of agents on machine learning code replication tasks.
2. It evaluates the performance of different models and agents under varying task settings (i.e., partial code replication and full replication).

### Weaknesses
1. Novelty of the Benchmark: The benchmark is presented in two settings: partial code replication and full code replication. For full replication, a significant body of related research already exists. The partial replication setting closely resembles many existing code-filling tasks studied in the context of code models.
2. Rigorousness of Experimental Setup: The paper uses the number of masked functions (n) as its primary analysis target and difficulty metric, concluding that a larger 'n' decreases the likelihood of successful replication. This conclusion is trivial and has already been well-established in numerous code-filling tasks. In practice, different functions vary dramatically in logical complexity, code length, and library dependencies. Using 'n' as a universal measure of difficulty is overly simplistic.
3. Lack of Diverse Experimental Subjects: The paper's main results are derived from models like GPT-4o and Claude-3.7, which have known capability limitations. As of 2025, many more powerful reasoning models have emerged that demonstrate superior performance on software engineering tasks, but this paper fails to evaluate them. Moreover, the agent setup itself is rudimentary, employing only a simple tool-using agent designed by the authors. This setup does not reflect the current state-of-the-art in agent capabilities and appears to be testing the backbone model more than the agent's autonomous abilities.
4. Reasonableness of the Benchmark Design: The paper states that with an increasing number of masked functions, the 85 samples from just 4 papers can lead to 275,990 possible combinations for n=5. The methodology of sampling only 100 of these combinations offers no guarantee that these samples are representative of that difficulty level. The design of this benchmark appears to be sloppy and lacks statistical rigor, as this sparse sampling introduces massive variance, making comparisons between different 'n' levels unreliable.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
2
