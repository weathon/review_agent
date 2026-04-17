# Beyond Software Development: Continuous Integration with coding agents

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Coding agents are popular aids for software development today. Starting from code completion mechanisms in GitHub co-pilot, they have evolved much beyond programming to be active aids for software development by managing and maintaining a code-base via issue resolution. Resolving software issues takes care of program improvement tasks such as bug fixes and feature additions. 
Yet they do not contribute to the integration and operationalization of such changes into a complex software project. 
In this work, we study technical challenges that maintainers will face with integrating AI-generated suggestions such as build-problems and testing software systems. We compare how a variety of existing software engineering agents can cope with such operationalization and systems integration.  This includes the open-source agent \texttt{OpenHands}, an existing agent to help in project builds called \texttt{ExecutionAgent}, and USEAgent, a general purpose SE agent ensemble.  Furthermore, we introduce USEAgentPlus, an enhanced agentic system that extends prior agents by incorporating dedicated tools for environment management. This enables effective solutions for systems-integration tasks and facilitates experimental comparison. Our results suggest that the proposed solution outperforms existing approaches, achieving success on diverse open-source projects evaluated against a standard software engineering research benchmark.
At a broad level, our work contributes in taking the automation offered by AI agents to the next stage of the software lifecycle - from  software development and maintenance to software systems integration.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses the gap between AI-generated code patches and their practical integration into real software projects. While existing coding agents can produce code changes, they often fail to ensure these changes can be successfully built and tested in the target environment. The authors propose USEAgentPlus, an enhanced multi-agent framework that introduces environment probing and a self-reflective advisor mechanism to improve system integration. Evaluated on 50 open-source projects and CI configuration tasks, USEAgentPlus outperforms baselines in building projects and executing tests, demonstrating its potential to advance AI agents from code generation to continuous integration.

### Strengths
1. The authors extend the focus from "development" to "integration," explicitly framing "Continuous Integration with Agents" as a more challenging and realistic task that closely mirrors real-world software maintenance workflows. This addresses a critical gap between AI-generated code and production deployment.

2. The paper presents USEAgentPlus, an enhanced agent framework centered on environment probing and a self-reflective advisor agent, offering a practical architectural improvement for complex system tasks.

3. The evaluation is conducted across multiple dimensions: automated build and test execution on 50 real-world open-source projects, validation of bug-fixing performance on SWE-Bench Verified, and assessment of CI script generation quality—demonstrating thorough empirical validation.

### Weaknesses
1. The paper contains a few typos. The last sentence of the Introduction (Line 95) reads "suggesting its its," with a duplicated "its." Additionally, in the Related Work section (Line 153), a citation is not properly enclosed in parentheses.

2. Section3: Approach lacks clarity. As this is primarily an engineering-focused paper with limited theoretical contribution, the methodology section should provide detailed implementation and operational mechanisms of the two key components: the environment probing stage and the advisor agent. However, the paper does not sufficiently explain how the environment probing stage gathers system information, decides which packages to install, or processes the collected data. Similarly, it is unclear how the advisor agent synthesizes and reflects upon this information. Highly suggest that the authors adjust Figure 2 as it currently lacks intuitiveness. For instance, placing "task" at the top left corner to signify the starting point of the entire system would be helpful. Additionally, straightening and thickening the line from the “1)Meta-Agent” to “2)probing” would emphasize this connection for readers. Furthermore, some of the grey arrows in the bottom right corner are overlapping.

3. Experimental details are insufficient. Is USEAgentPlus built on an LLM-based agent system? If so, which base model was used? Or is it rule-based? Furthermore, for RQ1, one of the evaluation metrics is whether test scripts were executed. Given that test suites in projects often have multiple levels of complexity, providing a concrete example of a successfully executed test script—including its difficulty level—would better illustrate the system’s capabilities.

4. The experimental setup for SWE-Bench Verified lacks clarity. The paper appears to demonstrate that generated patches can be automatically merged into original projects while maintaining buildability—a significant advancement over prior generative SWE-Bench agents that often ignored integration feasibility. However, the absence of baseline comparisons in the SWE-Bench experiments weakens the persuasiveness of the results. Moreover, the definition of evaluation data points is not explained, does it refer to the successful build of the SWE project?

### Questions
See weaknesses 2, 3, 4.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper presents USEAgentPlus that builds on top of prior general purpose agents to tackle integration tasks. The authors highlighted several issues specifically with integration tasks in the era of software agents. The evaluation focuses on two major scenarios: 1) create bash script for environment building and test commands 2) compose CI configuration scripts to support the correct environment and tests. The results on these tasks demonstrate that USEAgentPlus achieves better performance compared with more general software agents.

### Strengths
- the paper tackles an important and previously underexplored area of software agents for environment management and integration
- the example and failure reasons are interesting and can be useful for future work to build on

### Weaknesses
- unclarity in the approach
	- after reading the approach section in detail it is still unclear to me 
	- the approach used in the paper seems to build on top of prior work (UseAgent)
	- however, the authors do not explain what are the contributions made to improve UseAgent (for example, in Figure 2, it would be great if the authors had highlighted what are the new components in this work compared to UseAgent)
	- furthermore, from the text, the introduced technique does not seem to add any novelty to the software agent space.
	- for example, the introduced advisor (i.e., llm-as-judge) has already been explored
- evaluation unclarity
	- many of evaluation settings and process are unclear
	- for example in the SWE-Bench verified setting the authors claim that "we find most data points (70.2%) are able to build and run tests", how is this computed? I don't believe SWE-Bench verified are not designed to test the build capabilities (as the environment is already provided)
	- it is also unclear how some of the baseline approaches were chosen. Why is OpenHands agent used in the CI task but not in the bash script creation task?
	- additionally why does the author not compare against the baseline UseAgent that the technique is built on top of for the bash script task?
	- the ablation results are also lacking:
	- the authors only evaluate the affect of different iterations but not any other components of the approach

Minor issues:
- the overuse of italics is a bit overwhelming, for example: in approach section, does "lost" or "must" really need to be italized? Small suggest for the authors is to reduce the amount of italics and keep only the important ones.

### Questions
1. What are the reasons for the different baseline techniques chosen for different tasks in this work? Additionally why was UseAgent not used for comparison in the bash script task given the technique builds on top of this prior work?
2. Please clarify what are the major contribution and improvements made in this work compared to UseAgent
3. Why did the tool limit the iteration to 3? It seems from looking at the results there is no drop in performance with even higher iterations

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
In this paper, the authors aim to push evaluation of coding agents to another stage. This paper evaluate coding agents on whether they could successfully integrate changes to code bases that they proposed themselves. The authors performed experiments on SWE Bench Verified (subset), and found that their proposed method

### Strengths
- This project aims to push another step of software development from reparis to integration.
- The motivation of the study is good.
- The authors proposed USEAgentPlus, which could be useful for the research community, and could be helpful and effective with integration.

### Weaknesses
There are substantial inconsistencies in the paper.
- OpenHands is discussed in the abstract but not evaluated in the paper's Table 1. Also no description is provided in the paper to tell the readers what is OpenHands.
- The authors mentioned 7 programming languages are studied, but this information never appears anywhere else in the paper.

There are also a lot of missing details.
- What base language models are used for the experiments? This is very important. 
- The authors presents USEAgentPlus's average costs, but didn't include this information for any baselines.
- What are the success rates of correctly solving the issues with the agents? In addition to reporting the results of integration, the authors should also report the SWEBench accuracy too.

Besides, the paper substantially lacks citations.
- OpenHands, ExecutionAgent are not cited.
- No discussion of how existing work handle integration of code repairs.

Overall, the paper quality is low, and seems to be written at haste.

### Questions
- What are the limitations of your work? 
- What are some potential future work based on your paper?

### Soundness
2

### Presentation
1

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
This paper studies how AI software engineering agents can assist maintainers during the integration phase of the software lifecycle. While much prior work has focused on patch generation and automating pull requests, real software production and maintenance involves much more than editing source files. Agents still struggle in two key phases: building a project and running its test suite.
To address this gap, the paper introduces USEAgentPlus, an automated system capable of managing environments — from installing dependencies to managing configurations and running tests. USEAgentPlus is built on top of USEAgent, a multi-agent ensemble framework, and extends it with three main components: (1) an environment probing stage, (2) a self-critique mechanism for task description, and (3) a refinement strategy.

USEAgentPlus is evaluated on 50 open-source projects written in various programming languages. The authors also study how the system performs in writing CI configuration scripts for popular Python repositories. In both evaluations, the proposed framework outperforms baselines such as codex-cli and OpenHands.

### Strengths
- The paper addresses a relevant topic: while many recent works focus on PR generation (e.g., SWE-Bench-style agents), the software lifecycle also requires agents that can automate the integration phase.
- Having an agent capable of automatically building a project and running its test suite without human intervention would be highly valuable for developers: USEAgentPlus seems a good candidate, given its performances.
- The research questions are clear, and Tables 1 and 2 effectively demonstrate that USEAgentPlus can handle dependencies, execute test suites, and create CI scripts more reliably than existing systems.

### Weaknesses
1. Presentation and clarity:
    - The abstract does not clearly introduce USEAgentPlus, despite it being the main contribution
    - It is not immediately clear what new components are introduced compared to the original USEAgent
2. Ablation analysis should be expanded, in order to understand the individual contributions and performances of the Probing Agent and the Advisor Agent.
3. The Consensus Memory appears to be a key architectural element and possibly a novel component. If it is indeed new, it deserves a clearer explanation.
4. Improve evaluation clarity: 
    - The abstract mentions that baselines such as OpenHands and ExecutionAgent were "extended with tools for environment management" but it is unclear how this was done.
    - Line 266 states: "...two baselines while executing USEAgentPlus with three parameters." - it’s not clear what the three parameters are, nor which two baselines are being referenced.

### Questions
- Question regarding Weakness #3: Why is OpenHands included in Table 2 (CI evaluation) but not in Table 1 (project build evaluation)?
- Could you clarify section "SWE Bench Verified" (line 321)? The description of this evaluation is unclear — what was measured, and how does it relate to the main tasks?
- In the conclusion, you state that ReAct agents perform better than unadjusted agent ensembles, yet the proposed extension builds on the latter. If ReAct performed better, why not use that as the starting point?
- The paper presents an important idea. The experimental design is solid, but the presentation needs improvement to make the contributions, evaluation and architecture clearer. I would be willing to raise my score if the authors clarify the above questions and improve the presentation.


Minor comments: 
- line 130: double comma
- line 184: "Project" -> "project" 
- line 158: "softeware"-> "software"
- line 265: Tool Settings should start a new paragraph
- line 291: codex-cli is presented here as CodeX
- line 413: missing space before “("
- Inconsistencies in section titles: some of them with a period, others do not

### Soundness
2

### Presentation
1

### Contribution
3
