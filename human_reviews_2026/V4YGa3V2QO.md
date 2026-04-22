# SecTest-Eval: Can LLMs Verify Security Impacts of A Vulnerability?

- Avg Score: 2.00
- Decision: Reject
- Scores: 2, 2, 2, 2

## Abstract
As Large Language Models (LLMs) have demonstrated capabilities in exploiting software vulnerabilities, the potential misuse of LLMs in conducting cyberattacks highlights the urgent need for benchmarks to capture the frontier of their capabilities. Existing benchmarks primarily evaluate LLMs from a global perspective, where LLMs are tasked to generate exploits that call vulnerable code (e.g. function) from project entry points, and reveal significant performance gaps. Therefore, recent studies have explored decomposing the whole challenging exploit generation task into a series of relatively simple tasks, applying LLMs from a local perspective, particularly for generating exploits that directly call vulnerable functions. While such attempts have shown effectiveness, existing benchmarks may lead to unreliable model performance in these scenarios due to low label accuracy for vulnerable functions. To address this, we introduce SecTest-Eval, the first benchmark for evaluating LLMs in exploit generation form a local perspective, where LLMs are tasked to generate exploits that directly call vulnerable functions. SecTest-Eval incorporates a novel automated data labeling method achieving accurate vulnerable function annotation and features a sandbox framework that automatically evaluates generated exploits by monitoring unauthorized data access, data modification, and denial-of-service. Our evaluations show that, even from a local perspective, current LLMs still face challenge in exploit generation, achieving at most 56% success rate. Furthermore, we find that Chain-of-thought prompting yields no significant improvement, while integrating LLMs into security-oriented agents improves success rates by 7.5%. These findings underscore the effectiveness of SecTest-Eval and suggest that enhancing LLMs' capabilities in exploit generation requires either training on specialized datasets or incorporating security-specific tools.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a new benchmark, SecTest-Eval, for evaluating LLMs on the task of generating unit tests for verifying security impacts of vulnerable methods. 
This subtly differs from the more common task of generating tests for vulnerable code snippets in that it requires explicit generation of "sensitive statements" that lead to security impacts.
The paper argues that existing benchmarks either focus on project-level test generation which might not isolate the vulnerability test generation capabilities of LLMs (SEC-Bench), or focus on limited security impacts (SEC-Bench, CyberEval).
The proposed benchmark focuses on function/method-level vulnerable code and three security impact categories: confidentiality (reading and printing sensitive information), integrity (modifying sensitive information), and availability (early termination or unresponsiveness).
The function-level vulnerable code snippets (C/C++) are obtained from existing vulnerability detection datasets (PRIMEVUL) which are then filtered using LLMs and manual inspection to 204 instances which contain sensitive statements (balanced by the three categories, 14 vulnerability types).
Finally, the paper finds that 5 LLMs underperform on SecTest-Eval (~56% success rate by the best LLM) and lists the differences in terms of model type (general purpose vs. code LLMs), security impact category, function length, prompting technique, and vulnerability type.

### Strengths
- While there are benchmarks that evaluate detection of function-level vulnerabilities, the proposed benchmark is the first to look at the security impacts of these vulnerabilities. 
- The experiments section is well-written and easy to understand.

### Weaknesses
- The models evaluated in the experiments are LLMs with two simple prompting strategies. The state-of-the-art methods on several Software Engineering tasks are agentic frameworks such as SWE-agent [1] and OpenHands [2] which are not evaluated here.
- The experiments also use a temperature of 0 which does not necessarily guarantee determinism for modern LLMs. 
- There is no justification provided for the choice of security impacts. Is there a concrete definition or list of security impacts that can be cited here?
- While I appreciate the list of insights from the experiments in Section 4.2, I do not understand how this guides (a) practioners who would like to use LLMs for this task and (b) future research in this area. The proposed future directions in Section 6 are applicable to the use of LLMs for most tasks, are there any pointers specific to the studied task?

There are some instances in the main text that should ideally include citations. For instance, line 136 mentions "Recent researchers show that many labeled vulnerable functions..." but this is not supported with a reference. Similarly, line 098 mentions that the benchmark "covers the three main categories of security impacts" but this isn't supported with a reference / justification.

Overall, I think the paper would immensely benefit from including state-of-the-art LLM-based agents in the evaluation and listing actionable insights which can guide either practioners / researchers in this area. Further, supporting claims with references should improve the exposition of the text.


[1] Yang, J., Jimenez, C. E., Wettig, A., Lieret, K., Yao, S., Narasimhan, K., & Press, O. (2024). Swe-agent: Agent-computer interfaces enable automated software engineering. Advances in Neural Information Processing Systems, 37, 50528-50652.

[2] Wang, X., Li, B., Song, Y., Xu, F. F., Tang, X., Zhuge, M., ... & Neubig, G. (2024). Openhands: An open platform for ai software developers as generalist agents. arXiv preprint arXiv:2407.16741.

### Questions
I will summarize my questions from the weaknesses section above (please refer to that section for more details):
1. Could you provide a list of actionable insights which can guide practioners / future researchers in this area?
2. Could you comment on the exclusion of SotA LLM-based frameworks from the evaluation?
3. How were the three security impacts chosen? 
4. How well does the temperature = 0 ensure determinism of outputs?

Additionally,

5. How many samples are there in the benchmark? Line 218 mentions that SecTest-Eval includes 204 instances while Table 1 states a total of 203.

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
The paper proposes a new benchmark, SecTest-Eval, to evaluate LLMs’ ability to verify the security impact caused by a weakness. Different from prior work, SEC-bench and CyberEval, SecTest-Eval include additional information to generate the proof of concept test program, such as an impact description. SecTest-Eval also covers 14 weakness types, which are more than some prior work. The paper presents some analysis of LLMs' performance on this new benchmark, indicating that room for improvement exists, and there are some patterns of failure were discussed.

### Strengths
+ Security vulnerabilities are important weaknesses to study and detect. Having more benchmarks would allow better evaluation of detection and analysis techniques.

### Weaknesses
- The evaluation is lacking a comparison with prior benchmarks. It is important to justify the motivation for another benchmark and what this new benchmark contributes to the existing landscape. And it is also very crucial to quantitatively show the uniqueness of this new benchmark compared to the existing ones.
- The comparison with the prior benchmark is not complete; there is a need for a more detailed analysis of overlap and unique entries in terms of weakness types. Also, a separability and ranking agreement analysis between TestSec_Eval and prior benchmarks, such as Sec-Bench and CyberEval, would provide a better view of how the proposed benchmark contributes to the overall landscape of existing security benchmarks.
- Lacks a proper justification for why the three primary security impacts chosen are the most important. Is there any evidence or prior study that supports this selection?

### Questions
1. LLMs only achieve an 18% success rate on SEC-Bench, meaning the existing benchmarks are still quite challenging. Why do we need yet another security benchmark? Why do we need to change the usage setting and provide a different set of inputs for LLMs?

2. Since SecTest-Eval covers 14 weakness types, what is the overlap with vulnerability types of prior work (memory access and privacy issues)? 

3. Is there any evidence or prior study that supports the selection of three primary security impacts?

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
This paper proposes SecTest-Eval, a function-level benchmark, evaluates whether LLM can generate a good PoC test for vulnerabilities that can verify security impacts for C/C++ programs. Basically, the input for LLM is a triplet ⟨vulnerable function fv, its CWE type, and a target security impact I⟩, the model should output a self-contained unit test format program T that can be directly tested by execution and trigger the problem. The benchmark construct 203 function-level tasks, which cover 14 CVE types with 3 main impact categories. They evaluate the whole benchmark with five SOTA LLMs under "Direct" and "COT" mode prompting.

### Strengths
1. Clear and well-scoped formulation. The task is precisely defined ⟨vulnerable function fv, its CWE type, and a target security impact I⟩ -> T. The benchmark provides concrete, automatically verifiable oracles for each impact type—making evaluation reproducible and objective.
2. Enough experiment workload, provide an evaluation of the SOTA models and check with two kinds of prompts(direct and COT) to show the diverse results.

### Weaknesses
1. Limited realism of function-level scope. Generally, real vulnerabilities often span multiple functions or modules. Evaluating only isolated functions misses complex control/data-flow dependencies—meaning success on SecTest-Eval may not translate to real-world exploitation.
2. Minor contribution and limited diversity. The dataset largely comes from previous work PRIMEVUL, with additional filtering and relabeling. While this makes the benchmark easier to construct, it limits originality. Moreover, the dataset only covers C/C++ code, which may introduce bias and prevent a more comprehensive evaluation of LLM capabilities across languages and ecosystems.
3. Trivial evaluation method. The evaluation of generated PoCs mainly relies on executing the produced code and checking for explicit runtime signals (e.g., file modification or crash). This rule-based approach cannot capture more subtle or complex exploit behaviors.
4.  Limited experimental design. The experiments focus only on direct model prompting and do not consider agentic or tool-augmented settings, which are a core capability of many state-of-the-art LLM. Without including agent-based reasoning or tool use, it remains unclear whether the observed weaknesses stem from model limitations or simply from the lack of external reasoning support.

### Questions
1. It might be good to cite some secure code generation work as a shared related interest for LLM security.

[1] Seccodeplt: A unified platform for evaluating the security of code genai

[2] SafeGenBench: A Benchmark Framework for Security Vulnerability Detection in LLM-Generated Code

[3] CodeLMSec benchmark: Systematically evaluating and finding security vulnerabilities in black-box code language models

2. Do you consider testing with some SOTA code agent, like OpenHands, Claude-Code?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
LLMs show promise in analyzing security vulnerabilities but current benchmarks may underestimate their true capabilities due to excessive context length and limited coverage of exploitation methods and security impacts. The paper introduces SecTest-Eval, a benchmark that tasks LLMs with generating PoCs, unit tests to verify security impacts of 203 vulnerable C/C++ functions across 14 weakness types and 3 impact categories. Evaluation of five state-of-the-art LLMs reveals modest success rates of up to 56%, highlighting the need for further improvement in LLMs’ ability to accurately assess security vulnerabilities.

### Strengths
1. Addresses evaluating LLMs' capabilities for the problem of PoC generation, an essential and time-consuming step for assessing the security impact of any vulnerability. The problem is very challenging, it is timely to solve in the era of LLMs, and it is under-studied relative to its practical importance.

2. Proposes a novel benchmark for generating PoCs in the form of programs containing unit tests that are specifically designed to validate the security impacts of a given vulnerable function.

3. Curates a balanced dataset of 203 samples, each consisting of a C/C++ vulnerable function, the type of the contained weakness (across 14 types of CWEs), and 3 impact categories (Unauthorized Data Reading, Unauthorized Data Modification, and Denial of Service).

4. Demonstrates headroom for improving LLM's abilities to generate PoCs, with only 56% overall accuracy for GPT-4.1.

### Weaknesses
1. The benchmark is limited to function-level vulnerabilities. The authors consider this a strength since the latter can underestimate the true capabilities of LLMs due to excessive context length, but it is also a significant weakness to focus on single functions.

2. The algorithm for how PoCs are generated is not presented. This is a lost opportunity since it would be essential to understanding important aspects of your approach such as its effectiveness (e.g. in terms of running time, LLM inference cost, and any guarantees it provides) and generality (e.g. in terms of extending it to support multiple functions, and different kinds of CWEs and security impacts). Ideally, the presentation of the algorithm should be in terms of the formal notation introduced in the problem formulation.

3. The paper has other presentation issues. Most importantly, there is no example of a generated PoC. The appendix shows many examples of non-PoCs but not a single example of a valid PoC. I would expect at least one such example, ideally in the main body of the paper, and an illustration of how the method generates it.

4. The paper claims generality as a strength over existing benchmarks in terms of aspects such as coverage of exploitation methods and security impacts. But it focusses somewhat narrowly on C/C++, and a few CWE types and impact categories.

Minor comments:

- Figure 3 is not referenced at all, making definitions 1-3 hard to understand.
- I could not find statistics of valid PoCs such as lines of code. This would help in getting a sense of how complex PoCs your framework can generate.
- A start of Section 3, you say 204 task instances, but Table 1 says 203.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2
