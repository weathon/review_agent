# BlueCodeAgent: A Blue Teaming Agent Enabled by Automated Red Teaming for CodeGen AI

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
As large language models (LLMs) are increasingly used for code generation, concerns over the security risks have grown substantially. 
Early research has primarily focused on red teaming, which aims to uncover and evaluate vulnerabilities and risks of codeGen models. 
However, progress on the blue teaming side, which is challenging and requires defense with semantic understanding, remains limited. To fill in this gap, we propose BlueCodeAgent, an end-to-end blue teaming agent enabled by automated red teaming. 
Our framework integrates both sides: red teaming generates diverse risky instances, while the blue teaming agent leverages these to detect previously seen and unseen risk scenarios through constitution and code analysis with agentic integration for multi-level defense. 
Our evaluation across four representative code-related tasks—bias instruction detection, malicious instruction detection, vulnerable code detection, and prompt injection detection—shows that BlueCodeAgent achieves significant gains over the baseline models and safety prompt-based defenses. In particular, for vulnerable code detection tasks, BlueCodeAgent has integrated dynamic analysis to effectively reduce false positives, a critical but difficult-to-address problem.
Overall, BlueCodeAgent achieves much more effective and context-aware risk detection and mitigation. 
We demonstrate that the red teaming benefits blue teaming by continuously identifying new vulnerabilities, which could significantly enhance defense performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the underexplored challenge of blue teaming for code-generating LLMs—detecting and mitigating security risks with semantic understanding. While prior work focuses on red teaming (probing for vulnerabilities), the authors propose BlueCodeAgent, an end-to-end blue teaming agent enhanced by automated red teaming. Their framework uses red team–generated risky code examples to train a defense agent that combines constitutional reasoning, static code analysis, and agentic coordination for multi-layered protection. Evaluated on three tasks—bias detection, malicious instruction identification, and vulnerable code detection—BlueCodeAgent significantly outperforms baseline models and safety-tuned prompting. Notably, it integrates dynamic analysis in vulnerability detection to reduce false positives, a persistent issue in static-only approaches. The results show that continuous red teaming improves blue teaming by uncovering novel threats, enabling more effective, context-aware defense. This work demonstrates a practical and scalable path toward secure LLM-based code generation.

### Strengths
The proposed BlueCodeAgent framework is innovative in its integration of automated red teaming with multi-level defensive reasoning, combining constitutional principles, static analysis, and dynamic validation. Its demonstrated effectiveness—particularly in reducing false positives for vulnerable code detection—addresses a key practical challenge in code security. The evaluation across diverse tasks further strengthens the paper’s impact and relevance.

### Weaknesses
Computation Overhead: What is the computational or time cost associated with the proposed method? For blue-team applications, latency is critical since we must minimize response delays to users. The method relies on GPT-4o for constitution generation — please specify the average time per single request, without batch or multi-threaded averaging. The overhead should reflect the true per-request latency.

Online vs. Offline Constitution Generation: Why is constitution generation performed online? Please discuss the time–effectiveness trade-off between online generation (producing constitutions during testing) and offline generation (precomputing and storing constitutions in a database for retrieval based on similar inputs).

Model Choice and Scaling Behavior: Is GPT-4o strictly necessary? Describe the scaling law for constitution-generation models — for instance, how do different model sizes (e.g., 0.6B–6B parameter range) influence single-request latency and accuracy? Additionally, compare GPT-4o and GPT-5 in terms of cost, response time, and effectiveness.

### Questions
Please refer to the weakness.

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
This paper proposes BlueCodeAgent, a comprehensive blue teaming agent for code generation LLMs, which is powered by automated and diverse red teaming. The core idea is to use a pipeline where red teaming generates diverse and realistic risky instances, which are then used to create actionable constitutions guiding blue-team defenses and, in the case of code vulnerability detection, augmented with dynamic code testing. Evaluation across bias, malicious instruction, and code vulnerability detection tasks shows that BlueCodeAgent outperforms both safety prompting and recent knowledge-augmented models, achieving up to a 12.7% average F1 score improvement across several datasets.

### Strengths
- Integrating comprehensive automated red teaming with knowledge-enhanced blue teaming agents is an effective defense method and possesses novelty.
- This paper conducts a comprehensive evaluation of three benchmarks (bias, toxicity, and code vulnerability risks) and reports results across visible/invisible risk categories, multiple base models, and various prompt configurations, demonstrating an extensive experimental scope.

### Weaknesses
- BlueCodeAgent relies to some extent on the knowledge base constructed by automated red teaming, but the red teaming methods used are limited. This seems insufficient to cover all harmful categories and red teaming strategies. How does BlueCodeAgent handle cases that are not included in the knowledge base?
- BlueCodeAgent summarizes “constitutions” based on closest-matching knowledge base entries found using embedding search. However, this means blue teaming effectiveness could, in part, inherit biases or blind spots of the underlying knowledge/data, particularly if uncovered risks deviate semantically from those seen. There is little discussion or mitigation of this risk.
- The selection of baselines is relatively limited. While BlueCodeAgent demonstrates defensive capabilities, particularly in bias and malicious instruction detection, the chosen baselines are not specifically targeted defense methods. For instance, some defense strategies designed to counter jailbreaking attacks are not included. This results in BlueCodeAgent lacking a direct and meaningful comparison with such approaches.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces BlueCodeAgent, an end-to-end blue teaming framework for defending against security risks in code generation AI systems. The key contribution is leveraging comprehensive automated red teaming to enhance blue teaming defenses. The framework operates in two stages: (1) diverse red teaming that generates risky instances across bias instructions, malicious instructions, and vulnerable code, and (2) knowledge-enhanced blue teaming that retrieves relevant examples to generate "constitutions" (safety principles) and incorporates dynamic sandbox testing to validate vulnerabilities. Experiments across three representative tasks demonstrate improvements over baseline models and safety prompt-based defenses, with F1 scores approaching 1.0 for instruction detection tasks.

### Strengths
- The paper presents a novel perspective on connecting red teaming and blue teaming for code security. The idea of distilling red teaming knowledge into actionable constitutions for defense is creative. The integration of dynamic testing with LLM-based static analysis for vulnerability detection is a practical contribution that addresses the over-conservatism problem identified in prior work.
- The paper is well-structured and clearly written. Figure 2 provides a helpful overview of the framework. The distinction between principled-level defense (constitutions) and nuanced-level analysis (dynamic testing) is well-articulated.

### Weaknesses
- The evaluation covers only three risk categories (bias, malicious code, vulnerable code). Many other security concerns exist in code generation (e.g., privacy leaks, intellectual property violations, supply chain attacks). The "unseen risks" evaluation (Section 4) tests on different sub-categories within the same high-level risk type (e.g., different CWE types). True generalization to fundamentally different attack types remains unclear. Table 2 shows performance drops when moving from seen to unseen risks (e.g., 0.25→0.20 for bias detection), suggesting limited generalization.
- The red teaming process relies on manual enumeration of policies, bias groups, and CWE types (Section 3.2). This approach may not scale to discover novel attack vectors. The framework requires maintaining and updating knowledge bases as new vulnerabilities emerge. The paper doesn't discuss strategies for continuous learning or knowledge base maintenance.
- The safety prompt baselines are relatively weak. More sophisticated guardrail systems (e.g., Llama Guard, specialized code security models) should be compared. The LLM-ensemble baseline (Table 1) is interesting but limited to one experiment. More ensemble methods could be evaluated. There is no comparison with existing vulnerability detection tools (e.g., static analysis tools like Semgrep, CodeQL) or hybrid approaches combining LLMs with traditional security tools.

### Questions
1.  How does BlueCodeAgent perform on completely novel attack types not represented in the red teaming knowledge base? Can you provide evaluation on emerging threats (e.g., prompt injection attacks specific to code generation)?
2. What is the quality control process for generated constitutions? Are there cases where constitutions are incorrect, contradictory, or overly broad?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces BlueCodeAgent, a framework for "blue teaming" against code generation LLMs. BlueCodeAgent tackles security challenges like detecting biased instructions by integrating knowledge derived from an automated "red teaming" pipeline. This system leverages the offensive data to generate safety rules that guide the defensive LLM agent and employs dynamic testing to verify potential vulnerabilities and reduce false positives. Evaluation across multiple benchmarks demonstrates that this knowledge-enhanced, agent-based approach significantly improves risk detection and mitigation compared to baseline and simple safety-prompt defences.

### Strengths
- BlueCodeAgent achieves significant gains over the baseline models and safety prompt-based defenses, demonstrating much more effective and context-aware risk detection and mitigation. It consistently performs well on both seen and unseen risks
- Red-teaming can empower effective blue-teaming defenses, showing that red teaming benefits blue teaming by continuously identifying new vulnerabilities

### Weaknesses
- The proposed methods are mostly based on prompt engineering and the technical contribution is therefore limited for this venue
- The definition of blue teaming is not presented in the paper. It is only clear from the context, but I would recommend to add a clear defintion early in the paper to show the contribution
- Limited Scope of Risk Categories: The current evaluation focuses on three representative code-related tasks: bias instruction detection, malicious instruction detection, and vulnerable code detection
- To enhance its real-world utility, BlueCodeAgent needs to be scaled up to operate at the file and repository levels. This scaling effort would require the agent to be equipped with more advanced context retrieval tools and memory components

### Questions
- How are vulnerabilities detected?

### Soundness
3

### Presentation
3

### Contribution
3
