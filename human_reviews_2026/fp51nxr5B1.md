# Automatic Generation of Safety-compliant Linear Temporal Logic via Large Language Model: A Self-supervised Framework

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Converting high‑level natural‑language task descriptions into formal specifications such as Linear Temporal Logic (LTL) is essential for ensuring safety in cyber‑physical systems (CPS). Existing work, however, only optimizes translation quality without explicitly verifying the output against safety constraints. We present AutoSafeLTL, a self‑supervised, cloud–edge–collaborative framework that automates the generation of safety‑compliant LTL specifications while preserving logical consistency and semantic fidelity. A lightweight edge‑side three-stage-fine-tuned LLM offers real‑time conversion from natural language to LTL specifications (NL2LTL) and guarantees safety‑critical latency and data locality. Two larger‑capacity cloud‑side agents then iteratively refine the alignment: 1) LLM‑as‑an‑Aligner matches atomic propositions to safety constraints, and 2) LLM‑as‑a‑Critic interprets counterexamples from Inclusion Check to guide corrective regeneration. This collaborative architecture provides a safety-guaranteed alignment mechanism between high-level user intent and formally verifiable system behavior, demonstrating the potential of our framework to advance AI Alignment in safety-critical domains. Our approach achieves 0% violation rates on multiple benchmarks, enabling trustworthy specification generation and verification for both AI and critical CPS applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces AutoSafeLTL, a self-supervised framework for automatically generating safety-compliant Linear Temporal Logic (LTL) specifications from natural language task descriptions.
Unlike prior NL2LTL systems that only translate text into logical formulas, AutoSafeLTL ensures that generated formulas formally satisfy predefined safety constraints. Using automata-based inclusion checking, the system verifies and repairs LTL outputs until they pass all safety constraints. Experiments in traffic-navigation scenarios show 0% violation rate, outperforming GPT-4 and nl2spec baselines while maintaining edge efficiency and data privacy.

### Strengths
- Novel integration of LLMs with formal verification: 

The inclusion of automata-based inclusion checking and counterexample-guided refinement provides genuine safety guarantees missing from previous NL2LTL work.

- Practical and privacy-preserving architecture: 

The cloud–edge collaboration design is well-suited for CPS applications where latency and data locality are critical.

- Strong experimental validation: 

Demonstrates zero safety violations and higher efficiency than GPT-4, with comprehensive ablations across fine-tuning stages.

### Weaknesses
- Limited domain coverage: 

The framework and datasets are restricted to navigation/traffic scenarios; generalization to other CPS domains remains unproven.

- Dependence on predefined safety constraints: 

The system assumes an existing library of safety LTL rules, limiting adaptability to new or evolving domains.

- Scalability and verification overhead: 

The iterative counterexample-guided repair loop may be computationally expensive for complex LTLs or multi-agent settings.

- Evaluation dataset limitations:

The repair process prioritizes safety compliance over the user’s original intent, but lacks a quantitative or qualitative measure of “semantic drift.” This could lead to safe yet semantically distorted outputs. The fine-tuning and evaluation datasets are relatively small (200–300 curated examples), synthetic, and domain-specific. Real-world instructions with linguistic ambiguity are not extensively tested.

### Questions
Can the framework be extended to automatically derive or learn safety rules from natural language corpora, rather than relying on manually predefined sets?

When safety repair alters the original task intent, how do you measure or control semantic drift between the initial description and the final safe LTL?

How does the system perform when handling multiple interacting constraints or large-scale multi-agent systems? Are there strategies to parallelize or approximate inclusion checking for scalability?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents AutoSafeLTL, a framework that integrates a fine-tuned LLM with cloud based agents and formal verification tools to generate LTL specifications from natural language that are guaranteed to be compliant with pre-defined safety constraints. The proposed three-stage fine-tuning strategy and the cloud-edge collaborative architecture for iterative, counterexample-guided repair are novel and interesting concepts. The authors report a 0% violation rate on their benchmark.

### Strengths
The core vision of tightly integrating large language models with formal verification to provide hard guarantees is timely and important. The move beyond just measuring translation accuracy to ensuring formal safety properties is a significant step forward for the application of LLMs in critical systems.

The framework strives for a fully automated pipeline from natural language to a verified LTL formula, which is an ambitious and worthwhile goal. The proposed improvement to the counterexample path extraction algorithm to enable this automation is a good contribution.

The idea of using a formal counterexample from the inclusion check not as a mere failure signal, but as a structured feedback mechanism to guide an LLM creates a meaningful dialogue between formal methods and LLMs.

### Weaknesses
- Lack of temporal efficiency: lack of time performance metrics. The proposed framework involves multiple, iterative loops of LLM inference, LTL-to-automata conversion, and language inclusion checking. Given that the primary application domain is safety-critical CPS (exemplified by a traffic navigation scenario), where decision-making must occur in milliseconds or seconds, the latency measurements are important. The average of 4.3 semantic correction iterations (Table 7) suggests a potentially prohibitive computational and communication overhead. Without data on execution time, it is impossible to evaluate the framework's practicality for any real system.

- Over-reliance on pre-defined, exhaustive Safety Constraints: the entire safety guarantee hinges on the assumption that the set of safety constraints Ψ is complete and correct. The paper does not address the "open-world" problem:

What happens when a user introduces a novel Atomic Proposition (AP) not present in the safety library? The LLM-as-an-Aligner might map it incorrectly or drop it, leading to semantically incorrect or unsafe specifications.

An AP generated as go_straight_200m might need to match a safety library AP like straight_200m. This is a simple case. However, more complex divergences are likely. For example, the user's instruction "Navigate to the emergency bay" might generate an AP reach_emergency_bay, while the safety constraints use a more technical term like at_service_area. The LLM may fail to link these conceptually similar but lexically different terms.

How does the system handle unforeseen scenarios or edge cases not covered by Ψ? The guarantee of 0% violations is only valid with respect to the known constraints, which provides a false sense of security if the constraint set is incomplete.

- Computational scalability: the paper ignores the well-known state explosion problem in automata-based verification. The complexity of BA construction and inclusion checking is exponential in the number of APs. The evaluation uses a small, manageable set of APs (4-8). The framework's behavior and feasibility with a realistic number of APs (e.g., 30+ for a real autonomous driving system), are not discussed. This raises serious doubts about its scalability.

- Benchmark: the evaluation is conducted on a custom benchmark. There is no demonstration of generalization, raising concerns about overfitting to the authors' specific data style (Chen et al., 2023).

- Missing baselines: the paper fails to compare against established NL2LTL methods, such as grammar-based approaches (e.g., Fuggitti & Chakraborti, 2023), other LLM-based fine-tuning methods (e.g., Chen et al., 2023). Consequently, it is impossible to determine if the significant complexity of AutoSafeLTL provides any tangible benefit over existing, potentially simpler and more efficient methods.

- Data privacy: the paper emphasizes "data locality for privacy" as a key advantage of the edge-side User LLM. However, the generated LTL formula, which is a direct formalization of the user's intent and the environmental context, is sent to the cloud for the verification and alignment steps. While the raw natural language may stay on the device, the formalized semantic content does not. This nuanced privacy trade-off is not adequately discussed.

Fuggitti, F., & Chakraborti, T. (2023). NL2LTL – A Python Package for Converting Natural Language (NL) Instructions to Linear Temporal Logic (LTL) Formulas.

Chen, Y., Gandhi, R., Zhang, Y., & Fan, C. (2023). NL2TL: Transforming Natural Languages to Temporal Logics using Large Language Models.

### Questions
What is the end-to-end latency of your framework, from receiving the NL input to outputting a safety-compliant LTL, on your benchmark? Please provide end to end average and broken down by component average.

How does your system behave if a user input contains an AP not in the pre-defined library? Please provide a concrete example and explain the process. What happens in the case of an AP that is not just new, but semantically incompatible with the safety rules encoded in the constraints? Does the framework simply suppress the action, or is there a way to inform the user that their request is inherently unsafe given the current rules? Please walk through the specific system behavior in such a scenario.
What is the maximum number of APs your framework can handle while maintaining reasonable verification times? Can you show how verification time grows with the number of APs?

Why did you not compare your method against other specialized NL2LTL tools? (e.g. Chen et al., 2023,  Fuggitti & Chakraborti, 2023) Can you provide such a comparison on a publicly available NL2LTL benchmark to demonstrate superior accuracy or efficiency? (e.g. Chen et al., 2023)

Fuggitti, F., & Chakraborti, T. (2023). NL2LTL – A Python Package for Converting Natural Language (NL) Instructions to Linear Temporal Logic (LTL) Formulas.

Chen, Y., Gandhi, R., Zhang, Y., & Fan, C. (2023). NL2TL: Transforming Natural Languages to Temporal Logics using Large Language Models.

### Soundness
3

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
5

### Summary
This work proposed a framework to generate LTL based on natrual language description with saferty complience.

### Strengths
This work is focusing on an important problem in CPS research, and it is critical in safety-critical applications.

### Weaknesses
My major concern is the originality of this work and the novelty.

It is not novel that connects a external verification module to the contents generated by LLM, I think this work is just another instance of these works. The difference is the out of LLM is LTL itself in this work,and others' output should follow some formal specifications such as LTL and first-order logic. It is very intuitive to use the checking results as feedback or guidance for LLM, and this is not novel.
Some early attemts can be found below:
1. Counterexample Guided Inductive Synthesis Using Large Language Models and Satisfiability Solving (MILCOM'23)
2. Assuring LLM-Enabled Cyber-Physical Systems (ICCPS'24) (This work has open-sourced their implementation as well (https://weizhesyr.github.io/SafePilot_doc/))

My second concern is the evaluation, it is easier for conjuction formulas but harder for disjuction in general. And it is easier for global operator but harder for until operator, I think it is more convicing if these statistics can be shown and how complex LTL this framework can handle. Currently, these details are missing, the statistics in Table 8 is insufficient.

There are some minor issues for clarity as well. For example, it is unclear why LLM-as-a-critic is useful, the only statement is they discoverd this and this not validated in ablation study. Since it requires multiple rounds for a correct LTL generation, it is unclear why reasoing model is not tested.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes AutoSafeLTL, a self-supervised framework for converting natural language task descriptions into safety-compliant Linear Temporal Logic (LTL) formulas. Unlike previous NL2LTL approaches that focus on translation accuracy, AutoSafeLTL integrates formal verification into the generation loop to guarantee that outputs comply with predefined safety constraints.

### Strengths
1. Integrating formal verification directly into the language model generation loop is interesting.
2. Ablation studies and comparisons against GPT-4 and NL2Spec demonstrate measurable benefits, which have lower iteration counts and zero violation rates.

### Weaknesses
1. While the multi-stage fine-tuning strategy is practical, it mainly repurposes standard LLaMA fine-tuning with LoRA adapters, which is not a significant methodological innovation.
2. Will the approach generalize to other CPS domains (e.g., robotics?).
3. There is no assessment of whether generated LTL formulas remain semantically faithful to user intent after repair. The authors used safety compliance as the metric to measure, but alignment quality is not measured.

### Questions
Please clarify the concerns raised in the weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
