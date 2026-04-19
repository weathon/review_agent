# LLM4DV: Using Large Language Models for Hardware Test Stimuli Generation

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 8, 3, 3

## Abstract
Hardware design verification (DV) is a process that checks the functional equivalence of a hardware design against its specifications, improving hardware reliability and robustness. A key task in the DV process is the test stimuli generation, which creates a set of conditions or inputs for testing. A major challenge is that existing approaches to test stimuli generation require human effort due to the complexity and specificity of the test conditions required for an arbitrary hardware design. We seek an efficient and automated solution that takes advantage of large language models (LLMs). LLMs have shown promising results for improving hardware design automation, but remain under-explored for hardware DV. In this paper, we propose an open-source benchmarking framework called LLM4DV that efficiently orchestrates LLMs for automated hardware test stimuli generation. Our analysis evaluates five LLMs using six prompting improvements over eight hardware designs and provides insight for future work on LLMs for efficient DV.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes a framework using LLM to generate tests for hardware designs. The framework employs a coverage-directed strategy and incorporates well-designed prompting techniques. The authors evaluated the framework on various hardware designs including CPUs. The experimental results show that the framework achieves performance that is comparable to, or better than, traditional approaches such as CRT and formal verification.

### Strengths
The authors effectively design the prompting strategies to guide LLMs in generating tests for hardware designs. Additionally, the framework has been tested on various LLM models. The experiments are performed on some industry-level designs such as CPUs, highlighting the impressive capabilities of current LLM technology.

### Weaknesses
As a submission to the ICLR conference, the paper lacks novelty in its application of AI concepts, appearing more as an engineering effort. Additionally, there are several similar works on LLMs for hardware test generation that the authors do not mention in this paper, including:

Wang X, Wan G W, Wong S Z, et al. "ChatCPU: An Agile CPU Design & Verification Platform with LLM" (61st Design Automation Conference, 2024).
Qiu R, Zhang G L, Drechsler R, et al. "AutoBench: Automatic Testbench Generation and Evaluation Using LLMs for HDL Design" (2024 International Symposium on Machine Learning for CAD, 2024).
Ma R, Yang Y, Liu Z, et al. "VerilogReader: LLM-Aided Hardware Test Generation" (2024 LLM Aided Design Workshop, 2024).
M. Hassan, S. Ahmadi-Pour, K. Qayyum, C. K. Jha, and R. Drechsler. "LLM-Guided Formal Verification Coupled with Mutation Testing" (2024 Design, Automation & Test in Europe Conference & Exhibition, 2024).

While this submission provides valuable insights into prompting strategies for hardware verification, its contributions appear limited. I recommend that the authors discuss and compare LLM4DV with these related studies and consider submitting the paper to a conference more focused on hardware domains.

### Questions
Please compare with recent works in using LLMs for hardware verification.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper proposes LLM4DV, a framework that uses LLMs to generate test stimuli for hardware design verification (DV). The authors then use this framework to evaluate state-of-the-art LLMs on this task. The authors also propose LLM prompting strategies to maximize success in this task. They show that these LLMs achieve test coverage rates ranging from 71.76% to 100%.

### Strengths
-	The problem that the authors address is important and challenging.
-	The authors provide an interesting evaluation of SOTA LLMs on the task of hardware design verification.
-	The paper is clear and easy to read.

### Weaknesses
While the proposed work is important, the paper has a limited discussion of related work and a limited comparison with existing work (that does not use LLMs). The problem of hardware design verification is well-studied. Examples of papers worth discussing:

-	HIVE: Scalable Hardware-Firmware Co-Verification using Scenario-based Decomposition and Automated Hint Extraction.

-	Survey of Machine Learning for Software-assisted Hardware Design Verification: Past, Present, and Prospect.

As the paper is written now, it is not clear what is its main contribution. Is the main goal of the paper to say that LLMs do better than non-LLM state of the art in the task of DV? Or is the goal to evaluate LLMs on this task (and propose a framework to do so). The authors should clarify this. The paper shows indeed that LLMs can do the task of DV when evaluated using the benchmark proposed by the authors, but it is not written to convince the reader that using LLMs to address this task is better than using other state-of-the-art methods. If your goal is to propose LLMs as the new state-of-the-art in addressing the problem of hardware design verification, you should have a detailed analysis of existing work and should compare with state-of-the-art methods. You also need to use the same benchmarks used by state-of-the-art methods (instead of creating a new benchmark).

The current paper has a high value. It clearly shows that LLMs can perform DV with high coverage rates, but as it is written now it is not convincing enough to show that LLMs are better than existing work in solving the problem.

### Questions
Can you clarify what is the goal of the paper? Is the main goal to say that LLMs do better than non-LLM state of the art in the task of DV? Or is the goal to evaluate LLMs on this task (and propose a framework to do so)?

Can you please add a more detailed discussion of existing non-LLM based methods that address the problem of DV? In particular, you can discuss aspects such as their test coverage rates and scalability and compare them to LLM-based methods.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work introduces LLM4DV, a framework that uses prompted LLMs to generate test stimuli for hardware Design Verification (DV). The workflow supports a plug-and-play setup, allowing users to experiment with various LLMs, hardware designs, and test coverage plans. The authors develop six prompt enhancements for effective automated DV, establishing strong baselines within LLM4DV as a testbed for studying LLM agentic behavior.

Key contributions include:

A. The creation of three DUT (Design-Under-Test) modules: (1) a Primitive Data Prefetcher Core, (2) an Ibex Instruction Decoder, and (3) an Ibex CPU, and the integration of 5 additional open-source designs, providing a diverse range of DUTs for users.

B. Evaluation of LLM4DV on 8 DUT modules with custom metrics, demonstrating coverage rates from 71.76% to 100% in realistic scenarios with optimized prompts.

Additionally, the authors open-sourced LLM4DV, encouraging both the machine learning and hardware design communities to experiment with it.

### Strengths
This research work is highly interesting, offering a promising solution to a critical challenge in hardware design and verification. The manuscript is well-organized and well-written. The LLM4DV framework is thoroughly described, detailing each of its component modules. The evaluation setup is clearly explained, and the results and analysis effectively support the authors' claims. The appendix section provides extensive scientific details that clarify the motivation for benchmarking and the working principles of the framework.

### Weaknesses
I have one minor suggestion: Algorithm A.2 should be included in the main text rather than in the appendix, as it would aid in understanding the background more effectively.

### Questions
This has already been addressed in the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The article introduces LLM4DV, an open-source benchmark framework leveraging Large Language Models to automate hardware test stimuli generation, specifically for hardware Design Verification. Hardware DV is crucial for ensuring that hardware designs function correctly, typically involving extensive human input to design and execute test stimuli that cover a range of scenarios. LLM4DV streamlines this process by generating stimuli with LLMs, reducing human effort and improving test coverage.

### Strengths
LLM4DV reduces the need for manual testing inputs, saving time and resources. It also Includes multiple evaluation metrics that allow comprehensive assessment beyond just coverage, such as message count efficiency.

### Weaknesses
1. I have observed several works with approaches similar to this submission in applying prompting methods to hardware verification and testing. Notably:

*AssertLLM: Generating and Evaluating Hardware Verification Assertions from Design Specifications via Multi-LLMs*

*Towards LLM-Powered Verilog RTL Assistant: Self-Verification and Self-Correction*

*LLM-assisted Generation of Hardware Assertions*

I suggest the author to compare LLM4DV with these works.

2. Novelty Concerns and Venue Suitability:

While the LLM4DV framework in this submission introduces noteworthy adaptations of prompting methods tailored to Hardware Testing, many of these underlying techniques are not novel. Prompting-based methods have been extensively applied in other domains since at least 2022. Specifically:

The Best-iterative-message Sampling described in Section 4.4 resembles a method for summarizing conversational memory, as discussed in Generative Agents: Interactive Simulacra of Human Behavior.

The Few-shot Prompting approach in Section 5 has parallels with work on few-shot learning found in PPT: Pre-trained Prompt Tuning for Few-shot Learning.

3.  Reproducibility and Open-Source Framework Availability:

The authors claim to provide an open-source benchmarking framework, but I was unable to find any code, supplementary materials, or anonymous links for verification. Given my experience with works claiming to be open-source yet only offering minimal or incomplete repositories, I recommend the authors to provide anonymous links during the rebuttal stage would allow me to verify the authenticity and reproducibility of this contribution, which is crucial for enabling follow-up work by future researchers.

### Questions
See Weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper presents LLM4DV, an open-source benchmarking framework designed to automate hardware test stimuli generation using large language models (LLMs) for hardware design verification (DV). LLM4DV leverages six LLMs and introduces six prompt engineering improvements to tackle the labor-intensive DV process, aiming to enhance coverage by generating stimuli for various hardware devices under test (DUTs). The framework evaluates these LLMs across eight hardware designs, comparing their coverage rates and effectiveness against conventional constrained-random testing (CRT) and formal verification methods. While LLM4DV demonstrates notable performance improvements on simpler DUTs, its efficacy declines with complex hardware architectures due to limitations like state-space explosion, token window constraints, and a high dependency on prompt engineering. The study concludes that LLMs show promise for automating aspects of DV but require further adaptation to handle more complex verification scenarios effectively.

### Strengths
The paper tries to address a relevant problem and is well written, and well organized.

### Weaknesses
1) The LLM4DV framework relies on missed-bin sampling to manage the large number of uncovered coverage bins, but this approach introduces randomness that may hinder consistent exploration of harder-to-hit bins. This sampling method may lead to poor prioritization, with the LLM potentially focusing on simpler bins at the expense of complex corner cases that are critical for comprehensive hardware verification.
2)	The dialog restarting mechanism, designed to mitigate the issue of LLMs "getting stuck" by clearing dialog history, lacks adaptability. The fixed thresholds (like hitting fewer than three bins within 40 messages) are rigid and may not accommodate variability across hardware designs with different complexity levels. This static approach may cause premature restarts, losing valuable accumulated context and decreasing efficiency in achieving coverage.
3)	Including DUT source code in the prompt often exceeds the token limit for many LLMs, especially in the case of designs with complex logic or multiple functional modules. This limitation results in truncating prompts or excluding critical code portions, reducing the model’s ability to effectively map test stimuli to specific coverage bins.
4)	The framework’s effectiveness heavily depends on manually engineered prompts and specific configurations, such as coverage-feedback templates and best-iterative-message sampling. This approach introduces overhead and suggests a lack of automated prompt optimization, which could otherwise make LLM4DV more adaptable and less reliant on human intervention.
5)	For designs with large state spaces, such as components with storage elements or multiple queues (e.g., AMPLE Prefetcher Weight Bank), the LLM4DV framework struggles to maintain efficient coverage. This is because the LLM’s state exploration strategy does not scale well with increasing design complexity, leading to a "state-space explosion" where the LLM cannot feasibly cover all unique states or transitions.

### Questions
1) Given the fixed thresholds in dialog restarting, how does LLM4DV adapt to variability in coverage rates across designs with differing complexity? Could dynamic thresholds improve restart efficiency?
2)	For designs with detailed CPU architectures, like those with complex instruction sets (e.g., RISC-V), how does LLM4DV ensure that test stimuli generation adequately captures inter-instruction dependencies to achieve comprehensive coverage?
3)	Does LLM4DV include mechanisms to help identify or diagnose the root causes of missed coverage bins? How could such a feature enhance the utility of the framework for iterative debugging workflows?

### Soundness
2

### Presentation
2

### Contribution
2
