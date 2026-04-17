# AdAEM: An Adaptively and Automated Extensible Measurement of LLMs' Value Difference

- Decision: Accept (Oral)
- Scores: 8, 8, 4, 8

## Abstract
Assessing Large Language Models’ (LLMs) underlying value differences enables comprehensive comparison of their misalignment, cultural adaptability, and biases. Nevertheless, current value measurement methods face the informativeness challenge: with often outdated, contaminated, or generic test questions, they can only capture the orientations on comment safety values, e.g., HHH, shared among different LLMs, leading to indistinguishable and uninformative results. To address this problem, we introduce AdAEM, a novel, self-extensible evaluation algorithm for revealing LLMs’ inclinations. Distinct from static benchmarks, AdAEM automatically and adaptively generates and extends its test questions. This is achieved by probing the internal value boundaries of a diverse set of LLMs developed across cultures and time periods in an in-context optimization manner. Such a process theoretically maximizes an information-theoretic objective to extract diverse controversial topics that can provide more distinguishable and informative insights about models’ value differences. In this way, AdAEM is able to co-evolve with the development of LLMs, consistently tracking their value dynamics. We use AdAEM to generate novel questions and conduct an extensive analysis, demonstrating our method’s validity and effectiveness, laying the groundwork for better interdisciplinary research on LLMs’ values and alignment. Codes and the generated evaluation questions are released at https://github.com/ValueCompass/AdAEM.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes AdAEM, a dynamic and self-extensible framework for evaluating LLM value orientations, addressing the "informativeness challenge" in existing static benchmarks. By adaptively generating test questions that maximize value distinctions across different LLMs, AdAEM constructs an evaluation benchmark containing 12,310 questions and systematically evaluates 16 mainstream LLMs.

### Strengths
- The paper clearly identifies a critical issue in existing value evaluation methods—using outdated, contaminated, or overly generic test questions leads to indistinguishable and uninformative results. This is a genuine and important research problem.

- The optimization algorithm is grounded in an information-theoretic objective (Equation 1) and employs an EM-like algorithm to alternately optimize question generation and response selection.

-  The method is evaluated on 16 mainstream LLMs covering different scales, architectures, and regions, validated across multiple dimensions (question quality, validity, reliability, extensibility).

### Weaknesses
- Reliability of probability estimation: For black-box LLMs, off-the-shelf classifiers approximate p(v|y) and coherence measures approximate p(y) (Section 3.2), but error propagation from these approximators to the final optimization objective is insufficiently discussed, potentially introducing systematic bias.

- Limited Evaluation Scope: Only instantiated on Schwartz's value system; while Appendix H acknowledges limitations, no validation experiments on other value theories (e.g., MFT, Hofstede) are provided, limiting generalizability.

- Reproducibility Considerations: Selection criteria for P1 and P2 models unclear: Why these 9 specific models? Would replacing with other model combinations significantly affect generated question quality?

### Questions
-  When comparing with ValueDCG (4,561 questions) and ValueBench (40 questions), does AdAEM's 12,310 questions introduce a scale advantage? Can you provide fair comparison with controlled question counts?

 - What are the background distributions of the 5 experts (nationality, expertise)? The 300 samples represent only 2.4% of the total dataset—how do you ensure evaluation result representativeness?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces AdAEM, a novel, dynamic, and self-extensible framework for evaluating the value differences in LLMs. The authors correctly identify the "informativeness challenge" in current static benchmarks, which often fail to distinguish between models because they test for generic, shared safety values. AdAEM tackles this by automatically generating new, controversial, and timely test questions by probing the value boundaries of diverse LLMs. This dynamic approach allows the benchmark to co-evolve with LLMs, mitigating data contamination and providing a much-needed tool for assessing misalignment, cultural bias, and preference.

### Strengths
The paper addresses a critical and well-defined problem in LLM evaluation. Moving from static to a dynamic, self-extensible benchmark (AdAEM) is a significant conceptual and practical contribution.

The core idea of using in-context optimization to probe value boundaries across different cultures (diverse LLMs) and time periods (knowledge cutoffs) is clever and highly effective at generating informative questions.

The authors provide a thorough empirical analysis. The results clearly show that AdAEM yields evaluation results that are significantly more distinguishable than existing benchmarks (ValueBench, ValueDCG).

The controlled value priming experiments provide strong support for the benchmark's construct validity, showing it can accurately detect intended value shifts.

### Weaknesses
The framework's current implementation relies solely on Schwartz's Theory of Basic Values. While well-justified, this is just one of many value frameworks, and the paper could benefit from a brief discussion on how AdAEM could be adapted to other theories (e.g., Moral Foundations Theory).

The paper acknowledges the potential for misuse, but it is worth noting that a framework designed to find controversial topics could indeed be misused.

### Questions
N. A.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes AdAEM, an information-theoretic framework that automatically and adaptively generates and expands test items to measure differences in LLM value orientations. The core method optimizes a generalized JS-divergence with disentanglement regularizer and produces “high controversy, model discriminative” value-inducing questions via sampling and refinement loops. Within this framework, the authors also build AdAEM Bench with 12,310 items that have been validated on quality, diversity, validity, and reliability.

### Strengths
Rigorous, scalable measurement of LLM value orientations is a timely, high-impact question for safety, alignment, and governance communities. This paper provides a dynamic, adaptive evaluation of LLM values, helping to overcome the limitations of traditional static benchmarks—namely their inability to capture new events, the difficulty of updating datasets dynamically, and the reliance on manual maintenance.
The central claims and proposed contributions are supported by well-organized benchmark statistics and visualizations, along with human validation on reasonableness and value differentiation. Construct validity is evidenced via controlled value priming with significant shifts in target/opposite dimensions, and reliability is supported by five-fold internal consistency. Contrast and contextualization against existing benchmarks and prior works are adequate.

### Weaknesses
1. Methodologically, selecting questions that maximize divergence in LLM responses does not necessarily best reveal their value preferences; it may instead induce new biases. This could be a significant shortcoming of the work and requires careful justification.
2. The EM/IM-like alternating procedure lacks a formal convergence or monotonic guarantee in the main text.
3. Too much empirical approximation of mathematical modeling injures solidity.
4. Any reasons on your choice of hyper-parameters? No experiments or searches provided.

### Questions
1. Figure 2 can be polished as it directly provides an overall illustration of AdAEM framework at the very beginning. 
2. It would be better to unify the notation in main text, notation table and pseudo code, strictly avoiding the occurrence of unexplained and nonuniform symbols.
3. The “Figure 15” in AdAEM Question Generation on page 23 probably should be “Figure 12”.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a value difference elicitation framework AdAEM. The framework iteratively generates new questions and calculates scores for conformity, coherence, value difference and semantic difference, extending prior value benchmarks with questions that better highlight the value boundaries between LLMs. The authors show that the generated questions are more diverse compared to prior value benchmarks and calculate the value differences of 16 LLMs, using their framework.

### Strengths
- The framework is novel, extensible, and generalizable
- The framework and the generated dataset of 12k questions should be useful for researchers exploring value differences in LLMs
- The writing is easy to follow
- There is substantial analysis aiming to validate the effectiveness of the benchmark
- Includes a discussion on how the authors think about values for LLMs

### Weaknesses
- A lot of the analysis is conditioned on the Schwartz Value Survey. Though in principle the approach could work on other value frameworks, a proof of concept on a different set of questions would’ve strengthened the generalization capability of the framework.
- The validity analysis claims construct validity but assumes o3-mini’s capability of generating text with a particular value present, which is not a given.

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3
