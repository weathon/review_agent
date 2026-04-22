# Simulation to Rules: A Dual-VLM Framework for Formal Visual Planning

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 8, 2, 4, 4

## Abstract
Vision Language Models (VLMs) show strong potential for visual planning but struggle with precise spatial and long-horizon reasoning, while Planning Domain Definition Language (PDDL) planners excel at formal long-horizon planning but cannot interpret visual inputs. Recent works combine these complementary advantages by translating visual problems into PDDL. However, while  VLMs can generate PDDL problem files satisfactorily, accurately generating PDDL domain files, which encode planning rules, remains challenging and typically requires human expertise or environment interaction.
We propose VLMFP, a Dual-VLM-guided framework that autonomously generates both PDDL problem and domain files for formal visual planning. VLMFP combines a SimVLM that simulates action consequences with a GenVLM that generates and iteratively refines PDDL files by aligning symbolic execution with simulated outcomes, enabling multiple levels of generalization across unseen instances, visual appearances, and game rules.
We evaluate VLMFP on 6 grid-world domains and demonstrate its generalization capability. On average, SimVLM achieves 87.3\% and 86.0\% scenario understanding and action simulation for seen and unseen appearances, respectively. With the guidance of SimVLM, VLMFP attains 70.0\%, 54.1\% planning success on unseen instances in seen and unseen appearances, respectively. We further demonstrate that VLMFP scales to complex long-horizon 3D planning tasks, including multi-robot collaboration and assembly scenarios with partial observability and diverse visual variations. Project page: https://sites.google.com/view/vlmfp.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper presents VLMFP, a framework that uses two Vision-Language Models (VLMs) to automatically convert visual planning problems into formal PDDL domain and problem files. SimVLM is fine-tuned for visual understanding and action simulation. GenVLM generates the PDDL files and refines them based on SimVLM's feedback. The method works by iteratively generating PDDL files, checking their consistency against SimVLM's simulations, and updating them to correct errors. Evaluated on grid-world domains, VLMFP successfully generates valid plans for unseen problem instances and visual appearances, removing the need for predefined domain files or constant access to the environment.

### Strengths
It autonomously generates both PDDL domain and problem files from visual input, eliminating the need for human-predefined domains—a key bottleneck in prior work.

The separation into a specialized simulator (SimVLM) and a general-purpose generator (GenVLM) effectively leverages the strengths of different models.

The feedback loop that compares PDDL execution with VLM simulation is a reasonable mechanism for catching and correcting errors in the generated artefacts.

The framework demonstrates robust performance across unseen problem instances, visual appearances, and even some modified rules within the tested grid-world environments only.

It provides a preliminary basis for bridging the gap between high-dimensional visual perception and precise, symbolic reasoning.

### Weaknesses
The evaluation is confined to synthetic 2D grid-worlds. It remains unproven whether the approach can scale to realistic 3D environments, cluttered scenes, or tasks requiring complex physics reasoning.

The method likely relies on the discrete, structured nature of grid cells and positional predicates (e.g., pos-1-1). Translating continuous, real-world spaces into such a formalism is a major unsolved challenge.

SimVLM was fine-tuned on a massive, domain-specific dataset (430k samples). Curating a similar dataset for every new real-world domain (e.g., kitchen manipulation, navigation) would be prohibitively expensive.

The iterative process of generating, executing, and comparing action sequences for refinement is computationally intensive and may not be feasible for real-time applications.

While it handled some rule variations, performance dropped significantly on complex, novel rules (e.g., the "freezing" mechanic), indicating that reasoning about entirely new dynamics is still a limitation.

### Questions
1) The framework is demonstrated on 2D grid-worlds. What are the most significant technical hurdles you foresee in scaling this approach to more complex, realistic 3D environments, such as a simulated kitchen or a robot navigation task?

2) The current method relies on discrete positions (e.g., pos-1-1). How could the approach be adapted to handle continuous state spaces, which are common in real-world robotics and control problems?

3) The current setup assumes a full, top-down view of the entire state. How would VLMFP need to be modified to handle partially observable environments where the agent's view is limited?

4) SimVLM required a massive, fine-tuned dataset (430k samples) for the grid-world domains. Is this level of domain-specific data a fundamental requirement, or do you see a path towards a more general-purpose SimVLM that could be applied to novel domains with minimal fine-tuning?

5) The grid-worlds have a fixed set of object types with clear visual representations. How would the object recognition and spatial reasoning capabilities of SimVLM need to improve to handle novel, real-world objects with diverse and often ambiguous appearances?

6) The iterative refinement process involving multiple VLM calls and PDDL simulations seems computationally expensive. What is the latency of the full VLMFP pipeline, and is it feasible for any real-time decision-making scenarios?

7) Could the "simulation consistency checking" step be made more sample-efficient? Are there smarter strategies for generating the action sequences used for comparison, rather than random sampling, to identify logical flaws faster?

8) The results show a sharp performance drop on the completely unseen "freezing" rule (Rule 5). Does this indicate a fundamental limitation in the system's ability to reason about truly novel action dynamics, as opposed to parametric variations of known rules?

9) How robust is the iterative refinement process to persistent or cascading errors? If SimVLM itself makes a systematic error in simulation, could this lead GenVLM to converge on an incorrect but self-consistent PDDL domain?

10) In your view, what single advancement—whether in model architecture, training data, or the core algorithm—would be the most critical for bridging the gap from these compelling grid-world results to a useful real-world application?

### Soundness
2

### Presentation
2

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
This paper investigates translating 2D navigation planning environments into formal symbolic PDDL representations. The task is meaningful because accurate translation enables invoking PDDL solvers for automated planning. The proposed approach uses an iterative refinement process: a VLM first generates draft PDDL files, then executes random actions in the environment to verify whether the outcomes align with simulation results. Experiments show that the method achieves a reasonable success rate (70% on average).

Overall, the experimental results are generally sufficient, but the writing is poor and difficult to follow. The paper requires a major revision to improve clarity and readability before it can be considered for publication.

### Strengths
- This is an interesting and novel research direction, to the best of my knowledge.  
- Based on the experimental results, the proposed method appears to significantly improve translation accuracy compared to existing approaches.

### Weaknesses
- The writing can be improved. The paper is dense and difficult to follow, focusing heavily on implementation details while lacking a clear, high-level conceptual framework. It is hard to identify the core scientific research question amid the technical descriptions. The method section is overly detailed, and the ideas could be presented more clearly by abstracting the core principles. For example, the iterative refinement process could be framed as learning a formal world model (domain files) and structured state representations (problem files). A clearer separation between methodology and implementation would significantly improve readability.

- It is unclear whether dividing the process into two VLMs, SimVLM and GenVLM, is necessary. In principle, a single large VLM with self-refinement capabilities could fulfill both roles. Using a smaller model for specific subtasks makes sense from a computational efficiency perspective, but this seems more like an engineering optimization than a conceptual requirement. If the authors wish to emphasize cost-effectiveness as a contribution, this should be highlighted in the abstract and introduction, and supported with experiments comparing computation cost and accuracy between small and large VLMs.

- I recommend adding a discussion on extending this framework to more complex and realistic domains, such as robotic control. A common criticism of PDDL-based approaches is their limited flexibility; this framework could address that issue. It would strengthen the paper to explicitly discuss how VLMFP might generalize beyond grid-world settings.

- Some suggested related works:
    - *Efficient Exploration and Discriminative World Model Learning with an Object-Centric Abstraction*
    - *Planning with Reasoning using Vision-Language World Model*
    - *Predicate Invention for Bilevel Planning*

### Questions
See Weaknesses

### Soundness
3

### Presentation
1

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
This paper introduces VLM-Guided Formal Planning (VLMFP), a novel framework for the autonomous generation of PDDL files from visual inputs.

Existing VLM-based approaches in visual planning are limited in their ability to accurately generate domain PDDL and often rely on access to ground-truth domain files. 
To overcome these limitations, VLMFP employs a dual-VLM architecture: a fine-tuned SimVLM that perceives environmental scenarios from visual inputs and simulates action outcomes, alongside a large generative VLM (GenVLM) that generates and iteratively refines PDDL files by aligning their execution with SimVLM’s simulations. 

Evaluated across six grid world domains, VLMFP demonstrates strong generalization capabilities to unseen instances, visual appearances, and game rules.

### Strengths
1. The proposed dual-VLM framework is well-justified.
It effectively allocates distinct reasoning roles to each model: SimVLM, a fine-tuned smaller VLM, specializes in spatial reasoning from visual inputs, while GenVLM, the larger model, leverages SimVLM’s output to reason over and generate PDDL content.

2. The paper is clearly structured.
The introduction and Section 3.1 offer a coherent presentation of the problem formulation, related work, and the motivation behind the proposed approach.

### Weaknesses
The paper addresses a highly specific problem, which may limit the broader applicability of the proposed approach. 
The evaluation, confined to six grid-world domains, lacks demonstration on more diverse PDDL scenarios. 
Furthermore, the requirement for in-domain training data for SimVLM in each new domain could hinder the framework's practical scalability.

### Questions
My main questions regarding the evaluation are as follows:

1. In Table 1, the proposed SimVLM component is not compared against any baseline methods. 

2. In Table 2, the CodePDDL baseline shows comparable performance to VLMFP on the Frozenlake and Maze domains but performs poorly on others. What is the underlying reason for this significant performance disparity across domains? An explanation is needed to interpret these results correctly.

3. Several relevant papers mentioned in the text (Line 52-79) are not included as baselines in Table 1 or 2. Including these comparisons is crucial for a fair and thorough evaluation of VLMFP's performance against contemporary works.

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
3

### Summary
This paper proposes a dual-VLM framework that generates and iteratively refines PDDL from visual inputs using SimVLM as a world model; the idea is simple, technically sound, and the experiments are well-designed. The approach is well-motivated and potentially generalizable.
However, the  evaluation is limited to grid-worlds, leaving scalability to continuous/3D/POMDP settings unclear. Convergence is under-specified: EW score threshold, iteration limits, failure handling when the score doesn’t reach 1.0, and reproducibility under stochastic simulations. In adition, one risk is hallucination and safety—SimVLM-consistency is not correctness without an external oracle, so hallucinations can yield confidently wrong PDDL.

### Strengths
This work proposes a dual-VLM framework that can generate more accurate PDDLs given visual observations and raw language prompts. The idea is simple yet effective: leverage an additional VLM as a world model (SimVLM), which generates and iteratively refines PDDL files by comparing their execution with SimVLM's simulations.
Overall, the paper makes meaningful contributions to PDDL generation from visual inputs.
The dual-VLM architecture is well-motivated, addressing the complementary weaknesses of spatial reasoning and PDDL generation. This idea of dual-VLM self-refinement can potentially be extended to VLM-based planning frameworks to improve planning domain knowledge generation.
However, there are some concerns about (1) task generalizability, (2) convergence analysis, and (3) hallucination and safety issues.

### Weaknesses
### Limited Tasks
The evaluation is restricted to grid-world domains. Real-world planning often involves continuous spaces, 3D environments, and more complex dynamics. The paper doesn't discuss how the approach would scale to such scenarios. For example, SimVLM may have issues for cases involving POMDPs.
### Hallucination Issues Without Guarantees
The framework fundamentally assumes SimVLM provides accurate ground truth for action simulations. However, this framework does not provide a proof or verification step against an external oracle; it guarantees syntactic validity and SimVLM-consistency, not alignment with real dynamics. When SimVLM hallucinates, the entire framework can produce incorrect PDDL files that match the hallucinated behavior rather than actual dynamics. The paper provides no mechanism to detect or correct SimVLM hallucinations, creating a critical failure mode where the dual-VLM framework could perform worse than simpler baselines. This is particularly concerning given recent work showing VLMs' propensity for confident hallucinations in spatial reasoning tasks.
### EW Score and Iterations Unclear
The paper is unclear about the refinement process. Specifically: (1) The threshold for acceptable alignment is never explicitly defined (only implied to be 1.0 from figures), (2) There is no discussion of what happens when convergence fails after multiple iterations (e.g., when the EW score cannot achieve 1.0), (3) Since SimVLM uses ChatGPT-4o which may have randomness in simulation, how is reproducibility ensured? To clarify these details, it would be better to provide: explicit convergence criteria, iteration limits, failure handling procedures, and empirical analysis of convergence rates across different domains.

### Questions
**SimVLM Architecture Choices:** Why was Qwen2-VL-7B chosen specifically? Have you experimented with other VLM architectures? An ablation study comparing different base models would be valuable.

**Failure Analysis:** Can you provide more detailed analysis of failure modes? When VLMFP fails, is it typically due to SimVLM perception errors, GenVLM generation errors, or convergence issues?

**Model Hallucination Issue:** If SimVLM hallucinates and creates inaccurate transitions, could the dual-VLM framework potentially end up with worse generation results? Is there any way to guarantee correctness?

**EW Score and Iterations:** Please see the questions outlined in the weakness section above.

### Soundness
2

### Presentation
2

### Contribution
3
