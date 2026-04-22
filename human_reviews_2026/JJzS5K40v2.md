# Can Data-driven Machine Learning Reach Symbolic-level Logical Reasoning? -- The Limit of the Scaling Law

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 6, 2

## Abstract
With the qualitative extension of embedding representation and the method of explicit model construction, neural networks may achieve the rigour of symbolic level logic reasoning without training data, raising questions of where the limit of the scaling law for logical reasoning lies, i.e., whether data-driven machine learning systems can achieve the same level by increasing training data and training time. We show two methodological limitations that prevent supervised deep learning from reaching the symbolic-level syllogistic reasoning, a subset of logical reasoning: (1) training data can not distinguish all 24 types of valid syllogistic reasoning; (2) end-to-end mapping from premises to conclusion introduces contradictory training targets between neural components for object recognition and logical reasoning. Taking the Euler Net as a representative supervised neural network, we experimentally illustrate the challenges common to all image-input supervised networks, namely, they struggle to distinguish all types of syllogistic reasoning and to identify unintended inputs. We further challenge the most recent ChatGPTs (gpt-5-nano and gpt-5) to determine the satisfiability of syllogistic statements in four surface forms: words, double words, simple symbols, and long random symbols, showing that surface forms affect the reasoning performance and that ChatGPT gpt-5 may reach 100% accuracy but still provide incorrect explanations. As empirical training processes are stopped after reaching 100% accuracy, we conclude that supervised machine learning systems may follow scaling laws but will not reach symbolic-level logical reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper addresses whether data-driven neural networks can ever attain symbolic-level logical reasoning by simply scaling up data and computing. Using syllogistic reasoning as the test case, the authors identify two methodological limitations that they claim make this impossible:
(1) training data cannot distinguish between all valid syllogistic reasoning types, and
(2) End-to-end mappings in neural architectures introduce contradictory objectives between pattern recognition and reasoning.

They demonstrate these limitations using the Euler Net and a modified version (Super Euler Net), as well as an evaluation of GPT-5 and GPT-5-nano on symbolic syllogisms across different input forms (words, symbols, random strings). Their experiments show that even high-performing models can reach 100% accuracy but still produce incorrect explanations, which the authors interpret as evidence of an inherent boundary in scaling laws for reasoning.

### Strengths
- The topic is ambitious and philosophically interesting, touching on the limits of scaling laws and the intersection between data-driven learning and symbolic reasoning.
- The paper offers a systematic empirical investigation combining vision-based reasoning (Euler Net) and text-based reasoning (GPT models) under a unifying logical framework.
- The focus on syllogistic reasoning is appropriate for testing elementary logical inference, a well-understood and bounded task.
- The experiments are well-organized, and the claims are clearly structured around two proposed methodological deficits.
- The connection to both historical and modern theories of reasoning (Johnson-Laird, Knauff, symbolic logic traditions) adds conceptual depth.

### Weaknesses
Framing misalignment.
The title and introduction suggest that the paper addresses the general question of out-of-distribution (OOD) reasoning limitations in data-driven systems. However, the empirical work concerns a very specific and diagrammatic form of syllogistic reasoning. The chosen paradigm introduces additional confounds—such as visual representation and image-based processing—that are orthogonal to the general OOD reasoning question. The findings, therefore, do not substantively advance our understanding of why modern large models fail at OOD reasoning.

Conceptual overreach.
While the identified limitations (non-distinguishable training cases and conflicting submodule objectives) are valid observations, they are not unique to the presented setup and have been documented extensively in prior work (e.g., Guzman et al., Findings of NAACL 2024, and subsequent arXiv extensions). The novelty lies mainly in the framing, rather than the underlying insights. The claim that these two deficits constitute a fundamental limit on all data-driven reasoning systems is too strong, given the narrow empirical base.

Diagrammatic reasoning introduces confounds.
The use of Euler diagrams as the central experimental representation makes the results difficult to generalize. Diagrammatic reasoning blends visual recognition and logical inference, and the errors observed may stem as much from visual encoding issues as from reasoning failures. A purely symbolic or formal logical setup would provide a cleaner testbed for claims about reasoning limitations.

Weak connection to modern architectures.
The paper’s experimental systems (Euler Net and Super Euler Net) are structurally far from state-of-the-art architectures used for large-scale reasoning. Extending results from these simplified models to all “data-driven systems” is not justified without stronger empirical or theoretical bridges.

### Questions
Why is a diagrammatic syllogistic reasoning task an appropriate paradigm for investigating general reasoning limits in neural systems?

How would your conclusions differ if the experiments were conducted in a symbolic (non-visual) setup?

Could your two proposed deficits (non-distinguishability and contradictory objectives) be mitigated through architectural or representational interventions?

How do your findings extend—or differ—from existing analyses of reasoning depth and compositionality limitations (e.g., Guzman et al., 2024)?

How do you reconcile the visual-logic mixture of Euler Nets with claims about purely linguistic models like GPT-5?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the symbolic-level logical reasoning limits of data-driven machine learning systems, focusing on supervised deep learning and large language models (LLMs). The authors focus on syllogistic reasoning as a test case for whether neural networks can attain symbolic-level rigor.

Through experimental analyses, the paper identifies two key limitations in data-driven systems:

- Training data cannot distinguish all valid types of syllogistic reasoning.
- End-to-end learning introduces contradictory targets between the pattern recognition and reasoning components.

Using the Euler Net and its improved version, Super Euler Net (SupEN), the authors empirically demonstrate that even with iterative data augmentation and scaling, the model’s accuracy saturates below symbolic-level performance. Extending the argument to Transformer-based LLMs, including GPT-5 and GPT-5-nano, they find that although these models can achieve nearly perfect accuracy, they still produce logically incorrect explanations. The paper concludes that scaling laws allow for performance improvements but do not enable symbolic-level logical reasoning.

### Strengths
-  The paper tackles a critical research topic, whether data-driven machine learning can replicate symbolic reasoning.
-  The experiments with Super Euler Net, and GPT-5 variants support the paper’s claim

### Weaknesses
-  The experimental scope is narrow, with neural network methods limited exclusively to (Super) EulerNet.
-  the claim "we conclude that supervised machine learning systems may follow scaling laws but will not attain the rigour of symbolic logical reasoning." is overgeneralized

### Questions
First of all, I would like to thank the authors for their work. I agree with the authors that large language models (LLMs) and most deep neural networks currently lack robust logical reasoning capabilities. However, I respectfully disagree with the claim: “we conclude that supervised machine learning systems may follow scaling laws but will not attain the rigour of symbolic logical reasoning.”

This statement feels somewhat overgeneralized and represents a limitation of the paper. The experimental scope is restricted to (Super) EulerNet as the sole example of neural network-based reasoning. However, several other architectures have been proposed that demonstrate varying degrees of logical reasoning ability, such as Neural Module Networks [1], Logic Tensor Networks [2], and differentiable inductive logic programming (ILP) models [3]. These approaches are not discussed or compared in the current work.

Would the authors consider clarifying how these methods differ from the approach proposed in the paper, or including such models as additional baselines, or at least discussing their relevance in the broader context of neural reasoning?

[1] Andreas, Jacob, et al. "Neural module networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

[2] Badreddine, Samy, et al. "Logic tensor networks." Artificial Intelligence 303 (2022): 103649.

[3] Evans, Richard, and Edward Grefenstette. "Learning explanatory rules from noisy data." Journal of Artificial Intelligence Research 61 (2018): 1-64.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper explores a fundamental question: Can data-driven machine learning systems—especially supervised neural networks and large language models (LLMs)—achieve symbolic-level logical reasoning solely by scaling data and computational resources?
The authors approach this question by analyzing two representative paradigms: Euler Net, a vision-based supervised network for syllogistic reasoning; and GPT-5 / GPT-5-nano, large-scale transformer-based LLMs.
They identify two methodological limitations that fundamentally constrain data-driven systems: Training data cannot distinguish all valid types of syllogistic reasoning, because different syllogisms may share identical surface patterns in data space, preventing the model from learning symbolic distinctions; End-to-end architectures introduce internal conflicts between pattern recognition (which tends to fill in missing or perceptually inferred information) and logical reasoning (which must not introduce unseen entities).
To empirically verify these hypotheses, the paper develops Super Euler Net, an enhanced version capable of self-generating and validating new training data. Through iterative experiments, it shows that while performance can approach but never surpass 97–100% accuracy, models fail to reach the symbolic rigor required for logical validity and explanation consistency.
The evaluation of GPT-5 further supports this claim: though achieving perfect accuracy in some conditions, the models still produce logically incorrect or hallucinative explanations. The paper concludes that while scaling laws effectively enhance performance, they cannot bridge the gap between data-driven pattern learning and formal logical reasoning.
Overall, this paper provides both theoretical and experimental insights into the limits of scaling in achieving symbolic reasoning and contributes to a broader understanding of how machine learning interacts with formal logic.

### Strengths
1) The paper tackles one of the most fundamental open questions in AI—whether data scaling alone can yield true reasoning capabilities. By analyzing the structural mismatch between empirical learning and formal logic, it provides a well-articulated theoretical contribution with clear philosophical and computational implications. Unlike many conceptual discussions, this paper grounds its arguments in experimental evidence. The combination of the Super Euler Net (for visual reasoning) and GPT-5 experiments (for linguistic reasoning) effectively bridges symbolic, perceptual, and neural domains.
2) The two identified deficits—data indistinguishability and end-to-end conflict—are general and apply to a broad class of neural architectures, including both CNN-based and transformer-based systems. This insight is highly valuable for the theory of neural reasoning.
3) The introduction of Super Euler Net is innovative: it automatically generates and validates new training data, providing an interpretable testing ground for studying the scaling behavior of logical inference systems. It also offers a replicable way to study symbolic consistency in vision-based reasoning.
4) The experiments with GPT-5 demonstrate an important phenomenon: models may reach near-perfect accuracy yet still fail to produce correct logical explanations. This observation highlights the distinction between statistical competence and logical understanding.
5) The paper successfully combines perspectives from computer vision, formal logic, and cognitive science. It connects deep learning scaling theory (e.g., Kaplan et al., Bahri et al.) with cognitive models of reasoning (e.g., Johnson-Laird, Knauff), thus contributing to a richer theoretical landscape.

### Weaknesses
1) The study focuses exclusively on syllogistic reasoning, which, while fundamental, represents only a small subset of logical reasoning tasks. It remains unclear whether the same limitations would hold for more complex forms such as propositional, predicate, or modal logic. The claim of “methodological impossibility” thus feels somewhat overstated given the narrow scope.
2) The experiments rely on controlled and synthetic inputs (circles, hypernym pairs, etc.), which may not reflect the complexity or ambiguity of real-world reasoning contexts. Extending the analysis to noisy or real-world data could provide stronger evidence for generalization.
3) The evaluation of GPT-5 and GPT-5-nano, while illustrative, limits reproducibility and transparency. Without open-source benchmarks, it is difficult to verify the validity of the reported results or to replicate the symbolic explanation failures.
4) While qualitative analysis is thorough, the statistical treatment of results (e.g., variance across trials, significance testing) is absent. Including such analyses would make the claims more robust and empirically grounded.
5) The conclusion that “scaling laws cannot reach symbolic-level reasoning” is philosophically appealing but empirically underdetermined. It might be more accurate to phrase it as a “current limitation” rather than a “fundamental impossibility,” leaving room for potential hybrid or unsupervised solutions.

### Questions
1) Can Super Euler Net or the broader analytical framework be extended to address multi-step or relational reasoning tasks such as first-order logic proofs or causal inference? How might the identified deficits manifest in these more complex settings?
2) Have you explored the possibility of combining your approach with formal symbolic provers (e.g., Goedel-Prover, Isabelle, or LEAN)? Such hybrid systems might circumvent the “end-to-end contradiction” by separating perception from logical validation.
3) Could non-supervised objectives, such as contrastive learning or RL-based symbolic verification, mitigate the issues caused by incomplete training data or pattern injection? Would this change your conclusion about the universality of the deficits?
4) Since syllogistic reasoning can be expressed as triplet relations (subject–predicate–object), have you considered representing reasoning structures using graph neural networks or relational transformers? These may inherently encode logical compositionality and offer a path toward symbolic consistency.
5) To what extent do you view your conclusion (“data-driven ML cannot reach symbolic reasoning”) as a theoretical impossibility versus a practical limitation? Clarifying this distinction could prevent misinterpretation and better position your work within the ongoing debate on neural-symbolic reasoning.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
To address the question of where the limit of the scaling law for logical reasoning lies, the authors identify two reasons demonstrating that all image-input neural networks and LLMs fail to achieve 100% reasoning accuracy. The first reason is that the training data cannot effectively distinguish among all 24 types of valid syllogistic reasoning. The second is that the end-to-end mapping from premises to conclusions introduces conflicting training objectives between neural components responsible for pattern recognition and those responsible for logical reasoning.

### Strengths
The paper employs an automated setup to evaluate the reasoning abilities of state-of-the-art LLMs, which provides a strong motivation for assessing the realistic reasoning capabilities of these models.

### Weaknesses
The paper aims to investigate whether data-driven machine learning systems can achieve the same level of performance as symbolic reasoning systems by increasing training data and training time. However, the authors only use a single, small benchmark consisting of images containing syllogistic reasoning information applied to specific neural networks, such as the GPT-5. As a result, the problem setting is overly broad and cannot accurately characterize by the tested benchmarks and models.

### Questions
1. When using Euler diagrams as examples in Figure 3, why not employ a more formal language as the benchmark to evaluate the reasoning abilities of neural networks? Furthermore, in the experiments with image inputs, the main objective appears to be evaluating the reasoning abilities of all image-input neural networks, including LLMs. How can image data adequately represent the reasoning capabilities of neural networks? Most prior research on reasoning first tackles symbolic reasoning described in formal languages before applying neural networks.
2. The authors represent four situations with statements like “some W are U” and “some W are not U,” but such symbolic statements cannot fully capture the exact logical relationships between W and U. So, the textual description statements are not good examples for indicating the logical relations in Figure 3. Why the authors use these implicit textual statements?
3. In Line 24, the authors state that their experiments illustrate limitations common to all image-input supervised networks. What types of neural network architectures were used? Can you provide a detailed list? Based on the architecture presented in Figure 4, why is this architecture considered universal for representing any neural network with image inputs?

### Soundness
1

### Presentation
2

### Contribution
2
