# From Atoms to Chains: Divergence-Guided Reasoning Curriculum for Unlabeled LLM Domain Adaptation

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Adapting Large Language Models (LLMs) to specialized domains without human-annotated data is a crucial yet formidable challenge. 
Widely adopted  knowledge distillation methods often devolve into coarse-grained mimicry, where the student model inefficiently targets its own weaknesses and risks inheriting the teacher's reasoning flaws. This exposes a critical pedagogical dilemma: how to devise a reliable curriculum when the teacher itself is not an infallible expert. Our work resolves this by capitalizing on a key insight: while LLMs may exhibit fallibility in complex, holistic reasoning, they often exhibit high fidelity on focused, atomic sub-problems.
Based on this, we propose Divergence-Guided Reasoning Curriculum (DGRC), which constructs a learning path from atomic knowledge to reasoning chains by dynamically deriving two complementary curricula from disagreements in reasoning pathways. When a student and teacher produce conflicting results, DGRC directs the teacher to perform a diagnostic analysis: it analyzes both reasoning paths to formulate atomic queries that target the specific points of divergence, and then self-answers these queries to create high-confidence atomic question-answer pairs. These pairs then serve a dual purpose: (1) providing an atomic curriculum to rectify the student's knowledge gaps, and (2) serving as factual criteria to filter the teacher's original reasoning chains, yielding a verified CoT curriculum that teaches the student how to integrate atomic knowledge into complete reasoning paths. Experiments across the medical and legal domains on student models of various sizes demonstrate the effectiveness of our DGRC framework. Notably, our method achieves a 7.76\% relative improvement for the 1.5B student model in the medical domain over strong unlabeled baseline.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces the Divergence-Guided Reasoning Curriculum (DGRC), an innovative framework designed to address the critical challenge of adapting Large Language Models (LLMs) to specialized domains without human-annotated data. The DGRC framework cleverly leverages divergences in reasoning paths between a "teacher" and a "student" model as a trigger for learning. Instead of simply assuming the student is wrong, DGRC treats divergence as an opportunity for a neutral, diagnostic inquiry. The teacher model analyzes both reasoning chains to generate "atomic questions" targeting the specific points of disagreement. It then re-answers these questions to create a high-confidence "atomic curriculum." Concurrently, this verified atomic knowledge serves as a factual standard to filter the teacher's original reasoning chains, yielding a "verified CoT curriculum." The student model is then trained via an "atom-to-chain" paradigm, first rectifying foundational knowledge gaps with the atomic curriculum and then mastering compositional reasoning with the verified CoT curriculum.

### Strengths
1. The concept of "Cognitive Asymmetry" elegantly captures a key characteristic of current LLM capabilities. Reframing model disagreement from a simple "student error signal" to a "trigger for impartial diagnosis" is a highly intelligent and novel perspective. This successfully circumvents the fundamental dilemma of learning from a "fallible teacher" in unlabeled settings.
2. Through an automated pipeline, DGRC transforms abstract model disagreements into two concrete, complementary curricula. This design adeptly solves two core problems: (1) how to precisely identify and rectify the student's knowledge gaps (via the atomic curriculum), and (2) how to ensure the teacher's supervision is reliable (via the verified CoT curriculum).
3. The paper reports significant performance gains, such as a 7.76% relative improvement for the 1.5B student model in the medical domain. Furthermore, DGRC demonstrates remarkable parameter efficiency and generalization; for instance, the DGRC-adapted 7B model outperforms its 32B baseline counterpart, highlighting its practical value for deploying smaller models in resource-constrained environments.

### Weaknesses
1. The DGRC pipeline is quite complex, involving multiple rounds of LLM sampling, divergence diagnosis, atomic question generation and answering, and CoT verification. This process is computationally expensive. This point is particularly salient given that a primary motivation for distillation-based methods is to create smaller, efficient models. If the curriculum generation process itself is prohibitively expensive, it risks becoming counterproductive, undermining the very goal of achieving cost-effective performance gains in the student model. It would be beneficial for the authors to provide a quantitative analysis of the computational cost (e.g., total tokens or API calls per sample), as this is crucial for assessing the method's practical viability.
2. The experimental results (Tables 3 and 4) clearly indicate that DGRC's effectiveness is positively correlated with the teacher model's capability. While the authors acknowledge this in the limitations section, it is a primary practical bottleneck. The method's efficacy might be substantially reduced in scenarios where access to powerful models like GPT-4.1 is unavailable.
3. As the framework is triggered by divergence, it cannot detect or correct errors that are common to both the teacher and student models("shared blind spots"). The authors mention this, but its potential impact might be understated. A discussion of potential mitigation strategies (such as incorporating a third-party model or an external knowledge base as an judger)would strengthen the paper.

### Questions
1. Can you quantify the computational cost of the DGRC pipeline in comparison to the baseline distillation methods?
2. Regarding the "shared blind spots" issue, do you see potential in integrating the DGRC framework with external knowledge bases to detect and correct such errors?
3. In the peer-correction experiment (Section 4.3), Qwen2.5-Instruct-72B achieved a higher final correction rate than GPT-4.1, despite having lower atomic knowledge accuracy. This is an interesting finding. Could you elaborate on your hypotheses for this phenomenon?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies unlabeled LLM domain adaptation for student model based on the divergence-guided reasoning curriculum. The proposed method DGRC first decomposes complex query into atom queries which are derived from the divergence between teacher and student models. Then DGRC steps into the curriculum generation stage and student adaptation stage. Extensive experiments are conducted to show the effectiveness of the proposed method.

### Strengths
1. The studied problem of unlabeled domain adaptation is interesting and of practical use.
2. The proposed method follows a cognitive asymmetry principle that is reasonable.
3. The experiments are extensive and persuasive to show the model effectiveness.

### Weaknesses
1. The proposed method lacks novelty.
2. Some important experiments are missing.
3. Some details are missing, which decreases the readability of the paper.

### Questions
1. While the cognitive asymmetry principle holds, the method that decomposes complex query into simple and easy-to-answer queries is widely seen, which further decreases the appeal of the proposed method.
2. The introduction of divergence is very odd. For K responses from teacher model and J responses from student model, the number of divergence pair could be large. While the capability of the teacher model is generally stronger than the student model, the former is still fallible. The majority voting from responses could improve their truthfulness. Hence it is really confusing why we need to form so many divergence pairs, but do not first conduct majority voting and then form the divergence pair.
3. In the case of **incorrect result** from teacher model and **incorrect/correct** result from student model, can the teacher model as the diagnostician correctly identifies the reasons and generate the correct atom/COT curriculm?
4. The details of the atom queries are missing which are really confusing. I suggest the authors add some toy examples on the atom queries in the main texts.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes the Divergence-Guided Reasoning Curriculum (DGRC) framework to address the challenge of adapting Large Language Models (LLMs) to specialized domains (e.g., medical and legal) without human-annotated data.

### Strengths
First, the core insight of "cognitive asymmetry" is well-validated and innovative, effectively resolving the pedagogical dilemma of learning from an imperfect teacher by shifting focus from flawed holistic reasoning to reliable atomic knowledge. 

Second, the three-stage DGRC framework (divergence detection, curriculum generation, student adaptation) is structurally rigorous, with multi-step filtering (e.g., IFD and LLM-based evaluation) ensuring high-quality curricula and addressing limitations of coarse-grained distillation and costly external knowledge bases. 

Third, the comprehensive experimental design, covering multiple domains, model scales, and training paradigms (SFT, GRPO), provides robust evidence of DGRC’s versatility, with clear performance gains in both accuracy and generalization.

### Weaknesses
First, DGRC heavily depends on teacher model capability, as shown by the significant performance gap when using strong proprietary teachers (e.g., GPT-4.1) versus weaker open-source ones, limiting its applicability in scenarios where access to advanced teachers is constrained. 

Second, the framework fails to address "shared blind spots" where both teacher and student make the same error, as divergence (the trigger for curriculum generation) is absent, leaving such critical flaws undetected. 

Third, the self-teaching configuration yields only marginal improvements for smaller models (e.g., 0.9% for 7B Qwen2.5), revealing that weak models lack sufficient diagnostic ability to act as reliable teachers, which narrows DGRC’s utility for low-resource model adaptation.

### Questions
see Weaknesses

### Soundness
3

### Presentation
2

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
This paper introduces the Divergence-Guided Reasoning Curriculum (DGRC), a novel framework for unlabeled domain adaptation of Large Language Models (LLMs). The key insight is the principle of "cognitive asymmetry": while LLMs can be fallible in complex, multi-step reasoning, they are highly reliable on focused, atomic sub-problems. Leveraging this, DGRC transforms disagreements between a teacher and student model into a structured, dual-curriculum learning path. When a divergence in final answers is detected, the teacher model acts as a diagnostician to:

1. Generate and self-answer atomic questions targeting the root cause of the divergence, forming an Atomic Curriculum.

2. Use these high-confidence atomic facts to filter its own original reasoning chains, producing a Verified Chain-of-Thought (CoT) Curriculum.

The student is then trained in a two-stage process: first on the atomic curriculum to rectify foundational knowledge, and then on the verified CoT curriculum to learn compositional reasoning. Extensive experiments in medical and legal domains demonstrate that DGRC outperforms strong unlabeled distillation baselines, shows strong generalization to unseen benchmarks, and is particularly effective for smaller-scale student models.

### Strengths
Originality: The formulation of the "cognitive asymmetry" principle and its operationalization through a dynamic, disagreement-driven curriculum is a highly original and insightful contribution. The shift from passive mimicry to active diagnosis and repair is a powerful conceptual advance.

Quality: The work is technically sound and executed with high quality. The experimental validation is comprehensive, spanning multiple domains, model sizes, and configurations (including self-teaching and RL). The ablation studies and analysis of "cognitive asymmetry" are particularly strong.

Clarity: The presentation is a standout strength. The paper is a model of clarity, effectively guiding the reader through a complex framework with a well-structured narrative and, presumably, clear diagrams.

Significance: The method demonstrably improves model performance, especially for smaller models, which has practical implications for deploying capable models more efficiently. The "atom-to-chain" learning trajectory is a principled approach that could influence future curriculum learning designs for LLMs.

### Weaknesses
Incomplete Comparative Analysis: The most significant weakness is the lack of comparison with other modern unlabeled adaptation techniques. For instance, how does DGRC compare to self-training methods like Self-Rewarding Tuning or rejection sampling fine-tuning? Without these comparisons, it is challenging to gauge the true standing of the proposed method within the existing landscape.

Limited Scale of Human Evaluation: The manual assessment of the atomic curriculum's quality, while positive, is based on a relatively small sample size (n=100 per model). To make a stronger claim about the reliability of the generated data, a larger-scale evaluation or automated metrics would be more convincing.

Ambiguous Self-Teaching Utility: The self-teaching results, while presented as a strength for versatility, are actually quite weak for the 7B model (+0.9%). This suggests a key limitation: the framework's effectiveness is heavily dependent on the diagnostic capability of the "teacher," which diminishes for smaller models. This bottleneck should be discussed more critically.

Potential Data Contamination Risk: Given the use of standard benchmarks (MedMCQA, MedQA, MMLU) and large-scale teacher models (GPT-4.1, Qwen2.5), the risk of the teachers having prior exposure to the test sets cannot be ignored. While a common issue, a more diligent discussion or analysis of this potential confounder is expected in a top-tier publication.

### Questions
1. Broader Comparative Baseline: How does DGRC perform against other state-of-the-art unlabeled adaptation methods, such as those based on self-training or reinforcement learning from AI feedback (RLAIF), which also do not require human labels? Can you include a comparison with at least one such strong, non-distillation baseline?

2. Robustness of Atomic Knowledge: The manual evaluation of atomic Q&A is crucial but limited to 100 samples. Can you provide a larger-scale analysis, perhaps using a highly capable model (e.g., GPT-4o) as an automated judge to validate a larger subset of the generated atomic curriculum, thereby strengthening the claim of high fidelity?

3. Teacher Dependence and Self-Teaching Bottleneck: The self-teaching results show minimal gains for the 7B model. What is the minimum capability threshold for a model to act as an effective diagnostician in DGRC? Can you analyze the correlation between teacher/model capability (e.g., pre-adaptation accuracy) and the quality of the generated curriculum to better characterize this limitation?

4. Data Contamination Analysis: Given the use of powerful pre-trained teachers and standard benchmarks, what steps have been taken to assess or mitigate the risk of data contamination? Can you discuss this potential issue and its possible impact on the reported results?

### Soundness
3

### Presentation
4

### Contribution
2
