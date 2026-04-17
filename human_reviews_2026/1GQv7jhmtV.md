# Euclid-Omni: A Unified Neuro-Symbolic Framework for Geometry Problem Solving

- Decision: Reject
- Scores: 2, 6, 4, 10

## Abstract
Euclidean geometry presents a compelling testbed for AI reasoning capabilities, requiring seamless integration of diagram understanding, logical deduction, and algebraic computation. Existing systems have either been narrowly scoped or struggled with challenging problems. We introduce Euclid-Omni, a unified neuro-symbolic framework that combines a formal geometry system with Large (Vision)–Language Models (LLMs and VLMs) to address both calculation- and proving-style problems across formal and natural languages, up to Olympiad-level difficulty. At its core, we develop Euclidea, a versatile geometry symbolic solver that automatically generates human-readable reasoning steps through logical deduction and algebraic solving. On top of this, we implement a comprehensive data generation pipeline that synthesizes symbolic problems, renders diagrams, and translates problems into natural language, yielding large-scale, diverse datasets for training LLMs and VLMs in different reasoning settings.
Experiments on multiple benchmarks demonstrate that Euclidea can tackle a broader range of problems than prior symbolic systems. 
Our trained VLMs achieve superior results on calculation tasks, while combining LLMs with Euclidea remains competitive with state-of-the-art systems on Olympiad-level theorem proving problems, despite using orders of magnitude less compute and data.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper aims to address the limitations of existing AI-based geometry problem solvers, which are often narrowly scoped to specific problem types (e.g., theorem proving) and require enormous computational resources for training. The authors first propose Euclidea, which is a symbolic solver that integrates a deductive database with an algebraic engine to generate human-readable solutions. Then, they propose Euclid-Omni, which is a data generation pipeline that synthesizes problems, renders diagrams, and translates symbolic solutions into natural language for VLM training. Experiments are conducted to evaluate the symbolic solver and the models trained on the synthetic data. The results demonstrate that Euclidea can solve more problems than prior symbolic systems , and the trained models achieve competitive performance on various benchmarks while using orders of magnitude less training data.

### Strengths
1. The paper provides many examples in the main text and appendices, which are helpful for understanding the implementation details. 

2. The proposed data generation pipeline is a good contribution to the field. It addresses a well-known bottleneck of data scarcity in geometric reasoning.

### Weaknesses
1. The proposed framework, Euclidea, appears to be heavily reliant on manually defined components. It is driven by a Deductive Database of pre-defined theorems and an Algebraic System that categorizes all equations into only four pre-defined types. I am doubtful that this manually curated set of theorems and equation types can cover the vast and complex landscape of real-world geometric reasoning. The completeness and generalization of this symbolic system are potential issues that are not sufficiently discussed.

2. The paper's description of its contributions, particularly the term "neuro-symbolic framework," is confusing. The abstract introduces "Euclid-Omni, a unified neuro-symbolic framework that combines a formal geometry system with Large (Vision)-Language Models (LLMs and VLMs) to address... problems". This phrasing is misleading for two reasons. First, it suggests that Euclid-Omni is the reasoning framework itself. However, after reading the paper, my understanding is that Euclidea is the purely symbolic reasoning system, while Euclid-Omni is the name for the data generation pipeline, which uses both the symbolic Euclidea and neural LLMs (for translation). Second, the abstract's phrasing implies that the problem-solving process is a deep integration of symbolic logic and neural reasoning. In reality, the core reasoning is performed entirely by the symbolic Euclidea system. The "neuro" components serve auxiliary, non-reasoning roles, such as translating solutions into "human-readable reasoning traces". This discrepancy severely misrepresents the paper's core contribution.

3. The experimental improvements in reasoning performance over existing baselines are not overwhelmingly significant, and furthermore, many key explanations and analyses of the results are missing (please refer to Q1-Q2 below).

### Questions
1. In the experiments, the authors use different settings for the two main tasks. In Section 4.2 (calculation), 20K training samples and the Qwen2.5-VL model are used. In Section 4.3 (proving), 100K samples and the Qwen2.5-Math-7B model are used. Could the authors explain the rationale behind these different data sizes and backbone models? Given the paper's emphasis on a "unified framework," why not train a single, unified multi-modal model on a combined dataset and evaluate it on both tasks?

2. Regarding the training data, could the authors provide a qualitative comparison between the datasets used by the baselines and the data generated by Euclid-Omni? While the paper emphasizes that the proposed method uses less data, it is possible that this performance gain comes from data specialization rather than just efficiency. Could it be that the baseline datasets are more general or diverse, whereas the Euclid-Omni data is more narrowly tailored to the specific types of problems in the evaluation benchmarks? This would explain why "less data" appears to achieve better results.

3. What is the fundamental difference between Euclidea and existing formal geometry systems like DD+AR and Inter-GPS? The described pipeline of iteratively applying deduction theorems (i.e, rules in some previous works) and updating known quantities via an algebraic solver seems conceptually similar to prior approaches. Is the core contribution of Euclidea a more advanced piece of engineering (e.g., a more comprehensive deductive database and a more precise algebraic system)?

4. The reasoning process in Euclidea involves enumerating inference rules from the deductive database to find applicable theorems. This sounds like it could create a significant efficiency bottleneck, especially as the number of known facts and theorems increases for complex problems.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This manuscript introduces a unified neuro-symbolic framework named Euclid-Omni, designed to solve Euclidean geometry problems. The framework combines LLMs and VLMs with a novel, versatile symbolic geometry solver called Euclidea. The Euclidea solver, which is the core of this framework, fuses logical deduction (a deductive database) with algebraic computation (an algebraic engine) to automatically generate human-readable reasoning steps, enabling it to handle both proving-style and calculation-style geometry problems. Furthermore, the manuscript constructs a comprehensive data generation pipeline. This pipeline can automatically synthesize symbolic problems, render corresponding diagrams, generate solution steps using the Euclidea solver, and finally translate this symbolic content into natural language . The resulting synthetic datasets can be used to train LLMs and VLMs.

### Strengths
The manuscript is well-structured and content-rich, constructing a complex neuro-symbolic framework that represents a significant workload. The method demonstrates high efficiency in solving Olympiad-level problems. Compared to the significant overhead of AlphaGeometry, Euclid-Omni's hybrid system achieves competitive performance using only a small number of training samples, reducing the training data by orders of magnitude.

### Weaknesses
1. In Section 3.1, the problem formalization relies on both "metric relations" and "diagrammatic relations". The paper states that diagrammatic relations (e.g., SameSide(a, b, c, d)) are "implicit but can be extracted from the diagram" . This extraction process is a critical, non-trivial step, yet its implementation is not detailed. How is this extraction performed?
2. The algebraic system in Euclidea (Section 3.1) categorizes equations into four types. While the first three (linear and log-linear) are solved systematically via Gaussian elimination , the handling of the fourth "complex" category (e.g., trigonometric or higher-order polynomial relations) is vague. The paper states simplification and substitution work "in many cases". This raises a key question: What happens when a complex equation cannot be simplified by this method? Does the system abandon that reasoning branch, or are there fallback mechanisms?
3. The claim of generating "human-readable" reasoning steps is made throughout the paper. However, the examples provided in Appendix C.1 consist of highly symbolic, step-by-step logical and algebraic derivations . While these proofs are more compact than those from PyEuclid (e.g., 13 steps vs. 32) , they are not "human-readable" in the sense of a prose proof and remain difficult for a non-expert to parse. How do the authors define and evaluate "human-readability"?
4. In Section 4.2, the VLM is trained on a mixed dataset of 10K synthetic samples from Euclid-Omni and 10K samples from Geo170K . This introduces a significant confounding variable. The strong performance reported could be largely attributable to the existing Geo170K dataset, which the paper notes was added to "better match the out-of-distribution diagrams present in existing benchmarks". To properly assess the contribution of the proposed data generation pipeline, a crucial ablation study is missing. What is the performance of the VLM when trained only on the 10K (or 20K) samples generated by Euclid-Omni?
5. The entire framework's success in training models relies on the quality of its synthetic data (Section 3.2). However, the paper does not provide a clear validation of this synthetic data's quality. While the generation pipeline is described, what measures or standards were used to assess the quality, diversity, and difficulty distribution of the resulting problems?

### Questions
Details see the Weaknesses.

### Soundness
3

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
4

### Summary
This paper proposes Euclid-Omni, a unified neuro-symbolic framework for Euclidean geometric problem solving. The main contribution of this manuscript lies in two parts. 
1. A problem solver for proofs and calculations (Euclidea). The workflow of Euclidea contains three sub-modules. a) The problem formulation module transforms natural language text and visual diagram into symbolic formulations (which is defined as “problem state” in the manuscript; b) The Reasoning Engine leverages the rules in an associated deductive database (SQL database) to calculate or prove the goal; and c) The solution generation module translate the complete solution generated by the reasoning engine into human-readable reasoning traces. 
2. A Data synthesis pipeline (Euclid-Omni) with four steps: a) Synthetic problem generation step leverages artificial construction rules to generate symbolic questions; b) The diagram rendering step generate a diagram according to the synthetic problem; the c) natural language translation step transforms the symbolic question into natural language via verified templates and refine them with an LLM into fluent statements, step-by-step solutions, and MCQ variants with plausible distractors; d) the task-specific packaging stem assemble calculation-style problems for VLMs and formal language problems for LLMs. 

The manuscript claims that symbolic solvers trained on the synthetic dataset can address Olympiad-level problems.

Note: although the authors mentioned Euclid-Omni in the title, it can also be interpreted as the name of a data synthesis method as mentioned in lines 257-260.

### Strengths
- Despite having moved a considerable amount of content to the appendices, the paper is generally easy to follow. The figures can help readers comprehend the proposed pipelines. 
- The manuscript provides solid technical contributions. Either Euclidea or Euclid-Omni (the data synthesis pipeline) is sufficient to support an individual publication. 
- The proposed Euclidea framework shows to be significant, since it shows to be the first one that can simultaneously solve calculation and the 2 proving tasks.

### Weaknesses
- The current organization of the manuscript feels somewhat overcrowded: the Methods section takes up too much space, which compresses the room available to clearly describe the experimental setup and results.
- The current manuscript lacks in-depth analyses of experimental results. Apart from knowing that the proposed method achieves better results, this reviewer is interested in knowing how the proposed method achieves such desirable results. The combined Experimental Results in Sections 4.1–4.3 amount to about half a page, and they are mostly descriptive statements of facts.
- This manuscript lacks adequate ablation studies. It is hard for us to attribute the better experimental results to the proposed Euclidea solver and the quality/comprehensiveness of the synthetic data. 
- The results in Table 2 are not comparable to some perspectives, since the authors trained the baselines with the synthetic data, but the results of baseline methods “are taken directly from prior works when available”.

### Questions
- This reviewer suggests the authors to separate Euclidea and Euclid-Omni (the data synthesis pipeline) into two separate publications. By doing so, the authors will no longer be necessary to delay some key details in their proposed method to the appendices, which this reviewer believes, will significantly enhance the reading experience of potential readers.

This reviewer respects the decision of the authors and acknowledges the technical contributions they have made. 

- This reviewer requests the authors to add the following experiments:
    - Use the same set of synthetic data to re-conduct training for Euclidea and the corresponding baseline methods. 
    - Use different datasets to train Euclidea, and compare the results with the one trained on the proposed synthetic dataset. 
    - Try removing/replacing the implementation method for the three modules/steps in Euclidea
    - Try removing/replacing the implementation for each of the 4 steps in the proposed data synthesis workflow.

### Soundness
2

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
This work introduces Euclid-Omni, a unified neuro-symbolic framework for solving geometric proving and calculation at IMO-level performance.
As in AlphaGeometry, a famous and powerful geometric prover, Euclid-Omni involves a symbolic deduction engine, a language model to propose auxiliary construction, and a data generator to synthesize pre-training data.
Better than AlphGeometry, Euclid-Omni uses much less training data and supports calculation-based problems.
Besides, Euclid-Omni can also take natural-language and diagram inputs and will release the whole framework.
Extensive experiments show that (1) the improved symbolic deduction engine outperforms existing ones (2) Euclid-Omni achieves comparable or superior performance on problems with natural language and diagram as inputs (3) Euclid-Omni achieves performance comparable to AlphaGeometry on IMO-level problems with orders of magnitude less training data.

### Strengths
1. The motivation is very solid and exciting. AlphaGeometry-like geometric provers achieve IMO-level performance and are interpretable and verifiable. This work not only extends them to calculation-based problems but also supports natural language and diagrams as inputs. More importantly, those works do not open-source some key components; hence, it is hard for the research community to follow up. If authors release the whole framework (better with data) as promised, it will be very helpful for the research community.
2. The authors carried out considerable work, notably refining the symbolic deduction engine, synthesizing data, and training both VLMs & LLMs.
3. Related works are comprehensively discussed and clearly written.
4. Experiments are comprehensive.

### Weaknesses
As a suggestion, instead of a weakness, incorporating a symbolic engine when taking natural language and diagrams as inputs would be better, in terms of interpretability and verifiability.

In my opinion, this is already an excellent work.

### Questions
1. How is the completeness (in terms of formal logic) of the rules in the deduction engine?

### Soundness
4

### Presentation
4

### Contribution
4
