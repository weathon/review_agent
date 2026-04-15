# Exploiting Code Symmetries for Learning Program Semantics

- Decision: Reject
- Scores: 8, 3, 3, 6

## Abstract
Large Language Models (LLMs) hold significant potential for automating program analysis, but current code LLMs face challenges in grasping program semantics. Our paper addresses this by formalizing program semantics through code symmetries and integrating them into LLM architectures for code analysis. We introduce a group-theoretic framework that defines code symmetries as semantics-preserving transformations, enabling precise reasoning within LLMs. Our solution, SymC, employs a novel variant of group-equivariant self-attention that is provably equivariant to code symmetries. We extensively evaluate SymC on four program analysis tasks, comparing it to eight baselines against eight code transformations. Our results show that SymC generalizes to unseen code transformations, outperforming the state-of-the-art code models by 30.7%. SymC, by design, stays invariant to semantics-preserving permutations, while state-of-the-art code models like WizardCoder and GPT-4 violate these invariances at a high rate (i.e., 14% and 43%, respectively).

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes SymC, an approach to training a Transformer architecture that is invariant/equivariant to dependence-preserving reordering of code. SymC's formal foundation is a group theoretic definition of invariance and equivariance. The paper defines this symmetry group over program interpretation graphs, graphs whose nodes are program instructions and whose edges indicate whether there is any execution path in which there is a direct dependence between data computed by the two instructions. The paper then relaxes this to the sound overapproximation of dependence graphs, and shows an implementation of self-attention that is equivariant to actions of this symmetry group. The paper evaluates the proposed model on a range of invariant tasks, evaluating on code transformations that fall within and outside the scope of the invariance. The paper shows that SymC is competitive or surpasses baseline models on nearly all metrics of interest.

### Strengths
* Broadly, the paper is very well written. The paper provides a clear description of all the background knowledge of symmetry groups, clearly grounds the theory in the task (symmetry groups of code ordering), explains the implementation reasonably well (invariant/equivariant self-attention), and has a clear evaluation.
* The problem domain itself is interesting and important, and the proposed solution is novel
* The evaluation is quite extensive, with strong results on all metrics of interest

### Weaknesses
* The one part of the paper that could be more clear is in the precise discussion of the implementation of SymC in a Transformer. Specifically:
  * I found the definition of the Aut(PDG) distance matrix to be a bit hard to reason about
  * I also wasn't sure what the relationship between this distance matrix and the standard use of positional encodings is.
* I would appreciate some discussion of why the F1/AUC results in Table 2 are not monotonic in the percentage of semantics-preserving permutations.
* The evaluation also lacks any quantification of variance in the results (e.g., standard error across different training trials).
* Minor typo: Section 3.3: "tofuture"

### Questions
* Does SymC use positional embeddings?
* Why are the F1/AUC results in Table 2 not monotonic in the percentage of semantics-preserving permutations?
* Could the authors provide examples of code where the relaxation of the interpretation graph to the program dependence graph is too conservative?

### Soundness
4 excellent

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper introduces SymC, a novel approach that leverages code symmetries, defined through group-theoretic principles, to enhance large language models (LLMs) for program analysis. By embedding a group-equivariant self-attention mechanism within the Transformer architecture, SymC achieves significant improvements in understanding program semantics. The method demonstrates strong generalizability across various code transformations, outperforming existing state-of-the-art LLMs, including WizardCoder and GPT-4, by substantial margins in four specific tasks.

### Strengths
1. The paper presents a unique and innovative approach to harnessing code symmetry, grounded in group theory, which stands out from previous methods that rely on ad-hoc heuristics. Instead of using these transformations for data augmentation, as is common in prior work, SymC ingeniously incorporates them into the attention layers of Transformers, showcasing a novel application.

2. SymC's performance is noteworthy, as it surpasses the baselines across the various tasks presented in the paper, sometimes by a large margin.

### Weaknesses
1. The paper could benefit from a more comprehensive comparison with related works, such as DOBF (https://arxiv.org/abs/2102.07492), which exploits code symmetry in pretraining through a deobfuscation objective, and CodeT5 (https://arxiv.org/abs/2109.00859), which leverages code symmetry in pretraining with identifier-aware data augmentation. These related works were not discussed or compared to the proposed method in the paper.

2. The evaluation framework relies heavily on four artificial tasks created by the authors, omitting well-established, practical benchmarks used commonly in the field. For instance, important code generation tasks like OpenAI HumanEval  (https://huggingface.co/datasets/openai_humaneval) and MBPP (https://huggingface.co/datasets/mbpp), as well as code translation, clone detection, defect detection, and code repair tasks from CodeXGLUE (https://github.com/microsoft/CodeXGLUE), are all relevant to the domain but were not considered. This absence of evaluation on existing benchmarks and comparison with related works raises questions about the paper's soundness and the model's real-world applicability.

3. The paper does not discuss potential limitations of the SymC model, such as its requirement for input code to be processed by a parser and a static analysis tool. This requirement may limit the model's applicability when dealing with incomplete or syntactically incorrect code, such as in code completion tasks or when faced with an empty Python block. While it is acceptable to establish certain assumptions for input code, these assumptions must be explicitly discussed rather than overlooked.

### Questions
1. Can the authors provide a comparative analysis of SymC with related works such as DOBF and CodeT5 that also leverage code symmetry?
2. Why were the evaluation tasks limited to four artificial tasks created by the authors?

### Soundness
1 poor

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors introduce a group-theoretic framework that defines code symmetries as semantics-preserving transformations, enabling precise reasoning within LLMs. SYMC employs a novel variant of group-equivariant self-attention that is provably equivariant to code symmetries.
The evaluation results show that SYMC generalizes to unseen code transformations, outperforming the state-of-the-art code models by 30.7%.

### Strengths
The idea of defining code symmetries as semantics-preserving transformations, enabling precise reasoning within LLMs is somewhat interesting. 
To evaluate the approach, four analysis tasks that require a deep understanding of code behavior such that they are expected to stay invariant to code symmetries were considered. Also a set of real-world semantics-preserving transformations
beyond PDG automorphisms to evaluate SYMC’s generalization in the experiments.

### Weaknesses
The paper needs more evaluations, e.g. an evaluation of the robustness of SYMC using the adversarial attack methods based on code transformations.
Some contents are not well presented/stated.

### Questions
Q: As stated in the paper, current code LLMs struggle with generalization to new code. Have you tried to evaluate the robustness of SYMC using the adversarial attack methods based on code transformations? The evaluation may make your method more convincing. 

Q: Have you tried to compare with "Graphcodebert: Pre-training code representations with data flow", which is a state-of-the-art method considering the inherent structure of code, in your evaluation? 

Q: Page 5, "PDG (VPDG,EPDG) is a super graph of IG, sharing the same vertices but having a superset of edges (EPDG ⊇ EIG), because we consider all memory accesses as aliasing, making PDG a conservative construction of IG",
If you "consider all memory accesses as aliasing", which is apparently a very weak encoding of the program semantics, it seems there would be too many aliases in the programs, making most statements unexchangeable to accomplish semantics-preserving statement permutations?

Q: Page 6, "Each entry dij is a 2-value tuple (pij , nij), indicating the shortest path from the lowest common ancestor of Vi and Vj , denoted as Tij , to Vi and Vj , respectively", is pij the positive distances and nij the negative distances as denoted in the next paragraph? Also, what do you mean by positive distances and negative distances?

Q: Page 2, "SYMC enforces its output to stay invariant via keeping its learned representation G-equivariant, where the code representation (e1, e2, e3, e4) is transformed into (e2, e3, e4, e1) xxx", should "(e2, e3, e4, e1)" be "(e2, e1, e3, e4)" as shown in Figure 1a?

Q: Page 9, the lines labeled 2nd, 4th, 6th in Figure 4a. Which lines are for Aut(PDG)-equivariant self-attention layers and which are for the Aut(PDG)-invariant ones?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work explores invariance to symmetries in code that do not change the semantics of the code. This notion is formalized via automorphisms of program interpretation graphs. To achieve equivariance (and invariance) to these automorphisms, the authors use a self-attention based model with pairwise features based on an invariant distance matrix.

### Strengths
1. Formalization of code symmetries as automorphisms of graphs is nice and seems like the correct formalism.
2. SymC model achieves equivariance to the code symmetries under consideration in a natural way, which is not too different from existing Transformer-based models.
3. Empirical results show that SymC outperforms strong baselines, while being small and robust to code symmetries.

### Weaknesses
1. Hard to understand exactly what program interpretation graphs and program dependence graphs look like, which is crucial to the paper.
2. Experimental details are lacking. What is the training procedure for SymC, is it just supervised training on the downstream task? How about for the other models? For Function Name prediction, do the LLMs take in just the text as input, and what exactly does SymC take as input there?
3. Computation of graphs associated to code may be costly and restrictive.

### Questions
1. Could you illustrate example program interpretation graph and program dependence graphs? This would be quite helpful for understanding.
2. How costly is obtaining the code graphs?
3. Why does SymC require "40,162 lines of code"? I'm curious as to what makes it require so much.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
