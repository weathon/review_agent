# Refactoring Codebases Through Library Design

- Avg Score: 5.50
- Decision: Reject
- Scores: 10, 4, 4, 4

## Abstract
Maintainable and general software allows developers to build robust applications efficiently, yet achieving these qualities often requires refactoring specialized solutions into reusable components. This challenge becomes particularly relevant as code agents become used to solve isolated one-off programming problems. We investigate code agents' capacity to refactor code in ways that support growth and reusability. We first investigate what makes a good refactoring, finding via asymptotics analysis and a human study that Minimum Description Length best aligns with developer preferences for code refactoring quality. We then present both a benchmark and a method for refactoring: MiniCode, a benchmark where multiple files must be refactored into a shared library, and Librarian, a sample-and-rerank method for generating reusable libraries. We compare Librarian to state-of-the-art library generation methods, and study it on real-world code bases.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
2

### Summary
The paper introduces MINICODE, an open-ended benchmark for refactoring multiple source files into a reusable library, and LIBRARIAN, a sample-and-rerank approach to synthesize such libraries. A comparative study finds that Minimum Description Length aligns best with developer preferences for refactoring quality; the method is further validated on real-world repositories, showing promising practical implications.

### Strengths
+ The investigated problem is clearly defined, novel and important to the community. MINICODE emphasizes open-ended library design, objective verifiability via unit tests, and large-context understanding across multiple files, addressing gaps in prior repo-level benchmarks.
+ LIBRARIAN combines sample-and-rerank with semantic clustering and a progressive, cross-cluster library accumulation strategy, which is practical for long-context constraints.
+ The paper compares Tokens/MDL vs. CC/MI, shows MDL better promotes reusable abstractions, and corroborates this with a human study.
+ Evaluations show strong empirical results. LIBRARIAN shows above 90% pass rate on CodeContests, which is a successful practical application of library learning to real software projects.

### Weaknesses
- In practice, a single file often mixes heterogeneous concerns (utility helpers, adapters, domain logic, classes), so the reviewer is not sure that file-level clusters could be too coarse.
- Figure 6 is a case study refactoring the code from the HuggingFace Transformers codebase. However, it is a little complex to understand. Authors are suggested to walk through the end-to-end pipeline that produced this figure.

### Questions
No question.

### Soundness
4

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
This paper studies code refactoring in software engineering, focusing on maintainability and reusability. The authors analyze different metrics for evaluating refactoring quality and find that Minimum Description Length (MDL) is the best metric. To evaluate and advance research on code refactoring, the paper introduces the MINICODE benchmark and the LIBRARIAN method.

### Strengths
- MINICODE offers a practical and effective benchmark for evaluating code refactoring
- The proposed metric MDL is reasonable and interesting.

### Weaknesses
- The proposed sample-and-rerank approach is relatively simple, and the methodological insights it provides are limited.
- The main risk of using MDL is that it can be heavily influenced by a single model. The paper only briefly discusses cross-model agreement for MDL in Section 6; a more detailed analysis would make the claim more convincing.
- Even if the refactored code passes all unit tests, there is still a risk of semantic inequivalence with the original code. The paper lacks an analysis of this risk.

### Questions
- Are there any cases where refactorings produced by LIBRARIAN in real projects have been successfully merged into the community codebase?
- Typos: line 218: “set of set of”; line 273: “more more.”
- Missing citation for the MI metric.
- What does “change from non-refactored” mean in Figures 2 and 3, and how is it calculated?

### Soundness
2

### Presentation
2

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
This paper addresses the problem of refactoring existing code to generate library code that is then used for rewriting the original code snippets. The paper assembles a benchmark called MiniCode and proposes a technique called Librarian. The paper compares different ranking methods and finds that minimum description length (MDL) to be a suitable metric to rank different refactoring suggestions. The evaluation on MiniCode shows effectiveness of Librarian.

### Strengths
- Code refactoring is an important software engineering activity. This paper demonstrates progress on this problem using a pipeline of clustering of code by natural language summary, cluster-specific library extraction and then rewriting the complete code corpus.
- It assembles a benchmark taking code contest solutions, previous refactoring benchmarks and small sets of related files from transformers and diffusers libraries. The resulting refactorings are ranked using MDL and evaluated for correctness using tests. It reports better refactoring accuracy than closely related work Regal on a subset of the benchmark.
- The comparative analysis between MDL, tokens and software engineering metrics is interesting and justifies the use of MDL in ranking.

### Weaknesses
- While the problem of refactoring is important, the proposed method is evaluated in limited setting. It does not present results at large scale where refactorings are most important and useful. Though the paper states that the proposed method is evaluated on "real-world code bases", the scope is restricted to a total of 3 tasks with 10 files each from 2 repositories.
- The paper's novelty over past work, Regal, is limited as both of them apply clustering based refactoring. 
- The study of different metrics showing MDL > tokens > MI and that test-time scaling can help are the interesting parts in the paper. As an aside, I could find a citation for the MI metric; please add.

### Questions
- In coding contest setting, the same problem can be solved using different algorithmic techniques. How well does the clustering work in such a setting? What is the quality of clustering if you were give entire code repositories (e.g., transformers) rather than hand-selected 10 files?
- As noted in the experiments, Librarian can create functions that are used only once. Curious if they are used within the refactored code itself (e.g., private methods) or outside?
- How does the performance vary if you use GPT models instead of o4-mini on CodeContents?

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
This paper presents LIBRARIAN, a method for refactoring multiple code files into reusable libraries, and MINICODE, a benchmark for evaluating library design. The authors investigate what makes good refactoring metric through human study and asymptotic analysis, finding that MDL aligns best with developer preferences. They demonstrate their approach on competition programming and real-world repositories like HuggingFace Transformers and Diffusers.

### Strengths
- This paper is well written and achieves impressive real-world validation by refactoring HuggingFace production code with 67% MDL reduction while maintaining correctness.
- This work provides systematic comparison of multiple metrics through asymptotic analysis and human studies, finding MDL superior to traditional software engineering metrics.

### Weaknesses
- This evaluation covers only 10 Transformers files and 2 Diffusers tasks, which seems insufficient to support claims about general applicability to real software projects.
- This human study with only 12 participants lacks statistical power to distinguish between MDL and tokens metrics, yet the authors make strong claims about MDL superiority.

### Questions
- Could you provide concrete examples showing these improve code quality beyond just MDL score?
- How does LIBRARIAN's performance degrade with larger cluster sizes S? The paper fixes S=3 for CodeContests but S=5 for Transformers without justification.

### Soundness
2

### Presentation
3

### Contribution
2
