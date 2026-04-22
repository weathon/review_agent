# Divide and Abstract: Autoformalization via Decomposition and Abstraction Learning

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Existing approaches to autoformalization---the task of translating informal mathematics into formal machine-verifiable languages---rely heavily on pre-defined libraries and expect LLMs to directly generate complete formalizations. These approaches face three fundamental limitations: they are bottlenecked by existing abstractions, they have difficulty handling the complexity of realistic statements, and they do not transfer well across formal languages.
We propose $\textit{Divide and Abstract (DNA)}$, a zero-training framework that addresses these challenges through a two-phase approach.
First, $\textit{DNA}$  extracts common mathematical concepts from the entire corpus and formalizes them as reusable abstractions, extending the target language's capability.
Second, $\textit{DNA}$ hierarchically decomposes new statements into structured informal clauses, translates each clause using the learned abstractions, and composes them into complete formalizations.
Our evaluation on the LeanEuclidPlus and ProofNet-Hard benchmarks demonstrates consistent improvements across multiple model families, achieving up to $\textbf{8.6}\times$ performance gains over baselines.
Notably, $\textit{DNA}$  enables smaller models to match baselines using much larger models, and shows particularly strong performance on complex mathematical statements requiring nested reasoning. Furthermore, our framework requires no training on target languages, making it effective for low-resource domain-specific languages. Our code is available at https://github.com/marcusm117/DNA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents Divide and Abstract (DNA), a novel zero-training framework for autoformalization that addresses the generalizability challenges in translating informal mathematics into formal languages. The framework consists of two phases: Phase I extracts common mathematical concepts from corpora to extend the target language's capabilities, while Phase II decomposes complex statements into structured clauses for systematic formalization. Evaluated on LeanEuclidPlus and ProofNet-Hard benchmarks, DNA demonstrates significant performance improvements across multiple model families, achieving up to 9.6× gains compared to baseline approaches and enabling smaller models to match larger baselines on complex mathematical statements.

### Strengths
1.The framework's combination of corpus-driven abstraction learning and hierarchical statement decomposition represents a novel contribution to autoformalization research, addressing fundamental limitations of existing approaches that rely on direct generation of complete formalizations.
2. The comprehensive evaluation across diverse model architectures (GPT-4.1/5, Claude-4-Sonnet, Qwen3 variants) and benchmarks demonstrates consistent performance improvements, with particularly impressive results on challenging domains where specialized models previously failed entirely.
3. The zero-training design makes DNA highly practical for low-resource domain-specific languages, eliminating the need for extensive training data while still achieving state-of-the-art performance through its innovative abstraction learning mechanism.

### Weaknesses
1. The six-step language extension process and four-step decomposition pipeline introduce significant complexity that may limit adoption, particularly for researchers and practitioners without extensive expertise in natural language processing or formal mathematics.​
2. While the paper demonstrates performance improvements, there is limited qualitative analysis of how the learned abstractions compare to human-crafted ones beyond basic correctness metrics, which would strengthen the validation of the framework's core innovation.

### Questions
The abstract mentions that code is available at https://github.com/anonymousauthor567/DivdedAndAbstract, but the repository appears to be empty. Will the code be available?

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
This paper proposes Divide and Abstract (DNA), a two-stage, training-free framework for autoformalization of mathematical statements. The Divide stage first extracts common mathematical concepts from a corpus and formalizes them into reusable abstractions. The Abstract stage decomposes each statement into structured informal clauses and uses the abstractions produced above to formalize them. Concretely, the Divide stage consists of concept extraction, concept filtering, concept dependency graph construction, concept formalization, refactoring, and documentation update steps; the Abstract stage consists of decomposition, formalization, and composition steps. Evaluation on subsets of LeanEuclid and ProofNet demonstrates the effectiveness of the DNA method across many LLMs.

### Strengths
1. The overall approach is very intuitive, while the experimental results show that it is effective. The results in Table 5 and the analysis in Section 4.3 indicate that the Divide and Abstract stages are complementary and produce synergistic benefits; in particular, their combination appears to yield a substantial performance boost on LeanEuclidPlus.

2. The proposed method has high potential value for low-resource formal mathematical languages. Beyond low-resource languages, it should also be effective when facing rapidly evolving formal languages (for example, different versions of mathlib in Lean 4 as noted in Line 106). This aligns with the current trend of fast development in formal systems and their mathematical standard libraries.

3. The paper addresses a long-standing but thorny problem: [missing mathematical dependencies](https://github.com/jsm28/IMOLean). The lack of formalized expressions for certain mathematical concepts in libraries such as mathlib creates difficulties for formalization; this issue has been widely overlooked by many autoformalization efforts. This paper partially addresses the problem by constructing the Abstract stage. (Although the example in Figure 1 is a simple composition of concepts, fully and rigorously formalizing complex mathematical concepts and structures and feeding them back into libraries like mathlib remains beyond the reach of current models.)

### Weaknesses
1. The effectiveness of the Divide stage appears to be limited to problems the LLMs have not seen (e.g., certain geometry problems). From Table 5, the performance gains of Divide stage on the ProofNet-Hard dataset are marginal. Those problems largely depend on mathlib (which is likely included in LLM pretraining data).

2. The experimental baseline setup is somewhat misleading: Line 292 states that helper definitions are withheld in the baselines. Given that these Lean definitions are typically absent from training corpora and resemble a DSL, withholding helpers imposes an unnecessary handicap on the baselines and may exaggerate the contribution of Divide. A more reasonable comparison would retain the existing pipeline but add a baseline that is provided the helper definitions (without the Divide stage).

3. The method’s generalizability is questionable. The approach seems to cover a single topic well, but extending the same method to other domains or languages appears to require substantial prompt rewriting for each component of the whole pipeline, demanding significant effort and therefore limiting transferability across different DSLs or math domains.

4. Minor issues suggesting the manuscript may have been prepared in haste: the code link given on Line 29 does not contain any actual code; Line 37 mentions "three transformative applications" but does not specify what they are; and on Line 160 both quotation marks are closing quotes.

### Questions
1. How are the correctness and recall metrics in Table 1, Table 2, and Table 3 defined? The main text does not appear to provide definitions or evaluation protocols for these metrics—please supply precise definitions and describe how they were computed.
2. Dataset choice and subsets. The paper evaluates on subsets (LeanEuclidPlus and ProofNet-Hard) rather than the full LeanEuclid and ProofNet collections. Please clarify:
   - What specific practical difficulties may arise if one attempts to apply the DNA method directly to the full LeanEuclid and ProofNet datasets?
   - Will the selected subsets and any scripts used to produce them be released publicly to facilitate reproducibility?

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
4

### Summary
The paper “Divide and Abstract: Autoformalization via Decomposition and Abstraction Learning” introduces DNA, a zero-training framework for translating informal mathematics into formal languages. It combines abstraction learning, which extracts and formalizes reusable concepts to extend target libraries, with hierarchical decomposition, which breaks complex statements into manageable clauses for accurate formalization. Tested on LeanEuclidPlus and ProofNet-Hard, DNA achieves up to 8.6× gains over baselines and enables strong generalization to low-resource languages, showing that decomposition and abstraction jointly enhance autoformalization performance.

### Strengths
- Dividing a complex task like autoformalization into multiple stages is very natural, and I particularly appreciate the authors’ idea of constructing a dependency graph.
- The training-free framework is appealing and offers good portability.

### Weaknesses
- The core operations of 'abstraction learning' and 'hierarchical decomposition' are not rigorously formulated. For example, abstraction learning is described as extracting three types of concepts—definitions of mathematical objects, relations between objects, and functions mapping objects to objects—but the boundaries between these categories (e.g., relations vs. functions) are unclear. Moreover, the expected formal output for complex informal statements (such as “the space of continuous functions from a compact manifold to a Banach space forms a complete metric space under the supremum norm”) remains somewhat vague.
- The paper lacks a qualitative study, particularly on non-Euclidean problems. The chosen Euclidean geometry tasks appear somewhat artificial and simplistic. It would strengthen the work to validate and qualitatively justify the proposed framework on more realistic datasets such as FATE [1].

[1] https://frenzymath.com/blog/fate/

### Questions
- What are the algorithms for step 4 and 5? And how to interpret the results of Table 3?

### Soundness
3

### Presentation
2

### Contribution
2
