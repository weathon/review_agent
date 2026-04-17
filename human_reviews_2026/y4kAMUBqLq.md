# Local Success Does Not Compose: Benchmarking Large Language Models for Compositional Formal Verification

- Decision: Accept (Poster)
- Scores: 4, 2, 8, 4

## Abstract
Despite rapid advances in code generation, current Large Language Models (LLMs) still struggle with reliable and verifiable code synthesis in the presence of *compositional reasoning* requirements across multi-function programs. To study this systematically, we introduce DafnyCOMP, a benchmark for generating compositional Dafny specifications for programs consisting of multiple interacting functions with non-trivial data dependencies. Unlike prior benchmarks that focus primarily on single-function annotation, DafnyCOMP targets programs composed of 2-5 functions arranged in acyclic call graphs, requiring specifications that establish correctness across component boundaries.

DafnyCOMP contains 400 automatically synthesized programs: 300 chain-structured instances and 100 non-chain DAG instances generated from 10 topology templates. We evaluate frontier LLMs from major providers under a unified prompting and verification protocol. While these models achieve high syntactic well-formedness (>99%) and moderate end-to-end verification (>58%) on prior single-function Dafny verification benchmarks, they obtain near-zero end-to-end verification on DafnyCOMP. On the chain split, even the strongest evaluated model reaches only 2% verification at Pass@8, with most models below 1%; the difficulty persists under broader topologies and stronger test-time scaling.

Our analysis identifies three recurring failure modes that hinder cross-functional reasoning: *specification fragility*, *implementation--proof* misalignment, and *reasoning instability*. DafnyCOMP provides a diagnostic benchmark for tracking progress in verifiable code generation, highlighting that bridging local correctness to compositional verification remains a key open challenge.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces DAFNYCOMP, a benchmark targeting compositional formal verification for LLMs. It assembles multi-function programs (2–5 functions). Across a series of frontier models, the paper finds a big gap between high syntax success and very low verification results on the benchmark. The authors analyze failures into three primary failure modes: specification fragility, implementation-proof misalignment, and reasoning instability.

### Strengths
1. The evaluation is thorough, covering a wide range of models
2. The failure taxonomy is insightful, well-articulated, grounded with examples, and provided with the frequencies.
3. The benchmark is challenging, exposing the limitations of current LLMs.

### Weaknesses
1. The benchmark is only done on Dafny. We don't know whether results and claims will still hold for other frameworks such as Verus / Boogie.
2. The synthesis pipeline uses the LeetCode programs. LeetCode skews toward short, single-goal algorithms, but may deviate from the real-world software use cases.
2. There is a limited trial in trying to solve the challenges. E.g., adopting a verifier-in-the-loop repair / agentic setup would be beneficial.

### Questions
1. Why do you choose LeetCode problems as the pool programs? Is it possible to switch to other datasets as the initial pools?
2. Will the compositional strategy reflect the distribution of natural real-world programs?
3. How will this methodology translate to other frameworks such as Verus? Does such a similar dataset / idea exist in the Verus or other frameworks' prior work? If not, will your work's pipeline generalize to other frameworks?
4. How to improve the performance of your benchmark? Consider adding a baseline to improve the results

### Soundness
3

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
This paper presents a benchmark set, DafnyComp, to evaluate LLMs ability to perform compositional
reasoning. The benchmark set contains Dafny programs that are compositions of 2-5 functions. The goal
is to prove these programs correct. To do this, one has to generate pre and postconditions (requires
and ensures) for each function in a way that they compose to yield a proof for the main program.
The LLMs task is to annotate the functions enough to enable Dafny verifier to complete the proof 
(by discharging the verification conditions using an SMT solver).

The main result shows that LLMs across the board are really bad at doing this annotation task. 
They are able to do so for single functions to some extent, but really struggle to do that for
multiple functions in a way that the annotations compose.

### Strengths
Strengths:
1. DafnyComp is a valuable benchmark since state-of-the-art LLMs currently perform very poorly
(about 3% for the best models).
2. DafnyComp tests the ability to think compositionally, which is useful to evalaute.
3. DafnyComp is also interesting because models can generate correct syntax (for annotations) with
high probability, so it does demonstrate that models really struggle with compositional reasoning
(and not getting the syntax right for a potentially low-resource language).
4. The pipeline for constructing DafnyComp guarantees that the benchmarks are all high quality
(i.e. they all can be formally proved correct.)

### Weaknesses
Weaknesses:

1. The benchmarks in DafnyComp are artificial - the individual functions are natural, but the
compositions are contrived (authors, please correct me if I am wrong.)  So, the 92% drop from
single-function to multi-function programs can also be attributed to the fact that the single-function
programs are natural programs that people may write and care for, and the multi-function programs
are more arbitrary. In other words, if I were to write completely unnatural single function programs,
wouldn't LLMs performance already deteriorate? That puts the main claim into question: maybe it is 
not the lack of compositional reasoning, but just that the compositional programs are unnatural. 
If the single-function programs were unnatural, or the compositional programs were natural, then 
would we get different conclusions?

2. The paper makes very strongly worded claims all through the paper, but there are a lot of 
factors that could be responsible for the observed behavior. Many such factors are overlooked and
not discussed.

 2.1 If the function call graph is linear (which means it can be topologically sorted), then couldn't
we turn the multi-function benchmarks to single-function benchmarks? If so, then would LLMs 
perform better on these derived single-function benchmarks? And would this finding impact the main
claims of the paper?

 2.2 Is Dafny prover guaranteed to prove things if they are provable -- couldn't the SMT solver fail -- 
e.g. time out -- even when the verification condition holds true? Would this not somehow impact
the conclusions? For example, purely hypothetically, maybe the LLM is generating the correct 
annotations and the SMT solver is failing to prove them? I know that this is unlikely, but how 
are we guarding against that possibility?

3. There is not enough explanation for how benchmarks (and their correctness assertions) are
generated.

 3.1 It is unclear how the assertions of the composed program are generated. Line 215 says "We execute
each composed program against the reference unit tests from LeetCodeDataset" -- but my understanding
is that you are taking individual functions from LeetCodeDataset and then creating new compositions
(with 2-5 functions). So, LeetCodeDataset would not have reference unit tests for these composed 
programs. There is clearly some misunderstanding here on my part.

 3.2 Some of these misunderstandings could be avoided if you had a running example either in the main
paper or at least in the Appendix. It would be good to have an example benchmark in DafnyComp. My best
Source is Appendix G. But most of the programs there have 1 function -- clearly, they can't be part
of DafnyComp. So, that adds to more confusion. G.4.2 is the first place where there is a program that
uses multiple functions. It has one function, reverse_7, that is unused, that is a but confusing, but
maybe DafnyComp has benchmarks that contain unused functions. But what is really confusing is that
"nextBeautifulNumber_769" is not a linear composition of two functions, but a loop over the "isBeautiful"
function. 

 3.3 The examples in Appendix G just give some code snippet and a Dafny error. That is not helpful. 
More explanation is needed. I could not understand most examples. For example, take G.1.1: sum >= 0
is the only postcondition there. It is clearly implied by the loop invariant (because it contains
a conjunct sum >= 0) -- but Dafny error says "a postcondition could not be proved..." ?

Other questions:

- Contribution (ii) on Line 111 is related to a single function/procedure, right? So, it is not
related to compositions, is that true?

- Does DafnyComp have 300 benchmarks or 900 benchmarks (Table 2)? Or Table 2 is analyzing 3 runs for 
each of the 300 benchmarks?

- Line 422-423: "not a typo but a systematic failure to..." -- what do you mean by "not a typo"?

- Line 427-428: "attention mechanisms excel at ...implementations." I think there is no experiment
in the paper that provides evidence of this claim, and I am not even sure what this claim means.

### Questions
Please refer to the Weaknesses section for the questions.

### Soundness
2

### Presentation
2

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
The paper introduces DafnyComp, a benchmark to evalaute LLMs in . Unlike existing benchmarks, DafnyComp focuses on programs consisting of multiple functions interacting with each other. Experiments on 13 frontier LLMs show that this leads to significantly a more challenging and realistic benchmark.

### Strengths
* The paper introduces DafnyComp, a benchmark for evaluating LLMs on the generation of compositional specifications in Dafny. Existing benchmarks for verifiable code generation only test problems with a single function. DafnyComp focuses on programs composed of multiple interacting functions with necessary data dependencies, addressing a significant gap in the real-world applications of formal verification. 
* A comprehensive set of LLMs is evaluated on DafnyComp. Results show that multi-function compositional problems present a major challenge to LLMs compared to simpler problems.
* The authors performed a detailed and insightful analysis of LLMs' failures on DafnyComp.

### Weaknesses
* The benchmark is relatively small (300 programs). Is it possible to scale the benchmark construction method in Sec. 3 to collect more programs?
* The paper focuses on Dafny. It would be great to discuss potential extensions to other systems such as Lean and Verus.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents DAFNYCOMP, a collection of 300 programs used to provide stress on the compositional generation of specifications in Dafny by chaining 2-5 functions with real dependency of data. In 13 frontier LLMs, syntax is well-behaved (96%-99% Pass at t8) but the verification is still low (3.7 percent overall; best model about 7 percent), indicating that there is a big discrepancy between syntax and semantic correctness. The failures the authors consider are three in general, namely specification fragility, implementation-proof misalignment, and reasoning instability.

### Strengths
- Timely evaluation of compositional verification
- Clear, reproducible benchmark spec with Dafny ground truth.
- Concrete numbers that expose a large syntax verification gap. 
- Insightful error analysis

### Weaknesses
- Data programs and specs are generated using an LLM from a single family employed in evaluation, which might be biased toward one style and negatively impact others.
- The evaluation is limited to topologies: only chains are considered, whereas real systems include trees, DAGs, shared states, and more complex dependencies.
- There is a lack of evaluation on more advanced test-time scaling strategies, such as multi-turn self-refinement.
- The evaluation described in the main text does not consistently standardize or control details about compute and prompting across models (tokens, temperature, verifier retry policy).

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
