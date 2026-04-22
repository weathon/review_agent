# Post Hoc Neuro-Symbolic Verification on Instruction Following of Language Models

- Avg Score: 2.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2

## Abstract
Large Language Models (LLMs) are increasingly used for real-world problem-solving and decision-making. However, LLMs may not follow instructions, with subtle behavior that is hard to detect and diagnose. The impacts of instruction-unfollowing behavior may be further magnified in an LLM agent along its reasoning chain. This paper presents NSVIF, a novel framework for post hoc verification on instruction following of LLMs. At its core, NSVIF abstracts instruction-following verification as a Constraint Satisfaction Problem (CSP), where both instructions and LLM outputs are represented as structured constraints, including symbolic and neural constraints. NSVIF introduces a neuro-symbolic solver that embraces symbolic reasoning and neural inference—the former offers sound logic while the latter detects semantic violations. We curated a comprehensive benchmark, VIFBENCH, to evaluate instruction-following verifiers, and developed a neuro-symbolic-guided synthesis method to construct data in a scalable and high-quality manner. We show the effectiveness of NSVIF on VIFBENCH, where NSVIF significantly outperforms the existing baselines. Our work shows that unified symbolic verification with LLM-guided reasoning enables effective, reliable, and interpretable analysis of LLM instruction-following behavior.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper present a neurosymbolic verifier to check whether an LLM-generated response satisfies the constraints indicated in the input. It does so by generating a set of formal and neural properties to check for, and then checks if the generation satisfies the properties.

### Strengths
- The overall idea is clear (though there are questions about details, see below)
- The benchmark seems cool

### Weaknesses
- It is unclear how a neural verifier can truly be used for "verification" - at best it might be predictive, and with sufficient empirical testing for calibration, could provide some form of statistical results, but I wouldn't call it verification.
- The "Formula Contextualization" presentation is too vague to understand what is going on. It is unclear how the first-order predicates are grounded, or evaluated over the inputs.
- The compositional verification formula is not quite sound, or not accurately presented: it should be checking for all the constraints, but the fourth equation (unnumbered) just seems to check one?
- The novelty of the approach is not clear wrt the following papers. A better discussion of novelty is in order.

Olausson, Theo, Alex Gu, Ben Lipkin, Cedegao Zhang, Armando Solar-Lezama, Joshua Tenenbaum, and Roger Levy. "LINC: A Neurosymbolic Approach for Logical Reasoning by Combining Language Models with First-Order Logic Provers." In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 5153-5176. 2023.

Zhang, Yedi, Yufan Cai, Xinyue Zuo, Xiaokun Luan, Kailong Wang, Zhe Hou, Yifan Zhang et al. "Position: Trustworthy AI Agents Require the Integration of Large Language Models and Formal Methods." In Forty-second International Conference on Machine Learning Position Paper Track.

### Questions
- How are the first order literals in the formulas grounded?
- How is a neural verifier actually guaranteed?
- What is the novelty of this paper wrt previous work?

### Soundness
2

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
The papers aims to address the issue of verifying whether LLMs follow instructions given through post-hoc verification. It presents a high-level idea of encoding the given input text as a set of logical constraints, some symbolic and some neural, which are then jointly checked for on the output resulting in a system they call NSVIF. The work creates a new benchmark called VIFBENCH which contains logical specifications for the given tasks. This benchmark is then used to test the hypothesis that NSVIF is better at verifying the derived constraints than the baseline approach of using 'LLM-as-a-judge'. The results show that NSVIF does better than the baseline by about 20-25% in precision and 40% in recall.

### Strengths
- VIFBENCH is a nice concrete outcome of the paper, which could be used in evaluation in other works.

- The general plan outlined in Section 3 and Figure 2 aims for a modular pipeline. The purely symbolic constraints are checked with the Z3 SMT solver whereas the subjective constraints using LLMs.

- The overall direction on verifying outputs of LLMs using symbolic methods, at least partially, is good.

- The results are positive, although the experiments run are partial and don't test the hypothesis fully. The paper explains the suspected reasons why the evaluated LLMs get stuck and possibly why they don't perform well.

### Weaknesses
- The main text of the paper does not give much details about VIFBENCH. It says the benchmark is comprehensive but provides little justification for why so. One of the challenges in such evaluations is to avoid biasing the benchmarks to a particular outcome. How did you avoid it?

- The most problematic part of the paper is Section 3. Beyond giving a high-level architectural roadmap, there are almost no details on how and why NSVIF is the way it is. This makes it difficult to draw well-grounded conclusions from the reported experiments in the paper. I will highlight several sources of unsoundness next:

1) There are vague statements "use a neural parser to analyze and parse the implicit constraints" that elide details. What is implicit constraints? How did you check that these constraints are accurately derived for the benchmarks or test subjects?

2) A common issue when trying to solve neuro-symbolic constraints is the deal with variables that are constrained in both symbolic and subjective constraints (see for example early works like [a] which highlight the challenge in Section 3 and 4). How are the constraints really encoded? It is possible to encode them either as symbolic or as subjective ones. How do you choose? Multiple encodings in either to be used but the paper is quite thin on technical details. 

3) Another significant source of errors in your design is with the executor. The executor relies on LLMs and best-effort loop to check the constraint satisfaction, but its incomplete and the paper doesn't quantify how this affects the final conclusions drawn.

4) Another issue is in formula contextualization. The neural paraphraser can get things wrong or over-concretize things to a specific context. This appears to be a somewhat arbitrary choice in the methodology. When do you decide between concretizing values for a symbolic variable vs. keeping it symbolic? E.g. for "sports", does the paraphrases specialize it to a certain sport keep it fully abstract? The difference matters because providing the first-order logic formula might change the outcome significantly.

- The evaluation methodology is incomplete and quite problematic in its rigor in eliminating sources of unsoundness. 

1) Models that infinitely think or fail to respond cause issues in the approach. But they could have been eliminated or substituted with outputs of other models. 

2) The analysis of the experiments states that "LLM-based verifiers do not understand constraint instructions at all." I am not sure whether this statement is true of all such verifiers. And if so, then what is the point of reporting on the final result which is clearly conditioned on this working correctly but requires a lot else to work. 

3)  LLMs are reported as inconsistent due to non-determinism. The evaluation results present no standard deviations to quantify the same.

I understand that the experimental methodology has limitation inherent of evaluating black-box LLMs. Nonetheless, it is also difficult to draw scientific conclusions when the experimental methodology ran into so many internal inconsistencies.



[a] "Neuro-Symbolic Execution: Augmenting Symbolic Execution with Neural Constraints", Shen et al., In NDSS 2019.

### Questions
- Can you clarify why you believe the conclusion you draw is accurate, despite the numerous ways in which evaluation methodology itself could fail? You have stated several ways upfront in the paper in how so.

- LLMs are used in NSVIF such as for a neural parser. Is this the same as that evaluated against in 'LLM-as-a-judge' baseline? What issues do you foresee in using the same LLM in both vs. different ones?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a technique and benchmark for verifying instruction following of LLMs. They propose a technique that breaks the constraints into two kinds: symbolic constraints, and neural constraints, each of which are verified by independent modules that then get combined and verified together via an SMT program. They also propose a new benchmark that contains verifiable constraints to accurately evaluate how well their technique can verify the output of an LLM. They conclude with an evaluation of their technique in comparison with an LLM as a judge.

### Strengths
1. The authors are addressing an important problem with LLM evaluations, which is how do we know that all the instructions from the prompt are being followed. I also appreciate the construction of a benchmark that specializes in doing so.
2. The technique seems quite sound, with the authors offloading most of the reasoning parts to the SMT solver, while using LLMs to extract and validate intermediate constraints.

### Weaknesses
1. Evaluation: I feel that the evaluation baselines are a bit lacking. First off, not enough models are used here in the evaluations. There are only 2 families of models being evaluated, non of which are open-source, and only one family of models is being used with NSVIF. I would like to see a more thorough evaluation. Second, I find it very hard to believe that *all* 4 models simply declare the outputs as satisfying the instructions, and on closer observation, I see that the prompt is a one-shot prompt with only 'sat' as an output in the examples. This seems suspicious: at best, this baseline is a strawman that is instructed to only output sat unless you actually give an example of what 'unsat' means in the prompt. I am fine with the prompt being two-shot to achieve this. Third, you can come up with another baseline that takes an autoformalization approach: give the LLM the instruction and the answer, and ask it to come up with an SMT program to verify the answer. If the LLM cannot do so, that further justifies a need for NSVIF, but I would like to see that experiment.

2. Some details are missing from the benchmark creation section of the paper. How do you come up with the initial seed predicates? What is the distribution of neural and symbolic predicates? Did you try disjuncting constraints? What is the final distribution of the generated problems with respect to constraint complexity? How do you come up with symbolic predicates v/s neural predicates (e.g. do you actually write the symbolic module that validates the predicate)?

### Questions
See weaknesses. Also:
1. Did you try if LLMs can show more improvements on tasks if given feedback via NSVIF v/s LLM as a judge?

### Soundness
3

### Presentation
3

### Contribution
3
