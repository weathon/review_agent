# To Guide or Not to Guide: Sparse Transductive Guidance in Program Synthesis

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 6, 2, 2

## Abstract
Program synthesis faces the dual challenge of achieving high success rates while maintaining interpretability and generalization, motivating hybrid approaches that combine complementary learning paradigms.
Integrating transductive methods, which provide strong predictive power by directly mapping inputs to outputs but lack interpretability, with inductive methods, which excel at producing explicit and interpretable programs, creates a new opportunity for programming-by-example.
While recent work has explored this integration through transductive guidance, we show, that permanent transductive guidance can, and in practice does, mislead search by overriding inductive reasoning strategies that would otherwise succeed.
To address this limitation, we introduce TIIPS, a novel framework that, for the first time, applies transductive assistance sparsely and selectively to inductive synthesis. 
TIIPS adopts a teacher-student paradigm, where guidance is provided selectively, activated only when inductive synthesis fails, thereby preserving the natural problem-solving capabilities of inductive approaches.
Experiments on two standard programming-by-example domains (string and list manipulation) demonstrate that TIIPS outperforms related work, solving more tasks and producing more robust solutions, particularly under distribution shifts. 
These results show that the timing and extent of transductive guidance matter more than its mere presence, establishing them as key factors for robust, interpretable, and effective program synthesis.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors (1) show that transductive supervision of a certain kind of previously-proposed program synthesis system can hurt their performance and (2) propose an alternative framework that addresses this limitation while still leveraging the transductive and inductive modules of the prior work.

### Strengths
- The paper identifies precise limitations of prior work that motivate the reported investigations. In particular, the examination of the performance of transductive and inductive systems are valuable and well-designed.
- The methodology (both the experiments and the proposed framework) is well-motivated and supported with an ablation study of prior work.
- Results are communicated clearly.
- The proposed system is simultaneously conceptually simple and effective (relative to the prior work on which it is based).
- The work is clearly positioned with respect to prior work.
- The manuscript details amount of compute necessary to run the reported experiments

### Weaknesses
- The description of the proposed framework in the manuscript is hard to understand (section 4.1). I had to refer to the pseudocode in the appendix to understand it. I suggest revisiting this description and moving the pseudocode to the main text (the pseudocode is much easier to understand than both the natural language description and Figure 3).
- The proposed method and the significance of the results are limited by the substantial reliance of the synthesis algorithm on domain-specific structure (namely, the additive nature of strings and lists).
- The work is farily incremental in nature.

Minor writing issues:
- Section 2.2 should mention that the output is produced by computing prefixes (otherwise the example requires familiarity with ExeDec).

Minor typesetting weaknesses:
- Spacing between paragraphs is very small and makes the manuscript difficult to read.
- Quotation marks are used incorrectly (” is used to open quotations, instead of “)

### Questions
- What happens when the inductive model fails to solve the sub-task given by the transductive model? Line 12 of Alg 3.
- In line 349: "Tasks that TIIPS fails to solve but ExeDec succeeds on can be attributed to the last step not being guided: TIIPS may find a program that solves all training I/O pairs except the test pair, producing a false-positive program.".
	- Is this not the case for ExeDec too?
- To what extent do you think the results reported here can help the community design better synthesis systems for other domains?
- It seems the results correspond to a single "run" of each system (aggregated over a set of problems). However, there is some reliance on random numbers (e.g., ExeDec gets multiple attempts). Standard practice is to perform the experiments multiple times and aggregate to avoid spurious results and quantify variance. This would further strengthen the experiments.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper does a careful analysis of the impact of transductive guidance for program by example program synthesis, analyzing previously published system "ExeDec" which uses a model to propose subtask input and outputs directly (transduction) and a second model to generate programs that solve the subtask (induction). The authors find that using an inductive-only ablation performs almost as well as the ExeDec model. Motivated by this, they design a new approach which first tries to solve inductively, then adds increasing levels of transductive guidance as long as a satisfying program has not been found. The authors evaluate their approach on string editing and list manipulation tasks, and find improvement in the list domain, but no improvement for the string domain. The authors also do some analysis of when and how TIIPS helps.

### Strengths
- The paper is well written and presented well, and easy to understand. 
- The paper does a great job carefully analyzing the impact of transductive guidance. It is surprising and interesting that the inductive-only baseline performs just as well as the ExeDec model. 
- The proposed "ratcheting up" of transduction is well designed, and works well. It is a general approach that can be applied to any similar system, and is safe in that worst case it will match the transduction approach.
- Insights from combining induction/transduction are of interest to the program synthesis community right now, so this work is timely and significant.
- The "leave one out" evaluation approach is stronger than what ExeDec used

### Weaknesses
- The proposed approach does not improve performance on the strings domain. However, the analysis of when and why transduction fails (Figure 11) shows an understanding for why transduction should work well on the string domain, and it's good that the proposed approach still matches transductive performance on domains where transduction should perform well, so this is fine.
- The approach is only evaluated on one domain, and is mainly a follow up to one specific work, ExeDec. However, I think this is okay, because the insights from this paper (combining induction and transduction, and how the two relate to each other) can easily be applied to other domains based on what's seen in the paper.

### Questions
Q1. How does the inductive only baseline of this paper compare to the ablations used in the ExeDec paper? How do the performances compare? Were the ExeDec ablations worse, or was this issue apparent in the original ExeDec paper?
Q2. How does the computation/inference time compare between ExeDec, TIIPS, or inductive only?

Small suggestions:
S1. some double quotes incorrect direction (such as psueocode for exedec)
S2. line 240 "tough" should be "though"
S3. Figure 5: I would like a more visual indication of the sizes, instead of the overlap area being the same regardless of the percentage (something like figure 5(a) of Li et al 2024)
S4. Figure 6 is not very convincing. I would like a quantitative measure of how much "more in the top right" TIIPS is compared with ExeDec. 
S5. A formal definition of intent match and syntactical overlap might be good to have. From what I'm reading, syntactical overlap = exact same program as ground truth?
S6. Clarify whether number of steps of solution in Figure 14 is the ground truth or discovered program.

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper considers program synthesis, concretely, the programming-by-example setting. It recognizes two approaches: transductive methods and inductive methods, outlining their pros and cons. Afterwards, it discusses ExeDec - a method which combines transductive program synthesis (TPS) and inductive program synthesis (IPS) as follows: using TPS to split the target task into subtasks, and IPS to solve each subtask.

The paper then hypothesizes that ExeDec’s rigid way of combining TPS and IPS could lead to sub-optimal results, and demonstrates this empirically on string and list manipulation tasks. They show that IPS alone can solve some tasks that ExeDec cannot.

To address this, the authors introduce TIIPS - a more flexible way of combining TPS and IPS. TIIPS tries to synthesizes a program using IPS and if that fails, then TPS is used to define the first subtask. Afterwards, if IPS fails k times, TPS is used to define the first k subtasks. Eventually, if all tries fail, TPS is used to define all K subtasks, reducing to ExeDec.

It is shown that TIIPS outperforms ExeDec on list manipulation tasks where IPS has an edge, while achieving slightly lower performance than ExeDec on string manipulation tasks.

### Strengths
I found the paper to be well written and the method to be well motivated.

The paper identifies an interesting shortcoming of current hybrid approaches (that combine TPS and IPS) and verifies it empirically. This can be useful for follow-up work.

The paper makes the case for designing more flexible hybrid approaches. It can be used to justify further exploration in this direction.

### Weaknesses
Limited evaluation: The paper only compares against ExeDec, omitting other program synthesis baselines, which makes it unclear how these two methods compare against other approaches. Moreover, it is only evaluated on list and string manipulation tasks which (though I’m not sure if it’s typical) seems insufficient for characterising the strengths and weaknesses of TIIPS.

Limited novelty: TIIPS appears to be a simple modification of ExeDec, which could be fine if it lead to significant performance boost (like Hyperband), but this does not seem to be the case.

Limited discussion of the shortcomings. For instance, how much more computationally expensive is the resulting method, compared to other baselines? Where does the method break?

### Questions
What happens when TIIPS is applied to harder tasks (requiring longer programs and richer DSL), where inductive program synthesis does poorly due to the large search space?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper investigates when transductive guidance benefits inductive program synthesis in programming-by-example (PBE) tasks. The authors demonstrate that ExeDec's permanent transductive guidance can actually harm performance by overriding beneficial inductive reasoning. They propose TIIPS, which applies guidance sparsely and incrementally—starting with pure induction and adding guidance only after failures. Experiments on string and list manipulation show TIIPS matches or exceeds ExeDec's performance while producing more robust solutions.

### Strengths
- Clear Motivation: The observation that permanent guidance can harm performance is valuable and well-demonstrated
- Reproducibility: Detailed appendices with DSL specifications, algorithms, and hyperparameters enable replication.
- Limitations Section: The authors acknowledge the scope limitations of TIIPS

### Weaknesses
- Limited technical contribution: The proposed framework lacks technical novelty and is largely an iterative search with some heuristics. 
- Lack of Theoretical Insight: The paper lacks a theoretical explanation of why sparse guidance helps. 
- Lack of Generalizability: Results are specific to two DSLs with particular decomposition properties. 
- Comparison Budget:  TIIPS seems to be allowed many more synthesis attempts than ExeDec.

### Questions
- Could you please clarify the technical contribution of and any theoretical insights into the approach?
- Could you please conduct relevant ablations to demonstrate generalization and clarify details regarding the budgets used in the experiments?

### Soundness
2

### Presentation
2

### Contribution
2
