# Seemingly Simple Planning Problems are Computationally Challenging: The Countdown Game

- Decision: Reject
- Scores: 10, 6, 4, 4, 8, 0

## Abstract
There is a broad consensus that the inability to form long-term plans is one of the key limitations of current foundational models and agents. However, the existing planning benchmarks remain woefully inadequate to truly measure their planning capabilities. Most existing benchmarks either focus on loosely defined tasks like travel planning or end up leveraging existing domains and problems from international planning competitions. While the former tasks are hard to formalize and verify, the latter were specifically designed to test and challenge the weaknesses of existing automated planners. To address these shortcomings, we propose a procedure for creating a planning benchmark centered around the game called Countdown, where a player is expected to form a target number from a list of input numbers through arithmetic operations. We discuss how this problem meets many of the desiderata associated with an ideal benchmark for planning capabilities evaluation. Specifically, the domain allows for an intuitive, natural language description for each problem instance, it is computationally challenging (NP-complete), and the instance space is rich enough that we do not have to worry about memorization. We
perform an extensive theoretical analysis, establishing the computational complexity result and demonstrate the advantage of our instance generation procedure over public benchmarks. We evaluate a variety of existing LLM-assisted planning methods on instances generated using our procedure. Our results show that, unlike other domains like 24 Game (a special case of Countdown), our proposed dynamic benchmark remains extremely challenging for existing LLM-based approaches.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
This is a strong, interesting paper that investigates the planning capabilities of today's LLM. It proposes a simple, easy-to-use benchmark for assessing LLM planning, and it discusses the empirical results.

### Strengths
The work appears to be original. The paper is of high quality and clarity, and the empirical results show that the proposed benchmark allows us to better understand the limitations of today's LLM-based planning.

### Weaknesses
Even though the proposed benchmark is clearly valuable, the paper would greatly benefit for an extensive discussion of its current limitations and possible improvements. Such a discussion would be an excellent roadmap for future work on this area.

### Questions
N/A

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes Countdown as a planning benchmark with an NP-completeness proof, an instance generator, and a Planning Domain Definition Language (PDDL) model for planner baselines. The authors evaluate multiple LLM methods including Input-Output,  Chain-of-Thought, Tree-of-Thoughts,, and AutoToS. They find an easy-hard-easy difficulty pattern across instance sizes. Dynamic instances reduce ToT performance relative to the static 24 Game dataset, suggesting memorization rather than generalization.

### Strengths
Shows that prompting methods may lack generalization on unseen instances. Domain formulation enables reproducible evaluation. Strong baselines including symbolic planners.

### Weaknesses
Unlcear how presentative the Countdown problem actually is.

### Questions
Could you provide diagnostic data like branching factors, node expansions, or solution multiplicity distributions across sizes?
The NP-completeness proof needs clarification on numeric encoding.

### Soundness
3

### Presentation
4

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
The paper proposes a procedure for creating a planning benchmark based on the Countdown game. In this game, a player is expected to form a target number from a list of input numbers through arithmetic operations. It is shown that the problem is NP-complete, thus it is computationally challenging. The paper argues that the domain meets many desiderata associated with an ideal benchmark for planning capabilities evaluation. It includes an extensive evaluation of its instance generation procedure over public benchmarks. The conclusions drawn from the experiments are interesting. 

The paper lists several desired properties for a benchmark of planning abilities. It would have been helpful if the paper elaborates on these criteria (e.g., why is a criterium desirable?). Furthermore, there is no discussion that shows that the generated benchmark satisfies these desired properties. In my opinion, the fact that the instances are planning problem instances does not provide an obvious answer, for otherwise, any procedure that generates planning problem instances could be used.

The presentation is easy to follow. However, some definitions and proofs need to be clarified:

- Definition 3: do all numbers in X need to be used to construct $\omega$? For example, does the SAP problem with X = {4, 5} and $\omega=4$ have a solution? 
-  I am not sure if Lemma 3 is sufficient to conclude that the CDP constructed in Theorem 1 cannot contain + and -. What if there are a, b, c and x, y, z so that $10^{a \pm b \pm c} = 10^x \pm  10^y \pm  10^z$? (e.g., if a = x, b = c, y=z  then $10^{a +b - c} = 10^x +  10^y - 10^z$). So, it appears that Lemma needs to be proved for two arbitrary sets $\{x_1, \ldots, x_k\}$ and $\{y_1, \ldots, y_k\}$. This is, however, not correct for k = 3.  

The description of the generation of the dataset is really good.

The experiment section is extensive. It demonstrates that the generated instances are difficult for LLM-based planning.  

Overall, the paper proposes a new benchmark for planning. There is some disconnection between parts of the papers and a proof might need to be revised.

### Strengths
+ The paper is easy to follow. 

+ When comparing the performance of LLM planning methods on the 24Game instances and the CD dataset generated by the proposed method, all LLM planning methods show significantly worse performance on the CD dataset. The finding indicates the usefulness of the proposed data generator.
 
+ The manuscript clearly explains the setting and the metric. Aggregating 100 instances per instance size sees a solid design.
 
+ Three representative open language models are used. They are a good representation of LLMs.

### Weaknesses
- Please check my comments on the proof of Theorem 1. 

- The discussion on desired properties of a planning benchmark should be discussed and connected with other parts of the paper. 

- The data generation does not report the time used to generate CD, RG, and SoS. It will be good to include the running time.
 
- To show the superiority of the CD dataset, generated by the proposed method, it will be better to run the LLM planning methods on the CD datasets and other generated datasets, RG, and SoS. However, the paper only compares the performance of the LLM planning methods on the CD and 24Game datasets.

### Questions
- Please check my comments on the desired properties and the proof of Theorem 1. They need to be addressed. 

Other questions:
- The measure accuracy @5 chooses per task the maximal accuracy over the 5 trials. Why do not report the average accuracy over the 5 trials?
 
- It will be useful to investigate further the fundamental reason in the dip of Figures 2 &3.

### Soundness
2

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
The paper focus on the drawback of existing planning benchmarks, such as poorly defined tasks or benchmark memorization.
 To address this, the authors propose using the countdown game as a superior benchmark.
 An experimental evaluation showing that while traditional symbolic planners and an LLM-based solver-generator can succeed, popular LLM planning methods fail dramatically on these new instances, even those of small size. This suggests their high performance on existing benchmarks may be due to data contamination rather than genuine planning ability.

### Strengths
1. The paper provides a formal proof that the countdown problem is NP-complete, which gives a strong theoretical foundation for its use as a hard benchmark, moving beyond qualitative assessments.
2. The authors' proposed a automatic data generation method, which favors targets with fewer solution paths, directly tackles the problem of benchmark memorization.

### Weaknesses
1.  The paper fails to explain why there is an easy to hard to easy situation in Figure 3&4, undermining the complexity of such tasks.
2. The error analysis categorizes failure modes but doesn't provide a deep qualitative analysis of why these errors occur. It's unclear if LLMs fail at basic arithmetic, logical step-tracking, or strategic search.
3. The dataset is constructed in an over-simplified manner. Although it does evaluate the core planning ability, its over-simplification undermines its application range.

### Questions
1. Given that direct planning methods fail, while the successful AutoToS method that use the LLM to generate symbolic code for a traditional search algorithm, does this suggest that the most viable role for LLMs in complex, verifiable planning is not as direct planners but rather as problem-compilers that translate natural language descriptions into a formal, symbolic representation that a dedicated solver can then execute?
2. How can the benchmark be used to improve the performance of the base LLM?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Countdown Game as a new benchmark for evaluating long-horizon planning in LLMs. Unlike prior loosely defined or classical planning tasks, it formalizes the arithmetic-based game as a planning problem and proves it NP-complete via reduction from Partition. The authors develop a diverse instance generator, a PDDL formulation, and evaluate both classical planners and LLM-based methods. Results show Countdown is significantly harder than simpler tasks like the 24-Game, exposing key failure modes in current LLM planners. The work provides a rigorous, reproducible benchmark and insights into the limitations of LLM reasoning.

### Strengths
- The paper provides a solid theoretical foundation by formally analyzing the problem’s complexity.
- The authors design and validate a method to generate challenging and diverse puzzle instances, addressing potential issues of triviality or memorization. Instead of relying on fixed or scraped puzzles which could appear in training data, they propose a dynamic instance generator.
- The paper evaluates a broad range of planning approaches under consistent conditions, yielding credible and insightful results. The authors test multiple paradigms: a classical planner, a cutting-edge neuro-symbolic method, and popular prompt-based LLM strategies, making the evaluation well-rounded.
- The paper surfaces important observations that advance understanding of LLM planning. Two discoveries stand out: (a) a phase transition phenomenon in puzzle difficulty, and (b) evidence of overfitting on static benchmarks. The paper also delves into why the LLM planners fail, which is very useful for future improvements.

### Weaknesses
- The paper reports a surprising “hard-to-easy” phase transition in problem difficulty without providing an explanation, leaving a gap in understanding. While the discovery of two phase transitions is intriguing, the authors explicitly note that they cannot offer an explanation for why larger Countdown instances become easier again.
- The paper primarily compares LLM-based planners against one classical planner and each other, but other potential baseline methods are not explored. For instance, Countdown puzzles could be approached with a brute-force or heuristic search specialized for arithmetic outside of PDDL, or even formulated as a Constraint Satisfaction/Optimization problem (e.g., an ILP or backtracking solver). Including a straightforward depth-first search or an optimized backtracking solver for small instances would show how far pure symbolic search can go on these problems.

### Questions
- What might be the underlying cause of the second “hard-to-easy” phase transition around 20 input numbers, and can the authors provide any analysis or hypotheses here?
- Have the authors considered evaluating alternative solving approaches, such as a bespoke backtracking or integer programming solver for the Countdown puzzle, or integrating a planning heuristic beyond ENHSP’s capabilities?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 6

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper focus on the problem of countdown game, provided a theoretical analysis of countdown game to show that it is NP-hard. The problem is evaluated with some LLMs with existing LLM-based planning approach like CoT, ToT, and AutoTos to show that the proposed problem is still a challenging problem.

### Strengths
- The paper provides a theoretical guarantee of the proposed problem.
- The paper is overall well-written and easy to follow.

### Weaknesses
- The paper does not follow the paper format. 
- As the author acknowledged, the same problem has been previously proposed in Reasoning Gym [1].
- The evaluation is based on old models that are even reasoning models. As a dataset paper, this is highly insufficient. To the current stage of LLM research, it is no surprise that LLM will fail in some NP-hard problem, but it will be much more interesting if one can provide more insights into what scale wat ill the problem becomes unsolvable to the latest models, and how potentially it can be improved. The current contribution is far below what is expected from an ICLR paper. The current paper is overall much less informative than its prior work [1].

[1]. Stojanovski Z, Stanley O, Sharratt J, et al. REASONING GYM: Reasoning Environments for Reinforcement Learning with Verifiable Rewards[J]. arXiv preprint arXiv:2505.24760, 2025.

### Questions
No specific questions. While this paper is probably a technically correct paper, I believe the paper still needs substantial improvement to be accepted to a top-tier ML conference.

### Soundness
1

### Presentation
2

### Contribution
1
