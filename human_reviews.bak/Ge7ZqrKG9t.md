# Modeling Complex Mathematical Reasoning via Large Language Model based MathAgent

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 5, 5, 3

## Abstract
Large language models (LLMs) face challenges in solving complex mathematical problems that require comprehensive capacities to parse the statements, associate domain knowledge, perform compound logical reasoning, and integrate the intermediate rationales. Tackling all these problems once could be arduous for LLMs, thus leading to confusion in generation. In this work, we explore the potential of enhancing LLMs with agents by meticulous decomposition and modeling of mathematical reasoning process. Specifically, we propose a formal description of the mathematical solving and extend LLMs with an agent-based zero-shot framework named \textbf{P}lanner-\textbf{R}easoner-\textbf{E}xecutor-\textbf{R}eflector (PRER). We further provide and implement two MathAgents that define the logical forms and inherent relations via a pool of actions in different grains and orientations: MathAgent-M adapts its actions to LLMs, while MathAgent-H aligns with humankind. Experiments on miniF2F and MATH have demonstrated the effectiveness of PRER and proposed MathAgents, achieving an increase of 12.3% (53.9%$\rightarrow$66.2%) on MiniF2F, 9.2% (49.8%$\rightarrow$59.0%) on MATH, and 13.2% (23.2%$\rightarrow$35.4%) for level-5 problems of MATH against GPT-4. Further analytical results provide more insightful perspectives on exploiting the behaviors of LLMs as agents.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper focuses on modeling the mathematical reasoning process within LLMs. The authors propose a novel framework named Planner-Reasoner-Executor-Reflector (PRER) and implement two MathAgents within this framework, i.e., MathAgent-M and MathAgent-H, to tackle complicated mathematical problems. The experimental results verify that the two agents significantly improve the reasoning accuracy.

### Strengths
1. The proposed PRER is a general framework whose idea of decomposing mathematical reasoning process is rational. Besides, the architecture of the paper is clear to read.
2. This framework can be implemented with different LLMs and in different grains. The motivation of describing LLMs’ and human-like behaviors is reasonable, and the corresponding technique makes sense.
3. The accuracy improvements on datasets MiniF2F and MATH are significant.

### Weaknesses
1. It’s necessary to give an \emph{overall} introduction of its idea. What I mean is not the description of the workflow of Planner-Reasoner-Executor-Reflector framework. What I am concerned about is the reason of formalizing the actions as “infer”、“calculate” and so on (Details could be referred to question 1 below). Besides, this paper presents several modules realized by prompting LLM, without a clear introduction to the internal logic and reasons for model design. 
2. Some details lack enough descriptions or explanations, bringing difficulty in reproducing the proposed framework. Details could be referred to questions 2-4 below.
Please see the detailed questions below, which should be answered and addressed.

### Questions
1. How do the authors define the actions of different modules? For example, why does the “Mathematical” class in MathAgent-H only contain “associate” and “construct”? What is the behind idea of designing them? Are there other actions that should be considered? 
2. According to the paper, the actions of MathAgent-M is a subset of MathAgent-H's. Therefore, what is the necessity of proposing MathAgent-M independently from the perspective of technique? Besides, as described in Table 1, the "Infer" action in MathAgent-M has different meaning with the "infer" in MathAgent-H. The authors state that the action in MathAgent-H is more aligned with human actions. However, the description of "infer" in MathAgent-M, i.e., "Infer new rationales using deduction methods" can also be viewed as an action in human cognition.
3. What is the meaning of m^1_n, m^2_n in Eqs.(2),(3) in section 2.2. They lack descriptions or explanations.
4. In Eq.(3), why is t_n (i.e., topology of the inference) an output, not an input, and even if t_n is obtained, what is it useful for subsequent reasoning? Because t_n does not appear in Eq.(4) or other equations.
5. According to Figure 3, “infer” and “calculate” occupy the most important part for MathAgent-M and MathAgent-H, respectively. It's a little weird for me due to the following reasons. First, as the authors have stated, MathAgent-H is more aligned with human actions. However, the statistics reveal that the human-like actions (e.g., "induce", "rethink") take up a very small proportion in MathAgent-H's reasoning process, contradictory with the motivation of designing MathAgent-H. Second, why "infer" takes up such a small proportion in MathAgent-H? Intuitively, since the testset are the same, "infer" should also be the prominent action in MathAgent-H, since it's more relevant to mathematical reasoning. On the contrary, "calculate" in MathAgent-H is indeed a computation action, which intuitively should not have such a high frequency.
6. There exist some other works that also decompose the mathematical reasoning into several steps (e.g., Tree of Thought [1]) and adopt a generate-then-verify paradigm (e.g., [2,3]). The authors need to give more illustrations of how this work is distinctive, explain the differences with other similar works, and incorporate them in experiments.
[1] Yao S, Yu D, Zhao J, et al. Tree of thoughts: Deliberate problem solving with large language models[J]. arXiv preprint arXiv:2305.10601, 2023.
[2] Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318, 2023.
[3] Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning, pages 19274–19286. PMLR, 2023.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper tackles improving the mathematical reasoning capability of LLMs by proposing a set of modular decomposition actions. These actions range from infer, associate, observe, disprove, etc. All the actions are simulated by LLMs with few shot prompts. With the help of these actions, the paper shows a strong performance improvement on MATH and MiniF2F datasets.

### Strengths
The paper attempts to systematically break down various useful actions in mathematical reasoning. The design is interesting.

The performance gain from prompting the LLM with proposed actions is quite significant especially on MiniF2F datasets, where it solves 20% IMO problems that are not solved before.

### Weaknesses
The paper is not very well-written with many of the equations not explained clearly. The authors should provide more clarifications on these.

The design of various actions seem heavily engineered and the overall algorithm quite complicated. (See Algorithm 2) I wonder if the authors could break down the effect of various actions and only identify a few that contribute to the performance improvement the most. This is especially important since according to Figure 3, majority of the actions are "calculate".

### Questions
1. What does this sentence mean: "whereas the MATH dataset does not offer final answers for reasoning"? I am very certain MATH datasets have ground truth reasoning steps and final answers.

2. What actions are truly necessary in improving the reasoning performance of LLM? Can the authors perform more thorough ablation?

3. Given the strong MiniF2F performance with MathAgent-H on IMO problems, can the authors provide a few generated proofs for those problems? Also, I was not able to find the prompts associated with MiniF2F.

4. Figure A1: where is (c)?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper delves into the challenges LLMs face when solving intricate mathematical problems. To address these challenges, the authors introduce an agent-based zero-shot framework named Planner-Reasoner-Executor-Reflector (PRER) and two MathAgents, MathAgent-M and MathAgent-H. Experiments on miniF2F and MATH datasets show that the proposed approach significantly outperforms GPT-4, especially for level-5 problems of the MATH dataset.

### Strengths
1. The paper is well written and easy to follow.
2. The work pushed the state-of-the-art results on two datasets including MATH which seems to be a challenging dataset even for larger language models.
3. The implementation two MathAgents is innovative and shows promise in addressing the challenges LLMs face in mathematical reasoning.

### Weaknesses
1. Although the paper demonstrates advancements over GPT-4, it fails to specify the average model calls needed for each question within the PRER framework. This omission raises concerns about potential high costs. If the cost is, hypothetically, k times, would it still surpass k majority-voting? It would be advantageous to incorporate such an experiment. With succinct prompts, GPT-4 can outperform MathAgent-M on the MATH dataset. For instance, PHP achieves a score of 53.9.
2. The experiments are centered on particular datasets (miniF2F and MATH), both of which solely encompass abstract mathematical language. The efficacy of the proposed technique on other mathematical problem-solving datasets remains uncertain, especially concerning word problems akin to those in GSM8K. Such word problems can also be complex and may require more domain knowledge.
3. The paper's proposition of an approach that can “systematically decompose and model the solving process of complex mathematical reasoning” seems unsubstantiated with neither theoretical nor empirical backing. While using prompts to tailor LLM into a specialized expert is a prevalent strategy, the model doesn't acquire any fresh insights. Furthermore, there's an absence of empirical evidence emphasizing the significance or need for Executors.

### Questions
The paper introduces a technique to augment LLMs' aptitude in mathematical reasoning through an agent-based framework. Although the findings are encouraging, questions remain about the method's adaptability and the absence of exhaustive comparisons with alternative techniques.

**Correctness:** 3: Some of the paper’s claims have minor issues. A few statements are not well-supported, or require small changes to be made correct.

**Technical Novelty And Significance:** 3: The contributions are significant and novel, but there are areas that could be further explored or clarified.

**Empirical Novelty And Significance:** 3: The empirical contributions are significant, but the paper could benefit from a broader range of experiments and comparisons.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this work the authors develop a general agent-based framework, called Planner-Reasoner-Executor-Reflector (PRER), to model the problem solving process in mathematical reasoning (MR).
A feature of the proposed framework is that it only relies on LLMs, with no calls to external theorem provers.
The proposed approach is evaluated experimentally.

### Strengths
1) The experimental evaluation is rather thorough, the proposed approach is compared with several different frameworks.

2) The related literature is discussed in some detail, but it only mentions briefly related approaches that leverage on theorem-provers. Also, the paper does not discuss why avoiding the use of theorem-provers entirely.

### Weaknesses
1) The authors say that "to the best of our knowledge, systematical decomposition and meticulous
modeling of complex mathematical solving process have not been explored." However, there are a few pages on decomposition for mathematical reasoning, also cited by the authors themselves. Consider, for instance,

- Xueliang Zhao, Wenda Li, and Lingpeng Kong. Decomposing the enigma: Subgoal-based demonstration learning for formal theorem proving. arXiv preprint arXiv:2305.16366, 2023.

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837, 2022.

- Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. arXiv preprint arXiv:2305.10601, 2023.

and related:

- Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, and Ashish Sabharwal. Decomposed prompting: A modular approach for solving complex tasks, 2023.

2) Equation 1 is not entirely clear. It seems akin to the notion of deduction in logical systems, but its meaning is not formally specified. E.g., what is the meaning of symbol "|-"?

3) The different components of the proposed framework (planner, reasoner, executor, reflector) are presented rather in a hurry, in less than one page, by means of Equations (1) to (5), which are not explained in much detail either, especially as for the role of the different logical functions. Consider: "Planner
includes an addition function, preprocessing, to decompose the original problem into the form of (X, y)."
We don't get any more information about preprocessing in the paper.

4) As the authors themselves discuss limitations in the conclusions, "the current prompts are manually
crafted, heavily reliant on experts."
This might not be the most promising way forward.

### Questions
It is not entirely clear to me why two different agents, MathAgent-M and MathAgent-H, are required. What is the rationale for this choice?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor
