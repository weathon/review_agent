# RLAD: Training LLMs to Discover Abstractions for Solving Reasoning Problems

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
Reasoning requires going beyond pattern matching or memorization of solutions to identify and implement algorithmic procedures that can be used to deduce answers to hard problems. Doing so requires reusing primitives, intermediate results, or procedures across multiple problems. While RL post-training on long chains of thought ultimately aims to uncover this kind of algorithmic behavior, the depth-first and brute-force nature of reasoning traces learned by these models suggests that this is far from a fulfilled promise. To address more effective reasoning, we introduce reasoning abstractions: concise natural language descriptions of procedural and factual knowledge that guide the model toward learning successful reasoning. We train models to be capable of proposing several useful abstractions given a problem, followed by RL training that incentivizes building a solution while using the information provided by these abstractions. This results in a two-player RL training paradigm, abbreviated as RLAD, that jointly trains an abstraction generator and an abstraction-conditioned solution generator. This setup effectively enables structured exploration, decouples learning signals of abstraction proposal and solution generation, and improves generalization to harder problems. We also show that spending more test-time compute into generating abstractions is more beneficial for performance than generating more solutions at large inference-time budgets, illustrating the role of abstractions in guiding global exploration.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
A heuristic framework that improves benchmark performance by augmenting LLM prompts with an abstraction prompt composed by another more capable LLM.

### Strengths
Paper is very well written and clear.  The experimental results of Table 2 look promising.

### Weaknesses
By design, the solution model is weaker than the abstraction model.   
Lines 158-161: "Concretely, we utilize the Qwen3 series of models to produce solutions and a stronger reasoning model, o4-mini, to generate abstractions. " 

It is my understanding that for an apples-to-apples comparison, the solution model and the abstraction model should have the same capability. Otherwise, it is impossible to tell how much of the improvement was obtained from the abstractions rather than the  use of stronger model for the abstractions.  Perhaps this is already visible in Figure 2, leftmost panel, where using the weaker model for abstraction provides no benefit. The strength of the proposed approach depends on the results of a fair comparison.    

If my understanding is incorrect, am supportive of acceptance.  If my understanding is correct, support rejection.

### Questions
How would you factor out the performance gains from the abstraction prompts and the stronger model?   

Is there any improvement in performance when both the abstraction and solution models are the same?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes RLAD, a way to train LLMs to actually discover and use reasoning abstractions instead of just writing longer chains of thought. The idea is to have one model generate short natural language “abstractions” (like useful hints or lemmas) and another model solve the problem using them. Both are trained together with RL so the abstraction model gets rewarded when its hints help the solver. It basically teaches the model to explore breadth instead of just depth in reasoning. Across math benchmarks like AIME and DeepScaleR, it beats long-CoT RL methods by a large margin and shows that generating abstractions is a better use of compute than brute-force sampling.

### Strengths
The paper presents a novel and well motivated idea which is using reinforcement learning to discover and leverage reasoning abstractions, rather than merely extending reasoning depth which is what typical RL does. The approach is technically sound, with clear definitions, careful reward design, and strong empirical validation. The 2 player design is original and The writing is clear and well-structured, making a fairly complex setup easy to follow. Overall, it stands out as an original and solid contribution that meaningfully advances how we think about structured reasoning in large language models.

### Weaknesses
The reward design, while intuitive, could be better analyzed to show why the two-player setup works beyond simple prompt diversity effects. The paper would also benefit from more interpretability analysis of the discovered abstractions how stable they are, whether they transfer across domains, and what kind of procedural knowledge they actually encode. Finally, the training setup remains computationally expensive, which may limit broader adoption without lighter or more scalable variants.

### Questions
How sensitive is RLAD to the relative strength of the abstraction generator versus the solution generator? It seems like training stability could depend heavily on their initialization or model size. could the authors clarify if balancing their capabilities was necessary for convergence or performance?

Can the authors provide a deeper analysis of what kinds of abstractions emerge during training, for example, are they procedural (methods), factual (lemmas), or meta-level strategies and whether these abstractions transfer effectively to tasks outside math and ARC-AGI?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a two-player RL paradigm where they train a model that proposes "reasoning abstractions" based on the question and another model that actually learns to solve the problem with the proposed abstraction. They show that this leads to meaningful improvements against a baseline RL policy trained to simply solve the questions.

### Strengths
- The paper is written clearly and their contribution in terms of the proposed algorithm and its effectiveness over a the DAPO-based RL baseline is both intuitive and clear.
- Show some interesting analyses on how to allocate compute between the abstraction generator and the solver and points out that allocating more compute to the generator is often times more helpful than the solver.

### Weaknesses
- It is unclear that current models are not already good enough at generating abstractions based on their solutions. For example, [1] and [2] show that one can prompt LLMs to generate their own abstractions and create a "cheatsheet" of abstractions (whether it's cumulative or using a retriever), which also boosts their performance. So, a comparison to a baseline where one can simply prompt the model to generate its own abstraction would be compelling.
- Some details on the training are unclear: the generator and solver are different models? How many samples were used to run the warm-start of the generator? (how many abstractions per question were generated per problem for the warm start)

[1] Suzgun, Mirac, et al. "Dynamic cheatsheet: Test-time learning with adaptive memory." arXiv preprint arXiv:2504.07952 (2025).
[2] Didolkar, Aniket, et al. "Metacognitive reuse: Turning recurring LLM reasoning into concise behaviors." arXiv preprint arXiv:2509.13237 (2025).

### Questions
- How well does the abstraction generator generalize across different datasets? For example, can they be trained on AIME and then transferred to other math datasets? Or does it overfit to that distribution of math problems.
- Also, how much delta/performance gain are we getting from training both the solver and generator? If the solver does not go through the RL training and the only abstraction generator does, does the performance gain vary significantly?

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
4

### Summary
This paper introduces RLAD, an approach to RL post-training of language models. The approach trains both a reasoning abstraction generator and an abstraction-conditioned solution generator in a cooperative two-player game set-up. Experiments demonstrate the success of the approach, and analyses demonstrate the utility of reasoning abstractions.

### Strengths
1. *Originality:* Although the idea of reasoning abstractions has existed for a long time, this is likely the first work that explicitly uses RL to train the generation of reasoning abstractions as part of RL post-training.
2. *Quality:* Comprehensive experiments and analyses are used to demonstrate the utility of reasoning abstractions for solution generation and training to generate them.

### Weaknesses
### A. Evaluation

1. My main concern is with the rigor of evaluation, where there’s a reasonable doubt of unfair comparison. The model was warm-started with SFT data in the beginning, similar to DeepSeekR1, but was compared with DAPO, which doesn’t have warm-starting*.

*I’m not completely sure about this since I couldn’t find details on the DaPO baseline in the paper—also see my Question 3 below.

2. Furthermore, the SFT data for the abstraction generator was generated with models (o4-mini) significantly stronger than the models being post-trained (Qwen3-1.7B), which raises the question of how well the approach scales/how RLAD would be applied to models that are already among the strongest.

3. The ablation studies (Appendix B2) do not study how much the performance gains are from the SFT data generated with stronger models vs. other parts of the training pipeline. (Also see Question 4.)

4. The definition of iso-compute scaling curves is not well-motivated.
  - It is unclear what $k_0$ represents in the formula $m \times (k - k_0) = \mathcal C$, and why this formula is supposed to capture the notion of “iso-compute”. Also, although there is a high-level explanation about how the offset discounts samples that are too similar, subtracting an offset appears to be an arbitrary choice relative to many other possible options such as dividing by a constant.
  - Even in the formula $m \times k = \mathcal C$, this does not capture the extra compute used to generate the abstractions. 

### B. Related Work section

There is, in my opinion, an important line of past work that has been omitted in the Related Work section, namely, past work on learning and using reasoning abstractions. There’s a significant body of literature on learning and using abstractions (also often called concept learning or library learning) – see, e.g., [1], [2], [3], [4].

DreamCoder, LaSR, LINC, LEMMA,

[1] Ellis, Kevin, et al. "Dreamcoder: Bootstrapping inductive program synthesis with wake-sleep library learning." *PLDI*, 2021.

[2] Li, Zhening, et al. "LEMMA: Bootstrapping High-Level Mathematical Reasoning with Learned Symbolic Abstractions." *MATH-AI Workshop at NeurIPS’22*, 2022. 

[3] Grand, Gabriel, et al. "LILO: Learning Interpretable Libraries by Compressing and Documenting Code." *ICRL*, 2024.

[4] Grayeli, Arya, et al. "Symbolic regression with a learned concept library." *NeurIPS*, 2024.

### C. Presentation

1. The main number claimed in the introduction is “44% improvement over… DAPO on AIME 2025”, which should be clarified to mean relative (percentage) improvement instead of absolute improvement. Furthermore, 44% appears inconsistent with the reported numbers in the tables. In Table 2, the RLAD w/ abs (avg) result on AIME 2025 42.45%, and the DaPO w/o abs (avg) result is 37.92%, and 42.45/37.92 = 1.119, which is far from the claimed 1.44. (I did manage to find a ratio that’s close to 1.44, aka the ratio of RLAD w/ abs (best) 48.33% to no RL post-training w/o abs (avg) 33.75% is 1.432, but a) these are not the right numbers to take the ratio of, and b) 1.432 should round to 1.43 and not to 1.44.)

2. The way the paper is presented suggests that “reasoning abstractions” is a novel concept introduced by the paper, which is far from reality (see Weakness B on Related Work). The novel aspect of the paper is, in my opinion, not the idea of “reasoning abstractions” themselves, but the use of RL to post-train a model to generate reasoning abstractions in tandem with an abstraction-conditioned solution generator (last three lines of the Related Work section). Clarification in the paper would be appreciated.

3. High-level presentation would benefit from more explicit summaries at the beginning of each (sub)section. For example, the summary paragraph at the beginning of Section 4 would benefit from a sentence like “In this section, we describe our approach to generating reasoning abstractions via summarization of reasoning traces, and demonstrate how they improve reasoning performance by acting as hints.”

4. The “Modifying reward design” paragraph is not super intuitive and would benefit from further elaboration on how it “address[es] one of the challenges” and which challenge it addresses. Currently, the reward design appears to be mainly supported by the empirical fact that it improved performance (Appendix B2), but motivation from first principles would be nice to see.

5. Typos:
- Incorrect citations, e.g., DeepScaleR Hard (line 81-82) mistakenly cited e3.
- Line 116: “(a) … (c) …” -> “(a) … (b) …”
- Lines 260-261: “recent results RL”
- Inconsistent capitalization of DAPO as DAPO in the main text and DaPO in the appendices.

6. Appendix C is empty.

### Questions
1. For ARC, I’m curious how the results would compare with another baseline: best-of-N. It is known that in domains where external verification is available (such as ARC), best-of-N is a powerful strategy. So I’m wondering, is adding abstraction generation more cost-effective or best-of-N more cost-effective (Table 1)?

2. Lines 70-71 says “The abstraction generator is rewarded for the improvement in the
accuracy of the solution generator, stemming from conditioning on the abstractions it proposes.”
But Equation 2 is the expected reward of the solver conditioned on z, whereas from that sentence I would’ve expected it to be difference w.r.t. the reward of the solver without z. So does the actual reward subtract or not subtract the solver’s reward without conditioning?

3. Did training the DAPO baseline include SFT warm-starting similar to that used in RLAD? (Weakness A1 and Question 4 currently assume the answer is “no”, but I’m not completely sure because I didn’t find any details regarding the DAPO baseline in the paper.)

4. What is the performance of RLAD without warm-starting? Alternatively, how does RLAD compare with a version of DAPO that includes a similar warm-starting procedure? (Also see Weakness A1 regarding fair comparison.)

### Soundness
2

### Presentation
3

### Contribution
3
