# DecAEvolve: Decompose, Adapt, and Evolve, or Three Pillars of Effective LLM-based Scientific Equation Discovery

- Avg Score: 4.80
- Decision: Reject
- Scores: 2, 4, 8, 6, 4

## Abstract
Finding mathematical relations underlying natural phenomena and scientific systems has been one of the fundamental tasks in the history of scientific discovery. Recent advancements in evolutionary search with Large Language Models (LLMs), with their embedded scientific knowledge, have shown great promise for this task. However, discovering such mathematical models governing scientific observations still remains significantly challenging, as it requires navigating vast combinatorial hypothesis spaces with an explosion of possible relations. Existing LLM-based approaches overlook the impact of data on the structure of mathematical relations, and treat LLMs as a static hypothesis generator unaware of the observed scientific system. This leads to suboptimal and inefficient exploration of the hypothesis space with over-reliance on LLMs' internal priors. To bridge this gap, we introduce Decompose, Adapt, and Evolve (DecAEvolve), a framework that leverages granular feedback from symbolic term decomposition and LLM refinement through reinforcement learning (RL) fine-tuning to enhance both robustness and efficiency of evolutionary discovery frameworks.  Our experiments across diverse datasets demonstrate that DecAEvolve significantly improves the accuracy of discovered equations and the efficiency of the discovery process compared to the state-of-the-art baselines.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes DecAEvolve, a framework that integrates term-level decomposition, test-time adaptation via reinforcement learning, and evolutionary search, enabling large language models to discover scientific equations more efficiently and interpretably from data.

### Strengths
Proposes a unified framework (DecAEvolve) that combines decomposition, test-time adaptation, and evolutionary search in a coherent pipeline.

### Weaknesses
My concerns are as follows:
1. The evaluation of each decomposed term is conceptually problematic
1.1 Because the process still relies on refitting parameters using BFGS, the results are prone to overfitting and instability. Each refitting step changes the weights of the remaining terms and may converge to a local optimum, thereby failing to reflect the true importance of the removed term.
1.2 Moreover, the symbolic terms in an equation are often highly coupled and possess domain-specific semantic interdependencies. Removing a single term in isolation can damage interpretability and cannot faithfully represent the true contribution of that component.
1.3 In addition, the computational cost is extremely high. If BFGS optimization must be executed after every single term removal—and given that BFGS itself is relatively slow—this design imposes a substantial computational burden.
In summary, while the method aims to provide granular feedback, I believe such feedback may become distorted due to numerical instability and structural coupling. Furthermore, I did not find clear ablation or reverse-validation experiments demonstrating the actual utility of this design (please correct me if I missed such evidence).

2. As shown in Fig. 1, the proposed framework remains highly similar to the LLM-SR architecture (including the island-based evolutionary design and overall search pipeline). The main difference lies in the addition of a modified GRPO fine-tuning step. While this is not to say that LLM-SR is inadequate, the paper does not introduce a genuinely novel perspective on how LLMs can be applied to equation discovery. The LLM still functions primarily as a solver, and this role does not represent a breakthrough compared with traditional RL or GP approaches.

3. Experimental setup concerns
The experiments are conducted on the LLM-SR proposed dataset, yet the Qwen2.5 models were released after LLM-SR, raising a potential data leakage issue. To ensure fair comparison, the authors should either use the newer LLM-SRBench benchmark or restrict evaluation to models that precede or coincide with the LLM-SR release.
Overall, the LLM-SR dataset is not a mature benchmark—its authors quickly released the improved LLM-SRBench afterward. Therefore, I strongly suggest conducting evaluations on LLM-SRBench instead of the outdated LLM-SR.

### Questions
Please refer to the weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
# Review for DecAEvolve

## Summary
- The paper proposes a new LLM-based framework for scientific discovery called DecAEvolve. They explain the relevant context and background of other methods and show their proposed framework outperforms others. Their innovation is using test-time RL using GRPO and term decomposition to improve the generated results while iterating on scientific discovery.

## Recommendation
- Weak reject. I actually like the core idea here but there are lots of issues in the paper and experimentation that need to be ironed out. If these issues get addressed then I would be happy to recommend this paper for acceptance. I outline these issues in Questions and Feedback.

### Strengths
## Strengths
- The model architecture seems to be novel and results substantiate the design decisions.
- Lots of experiments are performed, including ablation studies, comparison with a similar LLM architecture, and non-LLM models.
- Figures and tables are legible and provide useful information.

### Weaknesses
## Weaknesses
- Not enough explanation/details for the proposed method, even in the appendix.
- Key experimentation details are missing/ not substantiated.
- No code is provided to verify the implementation and method is doing what is claimed.

### Questions
## Questions
- Where is the code? The method is not proprietary and to properly confirm the results provided here are not fabricated it is best practice to provide *reproducable* code. I should be able to look at and run everything that is claimed in the paper.
- "internal prior" and "scientific prior" are used a lot in the paper, but is never explained. Exactly what is this? Please include this description in the background section.
- For test-time GRPO, how is Score_T computed? The appendix B reference doesn't clarify this at all. It's mentioned there is a held-out test suite, but this test suite is not defined or made clear.
- What is the difference in computation and runtime for your model compared to others? I would imagine it runs slower due to directional feedback (section 3.2) as well as test-time GRPO.
- Why was SINDy not used as a non-LLM baseline? Why was it not mentioned in the paper at all?
- The experiment comparing with LLM-SR is not clearly stated. On one hand, it's mentioned in the main text (line 255) that 3,000 LLM calls were made. On the other hand, Figure 2 shows that there were 3,000 search candidates, implying 3,000 full loops of the method. My questions are:
   - If you used 3,000 loops per problem, what is the runtime comparison between LLM-SR and DecAEvolve per problem? I would expect DecAEvolve to take longer to run due to test-time model tuning using LoRA.
   - If you used 3,000 LLM calls per problem, did you count the LLM calls in GRPO (N times G=64 completions) for DecAEvolve? If so, then DecAEvolve should've had less total loops than LLM-SR due to DecAEvolve's test-time GRPO.

## Feedback
- Please add a citation to and briefly mention how SINDy fits in to the previous work section for neural-guided deep learning methods for discovering symbolic regression:
```
@article{brunton2016discovering,
  title={Discovering governing equations from data by sparse identification of nonlinear dynamical systems},
  author={Brunton, Steven L and Proctor, Joshua L and Kutz, J Nathan},
  journal={Proceedings of the national academy of sciences},
  volume={113},
  number={15},
  pages={3932--3937},
  year={2016},
  publisher={National Academy of Sciences}
}
```
- "Line 92, "and SGA": make this the start of a new sentence, otherwise you have one sentence spanning 6 lines.
- The first sentence of your key insight should be modified to "Our key insight is that equation discovery benefits from both adaptation (aligning the model with data distributions) and decomposition (understanding which symbolic components matter), neither of which prior LLM frameworks integrate. "
  - Mention that LLM frameworks don't do this explicitly, the above citation for SINDy does this exactly just without an LLM.
- For "Problem Formulation" please change $x_i$ to either have a bold $\mathbf{x}_i$ or a vector on top of the $\vec{x}_i$ to denote that it is in $\mathbb{R}^d$ and not just $\mathbb{R}$, since it currently looks identical to $y_i$.
  - The LLM-SR paper chose a bold $\mathbf{x}_i$.
- Algorithm 1 should have another loop in it, since you do Stage 1 and Stage 2 multiple times to get the final solution.
- Algorithm 1 is not clearly explained. What is $E$ and $e_j$? What is $SampleExp$? What about $MakeFewShotPrompt$? All of this needs to be rewritten or explained better. The point of the algorithm is to make it clear what your model is doing.
- Table 1, fix lowest score for column 1 to be LLM-SR (Mixtral) and second best score to be LLM-SR (GPT-3.5).
- It's weird to have a related work section that is very similar to what you have in your introduction. Consider consolidating the sections together and moving the related work section up to the introduction.
- Change line 439/440 "Zue et al." to an in-text citation `\citet{...}` and remove the end-of sentence citation on line 441/442.
  - Same problem end of line 447.
  - Same problem in line 449.
  - Same problem in line 451.
  - Same problem in line 453.
- Line 441, there is a run-on slightly incoherent sentence:
   - Change
     - "Despite these successes and potential benefits of test-time training in better adapting to novelty, test-time adaptation remains largely unexplored in evolutionary scientific discovery frameworks, leaving open how inference-time learning can directly align priors of pretrained model with the dynamics of specific scientific system during the evolutionary process of search and discovery."
   - to
      - "Despite the successes and potential benefits of test-time training in better adapting to novelty, test-time adaptation remains largely unexplored in evolutionary scientific discovery frameworks. It is still unclear exactly how inference-time learning can directly align the priors of a pretrained model by leveraging the dynamics of specific scientific system during the evolutionary process of search and discovery."
   - or something similar.
- Line 465: change "GRPO and, evolutionary" to "GRPO and evolutionary

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors present DecAEvolve, a framework that improves current uses of LLMs for scientific equation discovery. This new framework takes advantage of given data in order to fine-tune the LLM based on data-driven rewards. This allows the LLM to reason about the contribution of each symbol, leading to an interpretable guidance towards the correct expression.

### Strengths
This paper is an improvement on LLM-SR, integrating reasoning about individual symbols and an evolutionary algorithm in addition to the LLM’s prior knowledge.  The paper is clearly written, with figures describing the flow of information through the evolve, adapt, and decompose sections of the framework. The evaluation is substantive, with the ablation studies showing the contribution of both the adaptation and decomposition modules.

### Weaknesses
I would be curious to see more about how noise in the data would affect the system’s effectiveness. Scientific data naturally will have noise due to how it’s sampled, and robustness to noise would allow this system to be more effective.

### Questions
1.	How does this system deal with highly correlated variables? In some problems with a high number of features, correlation between those features could cause multiple valid equations or confusion in your system in understanding the underlying symbolic structure.

2.	Can you show the performance of your model in noisy settings compared to the other baseline models?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper builds off previous work in LLM-driven symbolic regression (LLM-SR) to address the problem of scientific equation discovery from data. It makes two modification to the LLM-SR framework:
A test-time-training component using GRPO to updated the base model weights prior to LLM evolutionary search.
2. Term-level feedback about candidate expressions during evolutionary search.
Evaluating on the same 4 synthetic examples as LLM-SR did, it shows that these changes (called DecAEvolve), outperform LLM-SR and other symbolic regression methods nearly universally across LLMs and tasks. Ablation on the 2 modifications reveal that both are beneficial, however the GRPO-based test-time-training provides most of the lift. An investigation of reward improvement with more GRPO iterations suggests that smaller, cheaper models can “catch-up” to larger models in performance given enough GRPO training.

### Strengths
The paper makes both a systems and technical contribution (adding GRPO-based test-time-training, and the term decomposition based feedback to evolutionary search), and ablates on both to show where the improvement comes from (primarily GRPO).
The quantitative results appear to be a significant improvement over LLM-SR (while questions of comparable computational resources remain, see below, the curves shown in Figure 2 for LLM-SR do not appear like they would ever overtake DecAEvolve.

### Weaknesses
This work adds two methods on top of LLM-SR; GRPO-based test-time-training (adaptation), and equation-decomposition feedback. Of these two, the decomposition is the real novel contribution, but looking at Figure 2 we can see that in most cases, the majority of the improvement over LLM-SR comes from the addition of GRPO, not the more novel contribution. Also, as near as I can tell, these charts are not taking into account the extra program samples involved in the GRPO-training, so these may not be fairly judging compute vs performance.
LLM-SR is only given one short paragraph of explanation, despite being largely reproduced as part of this method. While the mechanisms of adaptation and decomposition are (mostly) well described between the primary text and the appendix, *how* these fit into the LLM-SR framework are not. The relationship between Algorithm 1 and Figure 1 (and order of feedback through Figure 1) needs to be explicitly shown, and there are no details about the multi-island evolutionary search that is shown in Figure 1 and mentioned in the introduction.
The only results shown are error-based scores, which give no context for how large or small the effects are on the prediction results. The LLM-SR paper includes results that show the predictions of the equation discovery relative to ground truth – including such figures would go a long way to demonstrating the impact of this work.

### Questions
During the decomposition optimization for each term / pairwise interaction experiment, are the original parameters included in the optimization, or just the term weights?
In Fig. 2, efficiency is measured based on the # of search candidates, but on line 255 the computation budget is specified in terms of the # of LLM calls. Do the LLM calls in the budget take into account calls needed for GRPO sampling, and if not, how much more compute is needed with the GRPO-based adaptation

This work builds directly on the algorithm of LLM-SR, but does not explain that method in enough detail to be reproduced. I would suggest that the space used for the detailed explanation of GRPO in the preliminaries would be better used for a more full explanation of LLM-SR and how the modifications of DecAEvolve fit within that framework.

How important (qualitatively) are the error improvements overn LLM-SR? It would be great to include some prediction figures comparing the output of the best performing functions discovered by each framework to the ground-truth.
On line 115 it is claimed that Decomposition “enables LLMs to understand not just which hypotheses succeed, but *why* specific mathematical building blocks are effective.” Please explain this claim. It seems to me that decomposition is showing *which* mathematical building blocks are effective, but not why they are (e.g. “why” a sin block in an oscillator’s forcing function is effective would be “because the forcing function is periodic”, not that “removing the sin block increases the error by XXX”).

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on automated discovery of symbolic equation with observed data of that equation. It points out two weaknesses of previous methods, adaptation and decomposition. Here adapation is about LLMs do not update their parameters for task-specific update even if after long time doing inference on a task. Decomposition is about to identify the valuable part of the equation so to more efficiently discover the full picture of the equation consistenting of several valuable parts.

### Strengths
1. This paper proposes to use GRPO to update the parameters of an LLM working on that research direction of scientific discovery. It has not been done in the field of automated scientific discovery before as far as I know. It would be beneficial to the community on sharing the insights of how to adapt GRPO on automated discovery of symbolic equation task, and the difference of using GRPO in this task compared with using it on other more common language tasks.

2. The idea of using "decomposition" is reasonable. 

3. The experiments are sufficient.

### Weaknesses
1. One of the main idea, the "decomposition" has been proposed by [1] on automated discovery of chemistry scientific hypothesis with experimental feedback. The fundamental ideas are very similar, although the two papers work on scientific discovery of different tasks (math equation and chemistry). I think this submission should discuss [1] and clarify how their use of decomposition differs or extends prior formulations in this new setting.

2. It is a bit unclear on how the "adapatation" (GRPO) and "decomposition" parts "collaborate" and have the synergy effect. Currently from Algorithm 1, my understanding is that they are basically on two stages, and don't have interactions, is this understanding correct?

3. Algorithm 1 mentions many symbols such as P and E, which seems are neighther discussed in Algorithm 1 and the Method section.

4. More insights on using GRPO on the symbolic equation discovery task can be beneficial (e.g., difference with using it on other tasks) 


[1] MOOSE-Chem3: Toward Experiment-Guided Hypothesis Ranking via Simulated Experimental Feedback

### Questions
1. Do "adapatation" (GRPO) and "decomposition" have any "interaction" (synergy) with each other?

### Soundness
3

### Presentation
3

### Contribution
3
