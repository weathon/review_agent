# Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning

- Decision: Accept (Oral)
- Scores: 8, 8, 6, 8

## Abstract
Enabling LLMs to improve their outputs by using more test-time compute is a critical step towards building self-improving agents that can operate on open-ended natural language. In this paper, we scale up inference-time computation in LLMs, with a focus on answering: if an LLM is allowed to use a fixed but non-trivial amount of inference-time compute, how much can it improve its performance on a challenging prompt? Answering this question has implications not only on performance, but also on the future of LLM pretraining and how to tradeoff inference-time and pre-training compute. Little research has attempted to understand the scaling behaviors of test-time inference methods, with current work largely providing negative results for a number of these strategies. In this work, we analyze two primary mechanisms to scale test-time computation: (1) searching against dense, process-based verifier reward models (PRMs); and (2) updating the model's distribution over a response adaptively, given the prompt at test time. We find that in both cases, the effectiveness of different approaches to scaling test-time compute critically varies depending on the difficulty of the prompt. This observation motivates applying a "compute-optimal" scaling strategy, which acts to, as effectively as possible, allocate test-time compute per prompt in an adaptive manner. Using this compute-optimal strategy, we can improve the efficiency of test-time compute scaling for math reasoning problems by more than 4x compared to a best-of-N baseline. Additionally, in a FLOPs-matched evaluation, we find that on problems where a smaller base model attains somewhat non-trivial success rates, test-time compute  can be used to outperform a 14x larger model.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors are investigating the relationships between scaling inference-time compute and training-time compute. With that said, the paper first investigates and characterizes different types of inference-time compute. There are two main reasons: the first is to change the LLM proposal distribution by having an additional set of tokens, and the second is to use a verifier to rank the best response. Building on top of this, they try to unify those two inference-time compute strategies together, and create a compute-optimal inference-time strategy that's conditioned on question/prompt (more specifically, question difficulties). Lastly they draw comparison to training-time scaling, and find that it is more beneficial to scale inference-time for easier questions.

### Strengths
- The unification of inference-time computing methodologies is useful. There are a sea of inference-time compute strategies, and the unification of this makes the discussion of it much easier. Although I have concerns that some other papers talk about inference-time compute method categorization, for details, see Question (1) below. This is not major, but I wish the authors can address and clarify.
- This unification also makes the experiments fairly comprehensive and thus having more confidence in the results.
- Trying to do a fair comparison between inference-time scaling and training-time scaling is novel. I think the authors have framed a good problem and showed a nice analysis.

### Weaknesses
- I partially disagree with the claim that the authors made at the end of the introduction. I think what the authors have shown doesn't mean scaling test-time computing can be preferable to scaling pretraining compute. For both medium and hard questions, using the same compute to scale pertaining works much more in favor of scaling inference compute. The advantage only comes in for easier questions, which I would argue is less important. Plus one can always do inference scaling for larger models. I think this claim may need to be justified more. 
- When training a verifier, is it more fair to include the compute to train the verifier as part of the inference-time compute? Similarly for the fine-tuned revision model.
- It is a bit difficult to fathom what "compute optimal" is exactly. How is this obtained or how is it optimized? I understand that strategies are selected based on question difficulties but providing the exact detail would be nice. 
- The separation of Verifier and Revision is a bit confusing as both require a PRM. The main distinction I think between sections 5 and 6 is one is using search and another is using revision.

### Questions
- Both [1] and [2] talk about the categorization of scaling inference-time compute. [1] separates inference compute into proposer and evaluator which is fairly similar to what this paper has proposed. [2] uses a more fine-grained definition to demonstrate different techniques. I   think both of them should be cited and discussed accordingly.
- Figure 3 (left) is a bit hard to interpret. What does each color mean?
Writing and Styling
- Line 076 "We find that...". The wording is a bit awkward. 

[1] Wang et al.,  Reasoning in Token Economies: Budget-Aware Evaluation of LLM Reasoning Strategies

[2] Saad-Falcon et al., Archon: An Architecture Search Framework for Inference-Time Techniques

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper investigates the best way to scale test-time computation in LLMs to improve their performance on challenging tasks (MATH). The authors analyze two primary mechanisms: (1) searching against dense, process-based verifier reward models, and (2) updating the model's proposal distribution over responses through sequential revision. They find that the effectiveness of different test-time compute strategies varies significantly based on question difficulty. Using this insight, they propose a "compute-optimal" scaling strategy that adaptively allocates test-time compute per question, achieving 4× better efficiency compared to best-of-N baselines. In a FLOPs-matched evaluation, the authors demonstrate that for problems where a smaller base model shows some success, test-time compute can outperform a 14× larger model.

### Strengths
1. The paper provides a unified perspective on test-time compute, namely modifying either the proposal distribution or searching with an external verifier. This framework helps systematically understand and analyze different methods.

2. The authors conduct comprehensive experiments on the MATH dataset using PaLM-2 models, comparing different test-time compute strategies across varying problem difficulties. The analysis is thorough, with detailed ablation studies showing the effectiveness of both mechanisms they proposed and the trade-offs between different approaches. 

3. The paper's findings have significant practical implications for model deployment. The demonstration that test-time compute can sometimes substitute for model size suggests a path toward deploying smaller, more efficient models while maintaining performance through clever use of inference-time computation.

4. The paper provides very insightful analysis on searching with PRMs and iterative self-refinemet, both of which have drawn lots of attention from researchers but still lack in-depth understanding. I pretty much enjoy the detailed analysis on the three ways of use of a PRM.

### Weaknesses
1. All experiments are done solely on the MATH dataset, which not only raises concerns on the genralizability of the conclusion, but also seems unfair when compared to scale up training time compute. Given multiple downstream datasets, we only need to scale up training time compute for once while have to scale up inference overhead every time. In this case, if we match the total test time compute with training time one, each task will be allocated much fewer inference time budget, which may lead to very different conclusions. 

2. When scaling inference time compute, dedicated efforts are needed to improve the verifier performance or models' capability of self-correction, which may also consume a non-trivial proportion of compute. This may also lead to unfair comparison to training time compute scaling.

3. Most presentations are excelllent but I there are still some minor issues. was a bit confused when reading Figure 3 (Left) since there is no legends and the overlap of bars make colors even more complex. Also, the main body oof the paper should be self-constrained, so it may not be quite suitable to put the results in Sec 5.3 to Figure 14 which is inconvenient to readers.

### Questions
- It seems all PRM results are presented in the way of *weighted* Best-of-N. Is PRM alone not working? If so, is it fair to compare lookahead search with the other two, given that the value in lookahead search can not be augmneted by majority vote?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper target the challenges in effectively utilizing additional computation at test time to improve the accuracy of their responses, particularly for complex tasks. This is important to explore how to tradeoff inference-time and pre-training compute. This paper trying to understand the scaling behaviors of test-time inference methods. This work analyze main mechanisms, and observed that the effectiveness of recent methods varies depending on the specific problem and the base LLM used.The observation motivates applying a “compute-optimal” scaling strategy, which acts to most effectively allocate test time compute adaptively per prompt. This approach selects the most effective method for utilizing additional computation based on the specific prompt and question difficulty.

### Strengths
The problem addressed in this paper is highly meaningful. This paper provide systematic analysis of different approaches for scaling test-time computes in LLMs. The observations could inspire further research and is beneficial for advancing the entire field.

The paper is well written and easy to follow.

### Weaknesses
1. Both the PRM and ORM models utilize the PaLM 2-S* base language model, which is costly. However, the computational cost associated with PRM during test time is not clearly defined and calculated.

2. In a specific domain, analyzing the trade-off between pre-training and test-time computation is more meaningful, but the paper primarily focuses on pre-training. As mentioned in the paper, "we expect test-time compute to be most helpful when models already have all the basic “knowledge” needed to answer a query, and instead the primary challenge is about drawing (complex) inferences from this knowledge." Pre-training can be leveraged to learn the basic "knowledge," while fine-tuning might be more effective in teaching the model how to draw complex inferences. Additionally, fine-tuning can be more efficient, providing greater benefits for the same computational cost.

3. Whether the observations are consistent across different datasets and models remains unclear. 

4. I think some details are unclear, which may affect the strength of the observations. Since the main contribution of this paper is conducting analysis using methods proposed in previous works, clarifying these details is important.

### Questions
1. What do the different colors in Figure 3 (left) represent? It appears the legend is missing.

2. What is the accuracy of the difficulty prediction? If it is quite accurate, the results with model-predicted difficulty may not be compelling because they rely on a powerful model for estimating difficulty and do not account for the associated computational cost, which is unrealistic. In real-world scenarios, using a powerful model entails significant costs, while employing a less powerful model may result in inaccurate difficulty predictions. Therefore, it would be valuable to demonstrate whether the observation holds true with a less powerful difficulty estimation model.

3. What is the accuracy of the verifier? Does the performance of the verifier influence the observations?

4. Is the accuracy of the verifier consistent across different difficulty levels? The observations might be influenced by the accuracy of the verifier.

5. How many test questions are there in each difficulty bin? Is it consistently 100 questions per bin?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper carries out comprehensive study on test-time compute. The authors study the topic from the following perspectives:
1. The first question they study is "How to scale test-time computation optimally". The authors propose an adaptive "computer-optimal" strategy for scaling up test-time compute. They evaluate the method against test-time inference methods, e.g., searching against dense, process-based verifier reward models, and updating the model's distribution over a response adaptively, given the prompt at test time. They study how to optimally combine these methods to reduce the test-time budget and achieve better performance.
2. Then they study to what extent test-time computation can substitute for additional pre-training, by comparing test-time inference on a smaller model compared to pre-training a ~14x larger model, under a FLOPs-matched comparison.

Finally, the authors summarize their results with a finding that states *"in some settings it is more efficient to pretrain smaller models with less compute, and then apply test-time compute to improve model outputs"*.

### Strengths
1. The paper is fluent and coherent. The authors have clearly made their points so that the ideas they propose can easily be followed.
2. The paper carries out comprehensive experiments and analysis on the research questions they study, i.e. 1) scaling test-time compute optimally and 2) exchanging pre-training and test-time compute.
3. The experiments are sound.
- For the first question, the baseline they choose and develop over, e.g. best-of-n and revision, are strong, which makes their points more convincing. And the study on sequential to parallel ratio is also helpful for readers to better understand the merits of different test-time inference methods and how they contributes to the "compute-optimal" strategy.
- For the second question, the setup for FLOPs matched comparison is plausible and the ablations on model size and question difficulties are insightful, making their conclusions convincing and easy to understand.

### Weaknesses
1. Yet the adaptive "compute-optimal" strategy achieves great performance over other test-time baselines and can be used to substitute for pre-training in some cases, for me it's more like a proof-of-concept.
- The experiments are all on MATH and 
- there're still many hyper-parameters under-explored, e.g.,
  - different verifiers;
  - different design of difficulty bins (number of bins and how to estimate difficulties),

along with the ones that have been explored in the paper:
  - types of test-time methods to select from (my understanding is we can have a set of many test-time methods to adaptively select from)
  - test-time compute need for different methods

It'll be much clearer and more helpful if the authors can discuss more about the possible "hyper-parameters", compensating the framework described in Eq. 1. (I see some of them are in the appendix, e.g. design of difficulty bins, but it would be much clearer for me if they can be put and discussed at the same place)

### Questions
1. My understanding from the paper is that different FLOPs budgets of SFT can also be taken into consideration, and we may be able to teach LLMs better specialized skills (e.g., revision) with more deliberate post-training. I don't see much ablations on the revision SFT in the paper. Do you think that could be the case?
2. This is an interesting and insightful paper. I have another question (better called thought) -- what if we scale on both test-time compute and pre-training compute, would there be another optimal solution under FLOPs-matched comparison? The paper only focuses on 1-to-1 exchange (scaling test-time compute only) but maybe that would also an interesting topic.

### Soundness
3

### Presentation
3

### Contribution
4
