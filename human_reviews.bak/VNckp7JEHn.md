# Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving

- Decision: Accept (Poster)
- Scores: 8, 6, 3, 6

## Abstract
While the scaling laws of large language models (LLMs) training have been extensively studied, optimal inference configurations of LLMs remain underexplored. 
We study inference scaling laws (aka test-time scaling laws) and compute-optimal inference, focusing on the trade-offs between model sizes and generating additional tokens with different inference strategies. As a first step towards understanding and designing compute-optimal inference methods, we studied cost-performance trade-offs for inference strategies such as greedy search, majority voting, best-of-$n$, weighted voting, and two different tree search algorithms, using different model sizes and compute budgets. Our findings suggest that scaling inference compute with inference strategies can be more computationally efficient than scaling model parameters. Additionally, smaller models combined with advanced inference algorithms offer Pareto-optimal trade-offs in cost and performance. For example, the Llemma-7B model, when paired with our novel tree search algorithm, consistently outperforms the Llemma-34B model across all tested inference strategies on the MATH benchmark. We hope these insights contribute to a deeper understanding of inference scaling laws (test-time scaling laws) for LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents an empirical analysis of inference scaling laws for large language models, focusing on the trade-offs between model sizes and generating additional tokens with different inference strategies. The authors explore how varying the compute during inference affects model performance after the model has been trained. They compare various inference techniques, such as greedy search, majority voting, best-of-n, weighted voting, and two tree search algorithms, using different model sizes and compute budgets.

### Strengths
S1. The standpoint of this paper is interesting and can provide some insights into the LLM community.

S2. The paper provides an extensive empirical study on the relationship between model size, inference computation, and performance, which is valuable for both research and practical deployment of LLMs.

S3. The paper analyzes not only the performance of different inference strategies but also provides a theoretical analysis of voting methods, offering insights into their convergence behavior.

S4. The experimental results are clearly presented, making it easy to understand the performance trade-offs between different models and inference strategies.

### Weaknesses
However, there remain some concerns for me:

W1. The analysis is focused on mathematical reasoning tasks, which may limit the generalizability of the findings to other types of problems. In detail, the paper does not provide extensive discussion on how these findings might generalize to other domains beyond mathematical reasoning.

W2. The study is based on specific two LLMs, and it's unclear how the results would extend to other model architectures.

W3. Some of the figures are a bit unclear. For example, in Figures 2 and 3, although the description is vivid, the set font is a bit hard to read.

### Questions
Q1: How well do the findings generalize to non-mathematical reasoning tasks, such as natural language inference or summarization?

Q2: How sensitive are the results to the specific model architectures used in the study? Would different architectures yield similar findings?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this work, the authors study how varying the amount of compute utilized during inference (i.e., varying the number of samples collected during decoding) affects model accuracy on a task. In particular, they explore how to maximize accuracy given a specific compute budget (measured in FLOPs). They experiment with Pythia, Mistral and Llemma models on 2 mathematical reasoning tasks. In experiments, they vary the inference strategy among sampling and tree search methods, coupled with best-of-n selection or weighted majority voting among the generated candidate sets.  Their results show that in many cases, under small compute budgets, smaller models with more compute-intensive inference yields the best results; under large compute budgets, larger models perform better. The results also suggest that their proposed tree-search method–ReBASE–tends to yield the optimal compute-accuracy tradeoff across compute budgets and models. The paper also includes an asymptotic analysis of voting methods as the number of samples goes to infinity. Based on these results, the authors defend a need for more sophisticated inference algorithms.

### Strengths
**Topic and Formalization.** I think the question being explored in this paper is interesting and timely–especially considering recent papers, such as those discussed in the concurrent work section. I also find the formalization–maximizing performance under a compute budget–attractive and satisfying. As such, the work is well-motivated, easy to follow and likely to be relevant to many.  Practitioners would be well-suited to consider this type of analysis when deciding what models to deploy.

**Well-designed experiments & complete analysis.** For the experiments that were run, the authors did a good job designing the experiments, and analyzing and visualizing results. The methods selected for comparison are good. I believe that the compute budgets selected are _likely_ good however it would be helpful to report the number of samples drawn for each model/method under each compute budget to make sure they cover a sufficiently large, yet realistic, number of samples.  Generally, it is easy to use the visualizations to see which methods and models are best under various compute budgets. In addition to the discussion of the main results, the discussion of the affect of tree search on weaker models as well as performance saturation adds to the paper. The results make sense and are likely useful. I appreciate the inclusion of error bars.

**Well-written.** The writing is clear and the paper is easy to follow. 

Overall, the paper exhibits a number of good characteristics. It’s well written, the topic is interesting, it provides some theoretical analysis, the experiments are solid and the results are relevant.

### Weaknesses
**Mathematics Tasks Only.**  The experiments only study two datasets, both of which are related to mathematical reasoning. As such, I’m concerned about the generalizability to new domains and new tasks. I think this paper could be strengthened with experiments on additional tasks like question answering and/or its variants.

**Limited models tested.**  The set of models tested leaves something to be desired. In my understanding and experience, Pythia models tend to be weak in comparison to newer open-source models, such as the Llama3.1 family of models. Lemma is a special purpose model for the mathematical domain; Mistral-7B is generally understood to be a strong model.  As such, I’m a bit worried that the conclusions would not extend to better models. As is reported, the proposed method, ReBASE, tends to help the weakest models and might not provide the same advantages for a model like Llama3.1-7b-instruct. 

**Some results seem insignificant.** Figure 7 shows a lot of overlap amongst the methods tested. This is especially the case for Mistral-7B (which is perhaps the strongest model tested).  This overlap exists in Figure 6 as well; albeit, less so. Again, this makes me concerned about how much the results extend to better models. And, in Figures 4 and 5, ReBASE with the larger models doesn’t seem to be significantly better than sampling.

Overall, I’m concerned about the generalizability of the results to new tasks, and the extent to which the results would hold for the most performant models.

### Questions
- What is the difference between ReBASE and beam search with sampling turned on? They appear to be quite similar.
- Not a question, but I think that you may want to discuss the “Dot-by-dot” paper (https://arxiv.org/abs/2404.15758) since it also concludes that generating more tokens can be helpful; albeit their setup is much different from yours.
- In the expression denoting the reward of a node r_k (last line of page 6), what was q?
- In Figure 5, what happens if you continue sampling from the 7B model with ReBASE until the maximum number of FLOPs considered (i.e., if we extend the orange line all the way to the right, what does it look like)?
- Why is test error reported instead of task accuracy? Please report the test accuracy for the sake of comparison to other work.
- Can you report the number of samples you were able to draw from each model / inference method corresponding to the number of FLOPs used?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The study investigates the inference scaling laws for large language models (LLMs), focusing on finding compute-optimal inference configurations. The authors examine the trade-offs between different model sizes and inference strategies, analyzing how these factors impact computing costs and model performance. Through empirical analysis, the study finds that smaller LLMs can outperform larger models given the same computational resources by employing advanced inference strategies. The paper also introduces a compute-optimal inference algorithm which can help small LLMs achieving better cost-performance tradeoffs.

### Strengths
1. The study’s motivation of studying the selection of model size and inference strategy under limited computation budgets is meaningful and justified.
2. The formal definitions and proofs presented in the paper are clear and maintain good readability.
3. The integration of a reward model to guide LLMs' inference process is reasonable.
4. The experimental results are clearly presented and easy to read.

### Weaknesses
1. The paper lacks technical contribution. The proposed REBASE framework merely integrates process reward models (PRM) into the existing Monte Carlo Tree Search (MCTS) framework. Considering these components are well-established and the simplicity of the REBASE framework, there are concerns regarding the novelty of the work.
2. While the paper employs a PRM as the reward model within REBASE, it fails to detail the training and inference processes of the reward model. Moreover, it does not address how to ensure the reliability of the rewards generated by this model. An unstable or suboptimal reward model could significantly impact the overall framework, and the authors should consider this aspect.
3. The paper provides theoretical analyses for four inference strategies: greedy search, majority voting, best-of-n, and weighted voting, but does not offer comparative experimental results. A detailed experimental comparison involving different model sizes and inference strategies should be included.
4. Similar to traditional MCTS methods, REBASE necessitates extensive intermediate interactions with LLMs before obtaining the final generated results. Additionally, it introduces the extra PRM component. This could lead to significant resource consumption. The authors should provide data analysis on resource usage and runtime to demonstrate whether REBASE is more efficient than conventional MCTS methods.
5. According to the experiments in Section 4.2, both small and large LLMs show high error rates under limited computation scenarios, suggesting that both models are not good enough in these situations. Claiming that smaller models perform better in such conditions is not sufficiently convincing.
6. The experiments in Section 4.3 show inconsistent performance trends of REBASE compared with sampling methods on the MATH and GSM8K datasets. The authors attribute this inconsistency to differences in dataset difficulty, which seems unconvincing. The authors should provide more credible explanations.
7. In the scenarios where REBASE is purportedly advantageous (i.e., less compute-intensive situations), the performance on the MATH dataset with Llemma-34B and Mistral-7B models is either weaker than or merely comparable to the Sampling method. This indicates that REBASE does not consistently achieve improvements. An explanation for this should be provided.
8. The paper only uses two baselines: sampling and MCTS, without elaborating on their implementation details. It would be beneficial to include the latest baselines and provide detailed introductions and comparisons of the baselines and the proposed REBASE method.

### Questions
Please answer the questions in the weakness section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on the intriguing topic inference scaling laws. The authors conducted a detailed analysis using language models of various sizes on math-related datasets, GSM8K and MATH. They discovered that, under the same computation cost, smaller models might outperform larger ones. However, once the performance of smaller models saturates, larger models may achieve better results. These phenomenon is quite significant. Subsequently, the authors carried out theoretical analyses and developed a novel tree search algorithm called REBASE, aiming to achieve a Pareto-optimal trade-off between performance and computational efficiency, following by experiments to demonstrate the effectiveness of the REBASE method.

### Strengths
- This paper conducted experimental analysis using language models of varying parameter sizes on math-related datasets GSM8K and MATH. It was observed that smaller models can sometimes outperform larger models in terms of performance, given the same inference computational cost. Furthermore, the authors noted that once the performance of smaller models reaches saturation, their performance tends to be inferior to that of larger models. These observations are significant.
- The author conducted a theoretical analysis, demonstrating that the performance of both standard and weighted majority voting inference strategies will converge when there is an infinite number of samples and the performance after convergence depends solely on the distributions modeled by the language model and the reward model. These theoretical analysis aligns with the experimental results.
- The author proposes a new inference strategy that can achieve comparable or even better performance with less computational overhead than MCTS.
- The organization of the article is clear, and the writing is excellent. It's easy to understand when reading.

### Weaknesses
- The authors conducted an analysis of inference scaling laws on Pythia and Llemma. However, they did not perform REBASE experiments on Pythia. Instead, they used Llemma and Mistral-7B for these experiments. It would be more convincing if the author could also conduct REBASE experiments on Pythia.
- This paper only conducts experiments on mathematical datasets. Although the authors acknowledge this limitation, their title is "LLM Problem-Solving" rather than "LLM Math Problem-Solving". In fact, the analysis and method proposed by the authors are not confined to math-related problem-solving. A clear indication of this is that in Section 3 (theoretical analysis and the introduction of the REBASE method), the term "Math" is not mentioned. This suggests that similar phenomena may occur in other problem-solving scenarios. Conducting experiments in a broader range of problem-solving areas, such as coding and logical reasoning or others, would enhance the generalizability of the authors' findings and contribute more significantly to the community.
- The models Pythia, Mistral 7b, and Llemma were introduced quite some time ago. In the past year, more powerful open-source LLMs like Llama3 have been proposed, achieving remarkable advancements across various metrics. It would be beneficial if the author could continue to analyze these newly introduced and widely used models, such as Llama3.

### Questions
- The authors analyzed inference scaling laws using Pythia and Llemma but did not include REBASE experiments for Pythia, opting instead for Llemma and Mistral-7B. Including REBASE experiments on Pythia could strengthen the findings.
- The paper focuses on mathematical datasets, yet the title "LLM Problem-Solving" suggests a broader scope beyond math. The methods and analysis in Section 3 do not specifically mention "Math", indicating potential applicability to other areas. Exploring additional problem-solving domains, like coding or logical reasoning, could enhance the study's relevance and impact.
- While the models Pythia, Mistral-7B, and Llemma have been around for some time, recent advancements with models like Llama3 show significant progress. Analyzing these newer models could provide valuable insights and keep the research up-to-date.

### Soundness
2

### Presentation
3

### Contribution
3
