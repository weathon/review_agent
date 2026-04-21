# Routoo: Learning to Route to Large Language Models Effectively

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 3, 5

## Abstract
LLMs with superior response quality—particularly larger or closed-source models—often come with higher inference costs, making their deployment inefficient and costly. Meanwhile, developing foundational LLMs from scratch is becoming increasingly resource-intensive and impractical for many applications. To address the challenge of balancing quality and cost, we introduce Routoo, an architecture designed to optimize the selection of LLMs for specific prompts based on performance, cost, and efficiency. Routoo provides controllability over the trade-off between inference cost and quality, enabling significant reductions in inference costs for a given quality requirement.
Routoo comprises two key components: a performance predictor and cost-aware selector. The performance predictor is a lightweight LLM that estimates the expected performance of various underlying LLMs on a given prompt without executing them. The cost-aware selector module then selects the most suitable model based on these predictions and constraints such as cost and latency, significantly reducing inference costs for the same quality. 
We evaluated Routoo using the MMLU benchmark across 57 domains employing open-source models. Our results show that Routoo matches the performance of the Mixtral 8x7b model while reducing inference costs by one-third. Additionally, by allowing increased costs, Routoo surpasses Mixtral's accuracy by over 5\% at equivalent costs, achieving an accuracy of 75.9\%. When integrating GPT4 into our model pool, Routoo nearly matches GPT4's performance at half the cost and exceeds it with a 25\% cost reduction. 
These outcomes highlight Routoo's potential to significantly reduce inference costs without compromising quality, and even to establish new state-of-the-art results by leveraging the collective capabilities of multiple LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a LLM model routing system called Routoo that selects what LLM to route queries to from a pool of LLMs of various capabilities and inference costs. For a particular query, and a particular LLM, a predictor model (based on Mistral7b) predicts the accuracy score, and an average cost heuristic estimates cost. Given a list of queries and a total budget, an optimization heuristic assigns the appropriate LLM for each query to achieve the highest overall accuracy within the total budget. On the MMLU benchmark, the Routoo system achieves higher accuracy at lower costs than using any single state-of-the-art model. Higher accuracy is achieved since the pool of models have specialists for particular type of queries and lower cost is achieved since easier queries don't need more expensive models to solve them.

### Strengths
1. This work introduces heuristics for optimizing for multiple queries with a total budget which is novel.
2. This Routoo system has great results where it Pareto-dominates state-of-the-art single LLMs on inference cost and MMLU performance.
3. The gathering of non-MMLU dataset for training along with synthetic data is meticulous in avoiding data leakage during evaluation.
4. The approach is sound: the predictive model for score prediction and budget constrained selector heuristic makes sense. The per-LLM embedding allows for the model to be inferenced only once-per-query instead of once-per-query-per-LLM which saves compute cost.

### Weaknesses
While the approach in this work is sound, the contribution of the work over previous works may not be significant enough. Furthermore, the work is more appropriate for machine learning systems without a significant learning aspect.
1. There are related works regarding predictive routing with multiple-LLMs that are not discussed [3,4]
2. The advantages of specialist models improving scores over strong single-LLM models are covered in [1,2] so that aspect of the system is not very novel.
3. The concepts of budget conscious LLM routing is covered in [3,4] so that aspect of the system is not very novel either.
4. The only learning aspect of this work is the predictive model (Mistral7b with learned classification head). Without accuracy measurements, and comparisons with other models, there is little insight to be gained.

[1] Large Language Model Routing with Benchmark Datasets

[2] Harnessing the Power of Multiple Minds: Lessons Learned from LLM Routing

[3] Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing

[4] RouteLLM: Learning to Route LLMs with Preference Data

### Questions
1. The Mistral7b based prediction has an inference cost. Is that included in the inference cost?
2. Is a LLM needed for prediction or would a lightweight retrieval system that retrieves scores on similar queries from the training data work well?
3. In Section 3.3, step 3 of the algorithm (budget management) mentions that the next best model is chosen if the budget is exceeded. However, the budget is the total budget across all queries so how is the ordering of this step performed across queries? Wouldn't the earlier queries starve the budget of the later queries, leading them to choose worse models?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
Large language models (LLMs) with superior performance often come with high inference costs, making their deployment inefficient and costly. This paper developed a routing framework, Routoo, to optimize the selection of LLMs for specific prompts based on performance, cost, and efficiency. Evaluation results on MMLU benchmark and 57 models demonstrate the effectiveness of Routoo.

### Strengths
S1. This paper proposed Routoo, a routing framework that intelligently identifies the best-performing LLM for a given query while considering constraints such as cost and latency.

S2. Routoo achieves competitive performance with GPT4 at half the inference cost and surpasses it by reducing the inference cost by 25%.

### Weaknesses
W1. Unclear novelty and limited technical contribution. Training a router to harness the respective strengths of different LLMs has been widely studied [1,2,3,4,5]. Specifically, [5] studied a very similar setup - select LLMs to maximize (resp., minimize) overall performance (resp., cost) subject to a pre-defined cost budget (resp., performance threshold) - and developed various strategies including the greedy algorithm proposed in this work, which authors did not discuss or compare to.

W2. No baseline. Provided the rich literature on LLM routing as discussed above, no baseline included in the evaluation seems insufficient. This work can be greatly improved if at least one baseline from the general routing [1-4] and one baseline from the constrained routing [5] were included and compared to Routoo.

W3. Authors leveraged the Mistral-7B model as the query encoder, which may cause non-negligible cost & latency overheads. Authors may want to perform overhead analysis to address this concern.


References:  
[1] Routing to the Expert: Efficient Reward-guided Ensemble of Large Language Models, https://arxiv.org/pdf/2311.08692  
[2] Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing, https://arxiv.org/pdf/2404.14618  
[3] ROUTERBENCH: A Benchmark for Multi-LLM Routing System, https://arxiv.org/pdf/2403.12031  
[4] Large Language Model Routing with Benchmark Datasets, https://arxiv.org/pdf/2309.15789  
[5] Fly-Swat or Cannon? Cost-Effective Language Model Choice via Meta-Modeling, https://arxiv.org/pdf/2308.06077

### Questions
Q1. In Sec 3.3, authors consider an average cost for each model to estimate the model inference cost for a given test query. Since the inference cost heavily relies on the output length which is unknown beforehand, it is intriguing to see how good the cost estimation is. Analysis on estimation error and variance would be very helpful.

Q2. In Sec 3.3, the formulated problem is really a knapsack problem - select a subset of LLMs (items) such that the overall performance (profit) is maximized subject to cost budgets (weight threshold) - which has been widely studied in prior literature [1]. Provided that, greedy algorithms to knapsack problem has been known to be non-optimal in general cases. Authors may want to discuss related work and justify the specific algorithm design.

References:  
[1] Knapsack Problems, https://link.springer.com/book/10.1007/978-3-540-24777-7

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
- The authors propose Routoo, an architecture that learns to select an LLM best-suited to answer a specific query when weighing the factors of performance (accuracy) and cost. The idea is, we don’t need to pay extra for a larger or closed-source LLM on queries where we expect that a smaller, cheaper LLM will perform similarly.
- Routoo chooses the LLM for a query from among a universe of complementary models, and the authors describe a “Universe Constructor” that creates this set automatically.
- The key components of Routoo are the “Performance Predictor” and “Cost-Aware Selector”.
  - At inference time, we want to be able to predict which LLMs are likely to perform well on a given query. The Performance Predictor is an LLM that takes as input the query and a model from the available set, then predicts whether that given LLM will answer the query correctly. The performance predictor consists of an LLM for embedding the query, an embedding model for embedding the candidate LLM, and a linear layer for predicting the performance, given the embedded LLM and the embedded query.
  - The Cost-Aware Selector uses each LLM’s cost and Performance Predictor score to select an LLM to answer the query. This component uses a greedy algorithm that calculates the Performance-to-Cost ratio for each candidate model, then chooses the candidate model with the best ratio that fits in the cost budget.
- Evaluation is performed using the MMLU dataset. An open-source version of Routoo is built from each size of Llama2, Mistral 7b and Mixtral 8x7b, while a mixed-source version adds GPT3.5 and GPT4-turbo. 
  - For the main results, the closed- and mixed-source Routoos are compared for cost and performance against each of these individual models.
  - The performance-cost curve is compared for Routoo with different budgets and the individual LLMs
  - The performance of Routoo is compared against Llama2 70b and Mixtral 8x7b for each domain within MMLU

### Strengths
- The paper is approachable and well-written
- Framing the balancing of performance and cost as an optimization problem with a greedy approximation is an elegant approach
- The finding that much of the performance gap can be addressed by domain-specialized models is interesting and provides context on why Routoo performs well
- As the authors state, many prior routing approaches (such as the cascading technique) execute multiple LLMs in sequence. Routoo’s inference-time technique of predicting models’ performances before actually using the models is clever.
- Routoo is flexible in that the “cost” consideration can easily be expanded to include criteria such as latency, and that the “performance” consideration can be any integer-valued performance-related score instead of just accuracy.

### Weaknesses
- The experimental results are promising but a little bit sparse right now, I list some more experiments that could be useful in the Questions section below
- It seems challenging to extend this method beyond datasets like MMLU that have multiple-choice answers
- The Universe Constructor builds a universe of candidate models in accordance with the paper’s Equation 2. This equation uniquely takes into account the model performance, and does not consider model cost. This approach should be modified to also consider including cheaper models at times, for instance by maximizing for performance-to-cost ratio (or some weighted version of performance-to-cost ratio that allows users to specify how much they want to prioritize cost).
- “We could not find any publicly available trained models for routing or model integration”: looking online for a few minutes, it seems this research area does have a dearth of methods with publicly released code. However, there might be some things on GitHub (e.g., https://github.com/emingenc/llm_adaptive_router) and the cascading technique (using the cheapest model, then going to a more expensive one if the cheaper one fails) would be relatively easy to implement as a routing-based baseline. Right now, the evaluations only compare Routoo against each individual LLM, with no other routing methods included.
- I don’t understand the claim in the Introduction (and restated at the very end of evaluation) that “Routoo addresses … the development cost of building LLMs from scratch, creating a composite high-performance model without the need for extensive retraining”. This makes it sound like Routoo is making it less costly to develop novel LLMs, when really Routoo is about choosing which pre-existing LLMs to use. If this claim is saying that Routoo is cheaper than, for instance, combining all the LLMs using MoE into a single very large model, then the claim holds up. However, model routing is already an existing technique that addresses this claim from this non-MoE perspective: it’s not something that Routoo itself is contributing. Model routing (including Routoo) and MoE already have very different use-cases, so the author’s comparisons of Routoo to MoE don’t seem very natural or could perhaps be better-articulated.

### Questions
- What is the comparison of the cost for obtaining Routoo’s training data (including the cost of generating the synthetic training data from GPT-4) plus doing inference, versus the cost for just running on Mistral/GPT/etc?
- How accurate is the Performance Predictor with predicting whether a given LLM will, indeed, answer a query correctly? Does it struggle more with queries from specialized domains, or other specific kinds of queries?
- The Performance Predictor takes as input the difference between the LLM’s and query’s embeddings. Are there other ways of combining these two inputs, such as convolutions or concatenations, that could perform better?
- Experiment comparing performance for different budget sizes and model universe sizes
- I would be interested to see results for different hyperparameters, for instance especially K. Additionally, the choice of M=56 seems like a very specific number, how robust is Routoo to these sorts of changes?
- Can you provide more information on the training process? You mention in the “Architecture Setting” paragraph of Section 4.1 that the Mistral query encoder is being fine-tuned with LORA– are you training the model embedding function Emb() and the projection matrix Linear() without LORA, but during the same time you’re fine-tuning Mistral? Also, how are the Emb() and Linear() weights initialized?

Overall, I think that Routoo might be promising, but I do not have confidence in its capabilities without seeing additional experimental results. My score is 3, but I will consider raising it if the authors could include more evaluations.

### Soundness
2

### Presentation
3

### Contribution
3
