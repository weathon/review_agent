# Test-Time Scaling in Reasoning Models Is Not Effective for Knowledge-Intensive Tasks Yet

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
Test-time scaling increases inference-time computation by allowing models to generate long reasoning chains, and has shown strong performance across many domains. However, in this work, we show that this approach is not yet effective for knowledge-intensive tasks, where high factual accuracy and low hallucination rates are essential. We conduct a comprehensive evaluation of test-time scaling using 14 reasoning models on two knowledge-intensive benchmarks. Our results reveal that increasing test-time computation does not consistently improve accuracy and, in many cases, it even leads to more hallucinations. We then analyze how extended reasoning affects hallucination behavior. We find that reduced hallucinations often result from the model choosing to abstain after thinking more, rather than from improved factual recall. Conversely, for some models, longer reasoning encourages attempts on previously unanswered questions, many of which result in hallucinations. Our analysis reveals that extended reasoning can induce confirmation bias, leading to overconfident hallucinations. Despite these limitations, we observe that compared to non-thinking, enabling thinking remains beneficial.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates whether test-time scaling is effective for knowledge-intensive tasks that require high factual accuracy and low hallucination, similar to the observations for reasoning-intensive tasks such as mathematical reasoning. Though enabling thinking performs better than non-thinking LLMs, increasing test-time scaling cannot result in consistent improvements in factual accuracy and often fails to reduce hallucinations based on experiments on reasoning models. Note that the "test-time scaling" for open-sourced models such as DeepSeek-R1-Distill models actually refers to appending "wait" to extend the model's thinking and for proprietary models the thinking tokens/time is increased in the API call configurations.

### Strengths
+ The experiments are comprehensive, conducted using 14 models on two benchmarks.
+ The observations may be inspiring for future work in this direction about how to improve reasoning models on knowledge-intensive tasks.
+ The authors conducted a fine-grained analysis by comparing the performance between low-reasoning and high-reasoning modes and identified the correlation between the models' willingness to answer and hallucination changes, which may to some extent reflect the incapability of LLMs in answering specific questions.
+ several case studies are conducted on some of the most powerful proprietary models.

### Weaknesses
+ The evaluation is based on one LLM (GPT-4o-mini). As it is easy for LLMs to be hacked even when reference answers are provided, the authors should sample some test cases and report human agreement. The authors could also evaluate the results using two LLM evaluators. 
+ No promising solutions are proposed. The authors are encouraged to alleviate this problem by attempts such as using a confidence threshold to force abstention when confidence is low to reduce hallucination at a reasonable computational cost.

### Questions
see the weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper conducts case studies to understand the effect of test time scaling and more thinking on hallucination and factual accuracy of knowledge intensive tasks. They observe it may not reduce hallucination or increase accuracy, in fact in some cases increases it. Sometimes hallucinations may reduce but that often can be attributed due to model obstaining from further thinking.

### Strengths
There are some interesting observations from the case studies in the paper

### Weaknesses
- I feel the paper is better suited for a blog (or at best a short paper). I don’t think it has enough content or enough depth for a long paper 
    - While the observations are interesting, they are not entirely new and some of them are indeed intuitive.  The paper simply does not go deep enough into each of these observations or prescribe any specific method to mitigate this.
    -  The conclusions are quite high level, many of them are just some observations from some instances or examples. While I am not judging the correctness of it, these are probably correct - just that I don’t see any fundamental contribution of the work (like some principled approach or some significant novelty) beyond just listing these observations. 
    - 
- Test time scaling is a very broad term, there are multiple ways to do that each driven by different signals and each having its own complexity. There are also more strategic (for e..g self-evolutionary or tree search ) mechanisms of iteratively improving something, which can also test time scaling methods.  In my understanding this paper has taken a very simple approach to test time scaling which is simply setting the thinking budget or effort parameters for specific frontier models. While very simple test time scaling strategies (like controlling budget parameters) may not effectively improve performance (or even degrade it), there are other more principled methods (MCTS, AlphaEvolve etc) which may not suffer from the same conclusions. For meaningfully doing such a study it has to be more comprehensive in terms of models and baselines used. For closed methods, undoubtedly there would be less control on the budget parameters. To get deeper insights it might be more interesting to examine open source models more closely.

### Questions
See the weakness section

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper investigates the impact of extending models' reasoning length on their performance in factual question-answering tasks. Through extensive experiments on 14 models and 2 benchmarks, the authors show that simply increasing the reasoning length does not consistently improve factual accuracy. The paper further analyzes the changes in "no attempt" rates and hallucination rates for different models, finding that longer reasoning often leads to more hallucinations.

### Strengths
1. Insightful conclusion: The main finding that longer reasoning does not help models recall facts they do not know, but can instead induce hallucinations, is a significant and insightful contribution. This challenges the common assumption that more computation uniformly leads to better performance. 

2. Thorough experiments and nuanced analysis: The study is comprehensive, covering 14 models with different scaling mechanisms. The authors perform a detailed analysis of how different models behave, and the qualitative case studies are well-chosen to explain why hallucination rates change. The paper's clear distinction between the benefits of enabling thinking (vs. non-thinking) and the ineffectiveness of extending thinking is a crucial and well-supported nuance.

### Weaknesses
1. Superficial definition of hallucination: The paper's primary metrics rely on classifying the final answer as "correct," "incorrect," or "not attempted". Hallucination is simply defined as an "incorrect" final answer. This approach fails to analyze the emergence of hallucinations within the reasoning process itself. The paper's most interesting qualitative finding is that models fabricate facts during long reasoning, yet this is not systematically or quantitatively measured. Besides, the " overconfident hallucinations" is similar to the "overthinking" described in [1], but [1] focus on the correctness of reasoning process as well as the final answer. The study would be much stronger if it analyzed the factual correctness of the reasoning steps, not just the final output.

[1] Chen X, Xu J, Liang T, et al. Do NOT Think That Much for 2+ 3=? On the Overthinking of Long Reasoning Models[C]//Forty-second International Conference on Machine Learning.

2. Problematic experimental methods and limited scope: For Qwen3 and DeepSeek-R1-Distill models, the "budget forcing" method (appending "Wait") is a highly artificial intervention. This unnatural method of forcing a model to continue reasoning may be the cause of the observed hallucinations, rather than the extended reasoning per se. Furthermore, the paper's claim is about "test-time scaling" in general, but its investigation is limited to only one form: sequential (long CoT) scaling. It explicitly omits other traditional TTS methods, such as parallel approaches like multiple sampling and self-consistency. This limitation narrows the scope and impact of the main conclusion.

### Questions
1. When extending the reasoning length, does the frequency of factual errors (hallucinations) within the reasoning process itself increase?

2. The paper shows that models fabricate facts during long reasoning. In the tests on the FRAMES dataset, how many errors are due to such fabricated facts, and how many are due to multi-hop reasoning errors (e.g., errors in logic or composition)?

3. In the comparison between "thinking" and "non-thinking" models, what is the typical reasoning length (in tokens) for the "thinking" mode? How does this length compare to the "extended" reasoning lengths that were found to cause more hallucinations?

4. Does your analysis suggest that an optimal reasoning length selection strategy exists?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper benchmarks reasoning models' behaviors with test-time scaling on two knowledge-intensive benchmarks, showing that more reasoning does not consistently improve accuracy or reduce hallucination for almost all models. There are also a few exemptions where more thinking budgets do help the model, and the authors provide case studies. Specifically regarding hallucination, experiments show that the changes are due to the models' willingness to answer. However, for models that support both "thinking" and "non-thinking" modes, the thinking mode does improve performance.

### Strengths
The paper investigates on sufficient model families, concluding basically consistent behaviors while also providing case studies. The main message is clear and the paper is well presented overall.

### Weaknesses
* While the number of models is large enough, the number of datasets / benchmarks seems to be insufficient. Can you elaborate on more datasets to show the universality?
* The system prompt is fixed across the paper. Did you try out different prompts? Would all prompts lead to similar conclusions?
* The main conclusion is interesting yet intuitive. The authors fail to provide any theoretical justification. What is the fundamental difference between thinking mode and non-thinking mode since they former leads to better performance? Why further increasing thinking budget would not lead to better performance? 

Overall, this is a benchmarking paper with straightforward conclusions. It would be better if the authors could provide more theoretical analysis or intuitions, as well as appropriate solutions to improve model performance when the thinking budget is increased. Otherwise, the contribution is somewhat limited.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
