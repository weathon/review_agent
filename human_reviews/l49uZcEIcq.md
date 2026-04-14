# Verbosity $\neq$ Veracity: Demystify Verbosity Compensation Behavior of Large Language Models

- Decision: Reject
- Scores: 5, 8, 5

## Abstract
Although Large Language Models (LLMs) have demonstrated their strong capabilities in various tasks, recent work has revealed LLMs also exhibit undesirable behaviors, such as hallucination and toxicity, limiting their reliability and broader adoption. In this paper, we discover an understudied type of undesirable behavior of LLMs, which we term **Verbosity Compensation (VC)** — similar to the hesitation behavior of humans under uncertainty — where they respond with excessive words such as repeating questions, introducing ambiguity, or providing excessive enumeration.  We present the first work that defines and analyzes Verbosity Compensation, explores its causes, and proposes a simple mitigating approach. Our experiments, conducted on five datasets of knowledge and reasoning-based QA tasks with 14 newly developed LLMs, reveal three conclusions. 1) We reveal a pervasive presence of VC across all models and all datasets. Notably, GPT-4 exhibits a VC frequency of 50.40\%. 2) We reveal the large performance gap between verbose and concise responses, with a notable difference of 27.61\% on the Qasper dataset. We also demonstrate that this difference does not naturally diminish as LLM capability increases. Both 1) and 2) highlight the urgent need to mitigate the frequency of VC behavior and disentangle verbosity with veracity. We propose a simple yet effective cascade algorithm that replaces the verbose responses with the other model-generated responses. The results show that our approach effectively alleviates the VC of the Mistral model from 63.81\% to 16.16\% on the Qasper dataset. 3) We also find that verbose responses exhibit higher uncertainty across all five datasets, suggesting a strong connection between verbosity and model uncertainty. We will release our code and dataset upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces a novel concept of verbosity compensation in LLMs. They define this term, explore it's presence in different models using combination of different datasets, also they try to come up an algorithm to tackle this issue of VC. The paper defines verbosity compensation as a tendency of LLMs to generate overly verbose responses when the LLMs are unsure about the answer. Coming to the evaluation part there are different sections of it with combination of automated evaluation and manual human evaluations. The paper does have some interesting things but there are some issues with the actual implementation and also writing, it is not very clear the exact research question the paper is trying to solve.

### Strengths
The model has a new concept of verbosity compensation

They explore 14 different LLM for comparison and performance evaluation

The idea of different type of VCs

The work also introduces algorithm to mitigate the issue that has been defined in the paper

### Weaknesses
- > The problem doesn't seem to very well defined, I couldn't understand the exact research question the paper is trying to solve.
If the problem is about the length of output, I don't see it as a major issue.

 -> The dataset used for this experiment is creates using existing datasets from previous works, most of these are from previous years and there is high chance that these were in the training data that was used for training different LLMs used in the paper and the details of these data-leakage are not explored or mentioned.

-> The prompt used for the experiment is not clear, more details on why 3 words was choose and why is it mentioned multiple times single phrase if possible, but then you mention single phrase and also 3 words. Given that the work depends a lot on the prompt, more prompts should have been experimented and more details of those results are needed.

-> Though there is mention of verbosity detector the actual definition and formula on how you calculate it is missing.

-> There is introduction of algorithm, the paper mentions that VC might causes latency, there can be latency due to algorithm as well.

-> Writing can be improved

-> Most of the examples mentioned in the paper have some numbers in the answers, out of 3 correctly given answers 2 doesn't have any numbers, given that there is existing research that LLMs are not good at the numbers, it would be better to compare the behavior on those lines as well, if the output is wrong because of the numbers and not other factors.

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studied the verbosity issue of LLMs. The authors tested the frequency that LLMs generate verbose answers (a behavior defined as **verbosity compensation** by the authors), and showed the following findings:

(1) All mainstream LLMs generate verbose answers.
(2) Models have different performance when generating concise and verbose answers. Most models perform worse when their response are verbose.
(3) There is a positive correlation between answer verbosity and model uncertainty. This means that models are usually more uncertain about the answers when they generate excessively long answers.
(4) Using verbosity as an indicator to route models (if the smaller model generates a verbose answer, then we give this example to the larger model) is a promising approach to balance model performance and inference cost.

### Strengths
1. The authors are the first to formally define the verbosity compensation (VC) behavior of LLMs which refers to generating an answer that is more verbose than required.
2. The authors collected a benchmark from existing NLP task data to analyze the frequency of VC in LLMs. They further designed metrics to calculate the impact of VC on model performance.
3. The authors did extensive experiments to test the correlation between VC and (1) model performance (2) model uncertainty. These experiments led to the findings I listed in paper summary.
4. The authors proposed a simple but effective model routing technique based on their findings on VC: routing the instance to stronger models if the weaker model exhibits verbosity in answering. They further tested its effectiveness in reducing verbosity and balacing performance-cost trade-off.
5. The paper is well written and the experimental results are clearly presented.

### Weaknesses
1. The fairness of evaluation metrics is questionable. This includes two aspects:

(1) The authors wrote in section 3.1: *We define a response $r$ to be verbose if and only if it can be further compressed with fewer tokens
while keeping the same meaning.* However, in actual experiments, they directly used "the number of tokens in the response" as the verbosity detector. **They did not measure "whether the answer can be compressed while keeping the same meaning"**. This may contain bias, for example, if the gold answer is 3 tokens but the model generates a wrong answer that is 5 tokens, although the answer is wrong, but I don't think the wrong answer is verbose if it cannot be further simplified. However, such a case will be identified as verbose by the authors as their criteria of conciseness is "less than 4 words".

(2) In section 3.2, the authors define $\Delta$ (performance difference) as the model score on cases where its answer is verbose minus the score on cases where its answer is concise. This means, **the two scores are not calculated on the same test instances**. First, I suspect this is not 100% fair, because there may be only a few "verbose instances" or "concise instances" for some LLMs, which will lead to inaccurate estimation. Second, I suspect this metric can be affected by spurious correlations. The authors showed in section 5.3 that generating verbose answers indicate a model's uncertainty. And it is possible that model-uncertain cases are those that are more difficult. Therefore, it is reasonable that models perform worse on such instances.

2. The evaluation set is limited to scenarios where the gold answer is less than 4 words. However, I think such short-form QA is just one of the possible situations where verbosity may occur, and it might not even be the most critical one. Verbosity may be more harmful in open-ended generation cases that are not evaluated in this paper. For example, *repeating the question when answering a factual question* may be less annoying than *saying a lot of useless words when planning a trip*.

3. When evaluating the proposed cascading approach in model routing, the authors only compared it with randomly selecting models. However, they did not compare with more related baselines such as "model routing based on response uncertainty". For example, if the weaker model is uncertain about the response, then give this question to the stronger model.

### Questions
1. The equation in section 3.2 uses the score of cases where $V(x,y,r)=1$ as the minuend, and the score of cases where $V(x,y,r)=0$ as the subtrahend. However, the authors defined the formula as "the average score of the concise responses minus the average score of the verbose responses". Should the minuend and subtrahend be reversed?
2. I have concerns about the definition of verbosity. From Figure 1 and the definition of verbosity detector, it seems like answers like *Paris is the capital of France* will be judged as verbose for the question *Which city is the capital of France?*. However, I think such a response seems pretty natural and human-like, and it does not bring any confusion or inefficiency to the communication. Therefore, I wonder whether the definition of "if the gold answer is less than 4 words, then any answer longer than 4 words is verbose" is too strict and whether it will be widely accepted.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes a concept termed “Verbosity Compensation” (VC), where LLMs respond with overly verbose answers to compensate for uncertainty. VC is shown to decrease response efficiency, increase costs, and obscure clarity, highlighting an urgent need to mitigate this behavior. To address VC, the paper introduces a cascade algorithm that effectively reduces verbosity by replacing verbose responses with alternative model-generated responses, achieving a significant reduction in VC, particularly demonstrated with a 63.81% to 16.16% decrease for the Mistral model on the Qasper dataset.

### Strengths
Interesting research problem.

### Weaknesses
1. In the verbosity compensation and uncertainty section, how to quantity the uncertainty for each split is not clear. How to get the perplexity evaluation metric and eigenvalues is not clear.
2. In my understanding, In performance difference calculation, the recall is calculated by exact matching, checking if the generation contains the exact same string of the ground truth answer. There might be some correct but not exactly matched answer in LLM generation, especially in the lengthy (verbose) response. So would be better to design a partial credit to further validate the findings of this work.
3. The proposed cascade model selection method can leak the ground truth answer. In the selection process, the detector would take ground truth answer as input for answer selection.
4. In section 4.2, what does the |r|>3 mean? 
5. In line 365 and figure 3, what does the llama-3-80b mean? I believe llama-3 doesn’t have the 80B version.

### Questions
please see the weaknesses above

### Soundness
2

### Presentation
3

### Contribution
2
