# Assessing Large Language Models in Updating Their Forecasts with New Information

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 2

## Abstract
Prior work has largely treated future event prediction as a static task, failing to consider how forecasts and the confidence in them should evolve as new evidence emerges. To address this gap, we introduce EvolveCast, a framework for evaluating whether large language models appropriately revise their predictions in response to new information. In particular, EvolveCast assesses whether LLMs adjust their forecasts when presented with information released after their training cutoff. We use human forecasters as a comparative reference to analyze prediction shifts and confidence calibration under updated contexts. While LLMs demonstrate some responsiveness to new information, their updates are often inconsistent or overly conservative, revealing limitations in their ability to incorporate it in their predictions. Across settings, models tend to express conservative bias, underscoring the need for more robust approaches to belief updating

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes Evolvecast, a dataset to measure how well LLMs update their forecasts given new information. Evolvecast contains questions from Metaculus after October 2023 (which is before the release date of the R1 distilled models tested, i.e. January 2025). New information is obtained from news articles within a week of when a new comment is posted on the metaculus question. The original forecasting question is queried on the Google Search API, and the top article is selected using Sentence embeddings.

### Strengths
The paper sets out to study an interesting problem: how well can language models update their forecasts based on new information. The methodology of creating question-news pairs is original, and the capability evaluated is important, measuring in-context learning and belief updates on forecasting tasks where new evidence can ideally reduce uncertainty, and require bayesian reasoning. The paper includes some interesting ablations on directional and magnitude based measurements of confidence updates.

### Weaknesses
1.  The methodology assumes that human forecasting updates are optimal. However, accurate forecasting (and also belief updates) are extremely hard for humans. I don't think such an assumption can be made without validation on how good the human reference really is (which I recognize is difficult to validate). It is plausible that frontier models can become more optimal than humans. The paper cites the "strong empirical track record" of metaculus, however the cited source only shows metaculus beats naive baselines rather than it being an oracle. Further, it is not about the forecast update track record of metaculus, but rather the final outcomes.

2. The language models are only shown one article, obtained using a weak embedding model on top of the top 100 google search results when queried directly with the question. This places them at a significant disadvantage compared to humans who have access to far more, and higher fidelity information when updating forecasts. Further, weak language models (1.5 and 7B R1 distills) are used for evaluation, while the claims are made about "LLMs" in general, like "LLMs are far fro human-like in belief updating". I recommend benchmarking frontier language models (e.g. GPT 5 High, Grok 4, DeepResearch models) to make any claims about LLM capabilities, though I suspect the evaluation might start running into issues about the reference metric being weak (as pointed out in 1).

3. There is likely considerable leakage given that the models tested were released in 2025, and the google search api is used. The only mitigation I could find is prompting the model to say "you do not have access to updates after T", but there is no evidence shown that this is enough to avoid leakage. See [1] for a reference on issues with such backtests, and please include a discussion of how these issues were handled in future versions. It is claimed without evidence that comparing forecasting updates between two timestamps "largely cancels effects" of static knowledge in pretraining. The "new information" could still interact with the "pretraining knowledge" in non-trivial ways, for eg, models might update differently when "new information" contradicts or reinforces pretraining knowledge, and thus pretraining knowledge cannot be ignored in this setup without convincing evidence to justify this.

4. The related work misses a large amount of literature on forecasting, in-context learning, probabilistic / bayesian updates and confidence reporting in LLMs etc.

[1] Pitfalls in Evaluating Language Model Forecasters. Daniel Paleka, Shashwat Goel, Jonas Geiping, Florian Tramèr. May 2025.

### Questions
1. Why do you say forecasting is considered as a probabilstic task "more recently". It has long been studied as a probabilistic task, both in humans and language models [1, 2]

2. Do you think metaculus-based human forecasting updates are an ideal reference that the best AI must match?

[1] Expert political judgment: How good is it? How can we know? PE Tetlock 2017.

[2] Forecasting Future World Events with Neural Networks
Andy Zou, Tristan Xiao, Ryan Jia, Joe Kwon, Mantas Mazeika, Richard Li, Dawn Song, Jacob Steinhardt, Owain Evans, Dan Hendrycks 2022.

### Soundness
1

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
This paper introduces a new evaluation framework to assess whether Large Language Models adjust forecasts with new information after their training cutoff. The benchmark evaluates the direction and magnitude of belief updates, along with confidence calibration, and compares models against human forecasters. Experiments show that LLMs exhibit some responsiveness, but their updates are often overly conservative and poorly calibrated.

### Strengths
- This paper proposes a novel approach for evaluating whether LLMs meaningfully update forecasts based on post-training information, moving beyond point estimation performance.
- Well-presented empirical results demonstrating clear gaps and insights into model weaknesses.
- Uses human forecasters as a reference, making evaluation more aligned with real forecasting behavior.

### Weaknesses
1. News updates are retrieved via Google Search API and selected by similarity matching, but the paper lacks ablations on retrieval quality or its effect on downstream belief updates. Errors in retrieval may be misattributed as reasoning failures, reducing the interpretability of results.
2. The literature review on forecasting with evolving external information appears incomplete. The paper would benefit from citing and discussing prior work where news or textual signals are directly incorporated into forecasting models (e.g., SFT-based approaches that inject news context for prediction): 
   - Li, Shuqi, et al. "CausalStock: Deep end-to-end causal discovery for news-driven multi-stock movement prediction." Advances in Neural Information Processing Systems 37 (2024): 47432-47454.
   - Wang, Xinlei, et al. "From news to forecast: Integrating event analysis in llm-based time series forecasting with reflection." Advances in Neural Information Processing Systems 37 (2024): 58118-58153.
3. Experiments are limited to DeepSeek-R1 variants. More diverse model systems would strengthen the generality of conclusions. 
4. Evaluation of unresolved and never-occurring events remains unclear. Forecasting requires testing on future or unseen events to avoid training-time data leakage. The evaluation must exclude cases in which outcomes are already implicitly encoded in the model's pretraining data. Moreover, for events that are unresolved or never occur, current accuracy metrics may unfairly reward conservative (no-update) predictions, thus failing to distinguish between genuine belief calibration and low-information hedging.

### Questions
- How robust is the framework to retrieval quality? For example, top-k retrieval vs top-1, noise injection, or varied temporal windows. Can we disentangle reasoning failures from retrieval failures? 
- Can the benchmark capture structural belief shifts beyond probability nudging? For instance, recognizing when causal assumptions break due to disruptive news. 
- How do the authors ensure that events used for evaluation are not already partially observed or included during pretraining, avoiding unintended data leakage?
- Could the authors clarify how the reliability of human forecasters is ensured when using Metaculus aggregates as reference signals? For example, are contributors weighted by historical skill or expertise, and how is bias or low-quality input controlled?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces EvolveCast to evaluate how well LLMs update their belief in face of new information. The results find that LLMs are inconsistent their updates, and, get worse when provided with more news updates.

### Strengths
1. I think the problem is interesting to evaluate reasoning abilities in face of new data points. 
2. To isolate the effect of new information paper uses a simple mechanism loosely based on DiD style difference in movements for LLMs ad experts. This is a simple and useful mechanism.
3. Experiments provide coverage over several settings to give insights into LLM failures.

### Weaknesses
1.  The paper claims to propose first framework to assess dynamic belief updates. However, the related work cited Karger et al. also tested for dynamic updates as more new information surfaced over time. The method presented here is different; but it doen;t make it first framework. 

2. I found the counter-intuitive result of accumulated news results in performance degradation most interesting. However, these are based on family of deepseek model. How generalizable the observations are? Are these due to model family idiosyncrasies? More recent models such GPT 4o or 5, may not show this behavior. Need more experiments to substantiate this generalization.

3. It is hard to interpret the results about models’ conservative revision of prediction. This is because it is compared against human. Human group could be impulsive with the new information. While conservative update, may show maturity of LLM. From that perspective, it seems the results are about how well LLMs match group of human.

4. On a related note, the ground truth relevanc of human predictions haven’t been discussed. Essentially, the paper is measuring how well LLMs conform to human forecasters.

5. I think using general population of forecaster for cosntructing ground truth is useful at the same time may high high variance. How does a select group (e.g. experts ot to 1% of forecaster) differ from the general population?

6 .The belief update happens via news article that best matches user comments. It is not clear how useful the matched news are? How does model compare when prompted with random contemporaneous news?

7. How do we set the values for \epsilon? In some experiments it is st at 0 and while for other is non-zero. How does evolvecast fix \epsilon?.

### Questions
Noted above in Weaknesses

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
3

### Summary
This paper addresses a gap in Large Language Model (LLM) evaluation, arguing that current benchmarks for future-event prediction are static. They fail to assess a critical real-world capability: how models dynamically update their forecasts when new information becomes available.

To solve this, the authors introduce EVOLVECAST, a new framework and dataset for evaluating the belief-updating capabilities of LLMs.

While the paper has some good ideas, due to the number and nature of weaknesses identified during review I cannot recommend it for acceptance at this time. If the authors address my concerns, I would be willing to raise my score.

### Strengths
- The idea of measuring the reasonableness of forecasting updates by comparing the directional update of an AI's forecast against the human crowd is a good idea. It's a clever use of existing data that has gone overlooked.

### Weaknesses
- The core idea of this paper is that we can evaluate intermediate forecasts of forecasting systems using the crowd prediction as a ground truth. I think this core idea is flawed for two reasons: (1) the human ground-truth here is not necessarily something we *want* AIs to match, and (2) we can do some of what this paper is trying to do with the ground-truth resolutions anyways.
    - For (1), consider that humans often make errors in judgment, and prediction markets have various inefficiencies. We don't want to bake these into LLMs. Another way of seeing this is from an RL lens. In environments where verifiable rewards are easy enough to obtain, it is often counterproductive to treat human demonstrations as an optimal policy. They can make for great pretraining data, but there is a reason AlphaGo wasn't trained on human demonstrations alone.
    - For (2), take the confidence calibration measure in this paper. It isn't defined as the Brier score computed against the ground-truth resolution, but rather the Brier score computed against the crowd prediction. This is highly nonstandard and doesn't make much sense when we can just compute Brier score against the ground-truth resolution instead. One could argue that Brier score against the ground-truth resolution is bad because it conflates accuracy and calibration, and accuracy increases in many cases as the resolution date approaches. I would agree with this criticism; I always thought Brier score was a bad metric anyways. A much better metric would be to just measure calibration error like everyone does outside of the forecasting community. I would strongly encourage the authors to use this instead of Brier score, and to switch to measuring against the ground-truth resolution.
    - All of that said, I think the magnitude evaluation introduced in this paper is meaningful. Looser evaluations that are basically sanity checks on whether forecasts update in a reasonable direction on new information are useful.
- For the black-box verbalized confidence, why not just ask the LLMs for a probability and perform some post-hoc Platt scaling or isotonic regression to normalize it? In my opinion, that would make a lot more sense than a Likert scale.
- The models being evaluated are too small. Please use some frontier models, even if that means you can't use white-box logits. Otherwise the conclusion in line 311 of "LLMs are far from human-like in belief updating, regardless of whether confidence is elicited in black-box or white-box form" falls flat.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
