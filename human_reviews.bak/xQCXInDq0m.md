# Context Steering: Controllable Personalization at Inference Time

- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
To deliver high-quality, personalized responses, large language models (LLMs) must effectively incorporate context — personal, demographic, and cultural information specific to an end-user. For example, asking the model to explain Newton's second law with the context "I am a toddler'' should produce a response different from when the context is "I am a physics professor''. However, leveraging the context in practice is a nuanced and challenging task, and is often dependent on the specific situation or user base. The model must strike a balance between providing specific, personalized responses and maintaining general applicability. Current solutions, such as prompt-engineering and fine-tuning, require collection of contextually appropriate responses as examples, making them time-consuming and less flexible to use across different contexts. In this work, we introduce Context Steering (CoS) —a simple, training-free decoding approach that amplifies the influence of the context in next token predictions. CoS computes contextual influence by comparing the output probabilities from two LLM forward passes: one that includes the context and one that does not. By linearly scaling the contextual influence, CoS allows practitioners to flexibly control the degree of personalization for different use cases. We show that CoS can be applied to autoregressive LLMs, and demonstrates strong performance in personalized recommendations. Additionally, we show that CoS can function as a Bayesian Generative model to infer and quantify correlations between open-ended texts, broadening its potential applications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces the CoS method for controlling the personalization of LLM generation results during inference. CoS operates by calculating the difference between LLM outputs with and without personalized context, and subsequently incorporating this difference into the original outputs with a weight parameter, lambda, to adjust the level of personalization. A higher lambda corresponds to a greater degree of personalization. The core idea shares similarities with existing counterfactual methods; however, applying it to control personalization is novel. Besides proposing CoS, the paper presents a method for inferring lambda in reverse from a given generation result, aiding in the identification of implicit intents, such as the 'degree of hate in statements.'

### Strengths
S1. Controlling the level of personalization by using the difference between LLM outputs with and without personalized context appears reasonable and straightforward, with the entire process completed at inference time.
S2. The approach of inferring implicit intents from a given generation result is interesting.
S3. A variety of experiments are presented.

### Weaknesses
W1. The experimental evaluation appears insufficiently convincing. It would be beneficial to include more evaluations with objective metrics. For instance, incorporating experiments conducted on established benchmarks for LLM personalization [1] and recommendation [2] would strengthen the analysis.

W2. Some experiments and their results are difficult to follow, such as those related to movie recommendations and hate identification. In the recommendation experiments, it is unclear how the baselines—multi-turn Q&A and in-context learning—are compared under different lambda values. Moreover, the results indicate a higher win rate for these baselines. How do these outcomes demonstrate the proposed method's advantages? For the hate identification experiments, the results are not presented in a clear manner.

W3. The method's effectiveness seems dependent on the LLM’s existing ability to generate personalized responses for a given context. This suggests that the approach amplifies current personalization rather than fundamentally enhancing it. For example, if an LLM's personalization is flawed, the method cannot correct it. This limitation indicates that the approach may not serve as a replacement for traditional tuning-based methods.

W4. The advantages of this method over prompt-based approaches (e.g., the multi-turn Q&A baseline) or in-context learning are not clearly outlined.

W5. Table 2 does not include results for lambda=0. Providing these results would offer a more comprehensive view of the evaluation.

[1] LaMP: When Large Language Models Meet Personalization.
[2] BARS: Towards Open Benchmarking for Recommender Systems.

### Questions
The main concerns have been outlined under Weaknesses. Below are some additional questions:

Q1. When adding the difference to the original model's predictions, how do you ensure that the generated results remain fluent, coherent, and meaningful?

Q2. Could you provide an example illustrating how to compute lambda and the degree of hate using equations (4) and (5)?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Context Steering (CoS), a method to personalize large language model (LLM) outputs at inference time. This is done by providing the user's characteristics and preference as context, and adjusting the influence of provided context using a contextual influence function. The influence of this function on the token probabilities can be adjusted, to control how personalized the output is to the given context. Applications of CoS include personalized recommendations involving topics such as movies, travel, cooking, etc. Besides this, the paper also introduces a Bayesian inference model by inverting the CoS probability model. This is used for classifying and quantifying implicit hate speech. Further applications of this Bayesian model include identifying tones in open-ended statements and online content moderation.

### Strengths
1. CoS is a simple method of personalizing LLM outputs to context, without requiring fine-tuning, or prompt tuning. The method saves on the cost and effort needed for training or prompt-tuning, while being effective in the tests carried out by the authors.
2. The framework can be used directly across many personalization contexts. Fine-tuning or prompt-tuning would require re-tuning for each new context.
3. The experiments show promise, and include human evaluations, GPT4 evaluations, and comparisons with baseline models, across various personalization contexts and implicit hate settings.

### Weaknesses
1. Limited contexts: While CoS is effective for single, straightforward contexts (e.g., "I like {genre}"), user preferences are often more complex, involving various (possibly conflicting) likes and dislikes.  It would be interesting to see the method's performance under more sophisticated and detailed contexts.
2. The baseline experiments in Figure 4 are unclear to me. How are various values of lambda used in the case of in-context learning and multi-turn QA? Also, could the supposedly worse performance of ICL be fixed via prompt tuning?

### Questions
See weaknesses.

### Soundness
3

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
3

### Summary
This paper introduces Context Steering (CoS), a method for controlling the influence of context in Large Language Model generated text. The key idea behind CoS is to quantify the impact of context by comparing the output probabilities of the LLM with and without the given context. This key parameter lambda allows CoS to adjust the level of contextual influence on the generated text.  

The paper demonstrates the effectiveness of CoS in various applications. One application is generating personalized recommendations, where CoS can tailor the LLM's output to specific user preferences. Another application is inferring relationships between open-ended texts, which can be used for tasks like classification and quantification of implied statements.

### Strengths
-The paper is well written and easy to read.

-The proposed approach to achieve personalization is simple, novel, and training-free, applicable to various LLMs. 

-Extensive experiments demonstrate strong performance in personalized recommendations, identification of implicit intents and quantification of extent of “personalization”.  

-The experimental analysis is comprehensive.

### Weaknesses
-Focused primarily on a single context. The paper primarily focuses on scenarios with a single, dominant context. However, real-world situations often involve multiple, potentially conflicting contexts. For example, in the movie case, the user might be interested in comedy movies, science fiction but also movies with great storytelling. 

-Limited Discussion on Computational Complexity: While the authors mention that CoS requires twice the amount of compute compared to a vanilla forward pass, they do not provide a detailed analysis of its computational complexity. A more in-depth analysis of how the computational cost scales with input length, context size, and lambda values would be beneficial.  

-Limited discussion on the impact of CoS on other tasks such as reasoning and creativity.

### Questions
-How can CoS be extended to handle multiple contexts with varying levels of influence? How would the method resolve potential conflicts between different contexts?

-Can you provide a more comprehensive analysis of the computational complexity of CoS? How does the computational cost vary with different parameters and input characteristics (input length, context size, and lambda values)? 

-How does the CoS approach affect the LLM's ability for other tasks, e.g., reasoning and creativity? It might be worth some discussion here.

-The appendix seems missing from the manuscript. Is it accidentally omitted?

### Soundness
4

### Presentation
4

### Contribution
3
