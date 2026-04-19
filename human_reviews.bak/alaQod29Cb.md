# Multilingual Arbitrage: Optimizing Data Pools to Accelerate Multilingual Progress

- Decision: Reject
- Scores: 5, 3, 8, 8

## Abstract
The use of synthetic data has been crucial in achieving recent state-of-the-art breakthroughs. However, relying solely on a single oracle teacher model for data generation can lead to issues such as model collapse and bias propagation. These problems are particularly pronounced in multilingual contexts, where no single teacher model performs optimally across all languages. In this study, we propose a solution through multilingual arbitrage, which exploits performance variations among multiple models for each language. By strategically routing samples through a diverse set of models, each possessing unique strengths in different languages, we address these challenges. Our extensive experiments with state-of-the-art models demonstrate that our arbitrage techniques significantly enhance performance compared to relying on a single teacher model. Our multilingual arbitrage techniques result in large gains of up to 80% win-rates over state-of-art proprietary and widely adopted open weight models such as Gemma 2, Llama 3.1, Mistral v0.3. These gains, achieved through multilingual arbitrage and averaged across all languages, were most substantial in the less-resourced languages within our pool.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper explores strategies to make use of different teacher LLMs for synthetic data generation for fine-tuning student LLMs. In specific, the authors propose to use:
- fixed routing with prior information about the strength of each teacher
- reward-based routing: a reward model selects the best output among all teacher outputs.
- a learned router: it predicts the most suitable teacher based on the above information on the fly.
These methods are then compared with using a single teacher and randomly routing to different teachers.

The methods are tested on open-ended generations measured by LLM-as-a-judge win rates and some more structured tests measured by accuracy. All three proposed methods outperform a single teacher or random routing. Specifically, in open-ended scenarios: reward-based routing > fixed routing > learned routing; and on multilingual benchmarks the results are mixed. The paper then included three types of analyses which are informative.

### Strengths
1. I support the idea of "multilingual arbitrage" (although not necessarily the naming): it is interesting and important to study methods that can use different teacher models to produce synthetic data for different tasks, languages, or even prompts. I think the three methods are also reasonable.
2. The experimentation is well-structured and extensive with informative analysis components.

### Weaknesses
1. Score reporting: In many places, it's misleading to compare/report win rates (%) using relative percentages. In my opinion, it's better for these to be presented as absolute differences. For example, when WR improves from 10% to 20%, the improvement should be 10% instead of 100%.
2. Evaluation: All test sets used for evaluation are "translated multilingual", which is also the nature of the prompts/questions used for fine-tuning the LLMs and the learned router. Even worse, the dolly-200 test and the model training data are translated using the same model NLLB-3.3.B. Perhaps the authors can report some results on the test split of the Aya prompts?
3. Although the reward-based routing uses a different reward model/judge than the evaluation, the judges may still correlate in some way. This makes reward-based data synthesis reward hacking-prone. Another minor thing is that the results seem a bit "expected", because reward-based routing uses more resources than fixed routing, which is also more advantageous (having information about each teacher model) than random/single-teacher routing.

### Questions
1. Perhaps incorrect citations? BLOOMZ (Dac Lai et al., 2023), Dolly-15k dataset (Costa-juss` a et al., 2022).
2. Regarding the selection of teacher and student LLMs:
    - For the "basic set" experiment, is one of the cases using Aya-23-8B as both the teacher and the student in the "single teacher" setting? 
    - In other routing cases, sometimes Aya-23-8B would also be used to synthesize data for its own fine-tuning. This seems weird to me.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper introduces "mulgilingual arbitrage", a novel distillation method from state-of-the-art LLMs to a smaller student model, by using a router to pick the best teacher model for each input training sample. The student model after such training outperforms its untrained version, together with several other open LLMs. The paper also discusses the difference between three routing stratgies in performance and shows the "reward-based routing" performs the best.

### Strengths
1. The method is simple and effective, which can save teacher model inference cost and enhance student model performance.
2. The method can be easily applied to other teacher and student models.

### Weaknesses
1. The idea that we can use multiple teacher models to train a student model is rather trivial and intuitive. As a result, the contribution of this paper is relatively small to the research community.
2. Also, the method, training a router to pick the best teacher for each sample, is quite straightforward, while the optimality of which cannot be guaranteed. Actually, the result that "reward-based routing" (using the score from an open reward model to choose the best from all the teachers' responses) outperforms "learned routing" (using a trained router) shows that the trained router does't always choose the teacher with the highest reward for that sample. Furthermore, the reward model score can also carry bias and error, bringing error accumulation in the process, but no experiment in this paper can tell whether the teacher with the highest reward is the ground-truth-best teacher for that sample (reviewed by human experts, etc.). 
3. The paper doesn't tell whether the "multilingual arbitrage" method is better than simply mixing the responses from multiple teacher models. Though there is a experiment comparing the reward-based routing method with the "single 'oracle' teacher" method (which is merely using the teacher model with the highest average performance), it is not enough.
4. Some of the experiment settings are questionable, for example:
   - Aya-23-8B is used as both the student model and one of the teacher models
   - Part of the Dolly-15k dataset is used in both the router training set and the system testing set
5. The title and abstract of this paper put "multilingual" as a key point. However, throughout the methodology part, no specific design for the multilingual problem is proposed. Also, in the experment setting part, only 3 discriminative tasks are metioned to test the multilingual capacity of the trained student model, which is not satisfactory. In fact, the routing method can be applied to much more generation tasks, as long as there is need to distil open LLMs into a smaller model.

### Questions
1. Why is the router training loss the KL divergence instead of the normal cross-entropy? Since the goal of the router is to choose the single "best" teacher for a given sample, the cross-entropy loss against the highest-reward teacher of a given training sample seems more reasonable.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper looks at ways to leverage multiple teacher models for a student model to improve multilingual performance on a student model (Aya-23). Overall, this is an interesting idea and the experiments justify most of the claims. However, there are a few things that I would have liked to see in the paper.

One thing that wasn’t abundantly clear to me in the paper is that a prompt must only be routed to one expert? Why must this be the case? Is there a fixed budget? I would assume that using more than one model for each prompt would actually improve performance? Probably not all, but at least more than one. Assuming a fixed budget, I think the question then becomes, do we get more by using the same prompt and going through multiple models, or do we do better by getting more prompts and diversity in the data that way. I may be missing something here, but I think that is a better baseline.

However, overall, the idea of strategic sampling seems like a great idea.

Also, I don’t like the name arbitrage. This implies that the same thing is bought and sold (here I guess it means language data). Normally you can think of this as a cyclical graph where you are trying to find negative cycles (like Bellman-Ford algorithm). This is more of an efficiency routing algorithm, so the name doesn’t fit.


I’d also like to see results in an appendix broken down by language. Most of the results are averaged across all languages. What happens when you break it down? For instance, Table 2 is averaged across languages. It would be nice to see this broken down in the appendix. Table 8 A.7 is also averaged across 7 languages. This is already in appendix. It could be expanded.

### Strengths
A really interesting idea about strategic routing.

Multilingual tasks that are often overlooked.

### Weaknesses
Baseline in prompt.

Name "arbitrage".

### Questions
What are individual language scores?

What happens if the student model sees all models for a prompt? Or a subset as of all models?

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
3

### Summary
This work introduces the concept of multilingual arbitrage to generate better synthetic data by using multiple teachers, each of which is specialized in their corresponding language. This reduces the bias propagation and model collapse from using a single teacher model, which is especially prevalent in multilingual contexts. The authors introduce different kind of routing strategies, namely 

1)Fixed-routing with predefined set of expert teachers

2)Reward-based routing in which all the teacher completions are considered and only the top ranked teacher completion is used to train the student model

3)Learned routing to address the disadvantage of the reward based routing where all teacher completions have to be generated. In this strategy, the model learns to choose the best teacher based on the given prompt and uses the completion of that teacher to train the student model

The above strategies have been tested across 15 languages and 9 SOTA multilingual models. The experiments show that reward-based routing technique achieved significant improvement over the best single-teacher model. They also performed ablation against random routing and showed that all three strategies significantly improve the performance against random routing. Scaling up arbitrage setup also improved the win-rate significantly against SOTA multilingual models like Gemma, Llama 3.1, and Mistral v0.3. The authors also assess the generated text's verbosity, readability, and diversity of the student model, finding that multilingual arbitrage improves all these characteristics.

### Strengths
1) The authors introduces a unique "multilingual arbitrage" approach that routes samples to the most suitable model for each language, thereby creating better student models than from a single best teacher

2) The authors have done a comprehensive evaluation of the different types of routing strategies, including testing for textual quality metrics like readability, verbosity, and lexical diversity, contributing insights into how multilingual arbitrage affects text quality

3) Win-rate comparisons across different resourced languages show that medium resourced languages like Turkish and Ukrainian experience a higher gain than high resource languages like English, which helps in addressing a critical gap of improving performance for languages other than high resourced ones

4) The results in Figure 7 indicate that translation is the least effective method for synthetic data generation, as even random routing performs better. This result helps in generating synthetic data which are suited to many contexts like tasks requiring cultural cues or other tasks like XNLI which are translated from English

### Weaknesses
1)The authors mention that the multilingual arbitrage method is created to not rely on a single model which creates bias and model collapse but do not show any experiments on how this is avoided

2)The language families considered is limited, with significant gaps, particularly in Indian and other South Asian languages, which limits the generalizability of the multilingual arbitrage method

3)The authors mainly rely on quantitative metrics with limited qualitative analysis of generated text in terms of cultural accuracy, idiomatic expressions, or semantic consistency, which is crucial in multilingual contexts

### Questions
1)How are the reward models used for the "LLM-as-a-judge" and the reward-based routing strategy different, and what criteria are used to evaluate their effectiveness?

2)Have the authors tested using all the completions  to finetune the student model to verify whether the reward-based routing is truly competitive?

3)Why is only a single English model used for English-only prompts in the translation setting? Would a comparison with multiple English models provide more robust insights?

4)Why is Turkish categorized within the East-Asian language cluster?

5)In the basic set, the teacher and student models are essentially the same size. What is the rationale for performing distillation in this case?

6)The improvement achieved by learned routing is less than that of fixed routing. Are there any insights on why learned routing is less performant?

7)The figures and the corresponding discussion in Figure 5 seem inconsistent. Could the authors clarify these discrepancies?

### Soundness
3

### Presentation
3

### Contribution
2
