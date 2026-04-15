# Multi-Dimensional Insights: Benchmarking Real-World Personalization in Large Multimodal Models

- Decision: Reject
- Scores: 5, 5, 6

## Abstract
The rapidly developing field of large multimodal models (LMMs) has led to the emergence of diverse models with remarkable capabilities. However, existing benchmarks fail to comprehensively, objectively and accurately evaluate whether LMMs align with the diverse needs of humans in real-world scenarios. To bridge this gap, we propose the Multi-Dimensional Insights (MDI) benchmark, which includes over 500 images covering six common scenarios of human life. Notably, the MDI-Benchmark offers two significant advantages over existing evaluations:
(1) Each image is accompanied by two types of questions: simple questions to assess the model's understanding of the image, and complex questions to evaluate the model's ability to analyze and reason beyond basic content.
(2) Recognizing that people of different age groups have varying needs and perspectives when faced with the same scenario, our benchmark stratifies questions into three age categories: young people, middle-aged people, and older people. This design allows for a detailed assessment of LMMs' capabilities in meeting the preferences and needs of different age groups. With MDI-Benchmark, the strong model like GPT-4o achieve 79\% accuracy on age-related tasks, indicating that existing LMMs still have considerable room for improvement in addressing real-world applications. Looking ahead, we anticipate that the MDI-Benchmark will open new pathways for aligning real-world personalization in LMMs.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper presents the Multi-Dimensional Insights (MDI) Benchmark, developed to address the limitations in existing evaluations of Large Multimodal Models (LMMs) by assessing real-world personalization. This benchmark includes over 500 images across six key life scenarios, each accompanied by a set of questions that vary in complexity and are tailored to different age groups: young, middle-aged, and older adults. The MDI-Benchmark evaluates both basic perceptual capabilities (like object recognition) and more complex skills, such as reasoning and knowledge application. The study finds that while advanced models like GPT-4o demonstrate high performance, there is substantial room for improvement, especially in adapting to diverse age-related needs. The paper's contributions include providing a personalized benchmark and revealing insights into LMMs' real-world applicability, which could guide the development of more personalized AI systems.

### Strengths
The MDI-Benchmark evaluates multimodal models’ personalization across different age groups and life scenarios, filling a gap in current model evaluation. It sets a precedent for multi-dimensional, population-specific model testing. The paper is well-structured, with clear descriptions of evaluation dimensions (scenario, complexity, age group), and the tables and charts are easy to understand. The appendix provides examples, making it easier for readers to grasp the experimental content.

### Weaknesses
•	1. Limited Data Coverage and Sample Representativeness:
Benchmark should be a tool to comprehensively evaluate a certain capability of the model. Even if some shortcomings are mentioned in the limitation of the article, the MDI-Benchmark includes only around 500 images and 1,200 questions, which is relatively limited given the paper’s aim to address real-world personalization across diverse scenarios. This sample size may not capture the full range of complexities and variability in real-world interactions. To enhance representativeness, the paper could expand the dataset by incorporating more images and questions across a broader range of subdomains and scenario-specific contexts. This would make the benchmark more robust in testing model adaptability across nuanced human needs.


•	2. Poor scalability. There is no way to dynamically add some new real-life scenarios. The scenarios are pre-defined. Compared with the existing datasets for evaluating the basic capabilities and reasoning capabilities of LMM, the scale of dataset is too small. If we can analyze how the proposed dataset can improve the performance of the model in real-life scenarios and improve the personalized performance of the model.


•	3. Over-reliance on Multiple-Choice Questions:
Although multiple-choice questions offer a straightforward evaluation metric, they limit the depth of assessment, particularly in evaluating complex reasoning, creativity, and nuanced language understanding in multimodal models. To better assess these dimensions, the paper could diversify its evaluation methods, such as open-ended questions or tasks that require models to generate explanations or make decisions based on scenario analysis. This would provide richer insights into models' real-world reasoning abilities and capacity for personalized responses.


•	4. Limited Scope of Personalization Dimensions:
The benchmark primarily focuses on age as a dimension for evaluating personalization, which may not fully capture the variety of personalized needs in real-world applications. While age is a relevant factor, additional dimensions like cultural background, professional domain, and personal preferences could significantly enhance the depth of analysis. Adding these dimensions, even in a limited capacity, would align better with the goal of creating models that are adaptable to diverse user bases and provide more actionable insights for real-world AI personalization.


•	5. Sample Size Limitation:
Although the study’s sample of 2,500 survey responses offers a substantial base, it may not be sufficient to capture the full diversity and complexity needed for a benchmark intended to represent “real-world personalization.” Social science research often requires larger sample sizes when generalizing findings across highly variable populations, particularly for studies spanning multiple age groups, professions, and cultural backgrounds. Increasing the sample size, especially within each subgroup (age, gender, occupation), could help ensure that the collected data more accurately represents the broader population, strengthening the benchmark’s reliability.


•	6.Potential for Survey Response Bias:
Relying on self-reported survey data introduces the risk of response bias, as participants may interpret or prioritize certain scenarios based on personal experience, cultural background, or social desirability. This can lead to skewed data that may not accurately reflect the actual needs and preferences of the population. To mitigate this bias, the paper could supplement survey data with observational or experimental data, allowing for a more objective assessment of real-world needs. Additionally, employing stratified sampling or weighting techniques to adjust for potential demographic imbalances in survey responses could reduce bias and improve data representativeness.


•	7. Questionnaire Design Limitations:
The process of summarizing responses into selected sub-domains may inadvertently overlook less common, but still important, scenarios. Standardizing questions to reduce interpretation variance and piloting the questionnaire in diverse groups could improve question reliability. Furthermore, expanding the scope of questions to address a wider range of sub-domains would enhance the comprehensiveness of the MDI-Benchmark, ensuring a better reflection of real-world complexities in model testing.

### Questions
Please see weaknesses for details.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes the MDI-Benchmark, a tool designed to evaluate the capability of Large Multimodal Models (LMMs) in addressing human demands of personalization within multi-dimensional scenarios. The proposed benchmark comprises over 500 images and 1.2k corresponding requirements, encompassing six major aspects of human life, with a special treatment on question sampling based on three age groups, as well as question difficulty. With the MDI-Benchmark, 14 existing LMMs are evaluated, revealing their performance preferences in different scenarios.

### Strengths
1. It is good to evaluate the LLMs with multiple dimensions, as one single dimension is insufficient to understand the insides of the models. 
2. The complexity dimension and age dimension, considered in this work, are of course dimensions that should be considered.

### Weaknesses
While it is interesting to see that the capacities are different among different age groups, the observations drawn from the results are not surprising to me, especially for the complexity dimension, since the LMMs have no reason to perform better in case of problems with higher complexity level. Even for the age group dimension, I will also guess the models should generally work better for young people, as they are the main generator of the data for LMMs model training. 

I am not convinced that the bi-level complexity and tri-age group dimensions used in this work are sufficient to support insightful evaluations of LMMs. Actually, it is a good topic to study what’s the complete dimension set of a benchmark for LMMs evaluation. Besides age groups, there are many other dimensions to explore, e.g., the gender, the occupations, the races, the nationalities, the religion, the personality…. Have those dimensions considered in the literature? It is better the authors can discuss why they chose only age as the primary demographic dimension among the above-mentioned options. The authors can also include a literature review on personalization dimensions that have been explored in the existing LMMs evaluation benchmarks.

### Questions
See the weaknesses part please.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a new benchmark named Multi-Dimensional Insights (MDI) to support the comprehensive, objective and accurate evaluation of Large Multimodal Models (LMM). Compared to traditional benchmark, this MDI-benchmark gives each image with two types of questions and consider the factors of human age to provide the more reasonable evaluations for LMM. In addition, this paper also applies SOTA Large models to conduct extensive experiments on this benchmark so that the effectiveness of this benchmark could be demonstrated.

### Strengths
1.	This benchmark provides two novel perspectives (more complex questions, ages) for supporting the comprehensive, objective and accurate evaluation of Large Multimodal Models (LMM)
2.	The experiments demonstrate this benchmark could help the scores to reflect the performance of the SOTA LMM closer to the judgement in the real world.

### Weaknesses
1.	This paper introduces some complex scenarios, but authors do not explain the reason that these scenarios are complex enough to allow the model evaluation results to greatly reflect real world compared to other scenarios.
2.	The criteria (scores) used for evaluation seem too simple. Should other criteria be considered? In addition, in terms of scores, I guess $score_{L2}$ might be more important and should be assigned more weight, because the second tiers might be the more significant contributions in this paper.
3.	Prompt templates seemed still hand-craft, there is no automated design solutions being given.

### Questions
1.	Judging from the images shown by the author, the resolution and properties of these images are different. Does this mean that these factors will not affect the effectiveness of the benchmark?
2.	In this paper, we could find the importance of age factors, do you think other main factors for this similar benchmark?

### Soundness
3

### Presentation
3

### Contribution
3
