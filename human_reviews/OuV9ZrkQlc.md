# ImagenHub: Standardizing the evaluation of conditional image generation models

- Avg Score: 6.75
- Decision: Accept (poster)
- Scores: 6, 8, 8, 5

## Abstract
Recently, a myriad of conditional image generation and editing models have been developed to serve different downstream tasks, including text-to-image generation, text-guided image editing, subject-driven image generation, control-guided image generation, etc. However, we observe huge inconsistencies in experimental conditions: datasets, inference, and evaluation metrics -- render fair comparisons difficult.    
This paper proposes ImagenHub, which is a one-stop library to standardize the inference and evaluation of all the conditional image generation models. Firstly, we define seven prominent tasks and curate high-quality evaluation datasets for them. Secondly, we built a unified inference pipeline to ensure fair comparison. Thirdly, we design two human evaluation scores, i.e. Semantic Consistency and Perceptual Quality, along with comprehensive guidelines to evaluate generated images. We train expert raters to evaluate the model outputs based on the proposed metrics. Our human evaluation achieves a high inter-worker agreement of Krippendorff’s alpha on 76\% models with a value higher than 0.4. We comprehensively evaluated a total of around 30 models and observed three key takeaways: (1) the existing models’ performance is generally unsatisfying except for Text-guided Image Generation and Subject-driven Image Generation, with 74\% models achieving an overall score lower than 0.5. (2) we examined the claims from published papers and found 83\% of them hold with a few exceptions. (3) None of the existing automatic metrics has a Spearman's correlation higher than 0.2 except subject-driven image generation. Moving forward, we will continue our efforts to evaluate newly published models and update our leaderboard to keep track of the progress in conditional image generation.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Paper proposes a library for the evaluation of coditional image generation models. They consider 7 tasks (text-guided, subject-driven, control-gruided etc). They define two human evaluation scores (semantic consistency and perceptual quality), train the raters, and evaluate around 30 models for the various tasks.

### Strengths
1. paper is well written and presentation is good. 
2. a fair comparison based on human raters is interesting for many users and scientists in this dense research field. 
3. evaluation setup and comparisons are well-designed and seem fair (many methods use their own curated datasets and are here compared on the same data).

### Weaknesses
1. I think the paper would have been stronger if the authors would have directly compared the human evaluations with existing automatic evaluation metrics. It would be very interesting to know the correlations. 

2. Ideally we would have computable metrics which correlate high with human evaluations. The paper does not explain very well what the main problems are of existing metrics. It would also be interesting to see what parts of human evaluation are missed by currently used metrics.

3. It remains unclear how a third party with a new method could make use of this benchmark since it is based on human raters. It would be nice if there is some rater-training guide which would allow other researchers to also evaluate their method on the proposed benchmark. There are no safeguards for the maintenance of the benchmark by the authors.

4. I found the discoveries and insights not especially surprising. They were often more based on looking at results than referring to the human evaluation rates.

### Questions
I think the study is of interest for many people. However, I found the technical contribution still a bit shallow and the possible usage for future model evaluation unclear. If some results on weakness point 1 could be added to the paper, I would probably be willing to raise my score.

- please address the mentioned weaknesses. 

minor remarks:
- ref to Table 10 in section 4.2 is wrong (should be table 2)
- spellcheck 'Inference: ' text on page 2.

-------------------------------
POST REBUTTAL:
I thank the reviewers for their feedback. I apologize for mising Table 8 in Appendix. And I appreciate the effort done to ensure ongoing usefulness of the proposed benchmark for future users (A7-A9). I have raised my score.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes ImagenHub, a new benchmark for conditional image generation based on human evaluators.
The benchmark evaluates seven conditional image generation tasks. 
During evaluation, human evaluators will follow two major metrics, namely semantic consistency and perceptive quality. 
The formal metric (SC) ensures that the generated image is aligned with the given condition, while the second metric (PQ) ensures that the generated image is of good visual quality.
The authors evaluate major opensource image generation approaches based on the two metrics and report the results.

### Strengths
1. This paper first proposes a comprehensive benchmark to evaluate different conditional image generation tasks. It evaluates a bunch of image generation models with human evaluators, and provides comprehensive evaluation results to the community.

### Weaknesses
1. It would be better if the authors can involve more top-performing image generation approaches in this comparisons (though some of them may not be opensource), like MidJourney or DALLE-3 (in a later revision of the paper). 
2. Some previous works (like T2I CompBench) have also proposed some evaluation metrics for benchmarking conditional image generation models (related to the Semantic Consistency part in this paper). It would be better to discuss and see if such metrics result in similar trends compared to the human-based evaluation.

### Questions
1. How do the authors validate the annotation quality, and the measure the performance of annotators?
2. How much human labor is required to build this benchmark (in annotator x hour)?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents ImagenHub, a dataset and library for standardized evaluation of conditional image generation models. A large amount of models is evaluated using a unified evaluatoin protocol of human raters. Two metrics are proposed to judge semantic consistency and perceptual quality, and the evaluation protocol is adjusted for high inter-worker agreement.

### Strengths
- The paper tackles an important problem, namely inconsistent evaluation protocols of the large amount of recent image generation methods.
- The paper presents a sound approach for fair comparison using human raters.
- The paper contributes a library to standardize and ease the evaluation of future generative models.

### Weaknesses
- The paper states that 83% of the published results are consistent with the ranking, and the presented evaluation results often validate the results of published works.
  - a) Where does this 83% come from?
  - b) What about the other 17%? What kind of results from published work is not consistent with the presented work? Is it due to limitations of the presented paper or wrong claims by published work?

Two minor points:
- It would be beneficial to incorporate more automatic measure such as the commonly used FID or detection based scores to evaluate spatial fidelity, object recognizability as well as counts of objects.
- It could be interesting to see an analysis on the costs and time needed of such a unified evaluation protocol given that the method relies on human raters.

### Questions
- Is it possible to analyze drift of user ratings over time? In other words, how much is the rating influenced by the experience/exposure of a rater to the evaluation platform?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces ImagenHub, a standardized framework for evaluating conditional image generation and editing models, addressing inconsistencies in experimental conditions. It defines key tasks, creates evaluation datasets, establishes a unified inference pipeline, and introduces human evaluation scores. Results indicate that existing models generally perform poorly, except for Text-guided and Subject-driven Image Generation, and validate most claims from published papers while highlighting the inadequacy of existing automatic metrics, with plans to continue evaluating new models and tracking progress in the field.

### Strengths
This is an extensive endeavor! Comparison of several models plus human evaluation is presented. It is an important problem and the this is a timely study. In general, I am leaning towards accepting the paper but there are several issues and questions that need to be addressed. I would like to see the authors responses first.

### Weaknesses
A major contribution is human judgment which has some issues. First, the number of subjects is small, Second, details of how experiments are conducted and information about them is missing. 


Writing can be improved.
Typos here and there:
One of the most popular task —> tasks  [page 1]
We found that evaluation results from the published papers from are generally [page 3]
These methods rely on the statistics on an InceptionNet pre-trained on the ImageNet dataset. [page 4]
A limitation in this work is the reliance on human raters, which is not only expensive and time-consuming. [page 9]

Page 3 “The goal of conditional image generation is to predict an RGB image” —> I think predict is not the right word here

### Questions
Q: Fig 2 -> what does y axis show? No label

Q: Regarding the ImagenHub dataset: it seems like you are using data that is already been used by others. What is some researchers have already used this data to tune their models? Couldn’t you collect an independent new test set?


Q: ImagenHub Inference Library is a great job. How you ensured that the best parameter setting is chosen to generate best results for each model?


Q: What is the last row of Fig 3 showing?!


Q: In Eq. 1, why min is used? Isn’t min too stringent here? Why not mean instead!?

Q: why are there multiple errors bars for each condition? Not clear

Q:  “We assigned 3 raters for each model and computed the SC score, PQ score, and Overall human score”. 3 subjects is really not that many here and this makes the results less reliable.

Q: what is the keyword column in Table 3?

Q: How is the overall column is computed in table 4?

Q: No information about the subjects and biases etc are given.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
