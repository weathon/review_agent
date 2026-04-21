# Embedding-based statistical inference on generative models

- Avg Score: 4.40
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 3, 3, 8

## Abstract
Generative models are capable of producing human-expert level content across a variety of topics and domains. As the impact of generative models grows, it is necessary to develop statistical methods to understand the population of available models. These methods are particularly important in settings where the user may not have access to information related to a model's pre-training data, weights, or other relevant model-level covariates. In this paper we extend recent results on representations of black-box generative models to model-level statistical inference tasks. We demonstrate -- both theoretically and empirically -- that the use of these representations are effective for multiple model-level inference tasks.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper titled "Embedding-Based Statistical Inference on Generative Models" presents methods to leverage embedding-based representations of generative models for statistical inference tasks. These models, widely recognized for their expert-level content generation across domains, can vary significantly based on covariates such as benchmarks or model properties. The paper proposes a method to predict these covariates when direct access is unavailable.

The core contribution is the extension of embedding-based representations, specifically the "data kernel perspective space" (DKPS), into classical statistical inference. This allows users to infer model-level properties by using the embeddings of similar models. The authors demonstrate this through various tasks, such as predicting the presence of sensitive information or a model’s safety.

### Strengths
The paper is grounded in strong theoretical foundations, drawing from recent advancements in embedding-based representations and applying them to real-world inference problems. The extension of prior work on model embeddings to the statistical inference setting is supported by well-reasoned arguments and formal proofs. Empirical evaluations across multiple tasks, such as model safety and sensitivity analysis, reinforce the claims made, with results that demonstrate both the effectiveness and scalability of the proposed method. The experimental design appears sound, and the methodology is reproducible based on the description provided.

### Weaknesses
The paper claims to introduce a novel framework by extending embedding-based representations to infer model-level covariates, but this concept is only a modest extension of existing techniques. Embedding-based methods are already widely applied for clustering, classification, and regression tasks. The application to generative models may appear new, but it is essentially an adaptation of standard techniques rather than a breakthrough innovation.

The experiments are narrow in scope, and the datasets or tasks chosen do not convincingly demonstrate the real-world applicability or generalizability of the proposed methods. While the toy example (predicting if a model will say "yes" to "Was RA Fisher great?") provides an illustration, it lacks complexity and doesn't reflect more challenging, realistic tasks. The experimental design also fails to explore the scalability of the approach or its robustness to different hyperparameters or model sizes.

### Questions
How does your approach scale to larger models or model collections, especially when working with large-scale, real-world generative models like GPT-4 or other multi-billion parameter models? Did you consider any strategies to optimize or manage the computational cost in such settings?

The tasks chosen for your experiments, such as predicting the response to "Was RA Fisher great?" or detecting sensitive information, seem narrow and possibly not reflective of more complex model inference challenges. How do you justify the choice of these tasks, and do you plan to expand the scope of the evaluation in future work?

You provide results on performance improvements using DKPS, but there is a lack of direct comparison with alternative approaches. Could you explain why certain baselines (e.g., traditional statistical inference methods or other embedding-based inference techniques) were not included for comparison, and how you plan to address this in future work?

There is limited discussion on the sensitivity of your method to hyperparameters such as the number of queries (m) or the number of models (n) used in generating the DKPS. How sensitive is your approach to these choices, and what strategies do you recommend for optimizing these parameters in practice?

In the paper, you mention that DKPS can predict various covariates, but it's unclear how well the approach generalizes across different types of covariates (e.g., safety, hallucination rate, bias). Do some covariates work better than others with DKPS, and are there any limitations on the kinds of covariates that can be inferred using your method?

How does your method handle models of varying complexity, especially when combining simpler models with highly complex generative models in the same DKPS? Does the variance in model complexity affect the performance of the DKPS?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The submission proposes an embedding-based approach for statistical inference within generative models. Leveraging the Data Kernel Perspective Spaces framework, the proposed method allows to embed models into shared geometric space based on models' responses to a set of queries. These responses are embedded with an external embedder and are averaged. The distance matrix between the models in the collection is then formed based on the differences in the averaged embedded responses. The model embedding is then obtained by applying a multidimensional scaling on the obtained distance matrix. Thus, MDS provides the d-dimensional vector representations for the collection of models called data kernel perspective space. 

The submission also offers consistency results for the proposed approach, which essentially say that if one increases the number of queries, learning on the observed data kernel perspectives (model embeddings) will be close enough to learning on true unknown perspectives.

To enable analysis and model property inference, the approach requires a careful design of the query set and a set of labelled models which could either be obtained from external sources (scores from benchmarks) or fine-tuning models on the controllable training sets to form a reference model set with labels. By analysing distances and configurations within DKPS, one can infer similarities and differences in model behaviours, even if these models differ in architecture, training data, or parameters. The subsequent empirical study explores the viability of this approach.

### Strengths
The idea of substituting the direct model evaluation on existing benchmarks with an embedding approach and subsequent examination of relations within a collection of models in this space seem novel and more general in terms of applicability and computational efficiency.
This is especially important in scenarios where score function for assessing the model output is not formalised or unavailable (e.g., there is no respective benchmark).

Theoretical guarantees for the consistency of DKPS representations and further empirical validation are provided to ensure results are stable and reliable with increasing amount of query and model set sizes.

The computational efficiency makes DKPS a practical tool for large-scale applications where many generative models need to be evaluated and compared quickly.

In terms of clarity, the paper does a good job describing the approach and providing empirical illustration.
The authors also discuss limitations in terms of design choices for particular pipeline implementations, involving choice of query set formation and distance functions used in MDS.

### Weaknesses
1. Distinction with prior work is not clear to me. After a quick glance at [1], which current submission heavily relies on (DKPS, consistent representations, the empirical illustration with RA Fisher greatness, ablations), it is difficult to distinguish the contributions of the current paper because distinctions are not clearly stated. I would appreciate if authors could specify how their current work departs from the previous research.

2. The toxicity and bias experiments are not entirely clear to me. The model's covariate correlates with the size of the point in Figure 4 (right, top), which demonstrates that DKPS doesn't provide a simple configuration for these particular covariates unlike the previous example in Figure 2. Surely, this is expected, since both bias and toxicity are of a more complex nature. But this also highlights that limitation of the proposed approach. Although, the 1-NN in DKPS space for the particular example (green, blue and red models) shows benefit over other simple regressors, it would be more convincing to see this result at scale. As far as I understood, Figure 4. (right, bottom) is also computed for one unlabelled model and underlines the hardness of the problem.

3. The proposed approach highly depends on the design choices and it is not explored how much effort is required to produces pipelines that are descriptive and practical enough. If this work positions itself as a practical extension of the theoretical framework introduced in prior work, the undeniable contribution would be to provide a convincing practical application. For example, although the existing experiment with toxicity and bias helps understand that green model is closer to less toxic red model than the more toxic blue model, this doesn't answer the question whether it can be less toxic than the red one, which is probably the question practitioners would be interested in. Finding a better scenario to showcase the strength of the proposed approach, would help this submission greatly.

[1] Acharyya, Aranyak, et al. "Consistent estimation of generative model representations in the data kernel perspective space." arXiv preprint arXiv:2409.17308 (2024).

### Questions
1. How this work differs from [1], which seems to introduce the DKPS and provides similar empirical study?
2. Did I understand it correctly, that Figure 4 (right, bottom) consider 1 unlabelled model? I suggest demonstrating this at scale, meaning testing performance of the proposed approach on at least several unlabelled models to find whether the results hold.
3. Is my understanding right that we cannot see the clear dependence in the DKPS spaces with the covariates in Figure 4 right top because the complex nature of the covariates? Might this mean the query sets are not optimal for the covariates to demonstrate variation along one of the 2 top dimensions?
4. On line 112, do you mean $j$th row in $\mu_i$?

Addressing the distinction with prior work, providing more comprehensive empirical study (showing results hold for many models), and demonstrating a more practical setup where DKPS alleviates the need for direct model evaluation on a benchmark would help raise the score for this very promising paper!

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper proposes a framework for performing statistic inference on collections of generative models with a particular focus given empirically to language models. The approach is defined through a "data kernel perspective space" and aims to statistically evaluate decision functions over this space.

### Strengths
The general problem space is of broad interest to the community at large, with quantifying various aspects of ever-growing foundation / generative models. Any effort towards making these models less black-box-like is commendable, especially given how few assumptions this approach makes at the outset.

### Weaknesses
I found the work to be trying to be too general at the cost of messaging and clarity. Throughout the detailing of the methodology, it is unclear what the precise goal is when trying to construct this statistical inference framework for generative models. I understand the general intent to perform inference in the presence of multiple different models, different generations, and different settings; however, in covering so much it is not obvious if anything meaningful can come out of so many degrees of freedom in the setup. The empirical findings are not too convincing to me, as the theory developed does not appear to tie too closely to the results. For example, in Figure 1, could be equivalently achieved with a logistic regression model trained on the same embeddings. Any results concerning actual statistical tests appear to be from other established tests, executed after the initial embedding in this "data kernel perspective space".

Additionally, I feel that in striving to make this approach generic and applicable to many different settings, the text became fairly unreadable and indecipherable to me. The mix of background on the data kernel perspective space mixed with adapting it to a generic statistical inference framework introduced a lot of notation in what I found to be a very terse manner. 

Finally, I have some general concerns about the soundness of the foundations that this work builds upon. In section 1.1, a great deal of deference is made to previous works that this paper builds off of. In particular, the three listed below, with specific emphasis on the first. This paper claims to develop further results in this line of work, it does appear to heavily rely on the theoretical findings of these previous ones. I bring this up because all of these papers are preprints that have come out in the past 1.5 years (two in the past 6 months), with absolutely no citations (except between themselves) and no peer review. For more empirical-leaning work, this wouldn't cause me much concern but I feel for such a theoretically positioned paper, it is important that the foundation being built upon is trustworthy.
1. Acharyya, A., Trosset, M. W., Priebe, C. E., & Helm, H. S. (2024). Consistent estimation of generative model representations in the data kernel perspective space. arXiv preprint arXiv:2409.17308.
2. Helm, H., Duderstadt, B., Park, Y., & Priebe, C. E. (2024). Tracking the perspectives of interacting language models. arXiv preprint arXiv:2406.11938.
3. Duderstadt, B., Helm, H. S., & Priebe, C. E. (2023). Comparing Foundation Models using Data Kernels. arXiv preprint arXiv:2305.05126.

### Questions
No additional questions, please address my concerns above. In general, if I am misunderstanding the parts or all of this paper, please do let me know as I could very well have missed something critical in my reading.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
Authors consider embedding-based representations of generative models and demonstrate that it can be used for several model-level inference tasks (predicting whether or not a model has had access to sensitive information and predicting model safety).

### Strengths
As the landscape of generative large language models evolves, it is important to develop new techniques to study and analyze differences in model behavior. The authors show the potential of embedding-based vector representations for capturing meaningful differences in model behavior in the context of a set of queries.

### Weaknesses
1. The authors note that their work extend theoretical results of the paper "Consistent estimation of generative model representations in the data kernel perspective space" (arxiv, 2024; Aranyak Acharyya, Michael W. Trosset, Carey E. Priebe, and Hayden S. Helm). The literature review in sec 1.1 highlighted the following works as of particular relevance: "Comparing foundation models using data kernels" (arxiv, 2023; Brandon Duderstadt, Hayden S Helm, and Carey E Priebe) and "Tracking the perspectives of interacting language models." (arxiv, 2024; H Helm, B Duderstadt, Y Park, CE Priebe). I am concerned that the referenced works are unpublished preprints that are not cited by other works. This may make it difficult for the reader to assess the significance of the research and trace its continuity.

2. I can't understand the first paragraph of section 2. What does index "i" mean, what does index "r" mean?

3. The second paragraph in section 2 contains exactly two sentences that are not related to each other in any way.

4. Line 102: "is the average difference between the average embedded response between" - it's hard to understand.

5. Lines 189-192. The text is difficult to understand, perhaps it should be rephrased and the multiple repetitive "that" removed. Why is the word "optimal" in quotation marks here? In what sense do you understand optimality here?

6. In general, from the text after theorems 1 and 2, at first glance it looks like nothing at all follows from the theorems ("The result does not provide instructions";  "Nor does it provide insight"; "does not provide guidance"; "it is unclear how"). Perhaps this text should be rephrased and it should be more clearly outlined what follows from these theorems and how you use them further in the work.

7. Line 796: "we appended the name of ten random fruits, e.g., “banana” ... ". I don't understand how fruits came to be here, perhaps the authors should explain this.

Overall, it seems to me that the work is very relevant, but before presenting it at the conference, the text and style of presentation should be significantly revised. Therefore, I cannot recommend this work for acceptance.

### Questions
Please see above.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
Models have a set of attributes, such as the score on a benchmark or an indicator of whether they were fit to sensitive data. These attributes may or may not be known by the user. Sometimes similar models may be used to predict such model level attributes. But how is similarity defined? The authors propose using the data kernel perspective space as a basis for similarity. They demonstrate that similarity in this space is predictive of model level attributes.

### Strengths
Practical as it only needs generated responses and an embedding function.
Theoretically grounded.
Good empirical evaluation using relevant examples.

### Weaknesses
More discussion and evaluation on the choice of embedding function would strengthen the paper. It is not yet evident what makes a good or bad embedding function. For example, is it sufficient to use a random LLM? Could one use off-the-shelf token embeddings? For each experiment, an analysis of result sensitivity to this choice would be of practical interest.

The abstract is currently a bit of a disservice to the paper. For example, while it may be evident to those familiar with the data kernel perspective that access to a target or training model's internals is not needed, when you use terms like "embedding based representations" it gives the impression that such access is needed. I think simplifying the language, emphasizing the problem you are trying to solve (model attribute inference), then illustrating why it is important, and summarizing your approach and its positive features would help.

### Questions
I have no further questions at this point.

### Soundness
4

### Presentation
3

### Contribution
4
