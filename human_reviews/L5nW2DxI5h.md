# Predicting from Strings: Language Model Embeddings for Bayesian Optimization

- Avg Score: 3.75
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 3, 3

## Abstract
Bayesian Optimization is ubiquitous in the field of experimental design and blackbox optimization for improving search efficiency, but has been traditionally restricted to regression models which are only applicable to fixed search spaces and tabular input features. We propose _Embed-then-Regress_, a paradigm for applying in-context regression over string inputs, through the use of string embedding capabilities of pretrained language models. By expressing all inputs as strings, we able to perform general-purpose regression for Bayesian Optimization over different search domains such as traditional and combinatorial optimization, obtaining comparable results to state-of-the-art Gaussian Process-based algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper proposes a method called Embed-then-Regress that uses a pre-trained language model to embed string representations of solution candidates for a given optimization problem followed by estimating the mean and uncertainty of the solution candidates using a transformer-based Neural Process (NP) model that is trained on a set of tasks related to the problem being solved. The estimates from the NP are then used to compute UCB acquisition function values for Bayesian optimization.

### Strengths
1. The paper is written clearly and easy to follow.
2. The problem being addressed is important and a good solution has the potential to be impactful in several fields.

### Weaknesses
1. While interesting, my biggest concern with this work is that none of the ideas presented seem novel.  
    (a) Using language model embeddings for latent-space Bayesian optimization is not new [1, 2]. However, no comparisons have been demonstrated with prior works.  
    (b) Training an in-context regressor using a transformer was done in [3] and then later applied to BO in [4]; this work seems to swap the transformer architecture from [3, 4] (specifically the Riemann distribution mechanism) for an NP. If that is the major contribution, then a detailed comparison should be shown. Also, the authors should cite the original NP papers [5, 6].  
    (c) Meta-learning for BO, again, is not novel [7, 8, 4]. Given that this method requires pre-training on several related tasks, it seems natural that the compared baselines should also be exposed to additional training for a fair comparison, but no such analyses are included.
2. Regarding the "Traditional Optimization" experiments, it is unclear given the results whether using LM embeddings provides any substantial benefit over existing approaches, especially given that stronger baselines [9] have not been compared against. I do, however, see the value in using embeddings for combinatorial problems but, as mentioned in my previous point, comparisons with relevant existing methods [1, 4] should be included to come to an informed conclusion.
3. In the conclusion and future work section, the authors also mention that _"an ambitious and exciting direction is to pretrain a unified in-context regression model over multiple different domains, in order to obtain a “universal” in-context regressor."_ Please see [10, 11].

[1] Kristiadi et al., 2024. A Sober Look at LLMs for Material Discovery: Are They Actually Good for Bayesian Optimization Over Molecules?  
[2] Ranković et al., 2023. BoChemian: Large language model embeddings for Bayesian optimization of chemical reactions.  
[3] Müller et al., 2022. Transformers can do bayesian inference.  
[4] Müller et al., 2023. PFNs4BO: In-Context Learning for Bayesian Optimization.  
[5] Garnelo et al., 2018. Conditional Neural Processes.  
[6] Garnelo et al., 2018. Neural Processes.  
[7] Wang et al., 2024. Pre-trained Gaussian Processes for Bayesian Optimization.  
[8] Fan et al., 2023. HyperBO+: Pre-training a universal prior for Bayesian optimization with hierarchical Gaussian processes.  
[9] Cowen-Rivers et al., 2021. HEBO Pushing The Limits of Sample-Efficient Hyperparameter Optimisation.  
[10] Chen et al., 2022. Towards Learning Universal Hyperparameter Optimizers with Transformers.  
[11] Song et al., 2024. OmniPred: Language Models as Universal Regressors.

### Questions
1. In all experiments, am I correct in understanding that the search space is pre-defined over a fixed set of discrete points for which the embeddings are already computed? If that is not the case and the search is, instead, over the entire embedding space, please clarify how acquisition optimization, which will result in an embedding, provide the final text of the solution, i.e. how do you go from the embedding back to the string?
2. From L219, it appears that each optimization round requires a large number of forward passes. How expensive is the presented procedure compared to the baselines in terms of wall-clock time?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors proposes an icl regressor for black-box optimization with four desirada in mind:
1. Pretrainable
2. Flexible representation of inputs
3. Long-range in-context regression
4. Support diverse objective scales 

The main novelty in the proposed in-context regressor is using a text encoder to represent inputs $x$ as a fixed dimensional vector - by first converting to a json str $s$ and then embedding $s$ using T5 encoder. Another novel proposal is to use a text embedding model to encode metadata associated with the function being optimized as feature in in-context regression. 


On both traditional black-box optimization problems and combinatorial optimization problems, the authors find their pre-trained regressor combined with ucb acquisition is comparable and sometimes better than baselines that are based on GP regressors (Vizier, GP-bandit), as well as other baselines including Regularized Evolution, random, and quasi-random search. Ablation studies show that scaling both the input embedder and the decoder leads to improvements in BBO performance.

### Strengths
1. Method is simple and flexible.
2. Presentation of the method is mostly clear and easy to follow.
3. Discussion of related work and motivation is clear.
4. Conducted comprehensive experiments on synthetic functions and combinatorial optimization problems.

### Weaknesses
1. Lacking discussion of the effect of a fixed dimensional representation of inputs x regardless of how many inputs there are. This could in principle be a bottleneck, but maybe in practice the embedding dimension is large enough that this is not an issue. Some discussion or analysis of how performance changes with the number of inputs to the function being optimized would be helpful.

2. While easy to follow, some details about the method/experiment are missing from the paper. Please see questions.

### Questions
1. Details of the method/experiments:
a) What meta-data is actually used in the experiments?
b) What’s the architecture of the trainable projection used to convert y into an embedding?
c) Are inputs x scaled as well or left unchanged, before converting to json?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The work proposes "Embed-then-regress", a method for using in-context regression to perform Bayesian Optimization (BO). Specifically, the paper uses a pretrained encoder network to map points from the search space (represented as strings) to a fixed-length embedding. Then, an in-context regressor is trained using the past observations as the context set to predict the distribution of the objective value for a new query point. The proposed method was experimentally evaluated on a set of synthetic functions and classic combinatorial optimization problems.

### Strengths
1. The paper addresses a timely problem in machine learning, namely meta-learning or large-scale pretraining for optimization tasks (e.g. black-box, or combinatorial). 
2. The paper is written well, with sufficient details and context to make it easily parsable.
3. The proposed method is evaluated on both traditional optimization and combinatorial optimization tasks.

### Weaknesses
The main weaknesses are novelty, justification for design decisions, and thoroughness/fairness of empirical comparisons:

1. **Novelty:** The authors claim as a contribution that they demonstrate the 'versatility of using string-based in-context regression for Bayesian Optimization'. As far as I know, this has already been shown in prior works ([`R1`, `R2`]). Specifically, these works employed LLM-based optimizers and string-based representations of the search space, and has been shown to improve efficiency for (1) traditional regression, (2) Bayesian optimization, (3) combinatorial optimization (TSP). Can the authors crisply articulate the novelty of their method with respect to these works?
2. **Design decisions:** This relates the motivation for using a pretrained embedding network + an ICL regressor on top. An alternative approach could be to finetune a pretrained language model directly to do ICL with a context set of string-based representations of past observations. To me, there are two advantages of using this alternative approach: (1) ameliorates pretraining effort (especially a large ICL Transformer), (2) avoids the need for pooling techniques to aggregate over sequence embeddings (which can lose information)
3. **Empirical concerns:** 
* **3a) Weak baselines:** The authors mainly compare against GP (in Fig 3) and Regularized Evolution (in Fig 4). These are fairly toy baselines, and not appropriate for a proper comparison of the model's performance. The experiments should, at the *very least*, be compared against more competitive BO (HEBO, BOHB, Deep GP/Deep Kernel Regression [`R3`, `R4`, `R5`]) and combinatorial optimizers (population-based genetic algorithms, or local search). These are not pretrained or meta-learned. Perhaps more appropriate would be to compare against techniques with similar goals of *pretraining/meta-learning* (meta-BO methods e.g. PFNs4BO [`R6`], OptFormer (which the authors already cite) although the latter is geared towards hyperparameter tuning). 

If this work were the first to propose string-based representations + in-context pretraining for optimization, I would be more inclined to agree that the limited baselines are sufficient. However, given the existence of prior works [R1-R2], I feel strongly that the empirical evaluations should be more rigorous and fairer.

* **3b) Computational time:** I would like to see the wall-clock time or some metric of computational complexity. In the appendix, the authors mentioned they used 10k proposal points for each step. If my understanding is correct, that is a large number of forward passes just to acquire one point (even with parallel forward passes). This complexity would be orders of magnitude higher than GP or RS (which would take on the order of a few minutes on a CPU for the entire search).

* **3c) Performance improvements:** The performance improvements are relatively limited. In Fig 3, GP methods match embed-then-regress on around half of the problems. These results are not convincing, considering the use of large-scale training and significantly higher computational complexity.



---

[R1] Yang, C., Wang, X., Lu, Y., Liu, H., Le, Q.V., Zhou, D. and Chen, X., Large language models as optimizers. arXiv 2023. arXiv preprint arXiv:2309.03409.

[R2] Liu, T., Astorga, N., Seedat, N. and van der Schaar, M., 2024. Large language models to enhance bayesian optimization. arXiv preprint arXiv:2402.03921.

[R3] Cowen-Rivers, A.I., Lyu, W., Tutunov, R., Wang, Z., Grosnit, A., Griffiths, R.R., Maraval, A.M., Jianye, H., Wang, J., Peters, J. and Bou-Ammar, H., 2022. Hebo: Pushing the limits of sample-efficient hyper-parameter optimisation. Journal of Artificial Intelligence Research, 74, pp.1269-1349.

[R4] Falkner, S., Klein, A. and Hutter, F., 2017, December. Combining hyperband and bayesian optimization. In NIPS 2017 Bayesian Optimization Workshop (Dec 2017).

[R5] Wilson, A.G., Hu, Z., Salakhutdinov, R. and Xing, E.P., 2016, May. Deep kernel learning. In Artificial intelligence and statistics (pp. 370-378). PMLR.

[R6] Müller, S., Feurer, M., Hollmann, N. and Hutter, F., 2023, July. Pfns4bo: In-context learning for bayesian optimization. In International Conference on Machine Learning (pp. 25444-25470). PMLR.

### Questions
I also have some questions and minor concerns:

1. In L172-179, the authors describe the normalization process for $y$ values, but are input values (for each dimension of the search space) normalized too? This might present an issue if the search spaces between tasks exist on very different scales.
2. How exactly are pretraining tasks created/selected for the results shown in Fig 3 and Fig 4. How many pretraining tasks were included in total, and how many observations sampled from each problem?
3. The hyperparameters for regularized evolution seem inappropriate. What exactly are the hyperparameters used (for genetic operations etc), was the population size of 50 maintained throughout? That is a very small population size.
4. I understand that multiple parallel predictions can be made on a batch of query points simultaneously. But is the embedding process parallelized too?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper proposes _embed-then-regress_ a paradigm to perform in-context regression for bayesian optimization. The premise is that in order to get around the varying sizes and dimensions of problem specifications and inputs, they use an embedding model to convert the string representation of each trial into a fixed dimension vector. They are then able to pass this to the regressor Transformer which has been trained to perform in-context learning, similar to Garg et al. "What can transformers learn in-context? A case study of simple function classes." They then utilize expplore-exploit techniques using black box optimization to show that this technique achieves competitive performance over several tasks (primarily from the [BBOB test suite](https://numbbo.github.io/coco/testsuites/bbob).

### Strengths
1. The paper is extremely well written. The problem definition is clear and they provide sufficient (if not too much) context to the problem and the algorithm.
2. The experiments are fairly extensive covering a suite of tasks.
3. The method seems to get around some fairly problematic issues with scaling bayesian optimization and achieves reasonable performance across a wide range of tasks.

### Weaknesses
1. Despite a thorough description of the problem, some of the description around the final black box optimization is still left out. I may have missed this, but I would have appreciated a clearer definition of what explore-exploit technique they were using.
2. I understand that the problem setup is over black box optimization techniques and therefore optimality gap is the right metric to measure. But this does not seem to be part of their proposal. The contribution revolves around regression to make a prediction on $\hat{y}$. I would like to see the accuracy of their method on just this task compared to strong baselines. The optimization is a downstream task that I expect will be the same regardless of the mechanism of regression.
3. The authors suggest in the motivation that the main problem they are trying to solve is the large/unbounded size of the inputs. I presume this becomes infeasible very quickly, but do simple techniques like 0-padding work at all?
4. I am not sure why the authors insist on projecting $y$ to $\mathcal{R}^d$ and turning the input into a $2d$ vector. Since it is a real number, can't it be left as is and concatenated to get a vector in $d+1$? Or they could simply try the mechanism of Garg et al., and stack the vectors as different tokens and therefore the problem becomes next token prediction with the input $\\{x_i, y_i\\}_{i=1}^{n-1}, \hat{x}$. Did the authors try this method? Does it perform worse?
5. Why use custom attention patterns to make predictions parallel? Doesn't the inference mechanism in Garg et al. with k-v caching produce the same effect of avoiding redundant computations?
6. The authors claim that the inference cost needs to be cheap since zeroth order optimizers need to be called several times. Isn't this a choice though? Higher order optimizers could be used which are far more sample efficient.
7. The baselines in the experiments seem fairly weak. Why are there two random baselines?
8. Has the regressor seen some examples from each family of the test set? How different are the parameter values between the train set and the test set? If they are within the same range, it is possibly that the model is simply interpolating between seen values. This would be easy to check by comparing with a stronger baseline - perhaps even a simple 2-layer MLP?
9. My main concern with the paper is the true finding is that transformers can learn how to solve regression using ICL on varying families of functions and the black box optimization and the embedding are merely applications of this finding. If this is the case, then this has already been shown along with theory in Li, Yingcong, et al. "Dissecting chain-of-thought: Compositionality through in-context filtering and learning."

### Questions
1. I found it very useful to interpret the problem as something like reducing the problem of regression to that of RAG where the generator is the regressor given in-context samples and the corpus is retrieving a compact embedding of the said samples from the problem description strings. Is this somewhat accurate?
2. In 3.1 is $\mathcal{X}$ assumed to be of finite dimension?
3. Embedding of metadata: The authors mention that "we can also embed this data..." but it is not clear what is done in the experiments.
4. The larger encoders leading to better results is indeed quite interesting. Is it possible that some of these datasets may have been part of the T-5 pre-training dataset?

### Soundness
2

### Presentation
3

### Contribution
2
