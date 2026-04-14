# Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
Pretraining data of large language models composes multiple domains (e.g., web texts, academic papers, codes), whose mixture proportions crucially impact the competence of outcome models. While existing endeavors rely on heuristics or qualitative strategies to tune the proportions, we discover the quantitative predictability of model performance regarding the mixture proportions in function forms, which we refer to as the data mixing laws. Fitting such functions on sample mixtures unveils model performance on unseen mixtures before actual runs, thus guiding the selection of an ideal data mixture. Furthermore, we propose nested use of the scaling laws of training steps, model sizes, and our data mixing laws to predict the performance of large models trained on massive data under various mixtures with only small-scale training.  Experimental results verify that our method effectively optimizes the training mixture of a 1B model trained for 100B tokens in RedPajama, reaching a performance comparable to the one trained for 48% more steps on the default mixture. Extending the application of data mixing laws to continual training accurately predicts the critical mixture proportion that avoids catastrophic forgetting and outlooks the potential for dynamic data schedules.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper extends the scaling law approach to optimize data mixtures for efficient language model pretraining. Specifically, the authors fit a scaling law function (i.e. data mixing laws) between the validation loss and training in-domain sampling ratios. From extensive experiments, the approach can improve language model pretraining efficiency on RedPajama dataset, as well as mitigating the catastrophic forgetting by training on mixture of generic dataset and domain specific data.

### Strengths
1. The paper is well-written, where the methodology and experimental results are clearly demonstrated.
2. The effectiveness from the method is well-justified from extensive experiments on LLM pretraining and continual pretraining.

### Weaknesses
1. **Lack of a detailed analysis on computation overheads**: how many small-scale experiments are required to get an accurate prediction of the scaling law function? If the default baseline train with these extra computation flops, how would the performance comparing with the proposed method?

2. **unfair comparison between some baselines**: the DOREMI [1] and DOGE [2] weights from Fan et al.'s paper are both targeting at minimizing the model's performance on SlimPajama dataset, which has a great domain shift to the target used by the authors (the Pile). I would suggest to use the DOREMI weights from their original paper [1] which also optimized on the Pile and retrain DOGE with target domain as your validation set from the Pile.

3. trivial points: 
    - the footnotes (9,10) are not properly displayed on the paper.

[1] DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining
[2] DoGE: Domain Reweighting with Generalization Estimation

### Questions
1. **Computation costs analysis**: how many small-scale experiments are required to get an accurate prediction of the scaling law function? If the default baseline train with these extra computation flops, how would the performance comparing with the proposed method?
2. **Fair comparison between DOREMI and DOGE**: if apply the doremi weights from original doremi paper and doge weights using the correct validation domain, how the performance of the proposed data mixing law comparing to them?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel framework for optimizing data mixtures in pretraining large language models (LLMs). The authors introduce data mixing laws—predictive functions that estimate model performance based on the proportions of different data domains, allowing for the prediction of model outcomes across unseen data mixtures without extensive re-training.

Key contributions include:

1. Data Mixing Laws: The authors formalize a quantitative relationship between data mixture proportions and model losses, termed "data mixing laws." These functions enable the prediction of model performance with different data mixtures.

2. Pipeline for Scalable Predictions: By leveraging nested scaling laws for training steps, model sizes, and data mixtures, the paper proposes a pipeline that allows the prediction of large-model performance using small-scale experiments.

3. Applications in Continual Training: The framework also provides insights for maintaining performance in continual pretraining by dynamically adjusting mixture proportions, thus preventing catastrophic forgetting.

### Strengths
1. The paper is well-written and easy to understand.

2. The research topic on pre-training data mixture is timely and important for the LLM community.

3. The visualization of the model weights t_{ij} in Figure 5 is interesting because it shows the nuanced interaction between the validation domain and training domain proportions.

### Weaknesses
1. The validness of the underlying domain-independent assumption. Similar to other recent papers that model the data mixture problem using scaling law like [1], there is also an implicit assumption of the proposed method: different domain proportions are independent. In Eq. (1), no feature interaction term like r_i*r_j models the interactions between domain proportions. The loss is modeled by an exponential function over the **linear combination** of the proportions. We would like to understand how each domain affects every other domain, this interaction might be highly complex and not local. I would like to further clarify that the feature/proportions interaction mentioned here is different from those shown in Figure 5.

2. Missing baselines. There does not seem to be a sufficient comparison between the proposed method and other data selection methods, e.g. comparing with other token-level and sample-level automatic data selection methods [2, 3, 4].

3. Missing downstream performances. Only perplexity is evaluated, which is not sufficient to demonstrate the effectiveness of the proposed method.

[1] BiMix: Bivariate Data Mixing Law for Language Model Pretraining

[2] Perplexed by Perplexity: Perplexity-Based Data Pruning With Small Reference Models

[3] DoGE: Domain Reweighting with Generalization Estimation

[4] DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining

### Questions
See the Weaknesses.

I am open to raising my score if the authors address the above Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors carefully describe the steps they took to arrive at so-called data mixing law which is a simple formula that predicts the validation loss (potentially of arbitrary mix of diverse domains) of a model trained on a fixed data mixing ratio. This law has O(n_domain x n_domain + n_domain) parameters. Although there are a few arbitrary decisions here and there (e.g., choosing M4 over M1) and the validation (of the method and law) is limited to just one particular setup (training a language model on a relatively general set of domains), i enjoyed reading the authors’ thought process arriving at the proposed “law”.

### Strengths
I very much enjoyed reading how the authors present their thought process leading to the proposed law. Each step was reasonable, and some of the assumptions the authors put forward made sense (although it is never possible to know whether it’s post hoc or ad hoc by reading the manuscript alone.) Because of clear presentation of the steps, I was reasonably convinced of the utility of the proposed law for figuring out the data mixing ratio, despite its limited empirical validation. Furthermore, it was good to see that the authors showed that this law can be used in conjunction with so-called scaling law to improve large-scale training based on the mixing law from smaller scale training.

### Weaknesses
There are two main weaknesses. First, any such law is by nature empirical and thereby must be validated quite extensively with terms of their predictive power, before they can be established as the law. The experiments in the paper are reasonable, but are conducted in exactly one setup. Although the authors consider continual pretraining as a separate setup, i don’t believe it is separate enough to show that this law generalizes beyond this particular setup. I would’ve even appreciated some synthetic experiments, as long as some diversity was introduced. This is the major weakness, since the authors claim it to be the “law” of data mixing.

The second weakness is whether this particular form is the right form. As the authors pointed out, one can think of this as a particular form of a multilayer perceptron. A natural question is then whether this particular form is the one we should prefer. For instance, why do we want to choose the form out of M1, M2, M3 and M4; can’t we simply build a much simpler MLP and fit the MLP directly? If the goal is to use this so-called law to improve the efficiency of training, why would we care about a particular form of this law? Instead, we would much rather care about the predictive power of any predictor fitted on a large number of training runs.

### Questions
I would appreciate the authors’ responses as well as plans to address two weaknesses I have described above.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
It is important to determine the best mixture proportions over a set of domains (i.e., code, web text, for pre-training). However, naive approaches require manual tuning over the search space, and recent algorithms that learn the proportions are not well-understood. Instead, this paper discovers a log-linear "mixing law" that relates mixture proportions to loss. Experiments are done over 2, 3, 7 domains (Pile, RedPajama) to confirm that this mixing law is accurate. This mixing law provides a way of determining the optimal data mixture in a principled manner: conduct multiple training runs over various mixtures, fit the parameters of the mixing law, and use those to compute the proportions that minimize the desired loss. One drawback of naively using this approach is that this requires many full training runs. The paper thus proposes layering on training-step and model-size scaling laws so that the optimal mixture for some target model size and target number of training steps can be inferred by training on various mixtures for smaller models and fewer number of steps. The paper also discusses various ways this approach can be used: it can be applied when we are given a validation dataset that is disjoint from training domains, and it can be applied in the continual pre-training setting too. Empirical results show that conducting shortened runs on smaller models allows them to produce a mixture over RedPajama that attains low validation loss on the Pile faster than the default mixture.

### Strengths
Originality: 
- The paper departs from other data mixing works by presenting the following novel methodology: let's first focus on learning the parametric form of the relationship between proportions and loss, and then use this to find the proportions that minimize the predicted loss. The idea is inspired by work on scaling laws, but mixing laws are quite different. They must capture different factors, such as cross-domain interactions, and can be multi-dimensional depending on the number of domains.

Quality: 
- Beyond just presenting the mixing laws, which are interesting as a scientific artifact, the paper does a good job of discussing practical considerations (training budget) as well as insights from fitting the laws, such as the sparsity of the interaction matrix.
- The paper explains how mixing laws are broadly applicable, presenting practical settings that are not just limited to minimizing loss on the training domains (which e.g., DoReMi focuses on). In particular, the paper considers when we have a separate validation dataset, and when we wish to continually pre-train a model to learn some new domain without forgetting the original one.

Clarity: 
- Paper was well-structured in starting with results on the simplest mixing law and then gradually moving towards more practical settings.

Significance:
- The paper provides a feasible method of doing data mixing that can be much cheaper than brute-force searching---which is often the convention when training LLMs---and is more predictable than recent algorithmic approaches. It captures the intuition of the practical trial-and-error process that practitioners would make when tuning mixtures, and formalizes it into a rigorous procedure.

### Weaknesses
Clarity: 
- Minor typos: "the implementation details." on L328. Figure 6 has no x-axis. In L409, the second L(S) should be L(N). In L473 "date" should be "data".
- What does resampling mean in L456? The procedure here does not appear to match up with the algorithm since you only mention training 70M models. 
- My main clarity issue is in Algorithm 1. Presented as is, it is very difficult for someone to implement. What are the (x, y)'s you need to store in each loop and fit? What are the exact scaling laws? While L(S) and L(N) are defined in L409, this does not typecheck with the algorithm using only L(N, S, r)s.  

Quality: 
- Evaluation of DoGE and DoReMi: the mixtures taken from their paper are for the "universal generalization" setting; for DoGE these are the learned proportions that aim to minimize the average validation loss across domains, not a disjoint validation loss. For a fair comparison, you would need to use the OOD version of DoGE (their algorithm 2). Furthermore, it is unclear how much compute is used by your method vs DoGE vs DoReMi; the compute should be normalized to make a fair comparison, especially since DoGE and DoReMi require 1 extra and 2 extra training runs while this method has parameters scaling in the number of domains.
- It is unclear if the proposed mixing law holds on other domains. The 2 domain experiment is only done on Pile-CC and Github, and the 3 domain experiment is only done on Pile-CC, Github, and Books3. Is it possible that some property of these domains gives way to the linearity in the mixing law? It would also be interesting to explore if this mixing law held on tasks (i.e. in instruction-tuning). 
- Table 15 in the appendix suggests to me that the mixing laws are not that accurate for the nested approach.

### Questions
1. Can you explain Algorithm 1's nested structure more? It would be great if there was a longer version in the appendix that could stand on its own.
2. Can you comment on the mixing law fitting well in other settings (such as different subsets of RedPajama/Pile, or a new dataset overall)?
3. Is there a better way to evaluate DoGE and DoReMi in Figure 9 than using the mixtures reported in the DoGE paper? What do the DoGE mixtures look like in the OOD setting, which would account for the shift between RedPajama and the Pile?

Overall I think this is a very nice paper. It could benefit from more clarity in describing the algorithm and a fairer evaluation of the method.

### Soundness
3

### Presentation
3

### Contribution
4
