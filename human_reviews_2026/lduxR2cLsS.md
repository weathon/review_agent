# Efficient Metric for Distinguishing Memorization from Generalization in Large Language Models

- Decision: Reject
- Scores: 0, 6, 2, 2, 2

## Abstract
This work proposes a computationally inexpensive method to measure memorization of training data in LLMs (Large Language Models) while accounting for generalization. Prior approaches such as counterfactual memorization, have been computationally expensive, and therefore only been studied in limited settings.  However, our new metric, Prior-Aware memorization, does not require training any new models, and can thus be directly applied to existing LLMs trained on large amounts of data. We evaluate our metric on two pre-trained models, Llama and OPT, trained on the Common Crawl and The Pile, respectively. We discover that for the largest models, 55-90% of the sequences that would  be classified as ``memorized'' in earlier models are, in fact, generalizable sequences.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
Counterfactual memorization is proposed as a measure for memorization that can distinguish from generalization. Counterfactual memorization, however, requires retraining the model for each sample, with and without including it in training, resulting in an expensive and often impractical computational requirement.

The authors propose prior-aware memorization, which does not rely on training, yet can simulate memorization that is correlated with counterfactual memorization. The goal is to filter out statistically likely suffix. The authors propose that if a suffix incurs high generation probability even with the model is prompted with *any* randomly sampled text, then the suffix *may* be generated due to generalization, and hence not memorization. 

There are results relating/contrasting with existing implementation of counterfactual memorization. Also, the authors filter out a large amount of sequences, that would be classified as memorization in earlier models to be a case of generalization.

### Strengths
The motivation of improving the computation inefficiency of counterfactual memorization is indeed justified. 


The connection between conditional probability to Bayes formula is interesting: the conditional probability of generating a suffix $Pr(s | p)$ given a prefix is equal to $\Pr(s) * \frac{\Pr(p | s)}{\Pr(p)}$, where the first term denotes statistical frequency of the suffix, and the second term is the relative belief. The rest of the paper is based on using this decomposition to filter out statistically likely suffixes. I like this intuition, however the main results in Section 3 and onwards is flawed as explained below.

### Weaknesses
- Equation 1 motivates to decompose the conditional probability into suffix probability and belief ratio. However, definition 2 (and later analysis) **does not consider** belief ratio: $\frac{\Pr(p | s)}{\Pr(p)} \ne \frac{\Pr(s | p)}{\Pr(s)}$. If the goal is to approximate $\Pr(s | p)$, I do not understand why computing $\frac{\Pr(s | p)}{\Pr(s)}$ is relevant. This is fundamentally **wrong**. I encourage the authors to clarify this in rebuttal -- I am willing to increase my score if I am mistaken.

Other comments.
-  Line 37: John Doe may be statistically common, but it does not mean that asking the LLM the prefix *the murder was committed by*, it would generate John Doe. This does not make sense. Also, if some suffix is statistically common in *training*, it does not necessarily become generalization out of context. Reference Ghosh et al., as cited in the paper, has proposed a better differentiation between memorization and generalization.
- Line 102-104. Better explanation is required on the belief ratio. One should not naively assume $\Pr(p)$ is small relative to $\Pr(p|s)$.
- Line 186: what does the author mean by matching the distribution of $q_i$ and $p_i$? It is not immediately clear.
- Line 194 is not mathematically grounded.  
- References are poorly formatted throughout the paper. There is no parenthesis before and after any citation.

### Questions
Address the weakness.

### Soundness
1

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
3

### Summary
This paper introduces Prior-Aware (PA) Memorization, a computationally efficient metric designed to distinguish between true memorization and statistical generalization in LLMs. State-of-the-art approaches like counterfactual memorization require retraining models with and without specific data points, making them infeasible for modern LLMs. The authors propose a substitute measure that can be computed directly from a pre-trained model’s outputs without retraining. They observe that 55–90% of sequences previously classified as memorized were generalizable. Larger models show a decline in PA-memorized sequences, implying they rely more on generalization. Additionally, PA scores are positively correlated with Counterfactual Memorization, confirming the metric’s validity.

### Strengths
1. The authors propose a practical metric for memorization analysis that can be used with any model without retraining.
2. The derivation of PA-Memorization from Bayes' rule is interesting.
3. The methodological section is well-written.
3. Empirical results suggest that the metric is useful.

### Weaknesses
1. The experimental results need to be polished. The setup is clear but Sections 4.4 and 4.5 are not very clear in their takeaways.
2.  The abstract mentions "..55 − 90% of the sequences that would be classified as “memorized” in earlier models are, in fact, generalizable sequences." The paper doesn't seem to discuss on this aspect in the main paper.
3. More thorough experiments could have been undertaken -- for e.g. how is PA-memorization going to be for rare but sensitive items?
4. Figure visualisation can be improved -- e.g. for Figure 2, suffix length could have been a title or in the caption. Please consider writing the model names in the case as they are -- OPT, Llama (instead of opt, llama).
5. Spelling mistakes -- line 87 : "extrctable' --"extractable" , line 153 : "beleif" -- "belief".

### Questions
Same as Weaknesses above and below
1. What is meant by opt(Ratio) or llama(Ratio) -- Is it the fraction of sequences memorized as per relative belief ratio? It's not mentioned.
2. How can it be inferred that with decrease in PA-memorized sequences, "larger models are increasingly able to reproduce verbatim text by generalizing from common or near-duplicate data, rather than through true memorization" ? An experiment on showing this effect of generalization could have been more convincing.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper examines the problem of separating extractable memorization that arises out of generalization -- those being generally statistically likely -- from those that are memorized because a certain sequence was present in the pretraining corpus. The existing definition of this -- counterfactual memorization -- is highly inefficient to measure because it relies on an ability to retrain models for which a particular sequence was not present in the corpus. This work introduces prior aware (PA) memorization as an efficient approach to correct measurements of memorization for the presence of statistically likely common sequence. They estimate the general statistical frequency of a suffix by measuring its probability under randomly sampled pre-training prefixes. Using their score, they identify that 55-90% of memorized sequences can be attributed to generalization.

### Strengths
This paper explores the important problem of more accurately memorizing the amount that large language models memorize. They identify an important problem facing existing definitions -- statistically common text sequences -- and devise an intuitive approach for adjusting memorization measurements to account for them.

### Weaknesses
(1) I am a bit unsure of the novelty of the problem formulation of this work. For example, in Carlini et.al (2021), there is already an extensive discussion of "trivially" memorized subsequences which are those that are common and generally have high likelihood. In fact, they already propose ways to "baseline" whether a text is statistically common using independent models, zLib entropy, and perturbation such as casing. Thus, I am not sure that the observation/problem statement is particularly novel; nor is it made clear at all why the methodology introduced in this paper is superior to the existing works proposed in Carlini et.al.

(2) As the paper notes, the sequences used to test memorization are primarily randomly sampled from various public sources. As a result it seems somewhat unsurprising that a majority of such sequences are those that are more statistically likely under the model. Thus, I am not sure how easy it is to interpret the claim "We discover that for the largest models, 55-90% of the sequences that would be classified as ``memorized'' in earlier models are, in fact, generalizable sequences."  (as this fraction would depend significantly on the underlying source of sequences used to probe memorization). 

(3) Similarly, I also feel dubious about the claim that "Decreasing proportion of PA memorized sequences. Secondly, Figures 2 and 4a show that as model size increases, the proportion of PA memorized sequences generally decreases. This suggests that larger models are increasingly able to reproduce verbatim text by generalizing from common or near-duplicate data, rather than through true memorization". This may very well be the case for the types of sequences used to probe memorization in this paper. However,  this does not rule out that larger models would also memorize the rarer data at higher rates (which may not be significantly represented in the experimental settings here to be accurately measured).

### Questions
Please respond to the points raised in weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Given an output (or suffix $s$) and input (or prefix $p$) during training of a neural language model (NLM), the authors tackle the problem of identifying whether the output was memorized by the NLM. They do this while trying to differentiate between statistically common (suffixes that are presented a large number of times in the training set) and truly memorized suffixes.
In order to do this, they introduce the prior-aware memorization metric (PAm) based on the prior $\frac{\mathbb P(s|p)}{\mathbb P(s)}$. PAm is briefly compared with another memorization metric, also addressing this dichotomy (counterfactual memorization), in Section 3.3.3. PAm is also compared with the classical extractable memorization (Em), which does not differentiate between the two types of memorization, to show how PAm can solve the main problem.

### Strengths
S1. The problem of addressing the memorization of non-statistically likely responses is well-motivated and an important issue for NLMs.

S2. The motivation for the use of a different metric besides extractable memorization (Em) --- Since it can be large due to $s$ being a statistically likely output rather than a truly memorized sequence--- and counterfactual memorization (Cm) --- Due to its computational intractability --- was well presented. 

S3. The metric presented, prior-aware memorization (PAm), was well motivated by the gaps presented by S1 and S2, as well as by the explanation in Section 2.

### Weaknesses
W1. The rigorousness of the mathematical notation could be vastly improved. For instance, without knowing the domain of the sequences $p\|s$, one has to infer how the conditional probabilities are defined. Furthermore, multiple mathematical objects used throughout the paper were not introduced ($E,\hat =,\overset{\triangle}{=},\dots$) or sometimes used differently [$m$ (lines 148 vs 153) and $n$ (lines 148 vs 150)]. Furthermore, the arguments of the metrics, as introduced in (6) and (7), are also unclear --- Figure 1 seems to utilize a set of sequences ($p\|s$) while Def. 2 and (5) use a single pair $(p,s)$ as arguments. 

W2. The work presents Cm [Zhang 2023], which addressed the problems of Em. PAm was introduced as a computationally cheaper alternative to Cm. However, the main evaluation does not consider this newer and better-suited metric. 

W3. The presentation of the paper could be greatly improved. For instance, the Figures contain unnecessary plots (the ratios of two very clearly observable quantities) and are unordered (Figures 4a and 2 are presented before Figures 3 and 4b). Further, the wording is sometimes confusing (for example, "*We evaluate these metrics on various sizes 2 recent generative models* ($\cdots$)", line 320).

W4. The evaluation of PAm does not provide enough evidence to support the main claim of being an alternative that does not require the retraining of models. This is mainly driven by the lack of a comparison with Cm, but also by other evaluation decisions (See questions Q1, Q3, and Q4).

W5. Some influential metrics addressing the same problems of Cm and Em are missing in the related work and in the experiments (see Wang 2025). Without a proper analysis of related metrics (both theoretically and experimentally), there is not enough novelty in the article.

### Questions
In order to improve the manuscript, I kindly ask the Authors to assess the following questions and suggestions.

Q1. Lines 91-107 introduce the main idea for PAm. While it is true that $\mathbb P(s|p)$ does not differentiate between statistically likely suffixes and believable suffixes for the prefix, I do not see how using $\frac{\mathbb P(p|s)}{\mathbb P(p)} = \frac{\mathbb P(s|p)}{\mathbb P(s)}$ (the prior used for PAm) can differentiate between the two. As $\mathbb P(s|p) = \frac{\mathbb P(s,p)}{\mathbb P(p)}$, an increase on the joint probability $\mathbb P(s,p)$ does not need to be smaller than the increase of the individual probabilities of $p$ and $s$. This will, in turn, make $\frac{\mathbb P(s|p)}{\mathbb P(s)}$ bigger for a statistically likely sequence. This might call into question the validity of the prior (and hence, of PAm) to detect such differences. Since the near-duplicates used in the experiments are very dissimilar semantically (see Table 1 in the article ), an in-depth study of the distribution of PAm in a setting using more similar near-duplicates is needed to support the claim. 

Q2. As mentioned before, Cm is inversely correlated with duplication of the pairs $(p,s)$ (see Section 5 in Zhang 2023). How can the authors explain the increase in Cm in Figure 1? Further, the experiments in Section 3.3.3 are not explained in detail, and conflict with the definition of Cm and PAm given in Def. 2 and (5) --- See W1. Could the authors clarify this?

Q3. Cm is a well known and popular metric to address memorization on NLMs. Yet, it was barely analyzed in this article. Furthermore, PAm is presented as a computationally simpler alternative to Cm. To be able to be presented as an alternative, a proper comparison of its performance with Cm needs to be presented. Without this comparison, the evidence does not support the claim. Could the authors consider an in-depth comparison with this metric?

Q4. Why do the authors center in the use of a threshold value, rather than on the study of the distribution of the memorization metrics vs interesting statistics of the sequences, like in Zhang 2023, Wang 2025, and Jiang 2024? Understanding how the metric behaves for the entire distribution of values, not only the tails, is also relevant (and also eliminates the need to tune the thresholds). 

Q5. Some influential papers also tackling the same problem seem to be missing. More importantly, Wang 2025 also proposes an alternative to Cm that differentiates between generalization and memorization, while being computationally cheaper. Were the authors aware of this article?


While the authors propose an interesting and much simpler alternative to Cm, they do not compare PAm with Cm. Furthermore, a more modern alternative to Cm [Wang 2025] where not included. This alternative tackles the problems of Cm, while being computationally cheaper (the same claims as PAm). Without a comparison between all of them, there is **(a)** not enough evidence to support the main claims and **(b)** not enough novelty. Upon tackling Q1-Q5 above, I will reconsider my evaluation of the manuscript.


**References**:

[Zhang 2023] Zhang, C., Ippolito, D., Lee, K., Jagielski, M., Tramer, F., & Carlini, N. (2023). Counterfactual Memorization in Neural Language Models. Advances in Neural Information Processing Systems, 39321–39362.

[Jiang 2024 ] Jiang, M., Liu, K. Z., Zhong, M., Schaeffer, R., Ouyang, S., Han, J., & Koyejo, S. (2024). Investigating Data Contamination for Pre-training Language Models. https://arxiv.org/abs/2401.06059

[Wang 2025] Wang, X., Antoniades, A., Elazar, Y., Amayuelas, A., Albalak, A., Zhang, K., & Wang, W. Y. (2025). Generalization v.s. Memorization: Tracing Language Models' Capabilities Back to Pretraining Data. The Thirteenth International Conference on Learning Representations.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work proposes a computationally effective metric, i.e., prior-aware memorization, to approximate the counterfactual memorization. The authors have first conducted correlation analysis of the proposed metric and the counterfactual memorization measure on 350 GPT-2 models trained with injected sequences that are either exact duplications or near-deduped. The authors then use the proposed metric to measure memorization on pre-trained Llama and OPT models, which suggests that 55-90% of data that was classified as memorized were generalizable sequences.

### Strengths
The paper aims to solve an interesting problem, i.e., the counterfactual-based definition of memorization is accurate, however, computationally expensive to measure.

### Weaknesses
* The key assumption made in the paper (line 53-54) seems not well justified.
   * The authors argue that the differences between "genuine memorization" and "statistical likelihood" can be measured by "testing whether a sequence retains high probability even when the model is prompted with randomly sampled text from the training data. If a suffix appears with high confidence across many unrelated prefixes, its likelihood arises from generalization". This might be true if a memorized sequence happens to be split between the prefix $t$ and suffix $s$ for testing. However, if $s$ by itself corresponds to the memorized sequence, such as digits of $\pi$, i.e., $s=3.1415926535897932$, will it be counted as sequences with high "statistical likelihood"? $s$ might have low perplexity due to being repetitively optimized in training, especially for long sequences of $s$, where the model learns that reconstruction $s$ only depends on the first few tokens of $s$. This contradicts the counterfactual memorization definition since removing all occurrences of $\pi$ from training data, it is highly unlikely that a model can generalize this sequence from others due to its irregularity.
   * Although this assumption is empirically verified in section 3.3 through a correlation analysis on synthetic training runs, the setup here seems quite different from pre-training setup, where the near-dedup sequences are mostly noisy non-natural language sequences, while in actual pre-trianing data, near-dedup usually come in the form of textual variations and templated texts. Moreover, the comment on line 246-246 is inaccurate "Thus, injecting highly overlapping sequences may not realistically reflect modern training regimes, where such examples are typically removed." In fact, sequences in pre-training corpus can still repeat anywhere from 100 to tens of thousands of times, depending on the dedup algorithm used. These differences in the setup make it hard to tell if the assumption still holds on pre-training data distribution.

* **False claims on the property of SATML Challenge dataset**
   *  The authors claim that the SATML Challenge, **"consists of 1-eidetic sequences (sequences where each p||s is known to occur only once in The Pile)" (L343-L344), which is factually incorrect**. This dataset comes from https://github.com/google-research/lm-extraction-benchmark. The official document clearly states the properties of the sequences as "easy-to-extract", i.e., "there exists a prefix length that causes the model to generate the suffix string exactly" and "well specified", i.e., "given the 50-token prefix, there is only one continuation such that the entire sequence is contained in the training dataset." Neither of these descriptions indicate these sequences are "1-eidetic " or "occur only once in The Pile". Quite the opposite, these sequences actually occur very frequently in training data, which is why they are easy to extract (one could easily verify with tools like infini-gram, for example, the first sequence in the training split occurs about 7084 times in the Pile).
   * This discrepancy might explain the "surprising" findings that the authors have reported on L375-377, where the authors claim that "around 40% of sequences are common in nature". Considering the fact these sequences occur extremely frequently in the Pile, it perhaps not just these sequences are common in nature, but rather, there are just multiple exact copies. Overall, this might weaken the findings on line 21, where authors claim there are more "generalizable sequences" than measures used in previous work.

* No comparison with other memorization measurements that do not require training: For example, [Schwarzschild et al. 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/66453d578afae006252d2ea090e151c9-Paper-Conference.pdf) proposed a compression-based metric that is also motivated by the computation cost of counterfactual memorization. How does the metric proposed in this work compares with theirs in terms of efficiency and accuracy?

* Related work not discussed:
     * Contractual memorization: [Huang et al. 2024](https://aclanthology.org/2024.emnlp-main.598.pdf), [Lesci et al. 2024](https://aclanthology.org/2024.acl-long.834.pdf); both work uses counterfactual memorization to study properties of memorization in LLMs
     * Relationship between memorization and generalization: [Prashanth et al. 2025](https://openreview.net/pdf?id=3E8YNv1HjU)

### Questions
* Memorization and generalization are not well-defined in the paper.
   * On line 21, what does "generalizable sequences" mean here? Generalizable in what sense?
   * On line 53, what does "genuine memorization" mean here? What are the quantitive criteria?
   * On line 402, the authors claim that "larger models are increasingly able to reproduce verbatim text by generalizing from common or near-duplicate data, rather than through true memorization"; What does "true memorization" mean here? The authors cited Liu et al. 2025 as evidence, could the authors kindly point out which observations from Liu et al. 2025 support this argument?

* Could the authors elaborate the experiment setup in section 3.3. In particular, (1) what is the injected sequence length (2) are most near dedup sequences replaced with random tokens, i.e., the injected near-dedup sequences are no longer in natural language?

### Soundness
1

### Presentation
2

### Contribution
2
