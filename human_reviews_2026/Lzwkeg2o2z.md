# Beyond Raw Detection Scores: Markov-Informed Calibration for Boosting Machine-Generated Text Detection

- Decision: Accept (Poster)
- Scores: 8, 4, 2, 4

## Abstract
While machine-generated texts (MGTs) offer great convenience, they also pose risks such as disinformation and phishing, highlighting the need for reliable detection. Metric-based methods, which extract statistically distinguishable features of MGTs, are often more practical than complex model-based methods that are prone to overfitting. Given their diverse designs, we first place representative metric-based methods within a unified framework, enabling a clear assessment of their advantages and limitations. Our analysis identifies a core challenge across these methods: the token-level detection score is easily biased by the inherent randomness of the MGTs generation process. To address this,  we theoretically and empirically reveal two relationships of context detection scores that may aid calibration: Neighbor Similarity and Initial Instability. We then propose a Markov-informed score calibration strategy that models these relationships using Markov random fields, and implements it as a lightweight component via a mean-field approximation, allowing our method to be seamlessly integrated into existing detectors. Extensive experiments in various real-world scenarios, such as cross-LLM and paraphrasing attacks, demonstrate significant gains over baselines with negligible computational overhead. The code is available at \url{ https://github.com/tmlr-group/MRF_Calibration}.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors make an important contribution to zero-shot detection of machine-generated text (MGT). Building on two well-substantiated phenomena—(i) adjacent tokens tend to have similar detection scores, and (ii) detection scores exhibit randomness with typically higher values at the beginning of a sequence—the paper proposes a lightweight calibration method for metric-based MGT detectors. The approach is backed by strong theoretical analysis and extensive empirical validation. The exposition is clear, the methodology is sound, and each claim is convincingly supported by evidence.

### Strengths
1. The paper is very well written and logically coherent, with polished and visually appealing figures and tables.
2. The empirical results are strong and substantiate the authors’ theoretical claims.
3. The appendix is exceptionally well organized (I particularly appreciate its structure--the table of contents makes the appendix easy to navigate).

### Weaknesses
This paper, in my view, has no particularly obvious weaknesses, and I am casting a clear accept vote. A few minor improvements could further enhance the paper’s quality, though they are not required:

1. **Analyze failure modes.** In Figure 4, I observe performance drops in certain domains. Could the authors provide an empirical analysis of the underlying causes of these degradations?
2. **Include more attack methods.** The current evaluation considers only paraphrasing, a common attack. Extending the analysis to additional attacks—such as back-translation and synonym substitution—would further strengthen the contribution.

### Questions
1. In multilingual settings, does your theory still hold?
2. Could you include an analysis of **Binoculars**, a SOTA zero-shot detection baseline?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the issue that token-level detection scores in metric-based LLM-generated text detectors are susceptible to randomness, bias, and instability introduced by the LLM sampling process. To mitigate this, the authors propose a lightweight MRF (Markov Random Field) calibration module that models inter-token dependencies and positional weighting. Using Mean-Field Approximation, the method smooths and stabilizes detection scores, thus improving the reliability of existing statistical detection approaches.

### Strengths
- The paper clearly identifies the root cause of score bias in metric-based detectors, namely, randomness introduced by the LLM generation process.
- The proposed method is conceptually simple, computationally efficient, and effective.
- The experiments are broad and thorough, including challenging setups such as **DetectRL**, and report diagnostic metrics such as **TPR@FPR-1%** in addition to AUROC.

### Weaknesses
- Although the proposed method is intuitive and effective, there is concern about the practical significance of its improvement for statistical detection methods. As shown in Table 2, all metric-based detectors still perform poorly on the challenging DetectRL benchmark, far below the level required for real-world applications.
- The evaluated detectors are somewhat outdated. Incorporating more advanced methods such as Binoculars [1] and RepreGuard [2] could better showcase the effectiveness of the proposed approach.
- The proposed MRF module, while effective at mitigating random noise caused by the stochastic sampling process in LLM-generated texts, may not generalize well to adversarial perturbation attacks, such as random character insertion, token shuffling, or deletion. Moreover, I noticed that in the DetectRL setup, the authors only evaluate paraphrasing-based attacks, without considering perturbation-style adversarial attacks. Therefore, it is unclear whether the proposed method is fragile under such conditions. If this is the case, the authors should acknowledge this limitation and discuss potential solutions, which would strengthen the manuscript.
- The paper lacks comparison with other enhanced metric-based approaches, such as TOCSIN [3].

[1] Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text, ICML 2024

[2] RepreGuard: Detecting LLM-Generated Text by Revealing Hidden Representation Patterns, TACL 2025

[3] Zero-Shot Detection of LLM-Generated Text using Token Cohesiveness, EMNLP 2024

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The essential idea behind this paper is a good one. Broadly, what it does is take a token-level score based discriminator between human and machine generated text, where the final decision is based on averaging the token level scores, and replace that final averaging step with a Markov Random Field based approach. Essentially, instead of averaging the token level scores, the upgraded detector studies local patterns in the scores. This method can be bolted on top of many existing detectors.

An attempt is made to justify the effectiveness of this method in terms of mitigating randomness due to the fact that the process of generating machine text is usually stochastic. The authors provide empirical evidence for the effectiveness of their algorithm against a range of baselines.

### Strengths
The basic idea of the approach is sensible and works.

The analysis (Figure 2) of how the differences between the log-likelihood scores of pairs of tokens drawn from the text are correlated with the distance between the tokens is interesting.

Table 2 is very compelling in showing the uplift in performance when replacing 'averaging' scores from a detector with applying the authors Markov Random Field approach.

### Weaknesses
1. I do not rate the discussion of the current state of the art very highly, either in its breadth (there is much state of the art work which is missed, including many detectors which do look at local patterns in things like log-likelihood rather than just raw averages) or its accuracy. For example, there is a claim in the abstract and on page 2 that the weaknesses of token based discriminators between human and machine text stem from the randomness present in sampling algorithms for machine text. This is repeated on page 3: 'However, the detection error based on a single text may be large,because the randomness inherent in the LLM sampling mechanism may cause the MGT to deviatefrom these methods’ underlying assumptions, e.g., Log-Rank assumes that the generated tokenshave high rankings. In contrast, DetectGPT, Fast-DetectGPT, and DNA-GPT incorporate multipleperturbed (i.e., s′) or regenerated (i.e.,˜s andˆs) samples, which mitigates the errors caused byrandomness.' I think this is a fundamental mischaracterization of why DetectGPT and its successors were effective, they do not mitigate the randomness present in language model decoding strategies. Instead, they mitigate a fundamentally different issue, that the expected statistical properties of different types of text are different -  that poetry for example should have a higher inherent uncertainty about the next token than python code. 
2. In the discussion of score calculation (p3) there is a claim that aggregating a score (such as log-probability) over a sequence is problematic because there is randomness in the process of text generating. But surely averaging over a number of tokens is a reasonable way to deal with this randomness? There is an implicit claim being made that, if instead of aggregating log likelihood scores, one were to process the log likelihood scores to the authors algorithm, the effects of randomness in the generation process of machine text would be mitigated. No evidence is presented for this claim and I suspect it is wrong. (I'm not saying the author's text-detection algorithm is ineffective, I'm saying I don't believe the justification for its effectiveness).
3. Theorem 1 gives a theoretical proof of correlations between attention scores. The authors write 'The following
theorem will reveal the relationship between attention scores, which in turn help us understand the
relationship between detection scores of context tokens.' I am unconvinced that Theorem 1 is useful in understanding anything about correlations between log-likelihoods of tokens generated in a sequence, for example. I don't think that Theorem 1 contributes understanding to the current approach.
3. There is already work in the literature which seeks to exploit patterns in statistics such as log-likelihood, as opposed to just raw averages. For example, [Detecting Subtle Differences between Human and Model Languages Using Spectrum of Relative Likelihood](https://aclanthology.org/2024.emnlp-main.564/) (Xu et al., EMNLP 2024)], Xu, Yihuai, et al. "Training-free LLM-generated Text Detection by Mining Token Probability Sequences." The Thirteenth International Conference on Learning Representations. . These are just two examples of which I'm aware, there are probably many more. I am not confident that the authors have done a good literature review and I don't think they have compared their approach to a broad enough set of baselines or the current state of the art (I think all of the comparators are at least eighteen months old, in a fast moving field). What the authors do is upgrade raw average of quantities like log-likelihood to instead look at local patterns in log likelihood, so it is very poor that other papers which have the same overall aim are not included in the baselines.
4. The Markov random field approach is complicated relative to a simpler Markov chain approach. I would have liked to see the two methods compared.
5. I strongly disagree with much of the commentary assessing the current state of the field. For example, the authors write 'Our analysis reveals that they share a threshold-based detection criterion, with only minor differences, such as the inclusion of auxiliary data (e.g., perturbed texts).' I find this claim odd, the idea of DetectGPT which moved us from asking 'is the log-likelihood high' to 'is the log-likelihood higher than other texts conveying the same meaning' was revolutionary both conceptually and in terms of performance. The authors' method also uses a threshold-based detection criterion.

### Questions
Figure 2 is supposed to demonstrate that there is a substantial correlation between the log-likelihood score of a token and the log-likelihood score of the token's nearest neighbour. Certainly the graph shows some effect, but I find it hard to assess the strength of the correlation. Could you give us a clearer visualisation? For example, it would be useful to also include tokens at distance 100, so one could see a stark difference between the distance between the log-likelihood scores of nearest neighbours and distant neighbours. 

On the first line of page 2 you ascribe log-rank to Mitchell et al, which I think is wrong. Probably it should be the same reference that you have for log-likelihood and entropy, although I'm not certain.

I don't think Fast-DetectGPT has a significant compute overhead as stated on page 3, they collect all of the required info from a single forward pass through the text. 

I don't understand the point you are making in the following paragraph. Isn't the same true of your method, that having computed some number for each text you then use a threshold to divide into likely human and likely machine based? 'Detection. These methods all employ threshold-based detection mechanisms, whose effectiveness relies heavily on the accuracy of their calculated scores. As previously discussed, two factors compromise this accuracy: (1) the inherent randomness of LLM-generated text introduces bias
into score calculation, and (2) direct score aggregation fails to mitigate this bias. As a result, their detection performance is often unsatisfactory.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work observes the impacts of randomness (during text generation) on the metric-based LLM-generated text detectors, a specific type of detection that relies on statistical scores computed from token-wise probabilities. To alleviate this challenge, a Markov-informed score calibration method was proposed and a plug-and-play module was designed to improve the detection performance of existing metric-based detectors. Experiments were conducted to verify some claims mentioned in this work.

### Strengths
1. This work identified and analyzed randomness issues during token generation which might influence the detection performance of LLM-generated text detectors, which makes sense and the perspective appears new.

2. A Markov-informed score strategy was formulated and served as a plug-and-play for existing fake text detectors. 

3. Experiments on certain datasets were done to validate some claims proposed in this work.

### Weaknesses
1. Unclear presentation and lack of clarity. The definition and introduction of neighbor similarity is not clear both in the writing part and the illustration part. Particulary in Fig.2 and Fig.3, much important information is missing. E.g., how to compute the detection scores, what kind of distances being used, why there are two variables in the horizontal axis (e.g. position, and log-rank score, which variable do the numerical values 0.0 to 0.8 refer to?) 

2. Though some theorems were provided, the assumptions are quite strong to be useful in practical cases, e.g. Theorem 1 was based on a single-layer transformer model, which makes it difficult to generalize to LLMs, it is suggested to elaborate on the scope and guidance of these theoretical studies.

3. Unclear descriptions of the experimental settings. For the experimental comparisons, it is suggested to clearly present the threat models, being a white-box detection or black-box one, if black-box, what is the proxy model being used etc, the settings should clear to be reproducible. 

4. From Table 2, the advantages of the proposed module does not seem significant, particulary by integrating with FastGPT and DNA-GPT, please give some explainations. Besides, More benchmark datasets and baselines should be included.  For benchmark dataset, it is suggested to include experiments on datasets e.g., PubMedQA, in the black-box setting, and experiments on short-text detection are encouraged. Also, more recent metric-based baselines should be discussed and compared, e.g. ref 1 and ref 2, etc. 

ref1: Xu et al, Training-free LLM-generated Text Detection by Mining Token Probability Sequences, ICLR 2025.

ref2: Wu et al, MoSEs: Uncertainty-Aware AI-Generated Text Detection via Mixture of Stylistics Experts with Conditional Thresholds, EMNLP 2025.

### Questions
Please see Weakness part, eg. the vague description and definition of neighbor similarity, experimental settings etc. 
In addition to the weakness part, another question is, The presentation mostly relies on figures, why not provide precisely the numerical results? (It often seems hard to tell the exact number and comparision results given the figures and bars etc).

### Soundness
2

### Presentation
2

### Contribution
2
