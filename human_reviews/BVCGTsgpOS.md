# FactTest: Factuality Testing in Large Language Models with Statistical Guarantees

- Decision: Reject
- Scores: 8, 3, 3

## Abstract
The propensity of Large Language Models (LLMs) to generate hallucinations and non-factual content undermines their reliability in high-stakes domains, where rigorous control over Type I errors (the conditional probability of incorrectly classifying hallucinations as truthful content) is essential. Despite its importance, formal verification of LLM factuality with such guarantees remains largely unexplored.
In this paper, we introduce FactTest, a novel framework that statistically assesses whether an LLM can confidently provide correct answers to given questions with high-probability correctness guarantees. We formulate factuality testing as hypothesis testing problem to enforce an upper bound of Type I errors at user-specified significance levels. Notably, we prove that our framework also ensures strong Type II error control under mild conditions and can be extended to maintain its effectiveness when covariate shifts exist. Our approach is distribution-free and works for any number of human-annotated samples. It is model-agnostic and applies to any black-box or white-box LM. Extensive experiments on question-answering (QA) and multiple-choice benchmarks demonstrate that FactTest effectively detects hallucinations and improves the model's ability to abstain from answering unknown questions, leading to an over 40% accuracy improvement.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper offers a hypothesis testing framework to control the Type I error, while also showing that the Type II remains sufficiently low. The results hold for general binary classification tasks, and they are related to standard conformal prediction and calibration results. The authors frame their work in the context of LLM hallucination detection, and employ relevant scoring functions from the literature for this task. While the theory holds in distribution, they also provide an extension to out-of-distribution via density ratio estimation.

The methodology is justified by theoretical results. In practice, the method seems relatively simple to use, and it appears to provide notable improvements over baselines.

### Strengths
The paper is well written and theoretically justified. Experiments seem to support the theoretical claims.

### Weaknesses
In the conformal prediction literature, density ratio estimation is a fairly standard way to extend in-distribution guarantees to out-of-distribution. Yet, these estimates tend to be unreliable. Would be interested for the authors to comment on the robustness of their method with respect to the density ratio estimates.

### Questions
- Is there a typo in the legend of Figure 1? The ve10 and ve15 results are repeated twice.
- I feel the authors could bring a stronger connection with the calibration and conformal prediction literature. The UQ of LLMs paragraph in the work related section may mention previous work on calibration of confidence scores in LMMs.
- The threshold selection approach used in the paper essentially corresponds to a conformalized quantile regression. This helps achieving a marginal (on average over all the data) guarantee on the Type I error. However, as the error may not be homogeneous over different questions, one may wonder if ensuring a marginal guarantee is, in fact, sufficient. I wonder if the authors have experienced different error magnitudes across different segments of the datasets.

### Soundness
3

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
This paper proposes a novel statistical method that detects factuality of large language models (LLMs) with statistical guarantees, i.e. hallucination detection with guarantees. The main method assumes the iid assumption, and controls the type 1 error by thresholding uncertainty of LLM’s answers. This is further extended to hold in covariate shift by using rejection sampling. The efficacy of methods is validated over question-answering and multiple-choice tasks with white-box and black-box models.

### Strengths
I like this paper as mainly this tackles an important and timely problem. 

* This paper attacks an important problem, combatting hallucination in LLMs with guarantees.
* The paper is well-written and easy to follow. 
* This paper has Fairly extensive experiments.

### Weaknesses
This paper is quite interesting, but I am mainly leaning to rejection due to the novelty of this paper – this paper completely ignores existing papers in conformal prediction and selective prediction, which are popular methods for building trustworthy AI models in general. 

* (1) and (3) are equivalent to PAC-style conformal prediction (See proposition 2b in https://arxiv.org/abs/1209.2673 or other related papers). What is the novelty of the proposed method with respect to the PAC-style conformal prediction?
* In language tasks, obtaining the correctness of answers is the most important issue. Otherwise, we can apply traditional techniques for LLMs (e.g., conformal prediction). Conformal language modeling (https://arxiv.org/abs/2306.10193) proposes to extend conformal prediction for LLMs and this method can be used as a detector by checking whether a generated answer is included in a conformal set. What’s the novelty of the proposed method with respect to this conformal language modeling? How can you obtain the indicator variable y_i in Section 2.2?
* To my understanding, (4) is incorrect. This is barely an application of a binomial tail bound under the iid assumption, but as written in the paper it is not iid, i.e., the outer probability is taken over D but the inner probability is taken over a filtered distribution P_0. Can you justify the correctness of (4) if I miss something?
* If this paper’s method is equivalent to PAC conformal prediction, extension to covariate shift via rejection sampling is not novel. In particular, https://arxiv.org/abs/2106.09848 extends the PAC conformal prediction under covariate shift using the same rejection sampling techniques. What’s the novel part of the proposed method compared to this existing work?
* In experiments, an important baseline is missing. Can you compare your method and PAC conformal prediction in experiments for both no shift and covariate shift cases?

### Questions
Previously mentioned Weaknesses end with questions. Please refer to those questions.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper proposes a framework, FactTest to reduce hallucinations in LLM responses. FactTest uses Neyman-Pearson methods to make an uncertainty classifier and prevents LLMs from answering responses for which they are uncertain to reduce hallucination. The empirical analysis shows the method's constrained Type-1 errors and accuracy.

### Strengths
1. The paper applies the Neyman-Pearson method to construct an uncertainty classifier with guarantees on constrained Type-1 error. 
2. The work suggests a method to remove the typical iid assumption when designing Neyman-Pearson classifiers to cater to practical scenarios. 
3. The experiments show applicability of the method to black-box API models as well, thus enhancing its practical value.

### Weaknesses
I have the following major criticisms for the paper. With these, I think that the paper is not ready for acceptance. However, if the authors can address my concerns appropriately, I can consider increasing my score.
1. My main concern is the assumption of the equivalence of the notions of certainty of the model and its correctness. The method builds entirely on the premise that if the model is certain then it is going to be correct. The wording of the paper conveys this too, where the words "certain" and "correct" are often used interchangeably. Lines 97-104 mix up the notions of correctness and certainty. As the paper itself mentions in line 34 that models can generate incorrect responses with high confidence, such an equivalence of the notions of correctness and certainty is incorrect till certainty is formally defined differently from prior works, which is not done in the paper. I am especially confused by line 102, which says that when the null hypothesis is rejected, i.e., the model can answer q certainly, then M(q) aligns with a. There is no justification given for the same till that point of the paper. Same confusion is created in lines 124-125 where first $y_i$ indicates uncertainty of responses and then correctness in the following equation.
2. There is a mismatch in the definition of hallucination in the Introduction. Line 34 mentions hallucination as the models generating incorrect responses with high confidence and line 42 says that hallucination occurs when model is uncertain. The authors should consistently define the property.
3. I am doubtful about the theoretical generalizability of the uncertainty predictor $\hat{f}_\alpha$, which is constructed on the samples from $\mathcal{D}$ to samples outside of $\mathcal{D}$, to be useful as a general uncertainty calibrator. I believe that the authors should thoroughly study this aspect. Does the sample space of $P$ also contain elements outside of $\mathcal{D}$?
4. "We prove that our statistical framework achieves strong power control under mild conditions, ensuring that the predictor can also maintain a low Type II error." I don't see how this is a contribution from the main paper. The result of using a thresholding based classifier that has controlled Type 2 error appears to follow from Tong (2013). As the authors claim this contribution, they should provide at least a proof sketch for Theorem 1.
5. There are several instances of using terms before definition, some of which I enumerate below. 
    1. Lines 50-54 mention terms like Type-1 error, Type-2 error, and false positive rate, before clearly state the null hypothesis.
    2. [Line 62] The term "human-annotated samples" is used before definition/context. 
    3. $\epsilon_{\eta}$ is used in line 170 before definition.
    4. The paper does not clearly state the *mild conditions* (phrase used several times in the paper) under the Type-2 error is controlled.
    5. What is meant by the phrase "aligns with the correct answer"?
    6. What are $\mathcal{Q}$ and $\mathcal{A}$? They are used before definition in line 105.
    7. Line 269: How is the "probability distribution over distinct meanings" defined?
    8. What is FactTest-t that comes up Section 4.3, without any prior definition?
6. *Definition of M(q)*
    1. I sense ambiguity in the statement in line 116, where the authors mention that they consider the effects of the distribution of $M(q)$ as well in the probability term in equation 1. $M(q)$ is a certain realization from the distribution of responses, which is not explicitly captured in the expression. I would encourage the authors to explain this point more explicitly. 
    2. All through sections 2 and 3, the framework used M(q) as a single generation for a given question q. However, in the experiments, in equation 6, M(q) appears to be a list of responses. I would encourage the authors to be consistent in their notations.
8. Major typos:
    1. I think that in line 167, it should be "ability that 𝑀 answers the question *𝑞* (not q') certainly given any question 𝑞′ and the generated answer 𝑀(𝑞′)".
    2. It looks like the legends of Figure 1 have typos in them, as the plot names are repeated.
    3. Line 465: Shouldn't it be FactTestO-SE instead of FactTest-SE?
11. Line 223: The authors should provide the proof for $\tilde{\mathcal{D}}_0\mid\mathcal{I}\sim P_0$ and for iid. Moreover, what is meant by the notation: $\tilde{\mathcal{D}}_0\mid\mathcal{I}$, specifically, the conditioning? 
14. Experiments:
    1. The evaluation is just on Llama models and GPT-4o-mini. There are several other small open-source and closed-source models that must be evaluated to fully understand the efficacy of the method. Examples are Mistral, Gemini, Claude, GPT-4o, Phi, etc. 
    2. I think the evaluations should also report the % of willingly answered questions. Without that, it is hard to judge whether the method makes the models too conservative about QA. 
    3. Table 1 must also report the accuracy of the pretrained model on the subset of questions answered willingly in the FactTest experiments. 
    4. For practical significance level $\alpha=0.05$, the Type-2 error shown in Figure 2 does not appear to be controlled. It appears to invalidate the claim of the paper about controlling Type-2 error too. 
    5. The experiments consider only finetuning-based baselines and has no prior uncertainty quantification baselines or other hallucination mitigation methods to compare their uncertainty results against.
    6. It is not made clear what parts of the datasets used were for training and testing of the methods.
    7. The main text should mention how the ParaRel-OOD dataset differs from ParaRel.
    8. I don't understand how FactTest can work on just the pretrained model, without instruction tuning. The former models are known to not output the answer properly in most settings, which is the main motivation of instruction tuning. Why can't FactTest be applied to instruction tuned models?
25. The paper does not mention the relevant prior works on providing guarantees on the generations of LLMs. A non-exhaustive list is following:
    1. C-RAG: Certified Generation Risks for Retrieval-Augmented Language Models by Kang et al.
    2. Decoding Intelligence: A Framework for Certifying Knowledge Comprehension in LLMs by Chaudhary et al. 
    3. Language Models with Conformal Factuality Guarantees by Mohri et al.

### Questions
1. What does the parameterization wrt $\mathcal{D}$ mean in the outermost probability term in Equation 1? The paper mentions $\mathcal{D}$ as a dataset, which could be the sample space of the distribution. So what distribution is this probability defined over?
4. Do $\mathcal{D}_0$ and $\mathcal{D}_1$ contain of multiple answers $M(q)$ for same $q$?
5. Is $(q',M(q'))$ in line 167 from the datasets, or *any* possible pair?
6. Lines 172-173: How does assuming H to be an identity function ensure the condition $\|H\circ\hat{\eta}-\eta\|_\infty\leq\epsilon_\eta$? What is the point of having H in the first place, then?
7. Line 201: can the target distribution not be redefined and hence expanded to account for the covariate shift? Hence the previous theory can be reused. 
8. Lines 213-215: Do the source and target distributions have the same sample space?
9. Line 256: Why is the expected value of $\tilde{v}$ taken?
10. How is the frequency/probability term in Equation 6 calculated/estimated?
12. Lines 289-290: how will the distribution-free setting be violated for models requiring fine-tuning to answer factual question? I think most LLMs can do factual QA (perhaps not optimally) without finetuning. So what is the point of mentioning this?
13. Why does KLE have only a 15 generation variant in Table 1?
14. Does the experiment corresponding to Figure 3 suggest that after the construction of the certainty classifier (basically identification of its threshold), one needs to do another search for the accuracy maximizing threshold? I don't get the point of this experiment, if the search for the accuracy maximizing threshold is not a part of the method. 
15. Lines 459-460: How do you train a classifier to approximate density ratios? Is it unsupervised training? 
17. In the black-box APIs setting, is the open-source model used to get the uncertainty score even during testing?

### Soundness
1

### Presentation
1

### Contribution
2
