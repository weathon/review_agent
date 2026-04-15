# Has My System Prompt Been Used? Large Language Model Prompt Membership Inference

- Decision: Reject
- Scores: 5, 3, 6, 8, 8

## Abstract
Prompt engineering has emerged as a powerful technique for optimizing large language models (LLMs) for specific applications, enabling faster prototyping and improved performance, and giving rise to the interest of the community in protecting proprietary system prompts. In this work, we explore a novel perspective on prompt privacy through the lens of membership inference. We develop Prompt Detective, a statistical method to reliably determine whether a given system prompt was used by a third-party language model. Our approach relies on a statistical test comparing the distributions of two groups of generations corresponding to different system prompts. Through extensive experiments with a variety of language models, we demonstrate the effectiveness of Prompt Detective in both standard and challenging scenarios, including black-box settings. Our work reveals that even minor changes in system prompts manifest in distinct response distributions, enabling us to verify prompt usage with statistical significance.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper offers a method for detecting the use of proprietary system prompts by a third party.  The method is based on a statistical test that measures whether the response distribution of the third party model (which potentially uses the proprietary prompt) matches that of the same model with the proprietary prompt.  The statistical test is performed using the response embeddings taken from a BERT model.  In experiments, the method is applied to two existing system prompt datasets, as well as a novel LLM-generated dataset, showing positive results in some settings.  The authors highlight that their experiments show how minor changes to prompts can lead to significant differences in response distributions.

### Strengths
The paper is generally well-presented, with clear plots as well as psuedocode for their algorithm.  In addition, the extension of existing system prompt datasets to include more similar items creates the possibility for more interesting experimental results.

### Weaknesses
I find that this paper has many weaknesses, which I detail below.  

-After reading the introduction, I find myself questioning the exact nature and importance of the problem.  If prompt reconstruction is so difficult, then why is such a test important?  Also, I'm wondering if it is illegal to steal a proprietary system prompt?  If not, what actions would one take after receiving a positive result on your test?  I think better and more clear motivation is needed here.

-I think the related works section overemphasizes prompt engineering in general (as does the introduction), and does not make clear exactly where this paper is meant to be placed in the literature.  Also, one major concern I have is that line 62 states that "prompt reconstruction approaches usually do not offer certifiable ways to verify if the recovered prompt was indeed used"; given the "usually", I believe I am to understand that there has been previous work considering certifiable ways to verify prompt inclusion.  However, I cannot find those references in Section 2 (and also later wonder if that means that the experiments section is missing important baselines).

-Another place where I fail to grasp the motivation is Section 3.1, second paragraph.  This seems meant to be a motivating example for the methodology to come, but I find it to be confusing.

-The methodology composes existing tools (e.g., embedding model, statistical test), and lacks any significant novelty.  One opportunity to introduce novelty may have come from some method for producing especially effective task prompts (also small note, why is Section 3.3 not titled "Task Prompts"?), but this is not deeply explored.

-Related to the last point, I find that line 243 "The selection of task prompts q1, . . . , qn is an important component of Prompt Detective" is unsupported by the experiments; Lines 263-265 seem to indicate that task prompts are not very important.

-While I am not very familiar with these datasets, the tasks in Section 5.1 seem fairly trivial.  Also, why are false negative rates not included in Table 2?

-Are there any baselines that you might compare to?  Either from the prompt reconstruction literature, or membership inference literature?

-The extension to black box settings seems like another potential avenue to introduce novel methodology, but is ultimately given little attention.

-In the discussion, the authors state that "A key finding of our work is that even minor changes in system prompts manifest in distinct response distributions".  I believe this is very well known to LLM practitioners and researchers.  e.g. see Section 3.3.1 of [1]

[1] https://arxiv.org/abs/2404.09932

### Questions
Please see weaknesses for specific questions.  On a high-level, I would be interested to hear more on why is this an important problem, and how would you ultimately like to see your method used?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper discuss the detection of whether a blackbox service uses the same system prompt, or whether a system prompt of a victim was stolen. A statistical method called "prompt detective is used" which creates two generations one with a model that the victim can control its system prompt and one with the target model. The paper shows that system prompts with very minor differences can be identified.

### Strengths
- The paper has some surprising results, at line 448 ( system prompts that differ only by a typo as an example of extreme
similarity (see Appendix C for details). I think this result is quite important and may have broader implication beyond the specific problem of verifying whether the prompt is the same or not. For example, one can extract information from a blackbox model about the exact name of an api_call or some system prompt formatting even if the system designer is filtering out these information. 

- I like the idea of designing task prompts that are specifically designed to probe for the corresponding system prompt.

### Weaknesses
- I think verifying if the prompt used is the same might not be needed because one can perform prompt reconstruction or attacks to reveal the system prompt, which they are already existing work that shows it is possible [1-3]. The paper makes comparison to membership inference attacks on training data but this threat model is different as, unlike training data, system prompts can be easily reconstructed. I would at least expect the paper to make this comparison to show the advantage of the method over it. The discussion in section 2.2 while it mentions prompt leakage, does not mention why can't it be used to verify if the prompt is the same or not. 

[1] Debenedetti et al., Dataset and Lessons Learned from the 2024 SaTML LLM Capture-the-Flag Competition. NeurIPS D&B 

[2] Hui et al., PLeak: Prompt Leaking Attacks against Large Language Model Applications (https://arxiv.org/abs/2405.06823)

[3] Geiping et al., Coercing LLMs to do and reveal (almost) anything (https://arxiv.org/abs/2402.14020)

- Revealing system prompts have broader implications (e.g., one may use information in the system prompt in order to better launch prompt injection attacks). But the paper discusses that system prompts are valuable and therefore can be stolen to steal the functionality and we need to verify if that is the case. However, it does not show convincing setups where these system prompts are important. The current prompts are very generic. I would have expected to have prompts that are either large enough with a lot of detailed reasoning methods, contain private data, or have any optimized tokens that needs computation. There are two drawbacks of this: 1) we can't know if the method would generalize to these cases, 2) the generic system prompts can have overlap naturally and therefore they are not distinct. Even if the prompts are the same, that does not mean there exists an attack. 

- Regarding the above point, it seems that the test is costly in terms of the number of queries needed. If this scales with the length of the prompt, then the test may be very expensive to perform. 

- In figure 2 and section 4.1, the paper does not make it clear whether all of these scenarios are considered "stolen prompts", if the model is used in the same task, there should exist natural similarity in concept. It seems from section 5.2 that all of these are considered positive examples (similar to the system prompt). I think this is not a very reasonable assumption.

### Questions
- I don't understand Section 6. I think is not really blackbox because we assume the model exists. In principle this can be easily converted into the same setup (perform the test with each model individually and take the smallest p-value as the final one).

- Can you use the log prob to perform the test? (in this case you may not need to perform as many queries as there is no randomness for sampling).

- Are there any difference in the actual functionality of the model if one uses a system prompt with similarity 5? because in this cases, even if the method shows high similarity, the designer can easily argue that the system prompt is not the same and therefore the designer of the service may decide to always paraphrase the stolen system prompt.

### Soundness
2

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
This paper studies the problem of identifying whether an LLM uses a given system prompt, for the purposes of detecting the illegal leakage of system prompts. It applies statistical hypothesis testing to determine whether an LLM uses a given system prompt. The results indicate the effectiveness of the approach in showing that the target system prompts were not used by the given LLM for its generations.

### Strengths
1. The paper presents a novel application of statistical hypothesis testing to address an issue of economic value to LLM developers and users. 
2. The paper uses a formal approach to study and distinguish the impacts of different system prompts, that may share some commonalities. 
3. The results show the promising capabilities of the method in differentiating the system prompt used by an LLM from a target system prompt. 
4. The analysis is supported by a quite informative ablation study, which provides interesting insights into the importance of generation length versus number of generations on the statistical test.

### Weaknesses
1. The paper lacks motivation for the use of permutation testing for the task at hand. Firstly, what theoretical benefit does statistical testing bring about for accurately inferring the system prompt over existing methods? Secondly, why is this particular method of statistical testing picked from various other possibilities? 
2. As per my understanding of permutation testing, the "if" statement in line 184 (Algorithm 1), should have the condition $s^*>s^{obs}$. 
3. I believe that a better null hypothesis would be that the system prompts are not identical. This is because rejecting this would support the motivation of the paper, i.e., to detect reuse of system prompt and the privacy violation caused by that. The current method can only tell whether the system prompt was not used by the LLM, which appears to be not well-aligned with the motivation.
4. Given that this is not the first work on extracting the system prompts of proprietary LLMs, there is a lack of theoretical and experimental comparison with the various baselines/prior approaches. Examples of prior works on system prompt extraction include: Effective Prompt Extraction from Language Models by Zhang et al, Pleak: Prompt leaking attacks against large language model applications by Hui et al, Coercing
llms to do and reveal (almost) anything by Geiping et al. 
4. I see the following points missing from the current experiments section. I believe that including them will enhance the paper. 
    1. Include the results of open-source models used in Table 1, in Table 2 for hard examples as well.
    2. Include different models from the same model families (e.g., Llama-3) to see the variation in the effectiveness of Prompt Detective by number of parameters etc. 
    3. The ablation study in Figure 4 needs clarification. I think it is not appropriate to put the plot from the left panel exactly on the right panel. Although I understand what it means now, but it took me many re-readings to grasp the plot's structure. Moreover, in the left panel, the authors should clearly state the best values of the number of generations and task prompts, identified from their ablation study. The number is not possible to judge from the plot.
    4. Include more of the latest models in the experiments to thoroughly study the effectiveness of the method. Examples include: GPT-4, Claude-3.5-sonnet, Gemini. 
    5. The dependence of the results on the task probes is not adequately studied. I think there should be an ablation study of how the p-values vary with different choices for the task probes.
    6. The use of the BERT model specifically, as the projection function for the generations, is arbitrary and should be supported by an ablation study.
    7. I don't see the motivation of using universal task prompts for varying system prompts in the Awesome-ChatGPT-Prompts and Anthropic Prompt Library datasets. What is the challenge in generating custom task prompts for each system prompts?
5. The generalization of the approach to the Black-box setting, where the LLM is not known, makes a major assumption of the model being from either of 6 popular models. This does not appear to be a feasible approach in the general setting, where instead of the 6 models, another model or finetuned version of one of the 6 models could be used. Hence, I don't think that the approach is truly black-box.
6. Minor comments:
    1. I believe that the paper should include some background on the main technique used, i.e., permutation testing. This can enhance the readability of the paper. 
    2. Algorithm 1 is not described in detail in the paper. There are several constructs that are assumed to be known to the reader already. For example: what is $N_{permutations}$, $Shuffle(.)$? What does it mean to concatenate responses in line 179? Is it concatenating the *list* of responses or are the individual response strings concatenated element-wise over the 2 lists?
    3. The construction of "Hard examples" of system prompts is described in a very hand-wavy manner. For example, the distinction between "Same Prompt, Minimal Rephrasing" and "Same Prompt, Minor Rephrasing" is vague. Also, the word "minimal" has a very specific mathematical interpretation, which I don't see fit here. I would encourage the authors to clearly state the goals of each variation and the exact differences between them for the hard examples.
    6. Lines 346-347: Shouldn't "rejecting the null hypothesis of equal distributions" be "rejecting the null hypothesis of *different* distributions"?
    7. As the paper tries to verify LLM systems for copying a system prompt, it should contain the related works on verifying LLM systems in other contexts, especially the ones using statistical tests. A non-exhaustive list of related references is - 
        1. C-RAG: Certified Generation Risks for Retrieval-Augmented Language Models by Kang et al.
        2. Certifying LLM Safety against Adversarial Prompting by Kumar et al.
        3. Quantitative Certification of Bias in Large Language Models by Chaudhary et al.

### Questions
1. What is Claude 3's system prompt when generating the task prompts for given $\bar{p}$? How would that influence the quality of the generated task prompts?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces "Prompt Detective", a method to infer whether a chat LLM uses a particular system prompt.

**Threat model**: Given a target chatbot LLM with an unknown system prompt and a candidate system prompt, an attacker can query the chatbot with user prompts and observes responses (as text). The attacker's goal is to reliably guess if the candidate system prompt is different from the target chatbot's actual system prompt. The attacker generally knows the LLM being used (Sec. 6 considers a relaxation).

**Prompt Detective attack**: Given a set of user prompts, the attack samples multiple generations per user prompt, once for the target chatbot LLM with the unknown system prompt, and once for the same LLM with the candidate system prompt. The attack then performs a permutation test over the two sets of generations, permuting generations per user prompt, and using the cosine similarity of mean BERT embeddings as the test statistic. If the test's significance level is larger than some $\\alpha$, then Prompt Detective outputs that prompts are distinct, otherwise that they might be the same (i.e., do not reject null hypothesis).

**Evaluation**: The paper evaluates Prompt Detective on a broad set of prompts and models (ranging from Mistral 7B to GPT-3.5 in size) using both positive (equal) and negative (not equal) pairs of unknwon and candidate system prompt. The evaluation contains both an average-case setup and a "hard" setup that considers prompt pairs with a controlled level of similarity.

### Strengths
While prompting might not be a long-term business strategy, many current applications build upon proprietary system prompts. Hence, the problem this paper studies is relevant today. The threat model is further precise, sensible, and discussed thoroughly.

Prompt Detective is a straightforward hypothesis test without any unnecessary moving parts, yet done properly such that it achieves good performance in terms of what it set out to do. For example, in the average-case evaluation ("standard setup"), with a p-value of 0.05 (controlling false negatives, not false positives!), the method seems to work reasonably well; it yields almost no false positives, and a true positive rate consistently over 90%.

The paper is well-written, and the method is described well. The evaluation is mostly done well (except one issue; see weaknesses). The authors also investigate the Prompt Detective's degrees of freedom (e.g., more query prompts vs. more generations per query prompt), and show that the method still works in a more black-box threat model.

### Weaknesses
**Hypothesis test controls the wrong thing**: The hypothesis test is set up to control the false *negative* rate (i.e., predicting that a system prompt was not used when it was). However, as in e.g. membership inference, wrongly predicting that a system prompt *was stolen* has much stronger consequences than predicting *was not stolen* (see e.g. [Carlini et al., 2021](https://arxiv.org/abs/2112.03570)). Hence, the null hypothesis should be that the two system prompts are different (this might be hard with the used permutation test) to ensure a low false positive rate. The paper should also report full ROC curves when sweeping over significance levels $\\alpha$ (in general but especially due to the way the test is set up).

**Figures 3 and 4 lack positive pairs**: In Fig. 3 and 4, it is very much expected that the p-value decreases with more queries for *negative* cases (different prompts) due to the way the hypothesis test is set up. Hence, the p-value for positive pairs (same prompts) is much more interesting. This value should stay high for the method to work well, but I suspect that this p-value decreases too. Hence, the evaluation should also show $p^p_{avg}$ in Fig. 3 and 4.
(Additionally, both figures' plots might benefit from error bands around the lines.)

**Hard setup lacks positive pairs**: The hard setup only considers negative pairs (original system prompt plus a modified version). However, for the same reasons outlined above, there must be positive pairs to assess how well Prompt Detective works (i.e., include positive pairs and show the FNR/TPR in Table 2; simply looking at the FPR does not suffice).

### Questions
1. Why does the standard evaluation setup described in Sec. 4.3 only consider on negative pair instead of all possible pairs? As acknowledged, "negative pairs may not represent similar system prompts"; hence considering all possible negative pairs might provide a more complete picture.
2. The typo case study in App. C.2 sounds interesting; is it possible to get full results for that study?. In particular, what are the FPR/FNR and p-value for the positive case?
3. Is there any intuition why Claude Haiku behaves very differently from the other models in Table 1?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
System prompt is an important component of an LLM system as it directly controls the behavior of the base model by outlining the context of the conversation and the expected behaviors of the model. As a result, model developers can spend significant effort on developing proprietary system prompts that can boost model utility and hide them from the users. However, given the existence of various jailbreaking techniques that can drive the model to easily divulge its system prompt, it is important to have a statistical testing framework for model developers to perform end-to-end audits of a given LLM system on whether a specific system prompt has been used by the underlying LLM. To this end, the authors proposed Prompt Detective, a permutation testing framework to determine whether a given system prompt was used by a third-party language model. Through experiments, the authors demonstrate the effectiveness of their proposed method in different scenarios where the prompts are rewritten to various degrees.

### Strengths
1. This paper presents an important threat model as summarized in the previous section.
2. The paper is well-written with each figure sending a clear message. I can get the gist of it in my first pass.

### Weaknesses
1. While this work presents a valid threat model, have the authors considered possible adaptive attacks in which an adversary intends to hide the fact that they used a specific system prompt by optimizing the prompt in a way that preserves utility but results in inconclusive test results?
2. The current experiments exclusively use BERT models for embedding. It would be much appreciated if the authors could supplement experiments to demonstrate the effectiveness of the models on more embedding models.
3. In a more realistic scenario, an attack could use jailbreaking techniques to steal the system prompt of an LLM with blackbox access, through which the attacker can obtain a prompt that may or may not reproduce the original system prompt. Then, the attacker may choose to deploy the stolen prompt to their systems with a similar base LLM. Instead of manually varying the system prompt to validate the statistical testing framework, have the authors considered this threat model?

### Questions
1. Is the proposed method applicable to soft prompts and stolen in-context demonstrations?
2. My questions raised in “weakness.”

### Soundness
3

### Presentation
3

### Contribution
3
