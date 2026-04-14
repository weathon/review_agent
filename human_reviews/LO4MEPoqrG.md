# Does Safety Training of LLMs Generalize to Semantically Related Natural Prompts?

- Decision: Accept (Poster)
- Scores: 5, 6, 3, 6

## Abstract
Large Language Models (LLMs) are known to be susceptible to crafted adversarial attacks or jailbreaks that lead to the generation of objectionable content despite being aligned to human preferences using safety fine-tuning methods. While the large dimensionality of input token space makes it inevitable to find *adversarial* prompts that can jailbreak these models, we aim to evaluate whether safety fine-tuned LLMs are safe against *natural* prompts which are semantically related to toxic seed prompts that elicit safe responses after alignment. We surprisingly find that popular aligned LLMs such as GPT-4 can be compromised using naive prompts that are NOT even crafted with an objective of jailbreaking the model. Furthermore, we empirically show that given a seed prompt that elicits a toxic response from an unaligned model, one can systematically generate several semantically related *natural* prompts that can jailbreak aligned LLMs. Towards this, we propose a method of *Response Guided Question Augmentation (ReG-QA)* to evaluate the generalization of safety aligned LLMs to natural prompts, that first generates several toxic answers given a seed question using an unaligned LLM (Q to A), and further leverages an LLM to generate questions that are likely to produce these answers (A to Q). We interestingly find that safety fine-tuned LLMs such as GPT-4o are vulnerable to producing natural jailbreak *questions* from unsafe content (without denial) and can thus be used for the latter (A to Q) step. We obtain attack success rates that are comparable to/ better than leading adversarial attack methods on the JailbreakBench leaderboard, while being significantly more stable against defenses such as Smooth-LLM and Synonym Substitution, which are effective against existing all attacks on the leaderboard.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper presents a novel method for evaluating the robustness of safety training in Large Language Models (LLMs) using semantically related natural prompts, focusing on their vulnerability to jailbreaks. The authors introduce Response Guided Question Augmentation (ReG-QA), a technique for generating diverse, natural questions that can elicit unsafe responses from LLMs, despite safety fine-tuning. They show that aligned models like GPT-4 are susceptible to naive prompts, achieving high attack success rates.

### Strengths
+ The authors introduce the ReG-QA method, which provides a compelling approach to stress-test LLMs with in-distribution prompts, uncovering substantial safety weaknesses.
+ Using an unaligned LLM to generate answers to unethical questions and then regenerate the questions based on those answers is a novel jailbreak technique.
+ The paper identifies key failure modes in aligned LLMs: (i) susceptibility to cues embedded in the prompt from the answer, and (ii) the capability to generate jailbreak questions when toxic answers are provided.

### Weaknesses
- While the experiments are thorough, they are primarily centered around specific LLMs like GPT-3.5, GPT-4, and a few others such as Gemma-2. The paper does not explore whether the proposed method generalizes to different architectures or models fine-tuned with alternative safety techniques.
- The paper also lacks a comparison with other well-known jailbreak methods, such as GCG and GPTFuzzer. Notably, GPTFuzzer also leverages the concept of seed prompts in its attacks, making a direct comparison relevant.
- The underlying logic behind the motivation is still unclear, leaving readers confused about how the method effectively exploits the vulnerabilities in safety training. It does not provide sufficient theoretical insight into the mechanisms that make ReG-QA effective in bypassing safety fine-tuning.

### Questions
See weakness

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
3

### Summary
This paper proposes a method, ReG-QA, which uses natural prompts to jailbreak aligned large language models (LLMs). ReG-QA first utilizes an unaligned LLM to generate responses containing unsafe information for a given seed question. These responses are then used by a safety-aligned LLM to generate corresponding questions, which can be used to jailbreak aligned LLMs. This work leverages the discrepancy in safety-aligned LLMs between harmful questions and responses, achieving results that surpass baseline methods.

### Strengths
1. This paper introduces a novel jailbreak method based on natural prompts, which is more closely similar to human language. This similarity makes these prompts more difficult for safety filters to detect, posing greater challenges and providing a new perspective for safety alignment.

2. The authors conducted extensive experiments on several advanced closed-source models. Compared to baseline methods, the proposed approach demonstrates a significant improvement in ASR.

### Weaknesses
1. This paper only uses Para-QA as a simple baseline, which may not be sufficient to demonstrate the effectiveness of the proposed method. Could additional baseline methods be incorporated for more comprehensive evaluation?

2. The paper could benefit from an analysis of the specific factors contributing to the effectiveness of ReG-QA. For example, questions generated by the unsafe LLM may exhibit better jailbreak capabilities compared to seed or paraphrased questions. Such analysis could provide deeper insights and could support the development of more robust safety alignment strategies.

3. The study examines closed-source models exclusively from OpenAI, which may limit the diversity of the models analyzed. Given that different institutions often employ varied safety alignment strategies in their closed-source models, incorporating models from a range of sources could enhance the assessment of the proposed method's generalizability.

### Questions
Please refer to the weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper focuses on the performance of Large Language Models (LLMs) in generalizing to potentially harmful yet naturally phrased jailbreak questions after undergoing safety training. The authors propose a method to construct such natural jailbreak questions by initially having an unaligned LLM generate answers to malicious prompts, followed by an aligned LLM generating questions that would lead to those answers in reverse. The experiments demonstrate that this straightforward approach achieves a relatively high success rate in attacking models like GPT-3.5-turbo and GPT-4-turbo.

Although the research question is interesting, unfortunately, the paper lacks readability, and the experiments and claims are not yet of publishable quality. There are numerous aspects that are concerning and confusing.

### Strengths
I appreciate the visualization in Figure 1. Although it is not very readable, I understand its attempt to classify existing jailbreak attack types based on data distribution, offering a new perspective on jailbreak-related research.

### Weaknesses
**1. The most significant flaw lies in the writing.**

Specifically, Section 4 and Figure 1 are very confusing, with the caption of Figure 1 being especially unclear. The distinction between R0 and R1 and their labeling in the text do not seem to align. On line 185, where it states “with R2 being the region…,” was R2 a typo, and should it be R0 instead? What does the green cross in the figure signify?

Additionally, the table from lines 327 to 356 seems to belong before Section 6.2, while the content from lines 316 to 326 (the experimental analysis section) should perhaps be positioned after line 374. Table 1 would be more appropriate in an appendix rather than in the main text, and its content is unclear—what is meant by “agreement”? How were the numbers in this table assessed?

These issues make the paper feel like a hastily prepared draft that is challenging to read and not ready for peer review, requiring substantial revisions. In my opinion, **submitting an unprepared manuscript is a waste of reviewing resources and shows a lack of respect for reviewers’ time**.

**2. Some claims in the paper lack support.**

Recent work on jailbreak attacks has already demonstrated the ability to generate stealthy adversarial prompts (e.g., https://arxiv.org/abs/2310.04451, https://arxiv.org/abs/2404.16873). However, the paper seems to suggest that these jailbreak methods are not natural or in-distribution (which is the paper’s stated motivation). This type of claim requires empirical evidence; otherwise, it is unconvincing.

**3. The technical contribution of this paper is limited.**

The reverse generation method in safety alignment scenarios is not novel (e.g., https://arxiv.org/abs/2212.01810, https://arxiv.org/abs/2311.08685). There are likely other relevant papers, but it is the authors’ responsibility to conduct a thorough literature review. Thus, methodologically, this paper does not offer substantial innovation.

Regarding the empirical analysis, I find that the current experiments do not provide many insightful results, lack comprehensiveness, and raise concerns. For instance, what would the effect be of using different Q->A and A->Q models? The target model in the experiments does not include more advanced models, such as GPT-4o or Claude 3.5, even though GPT-4o is already used as the A->Q model. Would the capabilities of the Q->A and A->Q models impact the attack success rate? Due to the lack of a broad empirical analysis, it is impossible to draw these conclusions or gain valuable insights.

### Questions
I recommend that the authors make significant revisions in writing and claim articulation. Please refer to the Weaknesses section for specific critiques.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a jailbreak attack called Response Guided Question Augmentation (ReG-QA) that breaks the safety fine-tuning of LLMs. The main difference between this attack and the others is that the prompts generated by ReG-QA are natural and not optimized to jailbreak the model. ReG-QA first generates 100 answers for the seed prompt using a safety-**un**aligned LLM. Then, for each generated answer, generates 10 candidate questions. All these questions, semantically close to the original seed question, are prompted to the target model to evaluate the generalizability of the safety alignment of the target LLM.  One key result that they mention is even safety-aligned LLMs generate jailbreaking questions when a harmful answer is given (2nd stage of their algorithm).

### Strengths
- The question the authors ask is an important aspect of LLMs to analyze. Moreover, looking at the jailbreak samples from a natural perspective rather than an optimized unnatural perspective is important.
- ReG-QA shows good results.
- Interesting finding: safety-aligned LLMs generate jailbreaking questions when a harmful answer is given

### Weaknesses
- The presentation of the paper is poor. These are the points that I realized:

    - Some parts are explained under Section 6.1 Experimental details, and then starting Section 6.2 the experimental results are discussed for one paragraph. But then suddenly datasets and baseline methods were explained in the middle (lines 347-375) followed by the rest of the experimental discussion. 
    -  Figure 3 is put in the main paper with 2 subfigures but never referenced. 
    - Figure 1 explanation is unclear. R0 is referred to before defining R0. Region identities are miswritten, e.g "R3 denote a subset of R1" 
    - Some extra materials are referred to the supplement without stating the section (e.g. lines 307 and 460 )
    - Line 447 sentence not completed.
    - Line 455, figure 4 does not have results for gpt-3.5
    - correct citation scheme is not followed with some references, e.g. line 360
    - Typos that I've realized: Fig.2 description-answer, line 259-left quotation mark, line 313-agreement
    - Academic language is not followed in some places, e.g. line 325- "What is interesting is..."
    - I think it is better to put "Judge Selection" before "Judge Prompt and Model" 
    - The next point can also be considered a poor presentation 
- In Table 3, there are 2 columns with the same configurations (columns 2 and 5) but with different results. It decreases the reliability of the results presented in the paper. 
- It is stated that the generated answers should be long but the reason/intuition/explanation behind this is missing.
- Why choose "gpt-4o-mini-2024-07-18" as a judge while LLama-70B is better and cheaper?
- The criteria for filtering the questions and answers are not explained/under-explained. (even though stated that it is explained in section 5.2 - line 263  )
- It would be helpful information to provide the mean and the std of a number of questions per seed after filtering. It is stated that it is significantly less than 1000 (line 432) and referred to Figure 4 in line 293 but I don't see the connection.
- Only the naive baseline (i.e paraphrasing the seed question) is included in the experimental results. For a more comprehensive analysis, other baselines should be included (e.g. the ones mentioned in the related works). 
- For a comprehensive analysis again, one more dataset could be included. Some of the cited previous work uses AdvBech dataset for example. 
- Appendix B can be extended with more samples.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
