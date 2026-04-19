# Distinguishing Ignorance from Error in LLM Hallucinations

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5, 3, 6

## Abstract
Large language models (LLMs) are susceptible to hallucinations---outputs that are ungrounded, factually incorrect, or inconsistent with prior generations. We focus on close-book Question Answering (CBQA), where previous work has not fully addressed the distinction between two possible kinds of hallucinations, namely, whether the model (1) does not hold the correct answer in its parameters or (2) answers incorrectly despite having the required knowledge. We argue that distinguishing these cases is crucial for detecting and mitigating hallucinations. Specifically, case (2) may be mitigated by intervening in the model’s internal computation, as the knowledge resides within the model's parameters. In contrast, in case (1) there is no parametric knowledge to leverage for mitigation, 
so it should be addressed by resorting to an external knowledge source or abstaining. To help distinguish between the two cases, we introduce Wrong Answer despite having Correct Knowledge (WACK), an approach for constructing model-specific datasets for the second hallucination type. Our probing experiments indicate that the two kinds of hallucinations are represented differently in the model's inner states. Next, we show that datasets constructed using WACK exhibit variations across models, demonstrating that even when models share knowledge of certain facts, they still vary in the specific examples that lead to hallucinations. Finally, we show that training a probe on our WACK datasets leads to better hallucination detection of case (2) hallucinations than using the common generic one-size-fits-all datasets.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The paper addresses hallucinations in LLMs and introduces a new approach, Wrong Answer despite having Correct Knowledge (WACK), to differentiate between two types of hallucinations:

* HK-: Hallucinations due to lack of knowledge.
* HK+: Hallucinations despite the model having the required knowledge.

The paper constructs model-specific datasets to capture these types and demonstrates that such datasets outperform generic ones in detecting hallucinations. WACK probes internal states of the model to classify hallucinations more effectively across various prompt settings. This model-specific methodology generalizes between synthetic settings (Bad-shots and Alice-Bob) and allows better preemptive hallucination detection.

### Strengths
* This paper proposes a novel categorization and dataset, WACK, to study hallucinations based on knowledge availability.
* The experiments are well-designed and comprehensive, employing multiple models and tasks.
* The results highlight the importance of model-specific datasets and pave the way for future work on targeted mitigation of hallucinations.

### Weaknesses
* This paper primarily investigates two categories of reasons behind LLM hallucinations. However, the phenomenon of hallucination has been well studied and the contribution of this work is limited. The authors should further build on their findings to better alleviate hallucinations.
* The study is limited to three models in the 7B-9B range, raising questions about generalizability to larger models and commercial LLMs, e.g. gpt-series and claude.
* While the synthetic setups (Bad-shots and Alice-Bob) are innovative, real-world scenarios might involve more complex interactions, which are not captured.
* The focus on extreme knowledge scenarios (high- and low-knowledge) omits middle-ground cases, which could also be important.

### Questions
* How would the WACK approach extend to much larger LLMs like GPT-4 or PaLM?
* Is there any plan to incorporate more complex real-world prompts beyond the synthetic Bad-shot and Alice-Bob settings?
* How sensitive are the models to variations in the Bad-shot prompts? Could changing the bad-shots affect HK+ detection?

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
This paper proposes WACK, an approach for constructing model-specific datasets for hallucination detection with two types of hallucination samples. Leveraging existing closed-book question answering datasets, they first classify the samples into two categories based on whether the model can consistently answer the questions correctly in a closed-book setting. Then they prompt the model to generate hallucinated answers for those samples with correct predictions. They conduct analysis on the resulting dataset by training a linear probe on model’s inner states. Their experiments show that the proposed hallucination dataset is useful and better than a generic dataset for fine-grained model-specific hallucination detection.

### Strengths
- The presentation is good. The writing is well-structured and easy to follow.
- Hallucination detection in LLMs is important and timely. Compared with existing model-specific hallucination datasets, the proposed method can generate samples with a specific type of hallucination (i.e. the model is confused by the context despite having the relevant parametric knowledge), which enables more fine-grained hallucination detection.

### Weaknesses
- The motivation behind the prompting approach used in Alice-Bob setting is unclear. 
- The synthetic hallucination dataset is model-specific and the data creation process is sensitive to the in-context hallucination samples in use. It is not clear how such datasets can be applied for evaluation in real-world scenarios since we cannot compare different models or different hallucination mitigation approaches on it.

### Questions
- Can you highlight the technical novelty in your work?  Is bad shots prompting and Alice-Bob prompting part of your technical contributions or did you adapt them from prior work?
- Did you measure the distribution similarity between the dataset generated by bad shots prompting and the one generated by Alice-Bob for the experiments in Section 4.2?
- I find the experiment design in Section 5.3 is confusing. Can you explain in more detail about how the probe is applied in this case and how is the setup different from the experiments in previous sections?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper categorizes LLM hallucination into two types: (1) cases where the LLM lacks sufficient information within its parameters to generate a correct answer (denoted as HK-), and (2) cases where the LLM provides an incorrect answer despite having the relevant information within its parameters (denoted as HK+). Focusing on the second type, this paper proposes a method for constructing model-specific datasets. The classifier trained with these datasets demonstrates the ability to classify HK+ types based on the model’s internal state.

### Strengths
- Dividing hallucination into two types can be useful, as suggested by the authors. Each hallucination type (HK+, HK-) can be mitigated using different methods (e.g., HK- can be addressed with external knowledge).
- Providing datasets that include a new type of hallucination (HK+) can benefit the NLP community.

### Weaknesses
If these below points can be clarified through additional responses from the authors, I would be glad to adjust my evaluation accordingly.

- The justification for categorizing hallucination types is weak. The paper does not sufficiently verify whether HK+ and HK- are truly mitigated by different methods (e.g., the performance improvement with external knowledge for HK- may be greater than for HK+). Additionally, the analysis supporting the claim that HK+ and HK- have distinct characteristics is inadequate. For instance, the experiment in Section 4.1 does not effectively demonstrate that the model’s internal state changes according to these hallucination types, contrary to what is asserted in line 311. The figures (e.g., Figure 3) display accuracy by model rather than by class (hallucination type). In summary, this paper should include stronger evidence to support the claim that distinguishing between HK+ and HK- is beneficial.

- The experimental detail is inconsistent. Figure 5 discusses "Knowledge and hallucination differences". Does this refer to 'factually correct' results, HK+, or HK-? If not, is it something new? Additionally, is there a distinction between the "probe" and the "classifier"? Furthermore, the meaning of the dashed line in Figure 3 is not explained (it appears to be a random baseline).

- In my opinion, some experimental setups and their significance are unclear. For instance, what is the meaning of the experiment in Section 4.2? In line 359, the authors mention that the Bad-shot and Alice-Bob synthetic settings are suitable for examining HK+, but it is unclear how this is demonstrated through generalization between the two settings. Additionally, in the experiment described in Section 5.2, the generic dataset should not account for model-specific hallucinations or knowledge. However, according to Appendix B.1, Mistral is used. Considering that Mistral is one of the three models used in the experiment, wouldn’t this also induce model-specific hallucinations and knowledge?

### Questions
- Is it possible to provide null accuracy for each classification result? (Figure 3,4,6,7)
- Was the classifier trained on different QA datasets (e.g., TriviaQA or NQ)? Training a new classifier for each model and each dataset could potentially be costly; what are the training costs and how many steps or training data is required? Could you provide experimental results related to training data efficiency?

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
4

### Summary
The study aims to detect two types of hallucinations in LLMs, (1) the model does not hold the correct answer in its parameters; (2) the model answers incorrectly despite having the required knowledge. To this end,  the study introduce Wrong Answer despite having Correct Knowledge (WACK), an approach for constructing model-specific datasets for the second hallucination type. Experimental results show that training a probe on the WACK datasets leads to better hallucination detection of case (2) hallucinations than using the common generic one-size-fits-all datasets.

### Strengths
1. This study intriguingly explores the resilience of knowledge in large language models (LLMs) when faced with incorrect demonstrations.
2. The findings indicate that a probing based on the proposed WACK dataset is successful in identifying knowledge within LLMs that is susceptible to being misled by incorrect demonstrations.

### Weaknesses
1.  What is the value to analyze the types of hallucinations, namely HK- (hallucination caused by lack of knowledge) and HK+ (hallucination despite knowledge)? The scenario involving HK+ seems impractical; in real-world situations, users are unlikely to provide numerous incorrect inputs to elicit the correct answer. A scenario where incorrect information subtly influences the dialogue might be more realistic.
2. The approach to detecting factual accuracy and HK+ is strange. If a model is prompted with incorrect examples, why should it be expected to produce the correct response? It appears that the study is intended to test the resilience or robustness of the model's internal knowledge against incorrect prompts, rather than focusing on the hallucination of internal knowledge itself.
3. Given the study aims to assess the robustness of knowledge in various contexts, the primary focus should be on the impact of bad interventions. The author should undertake a thorough and detailed analysis that includes the effects of varying numbers of demonstrations, the reasons for the induced answers, and so forth.
4. Why only use three negative examples for the study? The selection and impact of these hyper-parameters, such as the number and content of these negative cases, should be discussed. Additionally, it might be beneficial to compare these with simple templates that lack substantive information.
5. The paper could benefit from more precise writing. For instance, lines 115-118 should clearly define what is meant by "high-knowledge" and "low-knowledge." The current wording is vague and difficult to comprehend.
6. The section on related work discussing 'Jailbreak' does not seem relevant to this study. A discussion on the robustness of knowledge within large language models should be included to provide a more informative background.

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a method to separate two causes for LLM hallucination – when the model does not have the related parametric knowledge, and when the model lies despite of having the required knowledge.

### Strengths
-Most of existing methods tend to misuse the term of hallucination and mix up these two cases. So this paper is potentially impactful to open up a new direction.
-The paper proposes an effective framework to generate good-shot and bad-shot examples.

### Weaknesses
-There are quite a few unexplained and confusing terms, such as "snowball effect" and "high knowledge".
-It would be great if the authors can add a discussion section about how to leverage this categorization for hallucination mitigation.
-The experiment section lacks of qualitative analysis to make the results more insightful. 
-Section 5.3 preemptive detection of hallucination is an interesting idea, but the description is very vague and would be difficult to duplicate.

### Questions
-In the bad-shot setting, how important is it to generate diverse examples? If it's crucial, how did you make sure the three random examples are representative?

### Soundness
3

### Presentation
3

### Contribution
3
