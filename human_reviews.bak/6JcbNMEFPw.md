# Large Language Models Can Be Good Privacy Protection Learners

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 5, 5, 5

## Abstract
The proliferation of Large Language Models (LLMs) has driven considerable interest in fine-tuning them with domain-specific data to create specialized language models. 
Nevertheless, such domain-specific fine-tuning data often contains sensitive personally identifiable information (PII). Direct fine-tuning LLMs on this data without privacy protection poses a risk of leakage. 
To address this challenge, we introduce Privacy Protection Language Models (PPLM), a novel paradigm for fine-tuning LLMs that effectively injects domain-specific knowledge while safeguarding data privacy. 
Our work offers a theoretical analysis for model design and delves into various techniques such as corpus curation, penalty-based unlikelihood in training loss, and instruction-based tuning, etc. Extensive experiments across diverse datasets and scenarios demonstrate the effectiveness of our approaches. In particular, instruction tuning with both positive and negative examples, stands out as a promising method, effectively protecting private data while enhancing the model's knowledge. Our work underscores the potential for Large Language Models as robust privacy protection learners.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents Privacy Protection Language Models (PPLM), a new approach to fine-tuning Large Language Models (LLMs) that safeguards against the leakage of personally identifiable information (PII). The authors offer both theoretical analysis and practical methods, incorporating multiple strategies including new training loss, instruction-based tuning, a PII contextual classifier, and direct preference optimization. The work is validated on three real-world datasets, demonstrating that PPLMs effectively balance domain-specific knowledge acquisition and privacy protection.

### Strengths
1. This paper addresses the critical issue of fine-tuning language models with an emphasis on privacy considerations.

2. The proposed approach comprises an integrated set of strategies designed to enhance privacy protection during the fine-tuning process.

3. Experiments provide empirical evidence for the effectiveness of the proposed methodologies.

### Weaknesses
1. The authors acknowledge that Differential Privacy (DP) offers a general framework for privacy protection, distinct from the problem presented in this paper. Nevertheless, DP could serve as a robust baseline for comparative evaluation. The authors might consider conducting experiments with differential privacy parameters set at (e.g., $\epsilon=1, 8$) for a comprehensive evaluation.  

2. I am a little confused about how the analysis in section 4.1 ensures the proposed method to safeguard privacy.

### Questions
NA

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors focus on the problem of fine-tuning LLMs on domain-specific data, which may possess a different distribution compared to the pre-training data so that the model can enhance its knowledge in a new domain. On the other hand, the authors consider the scenario where the domain-specific data may contain sensitive information such as personally identifiable information (PII), therefore, the fine-tuning has to be performed in a privacy-preserving manner. The authors introduce various techniques that enable LLMs to respond to queries without leaking PII from the domain-specific data and compare their methods through experiments with widely used NLP metrics for utility and define a privacy leakage metric for privacy. The authors demonstrate that by applying instruction tuning with demonstrations for privacy protection generation and PII leakage generation one can obtain favorable balance between the utility and the privacy of LLM generations.

### Strengths
* The paper is well written, which facilitates for the reader to understand the problem definition and introduced techniques.
* The paper focuses on an important problem that tackles fine-tuning on private domain-specific data while protecting the privacy of data samples.
* The paper utilizes state-of-the-art techniques from the NLP literature such as instruction-tuning and DPO algorithm to enhance the model's knowledge while not leaking PII from the underlying dataset.

### Weaknesses
* The paper mainly focuses on PII leakage regarding privacy protection of fine-tuning data. PII could naturally be thought as the majority source of a privacy violation, but actually privacy is more general than PII (e.g. according to GDPR). A generation without any PII may also cause a privacy violation if the content is linkable to an individual or a group of individuals. I think the paper should make it clear that in this sense the approaches do not guarantee any formal or comprehensive privacy guarantees.

* I don't quite follow how the task is different from Differential Privacy (DP). quoting the sentence from the paper: "Note that the task is different from the Differential Privacy (DP) LLMs (...) which is targeting to protect general differential privacy Zhao et al. (2022), our focus is the contextual privacy protection finetuning for knowledge injection, in which the PII to be protected in the tuning text data is annotated by users." DP provides a formal and comprehensive privacy guarantee to the individuals in the dataset. The mathematical guarantee is on their presence in the training data and this includes the protection of their PII as a special case. So it protects the whole text of individuals including any PII. Therefore, it's not clear for the reviewer how the problem in the paper is really different from DP.

* The authors state that instruction-based tuning provides favorable balance between the utility and the privacy for LLMs. It's slightly unclear how instruction-based tuning is performed and what advantages it provides over just generating regularly + scrubbing the PII as post-processing. I have a few specific questions about this particular approach, which is stated in questions part.

### Questions
1) I'd like to understand more clearly how instruction-based tuning is performed. It involves demonstrating positive and negative cases. Negative cases include generations with PII. How do you come up with negative cases and also the positive versions? Is it by using the scrubadub library? You mention in the introduction that "Directly applying techniques like Named Entity Recognition (NER) can lead to inaccurate identification of PII". So how does your approach for generating negative and positive cases with PII provide more accurate identification of PII?

2) Related to the question above, if you have PII identification tool that can prepare negative and positive cases, then why do we even instruction-based teach the model to generate without PII? Can we not just regularly generate and use this PII identification tool to convert it to a positive case? As the model learns from the negative and positive cases that are prepared with this PII identification tool, then should it not at best perform as good as this tool? By these questions, the reviewer is just trying to understand the use-case and advantages of the proposed technique and confirm her/his understanding.

3) There are various attacks shown that lets users make LLMs ignore their instruction and follow the user instructions? Can these be performed so that LLMs would leak PII instead of following the instruction?

4) Table 8 demonstrates example responses generated on the Wikidoc Patient Information Dataset by the vanilla model (original) and the model trained with the instruction (IT) strategy. But the questions are so generic such as "Could you provide a brief explanation of familial hypercholesterolemia?". Why would model even leak PII answering this question? Maybe I am missing something but this table does not really show the reviewer any guarantee on the efficacy of the approach. It also questions the reviewer if the validation in the experiments really measure the PII leakage issue properly.

5) As stated in the weaknesses, I think this problem is very relevant to DP. One would expect that fine-tuning with DP also enhance the model knowledge in domain-specific data and also protect against leakage of PII in the dataset. Do authors think differently that they don't provide and compare with the DP solution?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the privacy-utility trade-off problem studied for LLMs. A wide variety of techniques are considered and empirical results are provided pertaining the accuracy and privacy leakage.

### Strengths
1. The problem setup considered here is extremely relevant in real-world LLM systems today where a nuanced definition of privacy is required (as opposed to DP or PII extraction). 
2. Results are presented using real-world datasets and state-of-the-art language models.

### Weaknesses
The main weakness of this work is that, as a reader, I am unclear about the overall story and message of this paper. Please see the next section for my various questions.

### Questions
1. The last sentence of the first paragraph in Section 2 (on page 2) appears quite problematic. It's almost a tautology to say that "Differential Privacy is targeting to protect general differential privacy". Given that DP is a very rigorous mathematical notion that has consistently stood the test of time, if the authors here want to propose a new notion of contextual privacy instead, they must define what is clearly means. 
2. The first sentence in Section 4 says that this paper considers two techniques: corpus curation and SFT. But the rest of that Section covers various other methods such as in-context learning (Sec 4.2.4) and a auxiliary classifier model (Sec 4.2.3). 
3. In corpus curation method (Sec 4.2.1), the authors propose PII removal, but in the second paragraph in Section 2, they criticize removal of PIIs from training corpus as it "hampers the model's capacity to process and understand certain contexts".
4. In Sec 4.2.2, they mention using a service called "scrubadub" to determine n-grams associated with PIIs. But why is that needed, given that each token in each sample has a binary variable associated with it which indicates whether or not the corresponding token is contextually private?
5. On the evaluation side, a "privacy leakage metric" is defined in Sec 5.3. This metric uses a binary indicator corresponding to each token in a model-generated sequence. How is this indicator variable determined? 
6. The authors consider only LLaMa family of models for their experiments. However, do these results extend to other LLM architectures? I think a more comprehensive coverage on that front is needed in the empirical results. 

Minor typos:
1. In second from top paragraph on page 3, "Pareto" is misspelled as "Paterto".
2. Missing/extra parenthesis in the subscript in second line of Section 3.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper studies how large language models can be modified to protect personal identifiable information (PII). The authors propose several strategies: (1) scrubbing PII from the dataset, (2) modifying the loss to penalize PII tokens, (3) an output filter that consists of a binary classifier to identify PII tokens, (4) providing instructions and few-shot examples to avoid generating PII tokens, and (5) direct preference optimization. The authors present results for three dataset and find that instruction based tuning gives the best utility privacy trade off.

### Strengths
- The paper provides all necessary details and code to reproduce the experiments.
- The paper uses state of the art models and fine-tuning methods.
- The authors present a theoretical intuition why scrubbing should perform worse than an output filter.

### Weaknesses
- Privacy protection is a problem that needs to provide protection even against extreme cases. The authors chose metrics that capture the average leakage but this is not adequate to measure privacy.
- The ground truth is generated by a PII scrubber (scrubadub). It is not clear from the paper how well this scrubber performs. If scrubber produces many false negatives, then the resulting metrics may not accurately represent the actual privacy risk of the methods.

### Questions
- Can you give details how well the scrubber performs? Perhaps by testing it on the TAB dataset [1]
- Why does the leakage for instruction tuning go up for larger models. I would expect larger models to follow instructions better.
- Could you comment on the risk of privacy metrics that measure average leakage vs worst case leakage.
- Suggestion: There is also relevant work from our colleagues in the security space that could be worthwhile to include e.g. [2,3].

[1] Pilan et al https://arxiv.org/pdf/2202.00443.pdf

[2] Kim et al https://arxiv.org/abs/2307.01881

[3] Lukas et al https://arxiv.org/pdf/2302.00539.pdf

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
