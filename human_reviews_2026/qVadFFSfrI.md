# Diagnosing and Remedying Knowledge Deficiencies in LLMs via Label-free Curricular Meaningful Learning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 4

## Abstract
Large Language Models (LLMs) have demonstrated impressive generalization ability by learning from extensive unlabeled text. However, they still exhibit reasoning mistakes, which can affect their trustworthiness and reliability. Although users can interact with LLMs and provide diverse and comprehensive queries to expose the flaws of LLMs, obtaining sufficient and effective feedback is demanding. Furthermore, comprehensively evaluating LLMs with limited labeled samples is difficult. These make it a challenge to diagnose and remedy the deficiencies in LLMs through rich label-free user queries. To tackle this challenge and considersing that LLMs' reasoning mistakes often stem from knowledge deficiencies, we propose label-free curricular meaningful learning (LaMer), which first employs relative entropy to diagnose and quantify knowledge deficiencies of LLMs in a label-free setting. Then, LaMer adaptively synthesizes augmentation data based on deficiency severity and progressively remedies them with a curricular remedy strategy. Experiments show that LaMer effectively diagnoses and remedies knowledge deficiencies in LLMs, improving various LLMs across seven out-of-distribution (OOD) reasoning benchmarks, achieving comparable results to baselines with only 40% training data. LaMer even surpasses methods that rely on labeled data for deficiency diagnosis. In application, LaMer offers a diagnostic tool for efficient LLM development.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes LaMer, a label-free pipeline that (1) retrieves generic facts for each unlabeled query from an external knowledge base, (2) diagnoses “knowledge deficiencies” by comparing a model’s predictions with and without the retrieved fact, and (3) remedies those deficiencies by synthesizing training examples whose quantity scales with the diagnosed severity and training in an easy-to-hard curriculum. Using open-weight models, the method reports consistent gains over several augmentation baselines on seven out-of-distribution reasoning benchmarks, often matching or surpassing baselines with about 40% of the training data.

### Strengths
- The paper ties together retrieval, a simple distribution-shift diagnostic, and a curriculum that scales data to severity, turning unlabeled user queries into targeted training data without manual labels. The case study and “remedied examples” analysis make the mechanism concrete and inspectable.
- The paper contrasts label-free deficiency detection against perplexity and a label-reliant data-mining baseline, studies the curriculum order (vs. shuffling), and separates “helpful” vs. “misleading” retrieved facts—showing both expose repairable deficiencies. 
- It specifies the KB (GenericsKB with confidence filtering), retrieval embedding (FlagEmbedding), the number of matched facts per query, the synthesis protocol, and PEFT training settings, which lowers the barrier to trying LaMer in practice.

### Weaknesses
- The diagnosis hinges on retrieved facts being relevant and on a chosen threshold to flag deficiencies. If retrieval drifts (spurious or overly generic facts) or if the threshold is mis-set, the pipeline may teach to noise. 
- The severity buckets and the number of synthesized examples per bucket are fixed heuristics. While effective, it maybe interesting to explore the direction of an adaptive curricula (e.g., stopping rules per deficiency) or data-budget trade-offs across tasks and models. 
- While “remedied examples” are counted, there’s no probing of whether the method inadvertently forgets useful adjacent knowledge or whether it changes calibration/uncertainty properties.

### Questions
- I am a bit curious about the retrieval robustness, how sensitive is the deficiency diagnosis to the retrieval pipeline (embedding model choice, number of facts, KB domain coverage)? What happens if retrieved facts are partially wrong or overly generic?
- How did you select the deficiency threshold, and how stable are results across plausible ranges? Could you replace hard thresholding with an adaptive percentile per domain or model?
- How would you adapt LaMer when queries are private or streaming (e.g., enterprise chat), and when external KBs are domain-specific or scarce?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces LaMer, a novel framework for diagnosing and remedying knowledge deficiencies in LLM without relying on labeled data. The core idea is to use relative entropy to identify knowledge gaps by measuring the change in an LLM's output distribution before and after being provided with relevant external knowledge facts. Based on the severity of the diagnosed deficiency, LaMer employs a curricular learning strategy to synthesize a varying number of diverse examples to progressively fine-tune the model. The empirical results are strong and consistent across four different LLMs and seven OOD benchmarks, demonstrating the effectiveness and efficiency of the proposed method.

### Strengths
1. The primary contribution (using relative entropy for label-free diagnosis of knowledge deficiencies) is well-motivated and novel. It addresses a critical and practical challenge in the continuous improvement of LLMs: how to perform targeted enhancements without expensive human annotation.
2. The integration of curricular and meaningful learning is well-executed. The paper clearly demonstrates that diagnosing deficiencies first and then applying a targeted, progressive remedy (from easy to hard) is more effective than naive data augmentation or random-order training.

### Weaknesses
1. The framework's effectiveness is highly dependent on two external components: a comprehensive knowledge base (GenericsKB) and a powerful teacher model (ChatGPT) for data synthesis. This reliance may limit its applicability in scenarios where a high-quality KB is unavailable or the cost of using a powerful synthesis model is prohibitive. The paper does not sufficiently discuss the impact of these dependencies.
2. The RE thresholds used to categorize deficiencies into "Easy," "Normal," "Hard," and "Unfair" are presented as heuristics. The paper lacks a rigorous justification or a sensitivity analysis for these values, making it unclear how robust the method is to these choices. 
3.  The diagnosis step relies on knowledge retrieved via embedding similarity, which can be noisy and sometimes irrelevant. The paper does not address how such noisy knowledge might affect the stability of the posterior distribution Q and the resulting RE calculation. This raises concerns about whether the RE score is always a reliable indicator of a true knowledge deficiency.
4. The methodology description contains ambiguities that hinder clarity. For instance, the paper refers to the "negative log-likelihood (NLL) of each response oi conditioned on x,"(Section 2.2) but 'x' is not defined in the context.

### Questions
1. Regarding the knowledge retrieval process, how do you ensure the correctness of the knowledge recalled from the KB? Given that embedding-based matching can introduce noise, have you conducted any analysis on the impact of incorrectly retrieved or irrelevant knowledge on the RE calculation? How robust is the diagnosis mechanism to this noise?
2. In the Section 2.2, you refer to "The first situation suggests L might not grasp this knowledge or cannot properly apply this knowledge to problem-solving, while the second situation indicates that L does understand this knowledge but is easily misled by it.", providing a specific interpretation for two scenarios of knowledge impact. Can you provide the reasoning behind the claim that when knowledge has a negative impact (misleading), it indicates "L does understand this knowledge but is easily misled by it"? Why does this scenario not also suggest a failure to properly integrate or contextualize new information, which could be seen as a form of "not grasping" the knowledge in a given context?
3. Can you elaborate on the sensitivity of LaMer to the quality of its external components? Specifically, how would performance degrade if a less comprehensive knowledge base were used, or if a weaker, open-source model was used for data synthesis instead of ChatGPT?
4. Regarding the RE thresholds in Table 1, have you performed any experiments to analyze their sensitivity? How were these specific values (0.1, 0.4, 0.7, 1.0) determined, and how critical are they to the overall performance of the curricular remedy strategy?

### Soundness
3

### Presentation
2

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
They propose a method to first find the knowledge deficiency in LLMs which, based on the literature, is the source of mistakes in reasoning. To do so they use user queries and using an external knowledge base, they get the relevant info for each query. Then using relative entropy, they detect the knowledge deficiency. Afterwards, using curricular meaningful learning, they propose a method to remedy this knowledge deficiency. This method is label free and does not require human supervision.
They show that this method achieves comparable results to baselines with only 40% training data.

### Strengths
No need for human annotation or labels.
Rich experiments
Clearly written and easy to understand

### Weaknesses
This method is limited to adding knowledge from GenericsKB.
They do not show how much of knowledge in GenericsKB are actually missing in the LLM they study (maybe for some of them the knowledge is already there but fine tuning only makes that knowledge sharp.)
Heavy reliance on ChatGPT (if ChatGPT makes errors, their method will as well).

### Questions
1. in the paper you mention that: "Subsequently, we adopt ChatGPT (Achiam et al., 2023) to synthesize the specified number of examples for the deficiencies in each group.". How do you make sure that chatGPT has enough and correct information for that knowledge?
2. why in Gemma-1.1 your method beats others in most of the cases but with Qwen2, it does not?
3. "We only keep the examples that possess valid answers for evaluation." what percentage of answers did you throw away? and how do you define “valid” here?
4. "we use ChatGPT to generate m= 4 pieces of knowledge for GSM8K". What if ChatGPT makes a mistake? how do you guarantee ChatGPT is correct?
5. Are you sure numbers is Table 6 are correct?
6. "Finally, we synthesize 3,750 examples to enhance Mistral, Qwen2, and Gemma-1.1, while 1,250 examples are synthesized to enhance LLaMA-3 due to denser knowledge in it.". Why do you use different numbers for LLaMA-3?
7. "Therefore, Single enhances Mistral, Qwen2, and Gemma-1.1 with 1,500 examples, and it utilizes 600 examples to enhance LLaMA-3.". Is it fair to use different number of examples for LLaMA?
8. "Naive and Single could supplement some knowledge to them but cause them to forget more useful knowledge." Your method also supplies some knowledge but it does not hurt the numbers. why single hurts? how do you select the example for single one?
9. In section 3.6, item (2), you claim that “Naive and Single could supplement some knowledge to them but cause them to forget more useful knowledge.”. why this forgetting does not happen in LaMer?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a label-free framework for identifying and improving knowledge gaps in large language models (LLMs), which does not rely on costly human annotations. Specifically, the authors propose the relative-entropy-based diagnostic method that quantifies how much additional information external knowledge contributes to the LLM, which leads to detecting the areas of weakness. Then, the authors design the remedy process based on the curricular learning: synthesizing examples in proportion to deficiency severity and training the model from easier to harder deficiencies. The authors validate the proposed approach on four LLMs and multiple reasoning benchmarks, showing that it achieves consistent performance gains across them.

### Strengths
* The processes to identify and remedy deficiencies in LLMs are convincing. 
* The proposed approach clearly outperforms existing relevant baselines.

### Weaknesses
* In extracting the knowledge (needed to check and remedy deficiencies in LLMs), the assumption that, for each query, there should be relevant knowledge from an external knowledge base is very strong. In other words, what if the external knowledge base does not contain the relevant knowledge for each query? Additionally, the process of checking and remedying knowledge deficiencies can be done only for knowledge within the knowledge base, which seems a clear limitation of the proposed approach. Lastly, I am a bit confused whether the proposed approach is truly label-free: it requires the knowledge that is related and associated with the query, which may be considered as the label for the query. 
* There are relevant papers [A, B, C] that the authors should discuss and potentially compare with, especially [A] (which seems highly relevant). 
* The authors could more explicitly justify the advantage of the proposed approach over the unsupervised learning (or SFT) with experiments (i.e., the current setup does not fully justify its advantage over them, despite the claims in the paper). For example, the authors could train the LLMs with all the knowledge in the whole knowledge base and compare the proposed approach against it (i.e., the unsupervised setup) in both effectiveness and efficiency. 
* The performance of the baseline approaches on the Gemma-1.1 (2B) is inferior to the most basic setup (called Base), which may warrant more discussions. 

---

[A] Structural Entropy Guided Agent for Detecting and Repairing Knowledge Deficiencies in LLMs, 2025.

[B] R-Zero: Self-Evolving Reasoning LLM from Zero Data, 2025.

[C] Self-Error-Instruct: Generalizing from Errors for LLMs Mathematical Reasoning, 2025.

### Questions
Please see Weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3
