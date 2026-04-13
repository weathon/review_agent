## Human Reviewer 1

### Summary
Benchmark leakage will significantly impact a fair evaluation of the LLM ability. In this paper, the authors employ Length Normalized Entropy (LNE) to detect data contamination and achieve contamination mitigation evaluation. For the detection, the intuition is that memorized output may share an overlap with the ground truth. By observing the entropy pattern of the LLM on the ground truth, this work detects contaminated samples. This work also proposes a LNE-based method to block the generation of memorized samples.The results show that the methods can achieve excellent detection and mitigation results compared with baselines.

### Strengths
1. **Focus on important problems.** As more and more diverse benchmarks and LLMs are released, the contamination will mislead our understanding of the LLM progress. I acknowledge that both contamination detection and contamination-free evaluation are crucial research problems.
2. **Intriguing motivation**: I admire the motivation discussion in Section 2.

### Weaknesses
1. **Unclear effects of LNE blocking.** Even if the exactly memorized ground truth can be blocked, the impacts of the leaked data points on the LLM performance are not entirely removed. The alternate answers, as mentioned in Section 2, will be inspired by the leaked data points. It is a little strange that the performance after LNE-Blocking will become close to the original uncontaminated model.
2. **Crucial experiments are missing.** As we can never forecast the contamination behaviors, we may expect to employ contamination mitigation evaluation for all LLMs. What about the original model performance or any other uncontaminated LLM performance under LTE blocking?
3. **Realistic testbeds should be evaluated.** I suggest collecting realistic benchmark contamination reported by the LLM development teams instead of focusing on the manually contaminated LLMs. To be honest, training LoRA adapter for 20 epochs will result in a total memorization of the gold truths. The contaminated models are weak subjects for the task of contamination detection. For real cases, the leaked tokens will be mixed up with other riskless tokens and then be trained for a limited number of epochs. In such a case, it definitely leads to a higher chance of false positives for both contamination detection and contamination mitigation evaluation.
4. **Writing quality requires improvements.** 
- In Eq.1, I guess $N$ is equal to $l$, both corresponding to the output length.
- In Eq. 12, what about the meaning of $M_{original}$? The authors should define it.
- The setting for $\beta$ in line 229 is a little beyond comprehension. The postulation is not convincing and requires the support of at least empirical results. Even so, it is still unclear why an even distribution within the range of 0 to 1 is desirable.
- After walking through the whole paper, I am still confused about the identity of $y$ in Eq. 1. Does it correspond to the generated output by the LLM or the ground truth?

### Questions
- The LTE blocking is operated at the beginning of the generation. What's the rationale behind the choice? And how does it impact the contamination mitigation evaluation?
- Self-consistency is a widely adopted strategy for evaluating LLM abilities. Can you demonstrate how to integrate LTE blocking with self-consistency?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
3

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper tackles the critical issue of data contamination in LLMs, where training data unintentionally includes evaluation benchmarks, complicating fair model assessment. The authors propose LNE-Blocking, a two-part framework that first detects contaminated data using Length Normalized Entropy (LNE) and then mitigates contamination effects by dynamically adjusting blocking during text generation. This approach not only provides state-of-the-art detection but also enables efficient and fair evaluation of LLMs, requiring only two inferences for contamination mitigation. Robust across contamination levels, LNE-Blocking offers a practical and effective solution to this growing challenge in LLM development.

### Strengths
- The paper addresses a highly significant topic.
- The writing is well-structured and clearly organized.
- The proposed method demonstrates originality.

### Weaknesses
- **Concerns Regarding the Blocking Method**: While I appreciate the idea that "similar to how humans can rephrase their thoughts when interrupted, LLMs have the potential to generate alternative answers when their default response is blocked," there are practical concerns with the chosen approach for blocking tokens. Specifically, it may lead to issues if unique tokens at certain positions are blocked, making it difficult to generate correct answers. For instance, a long variable name might be split into several consecutive tokens, and blocking a critical unique token in the middle could result in inaccurate code generation.

- **Lack of Descriptions for Key Information**: Certain key terms and settings lack clear definitions, such as "Fixed Blocking 1, 2, and 5," which appear in Figure 1 and throughout the experiments. Further clarification is needed on what these terms specifically refer to.

- **Minor Typos**: There are small typographical issues, such as mismatched quotation marks in Figure 1's caption ("Cont."), which should be addressed.

- **Limited Experimental Evaluation**: It appears that the effectiveness of the DATA CONTAMINATION DETECTION approach was only evaluated on a single model. Additionally, the results suggest that the proposed LNE method outperformed the baseline only under Mild Cont. conditions, with limited improvement. To demonstrate the broader effectiveness of the approach, evaluations on additional models are necessary.

### Questions
1. How does the proposed blocking algorithm ensure that critical, unique tokens essential to answering the question are not blocked?
2. What are the specific definitions of Fixed Blocking 1, 2, and 5?
3. Could the authors provide additional evaluation results on DATA CONTAMINATION DETECTION across a broader range of models?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
2

---

## Human Reviewer 3

### Summary
The article deals with the problem of training set contamination in which one would like to validate that an LLM being evaluated was not trained with a given test sample. The proposed contamination detection method relies on the entropy of output tokens normalized by the output length. Following the detection mechanism the authors propose a method to evaluate strategies for mitigating the contamination techniques. The contamination detection is benchmarked against three competitors proposed in 2023-2024. The contamination mitigation method is benchmarked against one competitor published in 2024. The results show that the performance of the proposed methods is favorable under certain conditions.

### Strengths
*  Benchmarking against very recent models 


* Good motivation

### Weaknesses
* Marginal performance benefit 

Results reported in Tale 1 do not provide a sufficient margin over Min-k. Overall LNE does not significantly outperform Min-k.

In table 2, what about the other bold numbers? Is TED significantly outperforms LNE-Blocking  on CodeGen Mild cont?


* Missing theoretical justification of the contamination detection approach

It is not clear why is the input x necessary when defining LNE if it is not used in Eq 1. 
y is defined as the output and l is defined as its length. Yet the last token in the sequence y_1,...y_N is indexed by N. Both N and l are used in Eq 1 making its interpretation challenging.

According to eq 2 LNE signals when the token probabilities are skewed. Why should LNE work better than Min-k% Prob for detecting contamination?


* Readability and organization can be improved 

It is not clear what "extensive SOTA results" means in the second contribution bullet. Earlier, the authors refer to SOTA performance, implying that the proposed approach is now the best. In this context, the term "extensive" is not clear.

... whether this data has been trained by the model M. --> whether this data was used to train model M. OR .. whether model M was trained using this data.

Data and subject models are described three times in three different places: Sections 4.1 4.2, Section 5.1 first par. Section 5.2.1 first par

The explanations of the baselines on page 6 lack accuracy and important details. For example, it is stated that Min-k% Prob computes the probability of k% least probable tokens, but (1) it is not written that the low total prob indicates contamination, (2) the relationship between the reference answer and the prompt mentioned in that statement is not clear. The same goes for perplexity and CDD.

The related work section is expected to elaborate on the main competitors. Currently, they are missing from section 6.  When they are added the section should appear earlier in the paper before the baselines are actually used.



*  Benchmark is based on an unvalidated assumption that LLMs were not contaminated by the HumanEval dataset (line 269). 

* Missing some experimental details

It is not clear how the thresholds were chosen for all methods for computing F1 in Table 1. Unfairly chosen thresholds can bias the results. Luckily this problem does not affect AUC.

### Questions
Please provide a theoretical justification explaining why LNE should work better than Min-k% Prob for detecting contamination.


Why is contamination detection tested only on CodeLlama and not all four models?

### Soundness
3

### Presentation
1

### Contribution
2

### Rating
3

### Confidence
2