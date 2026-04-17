# RedacBench: Can AI Erase Your Secrets?

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 4

## Abstract
Modern language models can readily extract sensitive information from unstructured text, making redaction—the selective removal of such information—critical for data security. However, existing benchmarks for redaction typically focus on predefined categories of data such as personally identifiable information (PII) or evaluate specific techniques like masking. To address this limitation, we introduce RedacBench, a comprehensive benchmark for evaluating policy-conditioned redaction across domains and strategies.
Constructed from 514 human-authored texts spanning individual, corporate, and government sources, paired with 187 security policies, RedacBench measures a model's ability to selectively remove policy-violating information while preserving the original semantics.
We quantify performance using 8,053 annotated propositions that capture all inferable information in each text. This enables assessment of both security—the removal of sensitive propositions—and utility—the preservation of non-sensitive propositions.
Experiments across multiple redaction strategies and state-of-the-art language models show that while more advanced models can improve security, preserving utility remains a challenge. To facilitate future research, we release RedacBench along with a web-based playground for dataset customization and evaluation. Available at https://hyunjunian.github.io/redaction-playground/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors introduce RedacBench, a novel benchmark for evaluating redaction capabilities. Specifically, through 514 human-written texts and 187 security policies, it measures a model’s ability to redact diverse forms of sensitive information embedded in text. The authors conduct experiments on 9 language models with different redaction strategies, highlighting that effective redaction comes as a tradeoff between security and utility.

### Strengths
- The paper addresses a timely problem, is clearly written, and well-structured.
- The introduced dataset is valuable; the use of propositions effectively supports redaction evaluation and allows easy scalability with new policies.
- The playground is useful for exploring and understanding how the benchmark operates.

### Weaknesses
- Some redaction strategies are prone to hallucinations, with no mechanism provided to detect or account for them.
- No comparison with related work; RedacBench performance is not contextualized against existing studies.

### Questions
- Although the authors acknowledge hallucinations among the method’s limitations, they associate them solely with cases in which the model has been trained on the document to be redacted, and might therefore recall the redacted information.
My primary concern, however, is that hallucinations may occur regardless of whether the model is already familiar with the text. With adversarial redaction, a language model identifies sensitive information and rewrites the text accordingly. Regardless of whether the model has been previously exposed to this sample, a hallucination may still occur and compromise the rewritten text.
    
    For example, in the first redaction samples reported in Appendix B, we have:
    
    **Original Text:**
    
    *“Bob, regarding Patti Sullivan’s contributions to the west desk this year […]”*
    
    **Adversarial Redaction 1 Text:**
    
    *“Bob, regarding the employee’s contributions to the team this year […]”*
    
    However, a model could also redact that information by, for instance, replacing the employee’s name with another one:
    
    *“Bob, regarding Alice’s contributions to the team this year […]”*
    
    Although this could be considered secure (as a true negative), it may still affect the utility of the text if the meaning changes. This type of issue is not captured by the Utility Score proposed by the authors.
    
    My question, therefore, is the following:
    
    Have you observed this kind of phenomenon with Adversarial Redaction? If so, how frequently does it occur? And how could the Utility Score be modified to account for these types of failures in the redaction process?

- There is no analysis that relates the models’ performance on RedacBench to those reported in other studies. Moreover, in the Related Work section, there is no explanation as to why the cited works cannot be directly compared with the approach introduced by the authors. For instance, a comparison could have been made with the Adaptive PII Mitigation Framework regarding the redaction of specific PII data. Furthermore, for example, it would have been interesting to evaluate the performance of RedacBuster after applying the various redaction techniques proposed in RedacBench, in order to assess, for example, the robustness of the solutions with high Security or Utility Scores.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a benchmark for evaluating text redaction methods. The benchmark works by using an LLM autograder to determine whether key units of factual information from the unredacted text could be inferred from the redacted text alone. Data is collected from three varied sources (real human data). A human-in-the-loop approach is used, along with an LLM, to extract key factual units from the unredacted texts, along with privacy policies that this information may or may not violate. The autograder shows less than a 2% false-negative rate, making it fairly appropriate for evaluation of information leakage. Evaluations are conducted on frontier API-based LLMs, showing a Pareto frontier on redacted text informativeness and information leakage, with Claude 4 and the Adversarial Redaction (Staab et al, 2025) method performing best.

### Strengths
- Very important: The benchmark for redaction methods appears to be reasonably constructed, and I believe it will be useful for the privacy community.
- Important: The shift toward inference-based measures of privacy is very valuable. Using carefully validated LLM graders seems like the right approach for this.
- Important: A severe tradeoff between utility and security of current SOTA models/methods suggests that there is significant room for improvement of redaction methods on the benchmark.
- Of some importance: The web app also provided alongside the dataset is an interesting artifact and will hopefully allow for researchers to quickly prototype new ideas.

### Weaknesses
- Important: I’m struggling to get a sense of what optimal performance might look like on the dataset. Do you have examples that can be perfectly redacted by a human? It would help to know what the ceiling performance is on the benchmark and how far models are from this performance. Otherwise it may not be clear to the community how long to work on this benchmark, when it is saturated, etc.
- Important: I couldn’t see any detail to who the humans were in the human-in-the-loop data construction process. More detail on this is important.
- Of some importance: I was slightly concerned by two experiment design choices: using a weaker model grader than the redaction model, in some cases, (gpt-5 vs. gpt-4.1-mini), and the lack of ablation across redaction model and grader model families (grader models may unfairly favor their own redaction models efforts).

### Questions
Please feel free to respond to the questions in the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper makes an important contribution by introducing RedacBench, a comprehensive benchmark for the nuanced task of policy-based text redaction. Its core strengths lie in the novel proposition-based evaluation framework and the high-quality, diverse dataset constructed using a human-in-the-loop methodology. The experimental results clearly demonstrate the critical trade-off between security and utility, providing strong baselines for future work. However, the evaluation framework's reliance on an LLM (GPT-4.1-mini) as an automated evaluator introduces a potential risk of data contamination. Furthermore, the experiments primarily rely on closed-source APIs, which raises significant security and privacy concerns for real-world deployment.

### Strengths
- The work moves beyond simple PII detection, formulating a more realistic task of context-sensitive redaction based on specific security policies. The inclusion of 187 multi-layered policies, spanning from granular details to high-level abstract concepts, aligns well with practical requirements.
- The Security Score and Utility Score offer a clear, quantitative, and interpretable method for measuring the inherent trade-off in redaction, which is effectively visualized in the results. The authors establish a reasonable baseline for the evaluator's reliability by testing its false negative rate on known true propositions, achieving a low error rate of 1.45%.
- The benchmark dataset is constructed from diverse and challenging sources, including texts from individual, corporate (Enron), and government (Hillary Clinton) domains. This diversity ensures models are tested across a wide range of real-world scenarios. The public release of the benchmark and the accompanying interactive web playground (Appendix A) is a valuable contribution to the community, lowering the barrier for future research and enabling others to build upon this work.

### Weaknesses
- The evaluation framework relies on GPT-4.1-mini as an automated judge, which introduces a risk of "recall" from pre-training data contamination. Even if information is successfully redacted, the evaluator might incorrectly assess it as "preserved" (a false positive), thus artificially deflating the Security Score. The validation only checks the false negative rate and fails to assess the more critical false positive rate, posing a substantial threat to the validity of the reported scores.
- The experiments (Table 3) heavily rely on closed-source models (e.g., GPT-5, GPT-4.1, Claude-Sonnet-4). Sending sensitive text to third-party APIs is operationally infeasible in real-world applications and presents a significant data leakage risk. The paper fails to evaluate the performance of state-of-the-art, locally-deployable open-source models on RedacBench, leaving a gap for organizations that cannot use external APIs to assess practical trade-offs.
- The framework treats every proposition with equal weight, ignoring the differential risk associated with various policies. In a practical scenario, failing to redact a macro-level violation (e.g., "Strategic business plan") is far more severe than a micro-level one (e.g., "Instructor names"). The simple binary sensitive/non-sensitive classification and unweighted scoring risk evaluation distortion. This represents a missed opportunity to leverage the multi-layered policy information (Table 1) for a risk-weighted, granular metric design that would better reflect real-world priorities.

### Questions
- The evaluation framework relies on GPT-4.1-mini as an automated judge, which introduces a risk of "recall" from pre-training data contamination. Even if information is successfully redacted, the evaluator might incorrectly assess it as "preserved" (a false positive), thus artificially deflating the Security Score. The validation only checks the false negative rate and fails to assess the more critical false positive rate, posing a substantial threat to the validity of the reported scores.
- The experiments (Table 3) heavily rely on closed-source models (e.g., GPT-5, GPT-4.1, Claude-Sonnet-4). Sending sensitive text to third-party APIs is operationally infeasible in real-world applications and presents a significant data leakage risk. The paper fails to evaluate the performance of state-of-the-art, locally-deployable open-source models on RedacBench, leaving a gap for organizations that cannot use external APIs to assess practical trade-offs.
- The framework treats every proposition with equal weight, ignoring the differential risk associated with various policies. In a practical scenario, failing to redact a macro-level violation (e.g., "Strategic business plan") is far more severe than a micro-level one (e.g., "Instructor names"). The simple binary sensitive/non-sensitive classification and unweighted scoring risk evaluation distortion. This represents a missed opportunity to leverage the multi-layered policy information (Table 1) for a risk-weighted, granular metric design that would better reflect real-world priorities.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
- This work introduces RedacBench, a benchmark for evaluating LLM-based text redaction. 
- It provides a standardized framework to quantitatively assess the trade-off between security and utility, offering an objective basis for comparing diverse redaction techniques. 
- RedacBench lays the groundwork for trustworthy AI systems capable of securely managing and erasing sensitive information.

### Strengths
- The paper introduces a fine-grained, proposition-level analysis that captures semantic inferability of information, enabling more rigorous and interpretable quantification of redaction effectiveness than surface-level token or entity matching.
- Defines complementary metrics — Security Score (true negative rate for sensitive information) and Utility Score (true positive rate for non-sensitive information) — allowing quantitative assessment of both privacy protection and information preservation.
- Curates multiple human-written documents and 187 distinct policies, annotated into over 8,000 atomic propositions. This design supports fine-grained, cross-domain testing and reflects realistic privacy constraints
- Evaluates multiple redaction strategies (e.g., masking, rewriting, policy-based prompting) across various state-of-the-art LLMs, establishing meaningful baselines and empirically demonstrating the trade-off between model capability, security, and text utility.

### Weaknesses
- The benchmark evaluates models in a controlled, static setting, but it does not test interactive or context-evolving scenarios where redaction systems must operate dynamically (e.g., during live conversations or document editing).
- Although the paper positions redaction as a privacy defense, it does not directly compare or align its evaluation metrics with other privacy frameworks like unlearning, membership inference resistance.
- The current benchmark quantifies empirical removal, not privacy guarantees.
- The binary classification of propositions as “removed” or “preserved” may not capture partial or implicit leakage — cases where sensitive information is paraphrased, entailed, or inferable through context reconstruction.
- The paper focuses on improving redaction accuracy but does not address the ethical and practical risks of over-redaction, such as erasing legitimate public-interest information or introducing bias in released datasets.

### Questions
- Can your evaluation account for implicit leakage — cases where sensitive information is not directly stated but can still be inferred from contextual cues or correlated details?
- How does RedacBench relate to or complement benchmarks used for evaluating unlearning, differential privacy, or content filtering methods?
- The paper defines utility as the preservation of non-sensitive propositions. Have you considered complementary measures of semantic consistency?

### Soundness
3

### Presentation
3

### Contribution
2
