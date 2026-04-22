# White-Box Auditing of Large Language Model Unlearning

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Large language models (LLMs) can memorize sensitive information, raising serious privacy concerns. Machine unlearning offers a potential solution to remove such information, but it remains unclear whether existing methods truly erase it or merely hide it within the model. A key challenge is quantifying the persistence of sensitive data under a unified evaluation framework. To address this, we construct a synthetic dataset containing fake personal information and propose a white-box auditing framework to rigorously assess whether claimed-forgotten information is genuinely removed. Using this framework, we evaluate five existing unlearning methods and find that a simple “inverse greedy” decoding—selecting the least likely token at each step—can recover supposedly forgotten personal information. Our results reveal that current unlearning approaches often fail to fully eliminate sensitive information, highlighting the need for more reliable methods to ensure privacy in deployed LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses privacy concerns in large language models (LLMs) by proposing a white-box auditing framework to evaluate whether unlearning methods truly erase sensitive personal information (PI) or merely conceal it. The authors create a synthetic dataset called Fake Personal Information (FPI), containing attributes like year of birth, blood type, postcode, and social insurance number. They further introduce a restricted inverse greedy (RIG) decoding strategy to recover supposedly forgotten PI from the model's logits, in order to audit the unlearning effectiveness.

### Strengths
1. The structure of the paper is clear.
2. It is an important task to correctly audit the current unlearning method.
3. The PI inference task is suitable for unlearning auditing, considering PI are easily verifiable.

### Weaknesses
1. The contributions are limited. The proposed dataset is similar to previous TOFU or SynthPAI. Besides, traditional datasets for PI inference could be easily modified (or even directly used) to unlearning auditing tasks.
2. The paper is a bit over-claimed. Considering the auditing method may be limited to PI datasets (or similar tasks with verifiable answers) and fine-tuning unlearning methods, it is not a unified auditing method.
3. The proposed method is lack of empirical score to comprehensively quantify the ability for a given unlearning method. Besides, the idea itself lacks stronger (even theoretical guarantee), and it is intuitive for rigorous auditing task.
4. The key finding that existing unlearning methods can not robustly erase the target knowledge is not a new conclusion, many recent works have reveal this point. However, the authors fail to fully discuss and compare with them.

### Questions
1. What are the differences or novelty of the proposed dataset, compared with previous works or tradition PI datasets?
2. Can the auditing method be applied to broader tasks (e.g., open-ended generation), or other types unlearning methods?
3. The proposed auditing method needs stronger guarantee.
4. What are new findings compared to previous works, e.g., Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies whether current unlearning methods for large language models could erase sensitive personal information (PI) or merely suppress it. The authors propose a white-box auditing framework and introduce a synthetic Fake Personal Information (FPI) dataset that covers four attribute types (year of birth, blood type, postcode, social insurance number). They design restricted inverse greedy (RIG) decoding, which selects the least likely token from an attribute-specific restricted candidate set, to probe whether information targeted by finetuning-based unlearning remains latent in model logits.

### Strengths
1. The research problem is clear and timely. White-box auditing of PI unlearning aligns with real regulatory needs (e.g., GDPR) and provides a rigorous test of actual erasure vs suppression.
2.  FPI spans heterogeneous attribute types (numeric, categorical, structured sequences) and defines tailored metrics, enabling nuanced evaluation.

### Weaknesses
1. Experiments use only DeepSeek-7B with LoRA; robustness across model architectures and sizes is not assessed.
2. No comparison to retraining-from-scratch to quantify the ideal unlearning behavior and establish an upper bound on true erasure.
3. Previous unlearning audit methods often employed approaches with statistical guarantees, such as membership inference attacks, whereas the method proposed in this paper appears intuitive.

### Questions
- Can the authors replicate on additional open LLMs (e.g., Llama-3, Mistral, Qwen) and different sizes to test generality?
- How does the proposed audit method work on retraining, and does it produce false positives?
- Does the proposed method have theoretical guarantees to ensure the rigor of the audit?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a white-box auditing framework to verify whether personal information claimed to be forgotten through unlearning methods is genuinely removed. The authors construct a synthetic dataset (FPI) containing fake personal information and construct a Restricted Inverse Greedy (RIG) decoding strategy to recover supposedly forgotten information. The experimental results reveal that current unlearning approaches often fail to fully eliminate sensitive information, with recovery rates reaching up to 97% for some attributes.

### Strengths
1) The paper is well-written with clear motivation, methodology, and results. 
2) The proposed RIG decoding approach seems simple yet effective.
3) Unlearning auditing is a critical privacy concern in deployed LLMs.

### Weaknesses
1) The entire evaluation is conducted on artificially generated fake personal information. Real-world personal data often has different statistical properties, correlations, and contextual dependencies that may affect both unlearning effectiveness and recovery difficulty. The generalizability of findings to real privacy-sensitive scenarios remains unclear.
2) The paper dismisses prompt-based and decoding-based unlearning methods too quickly, focusing exclusively on finetuning-based approaches, which significantly narrows the contribution. 
3) The paper provides simple explanations for why RIG works but lacks formal theoretical analysis. Why does gradient ascent necessarily push information into low-probability regions? Under what conditions will RIG succeed or fail? What are the theoretical guarantees or limitations?
4) It is insufficient to make evaluations on a single model.

### Questions
1) Can you provide formal theoretical analysis or guarantees for when and why RIG is effective? Specifically, under what loss landscapes or model properties will gradient ascent-based unlearning push information into low-probability regions that RIG can exploit?
2) Can the proposed auditing method work on different types of unlearning methods? 
3) Regarding real-world scenarios, can the proposed dataset simulate different statistical settings (such as bias, where a personal information dataset may exhibit a long tail on certain features, and where forgetting high-frequency and low-frequency features presents differing challenges)? Additionally, can the auditing method be robustly applied to real-world considerations?

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
4

### Summary
This paper proposes a white-box auditing framework to evaluate whether personal information unlearning in LLMs is genuine or superficial. The authors build a synthetic Fake Personal Information dataset and design a novel Restricted Inverse Greedy (RIG) decoding strategy that explores low-probability logits to recover supposedly erased information. Experiments on DeepSeek-7B with five popular unlearning algorithms (GA, GD, GA+KL, PO, NPO) show that up to 97% of the forgotten content can still be recovered, revealing that most current unlearning methods fail to fully remove sensitive data and instead merely suppress its surface accessibility.

### Strengths
1. The paper offers a white-box auditing method that goes beyond surface-level black-box evaluations of unlearning.


2. The proposed RIG/RG decoding provides a simple way of revealing residual memorization hidden in low-probability regions.


3. Experiments on the FPI dataset are systematic, showing evidence that current fine-tuning–based unlearning methods often only suppress, rather than erase, sensitive information.

### Weaknesses
1. The proposed RIG auditing method relies heavily on the design of the restricted candidate set (e.g., top-k or probability threshold), making its results highly sensitive to hyperparameter choices. Small variations in these values can cause large differences in recovery performance, raising concerns about robustness and reproducibility.


2. The approach is mainly effective for low-entropy or format-constrained information (such as numbers, postcodes, or IDs). It is unclear how well it would generalize to unlearning of open-ended or semantically rich knowledge, where answers are expressed in natural language rather than within a fixed token pattern.


3. The evaluation framework is built entirely on a synthetic dataset with clearly defined personal information fields, which simplifies the task but limits realism. The lack of experiments on more diverse or real-world data makes it difficult to assess how the method performs under complex or noisy distributions.


4. The method is diagnostic rather than principled. It does not provide a formal definition or guarantee of what constitutes successful unlearning, nor does it explore theoretical foundations for why RIG should correspond to genuine memory recovery. As a result, while the findings are empirically interesting, their broader implications remain uncertain.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
