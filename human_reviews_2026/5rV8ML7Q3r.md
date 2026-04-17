# LLM as an Algorithmist: Enhancing Anomaly Detectors via Programmatic Synthesis

- Decision: Accept (Poster)
- Scores: 2, 4, 6, 6

## Abstract
Existing anomaly detection (AD) methods for tabular data usually rely on some assumptions about anomaly patterns, leading to inconsistent performance in real-world scenarios. While Large Language Models (LLMs) show remarkable reasoning capabilities, their direct application to tabular AD is impeded by fundamental challenges, including difficulties in processing heterogeneous data and significant privacy risks. To address these limitations, we propose LLM-DAS, a novel framework that repositions the LLM from a data processor to an algorithmist. Instead of being exposed to raw data, our framework leverages the LLM's ability to reason about algorithms. It analyzes a high-level description of a given detector to understand its intrinsic weaknesses and then generates detector-specific, data-agnostic Python code to synthesize ``hard-to-detect'' anomalies that exploit these vulnerabilities. This generated synthesis program, which is reusable across diverse datasets, is then instantiated to augment training data, systematically enhancing the detector's robustness by transforming the problem into a more discriminative two-class classification task. Extensive experiments on 36 TAD benchmarks show that LLM-DAS consistently boosts the performance of mainstream detectors. By bridging LLM reasoning with classic AD algorithms via programmatic synthesis, LLM-DAS offers a scalable, effective, and privacy-preserving approach to patching the logical blind spots of existing detectors.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces the use of large language models (LLMs) to augment training datasets, claiming that LLMs can synthesize additional anomalous samples. However, traditional methods for tabular anomaly detection primarily follow the one-class classification paradigm, where detectors are trained solely on normal data. By training detectors on a combination of normal samples and LLM-generated anomalous samples, the authors report significant performance improvements across various datasets and detection methods.

### Strengths
1. The figures, pipelines, and writing are easy to understand.
2. The paper tackles an interesting direction: using LLMs to augment the training dataset of anomaly detection in the tabular dataset, potentially as a way to alleviate the strong assumption posed by the existing detectors.

### Weaknesses
- The paper only considers general anomaly detection. However, there are multiple types of anomalies—such as point anomalies, contextual anomalies, and others—that differ in nature and characteristics. The proposed method does not take this diversity into account.

- Delegating the anomaly generation process to LLMs raises concerns. How can we ensure that the generated samples are indeed anomalous? More importantly, how is an anomaly formally defined in this context? Although the authors provide some visualizations in Figure 5, a more thorough discussion of what constitutes an anomaly point is necessary.

- It remains unclear how to evaluate the hardness of the generated anomaly samples. Can the LLMs generate anomalies with varying degrees of difficulty?

- The potential side effects of these synthetic samples are not discussed. For instance, could they negatively affect the generalizability of the detector, given that LLMs might exhibit biases toward certain types of anomalies and thus overrepresent them?

- Experiments: The ratio of blended synthetic anomaly samples is insufficiently studied. An ablation study examining the optimal proportion of anomaly samples needed to improve robustness would strengthen the empirical analysis.

### Questions
> ...This leads to tabular anomaly detection (TAD) usually being studied under a one-class classification paradigm, where the detector is trained solely on normal samples to distinguish anomalies at test time...

I have great concern about this argument. Other well-defined AD methods in the general domain, such as Self-Inf [1] can be easily applied to the tabular dataset. But Self-Inf is apparently not a one-class classification method. Following this concern, I would doubt whether the challenge of 'one-class classification' can well support your motivation for introducing LLMs to augment the training dataset.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles the task of tabular anomaly detection. The authors propose leveraging a large language model to analyze existing detectors and generate code that synthesizes hard-to-detect anomalies. A binary classification model is then trained and integrated with the existing detectors to perform anomaly detection. Extensive experimental results demonstrate the superiority of the proposed method.

### Strengths
1.	This paper is well written and easy to read.
2.	The proposed method sounds reasonable.
3.	The authors conduct extensive experiments to demonstrate the effectiveness of the proposed method.

### Weaknesses
1.	The paper employs an LLM to generate code for synthesizing pseudo-anomalies. However, it is unclear how the authors ensure the validity and reliability of the LLM’s outputs. The paper should include an explanation or verification procedure for this aspect.
2.	The proposed method utilizes the RandomForest as the binary classification model, the authors are expected to conduct experiments to analysis the robustness of the proposed method to various binary classifiers.
3.	The authors generate pseudo-anomalies for only 10% of the training set. It is unclear why this specific ratio was chosen. Moreover, the imbalance between normal and anomalous samples might affect the training of the binary classifier. Would using a balanced dataset improve the classifier’s performance?

### Questions
1.	How does the binary classifier perform if it is used alone for anomaly detection?
2.	Since the LLM is used to analyze the characteristics of existing anomaly detectors and generate “hard-to-detect” anomalies, it would be interesting to investigate whether the LLM’s analysis could also be leveraged to design detector-specific binary classifiers. This might further enhance the adaptability and effectiveness of the proposed framework.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces LLM-DAS, a framework that leverages Large Language Models to enhance tabular anomaly detection methods by generating detector-specific synthetic anomalies. Rather than using LLMs to directly process data, the approach positions LLMs as "algorithmists" that analyze detector mechanisms and generate reusable Python code for synthesizing "hard-to-detect" anomalies. These synthetic anomalies are used to augment training data, transforming the one-class classification problem into a more discriminative two-class setting. The method is evaluated on 36 benchmark datasets with five different base detectors.

### Strengths
1 - The repositioning of LLMs from data processors to algorithm strategists is novel and well-motivated, effectively addressing both privacy concerns and the inherent difficulties LLMs face with heterogeneous tabular data while leveraging their strengths in code generation and algorithmic reasoning.

2 - The framework demonstrates strong empirical results across 36 diverse datasets, showing consistent improvements over base detectors and outperforming recent LLM-based approaches like AnoLLM while being more computationally efficient.

3 - The approach is practical and scalable - the detector-specific synthesis code is generated only once per detector type and can be reused across multiple datasets, making it cost-effective compared to approaches requiring LLM fine-tuning or repeated querying.

### Weaknesses
1 - The paper lacks theoretical analysis or formal guarantees about why the generated "hard" anomalies should improve detector performance, relying primarily on empirical validation without providing theoretical bounds or convergence properties of the enhanced detectors.

2 - The synthesis strategies appear relatively simplistic (e.g., extrapolation along vectors for IForest), and it's unclear whether the LLM genuinely discovers novel algorithmic insights or simply implements predetermined heuristics that could have been hand-coded without LLM involvement.

3 - The evaluation is limited to classical anomaly detectors and doesn't explore integration with modern deep learning-based methods, nor does it investigate failure modes where the synthetic anomalies might degrade performance or introduce unintended biases.

4 - The paper doesn't adequately address the potential for adversarial exploitation - if the synthesis code is reusable and potentially accessible, malicious actors could use knowledge of the augmentation strategy to craft anomalies that evade the enhanced detectors.

5 - The ablation studies, while comprehensive in number, don't sufficiently isolate the contribution of LLM reasoning versus simpler rule-based approaches - a comparison with expert-designed synthesis strategies for each detector would better validate the value of LLM involvement.

### Questions
Can you provide examples (such as performing a casae study) where the LLM-generated code discovers non-obvious or counterintuitive synthesis strategies that human experts might not have designed?

How sensitive is the approach to the choice of LLM and prompt engineering - would the method still work with smaller, open-source models?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes LLM-DAS, a novel framework to enhance tabular anomaly detectors (TAD). It addresses the "fragile assumptions" of existing detectors and the privacy/usability issues of applying LLMs to tables. The core idea is to reposition the LLM from a "data processor" to an "algorithmist". In a data-agnostic phase, the LLM reasons about a detector's high-level logic to identify weaknesses and generates reusable Python code to synthesize "hard-to-detect" anomalies that exploit these flaws. In a data-specific phase, this code is executed on a dataset to augment the normal-only training data . This converts the one-class problem into a more robust two-class classification task. Extensive experiments on 36 benchmarks show significant performance boosts for mainstream detectors.

### Strengths
1. The central idea of using the LLM for abstract algorithmic reasoning and code generation—rather than data processing—is highly innovative. It leverages the LLM's core strengths and opens a new path for applying LLMs to non-linguistic domains.
2. The data-agnostic code generation phase fundamentally solves the data privacy issue. The resulting code is reusable, making the approach highly scalable: the LLM is queried only once per detector type, not once per dataset.
3. The framework's effectiveness is validated across 36 datasets and 5 diverse detectors. Crucially, ablation studies (Fig. 4b) and cross-detector experiments (Table 2) confirm that the "detector-aware" strategy is the key to success, significantly outperforming generic synthesis methods .

### Weaknesses
1. The framework's success is entirely contingent on the LLM's ability to correctly understand an algorithm's description and identify its true weaknesses . A superficial or incorrect logical analysis by the LLM would lead to a suboptimal synthesis policy.

2. The enhanced classifier is trained to detect anomalies from the LLM's specific synthesis strategy. It is not guaranteed to generalize to all types of "hard" anomalies, but may simply learn to spot the artifacts of those specific generated samples.


3. The final detector is an ensemble (a sum) of the original detector and a new classifier (e.g., RandomForest). This adds complexity and inference cost rather than directly "patching" or retraining the original model itself.

### Questions
1. For trainable detectors (like DRL or Autoencoders), did you consider using the augmented data to fine-tune the original detector itself, rather than adding an external binary classifier?
2. The framework seems to generate one primary synthesis policy per detector. What if a detector has multiple, distinct logical weaknesses? Could the LLM be prompted to find several weaknesses, and would a mixture of synthetic anomalies targeting all of them provide even more robustness?

### Soundness
3

### Presentation
4

### Contribution
3
