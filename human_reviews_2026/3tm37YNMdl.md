# CUMath: A Benchmark and Evaluation Framework for LLMs on Mathematical Reasoning in Undergraduate Computational Math

- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Large Language Models (LLMs) perform well on popular math benchmarks but still struggle with fundamental undergraduate tasks such as basic integrals. This suggests a diagnostic gap: existing datasets are either trivial, synthetic, or overly advanced, limiting their usefulness for exposing reasoning failures. To address this, we introduce CUMath, a benchmark of 2,100 real problems from undergraduate courses in Calculus, Linear Algebra, Differential Equations, and related fields. Each problem includes step-by-step solutions, enabling evaluation of both final answers and intermediate reasoning. Moreover, current evaluations treat accuracy and reasoning separately, overlooking their joint role in problem-solving. To address this, we propose a multi-layered evaluation framework that combines automatic metrics with an LLM-as-a-grader pipeline, integrating symbolic encoding and external verification. Using this setup, we evaluate 15 LLMs across various prompting strategies. Our results show that even advanced models often misuse symbolic methods and rely on shortcuts, leading to polished but flawed solutions. Our findings reveal the ongoing issue of inconsistent reasoning, highlighting the need for improved benchmarks, evaluation frameworks, and the development of models with enhanced consistency and reasoning capabilities. The code and data will be available upon publication.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a dataset called CUMath which can be used as a benchmark of 2,100 real problems from undergraduate courses in Calculus, Differential Equations, Discrete Mathematics, Linear Algebra, Multivariable Calculus, Precalculus, and Trigonometry. Each problem includes step-by-step solutions, enabling evaluation of both final answers and intermediate reasoning. It categorize the problems into three answer formats: Free Response (FR), Short Answer (SA), and True/False (TF). 

The motivation behind releasing this dataset stems from the observation that current generation LLMs perfom well on existing popular math benchmark dataset but still they strggle with math reasoning. 

Further, this paper proposes a multi-layered framework to jointly evaluate answer accuracy and the reasoning of the model.  This framework combines automatic metrics with an LLM-as-a-grader pipeline. 

Through the propsed CUMath dataset and evaluation framework, this paper shows that SOTA LLMs make mistakes in the symbolic manipulation and procedural reasoning even when prducing final correct answer.

### Strengths
- The dataset is balanced across sever core subjects of the Maths so that no single subject dominates or remains underrepresented.
- This paper proposes a multi-layered evaluation framework where it combines MathBERT for symbolic encoding, LLM for step-level reasoning assessment, and Wolfram Alpha for answer verification. This pipelines, thus, captures both answer correctness and reasoning quality. 
- The proposed benchmarking dataset in this paper would be valuable towards advancing the SOTA of LLM’s reasoning abilities.
- Sections 6.1 and 6.2 provide interesting insights about LLMs behavior at large when they fail in math reasoning.

### Weaknesses
- The technical novelty of the paper is limited but that is understandable because it is more of a data set contribution paper.
- It will be good to see the quantitative comparison of the proposed Dataset+Eval framework against well known Math reasoning dataset under the same proposed eval framework and the same set of models. This will help readers buy the key selling points of the paper. See my comment in the Questions section also.

### Questions
- In Section 4.1, why use two different notations $\hat{s}_i^j$ and $e_i^j$ for the same thing.
- In Line 228, the quantity $m_k(e_i)$ is not defined.
- In Line 269, it is written that “The encoded steps are passed to an LLM..” I was wondering how do you pass embeddings to an LLM? Can you elaborate? My understanding is that you are passing embeddings obtained from MathBERT (in Step 2) to an LLM in Step 3.
- In Table 2, it will be good if you can also add the performance of these models for some of the popular Math reasoning datasets but using your eval framework. This will help readers buy the points that your trying to drive home.

### Soundness
2

### Presentation
3

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
The paper aims to address two core issues in evaluating the mathematical reasoning capabilities of Large Language Models (LLMs) at the undergraduate level: (1) existing benchmarks are either too elementary or too advanced, lacking diagnostic value for reasoning failures, and (2) current evaluation paradigms often decouple final answer accuracy from the quality of the reasoning process. To this end, the authors introduce CUMath, a new benchmark of 2,100 problems from real undergraduate computational math courses, with each problem annotated with step-by-step solutions. Concurrently, they propose a multi-layered evaluation framework that combines automatic metrics with an "LLM-as-a-grader" pipeline, which is augmented with external tools like MathBERT and Wolfram Alpha for verification. Through an evaluation of 15 LLMs, the authors conclude that even frontier models exhibit systematic errors in symbolic manipulation and procedural reasoning, arguing for the necessity of integrated evaluations that assess both reasoning validity and answer correctness.

### Strengths
The paper accurately identifies a critical gap in the current landscape of LLM evaluation. As models approach saturation on benchmarks like GSM8K, there is a pressing need for more challenging, realistic, and diagnostically useful benchmarks. The focus on undergraduate mathematics is an excellent choice for a domain that can effectively distinguish between superficial fluency and deep reasoning abilities.
The construction of the CUMath dataset is a solid and meaningful effort. Its grounding in authentic instructional materials (quizzes, exams, textbooks) ensures the practical relevance of the problems. Most importantly, providing detailed step-by-step solutions is crucial for enabling fine-grained analysis of where models' reasoning chains fail, which will greatly benefit future research in this area.

The authors' advocacy for an integrated assessment of "answer correctness" and "reasoning quality" is insightful. Highlighting the evaluation blind spot of "correct answers derived from flawed reasoning" demonstrates a deep understanding of the limitations of current LLM evaluation. The conceptual direction of the proposed multi-layered framework, which attempts to synthesize automated metrics with qualitative LLM-based feedback, is both correct and worthy of exploration.

### Weaknesses
The authors repeatedly claim to evaluate "state-of-the-art LLMs" or "frontier LLMs." However, the list of evaluated models (Table 12) primarily consists of older models such as GPT-3.5, an early version of GPT-4.1, and smaller-scale open-source models (e.g., LLaMA 3 8B/70B). Given the rapid pace of development in the field (and a target publication date of ICLR 2026), these models are no longer representative of the cutting edge. More recent and powerful reasoning models, such as the latest GPT and Claude series or other specialized math models, are conspicuously absent. This outdated selection invalidates the paper's main conclusion that "even the strongest LLMs achieve an accuracy of less than 25%." A rigorous claim about the capabilities of "frontier models" must be substantiated by testing the models widely considered to be the most capable at the time of submission. Without such experiments, the observed failures could be limitations of the specific models tested rather than a general bottleneck for all LLMs.
While the concept of a "multi-layered evaluation framework" is appealing, its components are largely direct applications or combinations of existing work (e.g., SRS from ROSCOE, VR from ReasonEval). The main claimed novelty, the "LLM-as-a-grader" pipeline, lacks the most critical piece of validation: there is no quantitative analysis comparing its outputs to those of human experts. A reliable automated grading system must demonstrate high inter-rater reliability (e.g., using Cohen's Kappa or Krippendorff's Alpha) with human graders. Without this evidence, the reliability and fairness of the LLM grader cannot be trusted, rendering the scores it produces (the "LLM" column in Table 2) unsubstantiated.

The framework's reliability is highly dependent on its automated preprocessing modules, especially the "Math Segmentation" component. As described in Section 4.2, this module relies on simple heuristics—looking for explicit "step k" markers and defaulting to "line-based segmentation" otherwise. This approach is extremely brittle when processing the free-form, structurally diverse outputs of LLMs. A single complex reasoning step can span multiple lines, and models may use different delimiters or none at all. Incorrect segmentation leads to cascading errors in all subsequent evaluation steps (e.g., semantic F1, SRS, LLM-as-a-grader). Yet, the paper provides no evaluation of this module's accuracy (e.g., against a human-annotated ground truth) nor does it discuss its fault tolerance or potential impact on the final results. This oversight regarding a core component's robustness casts serious doubt on the entire framework's practical usability.

### Questions
1.	Could you explain the decision to exclude more recent, top-performing models renowned for their mathematical reasoning abilities (e.g., the latest GPT-4 series, Claude 3 series)? Given that your central conclusion is about the upper-bound capabilities of "frontier LLMs," how can this claim be supported by the current selection of models?
2.	Do you have any plans to conduct, or have you already performed, a study comparing the ratings from your "LLM-as-a-grader" pipeline against scores from human mathematics experts? Without such a comparison, how do you ensure the reliability and impartiality of the automated grader, preventing it from being merely a black box?
3.	Have you evaluated the accuracy of the "Math Segmentation" module? What is its error rate on LLM outputs that lack explicit step markers or follow non-standard formatting? How significantly do these potential segmentation errors impact the downstream F1, SRS, and LLM-grader scores?

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
4

### Summary
This paper introduces CUMath, a benchmark to evaluate LLM reasoning in undergraduate computational math. The authors state a diagnostic gap exists, as LLMs struggle with fundamental undergraduate tasks and existing datasets are either trivial, synthetic, or overly advanced. To address this, the authors provide a dataset of 2,100 problems, each with step-by-step solutions for evaluation. The paper also proposes a multi-layered evaluation framework that integrates automatic metrics with an LLM-as-a-grader pipeline. This pipeline uses MathBERT for symbolic encoding and external verification with Wolfram Alpha. The authors' analysis of 15 LLMs shows models misuse symbolic methods and rely on shortcuts, leading to polished but flawed solutions. The findings reveal failure modes, including invalid reasoning leading to correct results, and show accuracy alone is an insufficient measure of mathematical competence.

### Strengths
1.  The paper clearly identifies a diagnostic gap in current math benchmarks (being either trivial or overly advanced) for evaluating LLMs.
2.  The inclusion of detailed step-by-step solutions enables fine-grained evaluation of model reasoning processes.
3. The proposed LLM-as-a-grader framework offers a novel evaluation perspective by using external CAS verification loops.

### Weaknesses
1.  The low accuracy (less than 25% for even the best models) makes meaningful performance comparisons between models difficult, and we still do not have a comprehensive metrics to evaluate the ability of each model.
2. The paper shows a significant divergence between automatic metrics (like Accuracy) and its own LLM-as-a-grader score , and argues the LLM score is more comprehensive. However, this claim is weakened because the paper does not report consistency data between its LLM-as-a-grader framework and human expert scores.

### Questions
1.  What is the agreement (e.g., kappa score) between your LLM-as-a-grader and human experts on a subset of CUMath?
2. Given the divergence between automatic metrics (like Accuracy) and LLM-as-a-grader scores, why should the LLM-grader be trusted as a comprehensive measure without a reported correlation to human expert evaluation?
3. The results table  shows that smaller open-source models (e.g., LLaMA 4 Scout 17B Instruct) achieve scores similar to top-tier models (e.g., OpenAI o3). Does this lack of differentiation suggest the benchmark is not reliably capturing capability differences, or is this an intended finding?

### Soundness
2

### Presentation
3

### Contribution
2
