# VAULT: Vigilant Adversarial Updates via LLM-Driven Retrieval-Augmented Generation for NLI

- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
We introduce VAULT, a fully automated adversarial RAG pipeline that systematically uncovers and remedies weaknesses in NLI models through three stages: retrieval, adversarial generation, and iterative retraining. First, we perform balanced few‑shot retrieval by embedding premises with both semantic (BGE) and lexical (BM25) similarity. Next, we assemble these contexts into LLM prompts to generate adversarial hypotheses, which are then validated by an LLM ensemble for label fidelity. Finally, the validated adversarial examples are injected back into the training set at increasing mixing ratios, progressively fortifying a zero‑shot target NLI model. On standard benchmarks, VAULT elevates RoBERTa‑base accuracy from 88.48% to 92.60% on SNLI (+4.12%), from 75.04% to 80.95% on ANLI (+5.91%), and from 54.67% to 71.99% on MultiNLI (+17.32%). It also consistently outperforms prior in‑context adversarial methods by up to 2.0% across datasets. By automating high‑quality adversarial data curation at scale, VAULT enables rapid, human‑independent robustness improvements in NLI inference tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces VAULT, an automated pipeline for generating and validating adversarial examples to improve natural language inference (NLI) models. 
The approach integrates retrieval, LLM generation, automatic validation, and iterative retraining in a single loop. 
By generating challenging examples that existing models misclassify and filtering them through multiple LLM “judges,” the method aims to produce high-quality adversarial training data without human intervention. 
Experiments on SNLI, ANLI, and MultiNLI show that VAULT consistently improves model performance, achieving notable accuracy gains while using far fewer examples than previous large-scale synthetic datasets.

### Strengths
1. The paper presents a clear and coherent framework that combines retrieval, adversarial generation, validation, and retraining in a single automated loop. The methodology is easy to follow, and the ablation studies are detailed enough to support the main claims.

2. The use of both semantic and lexical retrieval provides a good balance between relevance and diversity, leading to more effective adversarial examples.

3. Empirical results are strong across multiple benchmarks, showing consistent improvements while using far fewer examples than large-scale synthetic datasets.

### Weaknesses
1. The unanimous validation rule may filter out useful but ambiguous examples, and there is no human evaluation to verify the accuracy of the validated data.

2. While comparisons to large synthetic datasets are provided, stronger baselines such as recent adversarial training or contrastive learning methods are missing.

3. The paper could include a deeper analysis of what types of reasoning errors are actually corrected by the proposed approach, to clarify the nature of the robustness gains.

4. This idea of having LLMs generate challenging examples and gradually incorporating them into training is not new; it’s just the first time it has been applied to NLI.

### Questions
1. Please report generation + validation cost (GPU hours) for producing the ~6–6.6k validated examples per strategy, and the end-to-end training time. How does cost scale with T iterations and k shots?

2. Do the three LLM judges ever agree yet be wrong relative to human labels? Provide a human audit on a random sample of accepted items to estimate precision of the unanimous filter. Also share inter-judge agreement matrices and typical disagreement cases. 

3. Break down gains by challenge types (negation, comparatives, quantifiers, monotonicity, syntactic perturbations). Which phenomena benefit most from VAULT?

### Soundness
3

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
3

### Summary
This paper proposes a method of LLM with RAG for natural language inference, which considers to incorporate adversarial examples for fune-tuning LLMs to improve its robustness.

### Strengths
- This paper proposes an end-to-end automated adversarial RAG pipeline, which fully automates retrieval, adversarial generation, multi-LLM validation, and iterative retraining.
- This paper provides a detailed procedure for the RAG pipeline, and experimental results on three NLI datasets show the proposed method achieves better performance by fune-tuning RoBERTa-base.

### Weaknesses
- The proposed method is not innovative, since adversarial generation has been proposed by prior works.
- As many LLMs can use RAG for implementing natural language inference, I want to see a direct comparison with these LLMs in the experiment.
- The structure of the paper could be improved, the figures and the hyperparameter settings can be put in appropriate positions.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a data augmentation framework VAULT for the NLI task that includes three stages: retrieval, adversarial generation, and iterative retraining. 
The retrieval stage uses both semantic (BGE) and lexical (BM25) similarities for few-shot sample retrieval; The adversarial generation phase employs LLM to generate hypotheses, with label accuracy validated through three LLMs; The iterative retraining stage fine-tunes NLI model by mixing the given original data and the generated adversarial data.
Experimental results on three NLI datasets show that VAULT improves the performance of a RoBERTa-base NLI model.
However, this paper still exhibits certain issues in paper writing and experiments.

### Strengths
1) The authors employ a data augmentation method for the NLI task which generate the adversarial data for training a model.
2) During the retrieval phase, the authors combine BGE and BM25 methods for the sample similarity assessment, effectively capturing the semantic information and the token-level feature. Experimental results further validated the effectiveness of this methodology.

### Weaknesses
1） This method only has been evaluated on NLI tasks, which limits its practical value. 
2） The experimental comparison is insufficient. Since this paper focuses on model training with labelled data, the baseline methods should include few-shot learning methods and LLMs based contextual learning methods. However, currently the paper only compared with LLMs, lacking comparisons with the aforementioned types of methods. Besides, the authors only use Roberta-base as the baseline model for NLI tasks, which cannot convince the effectiveness of the proposed method across a broader range of NLI scenarios. In the experimental setup, there is not the essential statistical information regarding the dataset. 
3） The authors do not explain clearly the process on how large language models (LLMs) generate hypotheses based on retrieval results. This process, however, critically influences the subsequent filtering and iterative retraining steps within the LLM framework.
4） The are some conflicts in the model section. In Section 3.2, the explanation of the formula contradicts the interpretation of the results in Figure 5. The claim that “the combined BGE+BM25 strategy consistently outperforms either alone” is inconsistent with the findings illustrated in Figure 5. In the retrieval part, there is an inconsistency in symbol usage—for instance, both the premise and the query are denoted by the symbol “p”.

### Questions
Here are some suggestions: 
1)	It is recommended to validate the proposed method across multiple tasks and compare it with additional baseline methods under identical experimental settings, particularly using labeled data of comparable scale.
2)	More details are desired on the method, including the adversarial generation process, the manner of organizing input information, and the prompt.
3)	The authors are recommended to provide more explanations when analysing experimental results, such as the reasons that cause the performance improvements or declines, rather than merely describing the observed changes, and analyze whether these changes can be attributed to your novel method.

### Soundness
2

### Presentation
2

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
This paper presents VAULT, an automated adversarial RAG pipeline for improving the robustness of NLI models. The method iteratively retrieves balanced few-shot contexts, generates adversarial hypotheses with LLM, validates them through a multi LLMs ensemble, and retrains RoBERTa with the validated data. VAULT achieves substantial accuracy gains on SNLI, ANLI, and MultiNLI, while requiring no human annotation, demonstrating that fully automated, LLM-driven adversarial data generation can effectively enhance model generalization and resilience.

### Strengths
1. The main contribution of this paper is the design of an automated “retrieve-generate-validate-retrain” closed-loop system, which provides a valuable engineering framework for improving the robustness of NLI models in a scalable manner without human annotation.
2. The paper conducts extensive ablation studies, demonstrating substantial experimental effort. The main experiments show that with a small number of adversarial samples, the method can achieve significant performance improvements across multiple NLI tasks.

### Weaknesses
1. The primary concern about this paper lies in its novelty. Each component of VAULT (including RAG, adversarial sample selection and refinement through iterative training, and the use of an LLM as a verifier) has been explored in prior work. VAULT appears to be more of an integration and adaptation of these existing techniques rather than a fundamentally new approach.
2. The paper lacks ablation studies across different models.

    a. Using different backbone models. It would be important to see whether VAULT remains effective when applied to stronger NLI models or to decoder-only language models such as Qwen3-0.6B or SmolLM2-360M.

    b. The study should also examine the effect of using generation and verification models of different scales. Evaluating stronger or weaker LLMs would help analyze how VAULT performs under different computational budgets.

    c. It is necessary to control the strength of the generator and the verifier to determine whether the performance gain mainly comes from the generation process or from the verification process.
3. [Minor] While VAULT indeed reduces the demand for large data volumes, it is model-specific. The method is tailored to address the weaknesses of Roberta-base-snli, whereas GNLI is a general-purpose data augmentation approach. Therefore, the comparison in terms of efficiency between VAULT and methods like GNLI is somewhat unfair.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
