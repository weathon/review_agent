# TrustGen: A Platform of Dynamic Benchmarking on the Trustworthiness of Generative Foundation Models

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Generative foundation models (GenFMs), such as large language models and text-to-image systems, have demonstrated remarkable capabilities in various downstream applications. As they are increasingly deployed in high-stakes applications, assessing their trustworthiness has become both a critical necessity and a substantial challenge. Existing evaluation efforts are fragmented, rapidly outdated, and often lack extensibility across modalities. This raises a fundamental question: how can we systematically, reliably, and continuously assess the trustworthiness of rapidly advancing GenFMs across diverse modalities and use cases? To address these gaps, we introduce TrustGen, a dynamic and modular benchmarking system designed to systematically evaluate the trustworthiness of GenFMs across text-to-image, large language, and vision-language modalities. TrustGen standardizes trust evaluation through a unified taxonomy of over 25 fine-grained dimensions—including truthfulness, safety, fairness, robustness, privacy, and machine ethics—while supporting dynamic data generation and adaptive evaluation through three core modules: Metadata Curator, Test Case Builder, and Contextual Variator. Taking TrustGen into action to evaluate the trustworthiness of 39 models reveals four key insights. (1) State-of-the-art GenFMs achieve promising overall trust performance, yet significant limitations remain in specific dimensions such as hallucination resistance, fairness, and privacy preservation. (2) Contrary to prevailing assumptions, open-source models now rival and occasionally surpass proprietary systems in trustworthiness metrics. (3) The trust gap among top-performing models is narrowing, likely due to increased industry convergence on best practices. (4) Trustworthiness is not an isolated property; it interacts complexly with other behaviors, such as helpfulness and ethical decision-making. TrustGen is a transformative step toward standardized, scalable, and actionable trustworthiness evaluation, supporting dynamic assessments across diverse modalities and trust dimensions that evolve alongside the generative AI landscape.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents TRUSTGEN, a dynamic and modular benchmarking platform designed to evaluate the trustworthiness of generative models. TRUSTGEN develops a dynamic evaluation pipeline encompassing three modules—the Metadata Curator, the Test Case Builder, and the Contextual Variator—to enable the automatic and iterative generation of evaluation data. The authors assessed 39 models and summarized their observations.

### Strengths
* The engineering effort behind this paper is impressive. The work involves the design and evaluation across dozens of trustworthiness dimensions and models. The authors provided the source code and datasets.

* The paper introduces several new, fine-grained trustworthiness dimensions, such as "Self-Doubt Sycophancy" and "Exaggerated Safety".

### Weaknesses
* The motivation for proposing the evaluation framework for T2I, LLM, and VLM models is not clear. While Table 3 suggests that the authors' evaluation framework covers a greater number of models and dimensions, the question remains: Why can't existing benchmarks, if combined, achieve the same purpose? What is the critical shortcoming of existing benchmarks? The potential advantage is the construction of dynamic datasets. However, this could also be a drawback, as the quality of evaluation data cannot be fully guaranteed before rigorous human checking.

* The structure of the paper is somewhat unreasonable for a benchmarking study. The main body should present the detailed information about dimension, dataset (size, case examples), and metrics. Furthermore, I note from Table 1 that the evaluation dimensions for different model types appear to be defined independently. Why should the dimensions of trustworthiness vary for each model? For instance, if LLMs can generate toxic output, why would VLMs be exempt from this concern [1]? I suggest the authors provide more rigorous definitions for their trustworthiness dimensions. By the way, safety and robustness are often discussed synonymously within the context of trustworthy AI, so their distinct roles in this work need more justification. Additionally, critical threats like backdoor attacks and watermarking are significant components of trustworthy AI, yet they are not discussed in this paper.

[1] On the trustworthiness landscape of state-of-the-art generative models: A survey and outlook

--- 

Note: My review is primarily based on the main body of the paper.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents TrustGen, a dynamic and modular benchmarking platform designed to evaluate the trustworthiness of Generative Foundation Models (GenFMs)—including large language models (LLMs), text-to-image (T2I) systems, and vision-language models (VLMs). The proposed framework standardizes trust evaluation through a unified taxonomy of over 25 fine-grained dimensions covering truthfulness, safety, fairness, robustness, privacy, and ethics. TrustGen consists of three core modules—Metadata Curator, Test Case Builder, and Contextual Variator—that enable dynamic dataset generation and adaptive evaluations. The authors benchmarked 39 generative models and reported four general insights, such as narrowing trust gaps among top models and competitive performance from open-source systems. The paper claims that TrustGen fills the gap in fragmented and static evaluations, offering a unified, extensible platform for trustworthiness assessment.

### Strengths
1. Timely and important topic – The paper addresses a highly relevant problem in the current AI landscape: how to evaluate the trustworthiness of diverse generative models. As GenFMs are increasingly applied in sensitive domains, systematic evaluation frameworks like TrustGen are much needed.

2. Comprehensive coverage and modular design – The proposed benchmark encompasses multiple modalities (text, vision, and multimodal) and integrates numerous trust-related dimensions. The three-module architecture (Metadata Curator, Test Case Builder, and Contextual Variator) is thoughtfully designed and ensures flexibility and extensibility.

3. Extensive empirical evaluation – The authors evaluate a large set of mainstream models (LLMs, T2I, and VLMs) and provide cross-model analysis. The large-scale experimentation adds credibility to the platform’s applicability and potential impact.

### Weaknesses
1. Unclear motivation for mixed cross-model evaluation – While the authors emphasize that existing works evaluate model families in isolation (LLMs or T2I), the paper does not convincingly justify why evaluating different modalities together is necessary or what unique insights such a unified evaluation yields. The presented findings do not clearly demonstrate benefits that could not be obtained through separate, modality-specific assessments. Thus, the key motivation and novelty of the “cross-family” benchmarking remain insufficiently supported.

2. Lack of surprising or deep insights – The four reported insights (e.g., narrowing trust gap, open-source competitiveness, correlation between trust and utility) are largely expected and do not provide new conceptual understanding of GenFM trustworthiness. The results read more as descriptive summaries rather than analytical insights that could guide future model design or policy development.

3. Inconsistent evaluation dimensions across modalities (Table 1) – Table 1 shows that different model categories (LLMs, T2I, VLMs) are evaluated under non-uniform trustworthiness dimensions and metrics, which undermines the claim of a unified benchmark. This inconsistency weakens the argument for the necessity of a cross-modal evaluation platform and raises questions about whether TrustGen truly enables comparable assessments across model families

### Questions
see the weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
TrustGen introduces a dynamic, modular benchmark for assessing the trustworthiness of generative foundation models in LLM, T2I, and VLM. It standardizes evaluation via a unified taxonomy, including truthfulness, safety, fairness, robustness, privacy, machine ethics, etc. The benchmark continuously generates up-to-date test cases via three modules: Metadata Curator, Test Case Builder, and Contextual Variator. Applied to 39 models, TrustGen yields several notable findings, including strong average performance but persistent gaps in areas such as hallucination, fairness, and privacy, with some open-source systems matching the performance of frontier proprietary models.

### Strengths
- Comprehensive framework: built on a broad, fine-grained taxonomy and a reusable cross-modal pipeline, validated by broad empirical coverage, evaluating 39 models across modalities.

-  Dynamic data generation: Moves beyond static benchmarks by iteratively producing fresh test cases with minimal human intervention, better matching a rapidly changing threat landscape.

- Human Review: the authors provide human verification for both test case generation and result evaluation, demonstrating the reliability of using LLMs to generate cases and to act as a judge within the TrustGen framework.

-  Reusability: Public dataset link and toolkit.

### Weaknesses
-  Dynamic-benchmark comparability: While dynamic is a strength, it complicates leaderboard stability. Strict versioning will be needed for comparisons.

- Expository clarity requires improvement: missing details make the core contributions hard to discern.

### Questions
1.  Writing and emphasis: I appreciate the substantial effort required to produce this work, including task selection, data-generation pipeline design, metric comparison, and the modular framework. However, the manuscript places disproportionate emphasis on analyzing evaluation results. Unlike HELM, DecodingTrust, MultiTrust, and Safebench, a core contribution here is dynamic data generation and adaptive evaluation. This aspect deserves more space and clearer exposition so readers can understand what “dynamic” concretely entails.
    
    -   Please expand the section 2.1, or align each subsequent evaluation sections to the three key modules in Figure 1 and provide the corresponding data generation pipeline. The appendix only briefly describes the dataset composition，lacks concrete details on how each component is constructed, as well as dataset sizes.
        
    -   For each dimension, the manuscript does not clearly clarify differences from prior work.

2. Benchmark fairness：how is fairness ensured for dynamic datasets?

3. Future work: I strongly recommend adding a dedicated future work section to outline follow-up plans, such as introducing additional modalities or model families, e.g., audio LLMs and text-to-video models. For a benchmarking platform that required substantial effort to build, sustained iteration and updates are essential, without an ongoing roadmap, its long-term value will be limited.

4. The paper covers many trustworthiness aspects, and different sections were likely written by different authors. Please align the writing template across dimensions.
    
    -   In the appendix, section sub-titles are inconsistent, e.g., many dimensions omit a Dynamic Dataset subsection.
        
    -   In F2.1, what is the purpose of introducing standard TrustLLM datasets and comparing them with dynamic datasets? It seems that the model achieves higher acc on dynamic datasets compared to TrustLLM. May I infer that dynamic datasets are less capable of fully revealing a model’s internal safety risks?
        
5. Minor: The model order in the main-text tables is unusual. For the GPT series for instance, the ordering is neither chronological nor by capability.

These questions require little to no additional experimentation. I would be pleased to revise my score if the authors provide reasonable responses.

### Soundness
3

### Presentation
2

### Contribution
4
