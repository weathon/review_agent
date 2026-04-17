# TableMaster: A Recipe to Advance Table Understanding with Language Models

- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Tables serve as a fundamental format for representing structured relational data. While current language models (LMs) excel at many text-based tasks, they still face challenges in table understanding due to the complex characteristics of tabular data, such as their structured nature. In this paper, we aim to enhance LMs for improved table understanding. We identify four key challenges: 1) difficulty in locating target data, 2) deficiency in table semantics, 3) numerical inaccuracies in textual reasoning, and 4) semantic inflexibility in symbolic reasoning. To address these issues, we propose TableMaster, a recipe and comprehensive framework that integrates multiple solutions to overcome these obstacles. TableMaster first extracts relevant table content and verbalizes it with enriched semantic context. Additionally, we introduce adaptive reasoning, a flexible approach that dynamically adjusts between textual and symbolic reasoning, tailoring the reasoning process to each query. Extensive analyses and experiments demonstrate our findings and the effectiveness of TableMaster. On the WikiTQ dataset, TableMaster achieves an accuracy of 78.13% using GPT-4o-mini, surpassing existing baselines. We hope this work will serve as a practical step toward more robust and reliable table understanding.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes TableMaster, a framework designed to enhance language models'  table understanding capabilities. The paper identifies several challenges: difficulty in locating target data, semantic deficiency in tables, numerical inaccuracies in textual reasoning, and semantic inflexibility in symbolic reasoning. To address these, TableMaster integrates multiple solutions including table-of-focus construction, table verbalization, program-aided reasoning, and adaptive reasoning that dynamically selects between textual and symbolic approaches. Extensive experiments on WikiTQ, TabFact, and FetaQA datasets show that TableMaster achieves good performance.

### Strengths
1. The paper provides a structured analysis of four fundamental challenges in table understanding, with each challenge directly linked to a targeted solution. 

2. TableMaster integrates multiple techniques (table-of-focus, verbalization, adaptive reasoning) into a pipeline. 

3. The paper conducts extensive experiments across diverse datasets and LLMs.

### Weaknesses
1. The core contributions of the paper are primarily engineering-focused. The paper lacks novel advancements in LM architecture or reasoning mechanisms specific to table understanding.

2. Experiments are concentrated on clean, structured tables from specific domains. The framework's performance on real-world noisy tables, hierarchical tables remains insufficiently explored, for example, the BIRD dataset. 

3. The dynamic strategy selection relies on LM judgment without robust error handling. 

4. While efficiency is discussed, no actual latency measurements or comparison with simpler baselines are provided. The multi-step process (verbalization, reconstruction, adaptive reasoning) likely introduces significant inference time overhead.

### Questions
please refer to the weakness

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
3

### Summary
This paper addresses table understanding with language models by identifying four key challenges: (i) difficulty in locating target data, (ii) deficiency of table semantics, (iii) numerical inaccuracy in textual reasoning, and (iv) semantic inflexibility in symbolic reasoning. The authors propose TableMaster, a comprehensive framework that integrates multiple solutions including table-of-focus construction, table verbalization, and adaptive reasoning that dynamically switches between textual and symbolic approaches. The method is evaluated on WikiTQ, TabFact, and FetaQA datasets, showing improvements over existing baselines.

### Strengths
- The paper provides a thorough empirical analysis of challenges in table understanding, with systematic experiments examining the impact of table size, verbalization, and different reasoning approaches.
- TableMaster achieves notable improvements across various large-scale LLMs (GPT-3.5-turbo, GPT-4o-mini, LLaMA-3 70B).

### Weaknesses
- While the integration is well-executed, most individual components (sub-table extraction, table verbalization, program-aided reasoning) have been proposed in prior literature. The novelty primarily lies in their combination rather than in introducing fundamentally new techniques.
- The section 3-4 can be condensed to leave more space for experiment and analysis. Currently, most the results are in the appendix.
- The evaluation focuses exclusively on large-scale models. It remains unclear how TableMaster performs on smaller (7–8B) models or what minimal model capabilities are required for it to function effectively.

### Questions
1. What are the minimum model capabilities required for TableMaster? Have you tested on 7-13B parameter models? At what model size does the framework start to break down?
2. Given that each component is well-established, what specific insights or contributions does TableMaster provide beyond engineering integration?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces TableMaster, a novel framework designed to enhance how Large Language Models (LLMs) understand tabular data. The research addresses four key obstacles: difficulty in data localization, semantic deficiency, numerical inaccuracies, and inflexible symbolic reasoning. TableMaster employs a multi-faceted strategy, beginning by isolating relevant data into a "table-of-focus" and then using "verbalization" to enrich it with semantic context. The framework integrates program-aided reasoning and features an adaptive mechanism that dynamically balances textual and symbolic approaches based on the query. This method has achieved state-of-the-art performance on the WikiTQ and TabFact benchmarks, notably reaching 78.13% accuracy on WikiTQ with GPT-4o-mini, significantly surpassing existing baselines.

### Strengths
1. The paper demonstrates strong empirical rigor through comprehensive experiments across multiple benchmark datasets and baselines. The thorough ablation studies effectively validate the contributions of individual components, providing clear evidence of the method's effectiveness.
2. The authors developed a robust system for analyzing and extracting information from general tabular data, with carefully designed modules compatible with various language model backbones.

### Weaknesses
1. Several key experiments are missing from the main paper, such as the analysis of adaptive reasoning. Including these results in the main body would strengthen the paper.
2. The framework is thoughtfully designed and comprehensive, but many of its sub-tasks have been extensively studied, with closely related methods already proposed. As a result, the incremental novelty appears limited. 
3. In the related work section the connections of similar methods to this work are not clear. It would help to position the framework relative to each major line of work (what is shared, what differs, and why those differences matter), and to articulate the specific gaps in prior methods that this paper addresses.

### Questions
Please refer to the weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
2
