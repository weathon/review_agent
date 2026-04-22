# Programmatic Context Augmentation for LLM-based Symbolic Regression

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 2, 8

## Abstract
Symbolic regression (SR), the task of discovering mathematical expressions that best describe a given dataset, remains a fundamental challenge in scientific discovery. Traditional approaches, primarily based on genetic algorithms and related evolutionary methods, have proven useful but suffer from scalability and expressivity limitations. Recently, large language model (LLM)-based evolutionary search methods have been introduced into SR and show promise. However, existing LLM-based approaches typically rely on scalar evaluation metrics, such as mean squared error, as the sole source of feedback during the search process, thereby overlooking the rich information embedded in the dataset. To address this limitation, we propose a novel LLM-based evolutionary search framework that incorporates programmatic context augmentation: by enabling code-based interactions with the dataset, our method can actively perform data analysis and extract informative signals, beyond aggregated evaluation scores. We evaluate our framework on advanced benchmarks, such as LLM-SRBench, and demonstrate superior efficiency and accuracy compared to strong baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The article integrates the mean field game with neural differential equations to construct a brand-new continuous sequence prediction framework: characterizing the temporal evolution as the game process of a vast number of agents and efficiently solving Nash equilibrium through virtual games. The theoretical derivation is rigorous, and the experimental results show that it surpasses the existing methods in terms of prediction accuracy. Although the computational cost of the model is relatively high, its ability to describe high-order temporal dependencies and complex inductive biases is outstanding, opening up a new theoretical perspective and providing practical tools for continuous sequence modeling.

### Strengths
This article is well-structured and has the following advantages：

1) The article seamlessly integrates mean-field games with neural differential equations to construct a continuous sequence generator, and introduces graphons to characterize the limit interaction structure. 

2) It theoretically derives the mean-field adjoint equation and experimentally improves the prediction accuracy on multiple public datasets.

### Weaknesses
Although the article has some novel ideas at the theoretical level, its innovation points are somewhat lacking and it essentially belongs to the "assembled" type. 

1) In terms of workload, the experiment only covered three medium-sized datasets. It lacked systematic statistics on computational costs such as training time, GPU usage, and the number of convergence steps. Moreover, it did not further verify its scalability in scenarios with millions of nodes or high-frequency sampling. 

2) The graphon kernel function needs to be manually specified and is sensitive to prediction errors. However, the article does not provide selection criteria or sensitivity curves. 

3) Moreover, the article pays less attention to the stability, robustness, and sensitivity analysis of Nash equilibrium solutions and hyperparameters. These issues result in insufficient interpretability and reproducibility of the article.

### Questions
If the author can address the issues raised in the weaknesses section and, on this basis, explain the rationality of the structure and algorithm of this article (rather than just piecing together the algorithm), I believe the content of the article will become more substantial.

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
5

### Summary
This paper proposes PROAUG, a framework that extends LLM-based symbolic regression by introducing a programmatic data-analysis phase. Instead of using only scalar loss signals, the LLM generates and executes Python programs to compute dataset statistics, which are then used as contextual input for equation generation. Experiments on LLM-SRBench show notable reductions in NMSE compared to LLM-SR and statistical-hint baselines.

### Strengths
The main advantage is that the idea of using additional information to guide equation evolution is interesting and potentially useful.

### Weaknesses
The results are not fully consistent with LLM-SRBench, and the ablation studies are insufficient.

### Questions
1. The reported NMSE values differ significantly from those in the original LLM-SRBench paper. Although different foundation models are used, some baseline models (e.g., Llama3.1 in Biology) achieve NMSE around 1e-6, much better than the values shown here. This raises concerns about evaluation consistency.
2. Section 3 presents a case where statistical hints help LLM-SR, but it mainly reflects a local search behavior. The authors should include a concrete example on LSR-Syn to demonstrate how GPT-generated statistical information specifically improves LLM-SR performance.
3. The proposed method does not use the same prompt as LLM-SR. An ablation is needed to confirm whether improvements come from the proposed augmentation or simply from a new prompt design. In particular, Appendix C.2 uses a linear regression model as a seed for PROAUG, which might bias the results.
4. The computational cost is not reported. The paper should clarify how much additional time is required for code generation and data analysis.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work introduces ProAug for LLM-based symbolic regression which allows LLMs to produce code that is then executed on the dataset to get statistics which is then fed back into the LLM. ProAug is then somewhat evaluated on LLM-SRBench, albeit not adopting the benchmark practices/results.

### Strengths
The introduction of a mechanism that allows LLMs to produce code that is then executed on the dataset to get statistics is novel for LLM-based SR.

### Weaknesses
Cites but lacks critical discussion or analysis with highly relevant work. A quick search on LLM-based SR methods yields RAG-SR [1] from ICLR 2025, which the paper cites briefly “These methods train networks on experimental data to either directly….or to accelerate the search for target expressions.”. Insufficient discussion and comparison are made to RAG-SR. The critical reason why RAG-SR caught my attention is because this paper claims that (and I quote from the introduction) “existing LLM-based approaches typically rely on scalar evaluation metrics, such as mean squared error, as the sole source of feedback during the search process, thereby overlooking the rich information embedded in the dataset”, yet RAG-SR addresses this exactly, with “semantics” (referring to vector of errors, instead of a scalar. The lack of discussion and empirical comparison to highly relevant work, especially from the same conference, is concerning.

[1] Zhang, Hengzhe, et al. "RAG-SR: Retrieval-augmented generation for neural symbolic regression.", ICLR 2025

Results (see Table 1) are inconsistent with literature, with errors several orders of magnitude higher (except “Material Science” category), even in the original LLM-SRBench paper [2]. Please see Table 1 in [2], and look at the LLM-SR results. In fact LLM-SRBench benchmarks on 3 other competitors, Direct Prompting, SGA, LaSR, which are not considered in this paper. Even LaSR beats most results that this paper presents, by orders of magnitude, which brings into questions the effectiveness of the algorithm.

[2] Shojaee, Parshin, et al. "LLM-SRBench: A new benchmark for scientific equation discovery with large language models." ICML 2025

Another point is that since the paper is using the experiments and standards borrowed from LLM-SRBench, the symbolic accuracy (SA) and numeric precision (Acc_0.1) should be reported too. LLM-SRBench’s value lies in its completeness and transparency, yet the current results include only a subset of algorithms and a single metric. I encourage the authors to include all benchmarked algorithms and multiple metrics to provide a fair, reproducible, and fully informative comparison.

Lack of details to reproduce experiment. Without additional details, I can only assess that prompts seem heavily reliant on the human designer. There is only one example in the Appendix as well, which makes it difficult to assess the approach.

Minor point, please use algorithm names consistently, e.g., “LLMSR”, “LLM-SR” are used interchangeably in the paper.

Finally, paper lacks confidence intervals, error bars and statistical tests for the results. It is hard to determine the significance of the results, even the issues above are resolved.

### Questions
Ironically, although the work is about symbolic regression, which discovers equations, I am unable to find an example of the equation that ProAug has discovered. Can the paper provide the equations that are discovered, or give a few clear examples of the f(x) discovered (alongside their metrics)?

How are the prompts developed? Are they ad-hoc or did they undergo a systematic process?

Were the prompts vetted or proofread? There are some grammatical and spelling mistakes in the Appendix, were these the actual prompts or is this just a typographical error in the paper?

Please address the weaknesses listed above as well.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents PROAUG, a framework for enhancing large language model (LLM)-based symbolic regression through programmatic context augmentation. The approach enables the LLM to generate and execute code that performs dataset analysis (e.g., correlations, transformations, descriptive statistics) prior to equation generation. The resulting analysis outputs are inserted back into the LLM prompt, enriching the contextual information available during symbolic search.

### Strengths
The paper identifies a clear limitation in existing LLM-based SR systems and proposes a practical solution that emulates human-like exploratory data analysis. 

The concept of integrating code-generated statistical insights into prompts is novel and well-aligned with the emerging field of neuro-symbolic and programmatic reasoning.

### Weaknesses
The main limitations of this work lie in its restricted autonomy, heuristic guidance, and limited evaluation scope. 

* The framework relies on a predefined set of data analysis operations, such as basic statistical measures and simple transformations, e.g., log, exp, sin, cos. This restricts its ability to discover equations that depend on highly complex or non-standard data relationships.


* The correlation metrics used in the statistical context are manually designed and confined to a small set of predefined transformations. This may result in the failure to detect intricate nonlinear or multivariate interactions.


* During supervised fine-tuning, the model occasionally generates non-executable code. While the authors mitigate this issue by augmenting training data with examples of failed code generation, it remains a limitation.

### Questions
How do you plan to address the limitation of relying on predefined data analysis operations to enable the discovery of more complex or non-standard data relationships in future iterations of PROAUG? 


Given the one-to-many mapping issue in the LSR-Synth dataset subset, do you have plans to adapt supervised fine-tuning (SFT) for such cases, or are there alternative approaches you are considering to improve performance on these tasks?

### Soundness
3

### Presentation
3

### Contribution
3
