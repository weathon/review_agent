# What Generative Search Engines Like and How to Optimize Web Content Cooperatively

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 8

## Abstract
By employing large language models (LLMs) to retrieve documents and generate natural language responses, Generative Engines, such as Google AI overview and ChatGPT, provide significantly enhanced user experiences and have rapidly become the new form of search. Their rapid adoption also drives the needs of Generative Engine Optimization (GEO), as content providers are eager to gain more traction from them. In this paper, we introduce AutoGEO, a framework to automatically learn generative engine preferences when using  retrieved contents for response generation, and rewrite web contents for more such traction. AutoGEO first prompts frontier LLMs to explain generative engine preferences and extract meaningful preference rules from these explanations. Then it uses preference rules as context engineering for AutoGEO$\_\text{API}$, a prompt-based GEO system, and as rule-based rewards to train AutoGEO$\_\text{Mini}$, a cost-effective GEO model. Experiments on the standard GEO-Bench and two newly constructed benchmarks using real user queries demonstrate the effectiveness of AutoGEO in enhancing content traction while preserving search utility. Analyses confirmed the learned rules' robustness and abilities to capture unique preferences in variant domains, and AutoGEO systems' ability to embed them in content optimization. The learned preference rules, our models, and the code is released at https://github.com/cxcscmu/AutoGEO

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors address the task of Generative Engine Optimization (GEO) and introduce a framework called AutoGEO, which aims to rewrite web content to improve its visibility on generative search engines. To achieve this, large language models (LLMs) are employed to extract preference rules by analyzing documents with contrasting visibility levels. The authors then propose two models to learn these rules: a prompt-based model and another trained using GRPO, which optimizes a reward composed of three components. Experiments conducted on three datasets demonstrate that AutoGEO achieves significant performance gains over baseline methods.

### Strengths
- S1: The problem addressed in this work is realistic and timely, given the increasing role of generative search engines in content discovery. The paper offers a practical and effective solution that aligns well with how web content creators may optimize for LLM-based visibility.

- S2: The proposed two-stage framework is conceptually clear and straightforward to follow. The writing is clean and the overall presentation is well-organized, making the methods understandable and likely reproducible. The implementation choices are also reasonable and grounded.

- S3: The experimental evaluation is comprehensive. The authors benchmark across multiple datasets from different domains and evaluate performance on several LLMs. The transferability analysis showing how rule sets may generalize across engines or domains is particularly interesting and contributes useful insight into how stable such optimization strategies may be in practice.

- S4: The results are strong and consistently show improvements over baselines.

### Weaknesses
- W1: The rule extraction process heavily relies on LLM-generated rationales, yet there is limited qualitative or human-driven analysis of the extracted rules themselves.

- W2: There is no ablation study breaking down the effects of each stage of the rule extraction pipeline. Since the rules are represented in free-form text, there is a reasonable possibility that multiple rules encode highly similar behaviors or overlapping preferences. A redundancy or compression analysis (e.g., clustering or semantic similarity grouping) would help clarify how many distinct principles are truly being captured.

- W3: While the method demonstrates that individual documents can be optimized for visibility, it is unclear how the system behaves under global optimization—i.e., if many or all documents adopt similar optimized styles. A small-scale controlled experiment (e.g., optimizing all documents within a subset and measuring relative visibility or stylistic homogenization) would provide insight into potential equilibrium

### Questions
- Q1: Are there instances where the rewritten documents introduce hallucinated or unsupported information?

### Soundness
3

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
2

### Summary
This paper introduces AutoGEO, a new GEO method for generative engines that generates responses based on retrieved contents.
AutoGEO leverages LLMs to extract preference rules from a generative engine and rewrites web contents based on these preference rules such that corresponding contents gain more traction in generative engines' generated responses.
Two strategies, including $AutoGEO_{API}$ and $AutoGEO_{Mini}$, are proposed.
The former leverages prompt engineering on LLMs with frozen parameters and the latter introduces a compact model fine-tuned using SFT and RL for cost-efficient deployment.
Experiments on several benchmarks demonstrates the effectiveness of AutoGEO.

### Strengths
1. Writing is good and the paper is easy to follow.

2. This paper focuses on GEO, which is an important and practical problem. Since generative engines become more and more popular, understanding and resolving GEO is crucial, just as the status of SEO in conventional search engine. The proposed method will be beneficial for future work in this area.

3. Experimental evaluation is extensive. This paper consider not only existing GEO benchmark but also constructing two new benchmarks, which enhances the diversity and robustness of the evaluation of AutoGEO. Besides, detailed analysis on GEO and GEU and ablation study strengthens the effectiveness of AutoGEO.

### Weaknesses
1. The novelty is limited. AutoGEO seems a direct application of existing techniques with little novelty in learning paradigm and theoretical understanding. The process of using a powerful LLM to explain, extract, and merge preference rules is essentially a form of automated data labeling and summarization, and using these extracted rules as prompts for a rewriting model (i.e., $AutoGEO_{API}$) is standard in-context learning. Besides, using rules as reward signals in RL for fine-tuning LLMs (i.e., $AutoGEO_{Mini}$) is also a direct application of existing methods. It develops a competent application system which achieves impressive GEO performance, but does not make a foundational machine learning contribution.

2. The analysis on cost is limited. This paper argues the cost efficiency of $AutoGEO_{Mini}$ compared to $AutoGEO_{Mini}$ (i.e., ~0.071x), but ignores the cost of rule extraction phase, which seems to be substantial and dominant. For evaluating a practical application, the cost of the entire process should be analyzed.

3. Analysis on preference rules is limited. The rule extraction pipeline obtains the final rule set through components like Explainer, Extractor, Merger, and Filter. However, clear metrics or methods to measure the accuracy, effectiveness or rationality of these rules themselves are missed and it is validated in an indirect way via downstream tasks. This approach only shows that rules are useful, but does not verify that a rule reflects the true preference of generative engines instead of noise from the dataset or inherent bias of LLMs.

### Questions
In summary, this paper develops an effective application for resolving GEO problem, which is an important application domain. However, the algorithmic or theoretical contribution is limited. So I think this paper is more aligned with the field of applied data science like KDD, WWW and SIGIR.

This paper can be further improved, including:
1. Adding theoretical analysis, e.g., discussing the condition under which the proposed method is guaranteed to work
2. Adding analysis of the effectiveness and rationality of preference rules

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This study aims to optimize web documents so that their content gains higher visibility. The key idea is to optimize the utilization of retrieved information via a generative engine. To achieve such a goal, a training-free framework and a RL-based training approach are proposed, which aims to extract preference rules for generative engine optimization in a plug-in or an automatic way. The experimental results based on common metrics and the designed evaluation setting demonstrate the effectiveness. Overall, this is an interesting paper that offers an important perspective on the future of search.

### Strengths
1. The target problem is important and interesting. What is the search engine like in the era of LLM and how to leverage the LLM to integrate with search paradigm is a timely topic. This study provides a new perspective for the community to facilitate our thoughts.

2. The proposed methods are effective, which include two different versions: plug-in one and an automatic one with training. The designed framework is reasonable and systematic.

3. Experimental results are good and a series of analyses are provided to support the claim of this study.

### Weaknesses
1. The motivation/illustration of the proposed search paradigm could be made more clear by comparing its differences with existing search paradigms, e.g., generative IR model, LLM-based dense retriever, LLM-based unified retriever and generator, and vanilla RAG. A systematic comparison could enhance the contribution in terms of the perspective of "What Generative Search Engines Like".

2. Similar to point 1, a task definition could be desirable before describing the methodology. The defined scenarios can connect with the used downstream datasets and evaluation settings.

3. The involved compared paradigms in point 1 could be included in the related work to better illustrate the connection and differences compared to existing search paradigms. Please see my question below as another enhanced aspect.

### Questions
1. What is the difference (and their optimization goal) between the generative engine (as introduced in related work) and the unified framework that aims to unify the retrieval and generation tasks using a single LLM as backbone (e.g., GRIT [1], UniConv [2])? It would be better to illustrate them in the related work, as the paper also wants to provide a perspective of "What Generative Search Engines Like"

[1] Generative representational instruction tuning.

[2] Uniconv: Unifying retrieval and response generation for large language models in conversations.

### Soundness
3

### Presentation
4

### Contribution
3
