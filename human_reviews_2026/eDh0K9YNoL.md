# OneSearch: A Preliminary Exploration of the Unified End-to-End Generative Framework for E-commerce Search

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Traditional e-commerce search systems employ multi-stage cascading architectures (MCA) that progressively filter items through recall, pre-ranking, and ranking stages. While effective at balancing computational efficiency with business conversion, these systems suffer from fragmented computation and optimization objective collisions across stages, which ultimately limit their performance ceiling. To address these, we propose OneSearch, the first industrial-deployed end-to-end generative framework for e-commerce search. This framework introduces three innovations: (1) a Keyword-enhanced Hierarchical Quantization Encoding (KHQE) module, to preserve both hierarchical semantics and distinctive item attributes while maintaining strong query-item relevance constraints; (2) a multi-view user behavior sequence injection strategy that constructs behavior-driven user IDs and incorporates both explicit short-term and implicit long-term sequences to model user preferences comprehensively; and (3) a Preference-Aware Reward System (PARS) featuring multi-stage supervised fine-tuning and adaptive reward-weighted ranking to capture fine-grained user preferences. Extensive offline evaluations on large-scale industry datasets demonstrate OneSearch's superior performance for high-quality recall and ranking. The online A/B tests confirm its ability to enhance relevance in the same exposure position, achieving statistically significant improvements: +1.67\% item CTR, +2.40\% buyer, and +3.22\% order volume. Furthermore, OneSearch reduces operational expenditure by 75.40\% and improves Model FLOPs Utilization from 3.26\% to 27.32\%. The system has been successfully deployed across multiple search scenarios in TEST, serving millions of users, generating tens of millions of PVs daily.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces OneSearch, a novel end-to-end generative framework for e-commerce search that replaces traditional multi-stage cascading architectures (MCA) with a unified model. The core idea is to generate item ranking list directly from user-query information, thus addressing the limitations of fragmented computation and optimization conflicts in MCA systems.

The proposed framework integrates three key innovations:
1. Keyword-enhanced Hierarchical Quantization Encoding: A hybrid RQ-OPQ tokenizer that preserves both shared and distinctive item features to improve semantic representation.
2. Multi-view User Behavior Sequence Injection: A method to encode user preferences using short- and long-term behavior signals, enhancing personalization.
3. Preference-Aware Reward System: A multi-stage fine-tuning process with an adaptive reward model to align generated rankings with user preferences.

Extensive offline evaluations and online A/B testing demonstrate significant improvements on multiple metrics.

### Strengths
1. The paper presents the first industrial-deployed end-to-end generative framework for e-commerce search, moving beyond the limitations of multi-stage architectures. This represents a significant contribution to both research and practice.
2. The paper provides comprehensive offline and online evaluations. Improvements in online metrics are significant and the proposed method claims to reduce the OPEX by a large extent.

### Weaknesses
1. The paper writing is a little bit hard to follow,  especially the tokenization part. The RQ-OPQ part does not give very clear description, the content in appendix A.3 includes a lot of details on experiment/implementation, but does not describe the method itself. And OneSearch includes many components, as shown in Figure 2, it is hard to get the major point of the proposed method.
2. No experiment results on open-source datasets. As the REPRODUCIBILITY STATEMENT pledges that the codebase will be released, perhaps it's better to provide results on open datasets.
3. Training details are not shown in the paper, like training cost, online update method, etc.

### Questions
1. The RM seems like a traditional ranking module, which seems the most critical part to the online AB test improvements shown in Table 4. If OneSearch still needs a "three-tower SIM" as the reward model to rerank the list, it is not fully "end-to-end". The proposed model is actually recall+pre-ranking?
2. Why the results in Table 2 uses HR@350 for both online MCA and w/o ranking? 350 seems like a length for testing pre-ranking or recall, not very suitable for the ranking model. Considering the proposed OneSearch (w/ RM reranking) is essentially recall+pre-ranking, its performance on HR@350 is not as good as "w/o ranking". 
3. In section 3.4, the paper claims to reduce the OPEX for 24.6%. How is it calculated? Because OneSearch with RM reranking still use a traditional ranking model with the recall&pre-ranking part replaced by Transformer structure, it is still a multi-stage complex system.

### Soundness
3

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
This paper introduces OneSearch, an end-to-end generative framework designed to replace traditional multi-stage architecture in e-commerce search. The authors propose a unified model that handles recall, pre-ranking and ranking. The core contributions seem to include a keyword-enhanced hierarchical quantization scheme, a novel strategy to input user behavior sequences, and a preference-aware reward alignment system. The authors report significant improvements in online A/B tests on a large-scale e-commerce platform, while achieving better GPU utilization and lower operating cost.

### Strengths
- The general idea of the paper is to unify e-commerce search into a single model, which is a highly interesting and relevant research question.
- The proposed OneSearch architecture seems to be highly optimized and is comprised of several meticulously crafted and fine-tuned parts, which is an impressive engineering accomplishment
- The model achieves impressive results over the previous production model, and several ablations validate individual aspects of it.
- In particular, the  proposed RQ-OPQ tokenization method appears to be a novel and effective idea to capture relevant remaining information after quantization.

### Weaknesses
While the paper tackles an important problem and presents promising results, it is difficult to follow and lacks clarity in key parts. It additionally reads like a system report rather than a research paper a lot of the time, making it difficult to assess its scientific novelty and contribution. In particular,

- The paper's clarity of writing could be greatly improved. In parts, the text is dense and convoluted, making it difficult to follow. There are several abbreviations that are used without proper introduction, forcing the reader to look up their definitions elsewhere and preventing the paper from being a stand-alone text. Since the paper proposes a pretty complex and intricate architecture, these shortcomings impact its readability and, to some extent, obscure its actual contributions.
- The proposed OneSearch method consists of a lot of components, and the paper does not always clarify which of these are novel contributions, and which are taken from previous work. These omissions cause the method section to partially read like a system report rather than a research paper.
- Some key methodological details are missing. For example, the loss functions in Section 2.1 are not self-contained. Terms like the "hard sample relevance correction loss" are introduced without any definition. While many of the non-explained terms are standard in retrieval systems, these omissions and the resulting ambiguity make it difficult to fully understand and appreciate the proposed method.
- The experimental evaluation lacks transparency. The dataset is described only as "TEST," which is not informative. The baseline, "onlineMCA," is a proprietary system, making it impossible to judge whether the reported gains are due to the novelty of OneSearch or simply differences in engineering and feature tuning. The paper sometimes states that OneSearch improves metrics by a certain percentage without clearly stating what it's being compared against in that specific context.
- Figures 2 and 3 are very large and cause physical lag on some PDF readers. Additionally, Figure 2 contains a lot of information, resulting in a somewhat overwhelming chart that does little to clarify the model.

### Questions
- Can the authors clarify the core scientific contributions of this paper, and differentiate it from the (impressive) engineering effort of the proposed method? What exactly are the main advances compared to existing work in dense and generative retrieval? What are durable takeaways for the wider research community?
- Related to this, how does the paper compare to recent generative frameworks like OneRec, which also unifies the recommendation pipeline, and GRAM, which applies generative retrieval to e-commerce. Could the authors elaborate on the key differences, and the reasons why OneSearch is particularly suited to the e-commerce setting?
- A significant part of the methodology appears to be highly tuned to your platform's data and infrastructure (e.g., the 18 structured attributes from NER, the specific six-level user behavior hierarchy). Which of the design choices of the method are specific to the chosen platform, and which would be expected to translate to other settings?
- The experiment section is based on the comparison to and relative improvement over the "onlineMCA" baseline. While I understand that the specifics are proprietary, could the authors provide a higher-level summary of the baseline? Similarly, could the method be, in part, compared to other existing work?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents OneSearch, an end-to-end generative framework for e-commerce search that aims to replace traditional multi-stage cascading architectures (MCA). The framework introduces (1) Keyword-enhanced Hierarchical Quantization Encoding (KHQE) that preserves hierarchical semantics while maintaining query-item relevance, (2) a multi-view user behavior sequence injection strategy incorporating both short-term and long-term user preferences, and (3) a Preference-Aware Reward System (PARS) with multi-stage supervised fine-tuning and adaptive reward modeling. Experimental results suggest that OneSearch achieves significant improvements in online A/B tests while reducing operational costs.

### Strengths
- The paper demonstrates substantial practical value with comprehensive online A/B testing results showing statistically significant improvements across multiple metrics. The successful deployment across multiple search scenarios is also meaningful.

- The KHQE module with RQ-OPQ tokenization and keyword enhancement addresses specific challenges in e-commerce search, such as noisy item information and strict relevance constraints. The multi-view behavior sequence injection strategy effectively handles both explicit short-term and implicit long-term user preferences, which is crucial for personalized search.

- The paper provides thorough offline evaluations, detailed ablation studies, and extensive online testing. The analysis covers various aspects like cold-start scenarios, different query popularities, and manual evaluation results.

### Weaknesses
- While the overall framework is novel, individual components largely build upon existing techniques. The core generative retrieval paradigm follows and adapts established patterns from Tiger and OneRec.

- The paper lacks discussion of scenarios where OneSearch underperforms or fails. There's limited analysis of the trade-offs between the unified approach and specialized multi-stage systems, particularly for edge cases or specific query types. The paper also doesn't adequately address potential scalability concerns or discuss how the approach might degrade with extremely large item catalogs.

### Questions
please refer to the weakness section

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work introduces a generative search framework with four key contributions:
1. Keyword-Enhanced Hierarchical Quantization Encoding: an encoding method that balances contextual understanding and collaborative information, while reinforcing relevance constraints
2. Multi-View Behavior Sequence Injection: a strategy that integrates user behavior patterns into ID representations, using both explicit and implicit prompts to improve the model’s reasoning about user preferences.
3. Preference-Aware Reward System: a personalized ranking mechanism utilizing multi-stage supervised fine-tuning (SFT) and adaptive reward modeling to enhance recommendation precision.
4. OneSearch end-to-end unified framework: deployed the first industrial-level, end-to-end generative search system that unifies all the above techniques for an eCommerce search setting with a compelling online A/B test results. 

There are some details left out in the paper: 
1. it would be valuable to discuss what challenges for production rollout, especially from infra perspective. 
2. Some experiments shared, it would be better to explain more rationals/intuitions for them. For example, in Table 2, OnlineMCA w/o ranking seems to have higher value for HR@750 for both order (30k) and click(30k) datasets, But the bold results are RQ-OPQ+Adaptive RS, inferring that we need to balance precision with recall. This was not very clear so maybe providing explanation to make it more clear. In Table 4, it seems that the combine effect of full optimizations +  RM is smaller than each individual ones. 
3. It was not clear on why online MCA was chosen as control for online A/B test, and if other benchmarks considered and why not.

### Strengths
This work introduces a generative search framework with four key contributions:
1. Keyword-Enhanced Hierarchical Quantization Encoding: an encoding method that balances contextual understanding and collaborative information, while reinforcing relevance constraints
2. Multi-View Behavior Sequence Injection: a strategy that integrates user behavior patterns into ID representations, using both explicit and implicit prompts to improve the model’s reasoning about user preferences.
3. Preference-Aware Reward System: a personalized ranking mechanism utilizing multi-stage supervised fine-tuning (SFT) and adaptive reward modeling to enhance recommendation precision.
4. OneSearch end-to-end unified framework: deployed the first industrial-level, end-to-end generative search system that unifies all the above techniques for an eCommerce search setting with a compelling online A/B test results.

### Weaknesses
Overall it was a good paper. There are some details left out in the paper: 
1. it would be valuable to discuss what challenges for production rollout, especially from infra perspective. 
2. Some experiments shared, it would be better to explain more rationals/intuitions for them. For example, in Table 2, OnlineMCA w/o ranking seems to have higher value for HR@750 for both order (30k) and click(30k) datasets, But the bold results are RQ-OPQ+Adaptive RS, inferring that we need to balance precision with recall. This was not very clear so maybe providing explanation to make it more clear. In Table 4, it seems that the combine effect of full optimizations +  RM is smaller than each individual ones. 
3. It was not clear on why online MCA was chosen as control for online A/B test, and if other benchmarks considered and why not.

### Questions
Overall it was a good paper. There are some details left out in the paper: 
1. it would be valuable to discuss what challenges for production rollout, especially from infra perspective. 
2. Some experiments shared, it would be better to explain more rationals/intuitions for them. For example, in Table 2, OnlineMCA w/o ranking seems to have higher value for HR@750 for both order (30k) and click(30k) datasets, But the bold results are RQ-OPQ+Adaptive RS, inferring that we need to balance precision with recall. This was not very clear so maybe providing explanation to make it more clear. In Table 4, it seems that the combine effect of full optimizations +  RM is smaller than each individual ones. 
3. It was not clear on why online MCA was chosen as control for online A/B test, and if other benchmarks considered and why not.

### Soundness
3

### Presentation
3

### Contribution
4
