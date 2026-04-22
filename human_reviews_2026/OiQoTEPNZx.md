# Cross-Scenario Unified Modeling of User Interests at Billion Scale

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
User interests on content platforms are inherently diverse, manifesting through complex behavioral patterns across heterogeneous scenarios such as search, feed browsing, and content discovery. Traditional recommendation systems typically prioritize business metric optimization within isolated specific scenarios, neglecting cross-scenario behavioral signals and struggling to integrate advanced techniques like LLMs at billion-scale deployments, which finally limits their ability to capture holistic user interests across platform touchpoints. We propose **RED-Rec**, an LLM-enhanced hierarchical **R**ecommender **E**ngine for **D**iversified scenarios, tailored for industry-level content recommendation systems. RED-Rec unifies user interest representations across multiple behavioral contexts by aggregating and synthesizing actions from varied scenarios, resulting in comprehensive item and user modeling. At its core, a two-tower LLM-powered framework enables nuanced, multifaceted representations with deployment efficiency, and a scenario-aware dense mixing and querying policy effectively fuses diverse behavioral signals to capture cross-scenario user intent patterns and express fine-grained, context-specific intents during serving. We validate RED-Rec on hundreds of millions of users in a world-leading UGC platform through online A/B testing, showing substantial performance gains in both content recommendation and advertisement targeting tasks. We further introduce a million-scale sequential recommendation dataset for comprehensive offline training and evaluation. We hope our work could advance unified modeling of users, unlocking deeper personalization and fostering more meaningful user engagement across large-scale platforms.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes RED-Rec, an LLM-enhanced multi-scenario sequential recommendation framework. The authors also collect a million-scale multi-scenario sequential dataset from a UGC platform.

### Strengths
S1: This paper studies an important topic on multi-scenario sequential recommendation.

S2: This paper is clearly written and easy to read.

S3: This paper provides a new million-scale industrial dataset covering different scenarios and channels.

### Weaknesses
1. The authors neglect an important paper [1] on the same topic, i.e., cross-domain sequential recommendation. The proposed method LLM4CDSR of the reference paper is an important baseline method which need to be discussed.

2. From my perspective, the potential impact and contribution of this paper is not significant. First, the necessity of introducing an LLM-enriched framework is not fully illustrated. Second, the previous work HLLM has already proposed a two-tower framework with a user tower and an item tower both adopting LLM architecture. Upon that, RED-Rec purely introduce multimodal information, 2-D positional encoding, and scenario-aware queries.

3. The experiments are only conducted on industrial datasets without validation on public representative datasets like Amazon.

4. The presentation needs improving. For example, Figure 3 is cluttered and unclear. Besides, the Appendix is missing and Appendix D is not mentioned.

[1] Bridge the Domains: Large Language Models Enhanced Cross-domain Sequential Recommendation, SIGIR 2025

### Questions
1. The Appendix mentioned is not included in the manuscript and there is no supplementary material.

2. In the experiment part, is it fair to compare with the pre-trained model variants (i.e., '-pt'), which are trained on a large-scale online dataset? This is because such implementation introduces additional information.

3. Some details of experiments are missing. For example, does the implemented HLLM use multimodal information?

4. Please refer to the third weakness. Will the authors include experimental results on public multi-scenario dataset? Generally this paper would be a better fit for the industry track of a data mining conference (e.g., ADS track of KDD).

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
4

### Summary
The paper presents RED-Rec, a large-scale LLM-enhanced hierarchical rec framework designed to unify user interest modeling across multiple scenarios or domains. Traditional recsys are typically scenario-specific and fail to capture cross-scenario dependencies, leading to fragmented user understanding. To address this, RED-Rec introduces a two-tower LLM-powered architecture that integrates textual and visual features through multimodal encoders and employs a 2-D dense mixing policy to fuse behavioral signals along temporal and scenario dimensions. It also incorporates scenario-aware queries to express fine-grained user intents. The authors construct a million-scale multi-scenario sequential dataset from a large UGC platform and validate RED-Rec through both offline benchmarking and online A/B tests on hundreds of millions of users. Results show significant performance gains over strong baselines, demonstrating the feasibility of unified cross-scenario recommendation at an industrial scale.

### Strengths
1. The paper is clearly written and logically well-structured.
2. Introducing sequential modeling into the multi-scenario domain is indeed a very interesting problem.

### Weaknesses
1. The related work section is not comprehensive, and many studies on multi-scenario recommendation have been overlooked, such as the work in this project [1].
2. In line 212, what does H_u represent?
3. How exactly is the model trained? The authors only mention in line 323 that NCE is used as the main objective, but the training procedure remains unclear to me.
4. Figure 4 refers to negative sampling multiple times, yet this is not explicitly described in the methodology section.
5. Where can the appendices mentioned in the text be found? I could not locate them.
6. Section 5.3 omits several classic multi-scenario recommendation algorithms, such as STAR [2] and M2M [3].

[1] https://github.com/Xiaopengli1/Scenario-Wise-Rec

[2] Sheng X R, Zhao L, Zhou G, et al. One model to serve all: Star topology adaptive recommender for multi-domain ctr prediction[C]//Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 2021: 4104-4113.

[3] Zhang, Qianqian, et al. "Leaving no one behind: A multi-scenario multi-task meta learning approach for advertiser modeling." Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining. 2022.

### Questions
please refer to the weakness section

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on the unified modeling of multi-scenario recommendation. The authors propose RED-Rec, a rather complex systematic modeling framework that aims to unify the modeling of various features across scenarios such as homefeed, ads, and search, mainly through semantic encoding and scenario-aware processing. The framework leverages LLM-based item and user encoders, along with a dense mixing policy to integrate multi-scenario signals. Experiments on an industrial dataset and online A/B tests suggest the method is useful and potentially highly effective in industrial settings. However, the approach is more of a comprehensive industrial implementation that integrates various existing technologies rather than introducing a clearly innovative model.

### Strengths
- The paper addresses a real-world and practical issue in large-scale multi-scenario recommendation systems.
- The proposed unified modeling approach is well-structured and demonstrates performance gains in industrial-scale deployment.
- Empirical validation includes both offline experiments and large-scale online A/B testing, providing strong practical evidence.
- The design effectively integrates multiple technologies, indicating high engineering value for industry applications.

### Weaknesses
- Limited novelty — the modeling method appears more like an aggregation of existing techniques (LLM-based encoders, multi-scenario mixing) into a unified industrial system rather than introducing fundamentally new algorithms.
- Missing comparisons with recent, relevant multi-scenario recommendation baselines (e.g., STAR, APG, AdaSparse, HierRec), which weakens the claim of state-of-the-art performance in this subfield.
- All appendix links are invalid, preventing reviewers from accessing potentially important supplementary definitions, dataset details, and experimental settings. This significantly impacts reproducibility.
- No anonymous, reproducible code or dataset provided at review time, despite the promise of releasing them upon acceptance. This limits the ability to verify implementation details.
- The evaluation leans heavily on proprietary industrial data; while practical, this limits independent verification and reduces the openness of the contribution.

### Questions
- The modeling method is more like an industrial implementation that integrates various technologies rather than an innovative research contribution.
- The paper belongs to the field of multi-scenario recommendation, but the experiments do not compare against recent multi-scenario recommendation methods such as STAR, APG, AdaSparse, HierRec, etc., only against some general recommendation methods, which is inappropriate.
- Although the paper states that the code will be made open-source upon acceptance, no anonymous and reproducible code is currently provided for review.
- All links to the appendices are invalid, preventing reviewers from accessing any of the appendix contents.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
RED-Rec is designed to address the limitations of traditional RS that operate in isolated silos (e.g., separate models for feed, search, and ads). The authors argue that this siloed approach fails to capture a holistic understanding of user interests, which manifest across these different interaction contexts.

RED-Rec is thus proposed as a unified, user-centric framework tailored for billion-scale industrial deployments. Its architecture is an LLM-enhanced hierarchical two-tower model that learns comprehensive user and item representations by synthesizing behavioral signals from multiple scenario

key contributions include:
* LLM-powered encoder, which area two-tower structure where both the user and item encoders are powered by Large Language Models, enabling rich semantic representations from content and user histories
* 2-D dense mixing/qerying policy, which is designed to effectively fuse behavioral signals from different scenarios. It operates along two axes <scenario, time> to address data imbalances (e.g., more feed interactions than ad clicks) and capture cross-scenario user intent. It also uses "scenario-aware queries" to express fine-grained, context-specific user interests during serving. I think this second contribution is more interesting.

The framework was validated through extensive offline experiments and large-scale online A/B tests on a major UGC platform, where it showed substantial performance gains in both content recommendation and advertising

### Strengths
Originality:

"2-D dense mixing and querying policy" is a novel and specific technical contribution designed to handle the practical challenges of multi-scenario data. By fusing signals along both temporal and scenario axes, it explicitly addresses data imbalance (e.g., more feed interactions than ad clicks), ensuring that infrequent but valuable user signals are not lost.

Quality
* The paper's most significant strength is its validation through online A/B tests on a world-leading UGC platform supporting hundreds of millions of daily users.
* They contribute to the quality of future research by introducing a new, million-scale multi-scenario sequential dataset.

Clarity:  The paper is mostly clear-written. 

Significance
* with the industrial a/b testing result, the work shows a high practical impact.

### Weaknesses
* The paper's title and abstract heavily emphasize the "LLM-enhanced" nature of the framework. However, the experimental section lacks a crucial baseline: a comparison against a similar architecture that uses a non-LLM encoder (e.g., a standard Transformer or GRU operating on ID embeddings). Without this comparison, it is difficult to quantify the actual performance gain attributable to the expensive LLM component versus the gains from the unified architecture itself. The current ablation study only explores variations of the RED-Rec model, not its core components against simpler alternatives. A non-LLM would clearly demonstrate the value added by the language model.

* The paper presents a sophisticated two-tower hierarchical architecture. While powerful, it also introduces significant engineering complexity. The paper mentions "system-level optimizations enabling stable, low-latency online deployment" but provides no details on what these optimizations are. This makes it difficult to assess the full cost and engineering effort required to make such a system viable in production. While some details may be proprietary, providing a high-level overview of the types of optimizations employed (e.g., model quantization, caching strategies, asynchronous embedding generation) would greatly improve the paper's practical value and reproducibility.

* Clarity: the paper is mostly clear though the implementation of "2-D dense mixing" policy is not fully detailed. The paper states that the Merge(·) function "deterministically fuses events - sorting by timestamp and concatenating per a fixed scenario order," but the clarity could be improved by providing a more explicit description of the merging logic, including the "fixed scenario order" and the rationale behind it.

* The paper's primary claim is that it advances "unified modeling" for multi-scenario recommendation. However, the baselines used in the experiments are primarily strong single-scenario sequential recommendation models, not models explicitly designed for the multi-scenario setting. The related work section mentions several such models, but they are not included in the comparison.

* The online A/B test results are a major strength, but the details provided are sparse. The authors should provide more context for the A/B test. Describing the control group and reporting on the trade-offs - for example, did the focus on ads have any negative impact on organic content engagement?

### Questions
See Weakness section above

### Soundness
3

### Presentation
2

### Contribution
3
