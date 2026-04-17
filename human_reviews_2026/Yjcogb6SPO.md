# No One Size Fits All: QueryBandits for Hallucination Mitigation

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Advanced reasoning capabilities in Large Language Models (LLMs) have led to more frequent hallucinations; yet most mitigation work focuses on open-source models for post-hoc detection and parameter editing. The dearth of studies focusing on hallucinations in closed-source models is especially concerning, as they constitute the vast majority of models in institutional deployments. We introduce QueryBandits, a model-agnostic contextual bandit framework that adaptively learns online to select the optimal query-rewrite strategy by leveraging an empirically validated and calibrated reward function. Across 16 QA scenarios, our top QueryBandit (Thompson Sampling) achieves an 87.5% win rate over a No-Rewrite baseline and outperforms zero-shot static policies (e.g., Paraphrase or Expand) by 42.6% and 60.3%, respectively. Moreover, all contextual bandits outperform vanilla bandits across all datasets, with higher feature variance coinciding with greater variance in arm selection. This substantiates our finding that there is no single rewrite policy optimal for all queries. We also discover that certain static policies incur higher cumulative regret than No-Rewrite, indicating that an inflexible query-rewriting policy can worsen hallucinations. Thus, learning an online policy over semantic features with QueryBandits can shift model behavior purely through forward-pass mechanisms, enabling its use with closed-source models and bypassing the need for retraining or gradient-based adaptation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes a contextual bandit method to determine which rewriting strategy to use to  minimize the risk of hallucination for a given query. Unlike other hallucination mitigation methods, this one can be used also for closed-weight models. Experiments show that contextual bandits outperform non-contextual bandits, which in turn outperform every single static rewrite strategy.

### Strengths
[S1] The problem of hallucination mitigation is very relevant.

[S2] The framing as contextual bandit seems original and appears to deliver gains.

[S3] The method is applicable to closed-weight models, which is an advantage.

### Weaknesses
[W1] There is no analysis of the computational cost. How many forward passes/generated tokens are required to generate the vector of linguistic features for the query? The cost/effectiveness trade off should be discussed.

[W2] The comparison with open-weight baselines (Table 5) should use stronger and more recent models. What would be the results with Llama 3.1 405B Instruct? Deltas should be given relative to the no-rewrite condition with the same model, not relative to a much larger model.

[W3] The advantage of contextual over non-contextual bandit seems a bit overstated: EXP3 is a strong contender, and does not require any additional forward passes to compute the linguistic feature vector.

[W4] It is not clear thaat $s_{fuzz}$ and $s_{BLEU}$ are useful. What would happen setting $\beta = \gamma = 0$?

[W5] It might be lost in the Appendix, but there does not seem to be an ablation over the linguistic features. Are all 17 really useful?

[W6] The proposed was tested only on top of GPT-4o: it is unclear if results would generalize to other models.

### Questions
Questions, Comments, and Suggestions:

* The argument that existing hallucination mitigation methods are only applicable to open-weight models is not clearly supported. Is there a survey or some other source that can be cited to support this?
* The introduction is too long and contains too many experimental results that should come in a later section, after introducing the method.
* It would be interesting to see a comparison with some white-box methods, even considering that the proposed method is black-box.

Abstract: It would be good to specify in the astract that the bandit context is given by linguistically motivated features.

L36-37: Why would post-hoc detection and correction approaches be limited to open-weight models?

L44: 'Footprint' usually refer to a quantity (e.g. memory footprint). In this context 'fingerprint' would be more appropriate

Figure1: this example involves a disambiguation as the original intent is unclear. The correct approach in such a case would be to ask the user to clarify, rather than tacitly assume one of two possible readings.

L80-81: [DeepSeek-AI et al. 2025] does not seem like the correct citation to use for the use of RL in fine-tuning LLMs. Consider using [Christiano et al. 2017].

L99-100: What dataset and labels were used for Figure 2(a)?

L154-155: The claim seems unsupported: why is it a sign of memorization rather than e.g. of the fact that rewriting degrades the query?

L215: Where does the information required to disambiguate come from?

L232-240: do these actions require a strong LLM to be effective?

L277-297: Remark 1 and 2 feel like a distraction in this place. Perhaps move elsewhere?

Figure 3: Cumulative regret grows linearly: is learning happening?

L421-431: do non-contextual policies collapse to static policies per task? If not, why?

L1117: $\epsilon$-greedy FTRL algorithm missing.

 




[Christiano et al. 2017] Christiano, Paul F., Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. "Deep reinforcement learning from human preferences." Advances in neural information processing systems 30 (2017).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents QueryBandits, a novel framework for mitigating hallucinations in closed-source large language models (LLMs) through adaptive query rewriting.

The core idea is to frame query rewriting as a contextual multi-armed bandit problem, where each incoming query is represented by a 17-dimensional vector of linguistic features, and the system learns to select among five rewrite strategies (e.g., Simplify, Expand, Clarify) in an data-driven manner.

The reward function combines three complementary signals—LLM-as-a-judge correctness, fuzzy matching, and BLEU overlap—with weights of 0.6, 0.3, and 0.1, respectively, optimized through Pareto frontier analysis on a human-labeled dataset (ROC–AUC = 0.973).

Experiments on 13 QA benchmarks (16 scenarios) using gpt-4o-2024-08-06 show that contextual bandits, especially Thompson Sampling, outperform both static rewrite heuristics and a no-rewrite baseline in accuracy, reward, and regret.

### Strengths
1. Conceptual originality: The contextual bandit framing of query rewriting is novel and elegant.
2. Practical relevance: QueryBandits is a plug-and-play method requiring only black-box access, directly addressing a key challenge in closed-model hallucination mitigation.
3. Sound methodology: Reward calibration and validation against human labels show strong discriminative reliability.
4. Interpretability: Per-feature regression analysis (Fig. 5) provides rare interpretability, showing which linguistic traits favor specific rewrite arms.
5. Quality of analysis and presentation: Extensive evaluation (13 benchmarks) and clear visuals make the findings convincing.

### Weaknesses
1. Limited baseline coverage: While the paper’s claim of being model-agnostic is methodologically valid—since the proposed framework does not rely on gradient access or internal model parameters—it lacks empirical comparisons with other strong model-agnostic hallucination mitigation baselines such as Self-Refine (Madaan et al., 2023) and RAG-based rewriting approaches (e.g., Rewrite-Retrieve-Read, Ma et al., 2023). Including, or at least discussing, approximate results from these paradigms would help contextualize QueryBandits within the broader landscape of adaptive prompt optimization and closed-model hallucination mitigation.
2. Reward circularity concern: Both the target and judge models are GPT-4o, which may share biases. The human-labeled validation helps, but a cross-family test (e.g., using Claude or Gemini as judges) would improve confidence.
3. Feature extraction reproducibility: The paper does not specify which LLM was used to extract the 17 linguistic features, nor assess whether these features are consistent across different model runs or random seeds. As these features are central to the bandit’s learning policy, clarifying the extraction model and testing feature stability would improve reproducibility and credibility.
4. Computation and token cost: The paper does not quantify the computational or token overhead introduced by different rewrite strategies (e.g., EXPAND increasing query length, CLARIFY TERMS adding definitions) or by the feature extraction and bandit update steps. For closed-source APIs such as GPT-4o, where token usage directly translates into cost, this overhead is a key practical factor.

### Questions
1. Have you evaluated the reward function’s robustness using an out-of-family judge (e.g., Claude or Gemini )? Does the learned policy generalize across models, or must it be retrained for each target LLM?
2. Which LLM was used for the 17-feature extraction? Are the extracted features consistent across different models or seeds?
3. Could you quantify the token or latency overhead per query relative to the baseline?
4. Have you considered including stronger SOTA baselines (e.g., Self-Refine or RAG-based rewriting) for a more complete comparison?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper, we propose a method called QueryBandits for the hallucination problem in LLM answering. The method models the problem of choosing a query rewriting strategy as a contextual multi-armed bandit problem. Specifically, the model can adaptively select the most appropriate rewriting method based on input query features to reduce the occurrence of hallucinatory responses. On multiple Q&A benchmarks, QueryBandits outperforms the static prompts or no-rewrite baselines in terms of accuracy and consistency, demonstrating good generalizability.

### Strengths
1. The proposed method can be deployed on black-box LLMs with high practicality. It is completely based on query rewriting and online selection and does not rely on access to model parameters or weights.
2. Experimental validation is extensive. The authors verify both the rationality of the modeling strategy and the effectiveness of the proposed method across 16 different scenarios.

### Weaknesses
1. The theoretical basis for directly associating bandit learning with hallucination mitigation is insufficient. The paper models query rewrite selection as a contextual multi-armed bandit with composite rewards to drive online learning, but lacks proof of formal links between optimal policy existence, convergence, and the minimization of hallucination rates. The relevant arguments are mainly empirical reward separability and AUC tests, not a formal connection to LLM hallucinations.
2. Contributions specifically related to LLM hallucination mechanisms are limited. The contribution of this paper focuses on the empirical calibration of composite rewards, the accuracy improvement due to contextual adaptation, and the interpretable relevance of feature–arm weights. However, the authors do not propose new theoretical perspectives on the causes of hallucinations or a dedicated mitigation framework.
3. Direct comparison with existing hallucination mitigation methods is insufficient. The experiments focus on internal strategy evaluations such as “no rewrite / static rewrite / non-contextual bandit / contextual bandit”, but lack comparisons with common community approaches such as decoding-based, internal modification, or retrieval-enhanced methods.
4. The paper lacks analysis of algorithmic complexity and system overhead. It does not report the time complexity, online latency, or token overhead of the algorithm. While the method emphasizes utility for closed-source models, this runtime overhead is especially critical in such settings.
5. The scope of the title is overly broad. It should specify LLM hallucination rather than implying general hallucination problems across other domains.

### Questions
Please examine the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
