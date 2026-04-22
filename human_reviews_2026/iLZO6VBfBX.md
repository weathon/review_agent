# CAFE: Causally-Guided Automated Feature Engineering with Multi-Agent Reinforcement Learning

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Automated feature engineering (AFE) enables AI systems to autonomously construct high-utility representations from raw tabular data. However, existing AFE methods rely on statistical heuristics, yielding brittle features that fail under distribution shift. We introduce CAFE, a framework that reformulates AFE as a causally-guided sequential decision process, bridging causal discovery with reinforcement learning-driven feature construction. Phase I learns a sparse directed acyclic graph over features and the target to obtain soft causal priors, grouping features as direct, indirect, or other based on their causal influence with respect to the target. Phase II uses a cascading multi-agent deep Q-learning architecture to select causal groups and transformation operators, with hierarchical reward shaping and causal group-level exploration strategies that favor causally plausible transformations while controlling feature complexity. Across 15 public benchmarks (classification with macro-F1; regression with inverse relative absolute error), CAFE achieves up to 7% improvement over strong AFE baselines, reduces episodes-to-convergence, and delivers competitive time-to-target. Under controlled covariate shifts, CAFE reduces performance drop by ~4x relative to a non-causal multi-agent baseline, and produces more compact feature sets with more stable post-hoc attributions. These findings underscore that causal structure, used as a soft inductive prior rather than a rigid constraint, can substantially improve the robustness and efficiency of automated feature engineering.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents CAFE, a two-phase AutoFE framework that first learns a sparse DAG to obtain soft causal priors, then uses a cascaded multi-agent DQN to select causal feature groups and transformation operators. Empirical results on 15 tabular benchmarks show up to 7 % accuracy gain and ≈ 4× smaller performance drop under covariate shift compared with non-causal RL baselines.

### Strengths
1. Problem relevance: Distribution-shift robustness is a timely pain-point for AutoFE; explicitly injecting causal structure is novel.  
2. Technical novelty: First work to integrate causal discovery (NOTEARS-Lasso) as soft inductive bias inside a multi-agent RL search; introduces causal-group-level exploration and a causally-shaped reward.  
3. Solid evaluation: 15 public data sets, 10 strong baselines, ablations, statistical tests, shift simulation, and SHAP-stability analysis.  
4. Reproducibility: Full algorithmic details, hyper-parameters, code URL, and extensive appendix provided.

### Weaknesses
1. Causal discovery bottleneck: All downstream benefits rely on a single linear-Gaussian DAG learner. When d≫n or hidden confounders exist, graph error propagates (Table 8, −4 % gap). No fallback beyond MI weighting.  
2. Scalability ceiling: Phase-I complexity O(d³) and group screening capped at 50×50 pairs limit real-world high-dimensionality (d>1000). No distributed or incremental variant.  
3. Limited operator set: Only 15 mostly univariate/scalar operators; no domain-specific, categorical-target or high-order interaction primitives.

### Questions
Q1. Robustness to graph misspecification: Please report CAFE with ensemble of DAGs (e.g., bootstrapped NOTEARS + GES) to quantify variance reduction.  
Q2. Non-linear mechanisms: Can Phase-I be replaced by NOTEARS-MLP without hurting sample efficiency? Provide ablation on 2-3 non-linear synthetic SCMs.  
Q3. Scalability: What is the largest d CAFE can finish within 24 h on a 32-core server? Include a runtime-vs-d plot.  
Q4. Fair comparison with LLM methods: Add LIFT-FE under identical XGBoost protocol; discuss token cost vs CAFE’s extra DAG compute.  
Q5. Interpretability: SHAP stability is useful but only explains final model. How interpretable are the constructed features themselves? Provide two concrete examples where domain experts validated causal meaning.

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
4

### Summary
In this work the authors propose a causal-based framework, CAFE, for automated feature engineering. The framework has two phases. In phase 1, an off-the-shelf causal discovery method is employed. The discovery serves as a prior guiding deep Q-learning multi-agents to generate informative and robust features. The experiments demonstrate the superior performance of CAFE over previous works on 15 datasets. The robustness of the generated features against distribution shift is also verified.

### Strengths
1. *The research topic is influential:* Trustworthy automation is crucial in ML model deployment. This work focuses on automated feature engineering, a critical component of the ML pipeline, and aims to improve it via a novel causal-based approach. The research direction can thus be potentially impactful.
2. *The approach is reasonable and elegant:* The ideas of feature grouping and the cascading multi-agent architecture are neat and seem reasonable to tackle the issue of an overly large search space. According to the experiments, these designed methods work effectively as well.
3. *The relevant studies are comprehensive:* The studies include sensitivity to causal discovery, computational efficiency, ablation studies with prediction performance, and robustness to distribution shift. This comprehensive analysis will greatly facilitate follow-up research.
4. *The writing is of high quality:* The paper is well-written and clearly structured. The quality of presentation extends to the visual elements. The appendix also maintains the high standard.

### Weaknesses
1. Some details of the experiment results are unclear. Please check *Questions*.
2. The practical impact of this work seems unclear on more complex datasets, where causal discovery is inherently challenging. Although the discussion about causal discovery method selection in Appendix K is relevant, the conclusion suggests a limited real-world utility for CAFE.
3. More recently proposed causal discovery methods are not discussed. Below are potentially related works:
   - Vashishtha et al., Causal Order: The Key to Leveraging Imperfect Experts in Causal Inference. ICLR 2025.
   - Wan et al., Large Language Models for Causal Discovery: Current Landscape and Future Directions. IJCAI-25 Survey Track.

### Questions
1. Regarding Table 1 (i.e., the overall performance comparison), some results of the main competitors, ELLM-FT, are inconsistent with those in Gong et al.’s work. Specifically, the performance on SVMGuide3 and Messidor_features are 0.836 and 0.757 in this manuscript, but were 0.856 and 0.760 in Gong et al. What is the cause of this difference?
2. According to Table 1, CAFE shows superior performance over all competitors on the classification and regression tasks. My question is, what is the root cause of this superior performance? I’d argue normally we observe a trade-off between prediction performance and robustness on datasets without strong distribution shift. Are the distribution shifts on the 15 datasets strong enough to account for this gain, or do other components of CAFE contribute to the exceptional predictive ability?

### Soundness
3

### Presentation
3

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
This work introduces two-stage automatic feature engineering approach that integrates causal discovery with multi-agent reinforcement learning. The first stage learns a sparse directed acyclic graph (DAG) representing the causal relationships between features and the target label. Then the second stage uses Multi-Agent Reinforcement Learning to construct features. Experiments across 15 datasets show up to 7% performance gains.

### Strengths
- Automatic feature engineering is an interesting topic. 
- Using causal graph, the proposed CAFE method constructs features that generalize better when data distributions change, addressing a key weakness of correlation-based AFE methods.
- The CAFE method is evaluated on many datasets, compared with many baselines, including statistical, RL, and LLM-based approaches.

### Weaknesses
- The causal discovery in Phase I is computational expensive O(d³) (Eq. 20 in appendix), expecially when deals with high-dimensional data. 
- The CAFE methods relies on the construction of a "correct" causal graph. The causal discovery backend  (NOTEARS-Lasso) assumes linear additive noise. I.e., The proposed method is unreliable when handling features with complex non-linear causal effects.

### Questions
- How does CAFE perform when the underlying causal structure is highly non-linear?
- How sensitive is CAFE to the choice of causal discovery algorithm?

### Soundness
2

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
4

### Summary
This paper introduced CAFE, a causally-guided automated feature engineering framework that leverages causal discovery as inductive bias with in a multi-agent reinforcement learning paradigm. This paper proposed a framework that reformulates AFE as a causally-guided sequential decision process,bridging causal discovery with reinforcement learning-driven feature construction.  PhaseI learns a sparse directed a cyclic graph over features and the target to obtains oft causal priors, grouping features as direct, indirect,or other based on their causal influence. PhaseII uses a cascading multi-agent deep Q-learning architecture to select causal groups and transformation operators, with hierarchical reward shaping and causal group-level exploration strategies.

### Strengths
1)This paper formulate AFE as a causally-guided sequential decision process, moving beyond correlation-based heuristics to leverage stable causal mechanisms.
2)This paper introduce three core principles—soft causal inductive bias,causal structure-aware exploration, and causally shaped reward function, integrating causal discovery with adaptive feature construction through novel multi agent coordination.
3)This paper develop CAFE, a two-phase AFE framework combining causal graph discovery with cascading multi-agent reinforcement learning that strategically constructs causally informed feature transformations.

### Weaknesses
1)The lack of detailed definition of each component under the reinforcement learning framework in the paper makes it difficult to read.
2)The introduction of the framework diagram is too brief, and most of the methods are described in text.
3)One of the comparative algorithms in the paper, ELLM-FT, has already used the large model method. Does this article consider using methods related to large models for feature engineering extraction.

### Questions
see the weaknesses

### Soundness
3

### Presentation
2

### Contribution
2
