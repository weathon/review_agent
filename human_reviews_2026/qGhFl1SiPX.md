# The Invisible Leash? Why RLVR May or May Not Escape Its Origin

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 4, 4, 10

## Abstract
Recent advances in large reasoning models highlight Reinforcement Learning with Verifiable Rewards (RLVR) as a promising method for enhancing AI's capabilities, particularly in solving complex logical tasks. However, it remains unclear whether the current practice of RLVR truly expands a model's reasoning boundary or mainly amplifies high-reward outputs that the base model already knows for improved precision. This study presents an empirical investigation that provides fresh insights into the potential limits of the common practice of RLVR. We examine how, under current training conditions, RLVR can operate as a support-constrained optimization mechanism that may restrict the discovery of entirely original solutions, remaining constrained by the base model's initial distribution. We also identify an entropy–reward trade-off: while the current RLVR recipe reliably enhances precision, it may progressively narrow exploration and potentially overlook correct yet underrepresented solutions. Extensive empirical experiments validate that while the current RLVR recipe consistently improves \texttt{pass@1}, \textit{the shrinkage of empirical support generally outweighs the expansion of empirical support under larger sampling budgets}, failing to recover correct answers that were previously accessible to the base model. Interestingly, we also observe that while RLVR sometimes increases token-level entropy, it results in greater uncertainty at each generation step and declining answer-level entropy. This indicates that these seemingly more uncertain paths ultimately converge onto a smaller set of distinct answers. Taken together, we reveal potential limits of the current RLVR recipe in extending reasoning horizons. Breaking this invisible leash may require future algorithmic innovations such as explicit exploration mechanisms or hybrid strategies that seed probability mass into underrepresented solution regions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper shows that RLVR rarely lets a model discover correct solutions outside the base model’s reach; it mainly refines what the base already knew.  
Across math and non-math tasks, RLVR raises single-sample accuracy but shrinks the set of recoverable correct answers, so high-k performance drops.  
Token-level entropy may rise, yet answer-level entropy falls, revealing local randomness paired with global collapse.  
Authors also prove RLVR cannot give non-zero probability to any solution the base model gives zero, establishing an “invisible leash.”

### Strengths
* The experiment is relatively detailed and covers a large number of LLMs.
* The paper is well-written with a clear structure, making it easy to understand.

### Weaknesses
* The conclusions and findings in the article have already been presented in recent works [1], with no newer insights or new algorithms proposed. 
* The concept of "Support of Correct Completions" proposed in the paper has no direct connection with the reasoning performance of large models.

[1] Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?

### Questions
* What is the relationship between "Support of Correct Completions" and the inference performance of large models?
* Is "Support of Correct Completions" equivalent to Pass@k?
* It is suggested to put forward the differences from related work and unique contributions.

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
3

### Summary
The paper studies whether RLVR truly expands a model’s reasoning capacity or mostly sharpens probability mass around solutions already within the base model’s action space. The authors provide formalized metrics (SRR, NDR, SDS, NSCR) to quantify preservation, shrinkage, and expansion of model answers. Across diverse benchmarks, they find high support retention but very limited genuine discovery; shrinkage generally outweighs expansion. They further report a diference between token-level entropy and answer-level entropy, arguing that RLVR increases precision while narrowing diversity.

### Strengths
1. The paper conducts comprehensive empirical studies on diverse model scales, domains, and benchmarks. Presents solid experimental data and detailed analysis to support the theories presented.
2. The introduction of metrics like SDS and NSCR addresses a gap in prior work, which often relies on pass@k.
3. The analysis of the difference between token-level entropy and answer-level entropy is novel and insightful.

### Weaknesses
1. While the analysis is strong, the paper does not propose or test a specific exploratory mechanism to break the leash. It is more of an evaluation study than a methodological contribution and lacks innovation.
2. The perplexity comparison of DeepSeek and Claude trajectories in Table 2 may be confounded by differences in style and format. Although the authors mentioned this in the paper, they did not provide quantifiable values ​​to evaluate the impact of style. The conclusions drawn from the results in Table 2 are not convincing.
3. Although the authors compared and analyzed numerous open-source RLVR models, they employed different RL algorithms, hyperparameters, training durations, and training data. The precise factors that contribute to the "shrinkage", and how they contribute, remain unclear, making it difficult to identify the primary factors.

### Questions
1. Is it possible to conduct some experiments to observe the changing of SDS/NSCR under the condition of fixed initial policy model, training data and RL algorithm?
2. Can you perform a sensitivity analysis on the choice of the ε parameter and report the confidence intervals and significance tests of SRR/NDR/NSCR?
3. Can you decouple the effects of “style” and model capability on perplexity in the experiments in Table 2?

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
This paper investigates the role and limitations of Reinforcement Learning with Verifiable Rewards (RLVR) in large reasoning models, exploring whether RLVR extends the reasoning boundary or merely reinforces the base model's known patterns through empirical and theoretical analyses. 
It proposes the concepts of "empirical support dynamics" along with a quantitative framework, and conducts experiments across multiple models and tasks. 
The findings reveal that current RLVR essentially functions as a support-constrained optimization mechanism, limited by the initial distribution of the base model. Although it can improve single-sample accuracy, the shrinkage of empirical support generally outweighs its expansion, and an entropy-reward trade-off arises.
Additionally, the study points out that breaking this "invisible leash" requires explicit exploration mechanisms or hybrid strategies, providing directions for RLVR optimization and the expansion of model reasoning capabilities.

### Strengths
- The research topic of this paper is important. Exploring the role and limitations of RLVR in the training paradigm of large models is a valuable topic.
- Some findings of this paper are interesting, such as the shrinkage of empirical support generally outweighing its expansion, and the existence of an entropy-reward trade-off.

### Weaknesses
- The core claims are not fully supported by the experiment results. RL training is a complex process with multiple coupled factors, and the observed phenomena cannot be directly attributed to RLVR merely through comparing model performance before and after training. It is suggested that the authors supplement experiments to further support the conclusions: under the condition that other factors (such as the base model and training data) are strictly controlled to be identical, compare the impacts of RLVR training methods and non-RLVR training methods on model performance.  
- The end-to-end performance comparison has weak credibility. It is recommended that the authors supplement the changes of metrics during RL training to further analyze the impact of the RLVR process on model performance.  
- There is a problem with the experiment conclusions of Table 2. For the "Top" and "Middle" columns, using the reasoning traces of one model as the reference and calculating the perplexity of the other model will inevitably result in a higher perplexity of the latter, which cannot prove the conclusions claimed by the authors.

### Questions
See Weaknesses

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper investigates whether RLVR truly expands reasoning models' capabilities or merely amplifies existing high-reward outputs. Through empirical analysis, the authors find that current RLVR practices improve precision (pass@1) but narrow solution diversity, functioning as a support-constrained optimization that remains tethered to the base model's distribution. They identify an entropy-reward trade-off where token-level uncertainty increases while answer-level diversity decreases, causing models to lose access to correct solutions previously available in the base model. These findings suggest current RLVR recipes impose fundamental limitations on reasoning expansion, requiring future innovations like explicit exploration mechanisms to break this "invisible leash."

### Strengths
**\[S1\] Meaningful and timely research topic**  
The paper addresses a highly relevant and timely research question, the effect of RLVR on reasoning capabilities, and conducts a rigorous analysis using various metrics. This makes the work both empirically and conceptually valuable to the research community.

**\[S2\] Offer results with a systematic framework**  
By introducing SRR, NDR, SDS, and NSCR, the paper offers a systematic framework for quantifying how RLVR reshapes the model’s reasoning space. This transforms an abstract discussion about “capability expansion” into measurable, interpretable quantities, setting a foundation for future comparative studies.

**\[S3\] Empirical rigor and statistical validity**  
The authors acknowledge the inherent limitations of conventional metrics like pass@k but compensate through extensive empirical sampling and thorough statistical validation, including experiments with up to 8192 samples per prompt. This large-scale approach enhances the reliability of the findings and lends weight to their conclusions. Also, providing theoretical proofs in Appendix C provides formal mathematical grounding that strengthens the empirical observations and demonstrates the rigor of their analytical framework.

**\[S4\] Cross-domain generalizability**  
The analysis extends beyond mathematical reasoning to include non-math reasoning and even multimodal tasks, demonstrating that the observed precision-diversity trade-off is domain-agnostic.

### Weaknesses
**\[W1\] Limited model diversity**  
Including a wider range of open-weight baselines such as Llama or OLMo would provide stronger evidence that the identified patterns are model-agnostic rather than implementation-specific, thereby increasing the robustness and external validity of the conclusions.

**\[W2\] Insufficient analysis of exceptional cases**  
Although the overall experimental results exhibit a clear global trend, certain exceptions deviate from this pattern. However, the paper lacks an in-depth analysis explaining these anomalies. A more thorough examination of such exceptional settings, identifying potential causes or contributing factors, would strengthen the empirical support for the claimed characteristics of RLVR and enhance the overall robustness of the conclusions.

### Questions
**\[Q1\]** The paper demonstrates that RLVR generally leads to support shrinkage, yet some Reasoning Gym tasks appear to exhibit genuine expansion. Do the authors have hypotheses about what task characteristics or properties might explain why certain tasks show genuine expansion while others do not?

### Soundness
4

### Presentation
4

### Contribution
4
