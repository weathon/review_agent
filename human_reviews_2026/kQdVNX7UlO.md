# RADAR: A Risk-Aware Dynamic Multi-Agent Framework for LLM Safety Evaluation via Role-Specialized Collaboration

- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Existing safety evaluation methods for large language models (LLMs) suffer from inherent limitations, including evaluator bias and detection failures arising from model homogeneity, which collectively undermine the robustness of risk evaluation processes. This paper seeks to re-examine the risk evaluation paradigm by introducing a theoretical framework that reconstructs the underlying risk concept space. Specifically, we decompose the latent risk concept space into three mutually exclusive subspaces: the explicit risk subspace (encompassing direct violations of safety guidelines), the implicit risk subspace (capturing potential malicious content that requires contextual reasoning for identification), and the non-risk subspace. Furthermore, we propose RADAR, a multi-agent collaborative evaluation framework that leverages multi-round debate mechanisms through four specialized complementary roles and employs dynamic update mechanisms to achieve self-evolution of risk concept distributions. This approach enables comprehensive coverage of both explicit and implicit risks while mitigating evaluator bias. To validate the effectiveness of our framework, we construct an evaluation dataset comprising 800 challenging cases. Extensive experiments on our challenging testset and public benchmarks demonstrate that RADAR significantly outperforms baseline evaluation methods across multiple dimensions, including accuracy, stability, and self-evaluation risk sensitivity. Notably, RADAR achieves a 28.87\% improvement in risk identification accuracy compared to the strongest baseline evaluation method.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a new risk evaluation framework for LLMs that decomposes the risk concept space into three subspaces: explicit, implicit, and non-risk. The authors propose RADAR, a multi-agent collaborative evaluation framework that uses dynamic update mechanisms to self-evolve risk concept distributions. The framework outperforms baseline evaluation methods in accuracy, stability, and self-evaluation risk sensitivity, with a 28.87% improvement in risk identification accuracy.

### Strengths
+ The design ensures comprehensive coverage of different risk types and mitigates bias through structured collaboration.
+ Evaluation goes beyond simple accuracy to include stability and sensitivity.

### Weaknesses
- The typical contribution of each agent to the final outcome is not clear.
- A few technical details are not clearly explained.

### Questions
1. Section 4.1
> This role should be instantiated with the strongest available evaluator.

I did not find how CAC (and also other roles) are actually chosen in the experiments. Also, what do you mean by the strongest?

2. Section 4.2, formula 6: the stubbornness coefficient lambda should have an impact on the final evaluation. There should be more discussion on it in the experiment section.

3. Section 5.1: The validity and completeness of the Hard Case Testset is not well justified.

### Soundness
3

### Presentation
3

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
This paper introduces RADAR, a multi-agent collaborative framework for evaluating the safety of large language models. The authors identify two key limitations in existing safety evaluation methods: evaluator-heterogeneity bias and self-evaluation bias. To address these issues, they propose a theoretical framework that decomposes the latent risk concept space into three mutually exclusive subspaces (explicit risk, implicit risk, and non-risk). RADAR employs four specialized roles—Security Standards Auditor (SCA), Vulnerability Detector (VD), Critical Argument Challenger (CAC), and Holistic Arbitrator (HA)—that engage in multi-round debate to reach safety evaluation decisions. Experiments on a proprietary "Hard Case Testset" and three public benchmarks demonstrate that RADAR achieves substantial improvements over baseline methods, including a 28.87% improvement in risk identification accuracy.

### Strengths
- The decomposition of the risk concept space into explicit, implicit, and non-risk subspaces is novel and well-motivated for safety evaluation tasks
- The integration of specialized role assignment with dynamic risk-concept distribution updates (Equations 5-6) provides a principled theoretical foundation for multi-agent collaboration
- Demonstrates consistent improvements across diverse datasets (Hard Case Testset, Red Team, Implicit Toxicity, DiaSafety)

### Weaknesses
- While the authors acknowledge that "multi-round debate mechanism incurs considerable computational overhead", there is no quantitative analysis of actual inference time comparisons with baseline methods or token consumption and associated API costs, this is critical for practical deployment considerations.
- While the paper cites recent multi-agent collaboration work, there are no direct comparisons with other multi-agent evaluation approaches (e.g., simple multi-agent voting, other debate frameworks). The ablation "RADAR w/o Role-Setting" partially addresses this, but more comprehensive comparisons would strengthen the claims.
- The paper mentions two error categories (Section 5.3) but provides no detailed failure case analysis. Understanding when and why RADAR fails would strengthen the contribution.

### Questions
1. Given the computational overhead, how feasible is RADAR for real-time content moderation scenarios? Are there strategies to reduce latency (e.g., early termination, cached evaluations)?
2. How were the optimal values T=3 and N=1 determined? Was grid search performed, and what is the sensitivity around these values?
3. How does RADAR's performance compare with expert human evaluators on the same test cases?

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
3

### Summary
This paper introduces RADAR, a multi-agent collaborative framework for evaluating LLM safety, built upon a theoretical foundation. Specifically, RADAR assigns four specialized roles — Security Standards Auditor, Vulnerability Detector, Critical Argument Challenger, and Holistic Arbitrator—to perform comprehensive safety assessments. The authors validate the effectiveness of RADAR through experiments conducted on both a proprietary dataset and three publicly available datasets, demonstrating its strong performance in evaluating LLM safety.

### Strengths
1. This paper develops a multi-agent framework for LLM safety, grounded in strong theoretical insights.  
2. The authors tackle the crucial challenge of ensuring LLM safety, which is essential for the responsible use of such models.  
3. The proposed approach demonstrates substantial improvements in LLM safety performance.

### Weaknesses
1.  The paper’s theoretical foundation offers limited insight and appears superficial, as it merely follows prior work [1]. The authors introduce specialized roles without sufficient justification. Since the paper’s claimed contribution is based on this theoretical foundation (Lines 72–73), the overall contribution seems weak.  
* Moreover, the paper does not actually establish a new theoretical framework; rather, it reuses results from existing theoretical analyses, overstating its contribution by framing it as a novel theoretical development.  

2. The discussion on the limitation of single evaluators, referred to as Self-Evaluation Bias, requires further clarification and deeper analysis. This limitation could be easily mitigated by using a different family of evaluators, which weakens the justification for introducing the proposed multi-agent collaborative framework.  

3. The use of a proprietary dataset to demonstrate the effectiveness of the proposed method seems to be problematic. The authors rely on this dataset to validate the effectiveness of RADAR, raising doubts about the credibility of the results. Since no analysis of the dataset’s quality or relevance is provided, its use appears inappropriate. Therefore, the authors should either evaluate the method using open-source datasets or provide a clear justification for employing the proprietary dataset.

4. The overall writing quality of the paper needs improvement. For instance, in the methodology section, the SCA component is described as using a rule-based approach to detect explicit safety violations, yet the specific rules applied are not clearly explained. Similarly, Equation 5 lacks a detailed description of how $\alpha$  operates. Overall, the paper remains at a high-level description without providing sufficient technical details, which makes it difficult to fully understand the proposed method.

[1] Multi-LLM Debate: Framework, Principals, and Interventions. Estornell et al., NeurIPS'24.

### Questions
Overall, this paper appears to merely introduce a theoretical framework from the previous work and use it as a justification for constructing a multi-agent collaborative framework, which limits the significance of its contribution. Moreover, the use of a proprietary dataset without sufficient explanation or analysis is inappropriate for demonstrating the method’s effectiveness. The authors should either justify the use of this dataset or provide a thorough analysis of it. Therefore, I recommend rejection of this paper.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a multi-agent collaborative and debate-style safety evaluation methodology strategy for LLMs. The multi-agent team consists of a Safety Criterion Auditor (SCA), which looks for explicit harmful/unsafe content, a Vulnerability Detector (VD), which looks for implicit harm/unsafe content, a Counter Argument Critic (CAC), to check, counter and critically examine the statements of the former and a Holistic Arbiter (HA) as a final judge. The authors also put forth a theoretical backing for their methodology and evaluated several strong LLM and safety-finetuned models to show the efficacy of their technique.

### Strengths
The paper presents an interesting methodology to evaluate safety using multiple dimensions using a collaborative agentic approach. 

I appreciate the theoretical aspect where the authors cleanly define implicit and explicit concept spaces and directly use them in their framework in the form of SCA and VD. 

The RADAR framework shows significant improvement from the previous SoTA on public benchmarks.

### Weaknesses
I find the paper slightly confusing in some places. For example, the authors provide a detailed formulation of the stubbornness coefficient and CAC feedback, but I don’t see it directly implemented or analysed in the paper. Honestly, I feel the math in the paper does not directly correlate with the experiments and is unnecessary in certain places. Bottom line, this paper discusses the safety evaluation from a multi-agent “jury-and-judge” perspective, rather than the standard single LLM-judge.

Further, it is not clear how the authors chose the LLMS for SCA, VD, CAC, and HA. Most of them are top proprietary models. It would be interesting to see if OSS model(s) can produce good results, as these closed-source models can be expensive in the long run. Additionally, running 4 LLMs for multiple turns is itself an expensive process. It would be interesting to see if all these different personalities can be discretely condensed into a single model, i.e., with multiple disjoint personalities. 

The authors strongly vouch for CAC in section 4.2, but there is no ablation for it, i.e., what happens when you remove CAC from the system, and how the “echoes” hurt the performance across turns. 

The authors vigorously oppose and discuss the self-bias/self-evaluation in the paper, but the ablation for it is missing, i.e., what happens with all the models in RADAR are the same? Will it amplify the echo exponentially? What happens when such a system evaluates the same, similar family model, or a different model?

### Questions
Was there any analysis of cases where the model accepted its initial incorrect evaluation and changed it, maybe due to the intervention of CAC or without it as well? Since the setup is heavily reliant on CAC, and the authors also mention using the strongest available evaluator, I urge some analysis to be done around it. Similarly, in how many cases was the model “stubborn” and refused to change its output despite intervention by CAC?

I am surprised why the Llama-3.1-8B-Instruct model was used for CAC, while the rest are the top large-scale models? What kind of evaluation was done to select this and to show that this is the strongest evaluator? 

What happens when you use test-time scaling or reasoning for a single evaluator, and then how does it compare to RADAR? Were SCA (GPT-5) and VD (Grok-3) also set to reason during their turns? I feel a study of RADAR-reasoning vs RADAR-non-reasoning teams would be an interesting takeaway.

I also feel Table 2 is slightly unfair, as RADAR has an undue advantage of multiple LLMs debating for multiple turns, while the others are standalone single-turn evaluators. For some consistency, an iterative reflection-based evaluation should be done by the other baselines as well. 

Further, another interesting study would be when RADAR is evaluating a model that is present in its team, especially when it is the CAC. How does the self-bias/self-evaluation affect the process in this case?

### Soundness
2

### Presentation
2

### Contribution
2
