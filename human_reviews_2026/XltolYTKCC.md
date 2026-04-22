# Where Reasoning Fails: Step-wise Confidence Attribution in Black-box LLMs

- Avg Score: 4.80
- Decision: Reject
- Scores: 2, 6, 4, 6, 6

## Abstract
Large Language Models (LLMs) have achieved strong performance on complex reasoning tasks by generating step-by-step solution traces, but diagnosing where a reasoning trace might fail remains difficult. Confidence estimation (CE) provides reliability signals but is usually restricted to the final answer, offering only coarse diagnostics. While recent studies have explored stepwise diagnostics, existing methods rely on white-box access, such as token-level logits or fine-tuned models, which are infeasible for closed-source LLMs.
We introduce Stepwise Confidence Attribution, a black-box framework for diagnosing errors, requiring only access to generated reasoning traces.
Stepwise confidence attribution applies the Information Bottleneck (IB) principle to assign confidence scores at the step level, treating consensus structures across correct solutions as anchors of reliable reasoning with high confidence. Steps that do not align with these consensus patterns are assigned lower confidence.
We propose two complementary methods: (1) a non-parametric overlap-based approach (NIBS) that measures consistency without graph context, and (2) a Graph-based IB model (GIBS) that learns subgraphs through a differentiable mask to capture structural variability.
Through extensive experiments on mathematical reasoning and multi-hop question answering, we show that our framework reliably identifies low-confidence steps strongly correlated with reasoning errors. Moreover, incorporating step-level CE improves overall reasoning accuracy, yielding up to an 12.3\% accuracy gain. Our framework provides a practical diagnostic tool for enhancing the reliability of LLM reasoning. Code can be found at https://anonymous.4open.science/r/ICLR_2026_-2801.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes methods on step-wise confidence estimation of black-box LLMs in reasoning tasks. Two confidence estimation are proposed, which are based on measuring the similarity of different steps among various sampled reasoning traces. Empirical evaluations are conducted against various baseline methods, where the proposed methods show better performance. Moreover, the utility of the proposed methods is demonstrated in guiding LLM self-correction, which is more effective than self-correction based on final prediction correctness only.

### Strengths
1. The proposed step-wise confidence estimation methods are principled and intuitive, which could be effective given strong implementations.

2. The empirical evaluation is thorough, which aims to demonstrate the effectiveness of the proposed methods through direct evaluations and downstream applications.

### Weaknesses
1. In general, this work lacks clarity in its problem formulation, descriptions of evaluation setup and implementation details. Specific issues are mentioned in the next sections.

2. The introduced confidence estimation methods introduce dependency on external models and/or additional requirements on the format of model outputs, which could make them brittle and hard-to-generalize.

    (a) The NIBS method requires either BERT embeddings or an NLI model. Therefore, its effectiveness depends on the reliability of the embeddings and/or the NLI model. (There is no clear description regarding the BERT embeddings and the NLI model. Which BERT was used? Was the NLI model trained? Which model is the NLI model based on?)

   (b) The GIBS method requires the model to output structured reasoning traces or a post-processor for parsing the reasoning trace. (There is no specific information in the paper on which approach was actually used.) If the former approach is used, it requires the reasoning model to be capable enough to generate structured reasoning traces and might alter model behavior and introduce computational overhead. If the latter is used, an additional post-processor is required for the method. Moreover, the GIBS method also requires an NLI model. (Again, the specific information of the used NLI model is lacking). For annotating the entailment, two hyper-parameters are introduced ($\tau_{e}$, $\tau_{v}$), of which the values seem to be arbitrarily set (Line 787). How sensitive is the method regarding the values of these?

3. The evaluation setup and the main objective of the step-wise confidence estimation should be better clarified.

    (a) Is there a definition of "gold-standard" step-wise confidence? Is it the LLM-predicted probability of the steps? This is critical since it determines how the confidence estimation methods should be evaluated.

    (b) Related to (a), in Section 5.2 (Table 1), how are the AUROC and ECE, etc. computed? These metrics require accuracy. Is the accuracy computed at the step level? If it is, how is the step-level accuracy computed?

### Questions
Please see the Weaknesses section.

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
- This paper proposes two confidence attribution methods called NIBS and GIBS. NIBS is a non-parametric method while GIBS employs graph-based modeling method to check the confidence of the steps in the reasoning chain. They are based on the information bottleneck principle. Authors conduct experiments on several LLM backbones and different reasoning datasets to present the effectiveness of their design.

### Strengths
- This is a novel paper discuss an interesting problem Confidence Attribution, which is important for LLM reasoning. Authors design new black-box methods to address the disadvantages of current white-box methods.
- The design of NIBS and GIBS demonstrates strong innovation, and the authors have conducted a substantial amount of work in their experiments.

### Weaknesses
- Regarding the process of constructing a graph, the author provides a rather brief introduction in the text. I believe this process could be explained more effectively with the inclusion of additional concrete examples.
- In my understanding, the model designed by the authors should be a GNN. Based on this, they developed subsequent training and inference methods for NIBS. However, the paper does not elaborate on specific details such as the choice of GNN backbone. I only confirmed this after reviewing the code.
- The method is tested under several datasets which all have definite answer labels, which may be unrealistic in certain scenarios (e.g., open-domain generation). Although the authors note this is a reasonable assumption for diagnostic tasks, they do not discuss the impact of label noise or partially correct answers.

### Questions
- Can the author explore how different graph construction methods impact performance? For instance, what performance differences arise when processing reasoning steps as homogeneous graphs vs heterogeneous graphs?
- Did the authors test the impact of different GNN backbones on performance? Based on the code, it appears they only experimented with the GCN model.
- Can the research topic being studied by the author positively impact RL-based LLM post-training? For example, by utilizing the model you propose to estimate the confidence level of the reasoning chain during the RL rollout process, thereby yielding additional benefits.

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
3

### Summary
This paper proposes a step-wise confidence attribution (SCA) method for diagnosing large language model reasoning traces. The key idea is to identify consensus steps derived from correct solutions and measure how well individual steps in a given reasoning path align with these consensus steps. The authors present two methods: a non-parametric overlap between reasoning step and consensus steps and a graph-based method which represents reasoning traces as graphs and learns to select subgraphs that align with consensus reasoning graph. Experiments on mathematical reasoning and multi-hop QA datasets demonstrate that the approach can identify erroneous steps and improve final-answer accuracy through error correction.

### Strengths
- Step-wise confidence attribution addresses a critical need in making LLM reasoning more interpretable and reliable. The ability to localize where reasoning fails is valuable for model analysis.
- The idea of using consensus steps from correct solutions as anchors for confidence estimation is intuitive and reasonable.

### Weaknesses
- The method assumes access to groundtruth answers to construct consensus steps. While this is acceptable for diagnostic evaluation, it undermines claims about error correction. Most baselines in Table 1 do not rely on such supervision, making comparisons somewhat unfair. In Section 5.3, the correction setting implicitly assumes oracle access to correctness feedback. This limits the practicality of the proposed use cases.
- The paper associates "confidence" with "contribution to correct answer," which are 2 distinct concepts. A reasoning step can be confident but incorrect (high model certainty, wrong conclusion). Conversely, a correct step might show low confidence if it's unusual or creative. The method assigns scores based on alignment with consensus from correct trajectories, which measures attribution to correctness rather than confidence.

### Questions
1. Why restrict consensus anchors to correct solutions? Have you attempted using consensus from all sampled trajectories rather than correct-only? Could frequent patterns across both correct and incorrect solutions provide better signal?
2. How sensitive is the confidence attribution to the number and diversity of sampled trajectories used for consensus construction?
3. Can the proposed framework operate without ground-truth correctness labels such as self-consistency?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a framework for diagnosing reasoning errors in large language models without requiring white-box access. The authors propose a Stepwise Confidence Attribution (SCA) framework, which assigns confidence scores to individual reasoning steps using only generated traces and final correctness labels. Specifically, two implementations are presented: NIBS, a non-parametric overlap-based method, and GIBS, a graph-based model leveraging the Information Bottleneck (IB) principle for structure-aware confidence attribution. Experiments across reasoning datasets (GSM8K, Math, MoreHopQA) show that GIBS outperforms baselines and improves reasoning accuracy. The framework also enables targeted self-correction and exhibits out-of-distribution robustness.

### Strengths
1. Exploring step-wise attribution frameworks for black-box models is a challenging task, and integrating information theory appears promising.

2. Cross-domain generalization experiments verified that the proposed method possesses a certain degree of scalability.

### Weaknesses
1. I think the paper's fundamental assumption that treating common steps as anchors is not fully convincing. From an entropy perspective, these anchors contain less information. A more effective attribution strategy should focus on identifying the correctness of non-consensus steps rather than treating them uniformly. 

2. In my view, the compared baselines are insufficient. LLMs used as judges can also assess the correctness of reasoning steps without requiring final-answer labels. Moreover, the utility of the proposed method is not particularly compelling, its advantages over existing techniques are unclear.

3. Lacks a formal analysis of computational complexity, and the computational cost of consensus construction and MCS operations remains high, limiting scalability.

4. Reproducibility requires further clarification, as the paper lacks detailed descriptions of consensus graph construction, model parameter settings, and training procedures.

### Questions
please address weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper Introduces a black-box method to detect where reasoning fails in large language models by assigning confidence scores to each step of a solution. 

The author's working hypothesis: correct solutions share common reasoning structures, and steps that deviate from this consensus are likely to be errors.

Proposed approach: two methods: NIBS, which uses semantic similarity, and GIBS, which models reasoning as a graph and learns structural alignment using the Information Bottleneck principle.

### Strengths
The black-box error diagnosis framework is something I can resonate with. Also the method is mostly reference free. The authors construct a proxy reference solution (reasoning trace). The application of the IB framework is also largely novel, from what I can tell. 

With that being said, I have some concerns. Please see weaknesses.

### Weaknesses
1. This "shared structure" hypothesis might not hold true across all domains. Think of a problem which is more creative, and does not follow a deductive reasoning like structure (the GIBS framework largely depends on structure of reasoning). I don’t see this hypothesis playing out there. 

2. Line 139. "We begin with the notion of answer-level..." seems like an incomplete sentence ?

3. The semantic similarity as explained in NIBS is already explored in [1] and [2]. The graph structure as explained in GIBS is also partially explored in [3]. 

4. Does the framework deal with the reasoning error that happen after the first wrong step ? This has not been explicitly mentioned. I would like to know the error identification accuracy of the first wrong reasoning step. the latter steps could skew the accuracy numbers.



[1] ROSCOE: A Suite of Metrics for Scoring Step-by-Step Reasoning

[2] RECEVAL: Evaluating Reasoning Chains via Correctness and Informativeness

[3] Premise-Augmented Reasoning Chains Improve Error Identification in Math reasoning with LLMs

### Questions
1. How do you use GSM, MATH for step level error eval ? Do you construct a proxy dataset here ? 
2. Can you share results on PRM800K or process bench ?  ( even a sampled subset should be fine )

### Soundness
3

### Presentation
3

### Contribution
2
