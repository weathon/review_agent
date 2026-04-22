# How can we assess human-agent interactions? Case studies in software agent design

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 8, 4, 2

## Abstract
LLM-powered agents are both a promising new technology and a source of complexity, where choices about models, tools, and prompting can affect their usefulness. While numerous benchmarks measure agent accuracy across domains, they mostly assume full automation, failing to represent the collaborative nature of real-world use cases. In this paper, we make two major steps towards the rigorous assessment of human-agent interactions. First, we propose a framework for more efficient human-centric evaluation of agent designs, which comprises collecting user feedback, training an ML model to predict user satisfaction, and computing results by combining human satisfaction ratings with model-generated pseudo-labels. Second, we deploy the framework on a large-scale web platform built around the open-source software agent OpenHands, collecting in-the-wild usage data across over 15k users. We conduct case studies around how three agent design decisions---choice of LLM backbone, planning strategy, and memory mechanisms---impact developer satisfaction rates, yielding practical insights for software agent design. We also show how our framework can lead to more robust conclusions about agent design, reducing confidence intervals by 40\% compared to a standard A/B test. Finally, we find substantial discrepancies in-the-wild results with benchmark performance (e.g., the anti-correlation between results comparing claude-sonnet-4 and GPT-5, underscoring the limitations of benchmark-driven evaluation. Our findings provide guidance for evaluations of LLM agents with humans and identify opportunities for better agent designs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper makes an effort to build a comprehensive assessment of human-agent interactions. The proposed framework consists of collecting real human feedback, training a machine learning model to predict satisfaction and generate pseudo-labels, and using prediction-powered inference to calculate valid confidence intervals. The framework is deployed in the real world, and a large amount of data was collected. Through three case studies, the paper demonstrates insights into bridging the gap between agent benchmarks and user experience.

### Strengths
1: The research topic and some of the results are interesting. This work directly solve the problem in effectively evaluate the utility of agents in practice. Also, the finding claude-4-sonnet outperformed gpt-5 in user satisfication despite the latter's strong performance on several coding benchmars is quite counter-intuitive 

2: The experiments are performed on a live platform and cover a wide range of users and context with different language, coding language and tasks, providing the findings with high external validity.

### Weaknesses
1: The feedback data collection is biased. As the authors mentioned, ratings are often only provided by a small percentage of users, and the ratings for the remaining sessions are predicted by a trained ML model. However, it is possible that the users who are willing to rate often had a very extreme experience, regardless of whether it was good or bad. This can result in a biased sample that does not represent the average user. It would be better if the authors acknowledged this limitation and discussed how this potential bias might influence their results.

2: The computational cost analysis is missing. The authors mentioned it is cost-effective in memory management part, but the paper lacks the API cost for gpt-5, claude-4-sonnet and alcude-3.7-sonnet. For instance, is the 5.86% improvement in user satisfaction from claude-4 worth its potentially much higher cost compared to claude-3.7?

### Questions
1: Regarding the prediction of human satisfaction, how did the authors validate the accuracy of the predicted ratings?

2: Still in the Table 1, the correlation between the best Random Froest and real ratings is 0.29, which is better than LLM-as-a-judge but still a relative low number. I'm curious how the accuracy of the prediction model $f$ impacts the effectiveness of the effectiveness of the PPI framework.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work introduces a human-centric evaluation framework designed to assess agent satisfaction in human–agent collaboration scenarios. The proposed framework consists of three stages: data collection, satisfaction prediction, and agent change validation. Using the open-source software engineering agent OpenHands as a case study, the authors show that the agent’s performance does not fully align with user preferences, highlighting the importance of human-in-the-loop evaluation for iterative agent development.

### Strengths
1. This paper is well-structured and methodologically sound. The authors develop a satisfaction-prediction model and conduct extensive user studies involving approximately 15k participants across 36k sessions. The reported statistics demonstrate significant variability, strengthening the reliability and generality of the conclusions.

2. The two proposed metrics—native effect-size estimation and augmented estimation—exhibit a high degree of correlation, underscoring the effectiveness and robustness of the trained satisfaction-label prediction model.

3. The authors conduct an in-depth analysis of software engineering tasks, covering seven common scenarios to compare human ratings with benchmark-based evaluations. Their findings suggest that existing SWE benchmarks may not accurately represent real-world human preference, highlighting the necessity of incorporating human preference signals in agent evaluation.

### Weaknesses
1. The case study primarily focuses on OpenHands, which may limit the generalizability of the findings to other software engineering agents such as GitHub Copilot or Cursor. Evaluating additional agents would help validate whether the conclusions hold across different system architectures and interaction paradigms.

2. Table 2 includes only two model comparisons, which may weaken the claim that “static benchmarks don’t tell the whole story.” Incorporating more model pairs—such as GPT-4.1 vs. GPT-5 or Claude 4 vs. Gemini-2.5-Pro—would provide a more comprehensive perspective and further substantiate the argument.

### Questions
1. How do you diagnose noise in user ratings? Additionally, what strategies do you apply to mitigate potential bias in the collected feedback?

2. In Figure 3 (Case 3), why does reducing the maximum number of steps result in improved user satisfaction? Could you elaborate on the underlying mechanism or user behavior that explains this outcome?

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
This paper proposes a three-step framework to evaluate human–agent interactions targeting software agents: (1) collect lightweight in-product user feedback at the end of each “work segment,” (2) train an ML model on hand-crafted trajectory features (user sentiment, message counts, task type; git actions; common failure tags) to predict user satisfaction for unlabeled sessions, and (3) combine human labels with model predictions.

It deploys the framework on web platform to collect data across over 15k users, and study how  three agent design decisions affect user satisfaction

### Strengths
- The user study is deployed large scale (36k sessions, 5k users,)
- The paper includes well-scoped case studies (LLM, planning, memory) with effect sizes, 95% CIs, and feature-shift analyses that explain why UX differs of software agents

### Weaknesses
The proposed method lacks novelty and is too specific to software agent: 1. The feedback data collection is common (asks the user to
rating on 5 star scale) 2. The feature selection is manual and targeting on software tasks. 
If the proposed pipeline is dedicated to Software agent, then it doesn't suffice as a general method. If the proposed method is claimed to be general, then significant effort is needed, and the results are not guaranteed. For example, while ML model is better in this case, we make such claim with another type of task.

### Questions
1. Please clarify whether the proposed method is specific to SWE tasks, or is supposed to be a general framework?
2. How do you handle noise in the user labels, or do you observe any? For example, users may give random ratings, or false ratings on purpose.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a framework for evaluating human–agent interactions, focusing on software engineering agents. The framework collects user satisfaction ratings during real-world interactions and uses a semi-supervised learning approach to train a predictive model estimating user satisfaction from interaction logs. The authors then apply the method to several case studies analyzing how different agent design choices (LLM backbone, planning, memory) affect user satisfaction, and compare these outcomes to traditional benchmark results.

### Strengths
1.Motivational relevance: The paper addresses an important gap — current agent benchmarks largely ignore human–agent collaboration. The argument that evaluation should reflect real-world human use is persuasive and timely.

2.Empirical grounding: The deployment at scale (15k users, 36k sessions) provides an unusually large dataset for human–agent evaluation, which is commendable given the difficulty of collecting real-world interaction data.

3.Clarity and organization: The paper is well structured and written in a clear, professional manner, making it accessible to a broad ICLR audience.

### Weaknesses
1.Limited methodological novelty. As the authors themselves note, prior work has already explored predicting user satisfaction in multi-turn dialogue using explicit user ratings. Extending this paradigm to software agent interactions is incremental and does not yield new theoretical insights.

2.Lack of reproducibility and data accessibility. The empirical findings rely entirely on proprietary user data that will not be released due to privacy concerns. Without access to the dataset or a simulator, one cannot verify, reproduce, or extend the study, substantially reducing its community impact.

3.Simplistic feedback signal and sparse coverage. The reliance on a 5-star scale at ~5% response rate introduces concerns about signal quality and representativeness. While the authors aggregate ratings across work segments to increase granularity, the fundamental sparsity may limit the model’s ability to capture satisfaction drivers. Incorporating implicit behavioral signals (dwell time, edit patterns) could strengthen the supervision signal.

4.Underdeveloped baseline comparisons. The study claims that the trained ML models “significantly outperform state-of-the-art LLMs across all metrics,” while providing only raw trajectory data to the LLM-as-a-Judge baseline, without any structured context management, prompt decomposition, or summarization. This setup limits the LLM’s ability to reason over long histories and identify latent satisfaction cues.

5.Questionable validity of semi-supervised inference. The PPI extension relies on the assumption that the prediction model f(X) is unbiased or has bounded bias. Given the low label response rate (~5%) and potential selection bias, the extent to which this assumption holds warrants further investigation. Sensitivity analyses varying the labeling rate or testing on held-out labeled data could strengthen confidence in the pseudo-label quality.

6.Limited scalability and generality. The framework’s dependence on large-scale real user deployment restricts its applicability. It is unclear how the approach would generalize to domains lacking constant human traffic or how to replicate such evaluations without a similar data infrastructure.

### Questions
1.Would releasing anonymized summaries or a simulator allow the community to reproduce at least part of the results?

2.Could richer feedback signals (e.g., implicit behavioral metrics like dwell time or edit frequency) enhance satisfaction prediction beyond the 5-star scale?

### Soundness
2

### Presentation
3

### Contribution
2
