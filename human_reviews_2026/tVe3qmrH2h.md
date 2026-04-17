# Rethinking LLM Human Simulation: When a Graph is What You Need

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 6

## Abstract
Large language models (LLMs) are increasingly used to simulate humans, with applications ranging from survey prediction to decision-making. However, are LLMs strictly necessary, or can smaller, domain-grounded models suffice? We identify a large class of simulation problems in which individuals make choices among discrete options, where a graph neural network (GNN) can match or surpass strong LLM baselines despite being three orders of magnitude smaller. We introduce Graph-basEd Models for Human Simulation (GEMS), which casts discrete choice simulation tasks as a link prediction problem on graphs, leveraging relational knowledge while incorporating language representations only when needed. Evaluations across three key settings on two simulation datasets show that GEMS achieves comparable or better accuracy than LLMs, with far greater efficiency, interpretability, and transparency, highlighting the promise of graph-based modeling as a lightweight alternative to LLMs for human simulation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper reframes a classic recommender-style GNN as an alternative to LLMs for discrete choice prediction. While competently executed, it lacks conceptual novelty, overstates its claims about “human simulation,” and provides limited scientific insight.

### Strengths
1. The authors clearly reframe discrete-choice human simulation (e.g., predicting survey answers or behavioral decisions) as a link prediction task on a graph, where nodes represent individuals and choices.
2. GEMS achieves comparable or even superior performance to strong LLM baselines (e.g., zero-shot, few-shot, chain-of-thought, fine-tuning) on three human-simulation subtasks.

### Weaknesses
1. The presented results are not surprising. It is already well known that large language models perform well on some tasks but not on others. Beating them on structured, discrete-choice problems is expected rather than a breakthrough. The only contribution here appears to be modeling a discrete prediction problem as a network-based one, as graph neural networks for link prediction have already been widely studied in prior work.

2. The use of the term “human simulation” exaggerates the scope of the paper and may mislead readers into thinking that it deals with cognitive modeling, psychology, or game theory, which it does not.

3. The paper’s central empirical claim that graph neural networks can match or surpass large language models is not particularly impressive once it is clear that the evaluated tasks are simple classification problems with small and discrete output spaces.

4. Overall, the work reads more like an engineering benchmark than a piece of scientific research that offers new insights or advances our understanding of the problem.

### Questions
1. How can your model be generalized to other tasks related to human simulation?

2. What are the contributions, except for modeling the discrete choice problem as link prediction?

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
4

### Summary
The paper introduces GEMS, a graph-based framework that reframes human simulation tasks where individuals choose among discrete options, as a link prediction problem on a heterogeneous graph of individuals, subgroups, and choices. Using relational structure and a GNN (plus an LLM-to-GNN projection when new questions appear), the authors conduct a comprehensive comparison of GEMS against multiple LLM-based baselines, achieving comparable or superior performance in most cases.

### Strengths
- The paper provides a comprehensive comparison between the proposed GEMS framework and multiple LLM-based baselines, covering diverse settings and evaluation dimensions.

- It addresses a highly relevant and practical problem and offering an efficient alternative to LLM-heavy approaches.

- The study provides valuable and insightful findings, demonstrating that relational structure and graph-based reasoning can rival or surpass LLMs while being more efficient and interpretable.

### Weaknesses
- The related work section is incomplete; it omits relevant studies exploring GNN-based approaches for multi-choice question answering, such as [1]. Including these works would clarify the connection to prior research and more accurately position the paper’s novelty.

- The paper lacks simpler and stronger baselines. Comparing GEMS against a more basic neural network or MLP classifier could better isolate the contribution of the graph structure, while incorporating newer or larger LLMs would help establish upper performance bounds.

- The paper lacks sufficient ablation studies to clarify the specific advantages of the proposed approach. For instance, conducting a user study to assess the claimed interpretability benefits would provide stronger empirical support for those claims.

[1] QA-GNN: Reasoning with Language Models and Knowledge Graphs for Question Answering.

### Questions
- The datasets examined in the paper mostly contain a small number of choices (4–6). In real-world scenarios, the number of options can be much larger. How do the authors expect GEMS to perform compared to LLMs in such settings?

- It would be interesting to explore the impact of demonstration selection in the few-shot setting, not only based on question similarity but also by incorporating attributes or other contextual features.

- The paper mentions that LoRA was applied only to the attention query and value matrices. Could the authors clarify the motivation for restricting LoRA adaptation to only these parameters?

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
5

### Summary
The paper presents GEMS, a framework for modeling human choices using GNNs, as an alternative to the use of LLMs. The case is made that GNNs can be at least as good as LLMs for such modeling with better efficiency and interpretability. GEMS uses relational knowledge between humans and tasks and uses link prediction for predicting the human choices on missing responses, new questions, and new individuals. There is a mechanism to transfer representation from LLMs to GNNs for the case of predicting responses on new questions.

### Strengths
The paper develops a nice model for predicting choices using link prediction.

The transfer of information from LLMs to GNNs is done well.

The performance and interpretability of the approach is good.

### Weaknesses
It is not unexpected that for many domains LLMs' performance can be surpassed through the use of GNNs or some other machine learning method. Thus, I find that the novelty low.

The setup and solutions are sound and along expected lines.

### Questions
Why is the approach novel?

### Soundness
3

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
4

### Summary
The authors propose using GNN approaches to model discrete choice human simulation tasks, by formulating the problem as a link prediction problem on a graph of individuals, choices and subgroups. They show that their methods achieve comparable performance to LLM-based approaches.

### Strengths
The paper is very well-motivated; The authors demonstrate a very good knowledge of the social simulation literature by covering all the main research progresses in the area. The three evaluation settings make sense. The writing is very clear and easy to follow. It’s also very nice that the authors included test-retest baselines.

### Weaknesses
The authors do not adequately address the fundamental reasons why LLM-based social simulator are so popular. 

1)	The end-users for these models are often researchers with limited computational resources or expertise. LLMs, accessible via APIs and natural language prompting, offer a near-zero barrier to entry. In contrast, the GEMS framework requires data preprocessing, graph construction, model training, and fine-tuning, all posing a significant technical hurdle.

2)	A key advantage of LLMs is their ability to generate natural language outputs. Even though the chain-of-thought is not a genuine cognitive process, these textual explanations are invaluable for social scientists seeking qualitative insights. GEMS is a purely predictive model and thus simply does not have this capability.

3)	While many papers focus on single-step discrete choice settings, this is certainly not the entire LLM-for-social simulation field, as there are many applications that requires natural language output, or multi-turn interactions. 

4)	The paper's "new questions" setting is a form of in-domain generalization. The true challenge, which LLMs are better poised to handle, is cross-domain or cross-dataset generalization (e.g., applying a model trained on political surveys to a new dataset on consumer preferences). The GNN's rigid structure and learned embeddings are unlikely to transfer, a critical limitation that is not discussed with sufficient honesty. The GEMS approach is not a general-purpose "human simulator" but a specialized prediction model.

5) The paper's empirical results are compelling, but their strength is contingent on the choice of baselines. The comparison is made against relatively small (7-8B parameters) and now somewhat dated LLMs (LLaMA-2, Mistral-7B-v0.1). To make a truly convincing case, it is essential to compare against a stronger "upper bound," such as a state-of-the-art proprietary model (e.g., via the GPT-5 or Claude 4.5 APIs) or a strong open model. These models exhibit far superior reasoning and in-context learning capabilities, and practitioners would most likely use them as their first choice. Without this comparison, it is unclear if GEMS's performance advantage holds against the models that are actually being deployed for the social simulation tasks.

6) Several claims need revision

a.	Prompt formulations for LLM (Section 5.1) capture at most 1-hop structure and do not naturally express higher-order dependencies… I would like to see some references on this; In general, as universal function approximators, sufficiently large LLMs can theoretically learn complex, higher-order dependencies from data, even if they lack a specific graph-based inductive bias. This claim should be rephrased as a hypothesis about the differing inductive biases rather than a statement of fact about LLM capabilities.

b.	GEMS makes predictions in a computationally simple and interpretable way -> While the dot-product mechanism and embedding space are more inspectable than an LLM's internal states, GEMS is still a deep neural network, which is fundamentally uninterpretable, compared to, say, a decision tree. 


7) Clarification questions:

(a) How does GEMS handle scenarios where the number of available options for a question changes between the training and test sets? This is a common practical issue that LLMs handle seamlessly but would likely require architectural changes or retraining for the GNN.

(b) While the appendices contain details, the main paper would benefit from a more explicit description of the train/validation/test splits, particularly for the more complex imputation setting (Setting 1), to ensure the comparison between methods is clearly understood as fair.

### Questions
see above

### Soundness
4

### Presentation
4

### Contribution
2
