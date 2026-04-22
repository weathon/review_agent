# ArtifactLinker: Linking Scientific Artifacts for Automatic SOTA Discovery

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 4

## Abstract
Scientific artifacts, such as models and benchmarks, are the foundation of machine learning research. With the rapid growth of repositories like HuggingFace, researchers now have access to millions of high-quality artifacts contributed by different researchers, yet the challenge remains: how can we automatically discover the state-of-the-art (SOTA) model for a given benchmark, fully leveraging existing scientific artifacts? We address this task, abbreviated as automatic SOTA discovery, by first modeling HuggingFace as an artifact graph, where nodes represent models or benchmarks and edges capture their relationships, labeled with evaluation results. Within this graph, we formulate the automatic SOTA discovery as the process of identifying new unobserved links with high potential performance that could advance future research. To enable scalable and efficient discovery of SOTA artifact links, we propose ArtifactLinker, a two-stage framework for automatic SOTA discovery: (1) prediction, which identifies promising links with Graph Neural Networks (GNNs) or graph-augmented LLMs, and (2) verification, which validates promising predicted links through reproducible and automatic coding experiments and agents. To evaluate ArtifactLinker, we further propose ArtifactBench, collecting 1,372 models and 308 benchmarks for systematically measuring prediction and verification performance and helping to develop new SOTA discovery agents. Our key results indicate that the graph-based prediction module in ArtifactLinker is effective in prediction. Moreover, an automatic verification pipeline in ArtifactLinker can verify that the identified promising links indeed achieve high performance on existing benchmarks in a fully automatic way.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper ARTIFACTLINKER frames Hugging Face as a bipartite artifact graph and defines “automatic SOTA discovery” as finding high-potential missing links, then validating them via an agent that executes model→dataset→metric pipelines. The authors release ARTIFACTBENCH, specify the SOTA objective, a δ-thresholded candidate filter, four prediction tasks, and one verification task. Results show simple GNNs and graph-augmented LLMs outperform metadata/random baselines.

The problem is crisp and the evaluation protocol is mostly sound. That said, several results suggest shortcutting and fragility in the system stack (see Weaknesses).

The paper is readable and structured, with a useful pipeline schematic (Figure 1) and compact result tables (Tables 1–4). However, verification details are scattered.

The novelty is largely at the system level (pipeline + benchmark), not in modeling/representation learning. Link prediction uses off-the-shelf GATv2 and prompt-wrapped TextGNN; verification is a ReAct-style agent with HF-specific checks. The dataset/benchmark contribution is helpful but incremental relative to existing artifact knowledge graphs (see, e.g., LinkedPapersWithCode).

### Strengths
1. Clear problem setup: automatic SOTA discovery framed as link prediction on an artifact graph.
2. Novel system idea: combining prediction and executable verification for reproducible discovery.  
3. Empirical evaluation across four prediction tasks + verification task.    
4. Public resource (ARTIFACTBENCH) could stimulate follow-up research.

### Weaknesses
1. Limited methodological novelty: uses standard GNNs and ReAct agents; innovation is mostly system-level.  
2. Strong shortcut bias: node-degree baseline F1=87.2, nearly matching GNN (88.4); no temporal or degree-controlled split.  
3. Fragile verification: success rises on “hard” cases (1.3→14.3%) but drops on “easy” ones (56→43%).
4. No scalability or compute analysis: verification throughput and cost per edge are unreported.  
5. Only Hugging Face is used as test dataset; unclear generalization to other artifact ecosystems and knowledge graphs (e.g., LinkedPapersWithCode, ORKG, etc.).

### Questions
1. How sensitive are your discovery results to the δ threshold in Eq. (5)? Did you analyze how δ affects the precision, recall, and number of verified links?  
2. Can you provide a detailed breakdown of verification failures by stage (e.g., dataset loading, model initialization, metric evaluation, runtime errors)?  
3. Have you evaluated the models under temporal or degree-controlled splits to rule out popularity or structural shortcuts in the graph?  
4. How do you ensure that the prediction models (GNNs and LLMs) are not simply exploiting structural shortcuts like node degree or community membership? Have you tested temporal or degree-controlled splits or examined feature importance to verify genuine generalization?

### Soundness
4

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper targets the problem of efficiently discovering SOTA models for a given benchmark by leveraging existing resources. The authors construct a model-benchmark graph and propose a method to predict the performance of models on benchmarks based on this graph structure. They collect a new benchmark to evaluate their approach and provide insights for future developments in this area.

### Strengths
1. The paper targets at an interesting problem of efficiently discovering SOTA models for a benchmark by using the existing resources.
2. The paper collected a new benchmark to examine its problem and built a systematic framework to validate the problem.
3. The paper provides valuable insights for future developments in this area.

### Weaknesses
1. The discussion on related works is not thorough enough, please check the references and provided necessary discussions. Essentially, the construction of model-benchmark graph and prediction over it have been studied in previous works, which weaken the novelty of the proposed work.
2. The paper appears as an engineering prototype, to strengthen its contribution, it is better to reveal some unique patterns from the proposed setting, such as what sort of models are more likely to perform well on certain tasks, etc.

[1] ADGym: Design Choices for Deep Anomaly Detection

[2] Structuring Benchmark into Knowledge Graphs to Assist Large Language Models in Retrieving and Designing Models

[3] Beimingwu: A Learnware Dock System

### Questions
1. Why should there be four evaluation tasks? Selecting SOTA method is more about ranking the existing models in the correct order, as the absolute performance may vary with benchmarks and hard to predict.
2. What happens if a totally new benchmark is added, i.e., no prior information is available for this benchmark? How does the proposed method perform in this scenario? Since this is an important practical case of finding SOTA.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
An interesting topic, automatic SOTA discovery, is proposed, which actually tries to filter and predict the performance of existing models on existing benchmarks. 

The paper first converts HuggingFace data into a bipartite graph. It then uses a straightforward information aggregation mechanism to predict performance, testing two routines based on existing methods. An LLM finally verifies the predicted potential SOTAs and runs the codes to obtain real performance. The experiment removes SOTA edges and then recovers them.

### Strengths
The paper is well-written, carefully wrapped up to emphasize its potential contribution.

The topic has an intriguing perspective, and the whole logic flow is self-contained.

### Weaknesses
1. The technical contribution is limited. While some LLM/agent papers adopt a similar high-level approach, this work lacks topic-specific design—such as deeper analysis of benchmark characteristics, dataset distributions, or model architecture encoding. Section 4.2.1 is particularly basic, using only multi-turn aggregations on 1-hop neighbors. The graph construction from HuggingFace is straightforward, and the ReAct-based verification appears to be a simple add-on. Table 3 lists only base models from existing works, which fails to support the claim that "the graph-based prediction module in ARTIFACTLINKER is effective in prediction."

2. The entire work relies on HuggingFace data, described as containing "millions of high-quality artifacts," yet produces a graph with only 1,372 models and 308 benchmarks. This small, single-source dataset is unconvincing. Discovery based on such limited data will be narrow and biased, regardless of method sophistication. 

3. Presentation and grammar issues:

a. Figure 1 shows edges between model-model and benchmark-benchmark nodes, contradicting the "bipartite" design (Line 151, page 3). These edges are not explained in the method section.

b. "we propose a novel two-stage framework: (1) prediction and (2) verification." → "we propose a novel framework with two stages: (1) prediction and (2) verification."

c. "Prior work has largely relied on static analyses" → "Prior works have largely relied on static analyses"

### Questions
Q1. For novelty, I think I can hardly change my mind during the rebuttal phase. I will check the comments from other reviewers. The authors can focus on other issues.

Q2. For the dataset, please explain the unmatched size mentioned above. Is it possible to extend the scope from Huggingface to other data sources (something similar to paperswithcode)? 

Q3. Please fix the grammar issues.

### Soundness
2

### Presentation
3

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
The paper models Hugging Face as a bipartite artifact graph between models and benchmarks, where edges carry evaluation scores.
Authors proposes a two-stage framework: prediction (GNNs or graph-augmented LLMs rank candidate links) and verification (an agent executes code to reproduce scores). They also build ArtifactBENCH to evaluate link/attribute prediction and reproduction.

### Strengths
- The paper is easy to follow. The problem formulation is clear. 
- Authors present a benchmark (ArtifactBench) from hugging face and a set of evaluation tasks based on the benchmark.
- Authors conduct extensive experiments using heuristics, GNNs and LLMs+graph method.

### Weaknesses
- The size of the dataset is relatively small and the graph is sparse. Many edges are concentrated around a few popular datasets (e.g., ImageNet, MMLU, as shown in Figure 4). This may limit the usefulness of the prediction tasks that are around the nodes with smaller degree.
- For link prediction task, degree-based baselines already achieve high F1 (87.2 vs. 88.4 for the best method), suggesting the task may be too easy and may not present meaningful discovery.  
- (Minor) ReAct-Linker appears to be a handcrafted pipeline running ReAct three times for different stages derived to solve the benchmark. Improvements are not consistent across easy and hard settings.  The contribution of this part feels small, though the authors also do not claim it as a major one.
- (Minor) The formulation of experiments is to exclude edges in the graph and predict them as targets. This setup focuses on re-discovery, which differs from discovering new, unobserved edges in a real-world setting (e.g., applying a model to a dataset that actually achieves new SOTA). Moreover, real-world discovery can be biased, especially with modern LLMs that already achieve SOTA across many datasets. This bias may limit the benchmark’s practical relevance.

### Questions
- Do you plan to scale the dataset and mitigate the concentration of edges on a few popular datasets?
- Do you plan to evaluate the framework on real discovery (unseen edges)?

### Soundness
2

### Presentation
3

### Contribution
2
