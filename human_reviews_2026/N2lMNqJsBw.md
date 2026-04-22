# RL Squeezes, SFT Expands: A Comparative Study of Reasoning LLMs

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Large language models (LLMs) are typically trained by reinforcement learning (RL) with verifiable rewards (RLVR) and supervised fine-tuning (SFT) on reasoning traces to improve their reasoning abilities. However, how these methods shape reasoning capabilities remains largely elusive. Going beyond an accuracy-based investigation of how these two components sculpt the reasoning process, this paper introduces a novel analysis framework that quantifies reasoning paths and captures their qualitative changes under each training process (with models of 1.5B, 7B, and 14B parameters on mathematical and code domains). Specifically, we investigate the reasoning process at two levels of granularity: the trajectory-level, which examines complete reasoning outputs, and the step-level, which analyzes reasoning graphs whose nodes correspond to individual reasoning steps. Notably, clustering of unique reasoning trajectories shows complementary effects: RL compresses incorrect trajectories, whereas SFT expands correct ones. Step-level analysis reveals that RL steepens (about 2.5 times), while SFT flattens (reduced to about one-third), the decay rates of node visitation frequency, degree, and betweenness centrality distributions in the reasoning graph. This indicates that RL concentrates reasoning functionality into a small subset of steps, while SFT homogenizes it across many steps. 
Furthermore, by evaluating the reasoning graph topologies from multiple perspectives, we delineate the shared and distinct characteristics of RL and SFT. Our work presents a novel reasoning path perspective that explains why the current best practice of two-stage training, with SFT followed by RL, is successful, and offers practical implications for data construction and more efficient learning approaches.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Summary:
This paper compares how reinforcement learning (RL) and supervised fine-tuning (SFT) shape reasoning in large language models. RL “squeezes” reasoning by concentrating paths, while SFT “expands” it by diversifying correct trajectories. Graph-based analyses explain why the SFT to RL pipeline works effectively.

### Strengths
The paper shows a novel analytic framework for The trajectory- and step-level “reasoning graph” approach provides a creative and rigorous lens for understanding LLM reasoning beyond metrics like Pass@k.

The work sheds light on why SFT+RL remains the dominant fine-tuning recipe for reasoning LLMs and offers potential implications for efficient data curation and model design.

### Weaknesses
The evaluated datasets (AIME24, AIME25, AMC23) are all from the same competition series and share similar problem distributions. This narrow scope makes it difficult to generalize the conclusions. For example, the claim that “applying SFT to the Base model increases the number of correct trajectories, showing that SFT teaches new solution strategies absent in the Base model” is not consistently significant, particularly for the AMC23 dataset and the 14B model.
 
The experiments use pretrained and post-trained models from the same family (e.g., Qwen and DeepSeek-based models). This raises concerns that the authors may not have controlled for shared training samples or overlapping data distributions, which could confound the observed “squeeze–expand” effects.
 
The paper omits several key implementation details required for replication. See question for details.

### Questions
The main findings rely heavily on clustering reasoning trajectories and sentences, yet the paper reports no clustering-quality metrics such as ARI, NMI, or Silhouette scores are reported. Without such validation, it is difficult to assess whether the clustering results genuinely reflect reasoning structure or are artifacts of algorithmic choices.


The paper adopts BGE-large-en-v1.5 (Xiao et al., 2024) for sentence embeddings instead of using the LLM’s own internal representations. What is the rationale for this choice? Given that the goal is to analyze reasoning behaviors within each model, it seems more natural to extract embeddings directly from the model under analysis, since external embeddings may not align with the model’s latent reasoning space.

The paper states that edge weights are defined by the Euclidean distance between cluster centroids, which suggests that every pair of nodes could be connected, effectively creating a dense graph. Could the authors clarify whether any thresholding, sparsification, or top-k selection was applied when constructing edges? If not, how should we interpret structural metrics such as degree, modularity, or centrality that assume sparse connectivity?

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
4

### Summary
This work provides a novel framework to analyze the properties of two post-training techniques of LLM - SFT and RL.
It investigates the reasoning processes of base, SFTed, RLed and SFTed+RLed models at two levels of granularity: the trajectory-level which examines complete reasoning outputs, and the step-level, which analyzes reasoning graphs whose nodes correspond to individual reasoning steps.

Based on the analysis existing models, covering Qwen2.5-Math-1.5B, Qwen2.5-Math-7B and Qwen2.5-14B, the authors get the conclusion that RL compresses incorrect trajectories, whereas SFT expands correct ones (trajectory-level); RL concentrates reasoning functionality into a small subset of steps, while SFT homogenizes it across many steps (step-level).

### Strengths
1. This work proposes a novel and solid analysis framework based on reasoning paths to better understand the effects of SFT and RL in LLM post-training, particularly from the perspective of reasoning diversity (e.g., pass@k vs. pass@1).

2. The framework design is rigorous and well-grounded in theory.

3. The paper is clearly written and well-presented, making the methodology and findings easy to follow.

4. The experimental analysis is comprehensive and covers multiple model scales and post-training settings.

### Weaknesses
## Representativeness of the SFT Models

I am concerned about whether the models chosen in this paper accurately represent conventional SFT. 

The authors use the DeepSeek-R1-Distill-Qwen series as SFT models; however, these models are trained via knowledge distillation (KD), which differs from standard SFT practices. In KD, for each prompt, multiple responses from teacher models are used for supervision (as described in Sec. 2.3.3 of Guo et al., 2025). This setup might make properties diverge from a pure SFT setting and potentially enhance the diversities by incorporating richer COT paths. (In fact, it is somehow close to offline RL).

Therefore, the behavior of KD-based models may not truly reflect that of conventional SFT models. If this is the case, some of the conclusions in the paper may not fully hold.




--------------_

- Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., ... & He, Y. (2025). Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

### Questions
1. Do the properties and construction methods of SFT datasets influence the resulting SFT models? If so, could this affect the validity or generality of the paper’s conclusions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper analyzes how SFT and RL on Verifiable Rewards shape the LLM reasoning process. They perform a two-level analysis: (1) trajectory-level clustering of sampled solutions, and (2) step-level reasoning graphs built from sentence embeddings of the models' CoT. The core findings are the following: (1) RL compresses incorrect trajectories while SFT expands correct trajectories, helping explain why SFT --> RL is the prevailing training recipe; (2) at step level, RL also consolidates reasoning graph functionality into fewer steps. Experiments are shown for 1.5B/7B/14B models on AIME24/25 and AMC23 with controlled sampling.

### Strengths
1. It presents a timely analysis for a RL/SFT fine-tuning on LLMs, which helps explaining and understanding  the use of these techniques for fine-tuning. 
2. The authors support their insights with a solid set of experiments which include multiple models. 
3. The authors perform a detailed analysis at the whole reasoning trajectory, and at the step level combined with fundament computer science analysis based on graphs.

### Weaknesses
1. Reasoning graphs depend on (a) sentence segmentation, (b) embedding model, (c) k=2000, and (d) Euclidean distances. Some ablation on of these points can help strengthen the derived insight. Some of these ablations should be easy to run. 
2. Local graphlet proportions can look similar across models while accuracies differ; more analysis is needed to connect which topological features track Pass@1/Pass@k
3.  In the graph analysis, step-level graphs aggregate all sampled responses for a problem without separating by correctness, so topology may conflate successful and failing behaviors.  Similarly, edge distance is Euclidean between K-means centroids from a single embedding model; no evidence is given that this correlates with semantic step difference, and cosine/other spaces aren’t tested.

### Questions
1. Any preliminary results on code or scientific reasoning to test whether RL still weakens community structure and SFT boosts global efficiency?
2. I'd suggest the Figure should provide some insight, and the caption should explain it. For instance, what's the value of the 4 versions of the same plot in Figure 4? Or similarly, what is the value of each metric in Figure 7?
3. The findings in the paper are interesting, but the authors fail to point out how they are useful for improving them, do you have some comments on how we can leverage these insights?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors examine how two training strategies, reinforcement learning with verifiable rewards (RLVR) and supervised fine-tuning on reasoning traces (SFT), reshape LLM reasoning. Their central result: RLVR compresses the reasoning space, reducing path diversity and concentrating effort on a small number of high-impact steps, whereas SFT expands it, introducing many correct solution paths and distributing functionality across more steps. This contrast clarifies why the standard SFT + RL pipeline succeeds: SFT first generates a richer set of good paths; RL then trims the weak ones to raise Pass@1.

### Strengths
1. The trajectory and step level analysis for both SFT and RL are interesting to me.  
Trajectory clustering shows RL compresses incorrect paths while SFT expands correct ones, motivating the common two-stage recipe (SFT then RL).

2. The paper builds “reasoning graphs” and profiles them with edge density, modularity, assortativity, centralization, global efficiency, etc., revealing RL concentrates functionality into hubs while SFT distributes it.

3. The paper is easy to follow.

### Weaknesses
1. Even though the analysis is interesting, the paper’s scope and novelty are not sufficient for ICLR.

2. The domain and model scope are narrow. The evidence centers on verifiable math datasets and a limited set of public checkpoints, leaving generalization to coding/science tasks and broader architectures untested.

### Questions
1. Can we test on other domain datasets to see if the observed effects replicate?

### Soundness
2

### Presentation
3

### Contribution
2
