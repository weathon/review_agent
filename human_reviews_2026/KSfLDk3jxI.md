# ALIGNING LLMS WITH GRAPH NEURAL SOLVERS FOR COMBINATORIAL OPTIMIZATION

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Recent research has demonstrated the effectiveness of large language models (LLMs) in solving combinatorial optimization problems (COPs) by representing tasks and instances in natural language. However, purely language-based approaches struggle to accurately capture complex relational structures inherent in many COPs, rendering them less effective at addressing medium-sized or larger instances (e.g., problem sizes greater than 30). To address these limitations, we propose AlignOPT, a novel approach that aligns LLMs with graph neural solvers for learning a more generalizable neural COP heuristic. Specifically, AlignOPT leverages the semantic understanding capabilities of LLMs to encode textual descriptions of COPs and their instances while concurrently exploiting graph neural solvers to explicitly model the underlying graph structures of COP instances. Our approach facilitates a robust integration and alignment between linguistic semantics and structural representations, enabling more accurate and scalable COP solutions. Experimental results demonstrate that AlignOPT achieves state-of-the-art results across diverse COPs, underscoring its effectiveness in aligning semantic and structural representations. Additionally, AlignOPT exhibits strong generalization capabilities, successfully extending to previously unseen COP instances.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes AlignOPT, a novel framework that integrates large language models (LLMs) with graph neural solvers for combinatorial optimization problems (COPs). The approach tries to address the limitations of pure LLM-based methods, which struggle with complex relational structures in medium-to-large COP instances. AlignOPT uses a multi-task pre-training strategy with two novel objectives: (1) Text-Graph Contrastive (TGC) loss to align semantic node embeddings from LLMs with structural embeddings from graph solvers, and (2) Text-Graph Matching (TGM) loss for fine-grained multimodal representation. After pre-training, the model can be fine-tuned without LLMs, improving computational efficiency. The framework demonstrates performance gains across diverse COPs and shows good generalization to unseen problem types.

### Strengths
1. The paper is generally well-written with clear figures illustrating the framework. The methodology section provides sufficient detail about the architecture and training procedures.

2. The paper presents a creative integration of LLMs and graph neural solvers through contrastive and matching losses. This is a significant direction for LLM and optimization researchers to investigate.

### Weaknesses
1. The paper claims state-of-the-art results but primarily compares against LNCS and GOAL as neural baselines. Given the rapid advancement in neural combinatorial optimization, more comprehensive comparisons with recent specialized solvers would strengthen the SOTA claims. For instance, several methods can approach nearly optimality on TSP-100 (fastT2T [1] and COexpander [2]) and can handle much larger problem sizes.

2. While the paper shows results up to n=100, it's unclear how the approach scales to truly large instances (thousands of nodes).

3. Regarding the validity of text-graph alignment, the assumption that textual descriptions and graph structures can be perfectly aligned may not hold in practice, especially for complex COPs where the textual representation might not capture all structural nuances. The paper would benefit from showing learning curves for the contrastive and matching losses to demonstrate effective alignment.


[1] Fast t2t: Optimization consistency speeds up diffusion-based training-to-testing solving for combinatorial optimization. NeurIPS 2024.

[2] COExpander: Adaptive Solution Expansion for Combinatorial Optimization. ICML 2025.

### Questions
1. Regarding the text-graph contrastive learning, how do you ensure that textual descriptions and graph structures maintain a one-to-one correspondence? Could you provide learning curves showing the convergence of both TGC and TGM losses during pre-training to demonstrate the effectiveness of the alignment process?

2. Your ablation study examines removing individual components but doesn't test the scenario where LLM features are completely excluded from the framework. What would be the performance of your graph neural solver trained without any LLM-derived representations? This would help quantify the true contribution of LLM integration.

3. What specific aspects of the pre-training enable such strong generalization to unseen COPs? Is it primarily the semantic understanding from LLMs, the structural alignment, or the combination? Could you provide more analysis of what knowledge transfers across different problem types?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
## Paper Summary
Paper is interested in solving combinatorial optimization problems (COPs), like, travel salesman problem, knapsack problem, etc, using machine learning. These problems can be viewed as optimization where variables can take integer values. In general, these problems can be NP-Hard and heuristics have been nowadays getting replaced by neural network models (e.g., to directly output a solution or to aid the search). The paper proposes to combine an LLM and Graph Encoder to solve this problem. Specifically, they LLM and GraphEncoders are "aligned" in their representation. They are trained to, once observed the same instance (respectively, in textual and graph form) output an embedding that is similar (high cosine similarity) or complimentary (i.e., an MLP absorbing both should have good classification accuracy of whether the corresponding inputs are instances of the same problem or not). After alignment, paper proposes to fine-tune the learned representations via RL. Results show that this alignment followed by RL finetuning produce SoTA models for COPs when compared to a range of Neural- or Heuristic-based baselines.

## Decision summary
I will vote to reject the paper. The method is not clearly explained and therefore is not reproducible. I am happy to revisit this decision if the authors fill the gap.

### Strengths
* Strong Motivation
* SOTA results
* Aligning Graph Encoders and Language Models is potentially applicable to many problems, even outside the COPs.
* I appreciate the Preliminaries. Smooth and informative read. While I didn't know much about this subdomain (COPs via ML), I feel that I learned a lot. Thank you!

### Weaknesses
The central weakness is the lack of information, preventing this paper from standing on its own.

* **Inference is not mentioned in this paper**. After training the model (pre-train by alignment followed by task-specific or many-tasks RL fine-tuning), it is not clear how is inference conducted. Part of me thinks it is only the graph-encoder that is being used, not the LLM. However, this goes against line 93 "our AlignOPT delves into general text-attributed COPs described in natural language". Further, the **unified decoder is not specified**. Please specify if inference needs the language, the graph, or both.

* It is not clear whether $\theta$ in Eq.3 corresponds to the LLM or the Graph Encoder parameters. What does the RL fine-tune? It is also confusing that $\theta$ is reused as a function-name (corresponding to cosine, Eq.7)

* The loss function $L_{TGM} $ is improper. It can be arbitrarily minimized by increasing the bias term of the MLP (at the last layer). A proper one is the log-likelihood of logistic function, in their notation, it would be

$$
L_{TGM}  = -\sum_{i, j} 1_{[j=i]} log(\sigma(..))  - \sum_{i, j}  1_{[j \neq i]} log(1 - \sigma(..))
$$

## Other weaknesses

* Why not also cite other ways to integrate graph with language? E.g., GraphToken
* The citations are against the guidelines listed in call-for-papers. Specifically, use the citation style "names (year)" when you use the author names in a sentence e.g. "A & B (2022) proposed ...". However, when you add a citation that certifies the sentence use "(name, year)" e.g., "it has been proposed ... (A & B; 2022)".

### Questions
* How is inference conducted? How is it guaranteed that the constraints are satisfied? Is this used to produce a final solution or to propose candidates that individually get scored to select the winner solution?

* What do you set $b(\mathcal{G})$ of Equation 3 to?

* Your experiments list-down the fine-tuned variants of AlignOPT. What about the performance without fine-tuning?

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
This paper proposes AlignOPT, a framework that integrates large language models with graph neural solvers to tackle combinatorial optimization problems. It aligns textual semantics captured by LLMs with graph structural information through Text-Graph Contrastive loss and Text-Graph Matching loss. A decoder fine-tuning stage further enables the model to generate effective and unified solutions across diverse COPs. Extensive experiments on multiple benchmarks demonstrate that AlignOPT achieves state-of-the-art or near-SOTA performance and generalizes effectively to previously unseen optimization tasks.

### Strengths
1. The proposed TGC and TGM losses offer a principled approach to fusing semantic (LLM) and structural (graph) representations, addressing a key gap in LLM-based COP solvers.
2. Extensive experiments across five classical COPs and two unseen ones demonstrate AlignOPT’s competitive or superior performance compared to both LLM-based and NCO baselines (e.g., LNCS, GOAL), with robust generalization.

### Weaknesses
1. While TGC/TGM losses are new, the overall architecture (text-attributed instances, multimodal alignment) heavily builds upon LNCS. The paper should better clarify the conceptual distinction and originality.

2. The alignment process lacks deeper theoretical grounding. Why this specific combination improves generalization, or how it avoids overfitting semantic bias from LLMs.

3. Since the LLM is removed during fine-tuning, it is unclear to what extent LLM pre-training influences downstream performance. The paper should quantify or visualize this transfer more explicitly.

4. The benchmarks are synthetic and the scale is not so visible. Some real-world tasks would better demonstrate scalability and robustness.

5. Some sections are dense and formula-heavy without sufficient intuition or visualization. The explanation of the mixed attention mechanism and task embeddings could be made more digestible.

### Questions
1. Could the authors clarify how much of the performance improvement actually comes from LLM pre-training? Have the authors quantified this transfer (e.g., through representation similarity or other means) to verify that semantic knowledge from LLMs persists in the graph solver?

2. The ablation results (Table 3) show that removing either TGC or TGM reduces performance, but it’s still unclear how these two losses differ in what they capture. Could the authors provide more intuition on why both are needed? For example, does TGC handle local node-level alignment while TGM enforces global instance-level consistency?

3. The current experiments mainly involve synthetic COPs with node sizes up to 100. How would AlignOPT perform in larger, real-world scenarios (e.g., thousands of nodes in logistics routing)? Are there computational or memory bottlenecks in applying the method to such scales, given the mixed-attention graph encoder design?

### Soundness
3

### Presentation
2

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
This paper studies neural combinatorial optimization and proposes AlignOpt. 

The idea is to align LLM generated text embeddings and graph-based neural decoder for optimization problems. This is fine-tuned, per problem or combined, using RL to constructively build solutions for combinatorial optimization problems. 

The pretraining stage uses to objectives, Text-Graph Contrastive Matching loss to map LLM features and Graph features into a shared latent space. 

Experiments span TSP, CVRP, KP, MVCP, SMTWTP etc as COP variants, including out-of-domain and ablation tests.

### Strengths
The idea of utilizing embeddings from text-based problem and instance descriptions is quite interesting (albeit not the first paper to do that) 
Two objective losses seem sensible and practical
I like that experiments cover traditional solvers, heuristics, meta-heuristics and other LLM-based comparisons
Ablations probing descriptions and each loss components are included
Quite practical to remove LLM at fine-tuning and inference stage.

### Weaknesses
Novelty vs. prior work: 
The idea of idea of aligning language and graph modalities with contrastive losses is well established in multimodal learning. This paper specializes on neural combinatorial optimization, and even in this setting the referenced LNCS articles uses this idea. (I should however note that, it's only an arxiv paper). 

The positioning of the paper with respect to LNCS and GOAL could be clearer. The idea of text alignment is already in LNCS and the unified encoder used from GOAL. So, it's not immediately obvious what exact gap AlignOPT fills (Q1)

Method clarity and technical details: 
I am quite surprised that the paper does not disclose important details. For example, how positive/negative sampling is performed across instances and nodes. How do extract LLM embeddings? are they aggregated, pooled, or used at token-level? 
There is no comment on training vs. test split and setup which is a major gap. On which instances and problems are you training this method and then testing it? How does the test differ from training? Are these synthetic instances? What are their problem distributions? How and what data are you tuning the hyper-parameters.  

Is there an ablation that compares AlignOPT with simply training the graph solver on raw numeric features without LLM inputs. Is this AlignOPT (w/o Task Rep)? Is this exactly the same setup without the LLM embeddings? I cannot tell from the insufficient descriptions and missing technical details. I assume this only removes the TASK description, but the LLM-derived text features of the instance are still used --which is not a direct ablation to see what the LLM brings to the table. Maybe it is AlignOPT (GNS) but I cannot tell.

Regarding the ablations; it is not clear to me (again, not sufficiently described) how does the pre-training change. Because the pre-training objectives TGM and TGC requires "text embeddings". Are you skipping multi-model training and train from scratch with equivalent compute budget? OR are you training pre-training with graph-only objectives (constrastive among nodes) in matching pretraining compute? OR sth else? 

The task description text is exactly identical for every instance, yes? Then I am having a hard understanding or the rationale behind why adding the same embedding again and again (which does not distinguish much) improves the performance. Have you considered a "control" that adds random text descriptions (but kept identical across instances) OR adversarial settings with on-purpose incorrect problem descriptions. IF task descriptions are to help, these ablations should perform worse, yes? I am suspicions of added model capacity (independent of what the text is saying) to somewhat help. 

We need the exact same data, same training budget, same RL fine-tuning to compare both 1) LLM-aligned model and 2) GNN-only model. So that we can attribute improvements correctly to LLM information. And within the LLM version, I would also be curious of random/adversarial task descriptions.

I am not sure I am following the difference between Table 1 and Table 2. Table 1 includes TSP, CVRP, and KP whereas Table 2 incudes again TSP, CVRP, KP and also MVCP and SMTWTP. Why? IF I take two same rows, say TSP LNCS from Table 1 and Table 2, are the results the same? why is one presenting the gap percentage and the other average objective? Btw, where does the optimal solutions come from? Why does Table 1 not present Goal? But then Table 2 has GOAL on TSP, KP, CVRP, so why was it missing from Table 1?  I really feel you can better streamline these Tables/Presentations. 

As rightly noted in the paper; LNC, GOAL, and AlignOpt are the closests. If I take the results for LNCS and AlignOpt, say from Table 1, for KP both are almost 0% so it is not immediate whether the difference is significant. If I look at the objective values, from Table 2, LNCS, Goal, AlignOpt are almost identical for TSP 20 and so on. In most cases, the differences are after the digit, and in cases in the 2nd or 3rd index. How significant is that? 

Minor suggestion: "aligns LLMs with graph-based neural solvers". I think this paper is the other way around: "align graph-based neural solvers with LLMs"

Typo: ". , thus"

### Questions
See above

### Soundness
2

### Presentation
1

### Contribution
2
