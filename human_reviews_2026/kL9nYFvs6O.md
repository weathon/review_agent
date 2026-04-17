# Divide, Harmonize, Then Conquer It: Shooting Multi-Commodity Flow Problems with Multimodal Language Models

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 8

## Abstract
The multi-commodity flow (MCF) problem is a fundamental topic in network flow and combinatorial optimization, with broad applications in transportation, communication, and logistics, etc. Nowadays, the rapid expansion of allocation systems has posed challenges for existing optimization engines in balancing optimality and tractability. In this paper, we present Pram, the first ML-based method that leverages the reasoning power of multimodal language models (MLMs) for addressing the trade-off dilemma—a great need of service providers. As part of our proposal, Pram (i) quickly computes high-quality allocations by dividing the original problem into local subproblems, which are then resolved by an MLM-powered "agent", and (ii) ensures global consistency by harmonizing these subproblems via a multi-agent reinforcement learning algorithm. Theoretically, we show that Pram, which learns to perform gradient descent in context, provably converges to the optimum within the family of MCF problems. 
Empirically, on real-world datasets and public topologies, Pram achieves performance comparable to, and in some cases even surpassing, linear programming solvers (very close to the optimal solution), and substantially lower runtimes (one to two orders of magnitude faster). 
Moreover, Pram exhibits strong robustness (<10\% performance degradation under failures or bursts), demonstrating MLM's generalization ability to unforeseen events. 
Pram  is objective-agnostic and seamlessly integrates with mainstream allocation systems, providing a practical and scalable solution for future networks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposed a LLM-driven solver for the multi-commondity flow problem. The problem is firstly decomposed as $|\mathcal{V}|$ sub-problems, then the sub-problems are transformed into images, and a MLM is prompted to solve these sub-problems. A multi-agent reinforcement learning algorithm with counterfactual objective is applied to fine-tune the MLM, so that the sub-problems can be solved in a cooperative way that maximize the global objective. Experimental results show that its performance is comparable to traditional baselines.

### Strengths
- This paper shows the potential of multimodal language models (MLMs) on the multi-commodity flow problem
- Proper techniques applied to the MLM to improve the efficiency/quality, including LoRA, global context, and the reprogramming technique
- Successful application of multi-agent reinforcement learning on MLMs with a counterfactual objective
- Good ablation studies
- Sufficient details in the appendix

### Weaknesses
1. The main claim of the paper, the necessity of MLMs, is not very sufficiently supported by empirical results. For example:
- The paper mentioned that "(MLMs) exhibit emergent abilities that were not explicitly programmed in to them during pre-training on massive data, such as mathematical problem reasoning and generalization to unforeseen conditions" (line 66). I generally agree with this. However, there is no empirical result shown in this paper that MLMs generalize better than other DNNs on unforeseen conditions. In Figure 8, no other ML-based baselines are presented.
- The paper mentioned that "(MLMs) mitigate the costs of retraining and handcrafting specialized DNNs for complex inputs" (line 69). However, even with LoRA, retraining an MLM is still typically considered more costly than traditional DNNs (especially considering the GPU memory requirements). I generally agree that MLMs can be a more "universal" solution than specialized DNNs, which would be easier to adapt between different type of optimization problems (ideally only with a change of the prompt) while other approaches may require a major change. However, this paper is scoped only on the MCF problem with very specific designs (e.g., the decomposion approach). In such a way, this advantage is not very relavant to this paper.

2. The only ML-related baseline approach (Valadarsky et al., 2017) seems to be very old. While some more recent approaches like GDDR (Hope & Yoneki, 2021) and MARL-GNN (Bernárdez et al., 2021) are mentioned in the paper, they are not included in the baselines. 

3. Some claims in the paper are not very clear to me, including:
- In the abstract, the paper says "dividing the original problem into local subproblems via an MLM-powered agent" (line 20). But if I understand correctly, the division is not done by an MLM agent but rather "rule-based" ("treating commodities from the same source as one subset", line 157).
- The convexity of the problem is illustrated in a case study and described as an assumption (and all the proof rely on this assumption). Given the closed-form formulation of the problem (appendix B.1), can we make it clear whether it is convex or not?
- The proof is based on the fact that "gradient descent can effectively solve the problem" (line 253). Unfortunately, gradient descent cannot effectively solve the problem. Gradient descent typically works for unconstrained problems or soft-constrained problems (so that the constraints can be transformed to a penalty term in the objective). However, this is a hard-constrained optimization problem, in which all the constraints in equation (1,2,3) should never be violated. In such a way, specific constrained optimization approaches should be applied (typically with KKT condition and lagrange multiplier). Or, empirically, if gradient descent is truely an effective choice, why not solve the problem simply with gradient descent?

4. Using bitmap images to represent graphs can be pretty lossy, especially when the graph is complicated. For example, in the examples provided in the paper, some edge capacities cannot be clearly identified due to heavily crossed links (e.g., row 3, column 1 of Figure 15, the capacity between node 2 and 12 cannot be identified)

### Questions
See the questions in "weaknesses".

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper presents PRAM a reinforcement pipeline architecture to solve
multi-commodity flow (MCF). PRAM works by decomposing MCF problems, compute the
representations of subproblems and recombine the subproblem representations to
return a complete solution. The context-free representation computation is
performed by a vision encoder and tokenizer (enriched with a learned global
context). Then graph and text vectors are passed to a multimodal language
model to combine modalities and compute MCF subproblem representations. The
global recombination is performed by multiagent reinforcement learning using
counterfactual policy gradient.

Communication between subproblems is implemented through LoRA on the multimodal
language model, and through the global context vectors which are the query of a
cross-attention mechanism.

Experimentally, the model shows strong results on real-world datasets and public
topologies, outperforming previous reinforcement learning approaches.
Interestingly, this model exhibits robustness as it is able to solve
problems with link failures and flow bursts with performance degradation under
10%.


The paper is very well written, and the method is novel (to the best of my
knowledge). The reason for using an off-the-shelf LM is clearly motivated, and
the adaptation mechanism is thoroughly described (although quite convoluted) and
justified in ablation studies.

### Strengths
1. Well written and clear
2. A novel method based on decomposition
3. Strong empirical results, with intersting robust predictions

### Weaknesses
- the neural architecture is quite complex

### Questions
1. the use of a MLM is justified, but the model w/o MLM process the input in a
  puzzling way. I can understand how the topology is processed by a GNN, but I
  don't understand how the textual descriptions are handled by a feedforward
  layer (size may change for instance). Can you elaborate? Is there a global
  context vector in this case?

### Soundness
3

### Presentation
4

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
The paper showcases an agentic AI application leveraging multimodal large language models (MLLMs) to solve the multi-commodity flow (MCF) problem through a multi-agent Reinforcement Learning (RL) framework. The method aims to resolve the scalability bottleneck of the classic solver by first dividing the network into subnetworks, then applying MLLM agents to solve each one, and finally ensuring global consistency through the harmonization of the subproblems.

### Strengths
1. **Novelty.** To the best of my knowledge, this paper is among the first to leverage the reasoning power of modern Multimodal Language Models (MLMs) for a classic combinatorial optimization problem (MCF). It bridges the gap between large-scale AI models and network optimization.
2. **Significant Performance (Speed and Scalability).** Based on the results, PRAM is significantly faster on large-scale topologies. The ablation study confirms that the partitioning is critical to this scalability.
3. **Near-Optimal Allocation Quality.** Unlike previous ML methods (like DRL) that sacrifice quality for speed , PRAM achieves near-optimal performance, often ranking just behind an LP solver with perfect future knowledge. In some cases (minimizing link usage), it even surpassed the LP solver's results.
4. **Strong Generalization and Robustness.** PRAM demonstrates a key advantage of ML-based approaches: it generalizes well to unforeseen conditions. It shows minimal performance degradation (<10-16%) during simulated link failures and demand fluctuations, scenarios where traditional prediction-based solvers struggle.
5. **Solid Theoretical Foundation.** The paper provides a convincing theoretical justification for why PRAM works. It connects the problem's structure (convexity of MCF objectives, Theorem 1) to the model's learned capability (MLMs can simulate gradient descent, Theorem 2).
6. **Thorough and Convincing Evaluation.** The experimental validation is comprehensive. It leverages multiple real-world and large-scale synthetic datasets , tests three different MCF objectives , and includes a detailed ablation study that validates the importance of every major component (partitioning, the MLM backbone, LoRA, context, and MARL).

### Weaknesses
1. **Cost in Training**. From my understanding, the superior scalability of PRAM is achieved by truncation of backbone layers and parameter-efficient adaptation. This can potentially be a problem during deployment.
2. **Memory overhead**. Input to the model are images of the subnetworks, which are sensitive to the partition parameters. I wonder if this can lead to significant memory overhead, especially when different subproblems (nodes) have similar network views that are processed redundantly.
3. **Dependency on Pretrained Backbone**. The success of PRAM is "heavily" dependent on the quality of the underlying pretrained MLM. I wonder how performance varies with different backbones.

### Questions
1. **Convergence**. My understanding of Theorem 1, Theorem 2, and Lemma 1 is that they only guarantee convergence for PRAM to locally optimal agents with a limited number of training steps. How can we guarantee that PRAM can strictly learn a global optimal set of agents?
2. **MARL Adaptation**. The MARL algorithm uses counterfactual reasoning, asking how the global objective would change if one agent acted differently. Does the computational cost of this "counterfactual baseline" scale linearly with the number of agents (i.e., the number of nodes )? If so, could this harmonization step become a new bottleneck for extremely large topologies, even if the problem is partitioned?
3. **Comparison to LP Solvers**. The paper notes that PRAM's results for minimizing link usage were "even better than those of LPs". Based on my understanding, since LP solvers are guaranteed to find a near-optimal solution given a specific demand matrix, how is this possible? Does this imply the LP solver was using predicted demands, while PRAM (using historical data) generalized better to the actual future demands used for evaluation?
4. **Role of _Reasoning_**. The paper claims the MLM "_...learns to perform gradient descent in context..._". Is this an emergent, implicit behavior, or is the model explicitly structured to perform GD steps? The proof for Theorem 2 constructs specific attention weights to demonstrate that this is possible, but does the model actually learn this specific structure during fine-tuning?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes PRAM, a novel framework that applies multimodal large language models (MLMs) to solve the multi-commodity flow (MCF) problem. Instead of handling large-scale MCFs as a single optimization, PRAM decomposes the process into a structured Divide–Harmonize–Conquer pipeline.

1. Divide:  
   The global network is partitioned into node-based subproblems. Each subproblem is encoded as a multimodal input, with network topology represented as images and demand profiles represented as text.

2. Harmonize:  
   A collection of MLM-based agents is coordinated via a lightweight multi-agent reinforcement learning framework, enhanced by LoRA (Low-Rank Adaptation) and learned context embeddings for efficient cross-agent communication.

3. Conquer:  
   The framework achieves global convergence through the implicit gradient-descent-in-context property of MLMs, allowing distributed subproblem solutions to align toward an optimal global flow.


Empirically, PRAM demonstrates near-optimal allocations across ten real and synthetic network topologies, often outperforming classical LP-based solvers by 1–2 orders of magnitude in runtime while maintaining <8% optimality gap. Theoretically, the paper establishes convergence guarantees under convexity assumptions and provides an interpretability analysis of the learned context embeddings.

### Strengths
1. Using multimodal LLMs as distributed solvers for MCF is genuinely new. Prior work on neural optimization largely relied on GNNs or RL; this paper reframes the task as multimodal reasoning over partitioned problems, which feels like a meaningful step forward.

2. The paper’s theoretical discussion—particularly the connection between convexity in MCF and the ability of MLMs to simulate gradient descent is well argued and gives the approach more credibility than most ML-for-optimization papers.

3. The experiments are comprehensive, covering several datasets and objectives. PRAM consistently performs close to LP while running 10–100× faster. The robustness tests (link failures, demand fluctuations) also support the generalization claim.

4. The choice to use LoRA and shared context embeddings keeps the model efficient. The MARL setup is simple but effective for ensuring coordination between agents.

### Weaknesses
1. The multimodal input design (topology as image + demand as text) is acknowledged to introduce layout sensitivity and encoding bias.
2. Although faster than LP in inference, fine-tuning large MLMs still requires considerable resources.

### Questions
1. How is the visual encoding of topology constructed, and how sensitive are results to the layout or resolution of the generated subgraphs?

2. Would representing the topology as a matrix or sequence (instead of an image) yield comparable results?

### Soundness
4

### Presentation
4

### Contribution
3
