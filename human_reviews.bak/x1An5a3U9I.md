# Advancing Graph Generation through Beta Diffusion

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Diffusion models have excelled in generating natural images and are now being adapted to a variety of data types, including graphs. However, conventional models often rely on Gaussian or categorical diffusion processes, which can struggle to accommodate the mixed discrete and continuous components characteristic of graph data. Graphs typically feature discrete structures and continuous node attributes that often exhibit rich statistical patterns, including sparsity, 
bounded ranges, skewed distributions, and long-tailed behavior. To address these challenges, we introduce Graph Beta Diffusion (GBD), a generative model specifically designed to handle the diverse nature of graph data. GBD leverages a beta diffusion process, effectively modeling both continuous and discrete elements. Additionally, we propose a modulation technique that enhances the realism of generated graphs by stabilizing critical graph topology while maintaining flexibility for other components. GBD competes strongly with existing models across multiple general and biochemical graph benchmarks, showcasing its ability to capture the intricate balance between discrete and continuous features inherent in real-world graph data. Our PyTorch code is available at https://github.com/xinyangATK/GraphBetaDiffusion.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents a Graph Beta Diffusion for the natural conflict between graph structure and diffusion model. A modulation technique is proposed to stabilize the topology of the key graph.

### Strengths
1. The motivation of this paper is very reasonable. Diffusion model and graph structure are incompatible.
2. The paper is well written and is easy to understand.
3. The Concentration modulation method is very cleverly designed. It seems to make good use of the topological properties of the graph structure.

### Weaknesses
1. The authors give no clear evidence as to why beta diffusion is effective at modeling the graph structure. The discrete distribution can take many forms. Or why not use BFN to model the distribution of a graph.
2. In terms of experimental results, if concentration modulation is not included, the effect of beta diffusion is not better than Digress and Grum. This does not effectively explain the effectiveness of beta diffusion. Another point of view is, can the ordinary diffusion model also use this idea to control the noise process?

### Questions
1. How to explain the beta distribution is better suited to the graph than other discrete distributions.
2. Can concentration modulation be used with other diffusion methods?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Graph Beta Diffusion to address the inherent conflict between graph structure and diffusion models. It also proposes a modulation technique aimed at stabilizing the topology of the key graph.

### Strengths
1.The paper is well-articulated and easy to comprehend.
2.The rationale behind this paper is quite sound, as diffusion models and graph structures do not align well.

### Weaknesses
1.Could the author offer a comparison of complexity? For instance, the calculation of concentration appears to be time-consuming. Authors should provide a comparison of complexity with other methods, preferably including both theoretical analysis as well as experimental data. (e.g. complexity comparison between GDSS and Digress)
2.Lack of detection for networks with other topological properties. BA networks, for example, follow a power-law distribution.
3.The sample rate analysis and scalability of this method are not discussed in detail. For example, in an SDE-based diffusion framework, we can use many different accelerated sampling methods, as well as perform conditional generation. But is this approach effective?

### Questions
1.For a graphset data, a concentration modulation is calculated for each graph and then concat them into a matrix. Will this cause a large consumption of time? And how should Concentration modulation be defined during generation? Because we don't know some of the properties of the graph in advance when sampling from the beta distribution?
2.What is the sampling rate of this method? And how does the sampling rate affect the quality of the generated graphs?
3.Can this approach be extended to conditional diffusion generation? Other methods like GDSS or Digress can easily combine with conditional generation in theory.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a novel graph generation model by adapting the recent diffusion process work (Zhou et al., 2023) to handle graph-structured data. The authors assess the performance of the proposed model against state-of-the-art methods on both synthetic and real-world datasets, including molecular datasets commonly used in graph generation tasks.

### Strengths
- Adapting the recent beta diffusion process to graph-structured data.
- The model is evaluated on both synthetic and real-world data, including widely used molecule datasets.
- The structure of the paper is generally well-organized.

### Weaknesses
- While the model adapts the diffusion process for graph data, there is limited discussion on unique contributions beyond this adaptation. Emphasizing the model’s distinct aspects and theoretical advancements would strengthen the paper's impact.
- Similarly, the paper heavily references Zhou et al. (2023), making it challenging for readers who are not familiar with this work to fully understand the technical content. 
- The reported results raise questions about the practical significance of the proposed architecture, with inconsistent comparisons across tables (missing baselines and metrics like Spec. and V.U.N.), lack of standard deviation reporting in Tables 2 and 3, and missing entries, all of which reduce confidence in the experimental evaluation.

### Questions
**Questions:**

- The methodology mainly adapts the work of Zhou et al. (2023) for graph-structured data and raises concerns about the originality of the paper. Could the authors emphasize their unique contributions beyond this adaptation?
- The reported improvements over baseline methods appear minimal. Could the authors clarify the practical impact of these results?
- Tables 1 and 2 use different baseline models, omitting certain comparisons (e.g., Wave-GD, GraphARM, GNF, GraphVAE in Table 1; GruM, GDSS+TF, SPECTRE in Table 2). Tables are missing standard deviation values (Tables 2 and 3) and certain metrics (e.g., Spec. and V.U.N in Table 1), with no explanation for missing entries. Could the authors clarify their choices?
- Does the proposed model offer any guarantees regarding the connectivity of the generated graphs?


**Additional comments:**
- The provided equations in the paper are a bit challenging to follow, especially for readers unfamiliar with the work of Zhou et al. (2023). The authors might provide additional background or a summary of this prior work to improve readability. 
- The graph images in the first row of Figure 2 are unclear, and it is difficult to see the nodes and edges. The authors might consider improving the quality or resolution of these visuals.
- Similarly, the SBM graph in Figure 3 is hard to interpret due to its size. Providing an adjacency matrix or alternative visualization might enhance readability.

Given the concerns outlined above regarding the experimental evaluations and originality of the paper, I recommend a rating of 5 for the paper.

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
2

### Summary
The paper proposes graph beta diffusion for the task of generating small graphs such as molecules. The core concept is approaching this task by adapting beta diffusion (Zhou et al., 2023), which is meant for generating data within bounded ranges (e.g., 0 to 1 for non-edge and edge) and is based on the beta diffusion rather than the Gaussian distribution. In addition to this core idea, the paper also explores several design ideas to strengthen the proposed method, including "concentration modulation," which modify the diffusion distribution for important positions in the graph. The paper concludes with experiments on real and synthetic datasets, showing that the proposed method is at least competitive with other recently proposed methods in terms of matching certain graph statistics. There are also ablation experiments supporting the lift from the design ideas.

### Strengths
- The core idea of applying a diffusion method based on the beta distribution to graph generation seems logical.
- The graph generation task that this paper tackles is enjoying a lot of attention recently.
- The writing is generally clear.
- The evaluation is fairly extensive, showing that the method is at least competitive with other recent approaches, and also includes ablations for the proposed design components.

### Weaknesses
- Since the main thrust of this paper is approaching graph generation with a diffusion method that is suited for bounded data, like the probability of each edge, there could be more discussion of alternatives to beta diffusion. What makes beta diffusion (Zhou et al., 2023) more suited for this task as opposed to, e.g., Dirichlet Diffusion Score Model (Avdeyev et al., 2023) or Dirichlet Flow Matching (Stark et al., 2024)?
- Relatedly, given that the main thrust is application of beta diffusion, there could be more intuition given for beta diffusion itself, given that it was proposed in a recent paper.
- With the exception of concentration modulation, little of the design seems specific to graphs, making this largely an application of a prior work to a specific domain. There could be more elucidation of what is gained from using beta diffusion vs standard diffusion in a graph context, e.g., some empirical or theoretical work showing that the beta distribution is more suited to capturing certain graph motifs.
- $\omega$ seems to be a key hyperparameter controlling the weight of two losses in beta diffusion. It seems unexplored here beyond reusing the value in the beta diffusion paper.

### Questions
Some questions are given above. Additionally:

- As stated on line 458, this method predicts $\mathbb{E}[\mathbf{G}_0 | \mathbf{G}_t$], which is not the standard diffusion setup, and as stated on line 160, this requires a neural network that predicts the conditional expectation of $\mathbf{G}_0$ given $\mathbf{G}_t$. Could the authors please expand on this design decision and its possible consequences? In particular, is it possible this reduces the diversity of samples?
- Perhaps relatedly, regarding concentration modulation: How is $\eta$, which corresponds to important positions in the sample, set at sampling time? Is the graph distribution generated by this method equivariant to permutations of the nodes?

### Typos
- Line 376 "Transofrmer"
- Line 522 "Centrailities"

### Soundness
3

### Presentation
2

### Contribution
3
