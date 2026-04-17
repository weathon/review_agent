# Exploring Effective Terminal State: A Null-Model-Guided Graph Diffusion Model

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Graph diffusion models have shown promise in generating complex networks, but they often suffer from two critical limitations: On the one hand, terminating the forward diffusion in pure Gaussian noise graph erases the intrinsic structural signatures of the original network, leading to sub-optimal generative outcomes. On the other hand, the unconstrained diffusion trajectory progressively obliterates topological characteristics, resulting in complete structural degradation. To address these issues, we propose Null-Model-Guided Graph Diffusion (NMG-GD), a principled framework with tailored designs for graph generation. 
First, we claim that traditional isotropic priors (e.g., Gaussian or fully structured graphs) distort salient topological features. Instead, we adopt a null-model distribution as the forward diffusion endpoint, which explicitly preserves critical network statistics such as degree sequences and clustering coefficients—ensuring global consistency. 
Second, we derive a null-model-guided continuous-time stochastic differential equation (SDE) and introduce the Position-enhanced Graph Score Network (PGSN). PGSN ingests both continuous and quantized adjacencies, fusing random-walk, shortest-path and null-model cues in a permutation-equivariant encoder,which can significantly elevates sample quality. Extensive experiments on three public datasets (including social and biological networks) demonstrate that NMG-GD achieves state-of-the-art performance. It shows the significant advantages in structural similarity and generation efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel graph diffusion model, NMG-GD, that addresses that existing graph diffusion methods use an unstructured or overly simplistic terminal state, which erases crucial topological properties of the original graph, forcing the model to reinvent complex statistics during the reverse process. NMG-GD guides the forward diffusion process to terminate at a first-order null model graph, preserving essential global statistics.  Extensive experiments on synthetic and real-world biological networks demonstrate that NMG-GD achieves competitive performance across multiple structural and neural network-based metrics.

### Strengths
1. This paper conducts comprehensive experiments on multiple datasets, showing that NMG-GD consistently outperforms a wide range of baselines, including recent diffusion models, across both classical structural metrics and modern neural metrics.

2. This paper conducts an ablation study comparing a variant without the full noise design and a parameter sensitivity analysis, helping to understand the contribution of its components.

3. The derivation of the null-model-guided SDE for both the forward and reverse processes is detailed and appears sound.

### Weaknesses
1. The paper emphasizes the use of directional noise as a key contribution over prior work. However, the connection between the proposed method and the concept of directionality is not fully clarified. The introduced noise $ε^'$ remains Gaussian, albeit with a shifted mean. It is better to discuss how this constitutes directional guidance rather than isotropic noise.

2. The paper does not explain why the proposed NMG-GD works so well. Is the primary benefit simply that it provides a better-initialized starting point for the reverse process? Or does it fundamentally reshape the loss landscape of the score function, making it easier to learn? It is better to conduct a deeper analysis, for example, by visualizing the diffusion trajectory.

3. The paper does not discuss the computational complexity or the practical overhead of NMG-GD compared to other baselines.

### Questions
1. Why is the discretization threshold set at 0.3 for converting the continuous adjacency matrix into a discrete graph? What was the rationale behind selecting this specific value? Was this threshold fine-tuned, and how sensitive is the model's performance to it?

2. NMG-GD uses a continuous adjacency matrix and applies a fixed threshold of 0.3 for discretization. What are the potential effects of exploring other discretization strategies, such as sampling edges from a Bernoulli distribution that is parameterized by the continuous values? Additionally, how does the choice of the discretization method influence the quality of the final discrete graph?

### Soundness
2

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
3

### Summary
This paper proposes Null-Model-Guided Graph Diffusion, a framework that uses a null-model distribution as the terminal distribution of the forward diffusion process. This terminal distribution is designed to preserve critical network statistics (e.g., degree sequences, clustering coefficients), which in turn guides the reverse process to generate more realistic and structurally valid graphs. The authors provide formal derivations for the forward process, the reverse (denoising) process, and the corresponding training objective. The proposed method is evaluated on three graph generation benchmarks (Community-small, Ego-small, and Enzymes), reportedly achieving significant improvements on several graph quality metrics.

### Strengths
- Theoretical Soundness: The authors provide formal derivations for the forward process, the reverse process, and the training objective, establishing a sound theoretical foundation for the proposed framework.

- Clarity of Exposition: The paper is well-written and logically structured. It begins with a clear and strong motivation, followed by systematic derivations of the diffusion process for using a null-model distribution as the prior, network architecture. This logical progression makes the paper easy to follow.

### Weaknesses
- Ambiguity and Lack of Ablation for the Null-Model Prior: 

The manuscript provides an insufficient treatment of the null-model prior. The precise mechanism by which it preserves structural statistics remains underspecified, and its **computational cost** is not analyzed. Crucially, the paper lacks an **ablation study** to justify the choice of preserved characteristics, leaving the question of which statistics are most impactful unanswered.

- Unaddressed Dependency on Ground Truth in Reverse Process: 

The reverse process, as formulated in equation 9, seems to depend on the ground-truth graph $A$ via the function $f$. This creates a **circular dependency**, as A is not available at inference time. The paper does not explain how this issue is handled, representing a significant gap in the methodology.

- Limited Experimental Scope and Inadequate Baselines: 

The paper's empirical evaluation is insufficient. It fails to compare against several key state-of-the-art graph generation models, such as DiGress[1] and GraphBFN[2]. Additionally, the experiments are confined to a limited set of datasets, omitting crucial and widely-used benchmarks like QM9 and ZINC250k. This lack of comparison against strong baselines on standard datasets makes it difficult to properly assess the method's performance and scalability. 

Additionally, the proposed method **underperforms SOTA on Enzymes dataset** in all classical metrics, which may suggest the method could not scale to large graphs.

[1]Vignac, C., Krawczuk, I., Siraudin, A., Wang, B., Cevher,V., and Frossard, P. Digress: Discrete denoising diffusion for graph generation
[2] Yuxuan Song et al. Smooth Interpolation for Improved Discrete Graph Generative Models

### Questions
What is the definition of $q_{null}$? Does it depend on the ground truth graph $A$?

How dose the null-graph distribution handles node features?

### Soundness
2

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
3

### Summary
This paper introduces Null-Model-Guided Graph Diffusion, aiming to address a key limitation of existing graph diffusion models, structural degradation when the forward process terminates in pure Gaussian noise. The authors propose using a null model as the terminal distribution, preserving global graph statistics such as degree and clustering. They derive a null-model-guided SDE and design a Position-Enhanced Graph Score Network to capture both continuous and discrete structural cues.

### Strengths
1. Innovative use of the Null Model as the terminal state of the diffusion model
The paper replaces the conventional Gaussian endpoint in diffusion with a null-model distribution, preserving key topological properties such as degree sequence, clustering, and preventing complete structural collapse during the forward process.
2. Enhanced score network design
The proposed Position-Enhanced Graph Score Network integrates continuous adjacency signals with discrete structural encodings, achieving permutation equivariance and improved structural recovery.

### Weaknesses
1. Lack of analysis on computational efficiency:
Although the authors claim improved sampling efficiency, the paper does not provide detailed runtime or memory comparisons against prior diffusion models (such as GraphGDP, Pard). The added null-model computation and SDE formulation may introduce nontrivial overhead.
2.No ablation on the score network architecture:
The proposed PGSN combines several structural encodings, but the paper does not isolate the contribution of each (e.g., RWSE vs. SPD). Without such analysis, it is unclear which component primarily drives the performance gain.
3. Insufficient exploration of parameter sensitivity:
The null-model weight η significantly influences generation quality, yet the sensitivity study is limited to a single dataset. The paper lacks a broader discussion on how different graph domains or diffusion schedules affect stability and convergence.
4. Missing qualitative or visual interpretability analysis:
While quantitative metrics are strong, the paper provides minimal visualization or structural interpretation of generated graphs. More examples or analysis (such as motif frequency, connectivity distribution) would help clarify what specific aspects of topology the null-model guidance preserves.

### Questions
1. Clarification on the null model formulation:
The authors are encouraged to provide a more detailed explanation of the term q_null. Specifically, what is its explicit mathematical form, and what does ​A-q_null represent in practice? Moreover, further justification is needed for why the term (A-q_null) effectively encourages the generated graphs to preserve key statistical properties (e.g., degree distribution, clustering). A clearer theoretical or intuitive interpretation would strengthen the readers’ understanding of this mechanism.

2. Potential overfitting and generalization concern:
Since the null-model constraint enforces structural similarity to the training graphs, it raises the question of whether the model might tend to memorize specific graph statistics rather than learning more generalizable topological patterns. If this is an important factor, it would be valuable for the authors to include additional evaluation metrics—such as diversity, novelty, or generalization scores—to quantitatively assess whether the generated graphs go beyond simply reproducing training-set statistics.

### Soundness
3

### Presentation
2

### Contribution
2
