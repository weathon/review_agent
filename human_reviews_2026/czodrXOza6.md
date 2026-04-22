# On Diffusion-based Multiplex Dynamic Attributed Network Generator

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 2

## Abstract
Multiplex dynamic attributed networks are essential for modeling complex systems, such as social platforms and telecommunication networks, where each layer represents distinct interaction types and attribute dynamics. However, existing generative models fall short in capturing their structural-semantic coupling, temporal evolution, and inter-layer dependencies, failing to reproduce network-level emergent behaviors like explosive synchronization and hysteresis. We introduce MulDyDiff, a diffusion-based generative framework that incorporates attribute-aware dynamic transition-based denoising, cross-layer correlation-aware denoising, and behavior-aware guidance. These components are unified through a novel Behavioral-guided Attributed Cross-layer Temporal (BACT) loss. Evaluations of three real-world datasets demonstrate that MulDyDiff consistently outperforms state-of-the-art dynamic graph generators, achieving 6%-9% improvement in terms of temporal metrics, offering a comprehensive solution for realistic multiplex dynamic attributed network synthesis.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes MulDyDiff, a diffusion-based framework for generating multiplex dynamic attributed networks that evolve across time and layers. It addresses well-known limitations of prior models by jointly modeling structure–attribute coupling, temporal evolution, and inter-layer dependencies. The method further introduces behavior-aware guidance to reproduce emergent phenomena such as explosive synchronization and hysteresis. Experiments on three real-world datasets show consistent improvements (6–9%) over strong baselines, with clear ablation evidence supporting each component.

### Strengths
1. Clear and meaningful motivation
The paper nicely motivates why existing dynamic graph generators fall short.
2. Novel diffusion-based architecture
The proposed MulDyDiff framework extends discrete diffusion models to handle multiplex dynamic attributed graphs, combining attribute-aware, cross-layer correlation-aware, and behavior-aware denoising processes. This integration is technically sound and original to me.
2. Strong experimental results
The model shows consistent improvements over competitive baselines on three real-world datasets. The ablation studies are also convincing and clearly show why each component matters.

### Weaknesses
1. Behavioral validation is limited
The “behavior-aware” part is conceptually interesting, but I’d like to see stronger evidence that the generated networks really show realistic emergent behaviors
2. Baselines could be more up-to-date
The comparisons mostly cover traditional dynamic graph models. It would be nice to include more recent diffusion-based graph generators for completeness.
3. Efficiency and scalability discussion is light
The paper reports training/sampling time but doesn’t really analyze scalability with graph size or diffusion steps. Some discussion on this would help assess practicality.

### Questions
1. Why did you choose KS metrics only? Have you tried other dynamic metrics like MMD or behavior-based measures?
2. Can you show results comparing with and without the behavior-aware loss L_behavior to clarify its real impact?
3. How does the computational cost grow with the number of layers and timestamps? Would this scale to very large networks?

### Soundness
3

### Presentation
2

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
This paper introduces MulDyDiff, a diffusion-based framework for generating multiplex dynamic attributed networks. The model integrates attribute-aware temporal denoising, cross-layer correlation modeling, and behavior-guided regularization using Kuramoto-based synchronization descriptors. The approach is technically interesting and addresses an important gap in dynamic network generation, extending diffusion-based methods to more complex, multi-layer temporal settings. However, the central claim that MulDyDiff can reproduce emergent, system-level behaviors (absent from prior models) is not substantiated by the empirical evaluation. The empirical results show limited and/or uneven gains across datasets/metrics, and it remains unclear whether these improvements stem from genuine modeling insights or simply increased parameter capacity. The paper would be significantly strengthened by targeted experiments that directly assess emergent behavior, a more systematic analysis of trade-offs across network measures, and clearer discussion of model parameterization. Overall, this is a promising direction, but the current paper falls short of demonstrating sufficient practical or conceptual impact.

### Strengths
The paper is technically well written and motivated by a clear limitation in current dynamic graph generators. The incorporation of cross-layer coupling and behavior-aware loss is novel, and the empirical setup includes multiple datasets and ablations.

### Weaknesses
- Unsubstantiated core claim. The paper’s main claim (that existing methods fail to reproduce emergent behaviors such as explosive synchronization or hysteresis) is not evaluated. The reported metrics (e.g., KS distance) capture temporal distributional differences but, because they (a) consider distributions over the full graph and (b) are averaged across time, they are unlikely to detect failures in system-level emergent dynamics. If the authors believe such behavior explains the gains in Table 1, a qualitative exploration of specific graph transitions that MulDyDiff captures (but baselines miss) would strengthen the case. Otherwise, evaluation using new metrics explicitly designed to quantify emergent behavior is needed.
- Limited and uneven empirical support. The model is compared to baselines on only two of the three datasets, and results are mixed: gains on some measures (node behavior, BC) come at the expense of degradation on others (RW). Without deeper analysis of these trade-offs, it is unclear whether the improvements reflect genuine modeling benefits or selective optimization of certain graph statistics. 
- Model complexity. The cross-layer coupling and behavior-guided components substantially increase model parametrization, yet there is no comparison of parameter counts, asymptotic complexity, or parameter efficiency relative to baselines. The modest empirical improvements could simply stem from higher model capacity. This is not inherently a flaw, but given that the gains are not consistent across all network metrics, it remains unclear how much of the improvement derives from MulDyDiff’s conceptual innovations versus its expanded parameterization.

### Questions
See weaknesses.

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
The authors of the work propose a method to generate synthetic dynamic multiplex graphs, where nodes are assumed to have time-varying attributes that influence the link generation in multiple layers. The proposed diffusion-based model considers dynamic node attributes, models correlation of edges across layers as well as desired emergent behavior based on the order parameter of the Kuramoto model for self-organized synchronization. The denoising process progressively generates subsequent network snapshots that are conditioned on all previous snapshots of the dynamic graph, thus giving rise to a sequence of snapshots of a discrete-time temporal graph. 

The method is evaluated in three data sets on multiplex graphs from Wikipedia, Twitter, and the Superuser forum. For two of these data sets, the authors compare the resulting distributions of structural metrics (e.g. centralities) to the ground truth and benchmark their approach in comparison to four baseline temporal graph generators. The results show moderate improvements in the chosen metrics. An ablation study investigates the impact of different model components in the third data sets, which suggests that both the temporal link generation and the cross-layer correlation mechanism contribute to the results.

### Strengths
- [S1] The paper considers a diffusion-based generative model for temporal graphs, which could have interesting applications in practice. 

- [S2] The paper considers the generatiion of multiplex attributed graphs, which have many applications. 

- [S3] The generation process consider the emergent behavior of a synchronization process in the generated graphs, which is a potentially interesting idea.

### Weaknesses
- [W1] The practical motivation of generating temporal multiplex networks that exhibit explosive synchronization under the Kuramoto model is weak, see Q1.

- [W2] The model decription is hard to follow and some aspects of the generation process are unclear and lack intution, see my detailed comments in Q2 - Q3 below.

- [W3] The dicussion of related works is weak and the contribution over prior works on generative temporal graph models is unclear, see my detailed question Q4.

- [W4] The experimental evaluation only uses two of the three mentioned data set, and the ablation study is conducted on a third data set, but not the first two. Evaluation metrics are not properly motivated and some of them are not explained and defined. It is unclear how hyperparameters for the main results have been chosen (and even what was their value). The experimental evaluation is generally weak and does not support the claims of the paper, see detailed comments in Q5.

### Questions
- [Q1] While I like the general idea, I did not get the practical motivation of the behavior guidance, i.e. why do we want to generate temporal graphs that specifically exhibit a certain synchronization behavior in the Kuramoto model in the first place? Could the authors explain this motivation of their work better? 

Also, the motivation mentions hysteresis as another important emergent phenomenon, but as far as I see this is actually not considered in the paper. How are hystereses effects considered by your model? I see that there are comments on hysteresis in the Kuramoto model in appendix G2 but I think it should be more clearly stated how this relates to the hysteresis example in the motivation (I think that the relation is weak at best).

Finally, the motivation suggests that multi-layer structure is a precondition for explosive synchronization and hysteresis, which is not the case as there many examples for such effects in dynamical systems on non-multiplex graphs as well. I think this should be reformulated. 

- [Q2] From the rather opaque description of the attributed evolution-aware forward process in section 4.1 I could not follow what kind of temporal patterns this part of the model is actually able to capture. Also, I find the notation hard to parse, which does not help to appreciate the underlying ideas. Explaining the meaning of variable s = 1, ..., S early on would also help the reader understand the process. The same holds for the description of the section 4.2, which simply gives the equations without much additional explanations.

Moreover, I have some questions about the cross-layer correlation aware denoising, which builds on a time-evolving cross-layer coupling graph which captures edges between nodes in different layers. What kind of cross-layer correlations does this capture? It seems that those are necessarily based on inter-layer links between specific nodes, which may miss more subtle patterns (e.g. nodes in different communities in different layers that are preferentially connected). Could the authors clarify this? 

I would generally recommend that the authors include an intuitive description of what the diffusion process actually captures and then potentially move part of the specific mathematical formulation to the appendix?

- [Q3] Similar to my comments in Q2, from the description in section 4.2.3 I could not follow how the behavior-guidance is actually done. Since this is a major point that could point to a more fundamental misunderstanding of the authors' work, I have structured this in a separate question.

In particular, the mathematical formulation in eq. 11 and 12 seems to suggest that we actually need to simulate the Kuramoto model to obtain the order parameter, which is then included in the diffusion model. However, this also requires us to set timescales for the simulation, i.e. at what speed does the (continuous-time) model evolve compared to the speed of the evolution of the temporal graph. This is not discussed in the paper, but I believe it is an important point that must be clarified.

Also, instead of actually simulation the model wouldn't it be possible to use spectral properties to implement the behavior guidance, e.g. the eigenration of the Laplacian matrix which is known to determine the synchronization behavior of Kuramoto oscillators in graphs?

- [Q4] I think it would greatly help the motivation of this work, if the authors could clearly work out a research gap that this work addresses. To this end, the current very brief description of related works is insufficient and - in my view - partly misleading. As an example, in the remark at the end of section 4.1 the authors state that: 

"Unlike prior temporal graph generators (Campbell et al., 2024; Liu & Sariyuce, 2023; Wang et al., 2022; Gupta et al., 2022; Zeno et al., 2021), our forward process explicitly encodes temporal dependencies between the current and previous timestamps."

It sounds curious that prior works on temporal graph generators did not consider dependencies between the current and past timesteps of a graph (which is the key point in modelling temporal graphs) and indeed a quick check reveals that some of the cited works include a conditioning on the previous timestamp much in the same way as the present work. I thus believe that the statement above is too general and kindly ask the authors to clarify the contribution over these prior works.

- [Q5] I was very confused by the experimental evaluation, which mentions three data sets and then only presents results on two of them (the third one only being used for the ablation study). Why did you not include results for the third data set. 

Also, it is not clear to me how the evaluation metrics have been chosen (why do we care about the KS-statistic between centrality distributions). Finally, it is not even clear what some of the metrics mean (node behavior, Random Walk) and how they are defined. This must be clarified before this paper can be published.

Also, in the sensitivity analysis the authors check the impact of the hyperparameters \gamma_0 and sequence length, but the values used for the main results are not mentioned (and it is not made clear whether these hyperparameters were optimized). Also, the model formulation includes another parameter (noise strength) which is not mentioned in the experimental evaluation. This must be clarified.

### Soundness
1

### Presentation
2

### Contribution
2
