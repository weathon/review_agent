# Adaptive Graph Denoising with Harmonic Grouping

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 2, 6

## Abstract
Diffusion models have become the leading paradigm for generative modeling, yet their effectiveness in graph domains remains constrained by the instability of topological structures under noise. Existing noise schedulers, however, are typically fixed or heuristic, failing to adapt to either diversity across graph datasets or variability within a dataset. We formulate the challenge of graph diffusion scheduling from two perspectives: inter-diversity, referring to differences across graphs, and intra-variability, referring to variations within individual graphs. To address the challenge in an unified manner, we propose ADHaG (Adaptive graph Denoising with Harmonic Grouping), a novel scheduling framework that explicitly addresses these limitations. Each graph is assigned to a group defined by Laplacian, within which we optimize a feature-conditioned schedule that parameterizes the base scheduler. Empirically, ADHaG demonstrates consistent improvements across both molecular and synthetic benchmarks. In particular, on the QM9 dataset, it attains dataset-level performance with respect to chemical validity; moreover, across all synthetic graph datasets, the orbit statistics consistently improve. These results demonstrate that adaptive scheduling can provide a principled approach to capture both inter- and intra-level variation of graphs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a new data-dependent and adaptive noise scheduling scheme for graph diffusion models, such that groups of graphs with similar structural characteristics and properties have their own noise schedules that are learned with a neural network. The grouping is done using the normalised Graph Laplacian eigenvalue spectrum, grouping similar spectrum graphs together, into G distinct groups. Then, an adaptive noise scheduler is designed that takes as input the group g as well as a vector z that describes some features of the particular graph, and can be obtained through a GNN encoder. In practice, the adaptive schedule is implemented as a modulation of a standard cosine schedule. The loss function is also modified to take into account changes in the noise level value at a particular diffusion step $t$, such that changes in the noise level also amplify the loss weighting. The experimental validation consists of synthetic graphs and molecular graphs with the 10k Polymer dataset and QM9. The method achieves improvements over various baselines.

### Strengths
- The method seems to achieve valid improvements over many of the baselines in many of the metrics. 
- The idea of adjusting the noise schedule for different types of graphs or graph datasets makes sense, and as such the method is quite well motivated
- The way the adaptive noise scheduler itself is implemented makes sense and seems like an approach that can bring benefits. I think that the research direction and the intuition about the problem statement makes sense.

### Weaknesses
The paper also does have some weaknesses:
- Overall, the paper was quite difficult to follow, and some details of the method are not very explicitly explained, and are potentially inconsistent with different parts of the paper. See the questions for more. 
- The usage of the z vector is particularly confusing to me. As I understand it, it is a GNN embedding of the graph. So we condition the schedule on individual graphs at training time. How do we get the $z$ at inference time? Further, how do we choose $g$ at inference time?
- Further, it seems that the $g$ and $z$ conditioning are also given as input to the denoiser itself. This raises the question: Do we need them as input to the noise scheduling function $\gamma$? It is not really at all obvious why would they provide added benefit over just conditioning in the denoiser, raising the need for (at least) the following ablations to make sure that both $g$ and $z$ are useful even for the base denoiser, as well as for the addition of the $\gamma$: 
1) $\gamma=1$ and $g$ conditioning for the denoiser but no $z$ conditioning
2) $\gamma=1$ and $z$ conditioning for the denoiser but no $g$ conditioning
3) $\gamma=1$, and both $z$ and $g$ conditioning for the denoiser
4) $\gamma$ conditioned on $g$, and both $z$ and $g$ conditioning for the denoiser
5) $\gamma$ conditioned on $z$, and both $z$ and $g$ conditioning for the denoiser
6) $\gamma$ conditioned on $g$ and $z$, and both $z$ and $g$ conditioning for the denoiser
- There is quite a bit of added complexity, and for the metrics in which the metrics outperforms the baselines, it is not obvious why this would be the case. Ideally, the paper would have a hypothesis for what kinds of issues does the graph diversity problem cause in the baseline models, and analyze quantitatively how the proposed method overcomes this issue and beats the baselines. Otherwise, looking at it from the outside, it is seems probable that the improvements on some measured variables could be due to small details like architecture design, hyperparameter tuning, including the $z$ conditioning to the denoiser, etc. This makes it difficult to have any confident takeaways from the paper.

### Questions
- It was a bit unclear to me while reading the paper how does the graph partitioning actually work. Do I understand correctly that we 1) calculate the normalized graph Laplacian eigenvalues for each graph separately, chunk the range of possible eigenvalues into G groups, and classify a graph in a particular cluster based on which group does it have the most eigenvalues? I am not sure if this was clearly stated, but Figure 2 seems to imply something like this. 
- What are we seeing in Figure 3? All the eigenvalues of all the graphs in a dataset? How come we have negative values and values over 2 in the plot, given that the normalized graph Laplacian eigenvalues are in the range [0,2] (to the best of my understanding). 
- What exactly is the role of the z vector? Why is it needed as an input to the schedule? 
- "for graphs, the diffusion process typically operates on the spectral domain. " -> what does this mean, since the paper seems to actually use the Digress formulation of graph generation? 
- Do the authors have an intuition for why do the schedules for the different groups end up like they do in Figure 5? The curves seem quite random to me, but perhaps there is some structure that is interesting? 

Due the reasons raised in the weaknesses, I am starting with a reject. I think that the paper would benefit from clearer writing and explanation of the method, more focused experiments showing the benefits, and potentially streamlining the method to the minimum added complexity that is necessary to achieve the potential improvements from the adaptive schedule.

### Soundness
2

### Presentation
1

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
This paper proposes ADHaG, an adaptive scheduling framework for graph diffusion models. The key idea is to introduce harmonic grouping, which partitions graphs based on their Laplacian spectra, and to train feature-conditioned schedulers for each group. The approach aims to handle inter-domain diversity (differences across datasets) and intra-domain variability (differences within datasets). Experimental results on several benchmarks (QM9, 10K Polymers, SBM, Planar) show modest improvements over classical baseline diffusion models.

### Strengths
- Introducing adaptivity at the scheduler level rather than model architecture is a fresh angle.

- The proposed framework is architecture-agnostic and can, in principle, be integrated into existing diffusion pipelines with minimal implementation overhead.

- Leveraging the Laplacian spectrum to capture both global and local structural variations is in accordance with established principles in graph signal processing.

### Weaknesses
1. The paper repeatedly claims to provide a principled and spectrum-aware approach, yet offers no formal justification for why Laplacian-based grouping should lead to improved noise scheduling. The relationship between spectral bins and generative performance is only empirical, not theoretically grounded. No ablation or analytical study explains why spectral similarity should correlate with optimal scheduler parameters.

2. The proposed harmonic grouping trains an independent scheduler for each spectral group. Thus, model size and parameter count increase linearly with the number of groups.  
However, the experimental results show no clear correlation between more groups (e.g., G=7) and better performance. The paper fails to quantify this trade-off in terms of memory, runtime, or parameter efficiency.

3. The most recent diffusion-based molecular graph generation model included is DiGress (2023). More recent and relevant baselines should be included.

4. The molecular experiments are restricted to QM9 and 10K Polymers, omitting other widely adopted datasets such as ZINC and MOSES, which are standard in graph generative model evaluation. This limitation weakens the paper’s generality claims for molecular domains.

5. For QM9, the paper only reports Validity and Uniqueness metrics, which are insufficient to evaluate generation quality. Established benchmarks also include Novelty, FCD, NSPDK, and KL-divergence against real distributions.  
The current reporting makes the comparison against DiGress, GDSS, and GSDM incomplete and potentially misleading. The authors are strongly encouraged to adopt the same experimental protocols and metric sets as these benchmark models to ensure fairness and comparability.

6. The paper does not mention a code release, nor does it provide sufficient implementation details to ensure reproducibility.

### Questions
1. Why have the most widely used benchmarks for molecular graph generation, ZINC and MOSES, been omitted, despite being used by the baselines (e.g., DiGress and GDSS)?

2. The reported results on generic graphs and QM9 differ significantly from the original results in benchmark papers (e.g., GDSS, DiGress) and recent models like GruM (2024). Could the authors clarify whether these differences arise from variations in data preprocessing, evaluation metrics, or training setups?

3. Can the authors provide any analytical or empirical evidence linking eigenvalue distribution to optimal noise scheduling? For instance, does spectral bandwidth correlate with denoising difficulty?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a trained adaptive noise scheduling method for denoising diffusion models for graphs (specifically the GDSM variety, i.e. diffusing only the eigenvalues and using training set eigenvectors).

For this, the top k eigenvalues are used to group the graphs into groups via “harmonic similarity” (histogram sorting and binning) and each  is assigned such a group during training and denoising.

the learned schedule is trained as a learning an offset to the scaling exponent of a cosine schedule to ensure it remains within reasonable bands. evaluation is performed on SBM,planar, communitysmall and QM9 datasets , as well as 10k polymers.

### Strengths
- originality: using spectral properties to automatically fine tune  the noise schedule is a decently non-obvious idea
    
- significance: the method  seems to *sometimes* yield strong improvements
    
- quality: a wide set of baselines and datasets is evaluated, the method is well described
    
- clarity: the paper is overall clearly written

### Weaknesses
- legibility improvement:
    
    - harmonic similarity is only defined by jointly reading the figure + prose, make clearer how the quantiles are constructed
        
    - F_phi should be defined in the main body or an explicit reference to appendix should be made
        
- lines 304 to 312 appear to require further editing, the  “that is… In contrast” construction does not parse to me  (the two paragraphs say the same thing, but are written as contradiction)
    
- I don’t think from the data presented it can be claimed the method consistently leads to improvements, needs to be evaluated across multiple training seeds and made rigorous e.g. as such:
    
    - train multiple models
        
    - check if the framework consistently improves metrics per batch of seeds on a model ( check for statistically significance/compute CI)
        
    - compute check p(intervention has significant difference) >0.5 => then can make the claim
        
- Figure 4 requires a baseline comparison without the harmonic grouping (maybe the methods are aleady good at modeling the EV distribution?)

### Questions
- the main issue is the statistical significance test from the weaknesses and more careful baselining, showing the actual improvement of the method
- other Question: am I correct that the scheduler is also used to generate noised samples (meaning that the t steps are now variably sized across training, and optimized to minimize loss/finding the easiest path possible?)
    
- do you  force zero SNR/full noising at noise limit? see [https://openaccess.thecvf.com/content/WACV2024/papers/Lin_Common_Diffusion_Noise_Schedules_and_Sample_Steps_Are_Flawed_WACV_2024_paper.pdf](https://openaccess.thecvf.com/content/WACV2024/papers/Lin_Common_Diffusion_Noise_Schedules_and_Sample_Steps_Are_Flawed_WACV_2024_paper.pdf)

### Soundness
2

### Presentation
2

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
The authors proposes a novel adaptive noise scheduler for graph generative models based on diffusion. This is certainly one important factor and design aspect that could help improve graph generation performance. The authors propose to group graphs according to spectral properties, and then learns a feature-conditioned scheduler for each group. The authors show performance that are competitive and slightly better on some perspective, with respect to recent graph generative models on QM9 and synthetic graph datasets.

### Strengths
Improving the noise scheduler is certainly an interesting direction for developing better graph generation models. The idea of noise scheduler that depends on graph properties is natural but a good idea. And developing different schedulers for each group of graphs is reasonable and novel. The performance of the proposed method on different datasets seems to confirm the benefits of the proposed graph scheduler.

### Weaknesses
Grouping graphs is certainly a good idea for designing a finite set of noise schedulers. It is however not fully clear that inter-diversity and intra-variability of graphs have to be considered on the same level: it seems that large graph variations across data/applications is handled separately at the end, even if this is not fully clear in the first part of the paper. Also, one might imagine that large variations across datasets can be handled differently than training a fully universal adaptive scheduler (e.g., by considering different generative models for graph datasets that are completely different).

The choice of grouping graphs by their spectral properties looks natural. However, it maybe lacks a strong convincing argument that this is the best way to adapt noise schedulers. Wouldn't it depend on the actual diffusion model? It seems to be mostly applied in combination with DiGress: even there, what is the exact connection between noise scheduling and spectral properties of graphs? (Btw, DiGress does not look like a spectral diffusion model, despite what is written in the present paper).

The conditioning block 'z' is described very shortly in the main paper, it is difficult to really appreciate its construction and benefits. 

The design of the adaptive noise scheduler makes sense. However, the proposed idea can probably be applied to other parametric forms than (3) - which is probably not a unique solution. It may be good to motive this particular form, or discuss alternative forms, or at least clarify that the solution is not unique, and maybe not optimal (there is no claim of optimality in the paper ofc, but it may still be good to clarify things).

Experiments are usually well chosen, and illustrate the benefits proposed by the adaptive noise scheduler. However, QM9 is a very small dataset, and state-of-the-art generative models like those based on flow matching, are missing. This is not a critical issue, as again the authors do not claim any optimal generation performance. Yet, the text can be clarified in that respect, and the choice of experiments can be motivated accordingly. 

From Table 5, it seems that most gains come from validity improvement. Whose value stays relatively low. It may be good to explain why benefits are essentially observed in terms of validity, and what further bottlenecks could lead to even better validity scores. 

Computational complexity is not discussed. In general, the diffusion process should not be more complex than competitors. Yet, two new components may deserve a short discussion: the construction of $\gamma_g$ and the spectral grouping (which apparently relies in eigendecomposition of graphs).

### Questions
see above

### Soundness
3

### Presentation
3

### Contribution
2
