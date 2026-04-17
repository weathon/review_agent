# CausalAffect: Causal Discovery for Facial Affective Understanding

- Decision: Reject
- Scores: 4, 2, 8, 6

## Abstract
Understanding human affect from facial behavior requires not only accurate recognition but also structured reasoning over the latent dependencies that drive muscle activations and their expressive outcomes. 
Although Action Units (AUs) have long served as the foundation of affective computing, existing approaches rarely address how to infer psychologically plausible causal relations between AUs and expressions directly from data. 
We propose CausalAffect, the first framework for causal graph discovery in facial affect analysis. 
CausalAffect models AU$\rightarrow$AU and AU$\rightarrow$Expression dependencies through a two-level polarity and direction aware causal hierarchy that integrates population-level regularities with sample-adaptive structures. 
A feature-level counterfactual intervention mechanism further enforces true causal effects while suppressing spurious correlations. 
Crucially, our approach requires neither jointly annotated datasets nor handcrafted causal priors, yet it recovers causal structures consistent with established psychological theories while revealing novel inhibitory and previously uncharacterized dependencies. 
Extensive experiments across six benchmarks demonstrate that CausalAffect advances the state of the art in both AU detection and expression recognition, establishing a principled connection between causal discovery and interpretable facial behavior.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes CausalAffect - a framework for causal graph discovery for facial affect analysis. Given facial images, this framework recovers a causal graph graph describing dependencies between facial activation units and facial expressions (i.e., AU -> AU and AU -> Expression). The authors introduce several specific mechanisms, including sample-adaptive learning and feature-level counterfactual interventions, designed to improve graph discovery. The authors validate the framework empirically on six benchmark datasets, illustrating that it improves the accuracy of AU and expression predictions, and recovers causal graphs that are consistent with existing psychological theories.

### Strengths
In this work, the authors propose the first (to my knowledge) approach for causal discovery in facial affect data. One key benefit of the proposed approach is that it can learn in a semi-supervised manner, and does not require joint annotations of AUs and facial expressions. This enables the proposed approach to ingest data from disjoint datasets during learning. 

The empirical validation of the work also has several string dimensions: the authors compare against a broad set of relevant baselines, illustrate accuracy improvements provided by the approach, and report interesting interpretive analysis of the scientific insights recovered by the framework (Figure 2; Appendix C). The authors also provide an interesting comparative analysis of the Social Smile and Duchenne Smile, and provide also perform hyperparameter sensitivity and ablation studies. While minor, CausalAffect is also a great name for the framework.

### Weaknesses
## Clarity of Causal Framework and Formal Assumptions

One of my key concerns with this work is that it provides an incomplete motivation for why a causal perspective is important to this problem, and why conceptually a causal approach improves over the status-quo in AU/Expression recognition. 

After reading the work several times, it remains unclear what cognitive or behavioral model this work instantiates. AU -> Expression, it could also be the case that Expression  -> AU, or that there is a third unmeasured variable that is manifesting in both AUs and emotion expressions. When investigating such theories, I would expect to see a formal Structural Equation Model or DAG with accompanying references supporting the model. The term "relations" used throughout the work maintains this ambiguity and does not clearly identify the proposed causal pathways that are under study. 

- For a causal discovery problem, I would expect a formal statement of causal and statistical assumptions needed to support inference, matched to the DAG or SEM above. 
- The soft DAG constraint enabling cycles in the AU → AU pathway violates the acyclicity requirement of causal DAGs. This makes me question the causal model under study. I think this could nicely be viewed as a time series problem where there is a fixed DAG with states that vary as a function of time t. This could obviate the cyclic issues while also supporting the dynamics the authors mention. Generally adopting a time-dependent formulation would be natural in this setting. 

In sum, there is a key gap between the paper's causal framing and its methodology. While the paper claims to "infer psychologically plausible causal relations" and "enforce genuinely causal relations", the absence of a formal causal model, stated identification assumptions, and the violation of acyclicity principles means that the learned relationships cannot be interpreted causally. The method may learn useful structured dependencies, but these are correlational rather than causal. I recommend the authors either: (1) formalize their causal framework with appropriate assumptions and modify the architecture to respect causal principles, or (2) reposition the work as learning interpretable feature relationships via a semi-supervised approach, which would still be a quite valuable contribution.


## Presentation of Technical Framework
Figure 1 is very helpful for understanding the framework and the paper is well-written overall. However, in places throughout Section 3, I found myself lost in the details of the approach with limited rationale for why these details matter. In particular, subsections 3.1-3.5 currently read as a recipe of technical details rather than providing a unified rationale for why these components are needed. For example, it's unclear why Counterfactual Interventions are necessary given HSIC-Based Disentanglement, or why sample-adaptive graphs are necessary to support heterogeneous nodes. The empirical ablation study illustrates the empirical impact of these decisions but offers a limited conceptual basis for why each component is necessary.  The authors could address this by easing the notation / moving some details to the appendix to make space for more rationale for the design decisions and sharing why these problems are technically challenging. 

Related to the comment above, one place where I was especially missing the rationale is Section 3.3. Why would Social and Duchenne smiles require *different graph structures* rather than *different paths* through the same graph? If we conceptualize each AU as a random variable, the causal graph structure should remain invariant across realizations—only the values of these variables should change. A graph that changes structure conditional on observed data is at odds with standard causal modeling. Note that addressing the first point (Clarity of Causal Framework) could also resolve this concern. 

## Empirical Validation

While the empirical validation has several strengths (noted above) there are also a few weaknesses. Foremost, the authors do not report statistical uncertainty in Table 1 and elsewhere in the results. This makes it challenging to assess whether the claims hold across settings. 

Further, based on the current results, it is unclear what the foundational goals of the empirical evaluation should be. I was surprised to see prediction accuracy featured so prominently in a causal discovery work. For causal inference, the primary validation should be whether the method recovers the true causal structure, not whether it improves predictive performance (which can be achieved by learning spurious correlations). To validate the approach, the authors may want to design a synthetic or semi-synthetic experiment illustrating that the approach can recover the ground truth when the true causal structure is known. This would provide valuable evidence substantiating that the learned causal relations (Fig 2) are valid. 

Finally, my conceptual confusion surrounding the causal framework extends to the  experiments. How would it be possible to recover a causal graph with edges that  vary conditional on a single image (Fig. 3)? This reinforces the concern that the  learned relationships may be correlational rather than causal.

More positively, a compelling visualization could show graph activation over AUs as a function of time—e.g., showing frames from a video sequence. This would illustrate the temporal activation patterns underpinning the graph structure and would align naturally with the time-series formulation suggested earlier.

### Questions
Conceptual:
- What is the proposed causal model underlying this approach? Specifically, does it posit that AU → Expression, Expression → AU, or both directions exist? Could you provide a formal DAG or SEM that represents your theoretical model? How does temporality connect to this formulation? 
- What causal and statistical assumptions are necessary for your method to identify causal relationships rather than correlations? For instance, do you assume causal sufficiency (no unmeasured confounders between AUs and expressions)?
- Could you clarify the theoretical justification for having graph structures that vary across individual images? 

Empirical:
- To what extent can the accuracy benefits reported in Table 1 be attributed to increased dataset size available to CausalAffect?
- How do the reported empirical results vary across training runs and random seeds?

### Soundness
3

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
5

### Summary
The paper proposes a framework for facial Action Unit detection that is based on a global feature extraction followed by a per-AU classification head and two graph neural networks that can do message passing to refine the detections. Through direct supervision, the graphs are learned following both data correlations and psychologically-based data. The whole network i.e. the backbone, heads and the adjacency matrices for the graphs is learned in an end-to-end fashion. The method is tested in standard AU benchmarks delivering competitive results.

### Strengths
The idea of trying to find a proper AU relationship as well as a relationship between AUs and expressions is appealing, despite not being new, and the authors try to approach it in a data-driven way. 

The results are compelling and the relation between AUs and expressions across datasets is investigated, showing similar correlations than that of existing work.

### Weaknesses
The paper is poorly written, and poorly presented, with many broken sentences. The narrative is very loose and the figures and notation do not serve the understanding of the paper. The paper is full of clutter and the tables and figures have been minimized to fit in the paper to an unacceptable level.

The method is not novel and combines many pieces of existing work. The discovery of knowledge-based AU graphs is not new, it has been presented in many works; as an example there are the following approaches not cited in this paper: 

Song et al. Dynamic Probabilistic Graph Convolution for Facial Action Unit Intensity Estimation. CVPR 2021
Wang et al. Spatial-Temporal Graph-Based AU Relationship Learning for Facial Action Unit Detection. CVPRW 2023
Fan et al. Facial Action Unit Intensity Estimation via Semantic Correspondence Learning with Dynamic Graph Convolution. AAAI 2020

What is exactly the contribution of the paper? The paper does not include any discussion wrt to prior work and how the proposed method advances existing research. 

There is no analysis of complexity and training and inference time. This needs to be included.

The use of external data to justify the results and its use to compare against state of the art methods is unfair. For a fair comparison the competing methods should have been trained in the same data. It is not good practice to claim state of the art results when the training includes a large amount of external data than that used by the competitors. Even when using additional data, the results are surprisingly close to state of the art, meaning that the method barely advances existing research. 

In summary, the manuscript is rather poor and needs a lot of work for it to be considered. The reading is unpleasant and the contributions are not properly justified. The results are far-fetched thanks to the addition of external data, and there is no proper comparison in the methodology against prior work on graph neural networks for Action Unit detection/intensity estimation.

### Questions
Please see my comments above. In particular, I would suggest the authors to properly illustrate in which ways their method is novel wrt prior work, and what are the main contributions they propose, in a succinct, to-the-point manner.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces a framework, named CausalAffect, for causal graph discovery in facial affective analysis. It aims to overcome the limitations of existing methods that lack psychological plausibility, rely on joint annotations, and ignore causality direction and inhibitory effects. Its main contribution is a weakly-supervised framework that learns a two-level (global and sample-adaptive) causal hierarchy for both AU→AU and AU→Expression dependencies, capable of modeling both excitatory and inhibitory relations. A key innovation is a feature-level counterfactual intervention mechanism that enforces true causal effects by perturbing latent AU features, eliminating the need for image synthesis.

### Strengths
The primary strength lies in its novel formulation of facial affective analysis as a causal discovery problem, moving beyond mere correlation to seek psychologically plausible mechanisms. The proposed framework, CausalAffect, integrates a two-level (global and sample-adaptive) graph structure to capture both stable population-level rules and context-specific dynamics. Its ability to model both excitatory and inhibitory relations, combined with an efficient feature-level counterfactual intervention, ensures the discovered dependencies are genuinely causal. It also eliminates the need for scarce jointly-annotated datasets by its weakly-supervised design. Well-designed and comprehensive ablation studies that confirm the necessity of each component.

### Weaknesses
-  The system complexity would be a concern to me. CausalAffect contains four different modules and that caused a large number of loss functions. Their corresponding hyperparameters (e.g., $\lambda_{ib}$, $\lambda_{DAG}$, $\lambda_{consist}$) also **increase the risk of training instability** and these factors would ** make it difficult for other researchers to reproduce the results**.

- The paper employs a significant number of mathematical symbols and formulas, which is commendable. However, to some extent, this comes at the cost of reading fluency and also compresses the available space for text, causing some of the results analysis to be relegated to the appendix.

### Questions
**Question 1: ** Validating psychological plausibility against existing literature like FACS is a clever approach. However, for the 'new' causal relations discovered by the model, for example: the subtle inhibitory ones, it seems that we don't have an objective 'gold standard' or ground truth. This makes it difficult to determine whether these are genuine psychological insights or merely statistical artifacts learned from the specific datasets. What's your thoughts on this?

**Question 2: ** In Table 3, the result for GFT at idx 13 should be bolded, assuming that bolding is used to indicate the best result.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces CausalAffect, a novel framework designed to learn causal relations among Action Units (AUs) and between expressions and AUs. The work features extensive experimentation and analysis, providing interesting insights into these relationships.

### Strengths
* The trained models and code will be released, which is a significant contribution to the research community.
* Extensive experiments, including ablation studies, are conducted on multiple datasets. The paper compares the method against competitive baselines and demonstrates promising performance.
* Overall, the paper is well-written and presented in good format.
* Interesting analysis and visualizations are presented in Section 4.2. These provide readers with a straightforward understanding of how the learned causal relations align with or differentiate from previous studies. (Although some results appear unusual, which is addressed in the Questions section.)
* Straightforward case studies are provided to illustrate the effectiveness of the proposed method.

### Weaknesses
* Figure 2 is cited in line 46 but is located on page 8, which is too far from the relevant text and disrupts the flow of reading.
* The paper lacks a Related Work section.
* While Table 1 shows promising performance, the comparison feels unfair because the best CausalAffect configuration utilizes additional training data sources. Without this extra data, the performance is very close to the best baseline, and a significant statistical test is missing to confirm the difference.

### Questions
* Regarding the EmotioNet experiment in Table 1, could you please explain why joint training (with AU datasets) appears to detrimentally affect the expression recognition performance?
* In Figure 2, the GNN-Learned Correlation visualization looks weird to me. It seems to imply that every AU is highly correlated with every expression. Could you elaborate on this specific finding? Similarly, for the AU-AU co-occurrence in BP4D, the strong correlation between AU6 and AU14 warrants further explanation.

### Soundness
3

### Presentation
3

### Contribution
3
