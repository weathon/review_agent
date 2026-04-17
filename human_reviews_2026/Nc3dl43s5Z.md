# ST-HHOL: Spatio-Temporal Hierarchical Hypergraph Online Learning for Crime Prediction

- Decision: Accept (Poster)
- Scores: 4, 8, 2, 6, 4

## Abstract
Crime prediction is a critical yet challenging task in urban spatio-temporal forecasting. 
Sparse crime records alone are insufficient to capture latent high-order patterns shaped by heterogeneous contextual factors with spatial and criminal specificity, while high non-stationarity renders conventional offline models ineffective against concept drift. 
To tackle these challenges, we propose a Spatio-Temporal Hierarchical Hypergraph Online Learning framework named ST-HHOL. First, we propose a hierarchical hypergraph convolution network that integrates crime data with heterogeneous contextual factors to uncover dual-specific crime patterns and their co-occurrence relations. Second, we introduce an iterative online learning strategy to address concept drift by employing frequent fine-tuning for short-term dynamics and periodic retraining for long-term shifts. 
Moreover, we adopt a Partially-Frozen LLM that leverages pre-trained sequence priors while adapting its attention mechanisms to crime-specific dependencies, enhancing spatio-temporal reasoning under sparse supervision.
Extensive experiments on three real-world datasets demonstrate that ST-HHOL consistently outperforms state-of-the-art methods in terms of accuracy and robustness, while also providing enhanced interpretability.  Code is available at https://github.com/777Rebecca/ST-HHOL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper, the authors designed a online learning model for crime prediction, and it has a good performance in three datasets.

### Strengths
1. A very clear introduction about the methodology. 
2. It is innovative to include online learning in this problem.
3. Numerical results show that this model is competitive. 
4. The case study is illustrative and useful.

### Weaknesses
1. Generally, there are many zeros in crime data; how do this model address this problem?
2. True zero rate is an important metric; however, it is overlooked in this paper.
3. Importance of e multi-source auxiliary data is not comprehensively discussed. Additionally, ablation study of the other two datasets should also be included since the importance of online learning should be shown.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes ST-HHOL, a spatial-temporal hierarchical hypergraph online learning framework for urban crime prediction. The framework captures dual specificity of crime patterns through hierarchical hypergraph convolutional networks, enhances spatial-temporal reasoning via partially frozen LLMs, and employs an iterative online learning strategy to address non-stationarity and concept drift, achieving state-of-the-art performance on three real-world datasets.

### Strengths
1. The introduction section accurately identifies multiple challenges in crime prediction, such as data sparsity and distribution drift. The proposed solutions to address these challenges are novel and well-motivated.

2. The "online learning" scenario presented in this work is more suitable for real-world applications, which enhances the practical value of the proposed framework.

3. The experimental evaluation is comprehensive, including multiple state-of-the-art baselines and validation of the model across multiple dimensions.

### Weaknesses
1. The proposed method exhibits considerable complexity in its design, which may lead to an excessive number of parameters. This could potentially affect the model's scalability and computational efficiency.

2. The description of how the data sparsity problem is addressed lacks clarity and detail.

### Questions
1. Could you provide a more clear and detailed explanation of how the data sparsity problem is resolved in your framework? Specifically, which components of ST-HHOL directly contribute to mitigating sparsity, and what mechanisms enable this?

2. While I understand the "online learning" scenario that the paper aims to address, I am uncertain whether the definition and usage of "online learning" in this context strictly aligns with the formal definition in machine learning literature. Could you clarify your interpretation and justify its appropriateness for this application?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors have proposed a spatio-temporal online learning model for crime prediction. It is a mix of hierarchical hypergraphs to model context-crime interactions with a partially frozen GPT-2 module for temporal reasoning and make use of a staged online training strategy to adapt to concept drift. The method is evaluated on three public city-crime datasets and shows improvements over baselines.

### Strengths
- The authors have addressed a real-world spatio-temporal problem with clear motivations in the paper.
- The proposed work combines hypergraphs and online updates in a sensible pipeline.
- Good experimental effort across multiple cities and metrics.
- The ablations and sensitivity studies are good to see.
- Interpretability figures are a plus in understanding and analyzing the work.

### Weaknesses
- The novelty in this paper is a little incremental. Hypergraph + online updates + partial LLM freezing feels like an engineering solution proposed by combing current methods.
- The claims around “hierarchical hypergraph” are mostly architectural rearrangements or engineering solution, not a fundamentally very new formulation.
- The “partial GPT-2 freezing” choice appears heuristic and could be swapped for many LLM-augmented temporal modules without changing the story.
- The concept-drift handling is also not theoretically grounded.
- There is no clear insight into why this model generalizes better, beyond “more components.”
- The ethical context is thin, considering the sensitivity of predictive policing. Have more discussions on that end.

### Questions
- What are the modeling insight that separates this from prior hypergraph-based crime methods beyond stacking modules?
- Why is GPT-2 the right backbone rather than a standard temporal transformer or Time-LLM? Is the gain coming from pretraining or just extra capacity?
- How is fairness monitored? Crime data is biased; what prevents feedback loops?

### Soundness
2

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
This paper addresses urban crime prediction by tackling two key challenges: (1) sparse crime records that fail to capture latent high-order patterns shaped by heterogeneous contextual factors, and (2) high non-stationarity that renders conventional offline models ineffective against concept drift. The authors propose ST-HHOL, a Spatio-Temporal Hierarchical Hypergraph Online Learning framework that integrates a hierarchical hypergraph convolution network with a Partially-Frozen LLM (PF-LLM) and an iterative online learning strategy. Experimental results on three real-world datasets demonstrate that ST-HHOL consistently outperforms state-of-the-art methods across various metrics while providing enhanced interpretability.

### Strengths
1. **Well-motivated problem formulation**: The paper is well-motivated to solve the sparse data problem and the concept drift challenge in crime prediction. Particularly noteworthy is the authors' approach to addressing concept drift through an innovative online learning mechanism that explicitly disentangles spatially invariant and temporally variant components. This separation allows the model to freeze spatial parameters (Θs) associated with crime patterns while adapting temporal parameters (Θd(t)) for co-occurrence relationships, providing a principled solution to handle both short-term fluctuations and long-term distributional shifts.

2. **Novel online learning mechanism**: The paper proposes a sophisticated iterative two-phase update scheme that combines frequent fine-tuning (every τ steps) for rapid adaptation to recent fluctuations with periodic retraining (every T steps) for capturing long-term gradual shifts. This mechanism effectively addresses the non-stationary nature of crime dynamics by balancing short-term responsiveness and long-term stability, which is a significant contribution to the online learning literature in spatio-temporal forecasting.

3. **Effective LLM integration strategy**: Experiments demonstrate that the Partially-Frozen LLM (PF-FFN) approach is effective in leveraging pre-trained sequence modeling priors while adapting to crime-specific dependencies. The ablation study shows that this approach achieves a better trade-off between retaining pretrained knowledge and adapting to the target domain compared to fully frozen or fully tuned alternatives.

4. **Comprehensive experimental evaluation**: The experimental evaluation is thorough and well-designed, featuring 14 diverse baselines spanning traditional methods (SVM, ARIMA), spatio-temporal forecasting models (DCRNN, STGCN, AGCRN, MTGNN, GMAN), crime-specific models (DeepCrime, ST-HSL, ST-SHN), and online learning models (DLF, FSNet, OneNet). Beyond performance comparison, the authors conduct extensive analyses including robustness studies, scalability analysis on NYC Taxi data, efficiency analysis, hyperparameter sensitivity studies, and interpretability case studies with visualizations.

### Weaknesses
1. **Limited novelty in hierarchical hypergraph design**: Though having some differences in motivation, hierarchical hypergraph neural networks and various GNN or hyperGNN designs have been fairly explored in previous works one to two years ago. The key innovation and contribution of this hierarchical hypergraph construction (heterogeneous G^e for crime patterns and homogeneous G^o for co-occurrence) is not very prominent considering the existing literature on hypergraph-based spatial-temporal modeling.

2. **Incomplete ablation study for LLM component**: The paper lacks a complete ablation experiment that entirely removes the PF-LLM component. While the authors compare different LLM variants (FPT, No Pretrain, Full Tuning, PF-A, PF-FFN), there is no comparison with a baseline that replaces the LLM with traditional neural networks (e.g., RNN, CNN, or MLP). This makes it difficult to assess the true necessity and contribution of the LLM component to the overall performance.

3. **Limited LLM baseline consideration**: The paper uses GPT-2, which has relatively low foundational performance compared to existing near-human-level language models. This raises questions about the necessity of fine-tuning such a weak model. It would be valuable to explore whether more powerful large models could achieve better results through prompt-based in-context learning rather than parameter fine-tuning, especially given the potential computational overhead of the current approach.

### Questions
1. Given that hierarchical hypergraph approaches have been extensively studied in recent years, what specific architectural innovations or theoretical contributions distinguish your hierarchical hypergraph design from existing methods beyond the particular application to crime prediction?

2. Could you provide an ablation study that completely removes the PF-LLM component and replaces it with traditional sequence modeling approaches (e.g., LSTM, GRU, or Transformer without pre-training)? This would help quantify the actual contribution of the LLM component.

3. Have you considered experimenting with more recent and powerful language models? Would a comparison with prompt-based approaches using stronger LLMs (e.g., through in-context learning) provide insights into whether the fine-tuning approach is truly necessary, or if the computational cost could be better justified?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a ST-HHOL for crime prediction to tackle the insufficiency of variety, high-order dependency, and non-stationality in sparse crime records. ST-HHOL includes (i) hypergraph modeling for integrating crime data with heterogeneous contextual factors and their co-occurrence relations, (ii) an iterative online learning for addressing concept drift by employing frequent fine-tuning for short-term dynamics and periodic retraining for long-term shifts, and (iii) a partially-frozen LLM for enhancing spatio-temporal reasoning under sparse supervision. The authors conducted extensive experiments on three real-world datasets and demonstrated that ST-HHOL consistently outperforms state-of-the-art methods in terms of accuracy and robustness, while also providing enhanced interpretability. In summary, this paper proposes an well-motivated designs for crime prediction to address the problems, conducts extensive experiments, and provides reasonable readability. However, this study would be further improved: (i) more theoretical justification on, for example, regret analysis of online learning and criteria for determining timings of fine-tuning and retraining; (ii) scalability and computational complexity of ST-HHOL; (iii) evaluation of concept drift (what types of concept drifts and how ST-HHOL adapt to them); (iv) empirical demonstration of ST-HHOL on datasets outside U.S.

### Strengths
- Well-motivated design: To address the challenges in crime prediction, this paper unifies three components: hypergraph modeling, online learning, and pretrained sequence prediction. 
- Extensive experiments: The authors conducted extensive experiments on many cities, ablation study, hyperparameter study, and visualization. 
- Readability: The paper is overall easy to follow

### Weaknesses
- Theoretical background: Most of the proposed algorithm are based on intuitive and empirical insights. In other words, the theoretical background is relatively weak. For example, regret analysis on online learning (Shalev-Shwartz 2012) and criteria for determining $\tau$ (fine-tuning) and $T$ (retraining) are missing. 
- Scalability: Hypergraph construction and periodic retraining may become expensive for large-scale city meshes. The computational complexity of ST-HHOL and its empirical results for larger datasets should be discussed. 
- Evaluation under Concept Drift: The experimental results do not provide what concept drift types are in the datasets (e.g., sudden, gradual, reccuring) (Gama et al. 2014) and how ST-HHOL adapt to such drifts. 
- Variety of Datasets: This paper conducts experiments on US cities datasets only, and the numbers of regions are up to order 100. 

[Shalev-Shwartz 2012] Shai Shalev-Shwartz: Online learning and online convex optimization. Foundations and Trends® in Machine Learning, 4(2), pp.107-194, 2012. 

[Gama 2014] Joao Gama, Indre Zliobaite, Albert Bifet, Mykola Pechenizkiy, and Abdelhamid Bouchachia: A survey on concept drift adaptation. ACM Computing Surveys, 64(4), pp.1-37, 2014.

### Questions
Please answer the points described in the Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
