# Multi-Integration of Labels across Categories for Component Identification (MILCCI)

- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
Many fields collect large-scale temporal data through repeated measurements (`trials’), where each trial is labeled with a set of metadata variables spanning several categories. For example, a trial in a neuroscience study may be linked to a value from category (a): task difficulty, and category (b): animal choice. A critical challenge in time-series analysis is to understand how these labels are encoded within the multi-trial observations, and disentangle the distinct effect of each label entry across categories. Here, we present MILCCI, a novel data-driven method that i) identifies the interpretable components underlying the data, ii) captures cross-trial variability, and iii) integrates label information to understand each category's representation within the data. MILCCI extends a sparse per-trial decomposition that leverages label similarities within each category to enable subtle, label-driven cross-trial adjustments in component compositions and to distinguish the contribution of each category. MILCCI also learns each component’s corresponding temporal trace, which evolves over time within each trial and varies flexibly across trials. We demonstrate MILCCI’s performance through both synthetic and real-world examples, including voting patterns, online page view trends, and neuronal recordings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies a matrix factorization problem specialized to tensor data commonly generated in many fields, including neuroscience. In the neuroscience application, the generative model for one trial has a loading matrix that is formed from a tensor of shape cells x components x category options (layers) x categories. (e.g., If category is 'task', layers can be easy, medium, hard. If it is 'choice', layers can be correct, incorrect.) The loading matrix for a given trial (e.g., with metadata task=easy, choice=correct) is formed by concatenating the layer matrices corresponding to that trial (e.g., layer 'easy' under task, and layer 'correct' under choice) The aim of the matrix factorization is to infer this loading matrix and the temporal trace matrix that it multiplies.

Matrix decomposition is performed by iteratively solving for the loading and temporal trace matrices. (in a loop: solve for the loading matrix when the trace matrix is constant, solve for the trace matrix when the loading matrix is constant) This iterative method uses multiple heuristic objective function components:
- data fidelity (L2)
- sparsity (L1; has a hyperparameter)
- temporal smoothness (L2 between consecutive time points of the trace matrix, has a hyperparameter)
- de-correlation of temporal trace components, has a hyperparameter)
- consistency between the cell x component matrices corresponding to different options of given category (L2, has a hypermarameter. This has one more hyperparameter for categories with ordinal labels, the st. dev. of the Gaussian kernel.)

The method is applied to a synthetic dataset, a large-scale neuroscience dataset, a US states-level voting dataset, and a wikipedia pageview counts dataset. On the synthetic dataset, the method is quantitatively compared against baseline methods.

### Strengths
- Analyzing data in relation to underlying complex metadata structure is a prominent problem.
- This paper takes advantage of prior knowledge on the structure of metadata, unlike other more generalist approaches. I think this is the paper's most prominent contribution.
- This paper offers a linear decomposition which improves interpretability.
- The paper demonstrates its method across three real-world datasets.

### Weaknesses
- Similar methods exist to analyze neuroscience data. In particular, not even citing the Pellegrino et al, "Dimensionality reduction beyond neural subspaces with slice tensor component analysis," Nature Neuroscience, 2024 paper is a big miss. I believe this paper should be added as a baseline, too.

- The method has multiple hyperparameters. Neither their sensitivity is properly discussed nor is it clear whether similar effort was spent to tune the competing models.

- The approach does not have a dynamics model. The only temporal processing comes from a penalty component to ensure temporal smoothness of the trace matrices. Therefore, it is not clear why the focus is on temporal data. Conversely, temporal smoothness could be added in a similar manner to some of the other tensor decomposition methods studied here.

- With such large datasets, probabilistic models (e.g., variational) that are optimized with gradient descent have shown remarkable performance across many applications. Here, the focus has been on matrix decomposition, which is readily interpretable. However, I consider a lack of any emphasis on noise modeling or analysis as a weakness.

- The consistency heuristic that enforces the cell x component matrices corresponding to different options of a given category is not clear to me.

- I find the wikipedia study anecdotal. There is a full page of text that describes seemingly cherry-picked results. Out of how many pages and components are the displayed results shown/discussed is not clear.

### Questions
- The SliceTCA method of Pellegrino seems relevant. Could the authors compare against that?

- Could you please add details on the tuning of hyperparameters; the procedure and sensitivity of the results to the hyperparameters. Was hyperparameter tuning performed for the baselines?

- Could you please explain the intuition behind the consistency heuristic? What is the effective assumption and does it apply to all datasets?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses the challenge of disentangling the effects of multi-category labels in high-dimensional temporal multi-trial data by proposing MILCCI, a data-driven method for component identification. MILCCI extends sparse per-trial decomposition to: (1) identify interpretable underlying components of the data, (2) capture cross-trial variability via label similarity within each category, and (3) integrate label information to distinguish the contribution of each category. It models components using category-specific sparse tensors, allowing subtle label-driven adjustments in component compositions while maintaining consistency across trials. Additionally, MILCCI learns flexible temporal traces for each component that evolve within and across trials.

### Strengths
1.MILCCI breaks new ground by modeling components as category-specific tensors, allowing subtle adjustments to label changes while avoiding the rigidity of fixed-component methods. This design uniquely enables disentangling multi-category label effects, a capability lacking in SiBBlInGS and tensor factorization.

2.The method is rigorously tested across synthetic and real-world data with varying characteristics. It outperforms baselines in synthetic component recovery and produces interpretable results in real domains.

3.The paper provides comprehensive details, including model equation, initialization steps, label similarity calculations, and code/environment specifications. This transparency ensures other researchers can replicate experiments and adapt MILCCI to new datasets.

4.MILCCI addresses unmet needs in multiple fields: it identifies decision-critical neural ensembles for neuroscience, uncovers actionable voting trends for political science, and reveals user/spider behavior differences for web analytics—demonstrating broad real-world utility.

### Weaknesses
1.The iterative training process may become computationally expensive for large datasets with many categories or trials (e.g., >10,000 trials). The paper does not report runtime comparisons with baselines or discuss optimizations (e.g., mini-batch training) for scaling.

2.While MILCCI uses sparsity (γ₂) and label consistency (γ₁) hyperparameters, there is no systematic analysis of how these parameters affect performance. For example, how does varying γ₁ impact component consistency across labels? This makes it difficult for users to tune MILCCI for new datasets.

3.The method assumes linear relationships between components and data, which may limit performance on non-linear systems. The paper acknowledges this but does not explore even simple extensions or discuss scenarios where linearity might fail.

4. For the Wikipedia dataset, PARAFAC and Tucker failed to converge, but the paper does not explore alternative implementations or explain why convergence failed. Additionally, SiBBlInGS comparisons are qualitative only—quantitative metrics would strengthen the case for MILCCI’s superiority.

### Questions
1.For large datasets (e.g., 100,000 trials or 10+ label categories), how does MILCCI’s runtime compare to baselines like SiBBlInGS or SVD? Have you explored optimizations such as mini-batch training for component updates or parallelization of trace calculations?

2.The paper uses γ₁ (label consistency) and γ₂ (sparsity) hyperparameters, but no sensitivity analysis is provided. Can you show how varying these parameters affects component interpretability and downstream performance on synthetic or real data?

3.MILCCI assumes linear decomposition—have you tested scenarios where non-linear relationships exist? Would a kernelized version of MILCCI improve performance, and how would you adapt the label consistency constraint for non-linear spaces?

4. For the Wikipedia dataset, PARAFAC and Tucker failed to converge with MILCCI’s component dimension. Did you test lower dimensions or alternative initialization strategies to enable convergence? If not, why is MILCCI more robust to high-dimensional, noisy data?

### Soundness
3

### Presentation
2

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
In this work, the Authors propose a model to decompose the variability in time series to attribute it to multidimensional labels of the data. They test the model on a synthetic dataset to show that it faitfully reconstructs the way the data was generated, to the US elections data to show that the model finds interpretable patterns, and to IBL neural data to assign roles to clusters of individual units in the brain.

### Strengths
- Strainghtforward model

The model proposed here is straightforward, attributing components of the time series segment to labels in a multi-hot label vector. The optimization is done in a kind of E-M algorithm, an alernation between optimizing the time series "templates" that correcpond to different labels and the "projection matrices" that determine the membersips of the templates in the labels.

- Comparison to baselines

The work compares the proposed algorithm to multiple beaselines including classical baselines (e.g. PARAFAC) and more recent baselines.

-- Principled buildup:

The work builds its way through a progression of datasets starting from the ones offering higher control of the input data and making way to real-world datasetss.
The work starts from **synthetic data**, enabling the Authors to show that the model faithfully reproduces the components used, manually, in the genaration of the synthetic data.
Second the work moves on to the case of **inherently interpretable data** (with the example of the US election data) which has been thoroughly analyzed semantically, allowing to show that the model finds known pattern in this simpler real-world data.
Finally, the work moves on to **novel data yet to be interpreted** (the IBL's biological data) where the model can make novel predictions abouth the biology of the brain.

- Finally, the model scientifically corroborates that Tucker is not the most reliable source when it comes to the interpretation of the US elections.

### Weaknesses
- No alternatives for the design choice

While the model puts forward a quite reasonable and quite principled architecture, the alternative design choices are not explored in the text. For the E-M-like algorithm that alternates between optimizing A and Phi, is lasso for MSE the best algorithm choice or could other alternatives (e.g. MSE -> entropy) would be better?

- No range description for the US election interpretability

While the readouts of the model on the US electios dataset are linked to known facts about the political events that took place in certain states in certain years, the significance of the effect is unclear from the proposed analysis. Notoriously, spurious correlations are often picked up on in election data, leading to widespread conspiracy theories. What would be a good baseline and the analysis to show that the model's interpretability capacities are significant?

- No thoroughly developed impact area

While the analysis of the IBL data is interesting and promising, this part of the work -- that could be its culmination and goal -- seems underdeveloped. While it is shown that certatin cells respond to certain labels, it is unclear form the text 1) which parts of it are already known and 2) what do we learn from this finding. For example, it would be interesting to analyze what brain regions these units belong to and to derive new knowledge about the neural circuitry related to decision-making in the brain. Overall, the lack of a defined goal / application domain and a corresponding result leaves me with the impression that the work is a bit unfinished

- Text is a bit heavy

The text seems to be a bit heavy, oftentimes obscuring othwerwise strainghtforward concepts. At the same time, some straightforward good propoerties of the model get lost in the wording. For example, I would highlight that limiting the number of components substantiates the need of separating Phi and A: the A tensor cannot be defined as a unit projection matrix from all labels that apply to their corresponsing templates in the time series domain because, under a limited number of components, it is forced to learn synergies between the labels (e.g. grouping states in the US election data.)

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes MILCCI, a flexible model that exposes interpretable sparse components underlying multiway data and reveals how they capture diverse label categories. This paper performs linear decomposition for each trial with respect to tuple containing the set of each category’s value in that trial.

### Strengths
- The setting is practical although challenging because we do not have ground-truth labels to justify the performance rigorously.

### Weaknesses
- The writing of the paper is not good with confusing mathematical notions and formulas, making the paper hard to read.
-  It is unclear how to use and interpret $A$ and $\Phi$.
-  The experiments have no ground-truth. Therefore, it is hard to justify the performance of the proposed approach.

### Questions
- After finishing learning, how to use two tensors $A$ and $Q$.
- What are the meaning of Q and A?

### Soundness
2

### Presentation
1

### Contribution
2
