# AutoDV: An End-to-End Deep Learning Model for High-Dimensional Data Visualization

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 2

## Abstract
High-dimensional data visualization (HDV) plays an important role in data science and engineering applications. Traditional HDV methods, such as Autoencoder and t-SNE, require hyper-parameter tuning and iterative optimization on every dataset and cannot effectively utilize the knowledge from historical low-dimension representation, which lowers the efficiency, convenience, and accuracy in real applications. In this paper, we present AutoDV, an end-to-end deep learning model, for high-dimensional data visualization. AutoDV is built upon a graph transformer network and an invariant loss function and is trained on a number of diverse datasets converted into multi-weight graphs. Given a new dataset, AutoDV outputs the 2D or 3D embeddings of all data points directly. AutoDV has the following merits: 1) There is no hyper-parameter selection during the data visualization stage; 2) The end-to-end model avoids re-training or iterative optimization when visualizing data; 3) The input dataset can have any number of features and can be from any domain. Our experiments show that AutoDV can successfully generalize to unseen datasets without retraining with 89.37\% precision of t-SNE and 91.05\% precision of UMAP on the unseen CIFAR10 datasets. Compared with existing parametric data visualization deep models, our method obtains a significant improvement with 86.65\% precision gain. AutoDV can perform even better than t-SNE and UMAP on gene and UCI tabular datasets. The project is available at https://github.com/DryDew/AutoDV.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work proposes an end-to-end high-dimensionality reduction that generalises to unseen datasets with various feature sizes without retraining. Given some historical dimensionality reduction representations as the learning example, the proposed mechanism acquires meta-knowledge of how to capture the structure of high-dimensional data and preserve it in the low-dimensional representation.
To the best of the reviewer's knowledge, this ability is rarely studied, and thus, this work is novel.

### Strengths
This is a very solid work, supported by strong theoretical arguments and valid experiments.

The strengths of this proposed algorithm are as follows:
1. It offers an end-to-end DR mechanism that generalises across many not new data points, but new datasets. It frees users from the need to retrain the DR method.
2. It does not need any hyperparameter tuning.

### Weaknesses
The reviewer found no significant weaknesses in this work.

The proposed algorithm was trained only on historical low-dimensional representations from t-SNE and UMAP. The reviewer understands that this is due to the page limitation. However, the proposed work's generality will be further strengthened if the authors can train AutoDV using other DR algorithms. Specifically, t-SNE and UMAP are non-linear and unsupervised DR methods. Hence, it will be interesting to observe the generality of the AutoDV against a linear DR such as traditional PCA or MDS, and against a supervised linear DR such as LDA.

### Questions
1. In pg. 3 subsection 3.1, the authors wrote: \theta^*_i denotes the optimal hyper-parameters selected using y_i (the labels). However, in the experiments, the proposed AutoDV was trained using historical DV, t-SNE, and UMAP, which are unsupervised DRs that do not account for dataset labels. So the statement does not reflect the experiments well.

2. The term "historical datasets" can be confusing. The term "historical low-dimensionality representations"  is better. Please review the usage of this terminology.

3. To further show the meta-learning ability to capture the preserved structure of high-dimensional data, it will be interesting if the authors can run additional experiments against t-SNE with various perplexity values (or UMAP with various numbers of neighbors). It is of interest to observe that, for a perplexity that generates poor maps, it will still be possible for AutoDV to capture the preserved structure.

4. pg. 2, "Lack of cross-domain....": min_theta E_i|f(x_i).....| ->  min_theta E_i|f_theta(x_i).....|

5. Fig.1 is too small to see

6. Some novel DR methods may benefit the depth of this paper. Please consider including them in the discussion and for future experiments to further argue the proposed method's generality

- A. Dehghani, et al.,Credit-based self organizing maps: training deep topographic networks with minimal performance degradation, ICLR 2025, https://openreview.net/pdf?id=wMgr7wBuUo

- P. Hartono, Mixing autoencoder with classifier: conceptual data visualization, IEEE Access, Vol. 8, pp.105301 -105310 (2020) DOI: 10.1109/ACCESS.2020.2999155

- P. Hartono, P. Hollensen, T. Trappenberg, Learning-Regulated Context Relevant Topographical Map, IEEE Trans. on Neural Networks and Learning Systems, Vol. 26, No. 10, pp. 2323-2335 (2015). DOI:10.1109/TNNLS.2014.2379275

### Soundness
3

### Presentation
3

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
This paper addresses the practical pain points of high-dimensional visualization (HDV): methods like t-SNE and UMAP require dataset-specific hyperparameter tuning and iterative optimization, and their results vary across domains and input dimensionalities. The authors propose AutoDV, an end-to-end model trained once to produce 2D/3D embeddings for unseen datasets without per-dataset tuning. The key idea is to learn from historical, high-quality visualizations and generalize that “visualization know-how” to new data.

Technically, AutoDV first graphizes any dataset—regardless of its original feature dimension—into multiple similarity graph views (e.g., Gaussian kernels at different bandwidths). It then computes graph positional encodings and processes each view with a per-view GNN; the view outputs are concatenated and fed to a Graph Transformer, followed by an MLP that yields the final low-dimensional coordinates. Training is supervised by “optimal” target embeddings obtained from conventional HDV (e.g., t-SNE/UMAP selected via Bayesian optimization on labels/metrics). Instead of matching raw coordinates, AutoDV minimizes an affine-invariant divergence between pairwise similarity structures of its prediction and the target, eliminating rotation/translation/scale ambiguities and stabilizing learning.

The paper’s main limitations are the dependence on historical “optimal” targets (which may require labels and careful tuning to curate) and potential quadratic costs from pairwise structures (mitigated by batching/anchors). Nonetheless, AutoDV offers a compelling step toward set-and-forget visualization: a single trained model that produces stable, near-optimal embeddings for diverse new datasets without per-dataset fiddling.

### Strengths
1. The idea of learning a mapping from multi-view graph topology to a low-dimensional layout is novel. It decouples from raw feature dimensionality, uses affine-invariant supervision, and targets cross-dataset generalization without per-dataset tuning.

2. The authors demonstrate strong command of related work and discuss it thoroughly. They situate the approach against classical DR (PCA/MDS), modern manifold learners (t-SNE/UMAP and variants), and parametric/inductive methods, clearly motivating the gap.

3. The experiments are extensive across many datasets and modalities. Image (with pretrained features), gene-expression, and tabular benchmarks are covered with multiple metrics, offering broad evidence for robustness and generalization.

### Weaknesses
1. Unclear data requirements for learning from historical visualizations. 

It is not specified how much and what diversity of historical visualizations are required for AutoDV to reliably replace UMAP in practice. The paper should quantify sample complexity and coverage (e.g., number of datasets, domain diversity, class balance, feature dimensionality ranges) and provide learning curves/ablation studies (performance vs. amount and variety of historical supervision).

2. Evaluation relies heavily on NMI, which can be a coarse proxy. 

When high-dimensional features (or labels) are imperfect, cluster labels are not a reliable indicator of visualization quality. The paper should include neighbor-preservation metrics and rank-based fidelity measures—e.g., kNN overlap/Jaccard@k, Precision/Recall@k, Trustworthiness/Continuity, triplet preservation, and (optionally) global distortion/stress or distance rank correlations—to demonstrate that local and global neighborhood structures are faithfully preserved.

### Questions
1. How do you expect the user should choose AutoDV over UMAP in practice?

2. What is the neighour-preserving rate between the high and low dimensional data?

### Soundness
3

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
This paper introduces AutoDV, a high-dimensional data visualization algorithm which leverages a graph transformer network to embed datasets into lower dimensional spaces. Datasets are first converted into a similarity graph, which is passed into a graph transformer, which is trained to predict the ground truth TSNE/UMAP values under optimal hyperparameters. The paper highlights two challenges in performing this end-to-end modeling task: different sizes/dimensions for the input points, and the presence of multiple ground truths for each underlying input (under rotation,  scaling etc.). To solve the first challenge, the paper ignores the dimensionality of the points, and relies on taking as input a similarity graph under a gaussian kernel function (Following Long, 2015). This makes it possible to apply a GNN to encode the graph, and reproduce it in a lower dimension. To solve the second challenge, the paper introduces an affine-invariant loss function, which attempts to align the pairwise squared similarity matrices in the lower dimension under a Bregman divergence (using either KL divergence for T-SNE or a structural similarity, given in Eq. 8, for UMAP). 

The paper is evaluated across several image datasets (MNIST, Fashion-MNIST, CIFAR-10), as well as several datasets for bioinformatics (Baron Huntman, Mouse Retina, Cambell and PBMC68K), and tabular data (UCI Machine Learning).  Evaluations of the NMI and Silhouette Coefficient show that AutoDV approaches, and sometimes exceeds the representation performance of the baseline approach across almost all results, while being somewhat faster than the existing methods (Particularly an un-optimized TSNE).

### Strengths
This paper tackles an interesting problem in data visualization: can you learn a function which end to end approximates t-SNE or UMAP (or really any of the dimensionality reduction methods), which doesn't require tuning the hyperparameters. AutoDV is a step towards that: there's no hyperparameter selection, and it generates what are numerically better visualizations of underlying data. The theoretical stability results are nice (albeit somewhat weak), and the method is much more efficient than un-optimized t-SNE, which means that there are potential gains here as well.

### Weaknesses
While there are some strengths, the paper does have significant weaknesses:
- For a paper on data visualization, there is very little actual visualization in the paper. It would be good to demonstrate qualitative results for several different datasets (Ideally beyond those in Figure 4), which honestly do not look remarkably similar to the underlying optimal TSNE/UMAP examples (even under transformation). What about visualizing some more classic datasets (swiss roll, lines, etc.)? 
- It's also not clear here that the learned parameters are that much better than defaults (For example, for Figure 4, how well does a single learned default parameter perform?) - Could you have a similar baseline, perhaps, that just uses the TSNE* data to inductively learn the optimal dataset-specific parameters? 
- It's not entirely clear if this model will generalize to new datasets, and the paper does little . While the paper does show that it generalizes from MNIST and FMNIST to CIFAR, it's not entirely clear how far that generalization goes: will a pre-trained AutoDV model generalize from CIFAR to a Genetic dataset? Or from CIFAR to something like the swiss roll? It would be really good for this paper to explore the extent of generalization - how far away to the datasets have to be structurally before the models start failing (CIFAR 10 has 10 classes, for example, which is shared with MNIST). There's also no datasets here that are not classification centric (i.e. all datasets have some larger class structure), which means that it's hard to tell if an AutoDV model will generalize between underlying manifold structures.
- There's a pretty large limitation in that the model can only handle 3000 point datasets (L364). It's a proof of concept, but it seems pretty bad, especially given the potential scalability of UMAP/TSNE.
- There's not really much analysis of the failure cases of the model - are there any? How does the model perform in situations where there are high-noise, or overlapping clusters, or even on random noise? 
- There are some really confusing wordings in this paper, which make it quite challenging to read, for example, the following sentence: "To allow a deep neural network to accept different input dimensionalities directly is barely possible, which is what the existing parametric data visualization models suffer from." which motivates the design of the paper. 
- The paper is lacking some fundamental motivation/ablations: Why were the base architectural choices made? Are there scaling properties to the AutoDV metric? The practical choices for $\mathcal{K}_\psi$, and particularly $\psi$ are not particularly well motivated - and no alternatives are explored. There's also no ablations on the different loss components (with/without something affine aware, with/without sign ambiguity, etc.).

### Questions
- How are the optimal parameters for TSNE/UMAP chosen (I might have missed this in the appendix)? It seems like while AutoDV is trained to approximate these methods, the optimal parameter choice chan highly influence the final outcomes.
- Are there any scaling properties? Does the AutoDV model have to be a particular size to achieve this performance? 
- Does the model generalize beyond classification datasets? How much does the underlying manifold have to change in order to break the hyperparameter approximation of this method? 
- Is this any different than just learning good initial hyperparameters for TSNE/UMAP? Does it outperform such methods?

### Soundness
2

### Presentation
2

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
The authors propose an automated way of learning to embed whole datasets similarly to a specific prescribed embedding algorithm, without the need of choosing its hyperparameters. The hyperparameters, used to generate each embedding, are different for each dataset and are chosen based on the labels of the dataset. These labels are used the validate the quality of the embedding, and are not used throughout the embedding process. Then, at inference time, the model produces an embedding that should correspond to an embedding with hyperparameters selected in the same manner for this dataset.

### Strengths
The fact that the neural network was able to produce an embedding based on the test data is interesting.

### Weaknesses
1. The terminology is not consistent with the manifold learning field. The authors propose to generate "k -view graphs", where in each of the k graphs they use a different bandwidth parameter. There are a whole field that deals with multi-view dimensionality reduction, where each view considers a different source. Here on the other hand, we get the same source. The more suitable term here will be multi-scale, as the algorithm generates the similarity matrix over the data in multiple scales of bandwidths. 

2.  Paper aims to solve too many tasks altogether. If I understand correctly, the paper aims to solve two problems together, including: (A) Extract good hyperparameters (and their corresponding bandwidth/s) for a given dataset, and (B) Generate an embedding based on these bandwidths. Each of these tasks are non-trivial, and I would expect the authors to show their superiority on each task independently.

3. Quantification metrics. The training data includes datasets along with their corresponding embeddings, which are the result of applying the same embedding algorithm with different hyper-parameters. These hyper-parameters were chosen based on some supervised metric. I wonder why the authors do not compare the embeddings they got from test data with the true embedding (of the embedding algorithm) using its "optimal parameters", one approach to do so is through their graphs.

4. Unsupervised bandwidth selection. The problem of unsupervised bandwidth selection is studied in the context of different embedding algorithms, and in other related contexts such as KDE. I did not see any discussion about this field.

5. Missing clarification about choices related to the neural network. For example, why is the SVD positional encoding good for this case, and why do embedding algorithms such as the ones considered in this paper relate to them. Going even further than that, you can then ask how this encoding corresponds to the initialization of the selected embedding algorithm.

6. The paper will really benefit from generating even a synthetic dataset, which will be easy to understand, and then the generated embeddings along with the scores can better illustrate the algorithm's output more clearly. 

7. Theoretical Configuration. The configuration of the problem is not well defined.

### Questions
1. NMI. How did you choose the number of K in K-means when you calculated the NMI? I am wondering whether you allowed K to be much larger than the number of clusters as the problem is related to Manifold Learning and not only to Clustering.

2. Same bandwidths used across all datasets. The optimal bandwidths may vary in orders of magnitudes between different datasets (and even within the same dataset). It can depend on all kind of factors, like the number of samples and the type of the data (e.g., curvature, density) . In a lot of cases, even z-scoring wouldn't remove this phenomena. How did you tackle this problem, or maybe this problem did not occur in your data as it had very similar characteristics (like in the image data that used the clip embedding space) ?

3. How many samples are there in each subset within the training data?

### Soundness
1

### Presentation
1

### Contribution
1
