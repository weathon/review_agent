# Pre-training Pure GNNs as Graph Learners

- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
Graphs from different datasets exhibit diverse numbers of features and labels, where each feature or label is associated with different semantic meanings. Such diversity poses challenges in adapting pre-trained graph neural networks (GNNs) to different datasets with a single set of input and output (I/O) module parameters. This raises a fascinating question: Can pure GNNs be pre-trained on diverse datasets, adapting to various datasets effectively without additional effort? To explore this, we propose unified I/O modules that enable pre-training with pure GNNs. Unlike traditional methods that tightly couple parameters to specific datasets, our approach decouples parameters through a shared relation function for the input and uniformly sampled points for the output. These designs effectively resolve the challenges in quantity inconsistency and semantic discrepancies of dataset features and labels. By integrating our I/O modules with various GNN architectures, we demonstrate that pure GNNs can be effective graph learners for direct adaptation to downstream tasks. Pre-training experiments under different setups show that increasing hidden dimensions and the average number of nodes per training dataset enhances model performance. Moreover, fine-tuning the I/O modules with frozen pre-trained graph operators significantly simplifies the model hyperparameter tuning process, achieving superior or comparable performance to supervised models on downstream datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes input (on node features) and output (for node predictions) modules for GNNs to allow pre-training them and performing downstream predictions on different tasks. Empirically, this paper demonstrates that downstream performances improves with increased training data.

### Strengths
- **(S1) Contribution (Originality)** Cross dataset training for GNNs is an important unsolved problem. This paper proposes a relatively clean solution to the key problems of cross dataset training (mapping input and outputs).
- **(S2) Empircial Evaluation (Quality)** The empirical evaluation is well done. The authors test on many datasets from different domains and evaluate many different backbones (=GNNs). They investigate the performance of their methods in different settings (1-shot, 3-shot, fine-tuning) and compare against a diverse array of models.
- **(S3) Performance (Significance)** Finally, performance of the model in few-shot and fine-tuning scenarios is good and often outperforms all other models.

### Weaknesses
- **(W1) Clarity of definitions:** Section 3 is dense and difficult to understand. In particular, the definition of the I/O modules is difficult to follow due to the density of notation. Here are some points that need clarification:
	-   $\mathbf{S}\_{\text{src}}$ and $\mathbf{S}\_{\text{tgt}}$  are unintuitive, are these simply learned weight matrices?
	- The definition of $f_\texttt{in}$ states that it maps from $\mathbb{R}^{n \times d_{\text{in}}}$ to $\mathbb{R}^{d_{\text{in} \times s}}$. However, this means it does not produce node-embeddings?
	- What is the difference between your input module and DeepSet?
	
- **(W2) Grounding of empirical results:** Figure 5 is difficult to interpret, it is supposed to show a performance gap between models on different domains / different heterophily but it is unclear to me what the figure shows (and how big the performance gap actually is).


- **(W3) Scaling:**
	- **(W3.1)** Figure 3 shows the relation between GNN model size (#layers or hidden dimension) against test set performance. We can see that for the best performing models such as GCN or mixhop the model size has very limited impact on test performance.
	- **(W3.2)** Figure 4 shows the relation between the average number of nodes and test performance. While there is a positive correlation it seems to be quite small (it would be good if the authors could quanitfy this). Furthermore, training on more datasets seems to have little impact on test performance.
	- Combined (W3.1) and (W3.2) seem to indicate a fundamental limitation of this approach. While pre-training does clearly give us good performance in (few-shot) settings, this approach seems to be unable to scale with model size and more diverse data.

	- **(W4) Homophilic data:** The model struggles with homophilic data (Figure S6). While I think that this is not a big problem since the model works well on heterophilic data, the authors should more directly state that in the main paper. (This is not clear enough about the reslts in the appendix: _"Results show that models generally achieve better results on heterophilic test graphs than homophilic ones ..."_)



- **Minor Weaknesses**
	- While Figure looks visually appealing, it is not helpful in understanding the architecture.

### Questions
- See weaknesses.
- What does "original split" in Table 1 mean?


**Overall,** while I think that this paper makes some good contributions there two primary things that need to be addressed. (W1) The writing makes it difficult to understand the architectural advances. (W3) The scaling results seem to indicate a fundamental limitation of this approach. Overall, I am voting 4 - marginally below acceptance threshold.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper explores whether pure graph neural networks can be pre-trained across diverse graph datasets without relying on language models. The authors propose unified input and output modules that decouple model parameters from dataset-specific feature and label spaces—using a shared relational function for input features and uniformly sampled pseudo-labels for outputs. This allows GNNs to generalize across datasets with different semantics. Experiments show that the proposed method enables effective cross-dataset pre-training, achieving competitive or superior results to supervised and LM-based baselines while simplifying fine-tuning and hyperparameter tuning.

### Strengths
1.	The paper is well-organized and easy to follow. It provides clear motivation, formal problem formulation, and theoretical derivations that make the proposed Unified I/O framework convincing and conceptually coherent.
2.	The paper tackles a highly relevant and underexplored problem—how to unify diverse graph datasets with different input and output spaces. This direction has strong practical significance for building more generalizable graph foundation models.

### Weaknesses
1.	The proposed method is only demonstrated on node classification tasks, while other settings such as node regression and graph-level tasks (where graph pre-training is often most impactful) are not explored. This narrow scope limits the general applicability and practical influence of the framework.
2.	Although the experiments are extensive, the performance of Unified I/O is not consistently competitive. In many settings, it falls notably behind the best existing methods. Moreover, several chosen baselines are relatively weak — for example, on the Cora original split, Unified I/O achieves 82.32%, yet many self-supervised approaches surpass this level by a clear margin. This undermines the strength of the empirical claims.
3.	The results in Figure 4 are disappointing — increasing the number of training datasets yields almost no improvement, which questions the necessity of such pre-training compared with simply performing self-supervised learning on a single dataset. In Figure 3, the claim that performance keeps improving with larger hidden dimensions or deeper layers is counter-intuitive; the curves flatten toward the end, suggesting the authors may not have reached the turning point where over-parameterization degrades results. If not, additional evidence is needed. Moreover, in scaling analysis, it would be more appropriate to examine total model parameter count, as commonly done in LLM research, rather than only hidden dimension or layer depth.
4.	The output module and final loss design of Unified I/O resemble a clustering process, raising concerns about convergence stability. On complex datasets, if the initial parameters are far from optimal, it is unclear whether the model can still converge to a good solution. This also raises potential cold-start issues during pre-training, as the model may lack meaningful gradient signals in the early stage.

### Questions
Please refer to the Weaknesses section above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a unified I/O module that effectively aligns the dimensions and semantics of features and labels across diverse datasets. This design enables GNNs to be pre-trained on diverse datasets and directly adapted to unseen downstream datasets. The authors conduct extensive experiments by integrating the proposed module with a variety of GNN architectures. The experimental results demonstrate that the module can effectively unify feature and label spaces and integrate well with different GNN architectures.

### Strengths
S1. The proposed output module is novel, as it enables the pre-trained models to directly adapt to diverse target spaces.

S2. The proposed method demonstrates strong performance in node classification under various settings.

S3. Figure 4 reveals an interesting and counter-intuitive phenomenon that increasing the node scale of training datasets improves the model’s adaptability to test datasets, while increasing the number of training datasets may not lead to better performance.

### Weaknesses
W1. The proposed input module is similar to the feature unification mechanism proposed by FUG [1]. The authors should provide a more detailed discussion on the specific distinctions

W2. The explanation of why the proposed shared relation function can unify feature and label semantics mainly relies on intuition, lacking a theoretical justification or semantic alignment analysis.

W3. The downstream task only includes node classification. Although Appendix A claims that the method can be applied to general graph learning tasks, there is no experimental validation to support this claim.

W4. The compared methods are limited. How does the proposed method compare with SOTA GFMs, such as FUG [1], SAMGPT [2], RiemannGFM [3]?

[1] FUG: Feature-Universal Graph Contrastive Pre-training for Graphs with Diverse Node Features. NeurIPS' 24.

[2] SAMGPT: Text-free Graph Foundation Model for Multi-domain Pre-training and Cross-domain Adaptation. WWW' 25.

[3] RiemannGFM: Learning a Graph Foundation Model from Riemannian Geometry. WWW' 25.

### Questions
Apart from the weaknesses, 

Q1. In some settings, the proposed method performs worse than GraphAny on homophilic graphs, but better on heterophilic graphs. As far as I know, GraphAny has designs for heterophily, while this method does not. Why does this method outperform GraphAny on heterophilic graphs?

### Soundness
3

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
4

### Summary
This paper proposed a new pre-training framework for graph data. Considering the varying input feature dimension and output label classes, the authors mainly devised I/O modules for input and output, which avoid direct connection with input data and actual downstrem tasks.

### Strengths
1. The motivation of finding graph foundation model is meaningful.
2. The organization is good to follow.

### Weaknesses
1. Techniques. \
At input side, the authors input feature similarity matrix, something like X^TX, rather than raw feature, followed by a all-one vector, so the model always receives the same length of input. But this operation is oversimplified, and lose a lot of information. Consider two cases, 1) one-hot raw features (e.g., user / item feature in recommendation), so X^TX is 0, 2) some datasets may have thousands of dimension features, and some datasets (e.g., molecule QM9) only have very limited feature dimensions, so X^TX could reflect totally different feature relations. How can we deal them uniformly? Further, features may appear very complex patterns, and only X^TX cannot fully capture them.\
At output side, different datasets have different number of classes, from single digits to hundreds / thousands. How to set the number of pseudo lables? Also, different classes sometimes are not totally independent, while pseudo labels are selected evenly in the sphere space.

2. Experiments. 1) In table 1, traditional semi-supervised GNNs are also need to compare, to show the necessity of pre-training. 2) In table 2, the proposed method has obvious improvement only on COMPUTERS if consider std.

### Questions
In Fig. 3 (b), why did the GNNs not occur over-smoothing, when stacking many layers?

### Soundness
2

### Presentation
3

### Contribution
2
