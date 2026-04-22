# Hydra: Towards Transferable Multi-Task Learning on Temporal Graphs

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Real-world evolving networks are naturally modeled as temporal graphs (TGs), where capturing temporal dynamics is essential for predicting future graph properties that support downstream decision-making. Existing temporal graph methods have been developed primarily for single-task prediction, and little is known about their generalization across tasks or transfer to unseen networks. This leaves the challenge of multi-task graph property prediction in TGs largely open. We address this challenge by introducing Hydra, a novel architecture that integrates local connectivity features from temporal GNNs with a spectral learning module that captures global connectivity patterns. This design enables joint learning of local and global information under a multi-task objective. In multi-task classification, Hydra achieves an 8.9\% relative gain in AUC over the strongest competitor. In multi-task regression, Hydra achieves competitive results in all three tasks, while obtaining the best results in two tasks with a 8.2\% relative gain in MAE compared to the strongest baseline. Moreover, Hydra delivers these gains with a 22× reduction in training time compared to temporal transfer models. These results provide the first systematic evidence that multi-task transferable learning on temporal graphs is effective. By delivering consistent top-ranked performance, Hydra highlights multi-task training on temporal graphs as a promising direction toward adaptable foundation models for temporal graphs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper propose a multi-task learning paradigm called Hydra that incorporates both spatial and spectral features for temporal graph property prediction tasks. The core design innovation in Hydra lies in the spectral memory module that utilizes density-of-states (DoS) descriptors as node-wise topological features that is efficiently aggregated via an attentive fashion which serves as the spectral feature that benefits temporal graph problems. Experimental results on MiNT tasks demonstrate competitive performance.

### Strengths
- Leveraging multi-task correlations in temporal graph prediction is an important aspects in temporal graph representation learning.
- The paper proposes a method to combine spatial and spectral mechanisms over temporal graphs that benefits downstream modeling.

### Weaknesses
- **Lack of innovations in multi-task paradigms** The core problem studied in this paper is multi-task learning (MTL). However, it seems that the solution to multi-task challenges proposed in this paper is simply the *shared-trunk with task-specific heads* paradigm, which is rather the ad-hoc way in MTL. Furthermore, in experimental session, the authors primarily compares the MTL approach with single model baselines that are trained on less data. While the authors referred to those baselines as *single-model*, I believe nearly all of these models could be modified with minimal effort to incorporate the *shared-trunk with task-specific heads* structural that naturally handles multi-task learning, as with basically any embedding-based prediction primitives. Therefore, I think from the perspective of multi-task learning, the paper lacks novelty.

- See questions below

### Questions
- **Actual Complexity of DoS** While the authors provided a complexity description in appendix B, I am still puzzled about the efficiency gain of Hydra, as in comparison to standard solutions in discrete-time temporal graphs like EvolveGCN, the paper seems to add one additional dimension of complexity that is induced via computing spectral features of each graph snapshots. Such types of spectral features are actually not that scalable------**I checked the code implementations in your appendix**, if you compute spectral features via *svd*, then with temporal graphs with magnitudes of ~10k per snapshot the computational cost would be prohibitive. If instead using the ``moments_cheb_dos`` implementation as detailed in your scripts, the complexity will then be dominated by a large number of matrix-vector computation involved in some power-iteration like computation traces (correct me if I am wrong). Therefore, **I am skeptical towards the efficiency reports in the paper that seem to overly promising**, but when applied to realistic temporal graphs at industrial scale, the primary innovation in this paper might be not applicable.
- **Comparison with alternative spectral embeddings** It appears to me that the proposed DoS approach is yet another spectral-based position encodings along with another temporal aggregation post-processing. As spectral-based position encodings have been studied extensively in recent years [1, 2, 3], I am curious about the difference between DoS and, for example, Laplacian eigenvectors [1] and more advanced approaches.

[1]. Dwivedi, Vijay Prakash, and Xavier Bresson. "A generalization of transformer networks to graphs." arXiv preprint arXiv:2012.09699 (2020).  
[2]. Rampášek, Ladislav, et al. "Recipe for a general, powerful, scalable graph transformer." Advances in Neural Information Processing Systems 35 (2022): 14501-14515.  
[3]. Kanatsoulis, Charilaos I., et al. "Learning efficient positional encodings with graph neural networks." arXiv preprint arXiv:2502.01122 (2025).

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the problem of multi-task temporal graph prediction, focusing on predicting global graph properties' evolution over time. The authors introduce an architecture which adds global spectral features to common temporal dynamic graph model primitives. This model is trained in an end-to-end fashion, on multiple temporal graphs and on multiple tasks at once, with the intent of being used in a zero-shot setting on unseen temporal graphs. The presented results show that this approach outperforms other temporal graph methods trained and evaluated on the target dataset directly.

### Strengths
- The paper is well written, has a nice flow and the presentation is very clear
- Experiments are well designed (with some caveats on the baselines, see Weaknesses) and the results well presented
- The results are impressive, particularly given that, as far as I understand, the authors are comparing zero-shot transfer performance to the performance of models trained and evaluated on the same datasets (with the caveat that datasets in the benchmark seem homogeneous in nature, all representing blockchain transactions)

### Weaknesses
### 1. Motivation
The main contribution of the paper is introducing a new problem setting of multi-task global property prediction for temporal graphs. In my view, its main weakness is that the authors spend little effort motivating this setting. 
Given that there is barely any prior work trying to solve this problem (making it easier to improve upon weaker baselines designed for others), the value of the paper is inherently tied to if the setting has some practical relevance and is not a purely academic endeavor. In this regard, I found the benchmarks used in the paper to be unconvincing. Global property prediction tasks seem arguably less relevant in temporal graphs than in static ones and the choice of tasks in the paper seem somewhat artificial. Examples with more obvious real world utility would, therefore, be a welcome addition.

An additional point is that, even in this new setting, the authors acknowledge limitations of their approach in learning classification and regression tasks at the same time, restricting the scope to one or the other, further limiting practical applicability.

### 2. Baselines
While the presented results look solid, it should be pointed out that temporal graph models are often designed with local tasks like link prediction in mind. The addition of global spectral features to the proposed model would, therefore, seem like an adaptation that greatly benefits this particular task (as demonstrated by the authors' ablation) but could just as easily be incorporated into existing models to level the playing field. 

### 3. Novelty
Besides the architectural improvements (mainly the addition of spectral features mentioned previously), this work appears to be a trivial extension of MiNT to multiple tasks. This is done simply by adding losses for the tasks together using multiple prediction heads (a very standard approach). It seems therefore that it would be trivial to train the MiNT model on the same multi-task objective, rather than train one model per task. 

Claiming a training speedup w.r.t. MiNT using such a different setup seems like an artificial attempt at claiming an advantage and a poor framing of the story. In my view these 2 aspects should be decoupled: model architecture and training approach (all tasks at once vs one model per task).

Finally, the authors only compare to MiNT in the classification setting, with the justification that this was the original setting in that paper. But it seems that adapting this approach to the regression tasks would be a trivial matter of changing the training loss.

### 4. Other Minor Issues

While I found the paper to be very well written, Section 3.2 is repetitive and overly verbose.

### Questions
1. What are some practical use cases and motivation for multi-task global property prediction in dynamic graphs?
2. Why tie together model architecture and training approach (all tasks at once vs one model per task). It seems like it would be trivial to adapt any architecture to predict multiple properties and train on multiple tasks.
3. Why not also compare to MiNT for regression tasks? Is there a concrete impediment?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper addresses the challenge of multi-task property prediction for temporal graphs by introducing Hydra, a novel framework that integrates local and global connectivity patterns. Hydra uniquely combines the strengths of temporal Graph Neural Networks (GNNs) to capture localized connectivity features with a spectral learning module designed to discern global connectivity patterns. This integrated design facilitates the joint learning of both local and global information under a unified multi-task objective. To validate its effectiveness, the authors conducted extensive experiments, demonstrating that Hydra not only outperforms strong baseline models but also significantly reduces training time compared to temporal transfer learning approaches.

### Strengths
S1: The paper investigates the under-explored problem of multi-task learning within the context of temporal graphs, an area of considerable value and current interest.

S2: The presentation is clear and easy to follow.

S3: The proposed method exhibits superior performance across diverse classification and regression tasks.

### Weaknesses
W1: Limited Novelty: The novelty of the proposed method appears limited, as it presents as a straightforward combination of existing techniques. More critically, the paper fails to sufficiently articulate how this specific combination is tailored to address the unique challenges of multi-task learning on temporal graphs. Simply applying existing technologies to a new problem is not enough. The authors should provide a clearer justification for why the "trunk" and "head" architecture is particularly advantageous for enabling knowledge transferability and robust performance in this specific context.

W2: Limited Scope of Multi-Task Learning: The evaluation of multi-task learning is constrained to scenarios involving similar task types (e.g., multiple classification tasks). Such tasks likely require similar feature information, which may simplify the model's optimization process. This experimental setting may not be representative of practical scenarios where a model must handle more heterogeneous tasks (e.g., a mix of classification and regression).  

W3: Insufficient Study of Multi-Task Capabilities: The investigation into the model's multi-task capabilities is insufficient. All experiments are conducted exclusively in a three-task learning setting. To improve the convincingness of the method's multi-task capabilities, the study should be expanded to include experiments with a different number of tasks (for instance, a two-task setting). This would provide a more comprehensive validation of the proposed method's flexibility and robustness.

W4: Lack of Comprehensive Training Time Comparison: The analysis of training time is incomplete. As shown in Figure 4a, Hydra is only compared against a single transfer learning model (MiNT), which is insufficient to fully substantiate the claims of its superior efficiency. A comparison against single-task models is necessary for a thorough evaluation. This comparison should include at least two standard settings: 1) training single-task models from scratch for each new task, and 2) fixing a pre-trained encoder (This encoder can be trained on the first task.) and only training a new head for each new task.

W5: Lack of In-Depth Analysis of Experimental Results: The paper lacks an in-depth analysis of certain experimental outcomes, which hinders a full understanding of the model's limitations. For example: 
-	In Table 2, Hydra fails to achieve the best rank and MAE on the "Influential Node Count" task, yet no analysis or potential explanation for this observation is provided. Discussing such "failure modes" is crucial for understanding the model's boundaries.
-	In Figure 3b, the performance trend for one task shows a different pattern compared to the other two tasks (LLC and edge), but this discrepancy is not discussed.

W6: Incomplete Ablation Studies: The ablation studies are incomplete, as they are conducted exclusively on classification tasks. To convincingly demonstrate the effectiveness of the model's core components, the ablation study should be extended to include the regression tasks as well. This would provide more robust evidence that the contribution of each component is consistent and valuable across different types of tasks.

### Questions
Regarding Figure 4a, what is the meaning of the dashed line connecting the results at 32 and 64 training networks?

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
2

### Summary
1. The paper propose Hydra, an architecture capable of handling multiple temporal graph prediction tasks without retraining.
2. Hydra incorporates Laplacian descriptors into TGNNs for graph-level prediction.
3. The paper compares Hydra with previous methods for single-model/task prediction using multi-task classification.

### Strengths
1. The submission provides with the implementation and code.
2. Hydra is the first model capable of handling multiple temporal graph prediction tasks without retraining.

### Weaknesses
1. The experiments do not report the the performance drop brought by Hydra. If the per-task AUC/MAE of Hydra is much lower than training models per task, this proposed new multi-task learning paradigm seems to have more disadvantages than advantages.
2. Lack of citations to relevant papers [1, 2].

[1] Temporal graph benchmark for machine learning on temporal graphs

[2] Benchtemp: A general benchmark for evaluating temporal graph neural networks

### Questions
If we directly apply the modifications required for the temporal graphs to multi-task GNN architectures, how does this combination compare to Hydra?

### Soundness
3

### Presentation
3

### Contribution
3
