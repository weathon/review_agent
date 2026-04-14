# DyGMamba: Efficiently Modeling Long-Term Temporal Dependency on Continuous-Time Dynamic Graphs with State Space Models

- Decision: Reject
- Scores: 3, 3, 8, 3, 3

## Abstract
Learning useful representations for continuous-time dynamic graphs (CTDGs) is challenging, due to the concurrent need to span long node interaction histories and grasp nuanced temporal details. In particular, two problems emerge: (1) Encoding longer histories requires more computational resources, making it crucial for CTDG models to maintain low computational complexity to ensure efficiency; (2) Meanwhile, more powerful models are needed to identify and select the most critical temporal information within the extended context provided by longer histories. To address these problems, we propose a CTDG representation learning model named DyGMamba, originating from the popular Mamba state space model (SSM). DyGMamba first leverages a node-level SSM to encode the sequence of historical node interactions. Another time-level SSM is then employed to exploit the temporal patterns hidden in the historical graph, where its output is used to dynamically select the critical information from the interaction history. We validate DyGMamba experimentally on the dynamic link prediction task. The results show that our model achieves state-of-the-art in most cases. DyGMamba also maintains high efficiency in terms of computational resources, making it possible to capture long temporal dependencies with a limited computation budget.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper presents DyGMamba, a modeling framework designed for continuous-time dynamic graphs. Built on a state space model (SSM), DyGMamba captures and leverages hidden temporal patterns from historical graph data. These temporal patterns guide the selection of critical information from a node’s temporal neighbors, enhancing the model's ability to predict dynamic links. DyGMamba achieves state-of-the-art results across seven datasets in dynamic link prediction tasks. Furthermore, extensive experiments have been conducted to demonstrate the framework's efficiency.

### Strengths
1. The efficiency improvement introduced by Mamba is sound compared to the Transformer.
2. This paper gives a clear and detailed description of the algorithm.
3. Both link prediction and node classification tasks are discussed.
4. The extensive experiments demonstrate that the Mamba structure significantly improves computational and memory efficiency compared to the Transformer.

### Weaknesses
1. **Low Novelty:** 
- Based on my understanding of the algorithm, Section 3.1 is identical to DyGFormer, except that it replaces the Transformer with Mamba. In Section 3.2, the mean pooling mechanism in DyGFormer is modified to a "Dynamic Information Selection" strategy
It appears, then, that **the primary contribution lies in the design of the pooling strategy**, which is more like a minor technical improvement and totally **unrelated to the Mamba structure**. This is somewhat misleading, given that both the name DyGMamba and the abstract emphasize the Mamba component as a key feature. 
- Furthermore, in the ablation study, Variant A uses mean pooling for the output in Equation 5. This modification makes Variant A essentially identical to DyGFormer, with the sole difference being the replacement of Transformer with Mamba. However, the performance of Variant A drops compared to DyGFormer, suggesting that incorporating the Mamba structure into DyGFormer negatively impacts performance. This observation implies that the performance improvements in DyGFormer stem from the "Dynamic Information Selection" mechanism, while the efficiency gains are due to the Mamba structure and the patching techniques utilized within DyGFormer.
For this reason, **DyGMamba’s design offers limited novelty, as its efficiency gains are entirely derived from prior studies**—specifically, the Mamba structure and the patching techniques originally developed for DyGFormer.  
- In addition, from Equation 6, the recent interaction number k is selected as a very small value, such as 10 or 30. Given this, is the Mamba block necessary in this context? Since Mamba primarily enhances efficiency, its advantages may not be realized with such short sequences. I argue that the Mamba structure is unnecessary in Section 3.2. And it contradicts the motivation of the paper, suggesting that **the Mamba structure may not necessary to your design**.




2. **Reference issues:** 
 - In line 176, from the description of the time encoding function, it is not the one from TGAT [1], but from GraphMixer [2]. 
 - In line 180, the Node Interaction Frequency Encoding is followed FreeDyG [3], which should be Yu et al. [4], as it closely aligns with Yu's neighbor co-occurrence scheme. Specifically, the only difference between your description and DyGFormer’s neighbor co-occurrence scheme is the appending of the self-node to the last position in the sequence, which DyGFormer also implements.



3. **Some key content illustrations are vague and hard to follow.** 
- In Section 3.2, lines 250–254, the part regarding "enabling batch processing" is difficult to follow. Specifically, how was the number $10^{10}$ derived? Is it accurate that using only 10 neighbors could result in such a large figure?
- The overall explanation of "Dynamic Information Selection" is also unclear. It’s not evident why this particular design of Eq.7 was chosen or what advantages it offers. Additionally, it’s unclear how this design effectively achieves selection.
- In the "Ablation Study," the design of Variant B is unclear, which is described as removing the Mamba SSM layers in Equation 4. Does this mean completely removing the layers, or does it involve replacing Mamba with Transformer? In my opinion, simply removing these layers lacks purpose, and replacing Mamba with Transformer would be more meaningful. 

4. **Lacking Experiments:** 
- From W1, it appears that the performance gains are entirely due to the "Dynamic Information Selection" mechanism. To thoroughly evaluate the impact of this design and the effectiveness of the time-difference encoding, a more detailed ablation study focused on this component would be beneficial. However, only the entirety of Section 3.2 is ablated as Variant A, which presents a rough experiment. To me, the only observation from Variant A is that the Mamba structure negatively affects DyGFormer’s performance.
- The paper lacks tuning key hyperparameters such as $\alpha$ and $\beta$ in time encoding, the $\gamma$ in Time-Level SSM Block. These hyperparameters are crucial for model performance, and optimal settings may vary across different datasets. Additionally, the reason behind the introduction of $\gamma$ is not discussed. 
- The model was only evaluated on seven datasets out of 13 datasets used in DyGFormer. Could be discussed more thoroughly about the statistics of data sparsity and give the reason why only seven datasets are used. This would improve the persuasion of the experiments part.
- Recently, a very related study, FreeDyG [3], has been proposed and might need to be compared in your experiments.


5. **There are no discussions regarding the limitations of this paper and possible solutions.**

**Conclusion:** \
This paper has a good example in the Introduction and is well-organized; however, the description of the key component in Section 3.2 is vague and difficult to follow. Additionally, the paper presents extensive experiments on efficiency comparisons, mainly attributed to the Mamba structure and patching techniques from the prior works. However, the effectiveness of the key component, "Dynamic Information Selection," is inadequately evaluated through experiments. \
Most importantly, the novelty of the work is low, and the design of the algorithm raises concerns, particularly regarding the use of Mamba. Such as the results from Variant A indicate that incorporating Mamba can negatively impact performance, and the use of mamba in very short sequences, as shown in Equation 6. Therefore, I think this paper is insufficient for ICLR.



**Reference**: \
[1] Inductive Representation Learning in Temporal Networks via Causal Anonymous Walks, ICLR, 2022. \
[2] Do We Really Need Complicated Model Architectures For Temporal Networks?, ICLR, 2023. \
[3] FreeDyG: Frequency Enhanced Continuous-Time Dynamic Graph Model for Link Prediction, ICLR, 2024. \
[4] Towards Better Dynamic Graph Learning: New Architecture and Unified Library, NIPS, 2023.

### Questions
1. In the experiment illustrated in Fig. 4, which evaluates the scalability with varying numbers of neighbors, the analysis is conducted using the Enron dataset. However, the Enron dataset has an average degree of 676, meaning that only a small number of nodes have more than 676 neighbors. Have you considered the implications of this situation, where only a few nodes exceed this average degree? Could you discuss how this may influence the results presented in Fig. 4?


2. In Figure 3a, why does the performance of DyGFormer decrease as the number of patches decreases, while the performance of DyGMamba increases? Could you provide some reasoning or discussion to explain this phenomenon?

3. DyGFormer is a general framework designed to address the Continuous-Time Dynamic Graph (CTDG) problem, employing mean pooling to merge final node representations. In contrast, DyGMamba utilizes a short time-difference sequence to parameterize a weighted pooling mechanism instead of mean pooling. Based on the results from Variant A, I believe that Section 3.2 contributes to the observed performance gains.
However, I have the following questions: 
- Is the Mamba structure necessary within the Time-Level SSM Block? From my perspective, both Mamba and Transformer can capture long-sequence dependencies, and Mamba has been shown to operate as a linear attention mechanism [2]. Therefore, the performance upper bound might be that of the Transformer itself or a Transformer with causal attention. **Could you explain why Mamba is used for such short sequences in Section 3.2? Additionally, please provide a comparison with the Transformer structure**. 
- Is time-difference encoding essential for the pooling strategy? After experimenting with your code and removing Equations (6a–7c) by setting $\beta=Softmax(Linear(H))$ in Eq. (7d) [1], I observed that the performance remains strong. This raises **the question of whether time-difference encoding is necessary in your design**. It seems likely that the performance gain results from weighted pooling versus mean pooling, rather than from time-difference encoded weighted pooling.
- To justify the inclusion of Mamba in Section 3.2, you should consider applying "Dynamic Information Selection" to DyGFormer. Possible comparisons could include: **1. DyGFormer with dynamic information selection; 2. DyGFormer with dynamic information selection using the Transformer structure instead;** 


4. As analyzed in W1, the introduction of Mamba appears to harm the performance of the DyGFormer structure, so further analysis of the reasons behind this would be helpful. More importantly, **beyond the efficiency gains from the Mamba structure and patching techniques, could you please clarify the contributions of DyGMamba?**



5. In Section 3.2, lines 250–254, the part regarding "enabling batch processing" is difficult to follow. It’s unclear how using only 10 neighbors could lead to such a large value. **Could you provide details on the batch processing—specifically, whether it follows the same batching process as DyGFormer?** Additionally, how was the figure $10^{10}$ derived?



6. The explanation of "Dynamic Information Selection" is generally unclear. There is no evidence about why the specific design in Equation 7 was chosen or what advantages it offers. Additionally, it’s not clear how this design effectively achieves selection. Could you provide a more thorough explanation of the intuition behind the Dynamic Information Selection mechanism and clarify **its benefits compared to other weighted pooling mechanisms?**


7. Furthermore, other references for the State space model on dynamic graph learning should be added, such as [3], and the state-space model's applications on graphs should be discussed, such as [4].

8. Finally, if the Mamba structure is not a key component of your design and the performance gains stem primarily from the pooling mechanism, demonstrating its effectiveness across other CTDG methods would be better for the paper. **Integrating "Dynamic Information Selection" with other methods**, such as FreeDyG, GraphMixer, and CTAN, to show performance improvements could clear the value of this component.


**References:** \
[1] Language Modeling with Gated Convolutional Networks, ICML, 2017. \
[2] Mamba: Linear-Time Sequence Modeling with Selective State Spaces, arxiv, 2023. \
[3] STG-Mamba: Spatial-Temporal Graph Learning via Selective State Space Model, arxiv, 2024. \
[4] Graph Mamba: Towards Learning on Graphs with State Space Models, KDD, 2024.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper studies the link prediction problem on continuous-time dynamic graphs. The main technique this paper used is the Mamba state space model (SSM).

### Strengths
S1. The introduction to the proposed method is detailed and useful for the readers to understand every module and architecture.
S2. The baseline methods used in this paper is state-of-the-art and recent.
S3. This paper includes 7 datasets with 3 different negative sampling strategies (NSS) which makes the experimental section comprehensive.

### Weaknesses
W1. Seems that the main body of the model is based on a mature model Mamba, e.g., Eqs. 4a-4d, and Eqs. 6a-6d. Looks like there are existing works using Mamba on graphs, e.g, [1], which undermines the novelty of the proposed method.

W2. Some of the model designs are not reasonable or questionable to me. I will detail them as follows.

W2.1 Lines 182-190 introduce the counting of interaction frequency. However, it is still pretty vague why the frequency features are formed like that in lines 184 and 185. E.g., why there are 5 rows in F_u^t, and why both F_u^t and F_v^t have 2 columns. In addition, why the encoding of 2 columns of F_u^t uses a shared MLP (in line 189)? In other words, why not use an MLP: from R^2 to R^{d_f}?

W2.2 Lines 191 and 192 mentioned that it can save computation resources. If I understood correctly, by such a patching, the time complexity to encode the X_u^t will increase because the matrix has more columns (features), but the SSMs module can be faster because there are fewer rows (in lines 198 and 199), right?

W2.3 The section named “Dynamic Information Selection with Temporal Patterns” (line 255) seems to be the core module of the proposed method, which is related to the “edge-specific temporal pattern”. However, I think the edge information has been included in the Node-Level SSM Block (Eq. 3). Then, why repeatedly use the edge information, and what is any rationale regarding this?

W3. Regarding the experimental results, looks like under the “random” negative sampling strategy (NSS) the proposed method performs very well, but under the historical and inductive NSS setting, the proposed method can only beat ~ half of the SOTAs. 

W4. The writing of this paper is somewhat sloppy. I detail them as follows.
W4.1 As some modules of this paper are modified from the standard S4 and Mamba SSM, I checked the dimension and introduction for the backbone model between lines 125 and 140. However, a lot of them are confusing. (i) In Eq 1a, do matrices Az(\tau) and Bq(\tau) with the same dimensions? Looks like Bq(\tau) is of the shape 1 x d1 but the matrix A is of the shape d1 x d1, so the matrix Az(\tau) is of the shape d1 x (?), which cannot have the same shape as Bq(\tau). The same problem to r(\tau), which is a scalar, but from the second eq in Eq 1a, r(\tau)=Cz(\tau), and the shape of the matrix C is d1 x 1; in other words, it is not clear whether r(\tau) is a scalar or vector.

W4.2 In line 136, I think d2 was not mentioned in previous content. Also, why is the model in a SISO fashion when d_2>1? I am not familiar with this backbone model, but this statement is not intuitive to me.

W4.3 The end of Line 161 should be |Nei_u^t| but not Nei_u^t.

[1] Behrouz, Ali, and Farnoosh Hashemi. "Graph mamba: Towards learning on graphs with state space models." Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2024.

### Questions
Please check the questions mentioned in the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes a Mamba model for representation learning on continuous-time dynamic graphs. It has a node-level SSM to encode node interactions over time and a time-level SSM to encode edge-specific temporal information. Both representations are interleaved for dynamic information selection.

### Strengths
This is arguably the first Mamba model for dynamic graph representation learning, and I think the long context modeling ability of Mamba is suitable for temporal graph learning. I appreciate the designs of two types of SSM blocks that consider both node-level and edge-level information, both of which encode critical information about temporal patterns. The proposed DyGMamba has a satisfactory performance by outperforming baseline methods with higher accuracy in link prediction, shorter training time (per epoch and in total), and less memory usage compared to DyGFormer.

### Weaknesses
The overall design of DyGMamba makes sense to me, as it basically employs SSM block for node features and edge features. I have a few questions regarding the experiments in this paper.

* All datasets used in this paper do not have node features. Based on TGAT, I assume the authors are using all-zero vectors as node features. I wonder how DyGMamba would perform when there are node features, e.g., on GDELT dataset.

* Meanwhile, ablation study in Tables 3 and 4 are interesting. As you shift from transductive setting to inductive setting, Variant A has slightly worse performance compared to Variant B consistently on LastFM, Enron, and MOOC. Could the authors offer theoretical insights on why this happens? I do not see any specific patterns on these 3 datasets from the statistics in Table 6. Is there any specific design of node-level SSM or edge-level SSM would cause this 
performance change? If yes, is there any insights that could offer to users when deploying this model to other data?

### Questions
See weaknesses part.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper proposes a novel continuous-time dynamic graphs (CTDGs) representation model named DyGMamba to address the challenges of learning long-term temporal dependencies. DyGMemba consists of a node-level state space model (SSM) and edge-level SSM, which aims to learn the node-level and edge-level representations, respectively. The outputs of node-level SSM and edge-level SSM are combined for future link prediction. The authors conduct extensive experiments to evaluate DyGMamba.

### Strengths
1. The experiments are extensive.

### Weaknesses
1. The novelty of this work is low. Some important modules are highly similar to existing works. In specific: 
   - In Section 3.1,  The "Encode Neighbor Features" and " Patching Neighbors" sections are highly similar to DyGFormer [1]. 
   - The main network architecture is taken from original Mamba [2], without incorporating structure information of dynamic graph (see Eq. 5, 6 and 7).  

2. The authors claim that applying Mamba for dynamic graph learning is to address the challenge of computation complexity in capturing very long-term temporal dependencies. This raises several concerns: 

   - Why should we learn from very long history? Intuitively, very long history should be discarded since it has minor impact on current event. In addition, in Fig 3 (a), the performance of DyGFormer and variant A is not increasing as sequence length increases. This supports that the model performance does not necessarily increase as sequence length increases.

   - In addition, I think DyGFormer has address the problem of learning long-term dependencies by its patching technique. It can learn very long history if its patch size is large enough. 

3. What is "critical temporal information" mentioned in Line 54? How to define it? Is there evidence that DyGMamba can indeed capture this?  

4. The performance improvement of DyGMamba is marginal (Table 1 and 2). Most improvements are within 0.5\%.
 
5. In Fig.3, why the experiments are only conducted on Enron? More datasets should be included.  

[1] Yu L, Sun L, Du B, et al. Towards better dynamic graph learning: New architecture and unified library[J]. Advances in Neural Information Processing Systems, 2023, 36: 67686-67700. 

[2] Gu A, Dao T. Mamba: Linear-time sequence modeling with selective state spaces[J]. arXiv preprint arXiv:2312.00752, 2023.

### Questions
See the weakness.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper studies how the Mamba model can solve the continuous-time dynamic graphs, especially focusing on the dynamic link prediction problem. The research problem is interesting, and investigating the new neural architecture for graph representation learning is exciting. A detailed review of the pros and cons can be seen in the following sections.

### Strengths
1. The research problem is interesting, and the link prediction is a pragmatic and important application.

2. The study has the potential to solve the dynamic representation learning and applications from the Mamba perspective.

3. The paper's organization is not hard to follow.

### Weaknesses
1. [Minor] The preliminary section is not informative. In a better way, the preliminary should pave the way for illustrating the proposed method. So far, the illustration of Mamba is not clear, and the connection of the illustration with the proposed method is not well aligned. Also, the dimension in Eq (1a) seems inconsistent for the matrix computation.

2. For Eq. (4a) the computation procedure for $Broadcast\_{4d}$ is missing. Also, in Eq (4d), the computation procedure for $SSM\_{A,B,C}$ is also missing. Only the sentence "similar to Eq.2" is inadequate, for example, what is the relation between $H_{\theta}^{t}$ with $p_{\tau}$ and $q_{\tau}$?

3. The motivation and the necessity of using Mamba for dynamic graph learning are not that clear. Section 3.2 and Section 3.3 pay much attention to introducing the equation. However, the motivation and intuition for those equations are missing. For example, for the proposed "Dynamic Information Selection with Temporal Patterns", what is being selected and what is being filtered? Correspondingly, the ablation study design is reasonable, but the explanation is not adequate.

4. The experimental design follows the SOTA, i.e., DyGformer, which makes it good and easy to evaluate the performance of the proposed DyGMaba. Why are the selected datasets and sampling methods truncated from the full version of DyGformer? For example, 6 of 13 given datasets are not included, and one sampling method is not fully explored.

5. Between lines 526 to 529, more derivation for the complexity conclusion will be appreciated.

6. Overall, the theoretical contribution seems incremental.

### Questions
Please consider the raised concerns in the above section.

### Soundness
2

### Presentation
2

### Contribution
2
