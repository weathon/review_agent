# Beyond DAGs: A Latent Partial Causal Model for Multimodal Learning

- Decision: Reject
- Scores: 5, 6, 6, 6

## Abstract
Directed acyclic graphs (DAGs) are often assumed in causal discovery, however, accurately identifying these DAGs necessitates various assumptions, particularly in latent causal models, which can be challenging to validate in real-world applications. This raises a critical question: Are DAG assumptions truly necessary for certain applications? In this work, we introduce a novel latent partial causal model for multimodal data, which features two latent coupled variables, connected by an undirected edge, effectively representing transferable knowledge across different modalities. We focus on a prominent learning framework, e.g., multimodal contrastive learning, and demonstrate that, with certain statistical assumptions, multimodal contrastive learning successfully identifies the latent coupled variables up to trivial transformation. This finding enhances our understanding of the mechanisms driving the success of multimodal contrastive learning. Furthermore, this finding reveals a unique potential for disentanglement in multimodal contrastive representation learning, improving the utility of pre-trained models like CLIP that are trained using this approach. Through experiments with synthetic data, we demonstrate the robustness of our findings, even in the presence of violated assumptions. In addition, we validate the disentanglement capabilities of pre-trained CLIP in learning disentangled representations, facilitating few-shot learning and improving domain generalization across a diverse range of real-world datasets.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
Post rebuttal: I appreciate the author's rebuttal and discussion, which resolve some of my main concerns. I am raising score to 5 but lower my confidence score, because (1) I still think the paper is written in a very confusing way, which is very easy to distract the reader to access the main idea; (2) the theoretical results depend on the b of exact modality matching, which is impossible in practice. Therefore, I am not sure how significant the results are.

==================

This paper questions the DAG structure in multi-modal contrastive learning in real world problems, and propose a latent partial causal model for multimodal data. The partial causal structure can be more representative for latent representation relation modeling. The paper derives several theoretical results to help understanding the multi-modal contrastive loss, and validate the disentaglement capabilities of pretrained CLIP in learning disentangled representations.

### Strengths
1. The idea of latent partial causal modeling in multi-modal contrastive learning is interesting.
2. Some theoretical results on the multi-modal contrastive loss can help one better understand the underlying mechanism of multi-modal contrastive learning.

### Weaknesses
While the problem studied in the paper is interesting, I find the paper very hard to follow. 

In my understanding, the paper proposes the latent partial causal structure in multi-modal contrastive learning. I thought there must be text to explain why this architecture is good, how to design algorithm algorithm to optimize the model and why properties this architecture could have. Unfortunately, I cannot catch what the paper is presenting. More specifically, Section 3.1 talks about the latent partial causal structure, then Section 3.2 directly analyze the standard multi-modal contrastive learning objective used in for example CLIP. But how is the objective related to your model? I believe the latent partial causal model should have a different objective because it has extra latent variables. Also, in Section 4, Theorem 4.1, I believe h is the composition of f_x and g_x, but why it only takes z_x (or z_t) as input? Where is m_x? And I did not follow the insights explained for Theorem 4.1. My concerns apply to Section 4.2.

Finally, I am not sure what the experiments try to tell us. There is not enough information to explain the results. For example, Table 1 tries to evaluate the identificabiulity of the model, but how? and how do you calculate the recovered \hat{z}_x? For the rest of the experiments, I think they are quite toy experiments, and I am not convinced about the advantage of the proposed method from the reported results.

### Questions
See the weakness part.

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
3

### Summary
The paper extends the identifiability results from previous work [1] to the case of multimodal contrastive learning. To validate their theoretical results, the authors show that shared information between modalities can be recovered from the representations obtained with a multimodal contrastive learning objective.  In addition, they demonstrate that their results are satisfied empirically even when certain assumptions are violated.  Finally, from their theoretical insights, the authors postulate that CLIP, which is trained with a multimodal contrastive learning objective, produces representations that can be well-disentangled in separate factors. This last claim is validated by applying ICA on top of the CLIP embeddings and showing positive results for few-shot classification and domain generalization.  

[1] Zimmermann et al. Contrastive Learning Inverts the Data Generating Process, ICML, 2021.

### Strengths
- To my understanding, the paper presents a solid theoretical analysis for identifiability in multimodal contrastive learning. 
- Empirical results validate the authors' theoretical insights
- The insight that ICA can improve the performance for robust few-shot learning with CLIP representations is useful and interesting

### Weaknesses
- The paper is strongly based on the results presented in [1] and provides similar insights as previous works [2,3], which limits the significance and novelty of the presented results.
- While "going beyond DAGs" is central in the title, to me it does not seem like the paper focuses on possible advantages of "non-DAG assumptions". To my understanding, the paper extends the identifiability results for multi-view contrastive learning from [1] to the multimodal contrastive learning case. Could the authors better clarify the centrality of the non-DAG assumption and why it is necessary for the insights in this work? 
-  The authors state "Informed by our identifiability results and the empirical evidence presented earlier, we have grounds to claim that the pre-trained CLIP model possesses disentanglement ability." I do not immediately understand what is the rationale behind this claim. Given the presented insights, one can claim that shared information can be disentangled from modality-specific information in the latent space, but why should individual factors (i.e. individual attributes in celebA) be disentangled?



[1] Zimmermann et al. Contrastive Learning Inverts the Data Generating Process, ICML, 2021.
[2] Daunhawer et al. Identifiability results for multimodal contrastive learning. arXiv preprint arXiv:2303.09166, 2023.
[3] Yao et al. Multi-view causal representation learning with partial observability. arXiv preprint arXiv:2311.04056, 2023.

### Questions
See Weaknesses

### Soundness
3

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
3

### Summary
The paper introduces a latent partial causal model for multimodal data that captures the undirected relationships between the two latent coupled variables corresponding to each modality. Further, the author provides identifiability guarantees for the commonly used multimodal contrastive learning loss and demonstrates its potential for disentanglement with CLIP embedding on real-world images.

### Strengths
- The paper focuses on multimodal learning problems and provides a novel perspective on the latent partial causal model setting.
- The paper provides identifiability analysis for the provided causal model on both hypersphere and convex body, and demonstrates its disentanglement implication.
- Empirical results on a synthetic setting and CLIP embeddings support the theoretical results.

### Weaknesses
- Section 3.2 about prior matching and information preservation shares a lot in common with many concepts being explored in self-supervised learning at the high level, for example, the alignment-uniformity perspective in [1], and the mutual information perspective in [2] (given that mutual information can be decomposed into the difference between an entropy and a conditional entropy term, corresponding to matching and information preservation). The paper lacks discussions/comparisons with these perspectives.
- The condition in Theorem 4.1, $\mathbf{f}_x \circ \mathbf{g}_x=\mathbf{f}_t \circ \mathbf{g}_t$, looks strong. It could be hard to achieve in real-world data.
- Compared to unimodal representations from SSL methods and the analysis in [3], it is unclear what specific gain can be attained from the multimodal representations.

[1] Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere.

[2] Representation Learning with Contrastive Predictive Coding.

[3] Contrastive Learning Inverts the Data Generating Process.

### Questions
- In Equation 5 of Theorem 4.1, is $z_t$ in $\mathbb{E}_{z_x\sim p(z_x)} [H(p(z_t|z_x),q_h(z_t|z_x))]$ the one corresponds to the current $z_x$, or it is from a random distribution $p(z_t)$?
- Can the authors give some intuition on how the modality-specific variables $m_x$ and $m_t$ are got rid of in Theorem 4.1 and how different  $m_x$ and $m_t$ would affect the results? Intuitively, when $m_x$ and $m_t$ dominate, it is harder to identify the latent partial causal variables.
- What is the reasoning behind the conclusion that Theorem 4.1 implies the potential for disentanglement of pretrained model embeddings?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a new model for understanding how different types of data (e.g., text and images) relate to each other, moving away from the usual directed acyclic graph (DAG) approach. Instead, it uses a structure with latent variables connected by undirected edges, which captures shared information between data types under different scenarios. The paper studies the identifiability conditions of these models. It also shows that this approach helps disentangle important features, making it possible to improve the use of pre-trained models like CLIP for tasks such as learning with limited examples and adapting to new domains. The method is empirically tested on various real-world datasets.

### Strengths
**Originality and significance.** This paper focuses on proposing a more generalized generative framework for studying multimodal settings, which is an interesting and important problem both from a theoretical and empirical perspective. It is clear how this work differs from previous contributions, and the analysis shows new insights. 

**Quality.** The paper is well written and the claims are mostly well-substantiated with either proofs or experiments. The authors include an “Insights” section following nearly every theorem, which greatly helps in understanding and building insights.

**Clarity.** The paper is well-organized and clear, with understandable figures and tables. 

Table 1 presents interesting results about how the robustness of the results, even in the absence of the assumptions made in the paper

### Weaknesses
- Although the authors show three different scenarios of the possible DAGs in Figure 2,  there are fundamental issues in this modeling. There exists a real world that we measure with various sensors, such as the camera (generating images from various views) or textual descriptions of the scene. Hence, for two modalities, each observation requires two types of latent representations: modality-specific representations ($m_x$ and $m_t$) that hold information exclusive to each modality, and a shared representation ($z=z_x=z_t$) that captures common information across both modalities. This eliminates the need to connect $z_x$ and $z_t$ through an undirected edge. Further, this idea represents the DAG shown in Figure 2(a). While the authors justify DAGs shown in Figure 2(b) and Figure 2(c) with examples of text-to-image retrieval and image captioning, these tasks can also be captured as inference on the first DAG i.e. Figure 2(a). 

-   Theorem 3.1 is interesting and provides insights into the CLIP objective in an infinite sample setting. However, as also mentioned in the paper, this seems to be a direct extension of Theorem 1 to the multimodal setting in Wang and Isola (2020). I am unsure if it adds a lot of value.

- The claims made regarding the disentanglement capabilities of CLIP seem to be unfounded. Firstly, the authors base their assertions on empirical analysis using only a single dataset, which is insufficient to substantiate claims of disentanglement. Further, the authors state that“the objective of disentangled representations is to learn features that lend themselves to be easily and robustly transferred to downstream tasks”. However, this is more of a consequence of achieving disentangled representations rather than an actual objective. The paper does not provide a clear or rigorous definition of disentanglement (which in itself is an involved question). Moreover, the baselines used for the disentanglement experiments are outdated (from 2018), while there are more recent works in this field. 

1. 
https://proceedings.neurips.cc/paper_files/paper/2023/hash/6818dcc65fdf3cbd4b05770fb957803e-Abstract-Conference.html

2. 
https://proceedings.neurips.cc/paper_files/paper/2023/hash/93e98ddf39a9beb0a97fbbe56a986c80-Abstract-Conference.html

Specifically, the first paper even shows the suboptimality of existing multimodal representation learning approaches such as CLIP. Overall, it appears that the authors use the notion of disentanglement to reinforce their theoretical findings, but their references to it are vague and unsupported. The experimental results presented do not imply “true” disentanglement.

- Minor: Figures 1 and 2 can be placed early on in the text. They are referred to in the introduction. It would be easier to have a main figure that explains the broad concept.

### Questions
Please see the weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
2
