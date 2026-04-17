# Healthcare Insurance Fraud Detection via Continual Fiedler Vector Graph Model

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Healthcare insurance fraud detection presents unique machine learning challenges: labeled data are scarce due to delayed verification processes, and fraudulent behaviors evolve rapidly, often manifesting in complex, graph-structured interactions. Existing methods struggle in such settings. Pretraining routines typically overlook structural anomalies under limited supervision, while online models often fail to adapt to changing fraud patterns without labeled updates.
To address these issues, we propose the Continual Fiedler Vector Graph model (ConFVG), a fraud detection framework designed for label-scarce and non-stationary environments. The framework comprises two key components. To mitigate label scarcity, we develop a Fiedler Vector-guided graph autoencoder that leverages spectral graph properties to learn structure-aware node representations. The Fiedler Vector, derived from the second smallest eigenvalue of the graph Laplacian, captures global topological signals such as community boundaries and connectivity bottlenecks, which are patterns frequently associated with collusive fraud. This enables the model to identify structurally anomalous nodes without relying on labels. To handle evolving graph streams, we propose a Subgraph Attention Fusion (SAF) module that constructs neighborhood subgraphs and applies attention-based reweighting to emphasize emerging high-risk structures. This design allows the model to adapt to new fraud patterns in real time. A Mean Teacher mechanism further stabilizes online updates and prevents forgetting of previously acquired knowledge.
Experiments on real-world medical fraud datasets demonstrate that the Continual Fiedler Vector Graph model outperforms state-of-the-art baselines in both low-label and distribution-shift scenarios, offering a scalable and structure-sensitive solution for real-time fraud detection.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles an important fraud detection problem characterized by limited label availability during pretraining and dynamically evolving fraud patterns. To address these challenges, it introduces a two-stage framework: (1) an unsupervised pretraining module that incorporates spectral information into a masked graph autoencoder, enabling node representations to capture global topological irregularities indicative of potential collusive behavior; and (2) an online adaptation module that constructs complementary subgraphs, integrates their embeddings through an attention-based fusion mechanism, and stabilizes unsupervised updates using a mean-teacher consistency objective. Extensive experiments demonstrate the effectiveness of the proposed approach.

### Strengths
1. The problem addressed in this paper is both novel and significant, with strong practical implications for real-world applications. 
2. Unlike conventional masked graph autoencoders that rely on random masking, this paper introduces a Fiedler vector-guided masking strategy, enabling the model to effectively capture global connectivity patterns. 
3. Comprehensive evaluations are conducted across multiple datasets and under various scenarios—including low-label settings and temporal distribution shifts—to thoroughly validate the proposed approach.

### Weaknesses
1. The design of the proposed method heavily relies on the core assumption that "connected nodes share similar smoothness values." However, it does not provide relevant statistical evidence on the medical insurance dataset to validate this assumption. Furthermore, the paper lacks an evaluation of robustness against adversarial fraud. In realistic attack scenarios, fraudsters may deliberately connect with normal users to conceal malicious activities, which would lead to significant attribute discrepancies between connected nodes—directly contradicting the fundamental assumption of the method.
2. The paper does not include a sensitivity analysis of its hyperparameters, such as the perturbation magnitude and the attention regularization coefficient.
3. The paper requires strengthening in its mathematical notation, as there are multiple instances where symbols are ambiguous. For example, the symbol $X_{ij}$ denotes a node representation in Line 192, whereas $x_i$ refers to a smoothness value in Line 213.
4. The paper lacks sufficient implementation details, such as the choice of optimizer and the learning rate schedule.

### Questions
See weaknesses.

### Soundness
2

### Presentation
1

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
The manuscript proposes ConFVG, a Fiedler vector–guided graph learning framework for medical fraud detection. By integrating spectral topology signals into pretraining and using attention-based online adaptation, it effectively handles label scarcity and evolving fraud patterns. Compared with POCL, ConFVG advances from contrastive learning to structure-aware self-supervision, achieving higher adaptability and stability in low-label settings.

### Strengths
- The paper addresses a realistic and practically important problem in medical insurance fraud detection. Its two-stage framework—pretraining followed by online learning—suggests strong potential for real-world application.

- Compared with the latest and most relevant baseline POCL, ConFVG introduces the Fiedler vector to alleviate data sparsity and adds a subgraph attention–based online adaptation module, showing clear methodological innovation.

- Experimental results demonstrate that ConFVG achieves the most stable and robust performance over continuous time periods, confirming its superior resilience for dynamic fraud detection tasks.

### Weaknesses
- The challenges outlined for medical insurance fraud detection appear too generalized and are not empirically verified, relying mostly on conceptual descriptions. While label sparsity is indeed a common issue across many domains, the claim of “adaptation deficiency” remains unclear and weakly justified. As a reviewer experienced in fraud detection research, I find these arguments repetitive and somewhat unoriginal, diminishing the paper’s novelty.

- The motivation for the model design lacks clarity—particularly, it is not well explained why the Fiedler vector can alleviate data sparsity, even though experiments suggest it works.

- The introduction of graph perturbation to ensure weak connectivity for computing the Fiedler vector seems somewhat ad hoc and theoretically awkward. While the paper provides an upper bound to justify this step, such a perturbation could distort graph structure, and this critical design choice deserves deeper discussion and validation.

### Questions
See above, thanks.

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
This paper introduces the Continual Fiedler Vector Graph (ConFVG) model, a novel framework for healthcare insurance fraud detection. The model is specifically designed to tackle two prevalent challenges in this domain: the scarcity of labeled data and the dynamic, evolving nature of fraudulent activities (non-stationary environments). The core of ConFVG consists of two main components. First, a Fiedler Vector-guided graph autoencoder is proposed for the pre-training stage. It leverages spectral graph properties—specifically the Fiedler vector derived from the graph Laplacian—to learn structural-aware node representations without relying heavily on labels. This helps in identifying anomalous graph structures often associated with collusive fraud. Second, for the online learning phase, the paper introduces a Subgraph Attention Fusion (SAF) module. This module, combined with a Mean Teacher mechanism, allows the model to adapt to new fraud patterns in data streams in an unsupervised manner, by focusing on emerging high-risk subgraphs and ensuring stable knowledge updates.

### Strengths
- The paper tackles a significant and practical problem in fraud detection: handling label scarcity and concept drift simultaneously. The proposed solution, combining spectral graph theory (Fiedler vector) for unsupervised representation learning with a continual learning framework (subgraph attention and mean teacher) for adaptation, is a novel and clever integration of ideas from different domains.

- The experimental results are comprehensive and convincing. The authors demonstrate that ConFVG consistently outperforms a wide range of state-of-the-art baselines, including both offline and online models, across multiple datasets (Medical, YelpChi, Amazon) and under challenging low-label rate scenarios (1% and 10%). The performance improvement is substantial, which highlights the effectiveness of the proposed components.

- The method is well-grounded in theory. The use of the Fiedler vector to capture global topological anomalies like community boundaries and bottlenecks is theoretically sound for identifying collusive fraud. The design of the Subgraph Attention Fusion module to handle evolving graph structures is intuitive and addresses a key limitation of traditional GNNs that focus only on existing connections.

### Weaknesses
- The core of the pre-training phase relies on calculating the Fiedler vector, which involves an eigendecomposition of the graph Laplacian matrix (𝐿). This operation can be computationally prohibitive for very large graphs, with a complexity that can approach 𝑂(𝑁3) where 
𝑁  is the number of nodes. While the paper demonstrates strong results on the given datasets, it does not address how this approach would scale to real-world graphs containing millions or billions of nodes.

- The online learning phase introduces a Subgraph Attention Fusion (SAF) module that unconventionally operates on the complement of the graph's largest components. The paper lacks a clear and deep intuition for this design choice. While traditional GNNs learn by aggregating information from existing connections (neighbors), it is not immediately obvious how aggregating features from non-connected nodes helps the model adapt to new and emerging fraud patterns.

- The method introduces a critical hyperparameter 𝜖  in Equation 4 to connect disparate graph components. The choice of this value is crucial, as too small a value may not enforce connectivity, while too large a value could introduce significant noise and distort the graph's original spectral properties, which are fundamental to the Fiedler vector's utility. The paper does not provide a sensitivity analysis for 
𝜖, making it difficult to assess the model's robustness.

### Questions
- Regarding the Fiedler vector calculation: Given the scalability concerns with eigendecomposition, did you explore or can you comment on the feasibility of using approximation algorithms for the Fiedler vector on much larger graphs? How might this affect the quality of the learned representations?

- In the online phase, the Subgraph Attention Fusion (SAF) module operates on a graph complement created from the "top-k largest components." Could you elaborate on the intuition behind using the complement? Traditional attention mechanisms in GNNs focus on aggregating information from neighbors (existing connections). How does incorporating information from non-neighbors via the complement graph specifically help in adapting to new fraud patterns?

- How was the hyperparameter 𝜖 for the graph perturbation (Eq. 4) selected? Could you share results from a sensitivity analysis showing how different values of 𝜖  impact the final performance?

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
3

### Summary
The authors propose the Continual Fiedler Vector Graph model (ConFVG), a framework designed to address two challenges in healthcare insurance fraud detection: label scarcity and non-stationary environments (i.e., evolving fraud patterns). The model consists of two key components. First, to handle label scarcity, it employs a Fiedler Vector-guided graph autoencoder for pretraining. This uses the Fiedler vector, derived from the graph Laplacian, to identify global topological structures like community bottlenecks and guide the autoencoder's learning process without relying on labels. Second, to adapt to evolving graph streams in an unsupervised online setting, it introduces a Subgraph Attention Fusion (SAF) module. This module constructs and fuses neighborhood subgraphs with their complements, using attention to emphasize emerging high-risk structures. A Mean Teacher mechanism is used to stabilize these online updates and prevent catastrophic forgetting. Experiments on real-world medical fraud datasets demonstrate that the approach outperforms baselines in both low-label and distribution-shift scenarios.

### Strengths
There are a few things I like about the paper:
1. The paper tackles a realistic real-world problem setup: pre-training with scarce labels and performing online learning.
2. The use of the Fiedler vector to guide a graph autoencoder in self-supervised pretraining is interesting.
3. The Subgraph Attention Fusion (SAF) module is an interesting mechanism for online adaptation. The idea is to complement the original graph with a complement graph to allow the model to capture evolving patterns that might not be visible from a single connected component's perspective.
4. The experimental results on multiple datasets support the paper's claims.
5. The authors performed an ablation study to show that the components described in the paper are relevant to the final prediction.

### Weaknesses
1. [Motivation]. The problem of label scarcity and non-stationary graph streams are actually not unique to health care insurance data. These two problems exist in most real-world fraud detection systems. I suggest the authors reframe the motivation to be more generic.
2. [Scalability and Complexity] I suggest the author provide discussion on the scalability and complexity of the proposed models. Particularly since the model is more complex than the baselines.
3. [Hyperparameter Sensitivity] The model introduces several new hyperparameters. Could the authors discuss more on sensitivity of the parameters vs results?

### Questions
Please address the weaknesses mentioned above.

### Soundness
3

### Presentation
3

### Contribution
3
