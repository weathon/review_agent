# PID-Guided Partial Alignment for Multimodal Decentralized Federated Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 6

## Abstract
Multimodal decentralized federated learning (DFL) is challenging because agents differ in available modalities and model architectures, yet must collaborate over peer-to-peer (P2P) networks without a central coordinator. Standard multimodal pipelines learn a single shared embedding across all modalities. In DFL, such a monolithic representation induces gradient misalignment between uni- and multimodal agents; as a result, it suppresses heterogeneous sharing and cross-modal interaction. We present PARSE, a multimodal DFL framework that *operationalizes* partial information decomposition (PID) in a server-free setting. Each agent performs *feature fission* to factorize its latent representation into *redundant*, *unique*, and *synergistic* slices. P2P knowledge sharing among heterogeneous agents is enabled by slice-level *partial alignment*: only semantically shareable branches are exchanged among agents that possess the corresponding modality. By removing the need for central coordination and gradient surgery, PARSE resolves uni-/multimodal gradient conflicts, thereby overcoming the multimodal DFL dilemma while remaining compatible with standard DFL constraints. Across benchmarks and agent mixes, PARSE yields consistent gains over task-, modality-, and hybrid-sharing DFL baselines. Ablations on fusion operators and split ratios, together with qualitative visualizations, further demonstrate the efficiency and robustness of the proposed design.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes the PARSE framework for decentralized multimodal FL scenarios, aiming to address the challenge that unimodal and multimodal clients have inconsistent gradient directions when updating shared parameters. To address this, the paper leverages Partial Information Decomposition (PID) to split features into redundant, unique, and synergistic slices. Due to client heterogeneity, slice-level knowledge sharing is performed only when they share alignable slices. The benefit of this approach is that it can prevent interference from irrelevant parts of the model. The paper compares the proposed method with 6 baselines on 4 datasets, and the results demonstrate that the proposed method outperforms baseline methods under various settings.

### Strengths
This paper uses information decomposition to address modality heterogeneity among clients, which is an interesting and intuitive approach. The experimental section appears clear, including various datasets, modality heterogeneity settings, class heterogeneity settings, different topologies, ablation studies, etc.

### Weaknesses
1. In my opinion, the biggest flaw of the proposed method is that it is impractical, in other words, no one would use this method to train models in the real world. The reason is that the method proposed in this paper modifies the model structure rather than being a pure FL algorithm. In contrast, almost all baselines compared in the paper are FL algorithms that are general to all models. This leads to the following consequences:
    - It cannot be adapted to any currently popular models without modifying their architectures. For example, if I want to fine-tune GPT-5 while keeping its architecture unchanged, the proposed method is not applicable.

    - More critically, even if I modify GPT-5 to add the three-slice structure, I cannot simply freeze the pretrained encoder and only train the classifiers. This is because the feature splitting (from the authors' code)
      ```python
      feat = self.encoders[i](x)
      feat_m, feat_c, feat_s = torch.chunk(feat, 3, dim=1)
      ```
      is completely meaningless if the encoder is frozen. the pretrained encoder was never trained to semantically separate its output into redundant, unique, and synergistic components. Therefore, the method requires training the entire model (encoder + classifiers) from scratch with the special three-slice architecture.
    
    - It cannot be proved that models with the special design of redundant, unique, and synergistic slices are better than models without these special designs. 
   
   - All experimental comparisons in the paper are: three-slice special architecture vs. standard architecture. This is not a fair comparison. FL algorithm vs. FL algorithm is acceptable, FL algorithm + special architecture vs. FL algorithm + special architecture is acceptable, but FL algorithm + special architecture vs. FL algorithm is not acceptable.

2. I believe the proposed method does not truly contribute to the DFL field, for the following reasons:
    - It does not solve the unique challenges in DFL, for example, asynchronous updates, dynamic topology, etc.
   
    - Many multimodal FL algorithms can be very easily adapted to DFL scenarios, such as the baselines in the paper: DSGD-Modality, DSGD-Task, and DSGD-Hybrid. I believe that at the code level, only about 20 lines of code need to be modified to adapt the CFL aggregation paradigm to the DFL neighbor averaging. Obviously, I would not call such modifications a contribution to the DFL field, and therefore the same logic applies to this paper.
   
    - I have to say that many experimental scenarios in the paper use the simplest and most basic ring topology. Although the paper also compares chordal ring and random gossip, these are all variants of the ring topology. This obviously cannot represent DFL. What about other topological structures, such as fully-connected structures? What about dynamically changing topologies? What about highly heterogeneous connectivity (some nodes have many connections, some have few)?
   
    - The paper's contributions emphasize server-free, but I do not know how to construct per-modality subgraphs in the real world without server coordination. How do clients know which neighbors have the same modality?
   
    - The paper's contributions emphasize topology-agnostic, but I do not know how the proposed method works under a fully-connected topology, i.e., a topology where all clients are connected regardless of modality?

### Questions
1. I suggest systematizing the experiments on modality heterogeneity, i.e., the agent ratios in the paper, by using a configurable parameter to represent, for example, the proportion of clients with all modalities. Additionally, why are the numbers of single-modality clients the same in all experiments? If the numbers of single-modality clients are different, for example, 5 audio clients and 20 video clients, how would the performance be, especially given that the paper constructs per-modality subgraphs?

2. I suggest testing on more topological structures, especially in cases where clients have more connections, which may better represent real-world P2P networks.

3. The paper mentions in its contributions that the proposed method is compatible with time-varying random graphs, but there are no experiments to prove this, especially in multimodal scenarios. Consider a scenario where some multimodal clients, for some reason, such as sensor damage or privacy policy changes, have some modalities no longer available. How should the proposed method handle this? Or consider, alternatively, single-modality clients who purchase new devices and collect new modality data. How should the proposed method handle this?

### Soundness
1

### Presentation
2

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
This paper explores the aggregation conflict challenges faced by different client-mode architectures in decentralized federated learning and proposes a server-free multimode decentralized federated learning framework named PARSE. The framework achieves peer-to-peer knowledge sharing between heterogeneous modal frameworks by partitioning the latent features of the data into three distinct slices: redundant, unique, and synergistic. This approach effectively resolves the collaboration issues among heterogeneous modal nodes. Extensive experimental results demonstrate the effectiveness of PARSE in practical applications.

### Strengths
1. PARSE employs a novel approach to knowledge sharing by partitioning data features into three slices. This method is quite intriguing, as it successfully facilitates knowledge sharing and transmission through the alignment of these slices.
2. The research focuses on the issue of modal heterogeneity among agents, a relatively new field that provides significant impetus for the advancement of multimodal federated learning.

### Weaknesses
1. This paper employs Partial Information Decomposition to partition features into three slices. Is there a theoretical explanation supporting its effectiveness in the multimodal domain? How is the specific feature partitioning process conducted?
2. The article mentions achieving knowledge sharing through feature fission, yet the specific design involves aggregating modules from the same modality model. For example, the optimization directions for single-modal and multi-modal clients sharing the same modality differ. How is this addressed when resolving gradient conflicts?
3. The expression of the method from a peer-to-peer perspective seems somewhat odd, as it does not reflect any special design for the peer-to-peer environment. Its aggregation design is similar to other methods, aggregating parameters of the same modality. In a peer-to-peer setting, the number of neighboring nodes for each client is typically limited, and the experimental details should briefly address this setup.

### Questions
1. How does the PID technique specifically partition data features into three slices?
2. The article needs to provide further explanation on how gradient conflicts are resolved for single-modality or multimodality.
3. More detailed clarification is needed regarding the improvements and settings in the peer-to-peer context.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces an interesting problem, multimodal decentralized federated learning in which different agents with various modalities need to collaborate without a central coordinator. To tackle this problem, the authors introduce PARSE, a new framework based on partial information decomposition (PID) theory to decompose modality-wise features into three components and perform selective alignment. The experimental results show that PARSE outperforms selected baselines in previous work.

### Strengths
* Good problem statement and clear motivation
* Good writing, easy to follow

### Weaknesses
* Novelty: The concept of modality decomposition, including PID-based variants, has been explored in prior work [1,2]. The authors should clearly articulate how PARSE advances beyond existing approaches and specify its distinctive contributions.
* Literature Review: The literature review should be broadened to encompass centralized multimodal learning methods or federated multimodal learning [1,2], not solely multimodal DFL. The authors are encouraged to discuss the challenges of applying centralized methods in distributed settings and to include additional baselines from these domains to substantiate PARSE’s robustness and design sophistication.
* Agent design: Assigning separate classifier heads to each decomposed feature is an unconventional choice. The rationale for using multiple classifiers within a single modality should be clarified, along with an explanation of how this architecture improves performance.
* Global collaboration: What is the difficulty of using DSGD with client design from federated multimodal learning ? How does PARSE handle this? The connection seems unclear to us.
* Main results: While the experiments span several benchmarks, all involve only a limited number of modalities. The current evidence is insufficient to demonstrate PARSE’s scalability as modality count increases (similar to [1])
* Ablation study: Given that each feature type has its own classifier, dropping certain features should not hinder inference. The authors should report PARSE’s performance when one or more feature types per modality are removed to assess robustness under partial feature availability.

[1] Nguyen et al., Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data, NeurIPS’25

[2] Liang et al., Quantifying & Modeling Multimodal Interactions: An Information Decomposition Framework, NeurIPS’23

### Questions
See Weaknesses

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
This paper introduces PARSE, a PID-guided feature decomposition and partial alignment framework for decentralized multimodal federated learning (DFL). The approach leverages partial information decomposition (PID) to factorize features into redundant, unique, and synergistic components, enabling selective peer-to-peer knowledge sharing without a central coordinator. Experiments across four public multimodal datasets demonstrate consistent performance improvements over representative baselines.

### Strengths
（1）Conceptual clarity and intuitiveness:
The proposed PID-based feature decomposition combined with partial alignment is conceptually simple yet elegant. It provides a clear and interpretable mechanism for handling cross-modal heterogeneity in decentralized settings.
（2）Strong presentation and experimental design:
The paper is well-written and well-structured, with comprehensive experiments and clear visualizations. The figures and tables effectively illustrate the advantages of the method, and the ablation studies offer solid insights into the model’s behavior and design choices.

### Weaknesses
（1）The method assumes that all agents solve the same underlying task (i.e., share the same label space). However, in many realistic multimodal decentralized scenarios, agents may work on related but distinct tasks. How would PARSE handle task heterogeneity? Would the PID-based decomposition and partial alignment still maintain consistent feature semantics across agents?
（2）The key designs—PID-based feature fission and slice-level alignment—could, in principle, also benefit centralized or server-based FL architectures. It would be valuable to clarify what aspects of PARSE are specifically tailored to the decentralized setting, beyond the lack of a coordinator. How does the framework uniquely address challenges such as gradient drift and topology variability in DFL?
（3）The notion of “synergy” is central to the method, yet its operational meaning in the reported experiments could be elaborated. In each dataset, what constitutes synergistic information in the DFL context? Which modalities contribute more to synergistic learning, and how is this reflected in the learned feature subspaces or cross-modal collaboration patterns?

### Questions
1)	Feature disentanglement is a well-established concept in conventional FL. The paper should clarify what is novel or specific about applying it in the decentralized FL scenario.
2)	The manuscript should specify the fusion strategy used for parameter exchange or aggregation among agents.

### Soundness
4

### Presentation
4

### Contribution
3
