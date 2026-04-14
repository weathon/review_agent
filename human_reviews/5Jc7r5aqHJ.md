# Energy-based Backdoor Defense Against Federated Graph Learning

- Decision: Accept (Oral)
- Scores: 8, 8, 8, 6

## Abstract
Federated Graph Learning is rapidly evolving as a privacy-preserving collaborative approach. However, backdoor attacks are increasingly undermining federated systems by injecting carefully designed triggers that lead to the model making incorrect predictions. Trigger structures and injection locations in Federated Graph Learning are more diverse, making traditional federated defense methods less effective. In our work, we propose an effective Federated Graph Backdoor Defense using Topological Graph Energy (FedTGE). At the local client level, it injects distribution knowledge into the local model, assigning low energy to benign samples and high energy to the constructed malicious substitutes, and selects benign clients through clustering. At the global server level, the energy elements uploaded by each client are treated as new nodes to construct a global energy graph for energy propagation, making the selected clients' energy elements more similar and further adjusting the aggregation weights. Our method can handle high data heterogeneity, does not require a validation dataset, and is effective under both small and large malicious proportions. Extensive results on various settings of federated graph scenarios under backdoor attacks validate the effectiveness of this approach.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The authors propose an energy-based backdoor defense for Federated graph learning. Experimental results on the node classification task under the mentioned backdoor attacks are provided, with comparisons to several defenses, and performance gain are observed in iid and non-iid scenarios.

### Strengths
1. The estimation of using energy distribution to recognize malicious clients seems to be novel.

2. Comparisons with some advanced defenses are provided and show the advantages.

### Weaknesses
1. Key details on the different energy distributions of malicious clients and others are not provided. In addition, the influence of this factor in the non-iid scenario needs further verification.

2. Some standard defenses are missing, such as FLAME， FL Trust, and G^2uard Fl. In addition, the authors should pay attention to their statements carefully, since the proposed work actually is not the first to conduct backdoor attacks in FGL. For example, "Distributed backdoor attacks on federated graph learning and certified defenses" published in CCS 2024 is a good work that should be compared.

3. It is strange that FedFreq has an unexpectedly low performance in the experiments which are not consistent in their own work, and the authors may clarify the reason.

### Questions
Besides the weakness, the authors may consider the following points to enhance the paper.

1. Does the proposed energy-based estimation work on traditional FL or other domains? 

2. Does the proposed scheme work for untargeted attacks?

3. Does the proposed scheme work for other types of graph learning tasks, such as graph classification and link prediction?

4. The verification attacking method needs more details, and other attacks should be further evaluated.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the issue of backdoor attacks in Federated Graph Learning and proposes a defense method based on Topological Graph Energy. The method injects structural distributional knowledge into the model. At the client level, it models energy to differentiate between benign and malicious samples. While at the server level, it constructs a global energy graph for energy propagation, effectively identifying and excluding malicious clients. Experimental results demonstrate that FedTGE performs well under various attack proportions and in heterogeneous scenarios.

### Strengths
- The paper is the first to introduce energy-based models into backdoor defense in FGL, effectively addressing the complexities of trigger injection locations and diverse attack forms by learning the energy structure distribution of the data through constructing an energy-based model, which leverages the unique topological characteristics of graph data.
- The methodology is well explained and structured. In particular, the authors provide a detailed step-by-step breakdown of their approach to help in understanding the core mechanisms of FedTGE. Furthermore, the framework diagram clearly explains the energy propagation and clustering processes, making it easier to grasp the interactions between different modules.
- By constructing an energy graph and propagating energy across clients, FedTGE overcomes the limitations of traditional federated defense methods struggle to adapt well of graph data. This approach combines structural knowledge to enable more fine-grained and accurate identification of malicious entities. Thus enhancing the robustness and adaptability of the defense system across diverse scenarios.

### Weaknesses
- Although the experiments demonstrate the superiority of FedTGE over traditional methods, the paper lacks sufficient discussion of existing backdoor defense techniques in traditional federated learning. For instance, RFA appears in the performance table but is not comparatively analyzed in the introduction. The authors are supposed to clarify the shortcomings of these methods in the given scenario.

### Questions
I noticed that the authors conducted extensive experiments under the "Renyi" setting and achieved excellent results. However, to my knowledge, there are other effective attack methods in FGL, such as GTA. Could the authors provide relevant experiments to verify the effectiveness of their approach against such attacks?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents FedTGE, a new method to protect Federated Graph Learning from backdoor attacks by analyzing energy patterns in data distribution. The method works by identifying and separating benign and malicious clients and adjusts how their data contribute to the final FL model. The authors claim that FedTGE performs better than current methods, which can handle high data heterogeneity, does not require a validation dataset.

### Strengths
-This topic is very important to the community, considering the backdoor defense methodology is developing.

-Well written and interesting.

-Thorough empirical results over a plethora of FL methods.

### Weaknesses
I have some concerns about the assumption and evaluation of this paper below.

### Questions
1. The paper assumes that calculating the energy distribution of clients can effectively tell malicious and benign clients apart. However, this method relies on the energy model accurately capturing the real data distribution. If the energy model fails to do so, especially in noisy or structurally complex data, the assumption is doubtful.

2. While the paper claims the method is effective under non-IID scenarios, I am not confident if the evaluation under non-IID-louvain setting is enough. For example,  there can be node label distribution skew and node feature skew. More non-IID conditions should be evaluated.

3.  TESP is used to adjust aggregation weights to defend against malicious clients. How to prevent from incorrectly including benign clients?

4. The computational complexity of the energy model and similarity propagation might become a bottleneck in large-scale cases, the authors should discuss this issue.

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
This paper addresses the challenges of backdoor attacks in Federated Graph Learning (FGL) by proposing an innovative defense method called FedTGE, which utilizes Topological Graph Energy. The method operates at both the client and server levels, injecting energy distribution knowledge into local models to differentiate between benign and malicious samples. It assigns low energy to benign samples while elevating the energy of constructed malicious substitutes, enabling the selection of trustworthy clients through clustering. At the server level, the energy elements uploaded by clients are treated as nodes in a global energy graph, facilitating energy propagation and enhancing the robustness of client selection. The experimental results validate the effectiveness of FedTGE under varying proportions of malicious clients and different data scenarios, demonstrating its capability to handle high data heterogeneity without requiring a validation dataset.

### Strengths
-	The introduction of the FedTGE framework is innovative, employing energy-based modeling to defend against backdoor attacks in federated graph learning (FGL).
-	The paper presents a comprehensive evaluation of the proposed method across various datasets and scenarios, demonstrating its effectiveness in both IID and non-IID settings.
-	The manuscript is well-structured, presenting a clear delineation of the proposed methodology, experimental setup, and results.

### Weaknesses
-	The efficacy of the FedTGE method heavily relies on the accurate modeling of energy distributions. To enhance the robustness of energy estimations, the authors should consider incorporating visual analysis of energy distributions prior to clustering. 
-	The computational demands associated with energy graph construction and similarity propagation may hinder scalability. The authors should discuss about the computational overhead and the costs of federated transmission.
-	The authors should investigate the defense performance across different trigger structures and patterns to better understand and enhance the system's resilience under varied attack scenarios.
-	The method requires careful threshold selection for clustering energy elements, which could be subjective and impact performance. The authors should describe how to determine these thresholds.

### Questions
Refer to the weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
3
