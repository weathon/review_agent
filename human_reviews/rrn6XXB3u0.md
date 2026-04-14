# Fast Few-Shot Graph Flow Prediction

- Decision: Reject
- Scores: 3, 3, 6

## Abstract
Accurate prediction of traffic flow is crucial for optimizing transportation networks, mitigating congestion, and improving urban planning. However, existing approaches like graph neural networks (GNNs) and traffic simulations face challenges in predicting flow for unseen road networks without historical data. Without abundant training data, GNNs often generalize poorly to new graphs, while simulations can be computationally infeasible for large-scale networks. This paper tackles the problem of few-shot traffic flow prediction in unseen road networks. We propose a novel traffic simulation algorithm that efficiently predicts flow based on node and edge attributes. Through theoretical analysis, we demonstrate our approach closely approximates true flow with asymptotically optimal runtime complexity. Experiments on real-world road networks show our simulation algorithm outperforms GNNs for predicting traffic in unseen cities after training on only three cities. While motivated by traffic prediction in road networks, we expect our contributions to have broader applicability to general graph flow prediction problems across domains.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
Here’s a summary for the paper "FAST FEW-SHOT GRAPH FLOW PREDICTION" with a focus on its innovations and shortcomings:

This paper proposes a traffic simulation algorithm designed for predicting the Average Annual Daily Traffic (AADT) in cities with limited historical data. The approach combines road network attributes with a few-shot learning framework, aiming to provide accurate flow predictions on unseen city road networks. The main contributions include the use of node and edge attributes to simulate flow and a theoretical analysis suggesting an optimal runtime complexity. However, the paper poses several significant problems that undermine its contributions. The use of AADT as a prediction target limits the approach’s applicability, as most traffic prediction studies focus on dynamic, real-time data that provide actionable insights for urban management. Additionally, the paper lacks adequate baseline comparisons, including only a single GNN model, which is insufficient for establishing superiority. The absence of common traffic datasets and unclear experimental details further weaken the study’s rigor and reproducibility. These substantial gaps, along with unaddressed advancements in transfer learning for traffic prediction, leave the paper unconvincing in terms of practical relevance and scientific contribution.

### Strengths
S1: Few-Shot Learning for Traffic Prediction: The paper leverages a few-shot learning approach, which is innovative in the context of traffic prediction, particularly for cities with limited historical data. This approach is beneficial as it reduces dependency on extensive datasets, which are often unavailable for smaller or new urban areas.

S2: Integration of Multiple Data Sources: The model integrates various data sources, including OpenStreetMap features, satellite imagery, and population density, to enrich the input attributes. This data fusion enhances the model's capability to capture the spatial complexity of urban networks, which can improve traffic flow predictions.

S3: Efficient Computational Design: The proposed method claims asymptotically optimal runtime complexity, which could make it scalable compared to traditional traffic simulations. This computational efficiency is valuable for applications on large-scale networks where real-time processing demands are high.

### Weaknesses
In my opinion, this article still looks like a semi-finished product. The weaknesses (questions) are listed below:

Q1: The existing challenges presented in the Abstract section is **completely incorrect**. The author said: "However, existing approaches like graph neural networks (GNNs) and traffic simulations face challenges in predicting flow for unseen road networks without historical data." However, to the best of my knowledge, this challenge has been deeply investigated for several years. Transfer Learning-based approaches such as [1] [2] [3] have been published in top AI conference, which shown very promising results in solving the challenge you mentioned in this article. I suggest the authors to make a further clarification about your motivations behind this work.

**Reference.**

[1] Mallick, Tanwi, et al. "Transfer learning with graph neural networks for short-term highway traffic forecasting." *2020 25th International Conference on Pattern Recognition (ICPR)*. IEEE, 2021.

[2] Yilun Jin, Kai Chen, and Qiang Yang. 2023. Transferable Graph Structure Learning for Graph-based Traffic Forecasting Across Cities. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '23). 

[3] Wang, Senzhang, et al. "Spatio-temporal knowledge transfer for urban crowd flow prediction via deep attentive adaptation networks." *IEEE Transactions on Intelligent Transportation Systems* 23.5 (2021): 4695-4705.



Q2: There are lots of uncleared **definitions and concepts** in the writing of this paper. For example, the main task is "Graph Flow Prediction", but the concept is rarely seen in general traffic prediction domain. As such, you must give a formal definition with necessary notations and equations to tell the reader what is "Graph Flow Prediction". Aside from that, you also need to formally define how you construct the traffic network graph in this research, What is the edge, what is the node, what are the node feature attributes, and how you calculate the element of graph adjacency matrix. Unfortunately, I don't see any formal preliminary definition in this article. The uncleared concepts make this paper hard to comprehend.



Although the method introduced in this work seems technically sound, there are numerous fatal questions regarding the experiments, in fact your current experiments fail to prove the superiority and effectiveness of the proposed simulation algorithm. See the following questions:

**Q3: Why did you choose AADT as the prediction target?** Based on my understanding, most traffic simulation methods focus on predicting traffic flow for specific future time intervals (e.g., the next 30 minutes, 1 hour, or 4 hours). These approaches offer time-variant, dynamic predictions that provide real-time feedback and insights into urban traffic conditions, which are invaluable for traffic management. In contrast, this study targets AADT, a fixed annual average daily traffic value. Given that AADT represents only a single value per year, what is the rationale for using a traffic simulation algorithm for its prediction? It appears inefficient to collect extensive data, extract features, and build traffic graphs and neural network models if the outcome is only a static, annual statistical metric.

Q4: Dataset statistics not clear. The three datasets you choose is not the widely recognized and open-sourced datasets in general traffic prediction domain. For instance, PeMS03, PeMS04, PeMS07, METR-LA, PeMS-Bay, are commonly adopted datasets for traffic prediction and simulation.

**Q5: The selection of baseline methods is clearly insufficient.** Comparing your proposed algorithm with only **one single GNN model** is inadequate to demonstrate its superiority. Typically, a comprehensive comparison includes at least 7 baseline methods, and many studies include even more. It is recommended to compare your approach with a broader range of GNN-based methods. If not with the latest state-of-the-art, you should at least include widely recognized baseline models from the past 3-4 years, such as DeepSTN+, ASTGCN[2], and AGCRN. Additionally, comparisons with other traffic simulation methods proposed in recent years are necessary, such as those in references [4] and [5].

Reference.

[1] Lin, Ziqian, et al. "Deepstn+: Context-aware spatial-temporal neural network for crowd flow prediction in metropolis." *Proceedings of the AAAI conference on artificial intelligence*. Vol. 33. No. 01. 2019.

[2] Guo, Shengnan, et al. "Attention based spatial-temporal graph convolutional networks for traffic flow forecasting." *Proceedings of the AAAI conference on artificial intelligence*. Vol. 33. No. 01. 2019.

[3] Bai, Lei, et al. "Adaptive graph convolutional recurrent network for traffic forecasting." *Advances in neural information processing systems* 33 (2020): 17804-17815.

[4] Liang, Chumeng, et al. "Cblab: Supporting the training of large-scale traffic control policies with scalable traffic simulation." *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*. 2023.

[5] Wenl, Licheng, et al. "LimSim: A long-term interactive multi-scenario traffic simulator." *2023 IEEE 26th International Conference on Intelligent Transportation Systems (ITSC)*. IEEE, 2023.



Q6: The significant implementation details are missing. For example, it is not clear your proposed simulation method is implemented using which programming language under which framework. In addition, the authors haven't yet open-sourced their code repository for evaluation, with both rarely used datasets and not open-sourced code, it's hard to say you achieved superior performance.



Q7: The authors lack a comprehensive knowledge about traffic prediction and graph neural networks.

### Questions
Q1: The existing challenges presented in the Abstract section is **completely incorrect**. The author said: "However, existing approaches like graph neural networks (GNNs) and traffic simulations face challenges in predicting flow for unseen road networks without historical data." However, to the best of my knowledge, this challenge has been deeply investigated for several years. Transfer Learning-based approaches such as [1] [2] [3] have been published in top AI conference, which shown very promising results in solving the challenge you mentioned in this article. I suggest the authors to make a further clarification about your motivations behind this work.


**Reference.**

[1] Mallick, Tanwi, et al. "Transfer learning with graph neural networks for short-term highway traffic forecasting." *2020 25th International Conference on Pattern Recognition (ICPR)*. IEEE, 2021.
[2] Yilun Jin, Kai Chen, and Qiang Yang. 2023. Transferable Graph Structure Learning for Graph-based Traffic Forecasting Across Cities. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '23). 
[3] Wang, Senzhang, et al. "Spatio-temporal knowledge transfer for urban crowd flow prediction via deep attentive adaptation networks." *IEEE Transactions on Intelligent Transportation Systems* 23.5 (2021): 4695-4705.

Q2: There are lots of uncleared **definitions and concepts** in the writing of this paper. For example, the main task is "Graph Flow Prediction", but the concept is rarely seen in general traffic prediction domain. As such, you must give a formal definition with necessary notations and equations to tell the reader what is "Graph Flow Prediction". Aside from that, you also need to formally define how you construct the traffic network graph in this research, What is the edge, what is the node, what are the node feature attributes, and how you calculate the element of graph adjacency matrix. Unfortunately, I don't see any formal preliminary definition in this article. The uncleared concepts make this paper hard to comprehend.

Although the method introduced in this work seems technically sound, there are numerous fatal questions regarding the experiments, in fact your current experiments fail to prove the superiority and effectiveness of the proposed simulation algorithm. See the following questions:

**Q3: Why did you choose AADT as the prediction target?** Based on my understanding, most traffic simulation methods focus on predicting traffic flow for specific future time intervals (e.g., the next 30 minutes, 1 hour, or 4 hours). These approaches offer time-variant, dynamic predictions that provide real-time feedback and insights into urban traffic conditions, which are invaluable for traffic management. In contrast, this study targets AADT, a fixed annual average daily traffic value. Given that AADT represents only a single value per year, what is the rationale for using a traffic simulation algorithm for its prediction? It appears inefficient to collect extensive data, extract features, and build traffic graphs and neural network models if the outcome is only a static, annual statistical metric.

Q4: Dataset statistics not clear. The three datasets you choose is not the widely recognized and open-sourced datasets in general traffic prediction domain. For instance, PeMS03, PeMS04, PeMS07, METR-LA, PeMS-Bay, are commonly adopted datasets for traffic prediction and simulation.

**Q5: The selection of baseline methods is clearly insufficient.** Comparing your proposed algorithm with only **one single GNN model** is inadequate to demonstrate its superiority. Typically, a comprehensive comparison includes at least 7 baseline methods, and many studies include even more. It is recommended to compare your approach with a broader range of GNN-based methods. If not with the latest state-of-the-art, you should at least include widely recognized baseline models from the past 3-4 years, such as DeepSTN+, ASTGCN[2], and AGCRN. Additionally, comparisons with other traffic simulation methods proposed in recent years are necessary, such as those in references [4] and [5].

Reference.

[1] Lin, Ziqian, et al. "Deepstn+: Context-aware spatial-temporal neural network for crowd flow prediction in metropolis." *Proceedings of the AAAI conference on artificial intelligence*. Vol. 33. No. 01. 2019.

[2] Guo, Shengnan, et al. "Attention based spatial-temporal graph convolutional networks for traffic flow forecasting." *Proceedings of the AAAI conference on artificial intelligence*. Vol. 33. No. 01. 2019.

[3] Bai, Lei, et al. "Adaptive graph convolutional recurrent network for traffic forecasting." *Advances in neural information processing systems* 33 (2020): 17804-17815.

[4] Liang, Chumeng, et al. "Cblab: Supporting the training of large-scale traffic control policies with scalable traffic simulation." *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*. 2023.

[5] Wenl, Licheng, et al. "LimSim: A long-term interactive multi-scenario traffic simulator." *2023 IEEE 26th International Conference on Intelligent Transportation Systems (ITSC)*. IEEE, 2023.



Q6: The significant implementation details are missing. For example, it is not clear your proposed simulation method is implemented using which programming language under which framework. In addition, the authors haven't yet open-sourced their code repository for evaluation, with both rarely used datasets and not open-sourced code, it's hard to say you achieved superior performance.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper investigates the problem of traffic flow prediction with limited viewpoints in an unknown road network. A traffic simulation algorithm based on node and edge attributes is proposed to effectively predict traffic flow.

### Strengths
The theory of the method is rich.

### Weaknesses
What is traffic simulation？ what is the difference between it and GCN-based methods? What's its advantage? Why simulations can be computationally infeasible for large-scale networks?

The motivation is unclear. Why is it difficult for GCN to generalize to unseen graphs with limited viewpoints? As far as I know, GCN is a type of inductive learning method that relies on message passing mechanisms to generate representations for new nodes.

The writing of the article requires significant modifications. While the introduction emphasizes the generality of the proposed method, the experiments only focus on a specific case of traffic flow prediction. The general potential of the method is not well explained.

The paper introduces a traffic simulation method, but related work is missing, making it difficult for readers to understand the gaps in existing research.

The theory heavily relies on two assumptions. What empirical support do these assumptions have? Exploring a broader theoretical framework that applies to general scenarios would make the theory more robust.

The paper is difficult to understand and requires major revisions. I suggest the authors add section hints or summaries to explain what they have done.

The experiments are not sufficient, and there is a lack of discussion on advanced baselines, such as cross-city migration methods.

The content of the paper is not substantial. I recommend adding more experiments, such as visual experiments.

### Questions
See the  Weaknesses.

### Soundness
2

### Presentation
1

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
This paper addresses the challenge of few-shot traffic flow prediction in unseen road networks by proposing a novel traffic simulation algorithm. The authors highlight the limitations of existing methods, such as graph neural networks (GNNs), when historical data is lacking. Experimental results demonstrate the effectiveness of the proposed method in accurately predicting traffic flow in unseen cities using data trained on only three cities, showcasing its computational efficiency.

### Strengths
1 The paper introduces a new traffic simulation algorithm that fills a gap in flow prediction when limited training data is available.
2 It employs real-world city data for experiments, providing compelling results that support the method's effectiveness.
3 The algorithm exhibits superior computational complexity, enabling efficient operation on large-scale networks.
Theoretical Support: The authors provide thorough theoretical analysis, demonstrating the method's convergence and accuracy.

### Weaknesses
1: The use of only three cities for training may limit the generalizability of the results.
2: While comparisons are made with GNNs, the inclusion of additional methods for a more comprehensive comparison would strengthen the paper.
3: The paper does not adequately explore the interpretability of the model, which is crucial in traffic flow prediction.

### Questions
NAN

### Soundness
2

### Presentation
3

### Contribution
2
