# FedGMR: Federated Learning with Gradual Model Restoration under Asynchrony and Model Heterogeneity

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Federated learning (FL) holds strong potential for distributed machine learning,
but in heterogeneous environments, Bandwidth-Constrained Clients (BCCs) often
struggle to participate effectively due to limited communication capacity.
Their small sub-models learn quickly at first but become under-parameterized in
later stages, leading to slow convergence and degraded generalization.
We propose FedGMR—Federated Learning with Gradual Model Restoration under
Asynchrony and Model Heterogeneity. FedGMR progressively increases each client’s
sub-model density during training, enabling BCCs to remain effective
contributors throughout the process. In addition, we develop a mask-aware
aggregation (MA) rule tailored for asynchronous MHFL and provide convergence
guarantees showing that aggregated error scales with the average sub-model
density across clients and rounds, while GMR provably shrinks this gap toward
full-model FL. Extensive experiments on FEMNIST, CIFAR-10, and ImageNet-100
demonstrate that FedGMR achieves faster convergence and higher accuracy,
especially under high heterogeneity and non-IID settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces FedGMR to address the limitations of Bandwidth-Constrained Clients (BCCs) and model heterogeneity in federated learning. In many real-world FL systems, clients have varying communication capacities and cannot always train or transmit full models, leading to slow convergence and poor generalization. To mitigate this, FedGMR progressively restores model capacity on each client over time, allowing BCCs to contribute continuously. The method also introduces a transmission and aggregation mechanism that aligns updates from heterogeneous clients and ensures consistent model integration across asynchronous updates. The authors provide theoretical convergence guarantees, showing that the average sub-model density influences the error bound and that the proposed gradual restoration narrows the gap to full-model federated learning. Experimental results on FEMNIST, CIFAR-10, and ImageNet-100 demonstrate that FedGMR delivers faster convergence and higher accuracy than existing FL baselines, particularly in settings with severe non-IID data and system heterogeneity.

### Strengths
+ Introduces a new concept of Gradual Model Restoration (GMR), which progressively increases each client’s model capacity.
+ Designs a mask-aware transmission and aggregation mechanism that aligns updates from heterogeneous clients, ensuring stable training despite differing model structures.
+ Provides formal convergence guarantees under the proposed mechanism.
+ Offers a mathematically grounded explanation of how gradual restoration narrows the performance gap to full-model FL.
+ Conducts extensive experiments on diverse and well-recognized benchmarks: FEMNIST, CIFAR-10, and ImageNet-100.

### Weaknesses
- As BCCs gradually restore larger sub-models, their computation and communication loads increase, making them more prone to straggling (delayed updates). This straggler effect can partially offset the benefits of gradual restoration, especially in asynchronous environments where delayed clients slow global progress or introduce stale gradients. The paper does not analyze or mitigate this trade-off, nor quantify how much density increase is sustainable before latency outweighs the learning gain. This limitation suggests that while FedGMR improves participation for BCCs initially, it may reintroduce asynchrony inefficiencies in later training stages as model sizes grow.
- FedGMR enables BCCs to train sub-models to mitigate the straggler effect. Recent works such as [R1] and [R2] leverage tiering approaches to mitigate the straggler effect, which should be discussed and compared in the paper.

[R1] Chai, Zheng, Yujing Chen, Ali Anwar, Liang Zhao, Yue Cheng, and Huzefa Rangwala. "FedAT: A high-performance and communication-efficient federated learning system with asynchronous tiers." In Proceedings of the international conference for high performance computing, networking, storage and analysis, pp. 1-16. 2021.

[R2] Mohammadabadi, Seyed Mahmoud Sajjadi, Syed Zawad, Feng Yan, and Lei Yang. "Speed up federated learning in heterogeneous environments: a dynamic tiering approach." IEEE Internet of Things Journal (2024).

- Experiments are conducted mainly on image classification datasets (FEMNIST, CIFAR-10, ImageNet-100); no evaluation on other domains such as NLP, speech, or sensor data, which would strengthen the generality of the approach. 
- Results focus heavily on accuracy and convergence, with no measurement of communication overhead, latency. Also, the results are given under a fixed wall-clock budget. It is suggested to also provide results under a fixed target accuracy and compare the convergence time.
- The performance of FedGMR on transformer models is not evaluated.
- The methodology section is dense, and key components like mask-aware aggregation are only briefly explained.

### Questions
1. How does FedGMR handle the growing computational and communication burden as sub-models expand for BCCs? Please analyze or visualize the relationship between model density and update delay. An ablation showing when the latency begins to outweigh the accuracy gain would help quantify this trade-off.
2. How does FedGMR compare conceptually and empirically with tier-based asynchronous frameworks like FedAT ([R1]) and Dynamic Tiering ([R2])? Please include a discussion or experimental comparison with these approaches, since both aim to mitigate straggler effects. Highlight how FedGMR’s gradual model restoration differs from or complements tiering-based synchronization strategies in terms of communication cost and convergence speed.
3. Can FedGMR generalize beyond image classification tasks? Include or discuss experiments in non-vision domains such as NLP or sensor data to show generality, since many FL applications (e.g., federated BERT fine-tuning) involve non-visual modalities.
4. Provide additional results under a fixed target accuracy and report the time to convergence for each baseline. This would allow a fairer assessment of FedGMR’s training efficiency. Include measurements of communication overhead and latency, since these are central to evaluating the benefits of gradual model restoration.
5. How does FedGMR perform with Transformer or attention-based architectures, which are now common in FL applications? Add an experiment or at least a discussion on how model restoration and mask-aware aggregation behave when applied to Transformers. The impact on gradient synchronization and sub-model alignment would be insightful.
6. How scalable is FedGMR in large federations (e.g., thousands of clients) where communication delays and resource diversity are more extreme? Discuss the scalability of  FedGMR.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
This work proposes a new heterogeneous FL frameowork, FedGMR. This framework consists of mainly three components, gradual density adjustment, buffered mask-aware aggregation, and incremental model splitting. In addition to those main components, the authors employ semi-asynchronous update aggregation scheme to alleviate the straggler effect (synchronization cost) in FL. The authors provides theoretical analysis of convergence behaviors as well as empirical results of accuracy comparisons across SOTA heterogeneous FL methods. Overall, the paper shows promising FL performance, however, I see several strong weaknesses that should be addressed.

### Strengths
1. The authors consider important and realistic issue in FL, the heterogeneous system environments together with data heterogeneity.
2. The theoretical analysis well incorporate the proposed method into the traditional analysis framework.
3. Asynchronous scheme looks promising. I especially appreciate this approach since most of recent FL studies just simply consider synchronous model aggregations that are not realistic.

### Weaknesses
**Comments on Main Idea**

1. The authors assume that weak clients become straggler mainly due to the limited network bandwidth. However, such weak clients tend to have limited system resources such as slow compute power and a limited memory space. Thus, gradually increasing density may make some weak clients with limited resources impossible to join the training any more. Therefore, I do not think the problem definition is convincing.

2. In this work, 'model-heterogeneous' term may mislead readers. In general, 'heterogeneous' indicates independently designed entities. E.g., heterogeneous data distributions do not have any dependencies across local datasets. However, this work assumes that weak clients have limited system resources and are assigned with models differently prunned. Basically, however, they share the same model architecture. Thus, this study does not cover model-heterogeneious FL. It would rather be a prunning-based sub-model distribution method.

3. How is GMR equation (4) built? What is the meaning of multiplying learning rate $\lambda$ to the ratio of time difference to the BCC's time? The main equation is not well substantiated due to the limiated explanation. The authors should focus more on **why** (4) is the best choice rather than **how** only.

4. In theoretical analysis, the assumptions are too strong. Especially, the bounded noise (assumption 3) and bias (assumption 4) are unrealistic. In addition, most of recent studies do not rely on the assumption of bounded gradient magnitude (assumption 2). Instead, the bounded gradient variance assumption is used popularly. Given these strong assumptions, the results are not that convincing.

5. Also, there are $f_1$, $f_2$, and $f_3$ in Theorem 2, but they have never been defined before. What does it mean by the subscripts?

6. Finally, I recommend analyzing the complexity of the derived convergence rate so that the performance can be easily compared with other methods. Since there are many indirect notations such as $\mathcal{A}$ and $\mathcal{B}$, it is not easy to compare the performance.

7. FedGMR obviously uses much more server-side system resources such as memory space for local models. However, other methods do not require such strong resources at the server-side. E.g., HeteroFL just align the submodels and directly average the parameters. Fjord also just randomly prunned local models are aggregated at the server-side. Thus, even though FedGMR achieves higher accuracy than other methods, the performance gain probably comes from using more resources. Table 1 does not consider such differences and just directly compare the achieved accuracy within the same amount of time. I do not think it is a fair comparison.

8. As compared to the best-performing SOTA method, the performance gain of FedGMR is not significant. E.g., FEMNIST non-IID performance gap between FedGMR and FedAsync is just 0.9% when non-IIDness is low. When non-IIDness is high, Fjord becomes the best. For CIFAR-10, the difference becomes 1.1~1.2%. Considering that FedGMR uses more server-side resources, I think this performance gain is not that impressive.
  
9. Some recently published and strongly related heterogeneous FL methods could be directly compared in Table 1 or at least discussed in Section 2. Some examples are as follows.

[1] Liu et al., Efficient Federated Learning with Heterogeneous Data and Adaptive Dropout, TKDD, 2025.

[2] Lee et al., Embracing Federated Learning: Enabling Weak Client Participation via Partial Model Training, IEEE Trans. on Mobile Computing, 2024.

[3] Liu et al., No One Left Behind: Inclusive Federated Learning over Heterogeneous Devices, KDD, 2022. 

**Comments on Presentation Quality**

1. The contribution summary at the end of Introduction section does not deliver meaningful information while taking up the space of several lines. I see many recent literature has this summary at the same position, but what is the point of having them? The summary includes nothing new, the proposed idea, the existance of convergence analysis, and experimental settings. I strongly recommend removing them and use the space for other more critical things, e.g., more experimental results or detailed discussion of main ideas.

2. Figure 1 does not clearly explain the workflow. First, what is the meaning of numbers? The caption explains the steps in the order of 3, 4, 1, and 2. It seriously confuses readers. Second, after the step 2, there are two horizontal flows, 3 and 4. The figure neither explains what they are nor why they appear in the figure. Overall, the figure should be improved much to deliver key ideas.

3. Algorithm 1 is not self-contained. What is IMSc? What is GMR? They should be at least connected to any equations or subsections where the method is explained. 

4. There are too many acronyms which seriously hurts readability. BAC, BCC, GMR, IMS, MHFL, MA, GA, MRI, etc... I suggest using their full names unless there are some special reasons. Just few are okay, but there are too many now.

### Questions
Some questions are included in the above weakness section. Please carefully address them.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes FedGMR-Federated Learning with Gradual Model Restoration under Asynchrony and Model Heterogeneity. It progressively increases each client’s model capacity during training, and a tailored transmission and aggregation mechanism is designed to better accommodate system-level heterogeneity. Theory and experiments on FEMNIST, CIFAR-10, and ImageNet-100 confirm the effectiveness of the proposed method.

### Strengths
1. This work proposes FedGMR, the first FL framework that dynamically restores sub-model capacity via GMR. Two auxiliary mechanisms, IMS for efficient transmission and BufferMaskFedAvg for robust aggregation under structural inconsistency, are designed to enhance FedGMR.
2. A convergence guarantee is established to show that the average sub-model density across clients and time governs error accumulation.
3. Experiments on FEMNIST, CIFAR-10, and ImageNet-100 demonstrate faster convergence and robustness of proposed method.

### Weaknesses
1. What’s the motivation of GMR? Although authors claim that GMR can dynamically restore sub-model capacity, the communication capacity is limited for BCCs. Is it essential for them to train the full model？ Furthermore, the current design of GMR fails to clearly demonstrate how the principles articulated by the authors in Section 3.1 are implemented.
2. The paper's logic seems confused. If the core contribution is GMR, what problems do the other two auxiliary components solve, and are they considered core contributions of the paper as well?
3. Although authors provide the convergence guarantee, the analysis simplifies the core asynchronous process, lacking sufficient theoretical rigor. Given that many recent works [1-3] on asynchronous federated learning have already provided comprehensive convergence analyses for their respective frameworks, this section needs to be strengthened.
4. The experiments only compare against the synchronous submodel training method. The authors should demonstrate the performance of the proposed method in an asynchronous scenario; it is suggested that combined asynchronous baselines be added for comparison to fully prove the method's effectiveness.
5. The paper needs to explicitly indicate which empirical results (figures or tables) demonstrate that the proposed method achieves faster convergence speed than other baselines.
6. The paper's readability needs improvement. A lot of abbreviations without a clear meaning are not feasible for readers to understand the work. Figure 1 fails to clearly show the complete process of the proposed method, and its caption is confusing and inconsistent with the legend.

---
[1] Sharper convergence guarantees for asynchronous SGD for distributed and federated learning.

[2] Asynchronous Federated Optimization

[3] FADAS: Towards Federated Adaptive Asynchronous Optimization

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
