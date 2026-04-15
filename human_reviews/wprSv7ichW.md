# Benchmarking Algorithms for Federated Domain Generalization

- Decision: Accept (spotlight)
- Scores: 8, 6, 6

## Abstract
While prior federated learning (FL) methods mainly consider client heterogeneity, we focus on the *Federated Domain Generalization (DG)* task, which introduces train-test heterogeneity in the FL context. Existing evaluations in this field are limited in terms of the scale of the clients and dataset diversity. Thus, we propose a Federated DG benchmark that aim to test the limits of current methods with high client heterogeneity, large numbers of clients, and diverse datasets. Towards this objective, we introduce a novel data partition method that allows us to distribute any domain dataset among few or many clients while controlling client heterogeneity. We then introduce and apply our methodology to evaluate 14 DG methods, which include centralized DG methods adapted to the FL context, FL methods that handle client heterogeneity, and methods designed specifically for Federated DG on 7 datasets. Our results suggest that, despite some progress, significant performance gaps remain in Federated DG, especially when evaluating with a large number of clients, high client heterogeneity, or more realistic datasets. Furthermore, our extendable benchmark code will be publicly released to aid in benchmarking future Federated DG approaches.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper explores the intersection between Federated Learning and Domain Generalization, namely, Federated Domain Generalization (FDG), in which different domains are separated among clients that will collaboratively learn a model that generalizes to unseen domains. This paper pushes on an important direction: on the methodology behind evaluating FDG algorithms. In this sense, the authors: (i) Present an interesting review of existing practice in FDG; (ii) Propose a novel way of partitioning clients in FDG; (iii) Propose new metrics for evaluating the hardness of benchmarks; (iv) provide extensive evaluation of FDG methods.

### Strengths
This paper plays the same role of [Gulrajani and Lopez-Paz, 2021] for FDG, i.e., an important paper that provides an in-depth discussion about how to evaluate existing methods. In this angle, the paper provides extensive experimentation and interesting insights. In this sense, the paper is quite important for the field.

Furthermore, as the authors discuss in the paper, the federated setting poses __new challenges__ to DG. This is especially related to the new partition method that the authors propose, and the main novelty of the paper. This contribution helps merging the fields of federated learning and domain generalization, making the evaluation of FDG algorithms more realistic. As a consequence, this direction is quite impactful and helpful for the field of FDG.

### Weaknesses
Overall, I have no major concerns with this paper. My only critique is a (minor) lack of clarity. In section 4., the term ERM is never defined, and the empirical risk minimization method is never properly described. While, for knowledgeable audiences, this does not hurt the understanding of the paper in general, this makes the paper harder to read for beginners. I suggest authors add a description of the ERM in page 3, on the Federated DG paragraph.

### Questions
__Q1.__ Since Domain Adaptation is a related problem to Domain Generalization, is it possible to apply the proposed partitioning scheme to Federated Domain Adaptation?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a benchmark for domain generalization (DG) in federated learning. Specifically, they (1) develop a novel method to partition a DG dataset to any number of clients, (2) propose a benchmark methodology including four important factors, (3) experiment with a line of baselines and datasets.

### Strengths
1. This paper contributes to an important topic. I believe federated DG is an important problem in FL, especially cross-device FL when the trained FL global model need to generalize to a large amount of clients that do not participate in FL training. 
2. The paper is well-written and easy to follow. 
3. The proposed benchmark includes a variety of datasets and algorithms. 
4. The experiments regarding the number of clients is highly related to cross-device FL, and indicate an important drawback in the current federated DG experiments.

### Weaknesses
1. In the context of domain generalization, we are particularly concerned with whether models trained on limited source domains can generalize to new target domains. This paper also uses held-out domains (in Appendix C.2). I believe that this aspect should be more explicitly explained in the main body of the text; otherwise, readers might easily misconstrue that all domains were used to construct training clients, which is misleading. 
2. Error bars are not provided for experiment results, the conclusion may be influence by random fluctuation. 

Minor: 
1. Page5 line 12: homogeneous -> homogeneity
2. When considering the second kind of DG, it is relevant to “performance fairness” in FL, which encourage a uniform distribution of accuracy across clients. Although works in this direction might emphasize more on participating clients, I believe at least the AFL algorithm [1] can be a good supplement to the benchmark. 

[1] Mehryar Mohri, Gary Sivek, Ananda Theertha Suresh. Agnostic Federated Learning. ICML 2019.

### Questions
1. In Eq. (1), two kinds of DG is mentioned. Which kind of DG is mainly used in your benchmark? 
2. The performance for some algorithm, for example, FedSR, is very low, and consistently lower than FedAvg. Could you explain the reason behind?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces a benchmark for federated domain generalization, which is a challenging problem that requires learning a model that can generalize to heterogeneous data in a federated setting. The paper presents a novel data partitioning method that can create heterogeneous clients from any domain dataset, and a benchmark methodology that considers four factors: number of clients, client heterogeneity, dataset diversity, and out-of-distribution generalization. The paper also evaluates 13 Federated DG methods on 7 datasets and provides insights into their strengths and limitations.

### Strengths
* The paper proposes a comprehensive and rigorous benchmark for Federated DG that covers various aspects of the problem and can be easily extended to new datasets and methods.
* The paper provides a clear and detailed description of the data partitioning method and the benchmark methodology, as well as the implementation details of the methods and datasets.
* The paper conducts extensive experiments and analyzes the results from different perspectives.

### Weaknesses
The paper mainly summarizes the experimental observations, but does not offer much theoretical analysis or explanation for why some methods perform better than others in certain scenarios, which would be helpful to gain more insights into the FDG problem and the design of effective methods.

### Questions
No.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
