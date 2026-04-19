# MotifDisco: Motif Causal Discovery For Time Series Motifs

- Decision: Reject
- Scores: 5, 3, 5

## Abstract
Many time series, particularly health data streams, can be best understood as a sequence of phenomenon or events, which we call motifs. A time series motif is a short trace segment which may implicitly capture an underlying phenomenon within the time series. Specifically, we focus on glucose traces collected from continuous glucose monitors (CGMs), which inherently contain motifs representing underlying human behaviors such as eating and exercise. The ability to identify and quantify causal relationships amongst motifs can provide a mechanism to better understand and represent these patterns, useful for improving deep learning and generative models and for advanced technology development (e.g., personalized coaching and artificial insulin delivery systems). However, no previous work has developed causal discovery methods for time series motifs. Therefore, in this paper we develop MotifDisco (motif disco-very of causality), a novel causal discovery framework to learn causal relations amongst motifs from time series traces. We formalize a notion of Motif Causality (MC), inspired from Granger Causality and Transfer Entropy, and develop a Graph Neural Network-based framework that learns causality between motifs by solving an unsupervised link prediction problem. We also integrate MC with three model use cases of forecasting, anomaly detection and clustering, to showcase the use of MC as a building block for other downstream tasks. Finally, we evaluate our framework and find that Motif Causality provides a significant performance improvement in all use cases.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
In this paper author proposed MotifDisco (motif discovery of causality), a novel causal discovery framework to learn causal relations amongst motifs from time series traces based on Granger Causality and Transfer Entropy. Used motif causality in down stream tasks like forecasting, anomaly detection and clustering.

### Strengths
The paper is well written. The literature survey is good. Work is mathematically sound and the author shows run time requirements. The idea of showing the model performance on three tasks was also good.

### Weaknesses
Marginal technical novelty. What is the contribution compared to  Lamp et el (2024) needs to be discussed. Pan et al., 2024; Lowe ¨ et al., 2022 Bonetti et al., 2024; Najafi et al., 2023) are already using  Granger Causality and Transfer Entropy then what is contribution compared to this work is not clear to me. 
Please explicitly state key technical contributions of this paper and how it differ from or improve upon the cited works, particularly in the context of motif-based causal discovery for time series.

I could not understand what you mean by discovery. I am requesting a clear definition of what the authors mean by "discovery" in this context would help clarify the paper's novelty.

The experiment section is incomplete. The author has compared with only one base model.  Also base model is built by the author. The author needs to provide the architecture of the base model. The author need to compare the work with other existing state of art motif causality models for all tasks like Pan et al., 2024; Lowe ¨ et al., 2022 Bonetti et al., 2024; Najafi et al., 2023. 
Also, it seems the author has compared the proposed model only for one data set for each task. Please compare on more real-world data sets. Also, provide details of the data set used.

### Questions
As above

### Soundness
3

### Presentation
2

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
The paper introduces MotifDisco, a novel framework for motif causal discovery in time series. The authors focus on causal relations among motifs, defined as short segments representing underlying phenomena within time series. MotifDisco leverages a combination of Granger Causality and Transfer Entropy to define Motif Causality (MC) and uses a GNN to learn causal relationships by solving an unsupervised link prediction problem. The framework is evaluated on glucose traces collected from continuous glucose monitors and further integrated into forecasting, anomaly detection, and clustering tasks. Overall, the authors claim significant improvements in performance for each of these downstream applications compared to non-causal baselines.

### Strengths
1. Novelty of Causal Discovery Framework: The introduction of Motif Causality (MC) for time series motifs and the development of MotifDisco fills an important gap in time series analysis, especially for health-related data. No prior work has explicitly targeted causal discovery among motifs within time series, which makes this a novel contribution.
2. Flexible Application Scope: Integrating MC into multiple use cases, namely forecasting, anomaly detection, and clustering, demonstrates the proposed framework's versatility and broadens its potential real-world applicability.
3. The experiments span various scenarios, including different motif extraction methods, motif lengths, and scalability. The performance gains in forecasting and anomaly detection validate the utility of incorporating causality into motif-based models.

### Weaknesses
1. Motif Construction Limitation: The method for constructing motifs is largely dependent on heuristic techniques (e.g., chopping or sliding windows). This may lead to arbitrary definitions of motifs that do not always correspond to well-defined physiological phenomena. The authors could consider more dynamic motif extraction methods.
2. Lack of Personalization: The majority strategy for causal inference used in the GNN might overlook personalized differences across individuals, which could limit the accuracy of BP estimation or understanding of other health parameters in highly diverse populations.
3. No Ground Truth for Causal Evaluation: A notable limitation is the lack of ground truth causal structures for motifs, which makes the evaluation of the learned causal graphs challenging. Although indirect measures such as downstream task performance are used to validate the usefulness of the model, a more direct assessment of the accuracy of causal inference is missing.\
4. No comparison with SOTA: this paper does not provide an extensive, systematic comparison against other state-of-the-art causal discovery frameworks and deep learning methods for forecasting, anomaly detection, and clustering.
5. Scalability Issues: The scalability analysis shows that training times grow significantly for large motif sets and numbers of traces. The current implementation may not be suitable for very large datasets, especially in real-time applications. Methods like parallelization for computing motif causality are suggested as improvements.
6. Limited Clinical Validation: The evaluation was limited to glucose data, and the clinical significance of the discovered causal motifs is not thoroughly validated. This limits the generalizability of the proposed method to other medical domains without further empirical evidence.

### Questions
1. How does the chosen motif extraction method impact the causality results? Would alternative approaches, such as clustering-based motif identification, lead to different outcomes?
2. How does the interpretability of the causal graphs change across datasets with different numbers of motifs and motif lengths?
3. How generalizable is MotifDisco to other health data, such as heart rate variability or electroencephalogram (EEG) signals?
4. How does the framework account for inter-individual variability, especially given that causal relationships can be highly individualized?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes to learn granger causal graphs over time series segments. The segments are defined as motifs as they have specific characteristics. The segments are then embedded into graph node embeddings using GraphSage, a message passing graph neural network. These embeddings are then passed through a link prediction model that maximizes the probability of links with high transfer entropy (conditional entropy gain achieved through adding node i to the conditional to predict node j). This process is then repeated over multiple epochs of the training time series data to learn the graph where links are added iteratively. Subsequently, in each iteration, links are removed by computing the reverse edge graph and their corresponding link prediction probabilities. Finally, this motif causal network is used for time series prediction, anomaly detection and clustering.

### Strengths
* Empirically , using the motif causal graphs, improvements in the 3 downstream tasks are identified demonstrate the use of motifs
* Comparison against chunked time series prediction should be presented
* Chunking provides a simple way of making all motifs of the same length

### Weaknesses
* Comparing against existing granger causal techniques applied directly on time lagged variables needs to be done to validate the necessity of motifs
* Identifiability of motifs is left as out of scope, but should be discussed as that defines the nodes used in the causal graph construction.
* Metadata such as time of occurrence and frequency of occurrence of motifs is not well presented in an interpretable manner in the link prediction task, how this might be captured in the motif representation is lacking
* A discussion of complex variable length motifs should be presented.

### Questions
Comparison with competing Granger causal baselines at various granularity should be presented to improve the soundness of the result.

### Soundness
2

### Presentation
3

### Contribution
2
