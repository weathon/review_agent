# Harnessing Diversity for Important Data Selection in Pretraining Large Language Models

- Decision: Accept (Spotlight)
- Scores: 6, 6, 8, 6

## Abstract
Data selection is of great significance in  pretraining large language models, given the  variation in quality within the large-scale available training corpora. 
To achieve this, researchers are currently investigating the use of data influence to measure the importance of data instances, $i.e.,$ a high influence score indicates that incorporating this instance to the training set is likely to enhance the model performance. Consequently, they select the top-$k$ instances with the highest scores.  However, this approach has several limitations. 
(1) Calculating the accurate influence of all available data is time-consuming.
(2) The selected data instances are not diverse enough, which may hinder the pretrained model's ability to generalize effectively to various downstream tasks.
In this paper, we introduce $\texttt{Quad}$, a data selection approach that considers both quality and diversity by using data influence to achieve state-of-the-art pretraining results.
To compute the influence ($i.e.,$ the quality) more accurately and efficiently, we incorporate the attention layers to capture more semantic details, which can be accelerated through the Kronecker product. 
For the diversity, $\texttt{Quad}$ clusters the dataset into similar data instances within each cluster and diverse instances across different clusters. For each cluster, if we opt to select data from it, we take some samples to evaluate the influence to prevent processing all instances. Overall, we favor clusters with highly influential instances (ensuring high quality) or clusters that have been selected less frequently (ensuring diversity), thereby well balancing between quality and diversity.  Experiments on Slimpajama and FineWeb over 7B large language models demonstrate that $\texttt{Quad}$ significantly outperforms other data selection methods with a low FLOPs consumption. Further analysis also validates the effectiveness of our influence calculation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes to balance the diversity and quality in data selection for LLMs pre-training. The author adopts the MAB method to dynamically estimate and sample data points from various clusters. Additionally, they introduce a novel influential calculation approach to estimate the data point more accurately while improving processing efficiency. Their experimental results on extensive benchmarks validate the effectiveness and efficiency of the proposed method.

### Strengths
* The introduction of MAB to LLMs' pre-training data selection is reasonable and interesting, addressing a practical problem about diversity and efficiency.
* The experiment results demonstrate that their proposed method significantly improves the pre-trained LLM's performance while maintaining efficiency.

### Weaknesses
* The experiments are not thorough. The experiment and analysis only take up 2 pages, which fails to thoroughly discuss many important components and designs of the proposed method. For example, whether the CS function could be designed in another way? Also, while the evaluation is conducted in various benchmarks, the authors only adopt experiments in a single LLM. This also makes people doubt the generalization of their method. I suggest the author to fully utilize the 10-page space and conduct further experiments.

* According to ablation study results, the model's performance is quite sensitive to the sampling threshold of influence. And there are no hyper-parameter selection strategies introduced in the paper. All these make me worry about the robustness of the proposed method to apply in real-world scenarios.

### Questions
* How do you choose the sampling threshold hyper-parameter in the main experiment? Have you tuned it in a validation set?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a new method for selecting LLM pretraining data. The core idea is to measure data quality with diversity. The authors introduce a new approach named Quad, which utilizes attention layers for influence calculation, and clusters data to maintain diversity. Experimental results on the Slimpajama dataset show that Quad surpasses other data selection methods in performance.

### Strengths
1. The paper presents a framework that is overall reasonable.
2. The influence calculation with attention layers technique is interesting and could potentially be useful in other settings.
3. Experiments (of the current limited settings) are complete and the authors do release the code for better reproducibility

### Weaknesses
The main limitation of this study is its experiment scale is too small (in terms of claiming the method for LLM pretraining). Only 30B out of 627B data are selected for training a small 1.3B decoder-only LLM. In practice, most LLMs will start with at least 7B so how the technique can generalize to reasonably sized LLMs is questionable. Also related to this limitation is the diversity of testing datasets, little (long-text) generation tasks are evaluated.

The second limitation of this study is that the authors do not fully motivate the problem setting. While cleaning noisy data and removing near-duplicate examples are useful for LLM pretraining, I am not fully convinced why you have to select a subset of "quality data" from the full dataset. For example, if we do a coarse-grained filtering of the original 627B to leave around 400-500B data and use all of them to do LLM pretraining, will the model performance better?

Finally, there are a few presentation issues such as the unclear method descriptions (see the below questions) and typos/grammatical issues (just list a few below).

1. Line 042: ”LLMs are generalizable a broad spectrum of downstream tasks”
2. Line 095: “..., which successfully addressing above challenges, achieves” 
3. Line 289, “.. will miss massive information of Consequently, …“

### Questions
1. Explain figure 1(a) better, what is the x- and y-axis?
2. How do you apply the “JohnsonLindenstrauss Lemma” (line 343)
3. Why only select “30B tokens out of the 627B” (line 360), what if we just do a coarse-grained filtering of the original 627B to leave around 400-500B data (e.g., via near-duplicate removal) and use all of them to do LLM pretraining, will the model performance better?
4. Is the Quad method robust to the clustering algorithms? What features are used in the clulstering?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents a method for data selection, namely Quad, which (i) uses the influence function with multi-head attention layers to estimate the potential impact of individual samples on the model's performance; (ii) uses clustering and multi-armed bandits (where each cluster is an arm) to balance between quality and diversity. The paper further proposes acceleration techniques to reduce computational cost and memory usage. The proposed method is applied on the SlimPajama dataset by training a 1.3B model, evaluated on a set of common text benchmarks, and compared with a set of data selection baselines.

### Strengths
The paper addresses the important challenge of data selection via a novel method and practical acceleration techniques. Influence function is an expensive operation and the proposed techniques make it more accessible for LLMs where computational overhead is an important factor. 

The paper presents a comprehensive evaluation on downstream tasks as well as ablations and statistics to help understand the proposed method. 

Despite the method's complexity, the paper is overall clearly written and easy to follow. The authors also attach the code, which helps the community to adapt this method for custom usecases.

### Weaknesses
The proposed method requires a complex set of steps and hyperparameters (e.g., $\alpha$,  $\gamma$, and $\tau$ for multi-armed bandit) and, even though there are ablations for some of these, it 
might be hard and resource-intensive to develop and tune in practice for diverse applications.

The experimental findings show that all evaluated methods have limited difference in downstream performance and it is not clear whether the observed improvements are statistically significant. It is also concerning that baselines like DSIR show no benefit compared to random selection, which might be contradicting with findings from previous work. Therefore, I would recommend a more thorough discussion behind the empirical benefits of the proposed method and the connection of the findings compared to the previous literature. To better demonstrate the benefits, it might be worth (i) considering more experimental settings, for example with models of varying sizes; (ii) plotting the training and validation losses for each data selection method.

### Questions
Could the authors report the results of statistical significance tests?

How does the selection of clustering algorithm and number of cluster affect the performance of the proposed method?

Can the authors clarify how they compute the FLOPS column in Table 1?

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
This paper explores the significance of data selection in the pretraining of large language models (LLMs) and the limitations of existing methods, including high computational costs and a lack of diversity. To address these issues, the authors propose a new method called Quad, which aims to balance data quality and diversity. Quad clusters the data and samples from each cluster to assess its influence, reducing computational burden while ensuring diversity. Additionally, Quad leverages attention mechanisms and the Kronecker product to efficiently compute data influence, further enhancing computational efficiency. Experimental results show that Quad significantly outperforms existing methods on the Slimpajama dataset and nine downstream tasks, achieving a 1.39% improvement in zero-shot accuracy with lower computational resource consumption. Overall, Quad provides an effective and scalable data selection approach, offering a significant breakthrough for LLM pretraining.

### Strengths
1. Clear Problem Definition: The paper clearly defines the problem of enhancing large models by selecting a subset of data from a large pool to fine-tune the model, aiming to minimize the loss on a reference set. This provides a concrete and well-defined goal for the research.

2. Balanced Quality and Diversity: The authors use Multi-Armed Bandit (MAB) techniques to balance quality and diversity in data selection. By treating each cluster as an arm and using the Upper Confidence Bound (UCB) to determine cluster scores, the method ensures frequent sampling of high-quality clusters while also exploring less-visited clusters to maintain diversity.

3. Efficient Influence Calculation: The paper introduces an efficient method to estimate the influence of individual data points on the model without full retraining. By extending the influence function to multi-head attention layers and using K-FAC for acceleration, the authors provide a scalable solution for large language models, significantly reducing computational costs.

### Weaknesses
1. Trade-off Between Sample Size and Accuracy: Although the MAB technique can make decisions under uncertainty, there is still a trade-off between sample size and accuracy when estimating the average influence score. Small sample sizes may lead to inaccurate estimates, while large sample sizes increase computational costs. Finding the optimal sample size in practical applications remains a challenge.

2. Complexity and Computational Resource Requirements: Despite the introduction of K-FAC and other acceleration techniques, the computational complexity and resource requirements remain high when dealing with large-scale language models. In environments with limited resources, this method may be difficult to implement or may require significant computation time.

4. Assumptions and Model Dependence: The method relies on certain assumptions when calculating influence scores, such as the approximation of the Hessian matrix and the independence of gradients. These assumptions may not hold in some cases, affecting the effectiveness and reliability of the method. Additionally, different model architectures may require different adjustments and optimization strategies.

### Questions
1. Clustering Quality and Sensitivity: How sensitive is the clustering quality to the choice of the number of clusters (10,000 in this case)? What criteria were used to determine the optimal number of clusters, and how might different numbers of clusters affect the performance of the MAB approach?

2. Generalizability and Robustness: The experiments were conducted on a specific set of downstream tasks and a particular reference set (LAMBADA). How well does the method generalize to other types of tasks or datasets? Have you tested the method on a broader range of tasks or datasets to ensure its robustness?

### Soundness
3

### Presentation
2

### Contribution
3
