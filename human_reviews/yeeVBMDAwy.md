# Variance-enlarged Poisson Learning for Graph-based Semi-Supervised Learning with Extremely Sparse Labeled Data

- Decision: Accept (poster)
- Scores: 6, 6, 6, 6

## Abstract
Graph-based semi-supervised learning, particularly in the context of extremely sparse labeled data, often suffers from degenerate solutions where label functions tend to be nearly constant across unlabeled data. In this paper, we introduce Variance-enlarged Poisson Learning (VPL), a simple yet powerful framework tailored to alleviate the issues arising from the presence of degenerate solutions. VPL incorporates a variance-enlarged regularization term, which induces a Poisson equation specifically for unlabeled data. This intuitive approach increases the dispersion of labels from their average mean, effectively reducing the likelihood of degenerate solutions characterized by nearly constant label functions. We subsequently introduce two streamlined algorithms, V-Laplace and V-Poisson, each intricately designed to enhance Laplace and Poisson learning, respectively. Furthermore, we broaden the scope of VPL to encompass graph neural networks, introducing Variance-enlarged Graph Poisson Networks (V-GPN) to facilitate improved label propagation. To achieve a deeper understanding of VPL's behavior, we conduct a comprehensive theoretical exploration in both discrete and variational cases. Our findings elucidate that VPL inherently amplifies the importance of connections within the same class while concurrently tempering those between different classes. We support our claims with extensive experiments, demonstrating the effectiveness of VPL and showcasing its superiority over existing methods. The code is available at https://github.com/hitcszx/VPL.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes effective graph-based semi-supervised learning approaches for sparsely labeled data. To improve the accuracy, the proposed approach adds a term of the label variance to the objective function of graph-based semi-supervised learning. The paper conducted experiments to show the effectiveness of the proposed approach.

### Strengths
- Graph-based semi-supervised learning is an important research problem in the field. 
- The proposed approach is simple and intuitive. 
- The theoretical properties of the proposed approach are well discussed in the paper.

### Weaknesses
- The compared approaches in the experiment are somewhat old.
- Graph structures used in the experiment are unclear from the description of the paper.

### Questions
The paper compares V-Laplace and V-Poisson to other graph-based approaches in the experiment. However, the compared approaches are somewhat old; the most recent one was published in 2020 (POISSON). Similarly, it compares V-GPN to other GNN approaches. However, they are not state-of-the-art, although GNN is a well-studied technique. Is the proposed approach more accurate than recent approaches?

Although k-NN graphs were used in the experiment, the detailed experimental settings are unclear from the descriptions of the paper. k-NN graphs are used in the experiment? What is the number of edges from each node? How do you set edge weight? Is the proposed approach used even if other graph structures are used besides k-NN graphs?

In the datasets used in the experiment, it seems that labels evenly exist. Could you tell me whether the proposed approach is useful for labels of screwed distribution? Please tell me whether the proposed approach is more accurate than other approaches even if labels do not sparsely exist (i.e., we have plenty of labels)? In addition, how do you determine the number of iterations in the proposed approach?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces a graph-based transductive semi-supervised learning method, which modifies Laplace and Poisson learning techniques by incorporating a variance-enlarged term for regularization. The authors present algorithms for these 'variance-enlarged' learning methods. Additionally, they propose a novel message passing layer with attention for Graph Neural Networks (GNNs) to enhance label variance, based on the 'label propagation' step of their algorithms. These contributions are tested in scenarios with limited labeled data and compare favorably against other methods.

On the theoretical front, the paper explores both discrete and variational cases. In the discrete case, the variance enlarged approach corresponds to a reduction of the edge-weights, which, under certain conditions, strengthens connections between nodes of the same class and weakens those between nodes of different classes. In the variational case, the minimizer of theoptimization problem is expressed as the solution of a PDE.

### Strengths
- The paper is well written.
  
- The paper presents a variety of contributions of theoretical and practical nature.
  
- The proposed algorithms are simple yet effective as shown in the experiments section.

### Weaknesses
I suspect Lemma 4.1 could have an error and therefore all proofs that derive from it. See questions.

### Questions
1. **Iterative vs. Linear Solution:**
   
   - In Algorithm 1, you've chosen an iterative approach for solving V-Laplace Learning. However, the conventional Laplace learning approach can be solved directly through a linear system ([1,2]). I'm curious if your method could also utilize a linear system solution. Is there a specific reason for the iterative approach? Is it faster?

2. **Relation to Previous Work:**
   
   - In [2], Laplace learning was associated with the probability of sampling a spanning forest that separates the seeds. Do you think your approach could also have a similar interpretation in this context?

3. **Consideration of Directed Graphs:**
   
   - As far as I understand, your approach does not consider directed graphs. Does your approach and the theoretical insights extend to the directed case as well?

4. **Convergence Dependency on Parameters:**
   
   - I'm interested in understanding how the convergence of your algorithms is influenced by the value of $\lambda$. Could you shed some light on this relationship?

5. **Clarification on Lemma 4.1:**
   
   - In the final step of Lemma 4.1, it seems that the sum of $q_j$ is factored out of the norm. However, this step isn't clear to me. Could you provide a more explicit explanation of how this is done? To illustrate, if I consider $(u_1,u_2)=(0.5,1)$, $(q_1,q_2)=(0.5,0.5)$, $\lambda=1$ and $w_{12}=w_{21}=1$, the equation doesn't seem to balance. The right term of the inequality is equal to 
$\sum_i^n\sum_j^n(w_{ij}-\lambda q_iq_j)||u(x_i)-u(x_j)||_2^2=$ 

$=2\left((w_{12}-\lambda q_1q_2)(u_1-u_2)^2\right)=2\left((1-0.5^2)(0.5-1)^2\right)=0.375$

While the left term is equal to

$\sum_i^n\sum_j^n(w_{ij})||u(x_i)-u(x_j)||_2^2-\lambda\sum_i^nq_i\left|\left|u(x_i)-\sum_j^nq_ju(x_j)\right|\right|^2_2=$

$=2\left(w_{12}(u_1-u_2)^2\right)-\left(q_1\left(u_1-\left(q_1u_1+q_2u_2\right)\right)^2+q_2\left(u_2-\left(q_1u_1+q_2u_2\right)\right)^2\right)=$
$=2\cdot0.5^2-\left(0.5\left(0.5-\left(0.5\cdot0.5+0.5\cdot 1\right)\right)^2+0.5\left(1-\left(0.5\cdot0.5+0.5\cdot 1\right)\right)^2\right)\\
     =0.4375$

This clarification is crucial as Theorem 3.1 and Proposition 4.3 depend on this Lemma.

6. **Typos**:
   
   - In proposition 4.3 references Theorem 4.1. It should be Lemma 4.1.
   
   - Table 7  does not contain the accuracies for the V-Poisson method. 
   
   - Table 5 and 6 do not contain any clarification regarding the meaning of the bold values.

[1] Grady, "Random Walks for Image Segmentation" (2006)

[2] Fita Sanmartin et al. “Probabilistic Watershed: Sampling all spanning forests for seeded segmentation and semi-supervised learning” (2019)

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper addresses the issue of traditional graph-based semi-supervised learning leading to degenerate solutions when labeled data is extremely sparse. It introduces VPL, which mitigates this problem by increasing the variance of predictions for unlabeled data. Furthermore, based on classical Laplace learning and Poisson learning, the paper proposes V-Laplace and V-Poisson as improvements. Extensive experiments have demonstrated the effectiveness of these approaches.

### Strengths
- This paper provides an overview of classical graph-based semi-supervised learning tasks, and the proposed idea, while simple, is highly effective. Its parameter-free nature makes it more appealing.
- The writing in this paper is of excellent quality, and the motivation and introduction of the proposed method are presented in a clear and easily understandable manner.
- This paper provides a thorough and reliable theoretical analysis.
- In addition to the general graph-based SSL methods like Laplace Learning, this paper also extends to GNN-based method and proposes V-GPN.

### Weaknesses
- Disclaimer: I am familiar with GNN-based semi-supervised learning and have knowledge of Laplace Learning and Poisson Learning, but I am not familiar with their applications in non-graph structured data. I noticed that the experiments primarily focus on datasets like (Fashion) MNIST and CIFAR-10. It would be beneficial to expand the experiments to larger datasets, such as ImageNet.
- Typos: e.g., lambda in the caption of Figure 1.

My other concern is the practical value of such graph-based (parameter-free) semi-supervised learning methods. As shown in Table 2 and Table 3, despite significant improvements over the baselines, the accuracy of the proposed method still falls short of being satisfactory. To my knowledge, parameterized models like ResNet and ViT-based self-supervised learning methods tend to perform better in cases of label sparsity. Therefore, in resource-abundant scenarios, it seems that having a parameter-free model with relatively poorer performance may not be very meaningful.

### Questions
I hope the authors can answer the last point in Weaknesses, and I am glad to raise my score if my concern can be addressed.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Semi-supervised learning (SSL) aims to leverage a vast amount of freely available unlabeled data alongside a small sample of expensive labeled data to improve the classification performance of a learnt model. Graph SSL techniques are a popular class of approaches where by constructing a graph with data points nodes and relationship edges, information can be propagated across the graph to make predictions on unlabeled data. A well-recognized limitation of typical Graph SSL techniques is the problem of degenerate solutions where, when the labeled sample is sparse, the nodes far away from any labeled sample can converge to a constant and uninformative value.

This paper proposes a simple and intuitive fix to degeneracy issue by regularizing the node values to be different from one another through a term that increases the variance between node values. Clear theoretical insights have been provided to show that, when the graph edges connect same class nodes more often than others, variance enlargement can amplify the importance of edge weights connecting vertices within the same class, while simultaneously diminishing the importance of those connecting vertices from different classes, thus leading to improved solutions.

Experimental results show salient gains due to variance enlargement regularize on a variety of datasets.

### Strengths
* SSL is an important problem and the paper addresses the crucial issue of node degeneracy in Graph SSL. As such, the problem is well-motivated
* The solution is simple and intuitive and theoretical connections are provided to explain the inner workings of the proposed technique
* Experiments are conducted on a wide range of datasets and show significant gains which demonstrates the utility of the technique

### Weaknesses
* Many variants of Graph SSL have been proposed in the literature. It will be interesting to discuss the effect of variance enlargement on those also beyond V-GPN that the paper explores.

### Questions
Can variance enlargement help with other Graph SSL approaches besides GPN? If so, a discussion on where and why it helps and does not help, can be useful and interesting.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
