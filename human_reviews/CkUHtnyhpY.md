# When narrower is better: the narrow width limit of Bayesian parallel branching neural networks

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
The infinite width limit of random neural networks is known to result in Neural Networks as Gaussian Process (NNGP) (Lee et al. (2018)), characterized by task-independent kernels. It is widely accepted that larger network widths contribute to improved generalization (Park et al. (2019)). However, this work challenges this notion by investigating the narrow width limit of the Bayesian Parallel Branching
Neural Network (BPB-NN), an architecture that resembles neural networks with residual blocks. We demonstrate that when the width of a BPB-NN is significantly smaller compared to the number of training examples, each branch exhibits more robust learning due to a symmetry breaking of branches in kernel renormalization. Surprisingly, the performance of a BPB-NN in the narrow width limit is generally superior to or comparable to that achieved in the wide width limit in bias-limited scenarios. Furthermore, the readout norms of each branch in the narrow width limit are mostly independent of the architectural hyperparameters but generally reflective of the nature of the data. We demonstrate such phenomenon primarily in the branching graph neural networks, where each branch represents a different order of convolutions of the graph; we also extend the results to other more general architectures such as the residual-MLP and demonstrate that the narrow width effect is a general feature of the branching networks. Our results characterize a newly defined narrow-width regime for parallel branching networks in general.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper challenges the common belief that wider networks generalize better. The authors focus on the narrow-width regime, where the network's width is significantly smaller than the number of training samples, and demonstrate that BPB-GNNs can achieve comparable or even superior performance to wide networks in bias-limited scenarios. This improved performance is attributed to symmetry-breaking effects in kernel renormalization, which enhances the robustness of each branch in the network. The paper provides a theoretical framework for understanding the narrow-width limit and validates it with empirical results, including experiments on the Cora dataset. The findings suggest that narrow-width BPB-GNNs can offer efficient alternatives to wide networks, highlighting their potential for optimizing network design in various applications.

### Strengths
- It is interesting to see the authors introduce the Bayesian Parallel Branching GNN (BPB-GNN) architecture, which differs from previous neural tangent kernel research.

- The analysis presented in Sections 3 and 4 is comprehensive.

- A statistical physics view is applied to the graph neural network, which is novel and interesing.

### Weaknesses
- One main concern is why this work focuses on graph neural networks. It seems that the analysis could be applied to other neural networks, such as MLPs and CNNs. It is hard to see the uniqueness of graph neural networks here.

- The rigor of this work is insufficient. For instance, the definition of $<>$ in Eq. 13 was not introduced.

- To strengthen the experimental section, it would be valuable to include additional real-world datasets (e.g., PubMed, Citeseer) (or tasks beyond node classification).

- To our knowledge, this is the first work to provide a tight bound on the generalization error for GNNs with residual-like structures. However, this may be an overclaim, as other works exist. It would be better to compare with the PAC-Bayes bound for Graph Neural Networks [1].

- A missing reference: [2]

[1] Liao, Renjie, Raquel Urtasun, and Richard Zemel. "A pac-bayesian approach to generalization bounds for graph neural networks." arXiv preprint arXiv:2012.07690 (2020).

[2] Huang, W., Li, Y., Du, W., Yin, J., Da Xu, R.Y., Chen, L. and Zhang, M., 2021. Towards deepening graph neural networks: A GNTK-based optimization perspective. arXiv preprint arXiv:2103.03113.

### Questions
- It is hard to see why $L=2$ directly reduces to the exactly to a 2-layer residual GCN

- In the teacher model (Eq 20) why do you apply a Gaussian distribution to the adjacency matrix? Please correct me if I am wrong.

- From your experiments shown in Figure 5, why don't you show the variance results? Additionally, how do you comment on the claim that increasing the depth can improve performance, despite the oversmoothing problem?

### Soundness
3

### Presentation
3

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
This paper studies the narrow width limit of the Bayesian Parallel Branching Graph Neural Network (BPB-GNN), a summation of several different one-hidden-layer linear Graph Neural Networks. The authors consider the regime that $P, N \rightarrow \infty$ but the ratio of $P/N$ varies where P is the number of training samples and N is the width of the networks. They find that when $P/N$ is small, the network can have better performance.

### Strengths
- this paper studies a new narrow-width regime for parallel branching networks using statistical physics tools and shows that the performance of a BPB-GNN in this narrow-width limit can perform better than its wide-width counterpart.

### Weaknesses
- The model considered in the paper is a linear model, which is very restricted considering the NTK and NNGP of nonlinear networks have been analyzed extensively.
- one of the major results is just a conjecture, which is not proved rigorously.
- The presentation of the paper is quite dense. There are many derivations in the main paper which are hard to follow and I am not sure if the derivations are rigorous. I would suggest the authors organize the main results and give more explanations and intuition for the results.
- See other questions below

### Questions
- $A^l$ is not defined. Are they all equal to $A$?
- I don't quite understand how the integration is calculated in Sec 3.3. How do you linearize and integrate the parameters in Eq (25-29)?
- In Eq. (10), $\alpha$ should be canceled with $N/P$? I don't see why the entropy term dominates when $\alpha \rightarrow 0$ and why it is the NNGP kernel.

---
After rebuttal, raised the score from 3 to 6.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
Existing works often study and characterize the infinite-width limit of neural networks. This work proposes to do the opposite of the study of the narrow width limit of neural networks, and the authors used Bayesian Parallel Branching Graph Neural Network (BPB-GNN) as an example. The authors show that when the width is small (as compared to the number of training examples), each branch of BPB-GNN exhibits more robust learning and empirically shows similar or better performance compared to the wide width limit when the bias term dominates in learning.

### Strengths
I’d like to begin my review by stating the caveat that I am not an expert in this area and I am not very familiar with the existing works; while I have a high-level grasp of the key findings of the paper, I will have to defer the assessment of this work with respect the literature to an expert reviewer.

- The results are pretty intriguing and challenge the common wisdom. As mentioned by the authors, previous works mainly focus on the analysis of the infinite-width limit; studying the narrow width limit is both interesting theoretically and can be potentially more practically useful, as one cannot scale neural networks to infinite width, so the results are mainly theoretical. In contrast, narrow-width can be attainable in real life.
- The paper is largely well-written, with a clear presentation (there are several minor areas of improvement, though — see “Weaknesses”). Experiments are also conducted on real (albeit toy) datasets, which strengthen the rigor and confidence of the theoretical results.

### Weaknesses
1. My biggest concern is the choice of BPB-GNN, which seems to be a very specific construction that should be better motivated. There are several peculiarities in the chosen architecture: for example, there is a distinction between the “number of branches” and “width of network” as a consequence of the non-weight-sharing branches of BPB-GNN. As the authors show in Fig 5, the behavior of the neural network behaves very differently to different values of L and N. My concerns are: 1) Unlike BPB-GNN, the infinite-width limit that the authors have made extensive references to is not limited to a graph learning situation or GNNs. Although not a weakness per se, I am curious why the authors have deliberately chosen a variant of GNN on a semi-supervised task to perform their analysis rather than something more “vanilla” like CNNs or MLPs on simple, fully-supervised setups like image classification. As a result of my previous question, I wonder to what extent the results would generalize to other architectures with different architectures and levels of supervision. 2) As mentioned above and related to my previous point, the distinction of L and N, which will not be present in a “vanilla” architecture without the branching structure, is particular to BPB-GNN. I would appreciate some discussion on to what extent the results will apply in such a case when such a distinction does not exist.
2. As the authors mentioned themselves in the limitations, it is unclear to what extent the results would be applicable when the variance term dominates,  which seems more likely for an over-parameterized network? Could it be the case that the robust learning phenomenon and the superiority over a wider network are caused by better regularization from a narrower network when the learning complexity is low, and that benefit will disappear for more complicated tasks? While it is good that the authors acknowledged some of these potential limitations themselves, I believe additional discussions will be beneficial.
3. Presentation: There are some presentation issues like Fig 1. The different lines (especially the red and orange lines) are difficult to read against the histograms of the same color. The legend fonts are also too small.

### Questions
Please address my comments under "Weaknesses".

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this work, the authors challenge the common belief that wider network widths improve generalization by studying the narrow-width limit of Bayesian Parallel Branching Graph Neural Networks (BPB-GNN). Unlike traditional models where increasing width leads to Gaussian Process behavior, authors show that BPB-GNNs with narrow widths exhibit stronger learning per branch due to symmetry breaking in kernel renormalization (when train data points are more compared to width). The series of experiments and theoretical analysis show that narrow-width BPB-GNNs achieve comparable or superior performance to wide-width models in bias-limited cases, with branch readout norms that reflect data properties rather than architectural hyperparameters.

### Strengths
1. Good work showing narrow-width BPB-GNN showcases better robustness and improved generalization compared to infinite-width approaches.
2. Theoretical justification supports empirical findings
3. The experimental setup is good.
4. Weakness is highlighted.

### Weaknesses
Mentioned below

======= After Rebuttal=====

Accept

### Questions
[1] did a similar study with other architectures. The only difference is that models are trained using SGD. Others [2-4] have focused on generalization and theoretical bounds. The authors should cite such work and comment on key differences between them.

Additional question to author, if we limit the width of GNN, the expressivity of model will reduce [5]. So, isn’t this the issue of learnability? In other words, even with wider widths, unstable models will struggle and have lower learnability; do the authors hypothesize that their approach leads to better stability and learnability? Similar to stable TM[6]. As it's obvious that expressivity would be reduced if you narrow the width, and even theoretically, the generalization would vary. In simple words current framework or architecture in no way is turing complete, it will be reduced to finite automata even with unbounded steps. Hence it would be ideal to mention how given approach trads off expressivity for better learnability and stability. Thus, it's important to mention these points and show comparison. 

The authors don’t provide details about hyper-parameter optimization or detailed analysis regarding the statistical significance of the result. Thus, it is very difficult to gauge the overall importance of the result. For instance, the experimental setup does not mention any details about training, the Size of N = width of hidden layer, the weight dimension W_(l), the number of layers l, the size of readout neurons a_l, 
 and the optimization steps (to update W and a). How are these learned and chosen? There should be information regarding this. p-value should be reported several trials to show results are statistically relevant. 

Can authors also comment on whether the results are true for both cases of homophilous and heterophilous graphs? 


1.	https://arxiv.org/pdf/2305.18411

2.	https://arxiv.org/pdf/2207.09408

3.	https://proceedings.mlr.press/v202/jaiswal23a/jaiswal23a.pdf

4.	https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9535500 

5.	https://arxiv.org/pdf/1907.03199

6.	https://www.sciencedirect.com/science/article/abs/pii/S0020025523016201?via%3Dihub

### Soundness
3

### Presentation
3

### Contribution
3
