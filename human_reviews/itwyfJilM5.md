# Graph Scattering Networks with Adaptive Diffusion Kernels

- Decision: Reject
- Scores: 3, 6, 3, 5, 5

## Abstract
Scattering networks are deep convolutional architectures that use predefined wavelets for feature extraction and representation. They have proven effective for classification tasks, especially when training data is scarce, where traditional deep learning methods struggle. In this work, we introduce and develop a mathematically sound framework for applying adaptive kernels to diffusion wavelets in graph scattering networks. Stability guarantees with respect to input perturbations are provided. A specific construction of adaptive kernels is presented and applied with continuous diffusion to perform graph classification tasks on benchmark datasets. Our model consistently outperforms traditional graph scattering networks with predefined wavelets, both in scenarios with limited and abundant training data.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper studies graph scattering network, and develops an adaptive kernels in diffusion wavelets. The authors further analyze its stability. The experiments show the general improvements over fixed kernel based diffusion wavelets.

### Strengths
1. As the authors mention, most of current scattering networks utilize fixed filter banks. Adaptability is a direction to improve them.
2. This paper is rigorous, giving strict definitions and theorems to support arguments.

### Weaknesses
1. Motivation conflict. The authors firstly acknowledged scattering network are advantageous with limited data availability at Line 31-33 because of no required training. However, the main motivation of this paper is to make existing scattering learnable and adaptive, which scarifies the internal advantages mentioned above.
2. Complexity. For a scattering network with $L$ layer and $h$ children for each parent node, the total number of filters is $\sum_{l=1}^{L}h^l$, an exponential function. If we make all filters learnable, the computing is very high and unbearable.
3. Experiments. The baselines are too old. More graph scattering methods are suggested to compare.
4. Writing. Starting from section 4, all following equations do not have a mark.

### Questions
The authors prove propositions 4.1 and 4.2, Lemma 4.1 and Theorem 4.1. However, only proportion 4.1 describes the proposed adaptive kernel, with language, while the other three prove something for general diffusion wavelet. Therefore, I am not sure their values for this paper, and if these general conclusions have been similarly proposed by previous papers.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper seeks to develop a generalized graph scattering transform which learns the transition matrix $A$ through a kernel inspired by the attentional diffusion method from Chamberlain et al. (2021). 

They take the initial node features $g_u$, map them into an embedding space by a learnable function $W$ and then build a diffusion operator via a kernel derived from the \{W(g_u)\}_{u\in V} and further add in learning of the diffusion operator via a multiheaded attention mechanism. 

After the attention mechanism, they then use the diffusion matrix $A$ to define diffusion wavelets of the form $\psi_j=A^{t_{j-1}}-A^{t_j}$ and use these wavelets to define a graph scattering transform. (This part is ``standard” and similar to other works such as Gama et al. (2019a) and Gao et al. (2019).) Additionally, they prove that their generalized graph scattering transform has similar theoretical properties to other versions of the geometric scattering transform and show strong numerical performance.

Overall, I think this is a good paper which needs a bit of work before it is publication worthy as described below. If these concerns are sufficiently addressed, I will likely raise my score.

### Strengths
The geometric scattering transform (GST) provides a theoretically solid framework for understanding multi-scale GNNs from a graph signal processing point of view. However, the original versions of it are limited in their numerical effectiveness because they are overly handcrafted. This paper shows viable ways of increasing the effectiveness of the (GST) while retaining its nice theoretical properties. This therefore helps bridge the gap between ``things that work well” and ``things which are well understood” which is important since GNNs etc are increasingly used in real-world tasks.

### Weaknesses
Right before the start of Section 4, I think $v$ should be defined in terms of the square-root of the degree vector (since you are using the symmetrized diffusion operator).

The discussion of Forward Euler etc in the end of Section 4.1 seems out of place in this paper. While it is indeed a useful insight from GRAND etc., I don’t see its relevance on diffusion wavelets which are already in discrete time

It seems to me that you should be able to take $N(\beta_A)=1$ 1 in Proposition 4.2 by imitating the proof of Proposition 4.1 of Gama et al. (2019a). (It might also be useful to look at the proof of Proposition 2.2 of Perlmutter et al `` Understanding Graph Neural Networks with Generalized Geometric Scattering Transforms” (2023).) I believe this would then allow you to establish the stability your method to additive noise (as is common in most formulations of the scattering transform).

Related Works:

Important: The second paragraph omits `` Graph Convolutional Neural Networks via Scattering” (Zou and Lerman 2020). This omission is particularly noteworthy because it is the first paper on graph scattering, predating Gama et al. by a couple of months. (The final publication date is later, but this is an artifact of the journal review process.)

Less important: Additionally, the discussion of incorporating learning into geometric scattering (Section 5) should likely also include ``Overcoming Oversmoothness in Graph Convolutional Networks via Hybrid Scattering Networks” (Wenkel et al. 2022). This paper introduced learning into the scattering framework in a different way than the Tong et al. paper that the authors mention. (As noted in Tong, these two forms of learnable scattering, as well as this one, are compatible and can be combined.) It also should likely include `` Scattering Networks for Hybrid Representation Learning” (Oyallon et al. 2018) and `` Separation and Concentration in Deep Networks” (Zarka et al. 2020) which incorporate learning into Euclidean scattering.

Notational inconsistencies:

There is inconsistent use of ``x” vs ``u” in Section 4.1

$A^*$ is used without being defined. Also, why do you use both $^*$ and $^T$ in equation 2? If the matrices are real this should be the same, right? 

In line 221, it would be more natural to call $p_\epsilon$ instead $d_\epsilon to be consistent with the proceed paragraph (or instead call them both $p$)

Very Minor: (Do not affect my score but should be fixed)

Line 54: ``we pursue on alleviating” is awkward. Please rephrase.
Throughout: Things like ``Section 5” are proper nouns and should be capitalized.
Throughout: Some quotation marks point the wrong way (which is an unfortunate artifact of LaTeX sometimes being a pain)

Throughout, some of the equations with $e^{stuff}$ are hard to read and it would be better to write $\exp(stuff)$.

### Questions
Why do we need a metric space, in addition to the measure space? Chew et al. `` Geometric scattering on measure spaces” (2024) should that the geometric scattering transform could be extended to measure spaces where there was not an obvious metric structure (e.g., digraphs). 

In Section 3.1, what norm is being used? This is relevant because $P$ is an asymmetric matrix (which is similar to \overline{P} used in Section 3.2). In Perlmutter et al `` Understanding Graph Neural Networks with Generalized Geometric Scattering Transforms” (2023) it was shown that the natural choice of norm differs depending on whether one uses a symmetric or asymmetric diffusion matrix (for asymmetric matrices you should use a norm weighted in terms of degrees).

In Section 3.1, could you provide examples of when there we be a strict inequality in line 140? It seems to me that in most cases, this will be equality. For the standard diffusion matrix, I believe that the bottom eigenspace will be zeroed out in the limit, but the other eigenspaces are never fully suppressed (which is why you need epsilon in the next paragraph). 

Could the authors please provide examples of setting where having dependence on the sampling density is preferable (as claimed in line 226, ``Naturally…”)? I thought this was typically a nuisance which one would attempt to normalize out (as in line 222).

The unnumbered equations in lines 246-251 are hard to understand. Could the authors please provide some intuition?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This work introduces a method for incorporating adaptive kernels into graph scattering networks. The paper provides theoretical stability guarantees against input data perturbations, ensuring robustness. Experimental results demonstrate that adaptive kernels offer advantages over traditional scattering networks.

### Strengths
1. The theoretical analysis to support the advantages of the adaptive wavelet diffusion.

### Weaknesses
1. The authors highlight the limitations of traditional methods under low-data scenarios. However, the paper lacks theoretical analysis or specific experiments tailored to illustrate how the proposed adaptive kernel-based scattering networks (AGSN) address performance in data-scarce environments.

2. The experimental results reveal that AGSN does not outperform some well-known graph classification techniques.

3. The limitation of The Related Works. There are some works also related to adaptive kernels for graph neural networks, such as [1-2]. It is not clear the advantages of the proposed adaptive wavelet diffusion compared to others.

[1] Sun, C., Hu, J., Gu, H., Chen, J. and Yang, M., 2020. Adaptive graph diffusion networks. arXiv preprint arXiv:2012.15024.

[2] Zhao, J., Dong, Y., Ding, M., Kharlamov, E. and Tang, J., 2021. Adaptive diffusion in graph neural networks. Advances in neural information processing systems.

4. The paper lacks the complexity analysis.

### Questions
What are the advantages of the proposed adaptive wavelet diffusion compared to other methods [1-2] ?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes a mathematically sound framework for applying adaptive kernels to diffusion wavelets, thus overcoming the limitations of traditional graph scattering networks with predefined wavelets.

### Strengths
* Considering the importance of selecting an appropriate kernel, it is promising to develop a framework for application of adaptive kernels in graph scattering networks.

* The proposed framework is bulit on mathematically sound foundation, and stability guarantees with respect to input perturbations are also provided, thus enhanceing its rationality and reliability.

* The experimental results also demonstrated that it consistently outperforms traditional graph scattering networks.

### Weaknesses
The main problem with this paper is that its experiments are not convincing enough.

* Baselines:
    * Given that graph deep learning has developed rapidly in recent years, this paper lacks comparisons against the latest graph deep learning methods.
    * More importantly, some typical graph scattering transform methods are not employed and compared in the experiments, such as GS-SVM [1] and GGSN+EK [2].

* The experimental results can not support the clained superiority. 
Although the authors have given some explanations, why not further conduct some experiments to prove it? 
For example, it's necessary to report the performance of deep learning methods when low training-data availability to prove the meaning of this work.

[1] Gao F, Wolf G, Hirn M. Geometric scattering for graph data analysis[C]//International Conference on Machine Learning. PMLR, 2019: 2122-2131.

[2] Koke C, Kutyniok G. Graph scattering beyond wavelet shackles[J]. Advances in Neural Information Processing Systems, 2022, 35: 30219-30232.

### Questions
Please see the **weaknesses** part.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces Graph Scattering Networks with Adaptive Diffusion Kernel, which enhances traditional graph scattering networks by incorporating learnable kernels while maintaining mathematical soundness. The novel part is that it bridges the gap between fixed wavelet transforms and learnable architectures while preserving mathematical guarantees.

### Strengths
1. the authors propose a novel framework that  incorporate learnable kernels in graph scattering networks.
2. prove that the adaptive kernels maintain symmetry and self-adjointness
3. provide stability analysis for learnable kernels

### Weaknesses
1. As far as I understand, the adaptive kernel is restricted to self-adjoint operators for mathematical convenience.
2. The weak performance raises questions about whether the theoretical advantages of the approach translate to practical benefits.
3. The fundamental question "Why adaptive scattering?" is not convincingly answered in the paper. The theoretical contribution might be interesting, but its practical necessity and benefits are not well established.

### Questions
1. What's the running time compared with other GNN methods, Could you provide runtime comparisons with baselines?
2. In what scenarios does adaptivity help/hurt performance? why we need adaptivity?

### Soundness
2

### Presentation
3

### Contribution
1
