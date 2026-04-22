# LipNeXt: Scaling up Lipschitz-based Certified Robustness to Billion-parameter Models

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 8, 6, 6

## Abstract
Lipschitz-based certification offers efficient, deterministic robustness guarantees but has struggled to scale in model size, training efficiency, and ImageNet performance. We introduce \emph{LipNeXt}, the first \emph{constraint-free} and \emph{convolution-free} 1-Lipschitz architecture for certified robustness. LipNeXt is built using two techniques: (1) a manifold optimization procedure that updates parameters directly on the orthogonal manifold and (2) a \emph{Spatial Shift Module} to model spatial pattern without convolutions. The full network uses orthogonal projections, spatial shifts, a simple 1-Lipschitz $\beta$-Abs nonlinearity, and $L_2$ spatial pooling to maintain tight Lipschitz control while enabling expressive feature mixing. Across CIFAR-10/100 and Tiny-ImageNet, LipNeXt achieves state-of-the-art clean and certified robust accuracy (CRA), and on ImageNet it scales to 1–2B large models, improving CRA over prior Lipschitz models (e.g., up to $+8\%$ at $\varepsilon{=}1$) while retaining efficient, stable low-precision training. These results demonstrate that Lipschitz-based certification can benefit from modern scaling trends without sacrificing determinism or efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies certified robustness in the direction of Lipschitz networks. Algorithmically, it (1) designs an approximate-then-correct optimization algorithm which speeds up the optimization in the orthogonal space, and (2) proposes a convolution-free method via shifting (rolling) data spatially in the orthogonal embedding space. The final method is able to scale to networks with billion parameters and achieves strong performance.

### Strengths
The proposed method exhibits strong rigor and novelty. Almost all designs are supported by strong theoretical analysis and well-motivated. The final overall performance demonstrates the effectiveness of the general algorithm. Detailed ablation studies are provided in the appendix to demonstrate the effectiveness of individual modules.

### Weaknesses
The overall algorithm seems costly. On the memory side, the optimizer requires a copy of the full parameter, thus doubles the memory cost, which is especially concerning for a model with billion-parameters. On the computation side, the main results on conducted on 8xH100 GPUs, which seems hard to reproduce by academic labs and not scalable to harder tasks. I will not attack the main contribution due to the costs though.

The parameter efficiency is of question. All comparisons, although meaningful in the story, are conducted with regard to models with far fewer parameters. I will not deny that it is meaningful to further scale and the baselines might suffer from scaling, but the key question is, are those parameters necessary to achieve strong performance; which part of the model consumes the most parameters?

As a minor comment, the introduction about certified training is misleading. Line 43 introduces randomized smoothing and Lipshitz networks as the two major trends, while other deterministic certified training based on bound propagation is as active, see [1] as an example.

[1] https://arxiv.org/abs/2406.04848

### Questions
1. An orthogonal projection, $R$, is conducted before the spatial shifting. Is this a learned orthogonal matrix or a manually constructed matrix?

2. Is the spatial shifting method meaningful without projecting into an orthogonal space first? The current design arguably introduces much more parameters (the $R$ matrix is a full-dimensional square matrix) than convolutions.

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
2

### Summary
LipNeXt introduces a novel 1-Lipschitz architecture that provides efficient, deterministic robustness guarantees for large models. By leveraging manifold optimization and a Spatial Shift Module, it achieves state-of-the-art certified robustness accuracy on CIFAR-10, CIFAR-100, Tiny-ImageNet, and ImageNet.

### Strengths
The paper provides solid empirical evidence, along with an ablation study, to support the main claims.

### Weaknesses
Table 2 presents results with additional data; however, I noticed that the total number of parameters for the proposed model is 256M, which is significantly larger than the competitors. Could the authors provide results for a smaller model configuration, such as L32W1024?

### Questions
* Could the authors provide a formal definition of certified robust accuracy (CRA) and clarify how the corresponding data is generated? As mentioned in line 40, ε represents the radius of a ball in the p-norm. However, the specific value of p does not seem to be provided. Additionally, since the introduction begins by discussing adversarial robustness, I assume that CRA refers to accuracy against adversarial examples. Could the authors clarify which attack methods were used for the assessments?

* I find the role of the β-Abs nonlinearity somewhat unclear. It is introduced between lines 320-357 as a replacement for MinMax or MaxMin. While I do not doubt the correctness of the proof, the connection to the overall proposed method is not immediately clear from a narrative perspective. Additionally, the improvement gains from β-Abs, as shown in Table 7, appear to be relatively modest compared to those in Table 1. Besides, could the authors clarify whether β-Abs could be applied to existing models, and if so, under what conditions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a new approach for constructing Lipschitz neural networks and demonstrates improved certified robustness compared with several baselines. The method primarily focuses on two key components. They are Manifold Optimization and Spatial Shift Module. The latter one is novel to me.
Overall, I lean toward accepting this paper. I may further increase my score if the authors can address my concerns properly.

### Strengths
1.	The paper is well-organized and clearly written.
2.	The proposed manifold optimization and spatial shift techniques are interesting and technically sound.

### Weaknesses
1.	Some important baselines are not discussed in the related work section, such as Sandwich and BRONet.
2.	The method integrates several known techniques, making it somewhat difficult to assess the individual effectiveness of each component.

### Questions
1.	I suggest separating Sections 3.1 and 3.2–3.3. This would allow the paper to present Sections 3.2–3.3 together with other orthogonal methods such as LOT, BRO, or Cholesky for comparison. I also recommend adding an ablation study combining those orthogonal baselines with Sections 3.2–3.3 to better highlight their contribution.
2.	In Table 2, the proposed model appears larger than the baseline. Could the authors include LipNeXt L32W1024 in the table for a fairer comparison? Additionally, I am curious about the scaling ability of the baselines, as the proposed method shows clear scaling behavior when synthetic data are used.
3.	In Table 3, could the authors provide the EDM results for the proposed method? Including these results would help illustrate how scaling up combined with EDM could further enhance performance.
4.	I do not fully understand the relationship between β-Abs and min–max. In Theorem 2, is it necessary to use this $R$? Also, please verify whether line 751 is correct.
5.	SOC introduces other activation functions such as Householder and GroupSort. Can the paper compare with these? Otherwise, the advantage of β-Abs is unclear.
6.	What does the dagger ($\dagger$ in line 351) symbol in Table 1 represent?
7.	Why does the paper use R in Equation (8)? Is this orthogonal mapping essential and trainable?
8.	The periodic polar retraction seems somewhat unusual. The paper claims it helps eliminate errors, but could the authors clarify its specific effect? Why not use other orthogonal projections such as LOT or Cholesky?
9.	It would strengthen the paper if the authors could include AutoAttack results to verify that the certified robustness implementation is correct.
10.	Regarding Section 3.1, I believe the following reference may be missing:
Bader, Philipp, Sergio Blanes, and Fernando Casas. "Computing the matrix exponential with an optimized Taylor polynomial approximation." Mathematics 7.12 (2019): 1174.

### Soundness
3

### Presentation
3

### Contribution
3
