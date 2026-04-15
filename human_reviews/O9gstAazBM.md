# Efficient Model-Agnostic Multi-Group Equivariant Networks

- Decision: Reject
- Scores: 5, 5, 3, 3

## Abstract
Constructing model-agnostic group equivariant networks, such as equitune (Basu et al., 2023b) and its generalizations (Kim et al., 2023), can be computationally expensive for large product groups. We address this by providing efficient model-agnostic equivariant designs for two related problems: one where the network has multiple inputs each with potentially different groups acting on them, and another where there is a single input but the group acting on it is a large product group. For the first design, we initially consider a linear model and characterize the entire equivariant space that satisfies this constraint. This characterization gives rise to a novel fusion layer between different channels that satisfies an invariance-symmetry (IS) constraint, which we call an IS layer. We then extend this design beyond linear models, similar to equitune, consisting of equivariant and IS layers. We also show that the IS layer is a universal approximator of invariant-symmetric functions. Inspired by the first design, we use the notion of the IS property to design a second efficient model-agnostic equivariant design for large product groups acting on a single input. For the first design, we provide experiments on multi-image classification where each view is transformed independently with transformations such as rotations. We find equivariant models are robust to such transformations and perform competitively otherwise. For the second design, we consider three applications: language compositionality on the SCAN dataset to product groups; fairness in natural language generation from GPT-2 to address intersec- tionality; and robust zero-shot image classification with CLIP. Overall, our methods are simple and general, competitive with equitune and its variants, while also being computationally more efficient.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper considers the problem of designing model-agnostic group-equivariant networks in an efficient manner. The authors consider two settings: (1) the network has multiple inputs each with a potentially different group acting on it, and (2) the network has a single input and the group acting on it is a large product group. For the former the authors consider linear formulations and characterize the entire space of linear equivariant layers. They then use the obtained equations to extend to non-linear models and show that there exist a design that is universal in approximating invariant-symmetric functions. For the second setting, they propose a method that is more efficient than existing works, at the cost of decreased expressivity.

### Strengths
I think the motivation is strong and the theoretical results are valuable.

### Weaknesses
I think the group-equivariant network that is designed for inputs having large product groups acting on them is claimed to be less expressive but more efficient than equitune (page 2). However the loss of expressivity is not discussed in section 3. I think this part is fundamental and should be stressed.

### Questions
1. For the results in Table 1, why do you consider the input as ordered? What if instead we consider the input as a set of images, and apply Maron et al 2020? How would it compare?
2. Related to 1, I think the multi-image task does not really test the first scenario, as the input groups are the same and therefore Maron et al 2020 can be directly applied.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper tries to address the computational problem of constructing model-agnostic group equivariant networks for large product groups, and provides efficient model-agnostic equivariant designs for two related problems with different input specifications. For different problems, this paper proposes new fusion layer designs, which can be extended beyond linear models, and model-agnostic equivariant designs for large product groups. Experimental results are provided for different applications, such as language compositionality, natural language generation, and zero-shot classification, showing high computational efficiency than the existing ones.

### Strengths
This paper is well-organized and well-written. The motivation is stated in a clear way, and the objective is easy to follow.

The theoretical findings are organized in a proper way, and the proofs are given in a rigorous way.

The computational complexity could have been given in detail.

### Weaknesses
The related work could be expanded to highlight the difference between this work and the existing ones. The novelty and contributions could have been highlighted.

More institutions and explanations of the theoretical findings could be provided after each theorem. It is not easy to figure out how important these theoretical results are and how they could be used to guide the practical designs.

In addition to the comparison of computational complexity, it is unclear how and to what extent the proposed designs are practically useful.

### Questions
Add more intuitions and explanations for theoretical findings and possible insights to guide practical designs.

Add more meaningful experimental results to show the practical usefulness of the proposed designs.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper aims at designing a model-agnostic group equivariant network for direct product groups.

### Strengths
The topic is interesting and important.

### Weaknesses
It could be that I misunderstand something. But from what I understand from the construction, the model is vacuous, and can only apply for the trivial symmetry.

In page 2, the notion of “invariant-symmetric” seems strange. Note that you simply define the identity action of $G_2$ on the range of $f$ in the space $Y$. Namely, for any $y=f(x)\in Y$, you assume that $g_2y=y$. In what sense is this an interesting model of symmetries? It describes no symmetry, or more accurately, just the trivial symmetry.

In the construction in (1), from what I understand, for the  “invariant-symmetric” property of

 $L^{IS}_{G_2,G_1}$ 

you need to assume that $G_1$ acts trivially on $X_1$, namely, $gx=x$, and for the “invariant-symmetric” property of $L^{IS}_{G_1,G_2}$ you need to assume that $G_2$ acts trivially on $X_2$, namely, $gy=y$. This means that your construction cannot describe any structure but the trivial symmetry. The proof of Theorem 1 in the appendix also shows that this is what you construct. Hence, the whole construction is vacuous. 

What am I missing?

If I misinterpreted the construction, please explain. I will be happy to change my score.

### Questions
Some other problems that I found in the paper before realizing that the model (1) is problematic:

The term “independent groups”, for example at the bottom of page 1, should be replaced by commuting subgroups, or by “all inputs are acted upon independently by separate groups”.

Page 2: “invariant-symmetric”: Note that you simply define the identity action of $G_2$ on  the range of $f$ in $Y$. Namely, for any $y=f(x)\in Y$, you assume that $g_2y=y$. In what sense is this an interesting model? It describes no symmetry.

Section 3.1 Multiple Inputs - you mean that the sequence $(X_1,\ldots,X_N)$ is the input, not that this is a set of inputs. You should also formulate the setting as follows: the direct product group $G=(G_1,\ldots,G_N)$ acts on the input space $\mathbb{R}^{d_1,\ldots,d_N}$.

Large product groups : ``the subgroups $g_i$ act in the same order.’’ The order does not matter since $G$ is a direct product group. The subgroups $G_i$ commute.

“ whereas for constructing G-invariant models we do not need commutativity” It is not a matter of need. Since the subgroups $G_i$ commute by definition of direct product of groups, and by definition of group action, the action of the subgroups $G_i$ must commute.

Equation (1): what is $L^{IS}_{G_2,G_1}$ 

?
 You did not define it. You also did not define $L_{G_1}^{Eq}$ and $L_{G_2}^{Eq}$. There is a problem with this construction as I wrote above.

At this point I have to admit that I stopped reading. If my assessment of (1) is correct, the paper should be rejected. If I misunderstood the construction I apologize.

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper considers the problem of constructing equivariant networks in a model-agnostic manner. In particular, this paper tackles the setting in which the group can be decomposed as a product group. This leads to two principal tasks: 1.) the multi-input setting in which each group in the product acts on its respect input and 2.) the single input setting but the symmetry on this data type has a product structure. For the first setting, the authors propose IS fusion layer and characterize the entire space of linear equivariant functions with multiple inputs. They then extend this to the non-linear setting and prove universal approximation capabilities. Experiments are conducted in both settings and include multi-image classification and downstream applications of compositional generalization and language. The proposed approach is sometimes competitive with previous approaches but has the benefit of linear computational complexity w.r.t. to the number of groups.

### Strengths
The paper has a few strengths that I would like to highlight. First, the paper builds upon two recent papers by Basu et. al 2023 and Kim et. al 2023. and this allows the paper to lean on existing methodology. Thus the overall idea is relatively straightforward to understand. Moreover, the presented theory can be understood equally easily as it largely follows from the author's definitions and results in Maron et. al 2020.

### Weaknesses
Despite the stated strengths above; I have strong reservations regarding this paper. The first one is on the motivation. It is unclear to me why we would want to make an existing trained model equivariant. This is certainly the assumption in the downstream language experiments, but this is not at all a convincing demonstration. The task is contrived and does not fit in the broader equivariant literature.

**Large discrete product groups?**

In addition, to the lack of coherent motivation, I also found the claims in the paper to be unsupported. A large theme of this work is on building equivariance across **large discrete** product groups. In the multi-input experiments (Table 1) you have $N=4$ and the $C_4$ group. This is not a large product or large individual group. Similarly, in the compositional generalization experiment you write "The product groups are made of three smaller each of size two, and the largest product group considered is of size eight". Again this is not a large product group. A similar criticism can be attributed to the intersectional fairness experiments. So I find the entire claim and motivation for doing this work lacking. In fact, I would argue that you can just as easily do frame averaging and canonicalization in these settings. Thus at **minimum** these should be baselines. 

**Experimental design and results**
The entire choice of experiments in this paper leans heavily on Basu et. al 2023's results. But I don't think the authors thought that the experimental setup in that paper might also be problematic in terms of highlighting the claims and goals. First, there is really no standard equivariant benchmark from many of the seminal equivariant papers. For example, you could have considered molecular datasets where you have $S_n$ and $SE(3)$. $S_n$ would be a much larger discrete group and you could consider discrete subgroups of $SE(3)$. Given the large literature of equivariant models in this space, the lack of this benchmark is alarming. In addition, if you really want to show discrete product group structure then there is a large body of work on latent space disentanglement via Linear symmetries which started from the seminal work of (Higgins et. al 2018). Completely ignoring this line of work and its benchmarks which are almost tailor-made to the setting considered in this work is questionable. Finally, I find the choice of compositional generalization and intersectional fairness via group theoretic notions quite a contrived task that runs counter to the actual practical goal of this work which is to scale up equivariant models for product groups.

With regard to results, there are many areas of improvement. To start off, the proposed approach does worse than equitune in some experiments (e.g. Fig 2). The authors do not have a convincing argument on why this is acceptable outside that their proposed approach has better-scaling properties w.r.t. to the number of groups in the product. Unfortunately, as I stated above this is not a large product so this is weird. Secondly, obvious baselines are missing. These include having an actual equivariant (architecture) model that is trained (not fine-tuned) post hoc in this manner. Also, frame averaging and canonicalization should be included.

### Questions
Please consider adding the following experiments.

1.) An experiment where $S_n$ is in the product. This can be a molecular task or not, but the equivariant literature has many examples of benchmarks.

2.) Adding an experiment with latent product symmetry. This would be a really convincing and better experiment in your setting.

3.) Can you please add the baselines mentioned in the weakness section?

4.) Can you please highlight the computational cost (iters/sec, flops, training time, inference time, etc...) of your approach versus frame averaging for your tasks. My guess is that it is quite similar given how small the group is.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor
