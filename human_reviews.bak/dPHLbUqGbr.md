# Fast, Expressive $\mathrm{SE}(n)$ Equivariant Networks through Weight-Sharing in Position-Orientation Space

- Decision: Accept (poster)
- Scores: 5, 6, 8

## Abstract
Based on the theory of homogeneous spaces we derive *geometrically optimal edge attributes* to be used within the flexible message-passing framework. We formalize the notion of weight sharing in convolutional networks as the sharing of message functions over point-pairs that should be treated equally. We define equivalence classes of point-pairs that are identical up to a transformation in the group and derive attributes that uniquely identify these classes. Weight sharing is then obtained by conditioning message functions on these attributes. As an application of the theory, we develop an efficient equivariant group convolutional network for processing 3D point clouds. The theory of homogeneous spaces tells us how to do group convolutions with feature maps over the homogeneous space of positions $\mathbb{R}^3$, position and orientations $\mathbb{R}^3 {\times} S^2$, and the group $SE(3)$ itself. Among these, $\mathbb{R}^3 {\times} S^2$ is an optimal choice due to the ability to represent directional information, which $\mathbb{R}^3$ methods cannot, and it significantly enhances computational efficiency compared to indexing features on the full $SE(3)$ group. We support this claim with state-of-the-art results —in accuracy and speed— on five different benchmarks in 2D and 3D, including interatomic potential energy prediction, trajectory forecasting in N-body systems, and generating molecules via equivariant diffusion models.

*Code available at [https://github.com/ebekkers/ponita](https://github.com/ebekkers/ponita)*

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work derives point-pair attributes uniquely identifying the equivalence class of the point-pair and uses these attributes for equivariant group convolutional network. Its application on 3D point clouds achieves state-of-the-art performance and scalability.

### Strengths
1. Conceptually simple, powerful, and efficient model verified with extensive experiments.
2. A detailed introduction of previous work.

### Weaknesses
1. There are so many contents from previous works that I can hardly tell the novelty. From my point of view, only Theorem 1 is new, but it is still quite easy to prove.

2. Though this work is titled with $SE(n)$, it only discusses $n=3$ case.

### Questions
1. Does this work have provable expressivity? For example, can it universally approximate continuous equivariant function?

### Soundness
3 good

### Presentation
2 fair

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
This paper studies the message passing scheme for signals define on space $\mathbb{R}^3\times S^2$ with SE(3) symmetry, which then can be used to construct neural architectures for pointcloud data, by lifting the pointcloud from $\mathbb{R}^3$ to $\mathbb{R}^3\times S^2$.   In particular, they give the explicit form of an invariant and bijective edge embedding $a_{i,j}$ for two points $x_i,x_j\in \mathbb{R}^3\times S^2$. This edge embedding serves to weight the pairwise message passing.  

They argue that the proposed method is more efficient than expressive $SE(3)$ group convolution (which requires integral on $SE(3)$) and meanwhile more expressive than convolution on $\mathbb{R}^3$ (e.g., SchNet). Experiments on potential prediction, molecule generation as well as trajectory prediction show the better performance and efficiency of the method.

### Strengths
- the analysis on invariant feature $a_{i,j}$ seems technically sound and solid to me
- the experimental part is extensive

### Weaknesses
- As there are many approaches for lifting a pointcloud, it may not be clear why the space chosen may find the best trade-off between efficiency and expressiveness. 
- So I think it could make the work stand out if we can find some applications that require to directly deal with signals on $\mathbb{R}^3\times S^2$. 
- Also just a minor personal idea: I think the word "weight sharing" makes me think of constructing linear transformation invariant/equivariant to certain symmetry. While here we are actually seeking for the full invariants describing the equivalence class.

### Questions
- There are two P$\Theta$NITA columns in Table 1. Should be a typo?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
When considering a homogeneous space $X\simeq G/H$ and features that are functions $f:X\to \mathbb{R}^C$, equivariant linear layers are convolutions (Equation 1). The authors propose to extend such layers to the case where $X$ is replaced by a discrete geometry (a graph of points of $X$), which allows for equivariant graph convolutions. They remark that the direct generalization (Equation 5) is not intrinsic as it depends on a represent of $x\simeq [g_x]$; they characterize filters that are intrinsic and give applications to quotients of SE(3) and SE(2).

### Strengths
The paper is very well written, the problem well posed and meaningful, and the solutions natural. I find the paper very interesting.

### Weaknesses
I don't see any

### Questions
None

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
