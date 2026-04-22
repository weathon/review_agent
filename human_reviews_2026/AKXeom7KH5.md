# Unrestrained Simplex Denoising for Discrete Data. A Non-Markovian Approach Applied to Graph Generation

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Denoising models such as Diffusion or Flow Matching have recently advanced generative modeling for discrete structures, yet most approaches either operate directly in the discrete state space, causing abrupt state changes. We introduce simplex denoising, a simple yet effective generative framework that operates on the probability simplex. The key idea is a non-Markovian noising scheme in which, for a given clean data point, noisy representations at different times are conditionally independent. While preserving the theoretical guarantees of denoising-based generative models, our method removes unnecessary constraints, thereby improving performance and simplifying the formulation. Empirically, _unrestrained simplex denoising_ surpasses strong discrete diffusion and flow-matching baselines across synthetic and real-world graph benchmarks. These results highlight the probability simplex as an effective framework for discrete generative modeling.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a diffusion model for graph generation based on a denoising process on the probability simplex. While previous simplex-based diffusion models for discrete data modeling are based on a Markovian process, the paper uses a non-Markovian noising that assumes independence across noisy states, which eliminates unnecessary constraints. Experimental results on graph generation tasks show improvements in small 2D molecule generation while being on par with SOTA baselines in unattributed graph generation.

### Strengths
- The writing is easy to follow and related works cover most of the relevant works.

- Application of n-dimensional simplex diffusion for graph generation seems to be a novel approach, to the best of my knowledge.

- The proposed method shows improvement in 2D molecule generation tasks.

### Weaknesses
- Why is the dependency across noisy states in Markovian diffusion considered unnecessary? This dependency is simply a byproduct of the Markovian design, which is not inherently necessary or unnecessary. Designing a non-Markovian process could potentially lead to better performance, though the reason for this improvement is not clearly explained. However, this alone does not imply that dependency across noisy states is unnecessary.

- Why is the proposed simplex-based diffusion model better than discrete diffusion models? The motivation in Figure 1 seems to apply to flow matching on simplex, and unclear how this connects with discrete diffusion models. Also, Dirichlet diffusion may show different results in the experiments in Figure 1.

- The experiments are limited to small-scale graph benchmarks: 2D molecules have at most 38 nodes, and unattributed graphs have at most 187 nodes (SBM). Is the method scalable to graphs with a larger number of nodes? This may be validated on the synthetic grid graphs, where controlling the number of nodes is possible and has been used in the baseline like Grum.

- The main claim of the paper is not validated sufficiently. The paper claims non-Markovian noising removes unnecessary constraints and improves performance, but results on unattributed synthetic graphs do not show this. Is the main claim wrong?

### Questions
Please address the questions in the weakness section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper is clearly written. The formulations for both the reverse and forward processes are very clean. Figure 1 clearly illustrates the comparison with linear interpolation. The experimental results are complete. Overall, I find the paper sound and complete.

### Strengths
The paper is clearly written. The formulations for the reverse process and the forward process are very clean. The formulas are easy to follow and convey the authors’ ideas. Figure 1 clearly illustrates the comparison with linear interpolation, as commonly done in discrete flow matching. I find the paper sound and complete.

### Weaknesses
1. Figure 1 clearly illustrates the disadvantages of previous discrete flow matching formulations. However, even with the thorough discussion of related work in Section 2.2, it is still difficult for me to clearly understand the differences or relationships between the proposed method and (1) Dirichlet diffusion, and (2) Dirichlet flow matching in spirit. Could the authors explain explicitly on this point? This will make the contribution of the paper more clear. At the same time, compared to the other group of methods that maps the prediction back to the simplex, what is the intuition that Dirichlet diffusion may function better?
2. There is related work that could be included in the comparison or discussed about, such as Graph BFN (Smooth Interpolation for Improved Discrete Graph Generative Models), which also performs denoising within the simplex space.
3. Additional ablations could further support the method: for example, does the noise schedule matter, or do any hyperparameters require tuning for different datasets?

### Questions
1. In line 346, the authors mention: “but we can also approximate this prior using a model trained with the probability path…”
Could the authors explicitly explain how this approximation is achieved without specifying the prior directly?
2. Another point of interest in the current research direction is generating graphs with fewer sampling steps. I wonder whether the authors have explored reducing the number of steps.

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
3

### Summary
The paper proposes a diffusion-based method for generating graphs with categorical attributes. Noised versions of categorical variables (node and edge attributes) are represented as points on a simplex, and the denoising process operates in this space. The motivation is to design a diffusion process that avoids the discontinuities of discrete noising. The forward noising is non-Markovian, i.e., noised samples are independent and depend only on the clean data. The method is evaluated on molecular graphs and synthetic (SBM, planar graph) datasets.

### Strengths
- Generative modeling for discrete data is a relevant topic. The idea of applying non-Markovian diffusion to graph generation is conceptually interesting.
- The authors provide a clear theoretical exposition, especially related to noising on the simplex. 
- In general, the paper is clearly written and easy to follow. The related work is presented well. 
- The experimental results are solid and show that relaxing the Markov assumption doesn't seem to harm empirical performance.

### Weaknesses
The downside of denoising in the simplex space is that graph sparsity is lost: the communication graph inevitably becomes fully connected. This is not explicitly discussed in the paper, though there is a related work section on scalable methods.

The authors note that simplex-based diffusion for graph generation has already been done in (Liu et al. 2025). The use of Dirichlet distributions for categorical variables appears to be a relatively straightforward generalization of the earlier Beta-distribution based approach in Liu et al.

Likewise, the challenges related to noising on the simplex are already studied in Stark et al. (2024). The proposed Voronoi-based probability construction is theoretically sound, but it does not by itself represent a major conceptual advance.

### Questions
The use of non-Markovian noise is interesting, though the theoretical analysis in Section 4.2 would benefit from clearer references to related work on this topic to highlight which aspects are novel.

**Typos and grammar:**
- l.16: "yet most approaches either operate
directly in the discrete state space, causing abrupt state changes and discontinuities." --> remove either
- l.234: the definition of L would be useful here
- l.273 $Cat(\pi)$ undefined
- l.303: duplicate "univariate case" 
- l.306: "Let assume" --> Assume
- l.365: "... framework supports both. Because ..." --> replace period with comma
- l.957: The denoisers are Graph Neural Network,  --> Networks
- l.981: $W^lsrc, W^ltrg, and W^ledge$   --> $W^l_{src}$; other similar errors occur in this section

### Soundness
3

### Presentation
3

### Contribution
2
