# Discovering Generalizable Governing Equations for Graph Dynamical Systems with Interpretable Neural Networks

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 6, 2, 2

## Abstract
The discovery of symbolic governing equations is a central goal in science; yet, it remains a formidable challenge, particularly for graph dynamical systems, where the network topology further shapes the system behavior. While artificial intelligence offers powerful tools for modeling these dynamics, the field lacks a rigorous comparative benchmark to assess the true scientific utility of the discovered laws. This work establishes the first rigorous benchmark for this task, moving beyond simple fitting metrics to evaluate discovered laws based on their long-term stability and, critically, their out-of-distribution generalization to unseen graph topologies. We introduce the Graph Kolmogorov-Arnold Network (GKAN-ODE), an architecture tailored for this domain, and propose a structure-aware symbolic regression method to leverage its inherent interpretability. Across a suite of synthetic and real-world graph dynamical systems, we demonstrate that symbolic models extracted from neural architectures, particularly our GKAN-ODE, achieve state-of-the-art performance and generalize to unseen networks, significantly surpassing existing baselines. This work presents the first systematic benchmark in this domain, clarifying the expressivity-interpretability trade-offs and offering a  pathway from observational data to fundamental physical understanding, providing a critical new tool for data-driven discovery in network science.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper addresses the challenge of discovering symbolic governing equations for graph dynamical systems directly from observational data. The authors propose a novel architecture, the Graph Kolmogorov-Arnold Network (GKAN-ODE), which adapts KANs for this task by modeling self-dynamics and interaction dynamics separately. The model is enhanced with internal multiplicative nodes to better capture physical interactions. To extract human-readable formulas, the paper introduces a principled, structure-aware "Spline-Wise" (SW) symbolic regression algorithm. The authors establish a rigorous benchmark, evaluating GKAN-ODE and other baseline methods on synthetic and real-world epidemic datasets. The core findings demonstrate that neural-based models, particularly GKAN-ODE, significantly outperform sparse regression methods in long-term stability and generalization to unseen graph topologies, successfully recovering ground-truth equations while being more parameter-efficient.

### Strengths
1. Comprehensive and Rigorous Benchmarking. The paper establishes a high-quality benchmark that addresses a clear gap in the literature.
2. Novel and Well-Motivated Method. The proposed GKAN-ODE framework is technically sound and introduces several valuable innovations.
3. Principled Symbolic Distillation and Interpretability Analysis. The paper offers a thoughtful approach to extracting and analyzing symbolic models.

### Weaknesses
I am not familiar with this field, and my expertise is insufficient to critique the article or offer suggestions.

### Questions
NA

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
4

### Summary
The paper focuses on the task of discovering governing equations in dynamical systems on networks.
In particular:
- It introduces GKAN-ODE, a method based on KAN, adapted for graph data (with static topology), and where the authors introduced multiplicative nodes.
- It introduces a symbolic regression algorithm, SW, that extracts a symbolic equation from a trained KAN.
- It compares GKAN-ODE with other methods, namely TPSINDy, LLC, and a method based on MLPs, GMLP-ODE.
- In doing so, it offers a benchmark to study and compare such methods.

They argue that GKAN-ODE, by exploiting the interpretability naturally inherent in the KAN, offers an accurate and transparent method for studying complex systems.

### Strengths
**Originality**

As far as I know, the new elements introduced in this paper are indeed original:
- the GKAN network, with multiplicative nodes,
- and the spline-wise symbolic regression algorithm to retrieve the symbolic equation from the trained model.

**Quality**

The methodology and evaluation are sound, with no overemphasised claims, and good support of ablations and experiments over diverse methods and datasets.

**Clarity**

This is one of the clearest papers I have read in a while: the discussion flows in a very logical way, and I haven't spotted any typos.

**Significance**

The studied problem is interesting and relevant. This work offers a solid and sound overview of some state-of-the-art methods, even without considering the authors' own contribution on the topic, this element would already make it significant.

### Weaknesses
Here I list some concerns I have, in no particular order:

1. As stated already in the title, this work focuses on discovering governing equations for graph dynamical systems. I believe a better effort could have been made to clarify how much more difficult this task is compared to the case with no graph. 
2. This work tries to find a difficult balance between proposing a new method and offering a benchmark of existing ones. In some parts, one has the impression that the authors used this approach to counter the limited results achieved on their proposed method.
3. In relation to point 2 above, it's a bit unclear what the advantage of the proposed spline-wise symbolic regression algorithm is. By looking at the results in Figure 1 (left), it's difficult to advocate for its use instead of GP. Table 2 also doesn't show encouraging results, both when compared with Table 1 and with Table 12. 
4. The authors argue that the cost above is balanced by a "more direct", "more faithful", "granular", "fully transparent" view of what the model learned. I understand that interpretability is tricky to measure, but I think these points should be better supported. In the field of discovering governing equations, it seems that the only way is to check if the discovered equations align with the ground truth, and the results seem to suggest that SW has the worst alignment. It doesn't help that the comparison to the original SR method by KAN's authors has only been included in the appendix. 
5. The addition of multiplicative nodes is justified solely via experiments.

### Questions
Referring directly to the weaknesses listed above:
1. Paragraph 4.2 shows that you validated the different methods also by changing the topology of the datasets. How did you do it? Could you maybe add to your benchmark methods that don't make use of the graph at all?
3. In paragraph 5.2, you argue that the coefficients obtained with SW are "*slightly* different", or that you get "additional *small*-coefficient terms", but, given that SW achieves worse MAE, can we really say that these errors are small?
4. Concerning the interpretability advantage of KAN+SW: 
  - I recommend expanding on why this approach improves interpretability and giving more substance to the above claims. 
  - Considering how the interpretability point was made, it seems that there is an overlap: the output of the method serves both to measure its performance and also to justify how interpretable the model itself is. Usually, in XAI, one measures the interpretability of a model by relating the output to the input. Is a similar approach viable here? For example, by linking the presence of some symbolic terms in the discovered governing equation to some features of the input?
  - I recommend adding the results of Table 10, where they lie without a direct comparison, to Figure 1.
5. The addition of multiplicative nodes is justified solely via experiments. Could you offer a theoretical justification too?



Two extra questions:

6. You use the five-point stencil method to build the time derivative of the trajectories. Have you performed an ablation on this? Since the $\text{MAE}_\text{traj}$ metric depends only on the trajectory itself, not on its derivative, I was wondering how robust these methods are when changing the way to estimate the derivative.
7. Have you considered using a KAGNN (*Bresson et al.*) in (2) instead of a simple KAN?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work focuses on graph dynamical systems (GDS) and proposes a novel GKAN-ODE framework that integrates Kolmogorov-Arnold Networks (KAN) with neural differential equations. Combined with an interpretable Spline-Wise symbolic regression pipeline, this framework enables the automated discovery of universal governing laws underlying graph-structured dynamical systems.

### Strengths
-By leveraging the spline-based structure of the KAN model and the white-box interpretability of the Spline-Wise mechanism, the proposed method successfully transforms deep network representations into interpretable symbolic expressions, which helps reveal the model’s internal learning dynamics.

-To promote reproducibility and support open science, all code and experimental configurations associated with this work have been publicly released.

### Weaknesses
-The current version of the paper lacks an intuitive illustration of the overall architecture. A well-structured framework figure would greatly enhance the presentation by clarifying the relationships and information flow among different components of the proposed model.

-The core evaluation relies on numerical integration to obtain X^(t) and compute MAEtraj(Eqs. 5–6). Since this process is autoregressive, the accumulated error may vary with the integration step size and the choice of integrator. However, the paper does not provide an analysis of integrator robustness, leaving the stability and reliability of the results insufficiently validated.

-The authors propose "multiplicative nodes without hyperparameters" to enhance physical interaction modeling, but they lacked a visual analysis of why it is superior to LLC/GMLP.

-The noise robustness analysis is not in-depth enough. The experimental design includes noise versions of different SNRS, but the systematicness of the derivative estimation and anti-noise mechanism in the paper is relatively limited.

-From the experimental results in Figure 1, it can be seen that the GKAN-ODE-SW method is not as good as GKAN-ODE-GP, failing to demonstrate the effectiveness of the proposed method in handling graph dynamical systems.

### Questions
Q1：To what extent does the long-term rollout error depend on the choice of numerical integrator and step size? While the paper reports overall evaluation results, why are the detailed experimental settings—such as the integrator type, step size, or tolerance—omitted?

Q2：Given that the primary metric MAEtraj relies on numerical integration, would different integrators (e.g., RK4, DOPRI) or step-size configurations change the ranking of model performance? Has the sensitivity of this metric to the integration method been analyzed?

Q3：In the synthetic tasks, the authors use OOD sets with varying topologies and initial conditions for selecting the final symbolic expressions, whereas in the real epidemic scenario, model selection is performed only on the training set due to the lack of OOD data. Has the potential “model selection bias” between these two settings been examined? 

Q4: The authors emphasize that MAEtraj is independent of the ground-truth equations and, therefore, more stringent due to error accumulation. Could the authors provide a correlation analysis between MAEtraj and MAEeul to show which metric better aligns with scientific correctness when their results diverge?

Q5：What is the impact of the proposed KAN architecture on the experimental results? Furthermore, please provide a detailed description of its specific implementation in the paper.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This is a nicely written paper that uses a multi-step learning process and new architecture to learn symbolic differential equations. They then try to extract symbolic structure from their learned network as well as a few alternative networks. They try this approach on a few symbolic ODEs on networks to define a benchmark set.

### Strengths
Their goal is valuable: interpretability for graph-structured ODEs. I also appreciate that they're trying fundamentally different architectures with edge nonlinearities instead of node nonlinearities. I think their tasks are reasonable targets.

### Weaknesses
Overall I'm sympathetic to the concept but I have multiple major concerns.

1) I think they overstate the importance of their benchmarks. It’s a limited set of tasks. These are fine tasks, I don’t object at all, but I would not elevate them to the level of a rigorous benchmark that makes its own contribution.

2) They talk about the greater interpretability of their GKAN networks, but if I understand correctly, they are really doing a non-symbolic fit with a different architecture, then using symbolic regression to distill their model. They do the same thing for other models too, and again get symbolic expressions that have similar complexity and performance. So it’s unclear what the advantage of their approach is. Their claim is that somehow putting nonlinearities on edges rather than nodes is valuable, but I don’t see evidence for this.

3) If they’re targeting symbolic nonlinearities, why not just use symbolic nonlinearities as their basis functions, instead of using splines as intermediaries?

4) They explore a particular family of composition, sums and products. This is creeping toward both node nonlinearities, and toward directly fitting symbolic structure. So it seems like they’re trying to have it both ways: “let’s use edge nonlinearities instead of node nonlinearities — except for here.” And “let’s fit neural networks and then later find symbolic structure — except for here.”

5) I don’t understand why their Spline-wise regression is unique to KANs. They’re just ways of fitting a symbolic function to another function, and that should work for general functions as well as for splines.

Overall, this is a reasonable approach, but I’m not convinced that it’s a substantial advance. I may be mistaken about any of these points, and I’m happy to be corrected.

### Questions
What is the main contribution here? Am I missing a substantive performance difference or improvement in interpretability from the GKAN versus the GMLP?

### Soundness
2

### Presentation
3

### Contribution
2
