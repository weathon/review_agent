# Leveraging Discrete Function Decomposability for Scientific Design

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 2, 6

## Abstract
In the era of AI-driven science and engineering, we often want to design discrete
objects (e.g., circuits, proteins, materials) in silico according to user-specified
properties (e.g., that a protein binds its target). Given a property predictive model,
in silico design typically involves training a generative model over the design
space (e.g., over the set of all length-L proteins) to concentrate on designs with the
desired properties. Distributional optimization, formalized as an estimation of distribution algorithm or as reinforcement learning policy optimization, maximizes
an objective function in expectation over samples. Optimizing a distribution over
discrete-valued designs is in general challenging due to the combinatorial nature
of the design space. However, many property predictors in scientific applications
are decomposable in the sense that they can be factorized over design variables in a
way that will prove useful. For example, the active site amino acids in a catalytic
protein may need to only loosely interact with the rest of the protein for maximal catalytic activity. Current distributional optimization algorithms are unable to
make use of such structure, which could dramatically improve the optimization.
Herein, we propose and demonstrate use of a new distributional optimization algorithm, Decomposition-Aware Distributional Optimization (DADO),
that can leverage any decomposability defined by a junction tree on the design
variables. At its core, DADO employs a factorized “search distribution”—a
learned generative model—for efficient navigation of the search space, and invokes graph message passing to coordinate optimization across all variables.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Decomposition-Aware Distributional Optimization (DADO) as a new algorithm for optimizing discrete functions that exhibit decomposable structures. The method learns a decomposed functional form of an objective and then incorporates a factorized search and message-passing schema to coordinate updates. The method is evaluated against Estimation of Distribution Algorithm (EDA) on both synthetic and real protein design tasks.

### Strengths
- To the best of my knowledge, the core idea of applying message passing for distributional optimization is novel and an interesting approach to the problem of scientific design.
- I appreciate the statistical tests reported in the manuscript.

### Weaknesses
At a high level, my main concerns/questions stem from the learned decomposition of the objective function. While there is evidence that DADO works if a good function decomposition exists, it is not clear that (1) such a good function decomposition always exists; and (2) that the method outperforms black box optimization methods that do not even take advantage of a function decomposition.

1. In the paragraph starting from line 100, the manuscript describes a method to "guess" the functional form of the predictive model and then fitting a model that enforces the guessed decomposition to the available data. However, this might result in a potential distribution shift between the fitted model and the true objective, which may or may not adhere to the "guessed" functional form. This is of particular concern in situations where the "high scoring" candidate designs are rare in the training data and might obey a different underlying functional form compared to other designs. It would be nice to better understand how much this possible distribution shift is actually a concern.
2. Somewhat related to the above comment, but it would be good to ablate the size of the training dataset used to learn the MLP-based function decomposition, instead of using all the data available (line 451). In particular, I feel it would be important to investigate the performance of DADO if only the bottom $x$th percentile of designs were used to learn the surrogate function.
3. The manuscript seems to specifically focus on applications of DADO for protein design (in addition to the synthetic function testing). However, it seems from the title, abstract, introduction, and other parts of the text that DADO is proposed as a general method for many different possible discrete design problems in AI4Science. In this light, it would be good to include additional evaluations across different domains (e.g., circuit, molecule, and material design, for instance) to better illustrate how DADO actually generalizes across different scientific domains. 
4. It seems like a number of baselines are missing from experimental evaluation - in particular, how does DADO compare with methods that do not involve learning a junction tree decomposition at all? This would include any black-box optimization method - PPO, BO, FDA, MCTS (some of which are non-distributional but can still be evaluated using the distributional optimization framework). This would help better clarify the added benefit of even learning a functional decomposition in the first place.
5. It would be good to include results on ablating $K$ in step 1 (line 154).

### Questions
6. Is it ever possible for the functional form to enforce the wrong prior over the input space (for example, if the training data is messy, noisy, affected by a confounding variable, or otherwise unreliable)? If so, what happens to the performance of DADO in these situations?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses a critical bottleneck in AI-driven scientific design—scaling distributional optimization to high-dimensional discrete spaces (e.g., proteins, circuits)—by exploiting function decomposability. Its contributions are well-aligned with the needs of both machine learning (ML) for optimization and domain sciences (e.g., protein engineering), making it a valuable addition to the field.

### Strengths
1. Unlike standard Estimation of Distribution Algorithms (EDAs) or reinforcement learning (RL) policy optimization, DADO explicitly leverages the decomposability of objective functions (via junction trees) to avoid optimizing over intractable full combinatorial spaces. This is a departure from "black-box" optimization approaches that treat the objective as monolithic.
 2. It generalizes classical max-plus message passing (used for exact global optimization) to distributional optimization, replacing hard maximization with soft, sample-based expectations. This enables DADO to balance exploration (via a factorized generative model) and exploitation (via coordinated message passing)—a key innovation not seen in prior work like Factorized Distribution Algorithms (FDAs) or chain-structured RL policies.
3. The paper also explores a practical tradeoff: tuning decomposability (via residue distance thresholds in proteins) to balance predictive model accuracy and optimization efficiency. This empirical insight bridges theoretical algorithm design and real-world scientific constraints.

### Weaknesses
1. The paper relies heavily on structure-based decomposability (AlphaFold3 contact graphs) for proteins but does not explore other practical sources of decomposability: For example, in protein design, decomposability could also be derived from sequence homology (conserved vs. variable regions) or functional annotations (binding sites vs. structural loops). Similarly, in circuit design, decomposability might come from modular components.
2. The paper empirically shows that "loose" decomposability (e.g., t=2.75 Å for GB1) retains predictive accuracy, but lacks theoretical bounds on how much decomposition error DADO can tolerate. 
3. The paper uses D=20 (amino acids) for all experiments but does not address scalability to larger discrete alphabets (e.g., D=100 for small molecules or circuit components)

### Questions
1. For domains without 3D structural data (e.g., novel peptides or synthetic materials), what alternative methods would you recommend to derive decomposability for DADO? For example, could unsupervised learning (e.g., clustering design variables by co-occurrence) be used to 
2. If you intentionally introduce errors into the junction tree (e.g., remove 10% of true residue contacts for TDP43), how much does DADO’s performance degrade relative to using the correct tree? Are there any heuristics (e.g., adding "redundant" edges to the junction tree) to mitigate this error?
3. For D=50 (e.g., small molecules with 50 building blocks), would DADO’s current MLP-based search distribution require prohibitive compute? If so, what modifications would you propose to scale DADO to larger alphabets?

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
3

### Summary
This paper presents an algorithm (DADO) for learning a distribution over a discrete design space to maximize an expected reward, where that reward function can be expressed as a kind of generalized additive model. The authors show favorable performance relative to classical EDA in both synthetic experiments as well as on a task involving the optimization of the score assigned by a protein property predictor developed with the required structure.

### Strengths
The authors are focused on an important and challenging problem in scientific design, as the design space in such settings tends to be discrete or combinatorial and therefore is challenging to explore or optimize over. Their focus on reducing the combinatorial complexity of this search problem to one that is significantly more manageable is a worthwhile pursuit and has real applications. The authors motivated the problem well and provided strong justification for an approach such as theirs for more efficiently optimizing the design space. Further, the generalized additive model structure induced by a junction tree that defines the relationship between the design variables and the property being optimized is a unique choice and lends itself to the paper's originality. Improved efficiency for exploring large combinatorial spaces is of great significance to the field, and the authors have proposed a method that performs capably on the considered problems.

### Weaknesses
The requirement that the decompositional form be known is a very strong one in practice. Indeed, in the protein property prediction experiments, the authors resorted to a very specifically designed predictor to make use of this decomposition--a design which limits the accuracy of the proposed property predictor. Hence, it is unclear to me whether this method has much utility to the community. It would be helpful if the authors could better profile the impact of the decomposition on the accuracy of the underlying property predictor.

Additionally, I found the the experiments to be rather limited. There is one set of experiments on small synthetically curated systems with shallow junction trees, and another on a protein property prediction task using the same model architecture for each of the four considered proteins. It is difficult from these limited experiments to build good intuition on the method's ability to scale to problem size and complexity. It would be helpful if the authors could better profile or characterize the behavior of their method on a more representative set of problems involving the optimization over a discrete design space. It would also be helpful to establish performance against other baselines for such problems, not just a vanilla EDA.

### Questions
How restrictive is the assumed compositionality? From the paper's definition "consider the form f(x) = C1(˜x1) + C2(˜x2), . . . , Cκ(˜xκ), where Ci denotes an arbitrary function on a set of design variables, x˜i," it seems rather general, but the experiments focus on a fairly narrow set of circumstances (i.e., shallow junction trees), so it would be helpful to understand what this definition covers. For example, an arbitrary neural network that takes x as input, transforms it through layers into some d-dimensional embedding, which are then transformed through a linear layer to a scalar output would seem to apply to this definition (in this case, the Ci functions are the individual embedding dimensions which are functions of the x's times the associated weight from the linear layer). Which would indicate to me that this can be applied arbitrarily to neural network predictors, although the experiments suggest this is not in fact the case. Can you clarify?

How well does the method scale to larger sequence lengths compared to those considered in the paper (L < 100)? What about to larger junction trees? Are there important qualitative differences in predictive performance for the underlying property predictors as sequence length increases?

How robust is DADO in settings where the assumed decomposition is incorrect or incomplete?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes Decomposition-Aware Distributional Optimization (DADO) that leverages decomposability and employs a generative model to guide the search for the target distribution.

DADO is compared to standard Estimation of Distribution Algorithm (EDA), a form of Expectation-Maximization that is not decomposition-aware.

The core of DADO is its use of an objective function decomposed into a junction tree, which enables node-level estimation of value (Q- and V-) functions that represent the choice of variables at each edge and node respectively. These value functions are used to update the search distribution via dynamic programming in the form of message-passing.

DADO outperforms standard EDA on synthetic functions as well as a multi-layer perceptron (MLP)-based predictive model of protein property functions.

### Strengths
- DADO is well-motivated and extensively derived
- DADO clearly converges faster than standard EDA
- DADO is a novel, more efficient algorithm than fills a gap in the literature

### Weaknesses
- This paper is hard to read due to large amounts of text, in-line math, and few subsections.
- The evaluation is limited to three synthetic functions and four learned protein property functions
- Standard EDA that is unaware of function decomposability can outperform DADO in the absence of ad hoc hyperparameter tuning.
- The GB1 evaluation appears prematurely ended, as the EDA does not appear to converge by the final training iteration

Typo:263 shaing (shaping)

### Questions
- How long does it take to run DADO and standard EDA?
- Can DADO still be used for larger alphabet sizes or sequence lengths?
- How would DADO change if positions did not share the same alphabet?

### Soundness
2

### Presentation
1

### Contribution
1
