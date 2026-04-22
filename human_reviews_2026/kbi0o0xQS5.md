# Optimal Affine Framework for Steering Generative Models

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 2, 4, 6

## Abstract
An idea of steering intermediate representations of generative models has recently emerged as a simple yet powerful approach for controlling aspects of generated texts and images. However, despite the simplicity of the approach, no theoretical framework has yet been built around steering. In this paper, we aim to bridge this gap, building theory around concapt steering. First, we provide theoretical link between steering and affine concept erasure framework, showing that widely used steering setup for erasing unwanted behaviours or concepts from generative models is a special case of LEACE, a closed-form method for affine concept erasure in neural networks. Next, we consider the task of concept switching, the aim of which is to change information about unwanted concept or behaviour in the model’s representations into another, more desired concept or behaviour. Here our contribution is two-fold: first, we formulate a theoretical framework for this task, adapting existing affine concept erasure framework used for concept erasure. Then, we identify weaknesses of the resulting framework, and propose a new, improved one, that we call MIDSTEER (MInimal Disturbance concept STEERing). Our results show that MidSteer performs favourably on a variety of tasks modalities and models, including image generative diffusion models and LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a theoretical connection between affine concept erasure and concept steering, and then adapts affine erasure to the task of concept switching. In addition, a new approach, MiDSteer is proposed, and experiments are conducted to compare it to the prior techniques in both LLMs and generative image models.

### Strengths
The proposed technique makes sense and the results show the improvement versus the state of the art techniques.

The theoretical bridging of the different approaches is a sdolid contribution.

### Weaknesses
My score is currently low because of the, to me, confusing presentation. If this could be rectified, then the score could improve. For example, it is unclear what LEACE is. It is not well introduced, and we have to infer what it is. Similarly, CASteer is mentioned just above equation (18), but it is not clear what it is and how it differs to LEACE. Even MidSteer itself is not well introduced. There is a tangential definition of it at line 310, which refers back to Thm 5. 

The results in section 4.2.1 experiments are for CASteer and LEACE. These are existing techniques and not the (new) MiDSteer. Why is this included?

Minor:
- "subscript" of $s^c^ should be "superscript"
- miscellaneous grammatical issues, e.g. "of the concept C in generation result of the model"
- Figure 1 is not referenced in the text as far as I could see.
- there seem to be two betas. One in line 322 and a different one (?) in eqn (20). That is confusing. (and the reference to eqn (21) in line 323 should presumably to to eqn (20))
- the heat map colour for beta in the figures does not work well
- in line 471, should "4" be "Fig 4"?

### Questions
Is the constraint in Theorem 5 on the rank being full satisfied in practise? Was that investigated? If it is not, how does it impact the results?

In section 4.2.1, unrelated concepts are used for testing. How is the lack of relation to the erased concept established? And would it be interesting to investigated related concepts vs unrelated concepts to see the impact of erasure on the different degrees of relatedness?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Topic: Steering intermediate representations of generative models

Definition of steering: adding a steering vector to the intermediate representations to control the generated results.
* concept deletion = Erasing unwanted behaviors or concepts
* concept switching = changing a concept with another

Problem statement:
* Steering is empirically developed without theory.
* Naive steering often perturbs unrelated features.
* Affine concept erasure does not solve concept addition or switching.

Contribution:
* a theory of concept steering: steering is a special case of LEACE, a closed form method for affine concept erasing in neural networks
* Minimal disturbance concept steering (MiDSteer), an improved version of concept steering

### Strengths
Originality: not sure

Quality:
1. L329 The experiments cover various models: Llama 2, Qwen 2.5, SDXL, and SANA.

Clarity:
1. L353, L371, L400 The task and desiderata are clearly noted.

Significance: not sure

### Weaknesses
1. The previous knowledge and the proposed knowledge should be separated. Which parts are the contribution? It would help the readers recognize the significance of this paper.
2. Significance of the contribution should be apparent to the readers. Why are the theorems and proofs non-obvious?
3. L171 Guardedness should be defined in the paper for integrity, even though it is cited. The connection between guardedness and Theorem 1 should be explained. Please be kind to the readers.
4. L310 MiDSteer, the proposed method, is not proposed. 
5. LEASE should be L054 cited and L196 explained.
6. Figure 4: The images and the caption do not match: the dog is still there in CASteer
7. Sentences should be easier to be understood. Please remove redundant or uninformative words and write precisely.

### Questions
My questions are apparent in the weaknesses. Resolving them in the rebuttal may improve my rating.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors provide a theoretical connection between representation steering affine concept erasure/switching. In particular, they first show that concept steering is a special case of LEACE, which is a closed-form method for affine concept erasure. Next, the authors study the task of concept switching. By arguing that vanilla concept steering can switch the two targeted concepts, e.g. untruthfulness to truthfulness and vice versa, they propose MIDSteer to only allow one way mapping.

### Strengths
* The paper is generally well-written.
* I think the problem of concept switching is a more controlled variant of concept erasure, which is nice.
* Empirical results for LLMs show superior concept switching results compared to other methods.

### Weaknesses
* The result section on LLMs seem to lack qualitative results, while the result section on diffusion models lack quantitative results. 
* While MIDSteer perform better than other methods for concept steering, I am not sure if CASteer and LEACE are appropriate baselines since they are purely designed for concept erasure.
* I think the paper would benefit from more justifications on why concept switch is an interesting problem or why is it more preferred than concept erasure.

### Questions
* Can the authors provide some qualitative results on LLMs output when MIDSteer, CASteer, and LEACE are applied?
* What concepts are used to measure erasure for the LLM experiments?
* Can the authors provide Pareto efficiency frontiers plot for the SDXL experiments?
* What is the purpose of Section 4.2.1, I do not see MIDSteer being compared for concept erasure, yet the authors are comparing it with the same methods for concept switching?
* For concept switching, I would be interested to see the Pareto plot using the y-axis as the concept score on c2 and on c1 separately, rather than just the difference between concept scores of c2 and c1.

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
This paper introduces a unified theoretical framework for optimizing affine mappings for concept steering in neural representations. They show that the existing approach for concept deletion and flipping can be viewed as special-case solutions for a general constrained optimization problem. Then, they propose MidSteer, a general framework for concept steering. The main novelty compared with the prior works is that they whitens activations and apply steering transformation on the standardized representation space.

### Strengths
The work elegantly connects existing techniques for concept deletion, flipping, and transfer within a single constrained-optimization framework, offering novel insights and guiding future work. MidSteer generalizes prior works by removing the assumption of standardized activations, making the framework applicable to real, anisotropic model embeddings.

### Weaknesses
From an implementation standpoint, MidSteer reuses the same affine transformation derived in earlier work, with $\beta$ now treated as a hyperparameter, and the only non-trivial technical novelty seems to be activation standardization. Similarly, the theoretical contributions, though sound, follow relatively directly from existing frameworks without introducing substantially new analytical insights. Happy to be corrected on this. 

The presentation of the final algorithm can be improved. I could not find any discussion of the computational cost of estimating $\Sigma_{X, X}$. I suggest that the authors add the pseudo-code for this algorithm. What will the dimension of $X$ be for the language model? Is it seq_len x hidden_dim? Would the algorithm require instantiating the whole matrix of $\Sigma_{X, X}$?

I do not directly work in this field, so I cannot comment on how significant the experiment results are. Therefore, I set my confidence to 2.

### Questions
See weakness.

### Soundness
3

### Presentation
2

### Contribution
2
