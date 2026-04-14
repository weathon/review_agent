# Feedback Schrödinger Bridge Matching

- Decision: Accept (Oral)
- Scores: 8, 8, 6, 6

## Abstract
Recent advancements in diffusion bridges for distribution transport problems have heavily relied on matching frameworks, yet existing methods often face a trade-off between scalability and access to optimal pairings during training. 
Fully unsupervised methods make minimal assumptions but incur high computational costs, limiting their practicality. On the other hand, imposing full supervision of the matching process with optimal pairings improves scalability, however, it can be infeasible in most applications.
To strike a balance between scalability and minimal supervision, we introduce Feedback Schrödinger Bridge Matching (FSBM), a novel semi-supervised matching framework that incorporates a small portion ($<8$% of the entire dataset) of pre-aligned pairs as state feedback to guide the transport map of non-coupled samples, thereby significantly improving efficiency. This is achieved by formulating a static Entropic Optimal Transport (EOT) problem with an additional term capturing the semi-supervised guidance. The generalized EOT objective is then recast into a dynamic formulation to leverage the scalability of matching frameworks. Extensive experiments demonstrate that FSBM accelerates training and enhances generalization by leveraging coupled pairs' guidance, opening new avenues for training matching frameworks with partially aligned datasets.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposed a novel approach, which introduces Feedback Schrodinger Bridge Matching (FSBM) as a semi-supervised matching framework for addressing distribution transfer issues. The authors initiate their study with the static semi-supervised Optimal Transport (OT) problem. They judiciously incorporate an additional regularization term to capture the interaction between uncoupled and aligned samples, thereby enhancing the transfer mapping process.

The paper's strength lies in its innovative use of a state feedback term to encode information from aligned samples, effectively guiding uncoupled samples' transfer. Additionally, the authors optimize the drift field using divergence, supported by a variational objective.

Starting from the Static Euclidean OT problem, the proposed term $G$ is seamlessly integrated into the cost function $c$, and feasible optimization algorithms are derived using Lagrangian and Hamiltonian formulations. Overall, the paper's motivation, methodology, and experimental results are partially well-presented and contribute to the existing literature.

### Strengths
+ Introducing Feedback Schrödinger Bridge Matching (FSBM): FSBM is the first matching framework capable of leveraging information on partially aligned data pairs, significantly improving efficiency.
+ Semi-supervised guidance: Accelerate training and enhance generalization capabilities by using a small number of pre-aligned pairs as state feedback to guide the transfer mapping of uncoupled samples.
+ Dynamic Target Rewriting: Reformulate the static entropic optimal transport (EOT) problem into a dynamic form to exploit the scalability of the matching framework.
+ Experimental validation: Extensive experiments show that FSBM has better generalization ability under various unseen initial conditions and is faster in training time.
+ Multiple task applications: FSBM performs well in various matching tasks, such as low-dimensional crowd navigation, high-dimensional opinion depolarization, and image translation.

In particular, I think this paper's theoretical derivation and experiments are good, and utilizing partially paired samples is also novel and exciting.

### Weaknesses
1. The key contribution is the proposal of the distance function $G$. However, the motivation of $G$ is not clearly illustrated. Can authors further illustrate why they give such a constraint to paired samples? What is the meaning of $G$? What is the meaning of $d(X_1,x_1^i)$ and $d(X_0,x_0^i)$? Is it relative to clustering? Authors can illustrate the behaviors of paired samples and unpaired samples more specifically. Is the behavior of FSBM related to the "clustering" of trajectories? More discussions, examples, and theoretical derivations to illuminate the motivation of the term $G$ are expected to be shown in the paper.

2. Paired samples and stochastic properties of SDE. As we know, randomness is an important property of SDE/Diffusion/Bridges. From $x_0$ to $x_1$, if $x_0$ is sampled from $\pi_0$, $x_1$ can be stocastic. However, paired samples seem to be deterministic. What is the relation between paired samples and the stochastic property of SDE/Diffusion/Bridges? Please address this concern explicitly in the methodology section.

3. Figure 3 directly shows the effect of $G$. However, Figure 3 is qualitative and not rigorous. The term $u^{\top}\nabla{G}$ forces the trajectory of unpaired samples to the trajectory of paired samples. What is the reason? Please provide a more rigorous analysis of how G affects the trajectories, perhaps through a theoretical derivation or a quantitative experiment that demonstrates the effect.

4. The proof Theorem 3.2. In Eq. 20, authors discretize $G(X_0,X_1)$ into a sum sequence $G(X_{t_i},X_{t_{i+1}})$. The authors seem to assume $\delta t \rightarrow 0$ and transform the sequence of $G$ into $G(X_0, X_1)$. Is the application rigorous? Authors and other reviewers can confirm that. Please discuss under which conditions the transformation (i.e., from the sum sequence to the time interval $[0,1]$) is valid. I suggest that authors provide a proof sketch to make the paper more readable.

5. The contradiction of formulations of Hamiltonian function. As for the formulation of Hamiltonian function, in line 860, authors use $\sup_{u} \langle u,\nabla{a} \rangle-\mathcal{L}(x,u,t)$. However, in Eq. 37, authors $\sup_{u} \langle u,a\rangle-\mathcal{L}(x,u,t)$, which is distinct from line 860. In the following derivations, authors use the formulation in Eq. 37, where the Lagrangian multiplier $a$ is not with the gradient operator. However, in Eq. 47 of Proposition A.1, from line 998 to line 1000, authors eliminate $\langle  \nabla{a^*, u^*}\rangle$ using the product term in Hamiltonian function $H^*(x,a^*,t)$. Then, the authors use the Hamiltonian formulation with $\nabla{a^*}$. Thus, in the proof of Proposition 3.5 and the proof of Proposition A.1, the formulations of Hamiltonian functions are contradictory. I am not familiar with Hamiltonian function, but the contradiction in the proofs should be claimed.

### Questions
Minor issues:
1. The resolution of figures should be improved.
2. Some typos. 
3. What are the advantages of the current FSBM framework's dynamic target design through state feedback? Is there any theoretical evidence that makes it more transparent?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
Summary of the paper 

The paper proposed a novel semi-supervised matching framework, termed as  Feedback Schrödinger Bridge Matching (FSBM), which was designed to improve the efficiency of distribution transport problems by incorporating a small percentage of pre-aligned pairs as state feedback. This framework was developed in order to minimalize  supervision on basis of scalability. It also addressed the limitations of existing methods, which often face trade-offs between computational efficiency and optimal pairing access during training. The authors claim that FSBM enhances generalization and accelerates training by reformulating the static Entropic Optimal Transport (EOT) problem into a dynamic one, thereby facilitating the matching of non-coupled samples.

Contributions 
+ Analysis begins with a static semi-supervised OT problems and derivation of a modified dynamic formulation out of it.
+ Introduction of FSBM, the first matching framework leveraging information from partially aligned datasets and Entropic Lagrangian extension of the variational gap objective used in BM frameworks to match the parameterized drift given a prescribed probability path.
+ Verification of remarkable capability of FSBM through extensive experiments to generalize under a variety of unseen initial conditions, while simultaneously achieving faster training times. These results were consistent across a wide range of matching tasks, from low-dimensional crowd navigation to high-dimensional opinion depolarization and image translation.

Merits
+  Proposal of a semi-supervised framework. This framework leverages minimal aligned pairs, which is a significant contribution to the field, because it effectively addresses the practical challenges posed by fully unsupervised and fully supervised methods. This method is novel and efficient.
+ The paper offers a solid theoretical basis mathematically  by deriving a generalized EOT objective and demonstrating its advantages in terms of efficiency and generalization capability.
+  The extensive experiments conducted across various tasks, including crowd navigation, opinion dynamics, and image translation, provide compelling evidence of the effectiveness of FSBM compared to state-of-the-art methods. Additionally  comparative experiments and their results were also shown to compare results with methods proposed in this paper and baseline methods.
+ The inclusion of detailed performance metrics and visual comparisons strengthens the claims made, illustrating the superior performance of FSBM in practical scenarios. The results were shown vividly.

Weaknesses:
+ While the theoretical aspects are robust, the manuscript could benefit from clearer explanations and simplifications, particularly in sections discussing the derivation of dynamic formulations and coupling optimization. A few formula faults are discovered in the paper.
+ The manuscript lacks a thorough discussion regarding the limitations of FSBM, particularly in relation to scenarios with highly variable datasets or when the fraction of aligned pairs is insufficient for effective guidance.

Suggestions for Improvement:
+ Simplify the presentation of theoretical concepts to enhance readability and comprehension for a broader audience.
+ Expand the comparative analysis to include more specific scenarios where existing methods excel, providing a more nuanced view of FSBM’s performance.
+ Discuss the limitations of FSBM more explicitly, particularly regarding the trade-offs associated with the number of aligned pairs used.
+ Include suggestions for future research that could build on the findings and address any identified limitations.

### Strengths
See Summary

### Weaknesses
See Summary

### Questions
See Summary

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
This paper tackles a semi-paired matching for generative models, based on Schrödinger bridge. In the semi-paired matching problems, along with non-coupled data, a small portion of pre-aligned data pairs are provided. In methodology, the paper is inspired by (Gu et al., 2023a) which investigates the guided optimal transport using a few paired data in static OT setting. The authors utilize a "distance-preserving" scheme to develop the dynamic OT model with entropy regularization, i.e, the Schrödinger bridge. Building on this, the matching algorithm is developed. Experiments on three settings are conducted to evaluate the proposed methods.

### Strengths
- It is interesting to explore the guidance of a small fraction of paired data to data translation in flow/bridge matching.
- It is reasonable to deduce the dynamic version of keypoint-guided optimal transport with a few paired data as guidance for developing the corresponding matching models (I did not check the proof and mathematical deductions).
- The paper writing is clear with clear organization and presentation.

### Weaknesses
- The main weakness to me is the unclear performance gain by introducing the keypoints, from the experiments. The authors reported the W-distance or FID between generated data and real data. However, to my understanding, the tackled problem in this paper belongs to the controllable generation, in which the transported data should preserve certain structures or information of the original source data. For example, in the image-to-image translation (Woman->Man) as in the experiments of this paper, the transported images should preserve as much information as possible of the input image as long as the transported image is a "man". The authors show some transported images, but I can not find the corresponding metric. Maybe a user study is helpful for this issue, e.g., ask the GPT to evaluate the generated images. Without such a metric, the importance of the keypoints can not be illustrated (the same issue exists in the other two experiments).
- It is quite confusing that adding in more keypoints worsens the performance. Is this because the distance-preserving is not effective for such cases? The authors mentioned that Gu et al., 2022 presented the relation-preserving scheme, which seems to be more effective than the distance-preserving scheme as shown in (Gu et al., 2022). Why not consider the relation-preserving scheme instead of the distance-preserving scheme? Unfortunately, both the above issues are left as future work.
- In Eq.(7), the $i$ is selected such that $x_0^i$ is closest to $X_0$. What if selecting $i$ such that $x_1^i$ is closest to $X_1$. The authors should prove that the designed strategy is optimal or justified.

### Questions
- How about including the metrics, e.g., LPIPS distance, perceptual similarity loss, SSIM, $L_2$ distance, etc. between the input and output images for the image-to-image translation? 
- For the user study, could you provide the generated images by different methods to distinct persons, and ask which one is better for the desired translation (e.g., better preserving the identity, background, stricture, etc.)? Meanwhile, could you ask the large language models (LLMs), e.g., GPT, to answer the question, as LLMs show a strong ability to understand text and images?
- For the other experiments, could you train a classifier on the target domain data and apply it to the generated data, to see if the accuracy could be improved with the keypoints as guidance?
- Could you try the relation-preserving scheme? At least, discuss the challenges faced. Meanwhile, could you provide more analysis on the effect of the number of keypoints in-depth, to figure out why a larger number of paired data do not bring performance gain?
- How about selecting $i$ such that $x_1^i$ is closest to $X_1$ or a weighted combination of selecting based on $X_0$ and $X_1$. Which strategy is optimal or most reasonable?

I am happy to increase the score if the questions are well addressed.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
Drawing inspiration from guided optimal transport schemes, this paper introduces Feedback Schrödinger Bridge Matching, a semi-supervised matching framework that uses a small set of aligned pairs to guide the transport map of non-coupled samples.
The proposed approach is evaluated on various distribution matching tasks, such as crowd navigation, opinion depolarization, and unpaired image translation, and compared against other state-of-the-art matching frameworks.

### Strengths
The proposed semi-supervised approach uses only a small percentage of paired data and achieves faster training time.

Experimental validation across a diverse set of tasks demonstrates the effectiveness of the proposed approach.

### Weaknesses
N/A

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3
