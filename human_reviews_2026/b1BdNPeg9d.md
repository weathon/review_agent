# Dual Privacy Protection in Decentralized Learning

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 2, 4, 2

## Abstract
In decentralized learning systems, significant effort has been devoted to protecting the privacy of each agent’s local data or gradients. However, the shared model parameters themselves can also reveal sensitive information about the targets, which the network is estimating. While differential privacy-based decentralized learning can protect network estimates, using excessively large privacy noise variance will significantly reduce the accuracy of network estimates. To this end, we propose a dual-protection framework for decentralized learning. Within this framework, we develop two privacy-preserving algorithms, named DSG-RMS and EDSG-RMS. Different from existing differential privacy distributed learning methods, the designed algorithms simultaneously obscure the network’s estimated values and local gradients, by adding a protective perturbation vector at each update and by using random matrix-step-sizes. Then, we establish convergence guarantees for both algorithms under convex objectives. In particular, our error bound and privacy analysis highlight how the variance of the random matrix-step-sizes affects both algorithmic performance and the privacy of local gradients. Despite using large-variance random step-sizes for stronger gradient privacy, the network’s estimation accuracy in our algorithms can still be improved by choosing a sufficiently small algorithmic parameter $\gamma$. Finally, we validate the practical effectiveness of the proposed algorithms through extensive experiments across diverse applications, including distributed filtering, distributed learning, and target localization.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes two differential-privacy algorithms, DSG-RMS and Enhanced DSG-RMS, to protect shared model parameters and local gradients in distributed optimization. The authors present convergence results for convex and strongly convex objective functions, respectively. Experimental results on distributed filtering, distributed learning, and target localization tasks demonstrate the effectiveness of the proposed approaches.

However, this paper is poorly written, which makes it difficult to follow. Moreover, the convergence results lack theoretical proofs (no appendix is included—perhaps the authors forgot to attach it), making it impossible to verify the correctness of the results given in Theorem 1 and Theorem 2. Furthermore, the paper provides no analysis of the privacy-protection strength/level of the proposed approaches, either qualitatively or quantitatively.

### Strengths
The experimental section is thorough and helps demonstrate the effectiveness of the proposed algorithms.

### Weaknesses
1. **Poorly written:** This paper lacks clarity in both motivation and technical presentation. For example, in the second paragraph of the Introduction, the authors simply list several privacy-preserving methods without discussing their limitations or drawbacks, which makes the motivation of the work unclear. Furthermore, Sections 3.1.1 and 3.1.2 only summarize two existing algorithms without sufficient explanation, making the presentation confusing and not reader-friendly, especially for readers unfamiliar with EDSG. 

2. **Lack of proofs for convergence results:** The proofs of Theorem 1 and Theorem 2 are omitted in the manuscript, and no appendix containing them is provided after the main text.

3. **Lack of differential-privacy analysis and discussion:** The authors aim to address privacy protection, however, there is no analysis of the achieved privacy level. The only privacy-related information that I can find comes from the algorithmic design itself. Specifically, the matrix $M_{k}$ is used to protect gradient information, which seems more like a heterogeneous stepsize design (similar to NIDS). Likewise, the use of the parameter $\tau$ to protect the transmitted messages seems more like a modification of the coupling matrix. Both designs appear to leverage constructive properties of distributed algorithms to achieve privacy protection. Nevertheless, the paper provides very limited qualitative or quantitative discussion on the strength or level of the achieved privacy protection, which is inappropriate for a work that claims to provide privacy guarantees.

### Questions
See the weaknesses above. In addition, I have the following questions:

1. In the Introduction section, the appearance of the parameter $\gamma$ is confusing, as it has not been defined at that point.

2. The paper lacks sufficient comparison with existing literature, which makes its contributions less clear. The authors should provide a more comprehensive review of existing privacy-protection results on distributed optimization (e.g., the PSGT method referred to by the authors), and analyze their limitations rather than merely stating that ``security risks remain." Such a discussion would better highlight the advantages of the proposed approach.

3. The second algorithm, EDSG, appears to be similar to the NIDS algorithm in [r1]. The authors should clarify the differences between the design of NIDS and that of EDSG-RMS.

[r1] Li Z, Shi W, Yan M. A decentralized proximal-gradient method with network independent step-sizes and separated convergence rates. IEEE Transactions on Signal Processing, 2019, 67(17): 4494-4506. 

4. In Fig. 1, why does the MSD of the proposed approaches exhibit sudden jumps?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents a privacy preserving technique for decentralized learning algorithms. A randomized matrix step size is used to protect against model inversion attacks and a non-zero mean perturbation vector is added to enhance the privacy. The paper additionally provides practical applications.

### Strengths
Although the idea of perturbing the step size for privacy is not original (as will be stated in the weaknesses section), to the best of my knowledge, this is the first work that attempts to give theoretical guarantees of the convergence. The paper presents convergence bounds of their method for the convex and strongly convex case under standard optimization assumptions.

### Weaknesses
- The authors do not provide any supplementary material that is necessary for the proofs.

- Contrary to what the authors claim, DP already accounts for statistical averaging attacks thanks to composition theorems. Hence, the purpose of the work is not clear. 

- The idea of perturbing the step size for privacy is not new. The authors do no cite key research paper such as : Enhancing Privacy Preservation in Federated Learning via Learning Rate Perturbation ICCV 2023. 

- Although the methods leverage noise to enhance privacy, no connection seems to be made with DP, and no formalization of the privacy guarantees is provided. 

- The network model assumption (Assumption 1) is not well explained in the paper. What does it mean for the (undirected) communication graph to be strongly connected ?

- The authors refer to inference attacks without specifying whether they mean membership inference attacks or model inversion attacks. 

- No intuition is given to the choice of adding the non-zero vector in the protection step.

### Questions
- Why does it help for the matrix to have non-zero non-diagonal elements ? 

- Why is the added vector chosen to depend on $w_k(n-1)$ for the first algorithm, and on $w_k(n-2)$ for the second one ?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the dual privacy leakage risks in decentralized learning systems: not only local data and gradients are vulnerable to exposure, but the shared model parameters themselves may also reveal sensitive information. To address these two types of risks simultaneously, the authors propose a dual-protection framework. Under convex objectives, convergence guarantees are established for both algorithms, and error bounds explicitly accounting for the influence of network topology are derived. Extensive experiments across various tasks—including distributed filtering, distributed learning, and target localization—demonstrate the practical effectiveness and robustness of the proposed methods.

### Strengths
1. The paper introduces a novel framework designed to simultaneously protect local gradient information and model parameters. Unlike conventional zero-mean noise injection schemes—which are vulnerable to statistical averaging attacks—the proposed approach combines random matrix step sizes with non-zero protection vectors to enhance robustness.

2. The manuscript is well-organized and clearly motivated. The authors center their design around the vulnerability of zero-mean noise to statistical averaging attacks in decentralized settings and use multiple remarks to explain the intuition and role of key algorithmic components. Overall, the presentation is coherent and easy to follow.

### Weaknesses
1. he paper lacks sufficient originality. The authors argue that existing privacy mechanisms relying on zero-mean noise are vulnerable to statistical averaging attacks and thus propose a dual-protection strategy combining non-zero protection vectors and random matrix step-sizes. However, the paper “Masked Diffusion Strategy for Privacy-Preserving Distributed Learning” has already introduced both non-zero-mean protective noise and random matrix step-size mechanisms for privacy preservation. The two works exhibit highly similar structures, core ideas, algorithmic designs, theoretical analyses, and experimental validations. The main differences lie only in the specific formulations of the random matrices and protection noise, while the overall framework and contributions substantially overlap.

2. The privacy analysis is primarily empirical, demonstrating resistance to DLG and statistical averaging attacks through experiments. However, it lacks formal and quantitative privacy guarantees—such as $ (\epsilon, \delta)$-differential privacy proofs or other theoretical formulations—making the claimed privacy protection less rigorous and theoretically unsupported.

3. The manuscript contains several noticeable typographical and notational errors. For instance, the initialization condition for 
$ w_k(0)$ is repeated twice in Section 3.1.1, and there is a subscript error in equation (6d). These issues reduce the clarity and technical precision of the paper.

4. The experimental section lacks sufficient baseline comparisons. The paper presents only the performance of the proposed methods without systematic evaluation against existing state-of-the-art privacy-preserving algorithm. As a result, it is difficult to objectively assess the claimed performance and privacy improvements.

### Questions
1. Could the authors further clarify the main differences between this work and “Masked Diffusion Strategy for Privacy-Preserving Distributed Learning”? Both papers appear to employ random matrix step-sizes and non-zero protection vectors. Please specify the key innovations—either in algorithmic design, theoretical development, or practical application—that go beyond existing work.

2. Have the authors considered providing a formal privacy guarantee for the proposed framework? If not, could they discuss whether such theoretical bounds are feasible or planned for future work?

3. The experiments currently lack comparisons with existing privacy-preserving baselines (e.g., MPD-SG or LDP-based methods). Would the authors be able to include such baselines—either in the rebuttal or supplementary material—to objectively demonstrate the claimed performance and privacy improvements?

4. How sensitive are the proposed algorithms to the variance of the random matrix step-sizes? Could the authors provide numerical or visual evidence (e.g., plots or ablation results) to illustrate the impact of step-size randomness on convergence stability?

5. There appear to be repeated or inconsistent notations—for instance, the initialization condition for w_k(0) in Section 3.1.1 and the subscript error in equation (6d).

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper claims to introduce two novel algorithm for decentralised learning with privacy. After an introduction, the paper introduce the algorithm recap mini assumption and divide very complex formula to characterise the convergence and then perform some numerical experiments

### Strengths
Unfortunately I can't find how to fill this section as the current presentation is very messy and does not allow to really understand what are the contribution of the paper

### Weaknesses
- the paper seems ignorant of all the related walk on the centralised learning in the past years not mentioning void 2006 no all the walk on gossip algorithm in the past year it remains also completely ignorant to past attacks in decentralised setting
- the result are not proved nor analysed
- the clarity of the paper is particularly poor, making hard to properly evaluate the eventual contribution of the paper
- while claiming privacy in the title and at the beginning of the paper there is no result of proof in this direction in the paper

### Questions
It seems unlucky that answers to questions can change the perception of the paper in this current stage so I do not list any question there but authors can choose to comment on the weaknesses listed above.

### Soundness
1

### Presentation
1

### Contribution
1
