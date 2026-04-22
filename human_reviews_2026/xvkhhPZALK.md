# Doubly Smoothed Decentralized Stochastic Minimax Optimization Algorithm

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Decentralized stochastic minimax optimization has recently attracted significant attention due to its applications in machine learning. However, existing state-of-the-art methods use learning rates of different scales for the primal and dual variables, making them difficult to tune in practice. To address this problem, this paper proposes a novel doubly smoothed decentralized stochastic minimax algorithm. Specifically, in terms of algorithm design, we update both the primal and dual variables using smoothed gradients and introduce novel approaches to handle the computation and communication of the auxiliary variables introduced by the smoothing technique. On the theoretical side, for nonconvex-PL problems, our convergence analysis reveals that the learning rates for the primal and dual variables are of the same scale. Moreover, the order of the condition number in our convergence rate is improved to $O(\kappa^{3/2})$. To the best of our knowledge, this is the first time it has been improved to such a favorable order.  Finally, extensive experimental results validate the effectiveness of our algorithm.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addressed the nonconvex-PL min-max optimization problem on a decentralized network. The proposed algorithm incorporates gradient tracking as the decentralized optimization framework, STORM variance reduced gradient estimator to improve the iteration complexity and the smoothing technique to control the progress of outer minimization and inner maximization.

### Strengths
- A better dependence on the condition number $\kappa$ is guaranteed.

### Weaknesses
### Algorithm Design
- Additional 2 smoothing variables, 2 variance reduction variables and 2 gradient tracking variables incur significant memory overhead.
- Each iteration requires the gossip communication of 6 variables, incurring significant communication overhead.

### Technical Contribution
- The proposed convergence guarantee depends on the notion of $\mathcal{O}(\epsilon_1, \epsilon_2)$-stationarity (Definition 4.1), which deviates from the standard notion of $\mathcal{O}(\epsilon)$-stationarity (also Definition 4.1). In Remark 4.5, the authors suggested applying a centralized algorithm (Yang et al., 2022) to transfer $\mathcal{O}(\epsilon_1, \epsilon_2)$-stationary solution to the standard $\mathcal{O}(\epsilon)$-stationary solution, possibly implying that the proposed algorithm cannot achieve $\mathcal{O}(\epsilon)$-stationary solutions in a decentralized setup.

### Experiment
- The presented numerical experiments are limited to small scale setting. Given the conference's emphasis on neural network training, larger scale experiments such as larger decentralized network, larger dataset and complex models optimization are preferred to demonstrate the practical performance of the proposed algorithm.
- It is not mentioned what kind of classifier is employed in the experiment.


### Writing
 - In line 055, "that" is redundant.

### Questions
- Can the authors explain / give intuition to why the proposed decentralized algorithm can achieve $\mathcal{O}(\kappa^{3/2})$ dependence, which is better than the state-of-the-art dependence of $\mathcal{O}(\kappa^2)$ in centralized setting? Can the same $\mathcal{O}(\kappa^{3/2})$ dependence be achieved in the centralized setting, or is it only achievable by decentralized algorithms? Furthermore, do the authors have any clue regarding the optimal dependence to $\kappa$, for instance, if there exists lower bound guarantees to the dependence of $\kappa$?

- Can the authors explain why in Remark 4.5, the optimization problem in line 372 satisfies PL condition in both $x$ and $y$?

- Have the authors considered whether variance reduction on the $y$ variable is necessary or not? Intuitively, the nonconvex minimization over $x$ (at a rate of $\mathcal{O}(1/\sqrt{T})$ [a] or variance-reduced acceleration of $\mathcal{O}(1/T^{2/3})$  (Cutkosky & Orabona, 2019)) should converge slower than the PL stochastic gradient maximization over $y$ (at a rate of $\mathcal{O}(1 / T)$ [b]). Therefore, I suspect that the variance-reduction over $y$ does not contribute to the acceleration of the overall convergence.

[a] Ghadimi, Saeed, and Guanghui Lan. "Stochastic first-and zeroth-order methods for nonconvex stochastic programming." SIAM journal on optimization 23.4 (2013): 2341-2368.

[b] Karimi, Hamed, Julie Nutini, and Mark Schmidt. "Linear convergence of gradient and proximal-gradient methods under the polyak-łojasiewicz condition." Joint European conference on machine learning and knowledge discovery in databases. Cham: Springer International Publishing, 2016.

### Soundness
2

### Presentation
2

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
This paper proposes \textbf{Smoothed$^2$-DSGDAM}, a doubly smoothed decentralized stochastic gradient descent–ascent with momentum for nonconvex–PL minimax problems.  The algorithm applies the smoothing technique to both primal and dual variables and incorporates variance reduction to reduce gradient variance. It eliminates the need for setting different learning rate scales between primal and dual updates, achieving a unified learning rate and improving condition-number dependence from $\mathcal{O}(\kappa^3)$ to $\mathcal{O}(\kappa^{3/2})$. Theoretical analysis shows convergence rate $\mathcal{O}(1/\epsilon^3)$ and communication complexity $\mathcal{O}(\kappa^{3/2}/((1-\lambda)^2\epsilon^3))$.

### Strengths
1. First decentralized minimax optimization algorithm that applies smoothing to both primal and dual variables.

2. Achieves same-order learning rates, facilitating hyperparameter tuning and improving the stability of the training process.

3. Improves dependence on the condition number $\kappa$ and achieves a faster convergence rate compared to previous works.

4. Introduces an effective combination of smoothing and variance reduction tailored for decentralized settings.

5. Provides clear theoretical comparisons with existing algorithms (e.g., DREAM, DGDA-VR, DM-HSGD).

### Weaknesses
1. Communication overhead may increase due to auxiliary variable exchange in both primal and dual sides. I hope the author can have more discussions about it.

2. Experiments are somewhat limited. The experiment part includes datasets such as a9a,w8a,and ijcnn1. I hope authors can add more real-world datasets from computer vision areas into the experiments to further validate the advantages of the proposed algorithms.

3.  This paper is restricted to nonconvex–PL settings, which may limit generality to broader minimax classes.

### Questions
1. How does auxiliary-variable communication affect scalability with increasing number of workers $K$?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a novel decentralized algorithm for stochastic nonconvex-PL saddle-point problems. It improves the best known complexity dependence on the condition number of the loss, and also has a favorable (experimentally observed) property that only one hyperparameter needs tuning.

### Strengths
The work makes a significant theoretical advance in this specific research direction by improving complexity by objective's condition number $\kappa$ from $O(\kappa^2)$ to $O(\kappa^{3/2})$.

Authors state that all hyperparameters except one do not require fine-tuning

Structure of the proof is clearly visualized in Figure 4.

### Weaknesses
The communication complexity is inferior to DREAM algorithm in terms of $\epsilon$ and the spectral gap.

There is no theoretical comparison for computational complexity with competing algorithms (e.g. in Table 1) only an experimental comparison of gradient‑evaluation counts is provided.

I cannot see theoretically, what is the benefit from having stepsizes of the same scale for primal and dual variables if the algorithm still has many parameters. It was stated in Section 5.2 that "method is robust to all hyperparameters except $\beta$", but the only justification is through a numerical experiment on a single problem instance, which is not a solid evidence.

The analysis is overwhelmingly technical, but this is, unfortunately, standard for the field.

### Questions
- Could communication acceleration techniques such as multi-round communication used in DREAM  be incorporated in your algorithm to match the optimal dependence on the spectral gap?
- Why is the nonconvex-PL setting important?

### Soundness
3

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
The motivation of this paper is to develop a decentralized smoothed minimax optimization method that achieves a better convergence rate while using same-scale learning rates for the primal and dual variables. This paper achieves this goal by introducing their method, Double Smoothed Decentralized Stochastic Gradient Descent Ascent (Smoothed$^2$-DSGDAM), which applies a smoothing technique to both the primal and dual variables.
While such double smoothing is challenging in incorporating the variance reduction technique and handling communication complexity, this paper overcomes these issues and improves the convergence complexity to $O(\kappa^{3/2})$. Finally, they present experimental results showing that their algorithm outperforms previous algorithms.

### Strengths
- This paper provides an algorithm based on a new design technique (double smoothing), which addresses the goal of developing a decentralized smoothed minimax optimization algorithm that uses same-scale learning rates for the primal and dual variables.

- Its novelty is not only based on its idea but also justified by its technical difficulty. This smoothing technique faces two challenges. The algorithm is carefully designed to apply the variance reduction technique and to bound the consensus error regarding their auxiliary variable, thereby achieving a faster convergence rate. Their high-level idea is introduced on page 6.

- The improvement of convergence complexity has its own merit as a contribution.

- The overall story of the paper is presented clearly, starting from the motivation to the novel aspects of the algorithm’s design, the challenges they faced, and how they overcame them.

- This reviewer personally finds the well-structured appendix very impressive. It seems that the authors put a lot of effort into helping readers understand the structure of the proofs and the appendix.

- Not only for the idea of overcoming challenges, this reviewer thinks that the fact they went through the heavy calculations required for the proof is worthy of applause.

### Weaknesses
While this reviewer didn’t find any significant weaknesses in the paper, this reviewer felt there are a few points that could be improved in terms of formatting.

- In line 219, instead of writing $>0$ repeatedly for every parameter like $a>0, b>0, c>0$, writing $a, b, c>0$ seems better. The same applies for $<1$.

- In lines 230–233 (update rule of $u$ and $v$), this reviewer thinks it would be better to have a line break.

- In the appendix, there are too many unused equation numbers. Since an equation number provides information about which equations will be used again and may be important, not only are the extra numbers distracting, but this reviewer also thinks they may make it harder to identify the important equations.

- The formal introduction of the algorithm name is on page 4, after going through motivation and challenges. But this reviewer personally thinks it may be better to introduce it earlier in the introduction.
This paper puts a lot of emphasis on the challenges they needed to overcome, using bold text several times, but seems to place less emphasis on how they actually overcame them. For example, to the best of this reviewer's understanding, lines 303–321 seem to detail how they overcame the challenges. This reviewer wonders whether this part could be more structured, indicating which section corresponds to which challenge, and highlighting the key ideas.

### Questions
- **Q1.** Could the authors provide a high-level explanation of the core ideas they used to overcome the challenges, and share their insights on why those ideas work?

- **Q2.** The potential function in equation (95) seems quite complicated. Could the authors provide the motivation for how they came up with such a potential function?

### Soundness
3

### Presentation
3

### Contribution
3
