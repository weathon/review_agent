# FIRE: Frobenius-Isometry Reinitialization for Balancing the Stability–Plasticity Tradeoff

- Avg Score: 6.00
- Decision: Accept (Oral)
- Scores: 6, 6, 6, 6

## Abstract
Deep neural networks trained on nonstationary data must balance stability (i.e., retaining prior knowledge) and plasticity (i.e., adapting to new tasks). Standard reinitialization methods, which reinitialize weights toward their original values, are widely used but difficult to tune: conservative reinitializations fail to restore plasticity, while aggressive ones erase useful knowledge. We propose FIRE, a principled reinitialization method that explicitly balances the stability–plasticity tradeoff. FIRE quantifies stability through Squared Frobenius Error (SFE), measuring proximity to past weights, and plasticity through Deviation from Isometry (DfI), reflecting weight isotropy. The reinitialization point is obtained by solving a constrained optimization problem, minimizing SFE subject to DfI being zero, which is efficiently approximated by Newton–Schulz iteration. FIRE is evaluated on continual visual learning (CIFAR-10 with ResNet-18), language modeling (OpenWebText with GPT-0.1B), and reinforcement learning (HumanoidBench with SAC and Atari games with DQN). Across all domains, FIRE consistently outperforms both naive training without intervention and standard reinitialization methods, demonstrating effective balancing of the stability–plasticity tradeoff.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce FIRE which views resetting a network in continual learning as a constrained projection that keeps weights close to the previous solution (minimize squared Frobenius error) while restoring isometry (zero deviation-from-isometry), yielding the nearest orthogonal weights via an Orthogonal Procrustes step implemented efficiently with a few Newton–Schulz iterations. This method is motivated by the need to balance plasticity and stability. Theoretically, the authors demonstrate that this limits representation drift, smooths the loss (via a Hessian bound), raises effective rank, and bounds neuron dormancy, thus improving plasticity without sacrificing stability. Over a series of continual learning experiments in vision, language, and RL, the authors show that using FIRE at task boundaries (or once mid-training in RL) consistently maintains plasticity, attaining similar or superior performance to the benchmark methods that the authors evaluate.

### Strengths
- The paper is generally well written.
- The paper utilizes a good mix of continual learning benchmark problems: vision, language, and RL along with a good mix of network architectures.
- The derivation of FIRE is very well done and posing the stability-plasticity tradeoff as a hard constraint optimization problem leads to an algorithm that does not necessitate an explicit $\lambda$ weight that requires the user to tune how much the method should prioritize stability vs plasticity. Many continual learning methods are sensitive to hyperparameter choice and require careful tuning, and outside the well structured academic benchmarks, it's not clear how robustly these methods will perform.
- The theory is solid and well justifies FIRE.

### Weaknesses
- While the experiments are thorough, the empirical conclusions would be more convincing had more competitor methods been evaluated. Given that FIRE is partially motivated by parameter resets, it would be useful to compare FIRE against reset methods such as Self-Normalized Resets, Continual Backdrop, and ReDO which have been shown to be effective at maintaining plasticity and against the regularization based method of L2 Init.
- Tangential to the above point, FIRE appears to show similar performance to its competitor methods on the class incremental and data incremental settings. While in the warm start setting FIRE does perform the best, given the scale the performance in marginally better. Had additional baseline methods been evaluated it is not clear if FIRE would be consistently outperforming these baselines.
- While test error is the metric of interest, many papers on plasticity loss consider trainability and minimizing training error as a diagnostic of plasticity loss. It would be beneficial to also report training error for the experiments as plasticity loss and generalization error can confound with regularization and the number of training epochs, see appendix A.2 in Self Normalized Resets for Plasticity in Continual Learning. 
- While FIRE is motivating by preserving stability, I don't quite see how the vision and language experiments evaluate stability, these look to me to be purely problems that evaluate plasticity. For instance in the vision problems, the network sees progressively more data, whether it is warm starting or class/data incremental. For the language problem, correct me if I am mistaken, but if the validation set is the second training set, then again I do not see how stability is measured. 
- Additionally for the language problem, I don't quite see how plasticity is present. It appears that every method is able to reduce the validation error, and a plateau begins to appear roughly at the same iteration count for each method. It would be helpful, for instance, to see an experiment where the baseline's error (training or test) increases over time.

### Questions
- Given the noted similarity with Parseval, could the authors compare and contrast their contributions with that of the Parseval paper?
- What are the with reset and without reset dashed lines in row (a) of Figure 2?
- Have the author's found any limitations to FIRE? For instance, many continual learning methods can be sensitive to the choice of hyper parameter. Could there be data streams that could be particularly challenging for FIRE to train on?

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
3

### Summary
The paper presents FIRE, a novel and principled method for reinitializing neural network weights to balance stability and plasticity in continual learning scenarios. The core idea is to formulate reinitialization as a constrained optimization problem: minimizing the Squared Frobenius Error (SFE) to the previous weights (stability) subject to the constraint of zero Deviation from Isometry (DfI) (plasticity). The solution is efficiently approximated via the Newton-Schulz iteration. The method is evaluated extensively across vision, language, and reinforcement learning tasks, demonstrating consistent superiority over naive training and baseline reinitialization methods like S&P and DASH.

### Strengths
- This paper introduces the novel use of Deviation from Isometry (DfI) as a unified and direct measure of plasticity loss, providing compelling theoretical evidence that minimizing DfI is mathematically equivalent to achieving multiple desirable plasticity properties, including flattening the loss surface, preventing neuron dormancy, and increasing feature rank.
- The solution, corresponding to the nearest orthogonal matrix via polar decomposition, is efficiently computed using the Newton–Schulz iteration, rendering the method practical with negligible computational overhead.
- The robustness of FIRE is convincingly demonstrated across diverse domains, including continual visual learning (ResNet, ViT), large language model pretraining (GPT-0.1B), and reinforcement learning (SAC, DQN), suggesting its broad applicability.

### Weaknesses
- The Newton–Schulz iteration used in FIRE is implemented with a fixed number of steps and lacks any adaptive stopping criterion or convergence check. For highly ill-conditioned weight matrices, the fixed step may be vulnerable to divergence, which could compromise the isometric constraint.
- While the theoretical claim of low time overhead is promising, providing concrete measurements would further strengthen its credibility. Additionally, a comparative analysis of memory footprint against baseline methods is lacking. 
- The study does not examine the interaction of FIRE with other effective continual learning strategies, such as parameter-efficient fine-tuning (PEFT) methods like LoRA, which also target the stability and plasticity tradeoff. Besides, how does FIRE perform with other regularization or sharpness-aware landscape techniques?
- Which stage or task is evaluated in Figure 2c, and do the later tasks perform differently from the earlier ones?

### Questions
please ref to the weakness

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
In this paper, the authors address the challenge of balancing plasticity and stability in continual learning by proposing a reinitialization-based method called FIRE. This approach is grounded in theoretical analysis and demonstrates its effectiveness across various learning paradigms, including continual visual learning, language modeling, and reinforcement learning.

### Strengths
The proposed method is strongly motivated and comes with theoretical guarantees (although I have not thoroughly checked the correctness of the proofs). Moreover, the authors validate the effectiveness of their approach across multiple learning paradigms, demonstrating its broad applicability through extensive experiments.

### Weaknesses
the authors only state the theorems without providing detailed analysis of the theorems and their corresponding implications, especially for Theorem 1 and Theorem 4. In addition, the choice of baseline methods for comparison is rather limited, which to some extent reduces the strength of the empirical validation of the proposed approach.

### Questions
In Theorem 1, it can be directly inferred that minimizing the SFE leads to a reduction in the variance difference. However, a more specific analysis is needed to explain how a smaller variance difference translates into strong stability for the model. For Theorem 4, the upper bound of $s_j$ is 1, but since $1<\sqrt{\frac{1+\epsilon}{1-\epsilon}}$ always holds, the derivation of the upper bound does not provide any meaningful insight. Meanwhile, the lower bound of $s_j$  increases as $\epsilon$ decreases, which actually implies an increase in the number of dormant neurons—contradicting the paper’s claim that the number of dormant neurons is reduced. Moreover, in Theorems 3 and 4, the assumption $\epsilon<1$ appears rather strong,  and its validity in practice should be justified. Additionally, the authors should provide a derivation for Equation (5).

### Soundness
2

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
3

### Summary
This paper proposes the FIRE algorithm for training Neural networks continually that is able to better balance stability with plasticity. The way it does so by doing a constrained optimization of the Squared Frobenius Error (SFE) between the weights and past weights, subject to the Deviation from Isometry (DfI) property being 0. The DfI property measures how close the weights are to orthonormal, which has been linked to high plasticity, and the SFE ensures the weights don’t deviate far when trying to achieve the plasticity objective. The paper evaluates the algorithm across continual visual learning, warm starting LLM pretraining, and RL.

### Strengths
This is a clearly written paper and the algorithm is very intuitive. The results are fairly strong across a variety of domains, showing the generality of the idea. The suite of experiments and ablations is fairly comprehensive. The theoretical results in the provide a further justification for why the objective used (DfI) is appropriate for plasticity, as it addresses several of the issues commonly associated with loss of plasticity.

### Weaknesses
- Given the similarity of the DfI optimization to the Muon optimizer, I feel like the paper should have more of a discussion on the work beyond what they've provided. The paper claims that the difference is in the coefficients chosen for the Newton-Schulz iteration, and that muon favors speed over stability, but it is not clear if that would actually matter. It seems that muon would also be a good baseline for this paper.
- I have a few questions which I've outlined below.

### Questions
- Do you apply FIRE only at the start of distribution shifts, or is it a regular intervention? In RL it seems reinitialization only happens once, what happens when you reinitialize more often? 
- I am assuming that before the reinitialization point, the different RL algorithms should be the same, so why are the results for the different algorithms different before the reinitialization point? If it’s just the result of randomness in the run, then maybe try an experiment where you take the same checkpoints/replay buffers and reinitialize them with the different strategies, rather than starting the runs from scratch. This would help disentangle the effect of what was potentially in the replay buffer with the effect of the reinitialization strategy.
- In Figure 5, what happens when you decrease the number of iterations even more? What is the breaking point?
- What is the type of change that happens with this reset? For example, on average, how much do the parameters move, is it generally a low loss path/is the reset to the same basin?
- This is lower priority, but I’d be curious to see how Hare and Tortoise does in your experiments as well, since it can also be thought of as a type of resetting baseline.

### Soundness
3

### Presentation
4

### Contribution
3
