# Cross-Domain Offline Policy Adaptation via Selective Transition Correction

- Decision: Reject
- Scores: 2, 6, 4

## Abstract
It remains a critical challenge to adapt policies across domains with mismatched dynamics in reinforcement learning (RL). In this paper, we study cross-domain offline RL, where an offline dataset from another similar source domain can be accessed to enhance policy learning upon a target domain dataset. Directly merging the two datasets may lead to suboptimal performance due to potential dynamics mismatches. Existing approaches typically mitigate this issue through source domain transition filtering or reward modification, which, however, may lead to insufficient exploitation of the valuable source domain data. Instead, we propose to modify the source domain data into the target domain data. To that end, we leverage an inverse dynamics model and a reward model to correct the actions and rewards of source transitions, explicitly achieving alignment with the target dynamics. Since limited data may result in inaccurate model training, we further employ a forward dynamics model to retain corrected samples that better match the target dynamics than the original transitions. Consequently, we propose the Selective Transition Correction (STC) algorithm which enables reliable usage of source domain data for policy adaptation. Experiments on various environments with dynamics shifts demonstrate that STC achieves superior performance against existing baselines.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper propose modifying the source domain data into the target domain data for better cross-domain learning and sample efficiency in cross-domain offline RL. The method learns an inverse policy model, a reward model and a forward dynamics model to achieve this modification and unified them into an algorithm called Selective Transition Correction (STC). Theoretical analysis and empirical evaluations are conducted to validate the framework's validity and effectiveness.

### Strengths
- The paper is well-structured, with clear introduction of the methodology and experimental setup.

- The paper proposes modifying source domain transitions to align with the target domain, as opposed to filtering data. In principle, this method is expected to improve the efficiency of data usage.

### Weaknesses
- The core idea is intuitive and reasonable, yet the proposed algorithm appears to be a forced assembly of distinct modules, lacking rigorous validation. In particular, the integration of the forward dynamics model undermines the paper’s logical coherence, while the designs of the inverse policy model and reward model also become largely meaningless.

- There is a large disconnect between the theory part and the proposed algorithm. While the THEORETICAL ANALYSIS aims to verify the validity of the inverse policy model and reward model, the authors subsequently state that "Since Dtar contains limited data, the learned inverse policy model may be unreliable in OOD regions." Yet they still need to incorporate a separate forward dynamics model. Ultimately, the analysis loses its value in guiding algorithm design and appears to be a proof created solely for the purpose of writing a proof.

- In my view, the three assumptions lack realism or have weak relevance to the experimental design. Take Assumption 1 as an example: it mandates that the dynamics difference between the source and target domains must be small. Yet the experiments provide no indication of how large this discrepancy actually is, making it impossible to verify the proposed bounds. For Assumption 2, considering the scarcity of target domain data, there is no explanation for how the assumption could hold—whether in a rigorous or even just an intuitive sense.

- The experiments use 5 seeds for repetition, but as illustrated in Figure 3, the standard deviations during learning are remarkably large. Such high variability, combined with the small number of runs, reduces the persuasiveness of the results. Moreover, the parameters are selected via a cherry-picking approach from a parameter search, and the authors even switch to different values post-search. This makes it difficult to believe the algorithm can generalize to other tasks when using the same parameters.

- The method requires learning three separate models, which makes it excessively heavy in terms of computational complexity. Additionally, the design logic behind these models is not sufficiently clear. Most notably, no experiments are conducted to justify the rationale for designing these specific models or to demonstrate the advantages of such a design. See Questions for more details.

### Questions
- Given the limited target domain data, how can we validate that the inverse policy model and reward model have been well learned?

- After the selection process, how many modified data samples remain?

- How to measure the difference between the two domains? What impact does this discrepancy have on selective correction and the final results? And does this align with the theoretical analysis?

- The inverse policy model always outputs a modified action, even when $s_{src}$ and $s'_{src}$ do not belong to the target domain. How should this case be handled? And does the theoretical analysis account for it?

- The reward model uses a first-order Taylor expansion around the original action. Why not predict the reward directly, similar to how the action is predicted?

- Line 191 states: "If the inverse policy model is sufficiently accurate, the corrected transition is expected to better align with the underlying dynamics of the target domain compared to the original transition." However, Line 274 mentions: "Note that we still include source transitions with large discrepancies for training since we believe there are still some underlying shared behaviors embedded in those data that can be beneficial for policy learning." Do these two statements conflict?

- Please refer to the "Weaknesses" section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
STC is a novel method for cross-domain offline RL, which modifies the source domain data into the target domain data.

### Strengths
* STC can explicitly make the source data align with the target dynamics.
* STC achieves superior performance against existing baselines.

### Weaknesses
*  The method trains an inverse policy and a reward model, which may bring more computational burden.

### Questions
* In Equation (4), could you clarify how to implement and obtain the $\hat{r}\_{\text{src}}$? 
  Specifically, how do you compute the term 
  $\nabla\_a r(s\_{\text{src}}, a)^\top \big|\_{a = a\_{\text{src}}} (\hat{a}\_{\text{src}} - a\_{\text{src}})$?

* The target domain dataset contains only 5,000 transitions. 
  I am concerned about whether the inverse policy model, reward model, and forward dynamics model 
  can be sufficiently trained under such limited data. 
  Have you tried using different numbers of transitions? 
  Is there a large performance difference?

* Could the authors include comparisons and a discussion with recent cross-domain offline RL studies (e.g., PSEC, DmC)? Incorporating these baselines and analysing the differences would strengthen the paper and better position the proposed method within the current literature.

Reference:

Liu, T., Li, J., Zheng, Y., Niu, H., Lan, Y., Xu, X., Zhan, X. Skill expansion and composition in parameter space. In International Conference on Learning Representations, 2025. 

Van, L. L. P., Nguyen, M. H., Kieu, D., Le, H., Tran, H. T., & Gupta, S. DmC: Nearest Neighbor Guidance Diffusion Model for Offline Cross-domain Reinforcement Learning. arXiv preprint arXiv:2507.20499.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the challenge of adapting offline RL policies across domains with mismatched dynamics by introducing Selective Transition Correction (STC), a method that corrects source domain transitions by leveraging an inverse policy model and a forward dynamics model trained on the target domain, specifically aligning actions and estimated rewards to the target environment. A selection mechanism filters corrections based on consistency with the target dynamics. Experiments demonstrate that STC typically outperforms several baselines across various MuJoCo tasks.

### Strengths
The paper theoretically analyzes the dynamics and value discrepancy induced by transition corrections with explicit assumptions and supporting proofs.

The paper conducts extensive experiments on various task domains and compares with sufficient baselines.

### Weaknesses
The success of STC depends heavily on the quality of the inverse policy and forward dynamics models. Insufficient or low-diversity target data will likely lead to suboptimal or even detrimental corrections.

The Taylor expansion in section 4.1 assumes local smoothness and is clipped for stability, but the limitations of this approximation—especially in highly nonlinear reward landscapes—have not yet been explored empirically or theoretically. Such approximations may produce inaccurate rewards for modified transitions, potentially introducing bias or overconfidence in out-of-distribution (OOD) settings.

Some common ideas have been investigated in other cross-domain RL settings, such as the correction module in CAT, and sequence consistency in [2]

For clarity, some notation is inconsistent (see e.g., mixing $\widetilde{\mathcal{M}}{\text{src}}$, $\widehat{\mathcal{M}}{\text{src}}$, and $\mathcal{M}_{\text{tar}}$ across proofs and main sections) that may confuse readers.

[1] Cross-domain adaptive transfer reinforcement learning based on state-action correspondence

[2] Cross Domain Policy Transfer with Effect Cycle-Consistency

### Questions
Could the approach extend to online learning settings or non-dynamics (observation) mismatch scenarios? 

Could the authors provide more insight into the computational overhead introduced by training/using three separate models (inverse, forward, reward) for transition correction and selection? How is the computation cost compared to other baselines? Would this scale to larger or more complex domains?

### Soundness
2

### Presentation
2

### Contribution
2
