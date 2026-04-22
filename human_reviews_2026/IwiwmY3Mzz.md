# A Reward-Free Viewpoint on Multi-Objective Reinforcement Learning

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 8

## Abstract
Many sequential decision-making tasks involve optimizing multiple conflicting objectives, requiring policies that adapt to different user preferences. In multi-objective reinforcement learning (MORL), one widely studied approach addresses this by training a single policy network conditioned on preference-weighted rewards. In this paper, we explore a novel algorithmic perspective: leveraging reward-free reinforcement learning (RFRL) for MORL. While RFRL has historically been studied independently of MORL, it learns optimal policies for any possible reward function, making it a natural fit for MORL's challenge of handling unknown user preferences. We propose using the RFRL's training objective as an auxiliary task to enhance MORL, enabling more effective knowledge sharing beyond the multi-objective reward function given at training time. To this end, we adapt a state-of-the-art RFRL algorithm to the MORL setting and introduce a preference-guided exploration strategy that focuses learning on relevant parts of the environment. Through extensive experiments and ablation studies, we demonstrate that our approach significantly outperforms the state-of-the-art MORL methods across diverse MO-Gymnasium tasks, achieving superior performance and data efficiency. This work provides the first systematic adaptation of RFRL to MORL, demonstrating its potential as a scalable and empirically effective solution to multi-objective policy learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a reward-free viewpoint for multi-objective reinforcement learning (MORL). It adapts the Forward–Backward (FB) representation from reward-free RL to MORL, aiming to learn a task-agnostic policy/value representation that can generalize to unseen preference vectors without retraining. Experiments on MO-Gymnasium tasks show improved performance and generalization compared with existing MORL baselines.

### Strengths
1. The paper provides a new perspective on MORL, suggesting that learning beyond specific scalarized reward combinations can accelerate MORL through more effective knowledge sharing across preferences.
2.The empirical evaluation is comparatively thorough for MORL: it includes multiple tasks (both discrete and continuous control), standard metrics such as hypervolume, analyses of generalization, and ablations (e.g., with and without PG-Explore and auxiliary losses).

### Weaknesses
1. The comparison with related work based on Successor Features (SF) and its variants is insufficient.
2. Some methodological and experimental descriptions are unclear. Please see the questions below.

### Questions
1. The proposed method and the goal of fast adaptation to unseen preferences appear conceptually close to SF. What concrete limitation of SF does the FB formulation address? Could the authors provide an ablation comparing FB + PG-Explore with an SF-based approach?
2. In Section 3 "Training objective function", the paper introduces both a measure loss and an auxiliary Q-loss. How are these two losses combined, particularly for the forward network? If they are combined, how sensitive is performance to their relative weighting?
3. In Figure 6, except for MORL-FB (w/o Q-loss), the hypervolume of other baselines is close to the proposed method, while UT performs significantly worse. Could the authors explain the reason for this discrepancy?

### Soundness
3

### Presentation
2

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
This paper proposes a reward-free viewpoint for MORL and an approach, MORL-FB, that integrates reward-free RL methods into MORL. Extensive experiments are conducted to show the effectiveness and generalization of the proposed method.

### Strengths
1. This work investigates the integration of a reward-free framework into multi-objective RL methods for better performance and transferability, which is novel yet natural.
2. A motivating example is provided to show the design consideration of the exploration of *z*, revealing the core insight of this work.
3. Extensive experiments are conducted on both discrete and continuous tasks, comparing with a wide range of baselines.
4. The visualizations are intuitive and show that the method is capable of handling multi-modal distributions, implying its improved generalization.

### Weaknesses
1. The layout on page 4 may need optimization. Algorithm 1 and Figure 1 can be presented on one line.
2. This work focuses on model design and experimental analysis, while the “reward-free viewpoint” in the title seems more theoretical. Further theoretical analysis may be needed, or the title and abstract should focus more on the model’s effect.
3. The title of Figure 20(a) seems mis-specified; it may be “Humanoid2d.”

### Questions
1. How is the transfer implemented through the framework of RFRL? A brief explanation based on the motivating experiment in Figure 1 would be appreciated.
2. How are $L_Q$ and $L_M$ balanced in line 20 of Algorithm 2? Is there any evidence that directly adding them up is plausible?
3. Why is the Q loss regarded as “auxiliary”? Does this mean the Q loss is less important compared with the measure loss?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper views MORL problems from a reward-free perspective and then solves them using the RFRL method. Based on RFRL, it introduces a preference-guided exploration strategy that incorporates reward functions and preferences into the original RFRL. Specifically, rather than directly sampling latent vectors from a normal distribution, it samples preferences uniformly and uses them to generate latent vectors. Finally, the proposed method is evaluated on a series of tasks in MO-Gymnasium and demonstrates superior performance and data efficiency.

### Strengths
- The paper is well-motivated. It solves MORL problems via RFRL, which is intuitive and novel and could serve as a good bridge between the two sub-fields.
- The empirical evaluation and ablation studies are comprehensive and insightful, carefully demonstrating the effectiveness of the proposed approach from multiple aspects.
- The paper is well-written and well-organized, with implementation details that are easy to understand.

### Weaknesses
I didn’t find major issues in this paper.

### Questions
- For completeness, I recommend briefly describing the future visited probabilities $M=FB$ in the preliminary section or appendix, as it is an important concept that helps readers understand why FB can be applied to arbitrary rewards.
- In Figure 1, the lines are too dense and intertwined, making it difficult to interpret. It may be better to eliminate the curves corresponding to batch sizes 256 and 4096, or adjusting the scaling to improve clarity.
- I recommend adopting more RFRL methods beyond FB to solve MORL in the future, which could lead to more insightful conclusions on RFRL's advantages in MORL.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper considers the problem of multi-objective reinforcement learning. Current MORL algorithms typically assign preference weights to individual objectives during the training process, modifying the underlying utility function to discover one or multiple policies to satisfy different preferences. 
This work proposes to instead use techniques adapted from reward-free reinforcement learning (RFRL), where no such preference weighting is included in the utility function. RFRL aims to learn a policy that is optimal for arbitrary reward functions. The authors note the conceptual connection to MORL problems, and suggest an extension of a state-of-the-art RFRL algorithm to the multi-objective setting. The proposed algorithm contains three important modifications: preference-guided exploration, training on latent vectors sampled from the replay buffer as auxiliary tasks, and auxiliary Q loss.
The algorithm is validated experimentally, showing clear improvement over other MORL algorithms in several standard multi-objective metrics.

### Strengths
- This paper draws a natural connection between RFRL and MORL, with a generally insightful discussion of the similarities and differences between the two. According to the authors, this is the first work to do so explicitly. This is a valuable contribution, and an interesting starting point for further work. (This reviewer is not intimately familiar with the current state of the art, and so must take the assertions about the novelty of the paper at face value)

- The paper is well-written and largely well-organised. Experiments are well-constructed, with a solid statistical approach, hyperparameter search, reproducible parameters, and meaningful ablation studies.

- The experimental validation of the algorithm shows good results compared to various state-of-the-art approaches and baselines.

### Weaknesses
- The paper occasionally seems to conflate the field of MORL as a whole with the single MORL technique of linearly combining objectives, e.g. in the introduction and Section 3. As mentioned in the related work section, other MORL approaches exist. This should be clarified.

- At times, this paper relies quite heavily on the appendix, in particular when relegating the discussion of related work from RFRL to the appendix. With the extension of the page limit during the revision phase, this should be moved to the main section to allow the paper to stand alone.

- The conclusion in its current form is also quite short, and could benefit from the addition of slightly more detail about the method and context. Similarly, some figures are quite cramped (see additional comments).


Additional (minor) comments

- Some of the figures, e.g. 1 and 3, are somewhat difficult to read. These could be made more readable despite the space constraints, e.g. by increasing font size and line thickness. Similarly, it is good that variance/error is represented in the plots, but in the current format this is barely readable.

- Some minor language and formatting issues:
1. line 050: “no prior work has explicitly adapt RFRL...”
2. “across various tasks in … benchmark” (line 080/081)
3. line 210: “where (a) holds by that ...”
4. Eq. 5 extends into the page margin.

- Occasionally, a citation is used as the subject of a sentence, e.g. in line 465 “(Felten et al., 2024; Mossalam et al., 2016) extended …”. In such cases, ‘\citet{}’ can be used in LaTeX to generate a textual citation.

### Questions
1. Perhaps this has been missed, but will the implementation of this methods be made available, ideally for integration with the existing benchmarking suite?

2. How exactly were hyperparameters chosen for the proposed method (MORL-FB)? The appendix only mentions a decision “guided by prior research and the HPO results”, which seems not quite comparable to the HPO-tuning of the remaining algorithms.

### Soundness
3

### Presentation
3

### Contribution
3
