# Retrieval-Guided Reinforcement Learning for Boolean Circuit Minimization

- Decision: Accept (poster)
- Scores: 8, 6, 6, 5, 6

## Abstract
Logic synthesis, a pivotal stage in chip design, entails optimizing chip specifications encoded in hardware description languages like Verilog into highly efficient implementations using Boolean logic gates. The process involves a sequential application of logic minimization heuristics (``synthesis recipe"), with their arrangement significantly impacting crucial metrics such as area and delay. Addressing the challenge posed by the broad spectrum of hardware design complexities — from variations of past designs (e.g., adders and multipliers) to entirely novel configurations (e.g., innovative processor instructions) — requires a nuanced 'synthesis recipe' guided by human expertise and intuition. This study conducts a thorough examination of learning and search techniques for logic synthesis, unearthing a surprising revelation: pre-trained agents, when confronted with entirely novel designs, may veer off course, detrimentally affecting the search trajectory. We present ABC-RL, a meticulously tuned $\alpha$ parameter that adeptly adjusts recommendations from pre-trained agents during the search process. Computed based on similarity scores through nearest neighbor retrieval from the training dataset, ABC-RL yields superior synthesis recipes tailored for a wide array of hardware designs. Our findings showcase substantial enhancements in the Quality of Result (QoR) of synthesized circuits, boasting improvements of up to 24.8\% compared to state-of-the-art techniques. Furthermore, ABC-RL achieves an impressive up to 9x reduction in runtime (iso-QoR) when compared to current state-of-the-art methodologies.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors proposed a new method for finding optimal synthesis recipes for unseen netlists. They proposed three methods, one based on MCTS and a trained policy on training netlist (MCTS+L) and the other, and more superior, is using MCTS and a trained policy with a variable to control the reliance on the trained policy or the online MCTS search (ABC-RL). The variable \alpha is determined based on a nearest neighbor search to determine if the netlist is close to the training netlists or not. The authors split the dataset of netlists into train, validation, and test sets and used the train to train the policy and used the validation set to set the hyperparameter that controls \alpha. The experimental results show that the ABC-RL can outperform the other methods and SOTA in most of the test netlists and can achieve the best goe-mean performance.

### Strengths
- Extensive experimental results.
  - The authors provided comparisons with multiple baseline including Online-RL, Simulated Annealing, MCTS, and MCTS+L and MCTS+L+FT.
  - The study on training the policy on specific benchmarks was helpful showing the effect of \alpha and closeness of netlist to the training dataset.
  - They perform an ablation study on the architecture of the policy to determine the impact of the transformer architecture on the performance.
- The authors motivated the need for a variable that can control the effect of policy vs MCTS very well by an example in the Introduction Section.
- The paper is well written and organized and can be followed easily by non-experts. 
- The related literature has been sufficiently reviewed and cited.

### Weaknesses
- The idea of using \alpha was interesting and novel to the best of my knowledge, however it is a simple and small contribution.

### Questions
- It would be great if the authors can also provide the results for the rest of the benchmark netlists to ensure that the similar performance gains hold up for the training and validation sets.
- In Section 2.4, in definition of \sigma_{\delta_{th}, T}(z), replace \teta with \delta_{th}.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a new logical synthesis optimization sequence method ABC-RL based on MCTS and pertained policy. Unlike previous works, ABC-RL can compute the similarity of the new circuit with previous circuits, thus determining how much experience will be used. Experiments show that ABC-RL achieves SOTA optimization results in most circuits.

### Strengths
1. The idea that the degree of retrieval is determined by the similarity function is interesting, and it shows the advanced performance in experiments.

2. The experiments are extensive.

3. The paper is well-organized and easy to read.

### Weaknesses
1. Some other ML method baselines are not included. For example, the DRiLLS [1] results are not included in the paper, but it is an important method recently.

2. The retrieval performance is not well reported. For example, it should give us examples of each circuit’s nearest neighbor circuit and similarity factor. 

3. ChiPFormer [2]  is an RL placement method with a pretrained policy and should be included in related work.

[1] Hosny, Abdelrahman, et al. "DRiLLS: Deep reinforcement learning for logic synthesis." 2020 25th Asia and South Pacific Design Automation Conference (ASP-DAC). IEEE, 2020.

[2] Lai, Yao, et al. "ChiPFormer: Transferable Chip Placement via Offline Decision Transformer." (2023).

### Questions
1. Could you compare ABC-RL and DRiLLS methods?

2. As weakness 2, could you give each circuit’s nest neighbor circuit and similarity factor?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Summary:
 This papers proposes ABC-RL method to smoothly change the impact of learned policy in the search objective based on the graph similarity. Similar designs to training examples will have more bias from learned policy, while novel circuit will use more search. Compared to prior method, the propose one outperforms in various designs in terms of area-delay product.

### Strengths
Strength: 

1.	The proposed method is based on good observation that the benchmark has a large diversity, and the learned policy sometimes is not helpful for novel circuits.

2.	The proposed method smoothly controls the importance of learned policy using GNN-based graph embedding similarity scores. The variable heuristic synthesis solution from the learned policy is encoded with transformer blocks for better performance.

3.	It also runs fast with runtime benefits.

### Weaknesses
Weakness:

1.	The overall framework basically does test-time augmentation. It assumes pre-trained policy is not generalizable to novel circuits, and by comparing the test example with the training example, it dynamically selects among two search strategies, but in a smooth way. The circuit similarity is a performance proxy for pretrained agent and MCTS method, and use that proxy to predict the weights to ensemble two models. The novelty, in this sense, is limited. Is it possible to combine more synthesis strategies based on a more general performance predictor at test time?

2.	The usage of BERT is not well justified. There are other simpler methods to encode variable-length sequences, e.g., RNN. The attention model is also data-hungry during training. Why BERT is the most suitable encoder?

3.	The assumption that learned policy cannot generalize to diverse benchmarks is not well supported. If there is generalizable knowledge in circuit representation and synthesis strategies, it should try to improve the generalization of the learned policy by using more data/better algorithms. If this problem in nature is not generalizable or learnable, then it is not necessary to use an RL agent to learn the synthesis strategy at the beginning. The proposed method does not fundamentally explain or solve the learnability of circuit synthesis problems, but rather uses two model ensembles to just cover some in-distribution data with learned model and out-of-distribution examples by search.

### Questions
Basically, it is listed in the weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents ABC-RL, a retrieval-guided RL approach to generate an optimized synthesis recipe for logic synthesis. ABC-RL tunes a weight that adeptly adjusts recommendations from pre-trained agents during testing. Given a circuit with high similarity to the circuits in the training dataset, ABC-RL assigns high weight to the policy by the RL agent. Otherwise, ABC-RL tends to rely more on the default searching strategy.

### Strengths
1.	This paper introduces a retrieval-guided RL agent to search for an optimized synthesis recipe for logic synthesis. ABC-RL selectively leverages the knowledge obtained during training based on the similarity between training and testing samples. This approach can address the issue of RL performance decrease caused by different data distributions. 

2.	The authors claim that ABC-RL can achieve up to 24.8% QoR improvement and reduce runtime up to 9x.

### Weaknesses
1.	ABC-RL calculates the cosine similarity between the embeddings of training and testing samples to determine the similarity score. Therefore, the quality of AIG embeddings is the key to the entire methodology. Unfortunately, the authors do not explore this issue and simply use a 3-layer GCN to do it. It's hard to believe such an approach could work properly. A circuit graph is a lot more complicated than a plain graph, not only it's directed, but more importantly, it has unique functions associated with it. Two AIGs could be very similar in terms of structure but differ significantly in terms of functionalities. Consequently, they may require different synthesis recipes, isn't it?

2.	The experiments are conducted with a very small dataset, which only includes 23 netlists for training. This is not convincing. The proposed agent should have seen sufficient circuit designs to come up with a good RL strategy.

### Questions
Please refer to the weakness part.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work introduces, ABC-RL, an MCTS+Learning algorithm to optimize the transformation recipes in logic synthesis to minimize the area delay product (ADP).  It employs GCN for AIG features and a transformer for applied transformations. It introduces a hyperparameter to adjust recommendations from pre-trained agents during the search. The hyperparameter is determined by similarity computed from GNN features learned during training. The key observation it leverages from hardware designs is that they contain both familiar and entirely new components. The empirical results show ABC-RL improves synthesized circuit Quality-of-Result (QoR) by up to 24.8% over SOTA methods and provides an average runtime speed-up of 1.6× compared to baseline MCTS.

### Strengths
1. The policy network architecture for recipe encoding and AIG embedding is reasonably selected.
2. The novel introduction of a similarity-score-based hyperparameter effectively enhances ABC-RL convergence, as observed in the results.
3. The study provides a comprehensive evaluation, comparing various search algorithms for logic synthesis recipe optimization, including MCTS, online RL, and simulated annealing. It also includes comparisons with MCTS+Learning with fine-tuning.

### Weaknesses
1. It would be beneficial to provide an estimate of the number of gates in each benchmark circuit. One concern with this approach is whether GCN-based AIG embeddings can effectively scale to real-world circuit designs.
2. With the integration of GNN + transformer features. It is likely that the compute complexity and runtime of each search iteration would be higher than the baseline MCTS.

### Questions
1. How is it compared to non-learning-based recipes? Is there an O3 flag for logic synthesis? 
2. How does its wall clock time (not iterations) compare to other algorithms such as MCTS and SA+Pred?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
