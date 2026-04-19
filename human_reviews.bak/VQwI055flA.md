# CARTS: Advancing Neural Theorem Proving with Diversified Tactic Calibration and Bias-Resistant Tree Search

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 8

## Abstract
Recent advancements in neural theorem proving integrate large language models with tree search algorithms like Monte Carlo Tree Search (MCTS), where the language model suggests tactics and the tree search finds the complete proof path. However, many tactics proposed by the language model converge to semantically or strategically similar, reducing diversity and increasing search costs by expanding redundant proof paths. This issue exacerbates as computation scales and more tactics are explored per state. Furthermore, the trained value function suffers from false negatives, label imbalance, and domain gaps due to biased data construction.  To address these challenges, we propose CARTS (diversified tactic CAlibration and bias-Resistant Tree Search), which balances tactic diversity and importance while calibrating model confidence. CARTS also introduce preference modeling and an adjustment term related to the ratio of valid tactics to improve the bias-resistance of the value function. Experimental results demonstrate that CARTS consistently outperforms previous methods achieving a pass@l rate of 49.6\% on the miniF2F-test benchmark. Further analysis confirms that CARTS improves tactic diversity and leads to a more balanced tree search.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper designs a new search algorithm for formal theorem proving called CARTS. Compared with the standard MCTS algorithm, CARTS encourages diverse tactics that lead to significantly different next states. This paper also trains a value function using pairwise feedback. When applied to three different models, this paper shows that CARTS outperforms baseline methods such as best-first search and MCTS on miniF2F-test and ProofNet at the test time (in other words, the model that generates the tactic is not trained).

### Strengths
-	The algorithm, CARTS, introduces a more aggressive exploration strategy compared with the standard MCTS algorithm, by encouraging the diversity of the next state using the MMR algorithm. While the idea of encouraging the diversity of next states is explored in prior works such as [1], using an embedding model to measure the diversity of the next state is novel.
-	This paper identifies the label imbalance problem when training the value function, and mitigate this issue by training the value function using pairwise feedbacks.

[1] Xin, Huajian, et al. "DeepSeek-Prover-V1. 5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search." arXiv preprint arXiv:2408.08152 (2024).

### Weaknesses
My major concern is the statistical significance of the results. On the one hand, the CARTS algorithm in this paper has many new components compared with the standard MCTS algorithm, and the design of the new components seems arbitrary (or at least heavily relies on vague heuristics). On the other hand, the improvement of CARTS on miniF2F-test is less than 2%. Since the miniF2F test set only has 244 questions, it means that the difference only comes from a few questions, which is not significant enough given the complexity of the algorithm design.

A few claims in the paper are not justified by empirical evidence:
 
-	Line 86: “This could result in label imbalance, causing the cross-entropy loss used in training, the value function to easily converge to local optima.” Why does label imbalance with cross-entropy loss lead to local optima?
-	Line 287: I don’t see how the adjustment term can mitigate the domain gap between the training and test datasets. The following sentences in this paragraph are irrelevant to the domain gap.

### Questions
-	How good is the off-the-shelf sentence embedding model when applied to Lean’s states? Since the standard sentence embedding model is trained on natural language data, it is unclear to me whether they can distinguish the subtle differences in Lean’s states. Can you show some concrete examples to demonstrate the performance of $f_{enc}$?
-	Is there a typo in Eq (1)? Currently, the bonus term might be bigger than 1 if the action a is only taken a few times.
-	In Table 3, what is the algorithm for the “Bias-resistant value function” row? Is the value function the only difference to CARTS, and how is the alternative value function trained?
-	The choice of the Bradley-Terry model sounds arbitrary, although the argument that the BT model effectively oversamples the positive pairs makes sense. Is the value function trained with BT model better than the value function trained with binary cross entropy loss with explicit oversampling of positive examples?
-	Can w(s,a), or MMR(s,a), be negative? If w(s,a) is negative, the exploration bonus (the second term in Eq. 1) is then increasing when action a is taken more times.
-	In the ablation for k, the conclusion is “as k increases, the pass@1 value generally decreases”. What about even smaller k such as k = 1,2,4? Intuitively the performance when k=1 should be very bad.
-	The case study in Section 4.3 shows that the ranking of the right tactic is improved. This could be a good place for more ablation studies. Which feature of the algorithm contributes to this improvement, length penalty, MMR, the new value function, or the adjustment term?

### Soundness
2

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This work proposed two ways to enhance neural theorem proving performance. One is to caliberate generated tactic, which can alleviate the problem of tactic redundancy and further helps to diversify the proof path. The second is to use bias-resistant value function by introducing preference modeling and adjustment according to valid tactic ratio. Experimental results show the combination of this two components could achieve remarkable performance gain.

### Strengths
1. The presentation of this paper is very clear, vitual demonstration is easy to understand as well.
2. The proposed method is reasonable and well-motivated, the first component is motivated by the redunancy of generated tactic and the second component is motivated by common gap-bias between different dataset. 
3. The reported results is good and stable, it can be applied to many existing methods and achieve a boost in performance.

### Weaknesses
1. The proposed method is not that generalizable for now, as it can be only applied to one-step tree search methods. 
2. The operating logit looks a little bit complex, raising concern about the running effiency, for example, in the tactic calibration process, this method need to run the tactic to get the next state at first and then perform MMR algorithm, which could be time-consuming.

### Questions
1. What's the comparison beween this method and other related method in terms of running efficiency? It would be great to clarify this point.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces two major improvement over the MCTS-based theorem proving framework: (1) diversified tactic calibration which re-scores generated tactics, by the [MMR algorithm](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf) for diversity-based re-ranking; and (2) bias-resistant value function, with embedding-based tactic filtering and preference modeling with the Bradley-Terry model. Take Reprover-Lean4 (a 0.3B model) as an example, the proposed method CARTS reaches 37.7%, while its MCTS counterpart is 36.5%.

### Strengths
- **Impact**: The paper is well-motivated -- improving the efficiency of tactic search and the accuracy of the value function addresses key challenges that practitioners in this field are eager to solve.
- **Originality**: Both designs of tactic calibration and bias-resistant value function are new and useful (incremental) contribution to the existing theorem proving framework. 
- **Writing**: The paper is very well-written and easy-to-follow.
- **Practicality**: The proposed method looks to be easy-to-implement, and authors have provided relevant codes. It would be nice if this could be made as a modular design that can be plugged into different pipelines/repositories.

### Weaknesses
My concerns are mostly on experiments.
- Both designs are non-trivial, however, it looks like the improvement over the MCTS benchmark is rather marginal. How many independent runs were used? Could the authors report the confidence interval, or provide some discussions on this?
- CARTS is based on MCTS (is this correct?); i wonder if it could be applied on top of BFS as well. Also, what is the value function used for BFS and MCTS in the tables? Do they use the model likelihood or some other value models?
- In Line 267, how to determine $\tau$? Could the authors do some ablations on this?
- Could the authors evaluate the method on more benchmarks, e.g., LeanDojo (i.e., mathlib theorems)?
- The paper uses the phi-1.5 model as the base model for the value function. There has been controversy or potential ethical concerns on this model (e.g., Schaeffer, Rylan. "[Pretraining on the test set is all you need](https://arxiv.org/pdf/2309.08632)." arXiv 2023), though not directly relevant to theorem proving tasks. Could the authors do some ablations with other models, like llama-3.2-3b? Orthogonally, it might be nice to check if the performance will improve as the value model scales up. By the way, please add the model parameter count to reprover and stepprover in the tables.

Overall, i think the paper is well-written and brings useful information to the community, and I lean towards acceptance. I am happy to raise my score if the authors can further address the above concerns.

### Questions
Please see the weakness part.

### Soundness
3

### Presentation
4

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
The paper propose to improve MCTS for Neural Theorem Proving. It adds a measure of diversity in the exploration that takes into account the diversity of the tactics searched.

### Strengths
The paper addresses a hot and interesting topic.
The experimental results show better results than alternative search algorithms.
The experiments are well illustrated

### Weaknesses
Only one step tree search is supported.
Lack of experiments with more processing power.

### Questions
Does your method scale with more budget?

### Soundness
3

### Presentation
4

### Contribution
3
