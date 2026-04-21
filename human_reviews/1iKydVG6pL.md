# Discovering Mathematical Formulas from Data via LSTM-guided Monte Carlo Tree Search

- Avg Score: 4.25
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 6, 5

## Abstract
Finding a concise and interpretable mathematical formula that accurately describes the relationship between each variable and the predicted value in the data is a crucial task in scientific research, as well as a significant challenge in artificial intelligence.  This problem is commonly referred to as symbolic regression, which poses an NP-hard combinatorial optimization problem.  Traditional symbolic regression algorithms typically rely on genetic algorithms;  however, these approaches are sensitive to hyperparameters and often struggle to fully recover the target expression.  To address these limitations, a novel symbolic regression algorithm based on Monte Carlo Tree Search (MCTS) was proposed this year.  While this algorithm has shown considerable improvement in recovering target expressions compared to previous methods, it still faces challenges when dealing with complex expressions due to the vast search space involved.  Moreover, the lack of guidance during the MCTS expansion process severely hampers its search efficiency.  In order to overcome these issues, we propose AlphaSymbol - a new symbolic regression algorithm that combines MCTS with a Long Short-Term Memory network (LSTM). By leveraging LSTM's ability to guide the MCTS expansion process effectively, we enhance the overall search efficiency of MCTS significantly.  Next, we utilize the MCTS results to further refine the LSTM network, enhancing its capabilities and providing more accurate guidance for the MCTS process. MCTS and LSTM hand in hand advance together, win-win cooperation until the target expression is successfully determined. We conducted extensive evaluations of AlphaSymbol using 222 expressions sourced from over 10 different symbolic regression datasets.  The experimental results demonstrate that AlphaSymbol outperforms existing state-of-the-art algorithms in accurately recovering symbolic expressions both with and without added noise.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a method to perform symbolic regression based on Monte-Carlo Tree Search (MCTS) with guidance from an LSTM.

### Strengths
I'm afraid I do not see much added value in this paper compared to existing work...

### Weaknesses
- Lack of novelty: the authors do not cite “Deep Generative Symbolic Regression with Monte-Carlo-Tree-Search” by Kamienny et al. The latter also performs MCTS for SR with guidance from a Transformer and should be cited — according to me, their method performs much better and is better validated empirically. There are also many other missing references in the SR literature.
- Experimental validation is poor. For example, the authors only report results when sampling only 20 points in the interval [-1,1], which is very small. They do not evaluate on the mainstream benchmark SRbench.
- Paper is particularly poorly written and presented, as detailed below.

### Questions
Important comments:
- I don’t understand what the \hat x_ij means in Eq 6; it does not seem to be defined anywhere. In symbolic regression, one typically predicts the labels \hat y from the inputs x_ij, but I don’t see how one can “predict the inputs”… This is very important as the authors consider this new loss function to be among their main contributions.

Comments on presentation:
- Many sentences are not capitalised 
- Many sentences are cut with inappropriate punctuation (e.g. “which cleverly combines LSTM and MCTS. And outperforms several baselines” or “thereby avoiding situations where each symbol is predicted with a similar probability. Improved the search efficiency of the algorithm.”)
- References are not separated from the text with a space
- Fig 3 is poorly described: what is the red line in panel (a) ? What exactly is plotted in panel (c) (what is compressive strength) ?
- Lack of details in many parts: 
    - “"No constrain" means no constraints are applied”, what are these constraints ?
    - Table 2 needs more details (“Yes/No”->”With entropy regularisation”/“Without”, “Time”->”Training time” etc)
Typos : 
- “times it is child” 
- “with the following expression:6”

Other things:
- “Anti-noise”->”demonising”
- “the algorithm’s reward function fluctuation is illustrated in the line graph (convergence proof)” : reward vs time is by no means a convergence proof…
- The computations after Eq. 7 do not make any sense : the partial derivatives are indicated as positive or negative without any justification on the range of the variables. Moreover, dy/dx>0 does not mean y is “proportional” to x.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The submission examines the performance of an AlphaZero-like approach, which they call AlphaSymbol, to the symbolic regression problem.

### Strengths
I'm not very familiar with the symbolic regression, but I'm not aware of AlphaZero having been applied to this problem setting.

### Weaknesses
The formatting of the submission makes it hard to read. There is not sufficient space between the paragraphs. There is clearly content that can be cut from the submission to make it easier to read. For example, the four phases of MCTS do not need to be enumerated in the introduction.

The structure and contextualization of the submission is poor. The submission is essentially applying AlphaZero to a new setting with problem-specific tweaks. However, the submission is written as if the AlphaZero methodology is largely original to the submission: AlphaGo Zero is cited one time for the definition of a running action value and AlphaZero is not cited at all. This lack of proper attribution is alone enough to disqualify the submission from acceptance. The appropriate way to structure the submission would be to include AlphaZero in a background section and describe problem specific tweaks in a methodology section.

There are also some strange deviations from AlphaZero that make me skeptical of whether the results should be taken seriously. For example, in equation (4), the submission seems to suggest that it uses the normalized logarithm of the visit counts as the policy (though it gives contradictory information elsewhere in the submission). If it is true that the submission is using the logarithm of the visit counts, it ought to better justify this modification (though I am skeptical that a justification exists). Also, it adds an entropy penalty to the loss function that is not typically present. The submission does ablations which seem to suggest that this entropy penalty is helpful. But these lead me to wonder whether this penalty is only necessary because of other unusual choices made by the submission. Overall, it is certainly possible that the submission's deviations from AlphaZero are necessary to achieve good performance, but the submission's poor presentation leaves the reader with the feeling that these deviations are haphazard rather than the product of careful study.

### Questions
> Think of the things where a response from the author can change your opinion

I think the submission requires significant revisions to improve readability, appropriately separate background from contribution, and discuss and investigate the reasoning behind deviations from AlphaZero.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper considers using a Monte Carlo Tree Search variant for discovering mathematical formulas. The MCTS variant uses PUCT for selection with an LSTM network providing the prior.
The empirical evaluation show that the algorithm is competitive with the state-of-the-art on several benchmarks.

### Strengths
The empirical results do show that the proposed algorithm can be a powerful tool for discovering mathematical formulas.

### Weaknesses
The proposed algorithm is a fairly standard MCTS, LSTM being the only slight deviation from a standard architecture used in games.

### Questions
Since the main deviation from the standard MCTS implementation (that uses deep neural networks as priors) is the use of LSTM, it would have been useful to explore the possible alternative architectures. LSTM seems a reasonable choice given previous suitability to formula discovery, but have you tested other architectures as well?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents AlphaSymbol, a new approach for symbolic regression for the discovery of mathematical formulae. The proposed approach augments a monte-carlo tree search with an LSTM to guide the search, a new reward function that addresses the problem of variable omission, and a new loss function for training the LSTM such that it produces distributions with lower information entropy. The experiments show that the proposed approach has a significantly higher recovery rate compared to the baselines.

### Strengths
Strengths:
- Important and well-motivated problem (symbolic regression for discovering mathematical formulae)
- New approach for the problem that consists of using LSTM to guide the monte-carlo tree search, as well as using a specialized reward function and a specialized loss function for training the LSTM
- Experiments show significantly higher recovery rate compared to the baselines

### Weaknesses
Weaknesses:
- Evaluation of experiments is not entirely clear: When is the search stopped and counted as "not recovered"?
- No comparison of running times between the proposed approach and the baselines. Or alternatively, comparison of rewards over time vs. the baselines.
- Some details about the technical approach is not entirely clear:
	* It is not clear how is the self-search phase and the use of LSTM are coordinated. For example, is the self-search used for several epochs while LSTM is being trained and then the algorithm changes to using the trained LSTM (if so, when is the change done)? 
	* There are two loss functions. Is the second one (S_{NRMSE}) only used for the reward computation (while the first one is used for the LSTM training)?
- Writing can be improved as some details are missing (examples above), format is quite dense with some subtitles appear inside a paragraph (e.g., "Ablation experiment for information entropy."), and several typos and inconsistencies (examples listed under "Minor issues" below). The appendix is used as part of the paper, simply transferring some figures there and referencing to them as if they are part of the main paper, which hinders the ability of the paper itself to be self-contained without the appendix and hurts the readability of the paper. Section 5 is entitled "Discussion" but reads much more like a "Conclusion".


Minor issues:
- in abstract: "MCTS and LSTM hand in hand advance together, win-win cooperation until the target expression is successfully determined" - this is a bit too informal and can be rephrased to be a bit more precise/clear.
- "which is not interpretable and analyzable": there are many post-hoc interpretation techniques that can be applied
- " visit count N increase": what is N?
- "regression. however" -> "regression. However"
- Section 4: the description of algorithms as "excellent", "superior" is not clear (is excellent better than superior?). It is also important to highlight the current state-of-the-art on this task.
- "method. the expression 5 shows" -> "method. Expression 5 shows"
- "matrixE.1,"

### Questions
Please see "weaknesses" above.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
