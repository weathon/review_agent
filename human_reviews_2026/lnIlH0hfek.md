# Mixture of Masters: Sparse Chess Language Models with Player Routing

- Avg Score: 4.40
- Decision: Reject
- Scores: 6, 8, 2, 4, 2

## Abstract
Modern chess language models are dense transformers trained on millions of games played by thousands of high-rated individuals. However, these monolithic networks tend to collapse into mode-averaged behavior, where stylistic boundaries are blurred, and rare but effective strategies are suppressed. To counteract homogenization, we introduce Mixture-of-Masters (MoM), the first chess mixture-of-experts model with small-sized GPT experts emulating world-class grandmasters. Each expert is trained with a combination of self-supervised learning and reinforcement learning guided by chess-specific rewards. For each move, a post-hoc learnable gating network selects the most appropriate persona to channel depending on the game state, allowing MoM to switch its style dynamically—e.g., Tal’s offensive vocation, Capablanca's positional dominance, or Petrosian’s defensive solidity. To quantitatively assess whether each expert captures a distinctive playing signature, we propose a behavioral stylometry metric by training a vision transformer encoder to classify grandmasters from game segments. When evaluated against Stockfish on unseen standard games, MoM outperforms both dense individual expert networks and popular GPT baselines trained on aggregated data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper shows that it is possible to combine multiple LLM style models via MOE to get a more powerful* chess engine than the equivalent non-MOE. They show that their experts when trained on different datasets behave differently via an interesting vision based classifier. Along with some analysis of using RL to train the underlying LLMs and interpreting the different experts.

*The definition of power is not the standard method (Elo/Glicko) used in chess.

### Strengths
It's an interesting result for the chess community and while LLM for chess is not a novel idea their RL methods and MOE are. The attempts at interpreting the models are also strong. The choice of a smaller LLM instead of a full GPT-2 model is also refreshing.

### Weaknesses
I find parts of this paper hard to follow as the authors use terms from LLM training to describe process that RL/chess AI researchers have been doing for decades. Why are you using GRPO? How does it compare to methods used to train much stronger models like mini-max (stockfish) or UCT(Alpha Zero Chess)? They also don't provide Elo or clear direct comparisons between the models, using FIDEscore instead.

The use of text to represent games was never discussed, why is a chess model getting less than 100% accuracy on picking a legal move OK?

My main issue with this paper is that it is unfocused. I don't see a strong direction to go with it. It's not very useful to LLM people as the methods are highly specialized to chess, and they don't clearly explain what chess or the traditional RL community can gain from it.

I'm giving this a weak accept as I think it's an interesting paper, but I am not confident on it's long term impact.

### Questions
What is the Elo of your models? Can you run a tourmanment with the different instantiations and Stockfish, then report the Elo of each model. I'd recommend starting all models at 1500, and using an deviation of 350, then using Glicko 2 for the Elo updates.

Can you explain how this is self-supervised learning? To me it looks like a standard supervised learning approach.

What does "decentralized" and "compositional" mean in this paper?

Why was FIDEScore used instead of raw wins? What is the draw rate in your results?

What is the takeaway for a chess engine builder? Is this a way to outperform stockfish for the same compute? More interpretable? More human?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces Mixture of Masters (MOM), a sparse chess language model that combines multiple small GPT-based experts, each trained to emulate the distinctive style of a world-class grandmaster. A learnable gating network dynamically selects which expert to consult for each move, enabling the model to switch between offensive, positional, or defensive styles depending on the game state.
MOM addresses the homogenization problem of dense chess models that lose stylistic diversity. The authors also propose a behavioral stylometry metric using a vision transformer to verify that each expert maintains a unique playing signature. Experiments show that MOM preserves creative variability and outperforms both dense GPT baselines and single-expert models when evaluated against Stockfish on unseen games.

### Strengths
- This work introduces a very insightful framework, which is the first persona-based sparse MoE chess language model. Each small GPT expert emulates a specific grandmaster and dynamically routes by game state. I can see similar ideas easily be extended to other domains where individual styles or strategic personas matter.
- The experiments present extensive ablations and Stockfish evaluations under strict legality rules, demonstrating consistent improvements and interpretable routing behavior across different difficulty levels.
- The content of this paper is well organized and easy to follow.

### Weaknesses
- My main concern is that the scalability and efficiency of the proposed method are not deeply analyzed. Training and maintaining multiple experts plus a gating network could be computationally expensive compared with a single dense model, limiting its practicality for larger expert sets.
- The acronym MOM is a bit off.

### Questions
- How exactly is the gating network trained? Does it receive supervision from expert identities, or only learn through downstream loss during move prediction?
- When merging weights across experts, which layers are merged and which are gated, and how is alignment between different expert representations ensured?
- Is it possible for MOM to generalize to unseen players or mixed-style games?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes mixture of masters, a technique to counteract homogeneity in chess models.

### Strengths
* The paper is well written.
* The illustrations and presentation are clear.

### Weaknesses
* The paper claims to contribute "behavioral stylometry" for chess, but this was first done in McIlroy-Young et al 2021 (not cited in the behavioral stylometry section). The method is even very similar, and the paper sounds like it is claiming it for itself (e.g. "Drawing inspiration from speaker recognition systems, we employ the generalized end-to-end (G2E2) loss", which is exactly what McIlroy-Young et al 2021 do). The ethics statement also makes it sound like the behavioral stylometry metric is an original contribution "Our introduction of a behavioral stylometry metric...". 

* Despite the substantial overlap with previous work, it is not used as a baseline in the experiments, so it's difficult to know how it compares.

* In Figure 8b, the position is illegal, since it is White to move but Black is in check. If simple errors like this example are already visible in the figures, this makes me seriously doubt the validity of the paper's results.

### Questions
How does this method differ from previous work?

How do other baselines do?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes to split Chess Language Models into distinct experts each trained following a different real Grandmaster (GM), and using those expert models collectively in a Mixture-of-Experts architecture. It grounds this idea in the notion that chess player have individual styles, including strength and weaknesses, where a given style might be most suited to a specific game situation.
The paper also proposes a systematic analysis of the different expert styles through a behavioral stylometry.

### Strengths
The main idea of the paper, splitting move selection based on position to use a more appropriate model is interesting, has potential to improve current Chess Language Models, and provide for some improved, though still nebulous, method to interpret move decision.

This paper providing new tools to have stylistically different Chess Language Models is welcome, although the ethical issues that could arise (and are well mentioned by the authors in the second to last section of the main paper) should be kept in mind.

### Weaknesses
The whole stylometry axis of the paper is quite confused:
- The explanation of the methodology is close to unintelligible, due to the explanation being far too compressed and out of order, as well as a notational mess. For instance, $h_k$ is defined as the average of all k-th tokens over the temporal window [i, i+F-i] (which is only vaguely mentioned by text), and are then used to define $h_j$, which has no dependency on j (all $h_j$ are equal), unless maybe j here refers to i in $h_k$??? In that case, the $h_j$ would be a representation of position j, as well as all the F-1 following positions. The whole loss section also has troubles, that are explicited further in the next section.
- The results of the stylometry analysis are incomplete, and poorly analyzed. So far as I can tell, they are only reported in Table 3 (labeled stilometry), and explained in RQ3. The table doesn't really show that "each expert model is stylistically closest to its corresponding target master" (L410). Only half of the experts have highest similarity with their master, and only 2 (maybe 4) can be said to be explicitly closest (3 and 10, maybe 5 and 9). 7 and 8 are not even really among the top 3.
- An additional stylometry table comparing real games (unused in training data) from the 10 sampled masters would be really helpful, either in the main paper or in an appendix, to support the main table results, and to provide more context on how well their "styles" can be differentiated.

Overall, the report and presentation of the results of this paper feel confused. In addition to the issues with the stylometry results,
- In RQ2 looking into the impact of RL on individual experts, figure 5 is cited to justify a substantial increase in the ability to generalize and explore diverse options, when the figure only shows that the generated moves are slightly more likely to be legal (maybe around +2%?).
- Maybe the correct figure for this should be Table 2? But even then adding RL usually lowers win rate and fide score compared to using only SSL. The explanation of the results might be mentioning the full MoM model, but it doesn't mention that, is between RQ1 and RQ3, which both only look at experts individually, giving the impression that the first 3 questions focus on the experts, while the later 2 on the full MoM model.

### Questions
The stylometry loss definitions were quite hard to follow:
- $G_p$ is used both as a set and a number. $\frac{1}{G_q-1}$ should be $\frac{1}{|G_q|-1}$ in Eq (3) for instance. also, $\\{G_q\setminus{}g\\}$ should just be $G_q\setminus{}g$. Also in Eq (3), $v_g^q$ is used. Is is meant to be $z$ instead of $v$? Also in this equation, the contribution of $z_g^p$ is suppressed on its own centroid, but is done in a weird way. When suppressed, there is no issue. When not suppressed, the centroid becomes the sum of the previous centroid and $z_g^q$, not the average of all $z_\cdot^q$.
- $L_{MoM}$ is almost impossible to parse. n_p and G_p are not appropriate notations in an equation summing over p independently. In the second term of Eq (4), g and g̃ are used, but not quantified. Is is supposed to be a quadruple sum and not a double sum? In that case, why is that term not normalized by $|G|^2$? And overall, the choice of normalization is a bit cryptic. the first and last term are normalized by |G|, while the second term is normalized by $n^2$. Is there a reason not to just have each sum be a mean instead, keeping both lambdas?

- For Figure 5, the move legality seems to be on average 80 to 90. What is the unit? I assume those are percentages for now. I am confused on how the models are able to play with any consistency when once in every 10 moves they instantly lose the game.
- In Figure 6, the FIDEScore seem to hover around the high 60s, being maybe lower for k=5. However, in Figure 7, MoM seems to only score around 59 (percent? total score over 100 games maybe? Which would be the same in practice). Maybe the legend of Figure 6 meant to mention Stockfish level 0 instead of Stockfish 1?
- In Figure 8, the second image shows the black king on f8, when it should have moved (out of check!) 2 plys earlier.

As a very minor point, the introduction mentions that the number of chess games (~$10^{50}$ in the paper, currently estimated more around $10^{45}$ [https://github.com/tromp/ChessPositionRanking]) "far [exceeds] the number of estimated atoms in the observable universe", which is around $10^{80}$. Maybe the authors meant the number of chess games? Which is higher than $10^{120}$.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Mixture-of-Masters (MoM), a sparse mixture-of-experts architecture for chess language modeling where each expert is trained to emulate a specific grandmaster's playing style. The authors combine self-supervised learning with GRPO to train individual expert models, then use a learnable gating network to dynamically route between experts during gameplay. They also propose a vision-based behavioral stylometry metric to help train the MoE model and verify that experts capture distinctive playing signatures.

### Strengths
1. This paper is well written and easy to follow.

2. Related works are carefully discussed in the main and extended literature review, and the authors place their work in a clear position for the chess AI community.

3. The design of mixture-of-experts routing among GMs is interesting and clearly motivated.

### Weaknesses
1. I have significant concerns about the two-stage fine-tuning process for creating expert models, particularly the introduction of GRPO. Using GRPO for next-move prediction tasks lacks clear justification, as there is no chain-of-thought reasoning involved here. Moreover, the effect of GRPO on legality (Figure 5) appears more like statistical noise than genuine improvement, despite the SSL model already achieving ~80% legality. With such a reasonable starting point, one would expect GRPO to deliver more substantial improvements if it were truly effective. Also, a chess foundation model with only 80% legal move accuracy provides an inadequate basis for subsequent chess-relevant analyses.

2. Table 2 lacks accompanying discussion in the text. From what I understand, the results suggest that GRPO post-training fails to demonstrate meaningful performance improvements across most metrics.

3. The stylometry model lacks independent evaluation. The paper does not discuss how accurately the stylometry model identifies in-distribution GM styles or its generalization capability to out-of-distribution GMs or players. Given that the entire MoE framework depends on this stylometry model, a thorough and careful evaluation is very essential.

4. The results in Table 3 are concerning. In 5 out of 10 cases, the diagonal similarity scores (which should be highest for correct style matching) are not even the best within their respective rows. Only Expert 10 (1 out of 10) appears to achieve clear correspondence between the model and its target grandmaster.

5. Limitations are not discussed. For example, the paper should acknowledge: (a) the relatively low legal move rates (~80-90%) which significantly lag behind state-of-the-art chess models, potentially limiting analyses and practical applications, and (b) the computational overhead of maintaining and routing between multiple expert models compared to a single dense model, which should be compared in efficiency and may not be justified given the marginal performance gains.

### Questions
1. Can you show the exact loss curves during GRPO post-training?

2. I believe there is an error in Figure 8 (right panel) - is the white light-squared bishop incorrectly shown as giving a check to the black king?

3. Given the concerning legal move rates for individual experts, how are games between the MoM model and Stockfish handled when illegal moves are generated?

### Soundness
1

### Presentation
3

### Contribution
1
