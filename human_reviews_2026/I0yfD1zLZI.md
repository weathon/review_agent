# Beyond Solving: A Closer Look at LLMs as Solution Verifiers

- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Large language models (LLMs) can act as both problem solvers and solution verifiers, with verifiers improving solver performance by selecting high-quality answers from a pool of candidates. However, prior studies of solver–verifier interactions have been limited, focusing mainly on self-verification and rarely examining how verifiers judge outputs from models in their own or in another model family. Modern LLMs also undergo extensive post-training, but its effect on verification remains unclear. We present a systematic study across 37 models spanning multiple families, sizes, and base vs. post-trained variants, evaluated on 9 benchmarks covering logical reasoning, structured puzzles, symbolic computation, mathematics, commonsense, factual recall, and domain knowledge. We compare self-verification with verification within the same family and across different families. To support this, we introduce and empirically validate *verifier gain*, a metric that predicts the performance improvements from *test-time verifier-based rejection sampling*. We analyze how metrics like verifier gain and false positive rate scale with model size and post-training, and characterize differences in dataset verifiability. Our findings show that cross-family verification is especially effective; post-training reduces self-improvement but strengthens cross-family improvement; and mathematical and logical tasks exhibit the highest inherent verifiability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a systematic study on the capability of LLMs to act as solution verifiers. The research investigates solver-verifier interactions across 37 different models—spanning various families, sizes, and post-training methods—on 9 diverse benchmarks, including logic, math, and commonsense reasoning. To measure the impact of verification, the authors introduce "verifier gain," a new metric that predicts the performance improvement achieved by using an LLM to select the best answer from multiple candidates. They found that cross-family verification is often more beneficial than self-verification and intra-family verification. The study analyzes how this verification ability scales with model size and post-training, while also characterizing the inherent "verifiability" of different datasets.

### Strengths
This work conducts extensive experiments on solver-verifier interactions and found that verifiers are more helpful when they are from a different family. This is insightful and could benefit future work.
The proposed verifier gain is a good indicator to simulate the gains.

### Weaknesses
1. The "verifier gain" metric is proposed but did not explain the reason behind it, which is confusing. In addition, since it is the upper bound, the name may not propriate. For example, if I have a verifier that can achieve 100% precision, the solver-verifier still cannot achieve 100% if the solver is too weak, even with unlimited sampling.
2. The observations are mainly empirical and lacks in-depth analysis. For example, they found that self-verification does not have much gain but did not explain the reason. One possible reason is self-enhancement bias [1]. To explain the reason, analysis of precision (in equation 1) in different conditions should be conducted. Is the precision lower when verifying the model's own responses?
3. Some observations can be anticipated, which may limit the novelty of this work. For example, self-verification is not so effective due to self-enhancement bias.

[1] Jonathon D Brown. Evaluations of self and others: Self-enhancement biases in social judgments. Social cognition, 4(4):353–376, 1986.

### Questions
In Figure 2, 5, and 6, what is the meaning of "r = 0.xxx" in the legend?

### Soundness
3

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
5

### Summary
This paper studies the relationship between solver-verifier interactions across 37 models and 9 benchmarks. More specifically, the authors study a verifier's ability to correctly judge a solver's outputs, comparing self-verification with verficiation of same-family and corss-family models.

In terms of methodology, the authors propose the verifier gain metric: Prec(SVD) - SolverAcc(SD). Moreover, they group models according to their size and pre-trained/post-trained. Then each solver-verifier pair can be categorized into 1) self-verification, 2) intra-family verification, and 3) cross-family verification. 21 post trained and 16 pre-trained models from 0.5-72B are compared on 9 datasets.

The first resuls show that solver accuracy improves with model capacity, especially post-trained models, and similarly, verifier accuracy improves with verifier model's own solver accuracy. While those are unsurprising, the authors identify less gain during self-verification and models that are more accurate solvers are not better at self-improvement, with the largest gap among cross-family verification. I think this was expected but it is good to have an experimental result to confirm it. Additionnally, the authors show that using the verifier gain is a good predictor for the performance improvement when using a verifier for rejection sampling.

The second set of results show that it is worth it to use a verifier with different solutio ndistributions than the solver which is a good finding. Moreover, and unsurprisingly, post-trained models improve solver accuracies. Interestingly, the same is not true for verifiers: a post-trained model can decrease its ability to improve itself or models from its own family, but it improves its capability as a cross-family verifier. Finally, the authors show that some tasks are inhenretly easier to verify than others.

Overall, this paper is motivated and confirm common (and potentially new for some readers) hypotheses that researchers had regarding solvers/verifiers. I appreciate that the authors answers those doing detailed experiments. I think one difficulty of this paper is the presentation: many plots, analysis, and text. I would encourage the authors to maybe aggregate all their results in one table or charts to simplify the take-away messages (although I acknolwedge the presence of take-away sections but I would go even one step further).

### Strengths
- Sound experimental design to confirm hypotheses about solvers and verifiers.
- Many models and lot of intra/cross-family experiments.

### Weaknesses
- On one hand, it is good to confirm experimentally intuitions that researchers may have already had. On the other one, it may sound rather incremental.
- Presentation could be improved.

### Questions
What would be your general guidelines when starting a new project involving solvers and verifiers?

### Soundness
3

### Presentation
2

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
This paper explores the effectiveness of using large language models (LLMs) for verification in problem solving. The study examines the interactions between solvers and verifiers across 37 models belonging to intra and cross family, covering various benchmarks such as logical reasoning, puzzles, symbolic computation, mathematical problem solving, commonsense reasoning, and domain knowledge. The authors of the paper found that verification accuracy alone does not provide a complete picture of the expected improvement from rejection sampling using a verifier. They propose a new metric called verifier gain, which better characterizes the improvement. They found that gains are often lower for self-verification and intra-family verification than for cross-verification, especially as model size increases or post-training is applied. The decrease in verifier gain is correlated with an increase in the similarity between the solution distributions of the solver and verifier. The authors also found that some tasks are inherently easier to verify with an LLM than others, with easier tasks involving logical reasoning, mathematical reasoning, or structured puzzle solving, while more difficult tasks require domain-specific knowledge or implicit knowledge about the world.

### Strengths
•	This paper explores the interaction between solver and verifiers across 37 models belonging to intra and cross model family which is a significant effort.

•	The paper proposes a new metric called verifier gain to study a verifier’s ability to correctly judge a solver’s outputs.

### Weaknesses
- While the paper provides a comprehensive study of LLM-based verification for problem solving, overall novelty is somewhat limited.

### Questions
- Line number 287-289 says “As we move to intra-family verification we begin to see more gains, but cross-family verification shows the greatest potential for improvement from a verifier.”  How this gain is related with a) the data used for post-training of the specific cross family model and b) individual evaluation data ?
- The clarity of Figure 6 is somewhat lacking. In particular, I am curious about the rationale behind the number of points (different colored dots) displayed for Cross-Family Verification. Could you provide an explanation for this and specify the total number of points involved?
- How about comparison with baseline solver-verifier methods? Can you please elaborate on this?

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
3

### Summary
The paper presents a comprehensive study o LLM-based verification for problem solving. The paper finds that verification accuracy paints an in complete picture of the expected improvement from rejection sampling from a solver using a verifier, and derive and validate a measure called verifier gain to better characterize that improvement. The paper presents extensive experiments on LLM validation.

### Strengths
This paper conducts extensive experiments on using LLMs as verifiers, and from an experimental perspective, the evaluation is quite thorough.
The proposed concept of verifier gain is well-motivated and reasonably supported by the experimental results.

### Weaknesses
First, I do not find the motivation of this paper convincing, like the statement:
>Yet, verification with LLMs remains underexplored. Prior work has mainly examined scenarios where solver and verifier are the same model or from the same family; far less is known about how verifiers behave when judging outputs from other families.
 
However, many models, even from quite early on, have extensively used LLM-based verification during the post-training stage. For example, models such as Kimi K2, Qwen-3, and Minimax-M2 all employ rubric-based approaches using LLMs for verification, which also include the so-called cross-family validation.


Second, a large portion of the experiments in this paper are conducted on verifiable tasks such as GSM8K, AIME, and GPQA. I do not believe these experiments have much practical value, as I fail to see any real benefit of using verifiers in production settings for tasks that are already verifiable.

Moreover, the majority of the experiments are carried out on open-source models. Why not test on more capable closed-source models such as GPT-5, Gemini, or Claude?

Finally, the contribution of the paper is somehow trivial. Many experiments remain at surface-level observations. For instance, the authors claim that “verifier models are biased toward accepting incorrect solutions during self-verification or intra-family verification”? What is the underlying reason? And how should experiments be designed to validate this claim?

### Questions
See Above

### Soundness
3

### Presentation
3

### Contribution
2
