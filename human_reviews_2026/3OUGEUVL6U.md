# ABS: Enforcing Constraint Satisfaction on Generated Sequences via Automata-Guided Beam Search

- Avg Score: 2.00
- Decision: Reject
- Scores: 0, 4, 4, 0

## Abstract
Sequence generation and prediction form a cornerstone of modern machine learning, with applications spanning natural language processing, program synthesis, and time-series forecasting. These tasks are typically modeled in an autoregressive fashion, where each token is generated conditional on the preceding ones, and beam search is commonly used to balance exploration and fluency during decoding. While deep learning models and Large Language Models (LLMs) excel at capturing statistical patterns in this setting, they remain ill-equipped to guarantee compliance with formal constraints.
In this paper, we introduce ABS: a general and model-agnostic inference-time algorithm that guarantees compliance with any constraint that can be compiled into a Deterministic Finite Automaton (DFA), without requiring retraining. ABS leverages the DFA to guide a constrained variant of beam search: at each decoding step, transitions leading to violations are masked, while remaining paths are dynamically re-ranked according to both the model’s probabilities and the automaton’s acceptance structure. We formally prove that the resulting sequences are guaranteed to satisfy the given constraints, and we empirically demonstrate that ABS also improves output quality. We validate our approach on three distinct tasks: constrained image-stream classification, controlled text generation, and text infilling. In all settings, ABS achieves perfect constraint satisfaction, while outperforming or matching state-of-the-art baselines on standard quality metrics and efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This is a constrained LLM decoding paper with a lookahead heuristic.  The constraint is provided by a DFA and a requirement to have a fixed output length $T$.  The algorithm is beam search.  At time $t$, the lookahead heuristic 
* kills off beam search candidates that cannot reach a DFA final state within the remaining $T - t$ steps
* penalizes beam search candidates that can reach a DFA final state only by using close to $T - t$ steps

(Dijkstra's algorithm was run on the reversed DFA during preprocessing, to precompute for each state the minimum number of steps required to reach a final state.)

### Strengths
The idea of using a lightweight lookahead heuristic to reweight hypotheses is reasonable.  Some constrained generation papers don't use any lookahead heuristic at all because they don't want to invest in learning.  But the reweighting heuristic here -- see (3) and Alg. 1 lines 9 and 15 -- is cheaply found by static analysis (given two hyperparameters).  It is sort of cute, so perhaps it could be an engineering sweet spot.

That said, the reweighting has no formal guarantees.  The only guarantees investigated here hinge on a hard pruning step that tries to kill off hypotheses that are not in the prefix language of the DFA.  And that idea is neither new nor done correctly here; see below.

### Weaknesses
Lookahead may indeed help in constrained generation.  However, the heuristic used here is hardcoded and doesn't seem so general.  First, with regard to the probability distribution:

* The method certainly is not sampling or maximizing from the posterior of the LLM given the constraint, which is the goal of some of the related work discussed at lines 462-476.
    * The particular heuristic is greedy and doesn't fully plan ahead.  It does "natural generation when slack is high" (line 236), which suggests that it will follow the LLM well in the early part of the string, but less well if necessary in the later part of the string.

* Dijkstra's doesn't know anything about the LLM probabilities (except that two scalars are optimized on the validation set: line 240).  So although beam search may be guaranteed to reach a final state, to get there it may have to generate a string that has low or even zero probability under the LLM, with the suffix tokens being especially improbable.  (This contrasts with lookahead methods that do learn an estimate of future probability cost, e.g., https://aclanthology.org/N18-1085/ .)

Second, with regard to the hard constraint:

* The proposed heuristic explicitly tries to avoid prefixes that don't make fast enough progress toward a final state that is far away.  But that's only one danger in constrained generation.  For example:
    * What about prefixes that progress too quickly toward a final state that is close?  Suppose the DFA accepts only a single complete English sentence.  If the given $T=1000$, then the challenge early on is to navigate to states that have long paths to the final state, not short ones (i.e., choose a sentence structure that will support a 1000-word sentence).  Dijkstra's algorithm can't help there.  
    * Nor can Dijkstra's algorithm help with parity.  E.g., it won't know to prune the current state if all paths from the current state to a final state have even length but $T-t$ is odd.

Appendix A claims to prove that the given construction works anyway.  But the proof is in error, which explains why the cases above can arise.  The bug is at line 674.  Suppose a current hypothesis at time $t$ has reached final state $q$.  Then $d_q=0 \leq T-t$ as claimed, and the hypothesis is considered "valid" in the terms of the paper.  But it is not guaranteed that there is any valid successor state $q'$ with $d_{q'} \leq T-(t+1)$, i.e., line 674 can be false.  In other words, although a final state trivially has a short path to a final state, its successors might not!

Since that bug seems fatal, I won't bother to study the experiments in section 5 unless urged to by the AC or the other reviewers.

Here for the record is the correct approach to the hard constraints:

If the only hard constraint were that the output must be accepted by the DFA, then there would of course be a very simple approach to ensuring what Theorem 1 calls "soundness": Trim the DFA to include only co-accessible states (that is, states that have a path of any length to a final state), and then require the LLM to follow paths in the DFA.  That is enough!

This is equivalent to pruning hypotheses that are not in the prefix language of the DFA.  And of course this obvious idea has been published, probably first in https://aclanthology.org/2021.emnlp-main.608/ (which went beyond DFAs and allowed any context-free constraint, using Earley's algorithm to keep track of whether each hypothesis was in the prefix language of the given CFG).

The paper here simply imposes an additional hard constraint that the output should have length $T$.  But that can be handled by intersecting the DFA (or CFG) with $\mathcal{X}^T$ before using it.  Done!  

Of course, this solution is correct only in the sense that it ensures soundness.  Because the pruning is greedy, it distorts the posterior distribution.  A number of papers have tried to correct this distortion, at least approximately, and perhaps the authors can find an effective heuristic with something like equation (3).

Minor comments:

* I was initially confused at line 087 about why there was no EOS symbol, since the paper said at 085 that the length $T$ "can vary from one sequence to another".  Answer: It appears that $T$ is specified as an argument to the constrained generation method.  Thus, $p(x_{1:T})$ denotes the probability of $x_{1:T}$ as a prefix, not as a complete string.  This should be clarified.

* The elements within the beam are usually called hypotheses or candidates, not beams.  The term "beam search" comes from an analogy to a flashlight whose beam lights up only a narrow portion of the search space, but the portion is wide enough for one to find a good route to one's destination.

### Questions
Am I mistaken?

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents ABS, a novel approach for performing constrained sequence generation by defining a DFA describing the required constraints and using a modified beam search that enforces that the generation complies with the DFA. To avoid dead-ends, ABS has a ramping push-up that gradually biases the search towards solutions that are accepted by the DFA. Theoretical analysis establishes the soundness and complexity and experimental analysis demonstrates the performance of ABS vs. several relevant baselines across three different benchmarks, namely constrained image sequence classification, constrained text generation, and text infilling.

### Strengths
Strengths:
- Novel approach for constrained sequence generation using automata-guided beam search
- The paper provides theoretical guarantees for the soundness and complexity of the approach
- The experiments demonstrate the promising empirical performance of ABS across three benchmarks

### Weaknesses
Weaknesses:
- The paper does not discuss various related works on constrained beam search using automata. This includes, for example, [1] [2] [3] [4] [5]. The paper should ideally highlight the key differences from previous works that are highly similar in nature (i.e., using automata to encode constraints as part of beam search generation).
- Comparison with baselines: It is not clear why baselines not applied uniformly: for example why Ctrl-G is not applied in the supervised setting of constrained image classification, and also not used in the other two domains (constrained text generation, text infilling). This is particularly important as in the only experiments where it is being used (Table 1 unsupervised) it seems largely comparable to ABS (better than ABS in two metrics and worse than ABS in two metrics).
- The paper shows that ABS is consistently faster than Ctrl-G. However it seems that the relative difference gets smaller with beam size and in terms of absolute value seems somewhat fixed at around 3 seconds (not getting bigger with beam size). This does not seem like a very strong point.
- Recent SOTA approaches that were shown to outperform Ctrl-G and Outlines are not considered as baselines, e.g., [6] [5]
- Mapping from LLM tokens to the alphabet seems to have an assumption that each symbol in the alphabet is a concatenation of tokens. Is that correct? And if so, it should be clearly stated.


[1] Anderson, P., Fernando, B., Johnson, M., & Gould, S. (2016). Guided open vocabulary image captioning with constrained beam search. arXiv preprint arXiv:1612.00576.

[2] Post, M., & Vilar, D. (2018, June). Fast Lexically Constrained Decoding with Dynamic Beam Allocation for Neural Machine Translation. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers) (pp. 1314-1324).

[3] Deutsch, D., Upadhyay, S., & Roth, D. (2019, November). A general-purpose algorithm for constrained sequential inference. In Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL) (pp. 482-492).

[4] Kuchnik, M., Smith, V., & Amvrosiadis, G. (2023). Validating large language models with relm. Proceedings of Machine Learning and Systems, 5, 457-476.

[5] Dong, Y., Ruan, C. F., Cai, Y., Xu, Z., Zhao, Y., Lai, R., & Chen, T. XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models. In Eighth Conference on Machine Learning and Systems.

[6] Collura, V., Tit, K., Bussi, L., Giunchiglia, E., & Cordy, M. (2025). TRIDENT: Temporally Restricted Inference via DFA-Enhanced Neural Traversal. arXiv preprint arXiv:2506.09701.

### Questions
Please see my list of concerns and questions above.

### Soundness
2

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
4

### Summary
The paper presents ABS, a DFA-guided method to constrain DNN output with regular languages. ABS leverages the DFA to guide a constrained variant of beam search. The paper presents three evaluation tasks: on constrained image-stream classification, constrained text generation, and text infilling.

### Strengths
+ The paper presents the algorithm for strict syntactic enforcement of properties during ML model generation. The algorithm integrates with beaming search decoding algorithm. 

+ The authors prove the soundness of their algorithm. 

+ The experiments span three domains: image classification (extended to include ordering), text generation and text in-filling. On these results ABS shows better utility than Outlines and Guidance.

### Weaknesses
- The approach is tightly tied to bream search as the decoding strategy. This may be a limitation, especially as other decoding strategies may be more efficient for some generation strategies. 

- The paper’s evaluation misses some closely related work. For text generation tasks, it should compare ABS experimentally with Syncode (Ugare et al. TMLR 2025), which similarly ensures hard-constraints, however using context-free grammars (which are more expressive than regular languages) to express syntactic rules, and Itergen (Ugare et al. ICLR 2025), which allows checking with arbitrary semantic rules during generation. 

- The experiments on the image generation and text infilling are done with relatively small and old models (CNN+Fashion MNIST and GPT2, respectively).  Text generation benchmarks only include Llama 8B as the base model. Other models with comparable size may give different results and are worth including in the evaluation. 


(Ugare et al. TMLR 2025) https://arxiv.org/abs/2403.01632

(Ugare et al. ICLR 2025) https://arxiv.org/abs/2410.07295

### Questions
Question: Can ABS be extended for constrained decoding beyond beam search?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The authors proposed ABS (Automata-guided Beam Search), a model-agnostic inference-time algorithm that guarantees generated sequences satisfy any formal constraint expressible as a Deterministic Finite Automaton (DFA). Abs first process logical constraints into DFA, and then calculate the chance of model being able to reach success state, and mask out the beam that fail to do so. The method ensures hard constraint satisfaction while requiring no additional auxiliary model.

### Strengths
1. The method is simple and efficient.
2. The paper is well written and easy to follow.

### Weaknesses
I have two main concerns about this paper: (1) Lack of detail reference to an extremely similar prior method Ctrl-G, and (2) lack of comparison to Ctrl-G on several baselines. I'll list the detail here:

1. The Abs method itself seems to be a simplify version of ctrl-g, which both process logical constraints into DFA format and mask out the tokens that cannot reach to the success state during generation. While Ctrl-G requires an auxiliary hidden markov model to provide soft guidence during generation, the simplify version, which only apply the mask, seems to do the exact same things as this Abs method. I hope the author can clarify the difference between these two work, and what's the advantage of the proposed method.
2. Following 1, the reference to Ctrl-G seems to be lacking considering the similarity between these two methods. Currently Ctrl-G is mostly just mentioned during the experiment section as a previouse SOTA.
3. Following the similarity of these two work, I think the authors need to have comparison with Ctrl-G on Text-infilling tasks and ordered commongen, since it's an important baseline and is capable of handling complex logical constraints and provided such results in their paper.
4. Following 3, since Ctrl-G shown that their method is able to handle complex constraints together (keyphrase generation, text-infilling, word counting, at the same time), it is important to show how Abs are able to handle such cases. From my understanding, without the auxiliary model to guide the generation, it seems to me that Abs are not able to handle such complex logical constraints.

In summary, the current results shown in the paper is not conprehensive and lack critical comparison to an extremely similar baseline. I hope the author can provide additional experiment results since Ctrl-G paper have provided both training code and their evaluation dataset, which also includes a human evaluaiton process. To propose this new method, a comprehensive comparison to the prior work is necessary. Please refer to Ctrl-G code and dataset.

I will be happy to raise score if the authors can provide the additional results and clarify the differences between the two methods. Also, provide clear reference to prior work in the paper.


Code: https://github.com/joshuacnf/Ctrl-G

Eval Dataset: https://billkunghappy.github.io/Ctrl-G/

### Questions
None

### Soundness
1

### Presentation
3

### Contribution
1
