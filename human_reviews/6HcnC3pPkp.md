# Token-Supervised Value Models for Enhancing Mathematical Problem-Solving Capabilities of Large Language Models

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
With the rapid advancement of test-time compute search strategies to improve the mathematical problem-solving capabilities of large language models (LLMs), the need for building robust verifiers has become increasingly important. However, all these inference strategies rely on existing verifiers originally designed for Best-of-N search, which makes them sub-optimal for tree search techniques at test time. During tree search, existing verifiers can only offer indirect and implicit assessments of partial solutions or under-value prospective intermediate steps, thus resulting in the premature pruning of promising intermediate steps. To overcome these limitations, we propose token-supervised value models (TVMs) -- a new class of verifiers that assign each token a probability that reflects the likelihood of reaching the correct final answer. This new token-level supervision enables TVMs to directly and explicitly evaluate partial solutions, effectively distinguishing between promising and incorrect intermediate steps during tree search at test time. Experimental results demonstrate that combining tree-search-based inference strategies with TVMs significantly improves the accuracy of LLMs in mathematical problem-solving tasks, surpassing the performance of existing verifiers.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents TVM -- token-supervised value models that can act as neural verifiers and provided tree-based LLM reasoning methods with a way to rank different parts of the search tree as more/less promising on reaching a solution in the domain of math problems. The authors state that existing SOTA like Outcome-based (ORMs) and Process-based (PRMs)  supervision models cannot rank paths effectively due to the way they are trained. ORMs only provide a reward if the answer is correct whereas PRMs provide the same reward to all steps. This causes issues in recall with PRMs since an entire step is marked as a failure even if only a few tokens are wrong.

To mitigate this drawback of PRMs, the authors introduce TVMs that utilizes different generated reasoning paths in the training set to provide different "weights" or rewards. For any substring of a correct solution, each token of that substring is allocated a weight based on the total substrings (or reasoning paths) that were generated. This can be thought of as a value function and the authors provide some soft theoretical analysis for it. 

The authors then conduct an empirical evaluation that showcases the strengths of their approach.

### Strengths
S1. The paper is very well-written and organized although it seems a bit verbose with some aspects not really required.

S2. The presented idea is very convincing and intuitive. The idea is novel and weighing up of tokens (or a partial order sequence) has been successfully explored as an idea in many different domains.

S3. The baselines are good and the empirical setup overall is good.

### Weaknesses
W1. The paper was a bit verbose even if it was well-written. There are several aspects that could have been cut. For example, I do not understand why  Tables 4, 5 were included since it does not appear to be a major contribution of the paper. Table 3 already showcases all methods with different tree-search paradigms so I am not sure why this was necessary. Please comment. An analysis of training effort would have been better here.

W2. The empirical evaluation is a bit lacking. The gains seem marginal in most cases and std. deviations are missing from these plots. Please provide them so that a better assessment can be made.  Also, it seems that this approach can be more general. I am not sure why the authors have limited to only math tasks. A more general evaluation would have improved the paper's impact.

### Questions
Overall I liked this paper and I think it will be a good fit in the program. I hope the authors are able to resolve my concerns.
Please respond to my comments on the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes token value models (TVMs), a new method for training verifiers used at test time to improve the problem-solving capabilities of LLMs. Existing training methods, outcome-supervised reward models (ORMs), and process-supervised reward models (PRMs) have drawbacks in that they can give only indirect supervision for intermediate steps. TVMs enable giving direct supervision over tokens by exploiting the probability that the token will read to a correct answer.  Experimental results show that TVM shows better predictive performance than ORMs and PRMs.

### Strengths
1. The proposed method is simple and easy to understand how it works.
2. Experiments are conducted with combinations of multiple benchmarks, LLMs, and search strategies. The experimental results support the proposed method's superiority.

### Weaknesses
Although the experimental results are interesting, the paper's presentation has some problems and thus does not explain the reasons why the TVM works well. Therefore, I think this paper needs a further revision.
  1. The paper says that TVM is superior to ORM and PRM since it uses token-level direct supervision (line 113). I'm not sure why we can say TVM is direct since ORMs, PRMs (without human annotations), and TVMs all rely on supervision signals for outcomes. No direct supervision on intermediate steps is given. Therefore, all these methods share the same problem described as the disadvantages of the ORMs (line 267).
  2. Moreover, the supervision of TVM (eq. (5)) based on conditional probability looks pretty similar to the reward for ORMs (eq. (3)). If we have $N_{tr}$ samples, the reward for a token $t_{n, k}$ in ORMs will be close to the success rate of the paths containing $t_{n, k}$. Therefore, the proposed supervision seems less novel.
  3. The paper says that one of its main contributions is that it is the first to disclose that PRMs produce a high false negative error. However, this is not true, as the explanation (line 306, Figure 2) assumes that PRMs do not use human annotation. The false-negative problem would not occur if we had human annotation on intermediate steps. Therefore, the statement is incorrect.

### Questions
1. (Figure 1) I need help understanding why ORM assigns low scores to correct steps and PRM assigns high scores to wrong intermediate steps based on this example. More convincing examples would help readers understand the superiority of the TVM.

2. I cannot understand the explanation of TVM's complexity (line 377). TVM is said to be a tree-search-based method (line 132), but the explanation here says that TVM computes eq.(6) from a set of fixed reasoning paths and does not need a tree search. I'm happy if the authors address this inconsistency. 

3. Moreover, I think the roll-out process (line 374) is also beneficial for the TVM since we can estimate conditional probabilities eq.(6) more precisely for a token if we have more sample runs. Why does TVM avoid this?

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
The paper proposes Token-Supervised Value Models, to enhance the mathematical problem-solving capabilities of LLMs. Traditional methods lack effectiveness when applied to tree search algorithms, leading to premature pruning of promising steps. TVMs address this by assigning a probability to each token that reflects its likelihood of contributing to a correct answer, enabling direct evaluation of intermediate solutions. Experiments show that TVMs outperform ORMs and PRMs in accuracy, especially in tree-search-based strategies, by reducing false negatives and improving the assessment of solution paths​.

### Strengths
1. TVM is a novel approach. It assesses each token's contribution to the likelihood of reaching the correct answer, enabling more precise evaluations of intermediate solutions during tree search.

2. The authors address that TVMs lower false negative rates by distinguishing between promising and incorrect intermediate steps, enhancing recall without sacrificing precision.

3. TVMs show consistent improvements in accuracy across various benchmarks, including GSM8K and MATH, indicating their robustness and versatility.

### Weaknesses
1. TVMs require token-level supervision, which could add complexity to the training and fine-tuning process compared to traditional verifiers.
 
2. The effectiveness of TVMs relies on reasonably accurate token-level probability estimation based on samples of successful and failed trials. For difficult problems, the sampling process controlled by the LLM can be challenging, as most samples may be incorrect, potentially reducing the effectiveness of TVMs to that of a PRM.

3. Some sections need clarification; please see the questions below.

*Typo: In several places, there is an extra "p" added after "%", e.g., "14%p".

### Questions
1. How is $N_{tr}$ determined, given its direct impact on probability estimation? Is it chosen through parameter tuning, or is there additional insight guiding this choice?

2. A follow-up question: During fine-tuning, do these $N_{tr}$ trials often share some prefix tokens, and how to control them? For example, in Figure 4, the three paths share the first $a-1$ tokens, allowing those tokens to have the value 0.33. If they don’t share any tokens, it seems the problem would reduce to a PVM.


3. How to interpreted Figure 3(b)? The "Verifier’s Scores Histogram for Correct Sampled Solutions" shows that the weights are more concentrated for correct solutions, but what about the histogram on the right side?

### Soundness
3

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
4

### Summary
This paper proposes token-supervised value models (TVMs) as a superior way to verify whether tree-search based math reasoning models are on the right track or not. The authors compare TVMs to output-supervised reward models (ORMs) and process-supervised reward models (PRMs). They argue that both of these existing verification methods, having been designed for evaluation of full trajectories, are not suitable for verifying partial trajectories, especially at the token level. The conduct experiments with two math datasets, showing that the use of TVMs improves performance. Additional experiments are also provided.

### Strengths
* The authors make a clear argument about what current verifiers (ORMs and PRMs) lack, and why TVMs should be able to help. Limitations of ORMs and PRMs are justified with some empirical support. Logically thinking, the progression from ORMs to PRMs to TVMs seems natural.

* Figures and examples (although small and hard to read) are useful.

* The method is clearly described, along with a formulation of prior (ORM and PRM) methods.

* Experiments show a notable improvement when using TVMs, especially for step-level beam search and for the MATH benchmark.

* Additional experiments around FLOPs used and execution time support the idea that tree-based search methods are more efficient, especially if one can pair them with effective verifiers.

### Weaknesses
* One obvious disadvantage of TVM compared to PRM is that during *training*, the same N_{tr} sampled reasoning paths per training problem are expected to provide a lot more supervision in TVM. Specifically, suppose a typical reasoning path has 5 steps and 50 tokens. Then, in PRMs, it is easy to imagine several reasoning paths sharing similar partial paths  steps, leading to a good signal for supervising the 5 steps. However, with ~50 token long paths, the chances of partial sampled paths overlapping would decreases quickly and would be very slim after the first 15-20 tokens. So I'm not sure what kind of token-supervision signal can one actually get for the latter parts of the reasoning paths -- unless paths are sampled differently or N_{tr} is higher. The paper doesn't seem to address / discuss this.

* The gains, when using TVM, are rather slim in step-level beam search for the GSM8k dataset, only about 1%. This is not very different from the gains when using best-of-N search. If fact, for Mistral, the gains from using TVM are slightly higher for best-of-N compared to step-level beam search. This goes against the motivational argument presented earlier, namely that ORMs and PRMs are suitable for best-of-N search but not tree search, and it's the latter than needs another verification method.

* In general, TVM doesn't seem to help that much on GSM8k.

* I don't understand what Proposition 4.1 is trying to say. Why is it not "trivially true" if the reward function is 0 at intermediate tokens and 1 at the output token iff is it the correct output? Is this proposition related to TVMs or also equally applicable to ORMs and PRMs (which seems to be the case)? I am not following the significance of this formal proposition.

### Questions
Please see the weaknesses section above. I am happy to consider increasing my score if I can get more clarity.

====
REVIEW RATING UPDATED AFTER AUTHOR RESPONSE
====

### Soundness
3

### Presentation
3

### Contribution
2
