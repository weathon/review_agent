# MarkovScale: Towards Optimal Sequential Scaling at Inference Time

- Avg Score: 6.00
- Decision: Reject
- Scores: 6, 6, 6

## Abstract
Sequential scaling is a prominent inference-time scaling paradigm, yet its performance improvements are typically modest and not well understood, largely due to the prevalence of heuristic, non-principled approaches that obscure clear optimality bounds. To address this, we introduce a principled framework that models sequential scaling as a two-state Markov process, uncovering its fundamental properties and providing closed-form expressions for key aspects, including the conditions under which sequential scaling enhances accuracy, the theoretical accuracy upper bound, and the convergence rate. Leveraging this formulation, we develop MarkovScale, a practical system that applies these optimality criteria to achieve a theoretically grounded balance between accuracy and efficiency. Comprehensive experiments across 3 backbone LLMs and 5 benchmarks show that MarkovScale consistently outperforms state-of-the-art parallel and sequential scaling methods, representing a significant step toward optimal and resource-efficient inference in LLMs. The source code will be open upon acceptance at https://open-upon-acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors pose sequential test time scaling for LLMs as a two state Markov chain: Given the present completion from the model (or the question initially) either the model generates a correct (C) or an incorrect (W) completion in the next step of inference. By analyzing this Markov chain they identify the conditions under which scaling is beneficial, neutral and detrimental. They also relate the expected convergence to the probability of the model moving from C->W or from W->C which then enables them to determine upper and lower bounds on performance when scaling in this way. Using this they develop a series of methods that aim to determine when to use scaling, and how much scaling to use. They then compare their methods with other test-time scaling methods, showing it has favorable performance on 3 models and 5 test sets.

### Strengths
- This work is timely and addresses a key question about test-time scaling, namely how much to do and in what circumstances it is beneficial to do so
- The performance bounds are nice and give a benchmark against which to compare methods
- Despite some problems with the exposition the central idea in this paper is quite elegant and uncomplicated.

### Weaknesses
- Figure 3 could do with some improvement. I think a bar chart or some other chart that doesn't imply an interpolation between benchmarks might be more appropriate.
- The exposition in section 3.3 is a bit sloppy. For instance, where does this theoretical bias term originate from? What does $q$ represent (a question I'm guessing)? Is $p$ in (6), (7) and (8) $p_0$ or $p_i$?
- I think the framing around section 3.3 could do with some justification: I'm a bit skeptical about model capability and problem difficulty being disentangled in this way. Surely the models capability is also related to the zero shot probability of correctness, just as the problems difficulty is related to $a$ and $b$? Additionally you frame $a$ and $b$ as being intrinsic to the model, yet you clearly estimate it separately for each dataset (or maybe example if $q$ represents a single question) or how else would the results in figure 2 be different? I would have like a bit more clarity about exactly what $a$ and $b$ represent and how they were estimated.
- I would have liked to have seen baseline results with no test time scaling.

### Questions
All of your methods appear to perform quite similarly, especially on the smaller models, are there any reasons to prefer one over the others?

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
2

### Summary
This paper proposes a new algorithm, based on a discrete Markov process, for inference time scaling of LLMs via sequental scaling. I am no expert in LLMs or methods research on top of LLMs, so I don't have high confidence in my review. What is more, I am short on time due to the semester start. Apologies if my reviews are a bit short. I am happy to engage in reviewer discussion should be concerns not be clear. 
That said, I think the suggested approach appears sensible, easy to understand and verify, and leads to improved results compared to baselines. So, from my far away perspective, I think this paper is a relevant contribution to the field.

### Strengths
- Clear theoretical framework.
- Easy to understand approach yet providing good empirical accuracy.
- Clear and consistent improvements against many benchmarks.

### Weaknesses
- I am too far away from the field to judge this in detail

### Questions
- The notation around EQ2 looks a bit messy. Sometimes there is an index i, sometimes there is not. a P seems to be missing in EQ2. Can you double check and fix the notation to be consistent?

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
3

### Summary
The paper first makes an observation that parallel scaling tends provide better performance but is token efficient compared to sequential scaling approaches. The paper aims to optimize the sequential scaling.
Given the inference-time scaling, it seems that the paper is trying to formulate the problem of scaling as a two-state Markov process. The work uses this formulation for the correctness probability to get the performance bounds: neutral, upper, and lower and use that as the theoretical basis to directly estimating the convergence accuracy.
MarkovScale proposed in the paper has several variants: (1) gating strategy, (2) MAP-based optimal scaling, and (3) training-free version. 
The paper evaluates the MarkovScale approach on combinations of several models and benchmark to compare against several methods including ones based on budget, early stopping, self-consistency, etc.
The paper demonstrates that it achieves better results (accuracy) given same token consumption.

### Strengths
* The paper is tacking a very important topic of token-efficient inference-time scaling.
* It seems the paper is trying to move from previous heuristic based approaches to an approach that is a little more backed by theoretical formulation which seems nice.

### Weaknesses
* Formulation seems rather oversimplified as overall reasoning process in inference-time scaling is not easy to boil down to simply correct vs incorrect. It is literally a reasoning process where things can go south then use that as a context to later converge on a better outcome. However, the oversimplification of the formulation seems to understate the significance of this.
* Seems rather unclear how the transition probabilities and zero-shot probability are computed in the MarkovScale.

### Questions
* Can you please elaborate how transition probabilities and zero-shot probability are computed in the MarkovScale. Some details would be great.
* Is there any projection to how this can be generalized to different models of different scale? It seems that the models evaluated in the paper are pretty small.
* Figure 1 stops the evaluation at around 700k tokens. What happens after that would also be interesting data to show whether the work (as well as the other approaches) are really controlling the inference-time scaling in an optimal manner.
* It is also interesting to see in Table 1 that MarkovScale0 seems to perform better than other variants a few times. Is this just error margin? Given that the MarkovScale based on MAP could be assumed to provide a good performance, does this mean that MLP was not trained enough? It seems that it is difficult to fully rule out the overfitting?
* Is there a good results showing how hyperparameters were determined?

### Soundness
2

### Presentation
3

### Contribution
3
