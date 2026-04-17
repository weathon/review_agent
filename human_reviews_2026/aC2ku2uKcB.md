# On the Limits of Test-Time Compute: Sequential Reward Filtering for Better Inference

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Test-time compute (TTC) has become an increasingly prominent paradigm for enhancing large language models (LLMs). Despite the empirical success of methods such as best-of-$n$ (BoN) sampling and sequential revision, their fundamental limits remain unclear. We address this gap by analyzing a mixture-of-reference policy model and proving that standard BoN is inherently suboptimal. To move closer to the optimal frontier, we propose reward-filtered sequential inference, a simple procedure that selectively incorporates only high-reward generations into the context. This mechanism concentrates computation on superior policy candidates and suppresses inferior ones. On the theoretical side, we show that reward-filtered sequential inference yields strictly stronger guarantees than standard TTC paradigms. On the empirical side, we evaluate our method across diverse benchmarks and observe consistent improvements over widely used approaches, demonstrating the practical effectiveness of our framework.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the fundamental limits of parallel Test-Time Compute (TTC) methods like Best-of-N.

By assuming that pretrained LLMs are a "mixture-of-reference-policies", the authors provide a clear proof that parallel TTC is inherently suboptimal and establish a tighter theoretical performance limit that can only be reached by sequential methods. This holds only when there is a unique response that attains perfect reward (such as a final numerical answer in math).

The paper proposes Reward-Filtered Sequential Best-of-N, a simple sequential algorithm. The method generates responses iteratively and selectively appends only high-reward outputs to the context for subsequent generations. 

Both the theoretical analysis and empirical results show that RF-SBoN outperforms parallel BoN and non-filtered sequential baselines, demonstrating a more effective use of the test-time compute budget.

### Strengths
The authors make a good contribution to our understanding of inference-time algorithms within their idealized theoretical setup. It helps to motivate a practical algorithm. 

RF-SBoN is compelling in its simplicity. It is intuitive, easy to implement, and directly motivated by the paper's theoretical goals. I foresee this potentially being a popular method if it gains traction and end-users validate its performance.

The experiments are well-structured, comparing RF-SBoN against the appropriate baselines (BoN and a non-filtered sequential approach). The use of multiple models, diverse benchmarks, and difficulty-stratified analysis provides solid evidence for the robustness and effectiveness of the proposed method.

### Weaknesses
- Improved "budget efficiency" is framed in terms of the total number of model calls. However, parallel, batched decoding is very efficient on our hardware. This makes vanilla BoN very appealing due to its *fixed* number of N samples generated in parallel. In contrast, RF-SBoN requires unknown N sequential steps, making it potentially slower and less GPU-efficient. Moreover, you are adding more input tokens because of appending and maintaining the "good" responses within the context.

- The RF-SBoN procedure actively conditions the LLM's future generations on outputs that the reward model scores highly. This creates a feedback loop that is highly vulnerable to reward model hacking. If the reward model has systematic flaws (e.g., a preference for verbosity over correctness), RF-SBoN will amplify this bias by repeatedly showing the model examples of it. This conditioning also reduces the diversity of the generation history, which could be detrimental, especially if the base LLM or reward model is not already very strong. I would encourages authors to discuss reward hacking in inference-time settings (qualitatively with respect to related work or quantitatively in their paper).

- This is less of a concern. However, the theoretical results rely on a "mixture-of-reference-policies" model of the pretrained LLM. While this is a useful analytical tool, it is a strong structural assumption. The paper's theoretical claims are a direct consequence of this modeling choice. They also assume the existence of a unique best response. For example, I don't see this applying for any real setup (outside math or Q&A) where you may have many equally good responses.

### Questions
-  Given that RF-SBoN is inherently sequential, how do you see its practical applicability in latency-sensitive applications?

- Your method refines the generation history based on reward scores. Did you notice cases where the generation process collapsed into a deterministic answer or into a "reward hacking" echo chamber? I feel that may be the case with PureSeq because the empirical performance of PureSeq is usually as good or worse than BoN. This is surprising unless we account for the two mentioned factors. Perhaps, to make the method more robust to reward model exploitation, you could filter on a lower confidence bound of the reward. However, that would add complexity.

 - Rewards are generally not calibrated across contexts, making it difficult to compare reward scores. Even within a prompt, Best of n only cares about the rank induced by the responses, rather than the scores themselves. So I feel extra care should be taken to make sure you can compare the evolving rewards to a fixed threshold. This may be valid in your setup since the reward for a new answer, r(a_i, x), is always evaluated against the static initial prompt x. Why not keep the top-k responses?

- The presence of plateaus and dips with AIME makes me question how you estimated the accuracy. Is there a way we can add bootstrap CIs to the plots? Is this behavior observed in practice with AIME?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a new algorithm, RF-SBoN (see Algorithms 2 and 3), that adapts sequential best-of-n (BoN) sampling by only including responses that fall above a certain reward threshold.
Furthermore, the authors develop theory for evaluating test-time compute (TTC) methods under a certain modeling assumption on the pretraining corpus of an LLM.

**I lean towards rejection of the paper.** As outlined below, the theoretical foundation hinges on unrealistic assumptions, and the algorithmic novelty seems to consist of only including responses into the context whose reward lies above a certain threshold, which is a marginal contribution.

### Strengths
Under certain assumptions on the LLM training data, the authors evaluate different TTC methods and e.g. show that vanilla BoN is suboptimal.

The experimental results show that the proposed methods outperforms vanilla BoN and a simple sequential baseline in terms of accuracy over $N$.

### Weaknesses
### Presentation
I find the paper somewhat difficult to follow. I'm not sure what's the best way to present things, but I feel like the paper could be improved by a more clear presentation, including a more precise problem statement, more clear definitions, and goals throughout the paper/sections. (E.g., a more clear distinction between "showing that BoN under these assumptions on the pretraining data is suboptimal", and "derive a new TTC method for sequential TTC".)

### Theory
The assumptions made on the pretraining data seem extremely unrealistic: it seems the authors are assuming all tokens in a pretraining sample (except a token prefix) are sampled independently instead of autoregressively (line 161). They briefly mention this is in line with previous work, such as [1]. After skimming [1], it seems to me like this work actually assumes autoregressive dependency though (compare the last equation on page 5 therein). If there is a reasonable explanation for this assumption, the authors should elaborate more.

The main theoretical claim here -- Theorem 4.3 -- depends on this assumption on the pretraining dataset, so a more careful evaluation of the assumption would be helpful. Much of the remaining theory  -- such as the assumption that for each prompt at test time, there exists a corresponding "optimal reference policy", which seems unrealistic in practice -- are also derived from these assumptions.

### Algorithm
The algorithmic novelty seems limited. Essentially, the authors are suggesting to do sequential TTC, but only append responses to the current context whose reward exceeds a certain threshold (possibly including a "burn-in" stage, where the context is not changed until the hidden state reaches a certain length $m$).

### Experiments
It would improve the experimental results to include error bars. Results on more than just three datasets would be helpful to better evaluate RF-SBoN.

A major difference between parallel BoN and sequential TTC methods is that parallel BoN can compute all responses in parallel, hence is much more efficient in terms of latency. However, in the experiments, the authors only compare to parallel BoN in terms of $N$. I believe a comparison in terms of latency is crucial for a fair comparison of the methods.


[1] Yufeng Zhang, Fengzhuo Zhang, Zhuoran Yang, and Zhaoran Wang. What and How Does In-Context Learning Learn? Bayesian Model Averaging, Parameterization, and Generalization. arXiv preprint arXiv:2305.19420, 2023.

### Questions
- the presentation of responses in Appendix C.3 could be improved: the orange is not very readable; would be nice to put each prompt-response pair in a box; I assume the "Error: ..." in the solution of the last problem is part of the generated solution, but it would help to clarify this; since each algorithm solves a different prompt, this doesn't really provide any information on the direct comparison between the methods. Instead, solving the same prompt with different methods might be more insightful.
- in the notation, the difference between $\mathcal{X}$, $\mathcal{A}$, and $\mathcal{V}^*$ is not made explicit (in particular, I would assume they're all mathematically identical? Is the action space supposed to consist of *complete* responses, or only partial responses/tokens?) Furthermore, shouldn't the policy $\pi$ be conditioned on the prompt, i.e. $\pi(\cdot|x)$ instead of $\pi(\cdot)$?
- Assumption 3.1 seems unrealistic in practice. Could the authors elaborate on this? Is this assumption commonly made, or do they have any justification for it?
- the definition of the family of reference distributions (line 156) is not very clean. What is $\mathcal{T}_{\text{ref}}$, in particular, is it countable/finite? What's the prior distribution?
- the axis labels in the plots could be improved: I assume on the y axis (e.g. Figure 1), the authors plot pass@N accuracy, and on the x axis, they plot N (however, the x axis label says "pass@N")
- can the authors provide experimental results that compare the proposed algorithm to parallel BoN in terms of *latency* instead of $N$? This seems to be a much more fair comparison.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents Reward Filtered Sequential best of n (RF-SBON) for sequential test time compute. RF-SBON sequentially appends n answers within its context window,  answers with rewards exceeding a threshold  are only  appended in this procedure.  

Under the assumption that the pre-training data comes from a mixture of reference policies and that a single trajectory $(x_t, a_1,..,a_N)$ comes from the same mixture component $\pi^{\tau}_{ref}$, and that given an $x$ that reference component is distinguishable from other components, authors shows that for BoN  there exits a reward such that BON is suboptimal in terms of regret to an optimal policy $\pi^*$, while for sequential Best of n there are regimes where sequential BoN improves on BoN. 

Authors show empirically that sequential BoN improves on BoN in terms of performance on Math500, GPQA diamond, AIME 24 using models in the Qwen family and a PRM.

### Strengths
The paper studies an interesting question on sequential test time compute generalizing Huang et al results for Best of N. The paper makes a lot of assumptions and derive sensible results under these assumptions.

### Weaknesses
* I would like to challenge the main assumptions in the paper relating to the mixture assumption. 

Reading through the appendix, models are instructed as follows: 
"The previous solution(s) may contain errors. Before solving, briefly critique the previous attempt(s)
in 2 to 3 bullet points.Then provide a COMPLETE and CONCISE corrected solution
from scratch that addresses those issues. End with exactly one line containing the final answer"
is it really that the trajectory is singling out a single component of the model and giving it some flexibility to win on Best of n, or the self correction / reflection aspect that makes sequential BoN work. 
As an ablation running sequential BoN without this instruction would it lead to same improvements?

* Are any of your assumption reflecting this ability of the model to judge its own result? along the trajectory is the reward  empirically monotonically increasing ? 

* If the model is a base model and not an instruct or thinking model , would sequential BoN still work ? imagine the model is a bad judge, I believe that this will not allow the model to judge its own generation and improve upon. I believe an inherent assumption maybe implied by yours, should be on the ability of the model to judge and recover. 

* As the context length is increasing in n, the paper does not baseline as it advertises in the introduction : Is Best-of-N the best one can hope for under a fixed test-time budget?  are the comparisons to BON fair in terms of test time compute? can you give wall clock for SeqBON, BoN, filtered sequential bon and rewind and repeat ?

### Questions
* What  n (maximum of appended answers)  is used in the experiments ? 

* When you evaluate with Pass@N do you mean N for sequential BON and filtered sequential BON the maximum number of appended answers , or you mean N samples  from RF-SBON for a given n ?

* For different gamma used , how many filtered answers end up in the context of LLM? 

Minor suggestion: 

* Avoid using SBON in RF-SBON as SBON is known as Soft Best of N in many works, you can perhaps use RF-SeqBoN

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
In this work, the authors analyze existing Best-of-N approaches to inference time scaling and note that under certain data/LLM settings, these methods fail to achieve low regret. They then propose a modification to a sequential variant of BoN known as Sequential Best-of-N, and show under certain conditions it can achieve better regret than BoN. They then implement an algorithm via reward-filtering, where they only feed back good (high reward) completions to the LLM for the next sample in the classical sequential BoN scheme, to encourage exploration of near-optimal solutions.

### Strengths
* The argument that in sequential settings we can leverage additional information is interesting and compelling.
* The authors propose a realistic mixture of topics setting where BoN can fail to achieve low regret, and show that their proposed algorithm can achieve low regret.
* The empirical results demonstrate the efficacy of reward-filtered sequential BoN.

### Weaknesses
* SBoN is already terminology used to refer to Soft Best-of-N [1], and is already referenced as such in the community [2,3]. I would recommend using a different abbreviation and rename Algorithm 1 such as SeqBoN .
* Also, Algorithm 1 is very vague. What is "Update h based on $x,a_1,...a_{i-1}$"? Can you describe a specific function or algorithm? For instance, what if I update $h$ to be $x$? Then this is just BoN, but falls under your description of SBoN. I'm not trying to make you say explicitly "update h based on x and not just x" but I think a little more clarity would be helpful here.


Theoretical Results:
* I'm a little confused by Thm 4.3: It is unclear to me what about the sequential nature of the algorithm leads to a different result. In fact, the proof for this and Prop 4.2 look the same, and the only difference is now looking at this mixture of topics model. Then, the result reads to me as that under certain models, the sample complexity bounds are in fact better: In 4.2, $n \leq M_{LLM}$ means we have at least $\epsilon$ regret, and here by 4.3 we have that $M_{\tau} \leq M_{LLM}$ and therefore if $n \leq M_{\tau} \leq M_{LLM}$ we have $\epsilon$ regret which is strictly easier to avoid. 
* Following on this, isn't the point to say that BoN fails under a mixture of policies thus motivating a modified Sequential BoN strategy? Why do you want to prove these converse results for sequential policies?
* For Thm 5.1, if you let your SBoN algorithm be $h \leftarrow x$ (so just the BoN algorithm as I noted above) then you just get a tighter bound. Can you comment on this? Your framing of the result and your remark on it makes it seem like something about the sequential nature is unlocking this better result, yet this example implies otherwise. If you can build some intuition or explain it in the sketch I would appreciate that.

Overall, reading this paper gave me the feeling (at times) that there was a lot of theory just to say something intuitive: we can incorporate additional information in a smart way during sequential BoN to get better results. The deep theoretical treatment came at the cost of empirical results validating the method and exploring its potential due to page restrictions. Furthermore, many of the results especially in the beginning were just restatements of Huang et al., or minor modifications to incorporate the mixture of policies approach. I think these results could be distilled which would make the reading process better. I would love to hear more from you and am willing to improve my score.

Notation
* On line 791, what is A.1 and A.2? If they are section headers, how are you referencing A.2 before getting there? I think maybe you swapped these because you also in A.1 reference the introduced reward function as being introduced during the proof of theorem 4.3 (which is A.2)
* Line 170, you have a trailing bullet point you should try to get on the previous page if possible.

[1] Verdun, C. M., Oesterling, A., Lakkaraju, H., & Calmon, F. P. (2025). Soft Best-of-n Sampling for Model Alignment. ISIT 2025. 

[2] Geuter, J., Mroueh, Y., & Alvarez-Melis, D. (2025). Guided Speculative Inference for Efficient Test-Time Alignment of LLMs. 

[3] Aminian, G., Shenfeld, I., Asadi, A. R., Beirami, A., & Mroueh, Y. (2025). Best-of-n through the smoothing lens: Kl divergence and regret analysis.

### Questions
* You frame Huang et al.'s work as a sequential BoN algorithm on line 136 but I was under the impression that the authors proposed a parallel BoN algorithm. Can you clarify this?

### Soundness
3

### Presentation
2

### Contribution
3
