# What Can an LLM Flip if It Fails to Flip Coins?

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4

## Abstract
Large language models (LLMs) can often accurately describe probability distributions using natural language, yet they still struggle to generate faithful samples from them. This mismatch limits their use in tasks requiring reliable stochasticity, such as Monte Carlo methods, agent-based simulations, and randomized decision-making. We investigate this gap between knowledge and sampling in the context of Bernoulli distributions. We introduce Verbalized Rejection Sampling (VRS), a natural-language adaptation of classical rejection sampling that prompts the LLM to reason about and accept or reject proposed samples. Despite relying on the same Bernoulli mechanism internally, VRS substantially reduces sampling bias across models. We provide theoretical analysis showing that, under mild assumptions, VRS improves over direct sampling, with gains attributable to both the algorithm and prompt design. More broadly, our results show how classical probabilistic tools can be verbalized and embedded into LLM workflows to improve reliability, without requiring access to model internals or heavy prompt engineering.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a verbalized rejection sampling method in which LLMs are prompted to generate random samples from a target distribution. This method contrasts with verbalized direct sampling, where LLMs are directly prompted to produce random samples from the target distribution. In verbalized rejection sampling, the LLM is prompted to propose a sample from a simpler distribution Q, and then to decide whether to accept or reject the proposal when both the proposal distribution Q and the target distribution P are provided in context. Importantly, the entire process is carried out within a single prompt: there is no explicit second stage of rejection, and the rejection sampling procedure is completed in one step. The authors demonstrate that verbalized rejection sampling mitigates some of the biases previously observed in verbalized direct sampling. They also provide theoretical arguments for why this method performs better, highlighting that explicitly prompting LLMs to act as rejection samplers is the key contributing factor.

### Strengths
The proposed verbalized rejection sampling method is interesting. Investigating the ability of LLMs to generate random samples is an important topic, and this paper makes a useful contribution to that line of research.

### Weaknesses
The contribution of the paper is rather limited. Verbalized rejection sampling is essentially a form of prompt engineering, where modifying the prompts allows the LLM to achieve better performance in random sample generation. The theoretical analysis suggests that improved performance arises because the LLM is explicitly prompted to consider acceptance and rejection in the context of both the proposal distribution Q and the target distribution P. An important control experiment, in my view, would be to prompt LLMs to sample directly from P (as in verbalized direct sampling), while also asking them to consider whether to accept or reject the sample --- but without reference to Q. This condition could be regarded as verbalized rejection sampling without Q, or alternatively, as verbalized direct sampling augmented with a superficial requirement to evaluate whether the sample should be retained. Including such an intermediate condition would potentially strengthen the paper’s overall contribution.

### Questions
Do the authors also provide a theoretical analysis of verbalized direct sampling? In particular, what explains the systematic biases observed in verbalized direct sampling across different LLMs?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies LLMs' ability to faithfully simulate a probability distribution, with a focus on the Bernoulli distribution. It is shown that while an LLM can identify the correct Bernoulli distribution from data, it fails to faithfully generate samples from a Bernoulli distribution. To reduce the sampling bias, the paper proposes Verbalized Rejection Sampling (VRS) which prompts an LLM to perform rejection sampling in language. Experiments show that the method leads to significant reduction in the LLM's sampling bias. The paper also provides some theory to prove that VRS outperforms direct sampling under certain conditions.

### Strengths
1. The problem of whether or not an LLM can faithfully simulate a distribution is interesting.
2. The paper gives both empirical and theoretical evidence to demonstrate the advantage of VRS over direct sampling.

### Weaknesses
1. My main concern is the limited scope of the paper. The empirical and theoretical investigations are almost exclusively focused on the Bernoulli distribution (with some additional results for the binomial distribution in the appendix). The leap from the single-parameter binary Bernoulli distribution to multivariate, continuous, or complex categorical distributions is substantial, and the paper does not provide empirical or theoretical evidence that VRS would be effective in these more complex settings. This renders the paper a specific solution to a very narrow problem. For the sake of sampling from a Bernoulli or binomial distribution, why not just use a standard software?
2. Moreover, as is shown in Section 6.1, the effectiveness of VRS can be heavily impacted by the prompt. In particular, a prompt that instructs the LLM to compute $M$ correctly can even degrade its sampling performance. This instability, combined with the lack of theoretical guarantees, makes VRS feel more like an ad hoc method rather than a robust and general approach.

### Questions
1. The paper positions itself as a first step towards studying LLM's probabilistic reasoning skills, and is motivated by practical applications such as agent-based modeling and randomized decision-making. Do the authors think that the insights in this paper (e.g., the knowledge-sampling gap) will carry over to those more practical scenarios? I feel that the nature of stochasticity in this paper's setting and in the more practical settings is quite different. The experiments in this paper are conducted in an isolated "lab" setting, where an LLM is explicitly prompted to generate a sample from a given distribution. But in the more practical settings such as agent-based modeling and randomized decision-making, stochasticity is often implicit (instead of explicitly specified) and embedded in a much larger reasoning chain.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies if LLMs can generate faithful random samples from Bernoulli distributions described in natural language. The authors identify a knowledge–sampling gap: models can accurately describe probabilities but fail to sample according to them, often showing bias (typically toward “1”). They propose a language-based adaptation of classical rejection sampling, making  the model verbally reasons about whether to accept or reject a proposed sample. Empirical results across 4 LLMs show rejection substantially reduces sampling bias. Theoretical analysis supports that the improvement arises from the rejection sampling mechanism itself, not just prompt effect

### Strengths
The paper tries to look into a fundamental problem and is not incremental though feels like an early stage work.
Comprehensive experiments across multiple models, phrasings, and CoT settings; results consistently demonstrate bias reduction.

### Weaknesses
Prior work has indicated that LLM agents despite understanding the notion of probabilities struggle with probability sampling (the paper also cite them) "Do LLMs play dice?" Further more developed by works like a "theory of response sampling" which explains distributional shift in sampling/ deviation from distributional statistics based on some kind of value maximization. I think these together might explain the deviation towards 1 in the experiment. 

Also the knowing doing gap of LLM, as knowing a concept and unable to use this in sampling/action is not really new. The 'knowing doing gap' has been discussed in many works, seminal work as "LLMs are Greedy Agents: Effects of RL Fine-tuning on Decision-Making Abilities". I dont see this a new finding though the finding here gives a nicve spin to it. 

The paper to my understanding argues the algorithmic process of probabilistically filtering of samples mitigates the LLM's inherent sampling bias. So as a verifier it works but give consistant bias in sampling. I think the paper is missing prior work that explains similar observation across similar task lile exploration where verbalized verifiers improve perfromance.

Also a minor concern on why the study limits to one distribution? and do the authors think this will scale? Can you try 0,1 and 1,0 in prompt to reduce order bias or use random tokens to amke sure the encoded value of 0 and 1 does not come into play?

I think the attempt to understand the underlying sampling pheneominon can be a good contribution thought the findings are a bit early stage but the paper fails to place itself and build on existing framework to explain the observation. Also for wider audiance may be say why using tool to fix the sampling problem is not good enough. I am ready to review my score and accept post rebuttal post rebuttal if the discussion help clarify my understanding of how the authors explain the observation in light of prior work. Thanks you.

### Questions
Please see weakness

### Soundness
2

### Presentation
3

### Contribution
2
