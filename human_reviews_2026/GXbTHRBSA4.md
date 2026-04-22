# ACTS: Adaptive Control for Test-time Scaling

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 4, 8

## Abstract
Controlling the generation length of Large Language Models (LLMs) presents a difficult trade-off between computational cost and output quality. We tackle this challenge with the ACTS (Adaptive Control for Test-time Scaling) framework, which leverages a novel ``termination signal'', the rich probabilities a model assigns to control tokens like End-of-Sequence. By framing generation as an optimal stopping problem, ACTS uses this signal to dynamically decide when to terminate. Experiments demonstrate that our policies significantly outperform baselines, reducing token usage on conversational tasks by 35\% while boosting mathematical reasoning accuracy by 13.3\% on AIME and 9.8\% on Math 500 through more efficient thinking, which is achieved simply by changing the token-sampling process. ACTS thus enables a new class of signal-driven, principled control over LLM generation, paving the way for more efficient and adaptable model inference.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Adaptive Control Token Sampling (ACTS), a framework that dynamically regulates the reasoning length of LLMs at test time using the sub-argmax probabilities of control tokens (e.g., EOS, EOT). The authors frame generation as an optimal stopping problem and design several policies to balance “underthinking” (premature termination) and “overthinking” (unnecessary reasoning). Experiments across reasoning and instruction-following (AlpacaEval) benchmarks show small gains in accuracy or efficiency.

### Strengths
- Using control-token probabilities for inference-time control is original and interesting, potentially offering a lightweight alternative to external reward models or verifier signals.

- The paper correctly situates itself within the growing literature on test-time scaling, referencing S1, Self-Consistency, and Speculative Rejection works.

### Weaknesses
- The submission appears rushed and lacks careful proofreading. Algorithm 2 and Figure 5 overflow the page boundaries; Figures 6 & 7 have inconsistent spacing; some sub-figure captions (e.g., “accumulated,” “last-interval”) are unclear; Table 2 includes ambiguous terms such as “unconditional forking.” Overall readability and figure organization require substantial revision.

- Improvements in Table 1–2 are small (≈ 2–3 % absolute at best) and sometimes trade off token efficiency inconsistently. It is unclear whether these changes are statistically meaningful or justify a new inference-control paradigm.

- The paper claims ACTS mitigates overthinking and underthinking, yet provides no quantitative trajectory analysis or diagnostic evidence (e.g., token-level reasoning-trace inspection, error typology, or heuristic measures of thought quality). Without such evidence, the claimed cognitive interpretation remains speculative.

- The spike thresholds and critique-trigger parameters appear hand-tuned, but no ablation or development experiment explains their choice or sensitivity.

- Operating on the “stopping probability” is a modest extension of known heuristic stopping rules. That said, the conceptual leap from S1’s budget forcing or early-termination heuristics is incremental.

### Questions
See weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces Adaptive Control Token Sampling (ACTS) that effectively mitigates the negative effects of "underthinking" and "overthinking" through a fine-grained control over special tokens.

### Strengths
The paper proposes several effective methods for mitigating overthinking and underthinking. Experiments do show their effectiveness over benchmarks where these phenomena are observed.

### Weaknesses
1. I must put into question the contribution of the paper, as it appears to me overclaimed. The main idea is three-fold: when underthinking is a problem, make the model think longer by saying "wait"; when overthinking is a problem, cut the model short once it has had several opportunities to end its thinking; self-critique, which is a rather self-contained method. The rest of the paper deals with engineering these approaches. However, there is a lack of central theme around these methods. It feels as if the authors tried these methods out, saw some performance improvements, and attempted to put them under the same framework, while they really should be treated as separate engineering tricks and studied individually. The most interesting method by far is the self-critique framework, with an intuitive explanation of effecient tree search for LLM problem solving, but it is only studied superficially through performance gains, without any attempts and theoretical analysis or a closer look into its mechanisms. 

2. The mitigation of underthinking and overthinking are restricted to reasoning and instruction following tasks, respectively. This is a problem here because the methods appear to target these two problems separately and only within their respective applications, putting into question the generalizability of the proposed methods. 

3. The general presentation of the method was not intuitive to me. I think the paper tried to go from a general framework to a more specific implementation, where Section 3 establishes the ACTS framework, and Section 4 specfies it into proposed methods. This overcomplicated the narrative because Section 3 taken out of context is rather confusing, for example "forcing the emission of the appropriate control token" is too general on Line 151. 

4. Section 4 has numerous writing problems that also overcomplicates things. First, the titled paragraphs are not individual policies, but a progression of different scattered ideas. Among the actual methods used in Section 6.2 / 6.3, Accumulated / Last-interval / N-Spike, only one is explained in the main text. Next, many concepts are mentioned but not discussed adequately, such as $N_{patience}$ mentioned without definition on Line 178 (minor issue), the unspecified directive for critic self-evalution through Lines 185-192, an unfinished sentence on Line 203 (also I would caution comparing LLM thinking procedure to humans unless it's highly relevant), unspecified "opportune" moment on Line 205, etc. Algorithm 1 is also not really needed in main text as a straightforward textual explanation makes it clear enough for me.

### Questions
Q1. Most importantly, I think the narrative of the paper needs to be revised. Targeting my weakness 1 and 2, can you clarify what main methods are used in the paper and how they fall under the same framework? Just to be clear I'm not asking for a simple restatement of contributions, but a more structured discussion of the methods' relations, synergies, etc. 

Q2. Can you also clarify what the requirements are before applying your methods? Specifically, beyond the tested benchmarks, do we need to know if the model is underthinking or overthinking as a priori, and what other metrics need to be assessed if any? 

Q3. For Section 6.1, are all the plots obtained with a single prompt? For Figure 2, did you do a cutoff at 0.0001, if so why did you choose this cutoff and what's the largest observed value, and if not why does the probability reach this value and not beyond so frequently? For Figure 3, can you explain why the spikes become much sparser after a while? Did the model rollout collapse in a way so as to never stop thinking? 

Q4. What are "Accumulated" and "Last-Interval" methods in Figures 5 and 6? May be related to Q1. 

Q5. For Figure 6, since we're dealing with underthinking, I assume everytime the model terminates prematurely, a "Wait" token is appended to keep it thinking. In this context what does N-spike refer to? Why is there a distinction between different amount of "Wait" tokens? Or is it "whichever comes first"? 

Q6. For Section 7, the main table sees accuracy increase over baseline as well as token count increase. In fact, the Avg. Token Count column is marked wrongly in terms of best performance. Shouldn't this be an application of mitigating underthinking instead of overthinking, so both accuracy and token count increases over baseline?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
Current language models often operate in "thinking" mode, allocating a fixed budget to reasoning. However, models may overthink or underthink in these settings; there is a need for calibration. This paper introduces an inference time framework - ACTS - which leverages EOT probability spikes to mitigate underthinking in complex reasoning scenarios. Across a variety of reasoning tasks and models, the proposed method improves over naive decoding strategies, pushing the pareto frontier of the best performance with a reduced token budget.

### Strengths
* The proposed approach is training-free, making it lightweight and accessible (to models that provide token probabilties)
* The problem is clearly motivated and timely
* Analysis is done that justifies the method (ie, token probabilties of eos)
* Section 4.1 is well-justified and seems to cover the most likely scenarios encountered during decoding.
* ACTS explicitly frames reasoning control as an optimal stopping problem; this provides a more principled lens for test-time scaling. This formulation unifies stopping, critique, and branching decisions under one controller.
* ACTS improves accuracy while reducing token usage on reasoning and instruction-following tasks.

### Weaknesses
* Though the formulation is interesting, it is unclear how it will generalize to models outside the ones tested; reasoning is only tested on two families of model, and instruction-following on one. Since this is a lightweight, inference-time method, it can be more easily verified by running on more models.
* This space is saturated and it is unclear how significantly this improves over existing work like S1, and other early-stopping methods (including ones that are inference-time only as well). In addition to further explaining this novelty, it would be helpful to show empirically it works well against more baselines, instead of against fixed decoding strategies (like greedy, or inserting wait x times).
* There are several formatting errors which make the paper slightly harder to read. For example, Figure 5 extends beyond the page boundary, and also impacts the next page by forcing a single column. Same thing for Algorithm 2.
* The paper shows spikes correlate with reasoning boundaries but doesn’t explore more in-depth here. It would be interesting to know why these spikes emerge or whether they consistently indicate correctness vs. hesitation.

### Questions
* Can ACTS improve performance on other benchmarks besides reasoning/instruction-following, or is this a specialized phenomenon in these domains?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposed ACTS (Adaptive Control for Test-time Scaling), which measure the spike of EOS and EOT token to detect models intention of termination. The proposed method treats generation as a control process, which monitors the control-token probabilities and decides whether to continue reasoning, critique itself, or terminate at each decoding step. The strong empirical results demonstrate that ACTS can effectively reduce the average length of generating tokens during decoding, meantime achieving comparable or better results, and can effectively help solve the underthinking and overthinking issue of recent reasoning models.

### Strengths
1. The idea of monitoring the spike of the signal tokens (EOS and EOT) to determine models intention to terminate is novel and interesting. This also well aligns with recent work that measure token entropy or probability for tracking major transition during LM reasoning process. [1]
2. The empirical results are strong, showing significant improvement on MATH500 and AIME dataset, meantime reducing the average token length to reduce computing.
3. The method is simple and easy to implement, can easily be applied on all kinds of reasoning tasks.
4. The paper is well written and easy to follow.

[1] Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning

### Weaknesses
1. Minor issue: The threshold choices for spikes and critique triggers are empirically tuned. Unable to automatically determine the threshold can bring difficulty to implementation when trying to apply it across tasks.

### Questions
None

### Soundness
4

### Presentation
4

### Contribution
4
