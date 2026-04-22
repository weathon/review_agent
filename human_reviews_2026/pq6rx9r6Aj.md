# Jailbreaking Jailbreaks: A Proactive Defense for LLMs

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
The proliferation of powerful large language models (LLMs) has necessitated robust safety alignment, yet these models remain vulnerable to evolving adversarial attacks, including multi-turn jailbreaks that iteratively search for successful queries. Current defenses, primarily reactive and static, often fail to counter these search-based attacks. In this paper, we introduce ProAct, a novel proactive defense framework designed to disrupt and mislead autonomous jailbreaking processes. Our core idea is to intentionally provide adversaries with "spurious responses" that appear to be results of successful jailbreak attacks but contain no actual harmful content. These misleading responses provide false signals to the attacker's internal optimization loop, causing the adversarial search to terminate prematurely and effectively jailbreaking the jailbreak.
By conducting extensive experiments across state-of-the-art LLMs, jailbreaking frameworks, and safety benchmarks, our method consistently and significantly reduces attack success rates by up to 92\%. When combined with other defense frameworks, it further reduces the success rate of the latest attack strategies to 0\%. ProAct represents an orthogonal defense strategy that can serve as an additional guardrail to enhance LLM safety against the most effective jailbreaking attacks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes PROACT, a proactive defense that “jailbreaks the jailbreak” by returning spurious, non-harmful responses that look like successful jailbreak outputs to an attacker’s evaluator, thereby misleading and prematurely terminating the adversarial search. PROACT is instantiated as a three-agent pipeline: a User Intent Analyzer routes only malicious queries into the defense, a Proactive Defender generates topic-consistent but benign “encoded” outputs that appear harmful, and a Surrogate Evaluator iteratively critiques and refines these spurious responses until they pass a jailbreak evaluator. Across different target LLMs and attack frameworks, PROACT significantly lowers while preserving instruction-following utility.

### Strengths
1. The idea of "Jailbreaking jailbreaks" is very interesting and novel. 

2, The paper is well-written and easy to follow

3. The empirical result of the PROACT method seems very good, and it can work with other defense methods together.

### Weaknesses
1, 
The Achilles heel of the PROACT system might be its User Intent Analyser, as it is basically acting as an LLM judge that detects the harmful content in the user input. One ICML 2025 paper (https://icml.cc/virtual/2025/poster/45356) specifically talks about how the LLM detection can be bypassed with their attack. The paper might need to address the usage of the LLM as a User Intent Analyser further. 

2. 
Some work, such as the persuasion attack (https://doi.org/10.18653/v1/2024.acl-long.773), also use a LLM judge to score the response's harmfulness and I am curious if that will work against the surrogate evaluator. 

3, 
Some jailbreak attacks do not involve iterative optimizations. For example, long-context jailbreaks (https://arxiv.org/pdf/2402.16717) or many-shot jailbreaking (https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea456e232efb72d261715e33ce25f208-Abstract-Conference.html). Is PROACT defenseless against these types of attacks?

4. 
It seems like there are many back and forth between LLMs within the PROACT system. It raises the concern of increasing latency and resources of computing.

### Questions
1, How will the PROACT work against the jailbreak attack that was designed to bypass LLM detection (https://icml.cc/virtual/2025/poster/45356), or does not need iterative optimization (https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea456e232efb72d261715e33ce25f208-Abstract-Conference.html)?


2, What is the extra latency of inferencing with PROACT?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a jailbreak defense method called ProAct against multi-turn seach-based attacks. The defense method first identifies jailbreak attempts and returns perturbed responses when spotting mallious inputs.

### Strengths
1. ProAct seems to be effective against PAIR and TAP, reducing ASR to nearly zero. On average, it reduces attack sucess rate for more than 50%, which is notable.
2. The authors conducted several ablations and experiments across different models to validate the method's effectiveness.

### Weaknesses
1. My major concern is about the User Intent Analyzer. Why is it necessary to return nonsense strings to the attackers when we can actually spot them? Would be much more easier and safer to just terminate the conversation or connection as it is done in most commercial chat websites like ChatGPT or Claude. In summary, I simply do not understand why such defense is needed when we can actually identify the attackers. For me, identifying the attackers is the most vital part of the defense.

2. The perturbation algorithms used by ProAct are trivial string operations, which is not adaptive to specific attackers. This might explain why the ASR is still above 50% for X-teaming.

### Questions
Is ProAct effective when we cannot identify the attackers?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces ProAct, a proactive defense framework that thwarts adversarial jailbreaks by generating deceptive responses that mislead attackers without producing harmful content. Extensive experiments show that ProAct reduces attack success rates by up to 92%, and to 0% when combined with other defenses, offering a powerful complement to existing LLM safety measures.

### Strengths
Innovative idea of generating spurious responses to actively mislead jailbreak attackers.

Good writting structure and easy to follow.

### Weaknesses
The defense's effectiveness relies on the attacker's evaluation mechanism `User Intent Analyzer`, which could be imperfect.

### Questions
null

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
2

### Summary
This paper proposes a jailbreak defense method called ProAct. The proposed method use a three-agent system that first identifies the users' intention, and then craft a spurious response (which serves as the proactive defender). This response is refined iteratively in order to deceive the  surrogate evaluator. The experimental results in Table 1 shows significant improvement on the models' robustness against jailbreaking attacks.

### Strengths
1. (Clarity) This paper raises 5 RQs and provide detailed discussion regarding each of them, which improves the readability of this paper.
2. (Significance) The experimental results in Table 1 shows significant improvement on the models' robustness against jailbreaking attacks.

### Weaknesses
See the quetions part.

### Questions
To what extent does the method presented in this study affect the efficiency of LLM serving? Is it possible to provide a quantitative assessment of the extra token consumption introduced by this method, and how does it perform in comparison with alternative methods?

### Soundness
3

### Presentation
3

### Contribution
3
