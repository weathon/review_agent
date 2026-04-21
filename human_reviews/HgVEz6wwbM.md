# What's the Magic Word? A Control Theory of LLM Prompting

- Avg Score: 4.60
- Decision: Reject
- Scores: 3, 5, 5, 5, 5

## Abstract
Prompt engineering is effective and important in the deployment of LLMs but is poorly understood mathematically. 
Here, we formalize prompt engineering as an optimal control problem on LLMs -- where the prompt is considered a control variable for modulating the output distribution of the LLM. 
Within this framework, we ask a simple question: given a sequence of tokens, does there always exist a prompt we can prepend that will steer the LLM toward accurately predicting the final token?
We call such an optimal prompt the magic word since prepending the prompt causes the LLM to output the correct answer. If magic words exist, can we find them? If so, what are their properties?
We offer analytic analysis on the controllability of a self-attention head where we prove a bound on controllability as a function of the singular values of its weight matrices.
We take inspiration from control theory to propose a metric called $k - \epsilon$ controllability to characterize LLM steerability. 
We compute the $k-\epsilon$ controllability of a panel of large language models, including Falcon-7b, Llama-7b, and Falcon-40b on 5000 WikiText causal language modeling tasks. 
Remarkably, we find that magic words of 10 tokens or less exist for over 97\% of WikiText instances surveyed for each model.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper suggests to view prompt engineering through the lens of control theory. The authors define a notion of probabilistically approximate controllability and verify study this property in various language models empircally. They also provide a sort of theoretical "converse", showing that if a certain condition is met, the language model cannot be controllable as defined here.

### Strengths
- I think the papers main selling point is to offer the perspective to analyze prompt engineering through a control theory lens. Unfortunately, it not novel and has been previously suggested in [1].

- Nevertheless, the idea to use control theory---and notions of controllability/reachability in particular---to analyze LLMs is quite elegant. 

- The paper asks a number of interesting questions in section 7. 

[1] Soatto, Stefano, et al. "Taming AI Bots: Controllability of Neural States in Large Language Models." arXiv preprint arXiv:2305.18449 (2023).

### Weaknesses
- It is not clear that the analysis in section 4 is of any relevance as it stands for two reasons:
 1)  I am not sure it makes sense to give a "topological/norm/metric" controllability condition as in (4). Is there a natural topology here to justify this? As far as I understand tokenization imposes a more or less arbitrary choice of topology so it is not clear to me that characterizing controllability (an algebraic concept) by a topological one makes any sense. 
2) Moreover, I cannot find any suggestion in the paper that the bound is of any practical use---here the obvious questions is: is it sharp and if not how loose?

- Control theory is fundamentally a study of dynamical systems. The statement that " For simplicity, the time set T and the transition map ϕ are omitted from this
definition" made in section 3 then suggests that you are actually no longer really in the realm of control---which was your main selling point to begin with.


- The experiments are somewhat sparringly commented and the reader is left wondering how these were conducted. What is the precise definition of a "solved instance" and how were these instances generated? My critique pertains to section 6 mainly but the appendix also suffers from uncommented plots.

- While satisfactory, the level of writing could be better. There are a number of grammatical errors, but beyond that and more importantly, the paper does not feel very well structured. I think this is due in part to a lack of clear delineation of what the paper's contribution is. 

- It would be better if the analysis was stated in standard thm/proof style. Currently, commentary is interwoven with the proof of claim (4) making the derivation unnecessarily obtuse. Ideally, the proof should be prefaced by exactly that which is to be shown and then broken down into an overview of the relevant components, followed by the proof itself. Let me finally say that I find this particularly strange since the paper has a number of definitions (potentially too many...) stated in the standard mathematical style.

### Questions
- Is the norm in (4) the Euclidean norm? 



See also questions I raised in weaknesses above.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work presents a control perspective of the LLM steerability by introducing the concept of $k-\epsilon$ controllability. They also proved a bound on the controllability of self-attention in terms of the singular values of its weight matrices. In addition, several experimental studies have performed to compute the $k-\epsilon$ controllability of LLMs (Falcon-7b, Llama-7b, Falcon-40b). The authors conclude that LLMs are very controllable and the control prompts of 10 tokens or less are sufficiently enough to ensure the LLM output the target token.

### Strengths
The idea of introducing $k-\epsilon$ controllability is interesting and it is a nice connection between LLM and control theory. The theoretical bound presented in Section 4, though hard to check in reality, is an initial step towards understanding the steerability of LLM theoretically. In addition, I do appreciate the authors provide an interesting discussion on the open problems of LLM in control.

### Weaknesses
Section 4 only considers a self-attention head, which is quite simple and limited (compared to the current model used in LLM). What are the difficulties in generalizing such results to a more complex model?

The presentation of Section 4 can be further improved. The relationship between state controllability (Definition 7) and $k-\epsilon$ controllability (Definition 6) should be discussed, i.e., implications of your theory result in Section 4. In addition, I am confused about some notations: are $u_i$, $x_i$ the embeddings of the tokens? Previously the u and x are presented as tokens, it does not make sense to make $\|u_i\|\le 1$ and $\|x_i\| \le 1$ if they are tokens. In addition, is this assumption valid in real LLMs? 


Although introducing the controllability of LLM from a control perspective is interesting, the experimental results of checking the controllability of the LLMs are not very exciting given the existing results from previous work [Zou 2023]. The experiment setup is almost identical to GCG work and the obtained results are also within expectation. Instead, proposing a new method to study the controllability of black-box LLMs will be more interesting.

### Questions
1. What are the connections between state controllability (Definition 7) and $k-\epsilon$ controllability (Definition 6)? Does the former imply the later?
2. Are the assumptions $\| u_i\| \le 1$ and $\| x_i \| \le 1$ realistic? If not, is your results in Section 4 still hold?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents a control theoretic perspective on steering LLMs via prompting. After formalizing the problem and providing a theoretical analysis of the conditions when self-attention is able unable to be steered, the paper provides an empirical investigation of the steerability of LLMs in practice on Wikitext.

### Strengths
- The paper presents a compelling perspective, which could be fruitful.
- Despite being a little bit disorganized, I appreciate the basic idea behind the theoretical analysis, showing that there is some fundamental bottleneck that makes LLMs not arbitrarily steerable (under some assumptions).

### Weaknesses
- I have concerns on the significance of the empirical results. In particular, I suspect LLMs might be very easily steerable, and that any limitation in the ability to push them to a particular output is just due to limitations in the optimization method that is used to find the prompt. As a noticeable example, one might say that, for the definition of steering that has been employed in the paper, a sufficiently capable LLM can always be steered by prepending to it an prefix that reads like "after reading n words/token, output this particular word/token". If the optimization procedure does not find such a solution, it seems more a limitation of that than a fact related to the inability of LLMs to be steered.
- The paper is at times not well-organized. The discussion section, that is usually a summary of the takeaways from the paper, looks more like a discussion of related work, and also has some incomplete points, the theoretical results would be better understood inside of a theorem latex environment, and so on.

### Questions
- How is your definition / empirical analysis entangled with the prompt optimization technique that is being used?
- Can you tidy up some parts of the paper to make them clearer to a machine learning audience?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies prompt engineering, a key factor in exploiting LLM, from a point of view in control theory. The main contribution is that:
1) Formulate the prompt engineering problem as an optimal control problem (e.g., defining the k-epsilon controllability). This provides a mathematical framework to study this problem rigorously.
2) Prove that if a certain eigenvalue bound about the weight matrices is satisfied, a single attention head is state controllable. That is, for any input, there exists a prompt, i.e., the "magic word", that can force the LLM to give the desired output.
3) Proposed two prompt searching algorithms to validate the controllability of Falcon-7b, Llama-7b, and Falcon-40b. Experiment results show that there generally exists a magic word of length 10 or less for over 97% of Wikitext dataset.

### Strengths
The primary novelty of this paper lies in its formulation of the prompt engineering problem as an optimal control problem.  I truly appreciate this idea. It is widely acknowledged that the choice of prompts heavily influences the LLM performance. The interplay between the LLM's weights and the input prompt jointly determines the "states" of the LLM. This insight is intuitive but non-trivial, very different from conventional supervised learning approaches. It is commendable that the authors have translated this observation into a mathematical framework and provided an initial analysis.

### Weaknesses
Weakness or Questions

1. I have a concern regarding the controllability metric. When an LLM is controllable, there exists a prompt capable of compelling the LLM to produce a desired output, even if that output is factually incorrect. Thus, this formulation does not seem to establish a real connection with the specific capabilities of LLM, such as reasoning and knowledge memorization. Moreover, The attribute of controllability appears to lean more towards a negative property, signifying the LLM's susceptibility to prompt manipulation and potential vulnerabilities.

2. The theory only requires the weight matrices to meet a particular bound on their largest eigenvalues. The analysis framework does not depend on any training data or training algorithms. Even randomly generated weights (as long as they satisfy the specific bound) can render the LLM entirely controllable. This raises concerns about whether this theoretical framework can effectively explain the underlying mechanism of a well-trained LLM.

3. The results do not seem to offer practical guidance for prompt design. The analysis does not appear to provide insights into how to stimulate the capabilities of an LLM effectively. The proposed methods both require access to the desired output (the ground truth) during the search process.

### Questions
My concerns and questions are merged into the "weakness" section. Please refer to the last section.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper uses control theoretical concepts to analyze the existence of specific prompts (magic words) that allow for controllable text generation. The authors do a thorough review of the state-of-the-art literature on the topic, and introduce the necessary control theoretical concepts. Building on these concepts, the authors introduce the definition of k-\epsilon controllability, which is a measure of the existence of a “magic word” (k token prompt) that would lead to the desired output token. The authors present a theoretical result to bound the k-epsilon controllability of the attention head, and two heuristic algorithms that allow to compute magic words. They provide simulation results that show that in 97% of the cases surveyed, “magic words” exist.

### Strengths
This paper is one of the first of its kind in applying a control theoretical concepts to the study of LLMs. Control theory offers very powerful tools that have the potential of providing a more formal understanding of LLMs behavior. For this reason, the approach introduced in this paper has great potential to advance our understanding of language technologies. This reviewer positively values the originality of this paper. Moreover, the topic addressed of controlling language generation with the appropriate prompt is very relevant, and having rigorous tools opens the door to a formal treatment of LLMs. In this paper, they provide an algorithm to find “magic words” based on a mathematical result inspired by control theory concepts. More importantly, they list a series of open questions that could be addressed from a control-theoretical point of view.

### Weaknesses
This paper has some issues with the formalization of the theoretical concepts, as well as with the presentation of the results. Since this paper introduces for the first time control theoretical concepts in the light of LLMs, it is paramount that the definitions are accurate and properly capture the ideas underpinning dynamical systems. This is not the case for this paper. The control theoretical concepts in this paper are not properly communicated, and Definition 4 has several flaws. The definition given corresponds to an input-output system, as opposed to a dynamical system: in order for it to be a dynamical system, the state space should be V as opposed to V^*. Moreover, the definition given for k-\epsilon controllability is very far the definition of controllability in dynamical systems. Please refer to Feedback Control Systems (Amstrom and Murray, 2009) for details. This reviewer is concerned that, if published in the current form, this paper can introduce more confusion than clarifications in the realm of using control theoretical tools for LLMs analysis. Aside from this, the main result should be framed as a theorem and provided in the body of the paper, together with the algorithms, not in the appendices, as these are the main results of the paper. Furthermore, the paper currently lacks necessary formal definitions (such as the definition of the V^* set), and the control theoretical section lacks clarity in the exposition. Substantial modifications are needed to improve the rigor of the presentation in Sections 3 and 4.

### Questions
This reviewer would like to know why the state space is defined as V^* as opposed to V. Defining it in such manner strips away the dynamics of the system (progression with time), and reduces it to an input/output system. This reviewer would also like to know how map \phi is defined in the context of LLMs, since only h was defined. Since the state space is defined as V^*, the map \phi cannot properly be defined as the dynamics of the system. Moreover, the question under study is more a question of system excitation under an initial condition than a control input. In the context of control theory, an input is very rarely seen as a one-time excitation, but rather as a feedback signal. The setup described in this paper corresponds to an input/output behavior under an initial condition, and as a result, the definition for k-\epsilon controllability is very different from the standard controllability definition used in control theory. Moreover, this reviewer suggest the use of “reachability”, as opposed to “controllability”, as it more accurately captures the problem under consideration.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
