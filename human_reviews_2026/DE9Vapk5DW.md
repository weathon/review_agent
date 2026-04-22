# Automating Thought of Search: A Journey Towards Soundness and Completeness

- Avg Score: 3.60
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 4, 4

## Abstract
Large language models (LLMs) are being used to solve planning problems that require search. Most of the literature uses LLMs as world models to define the search space, forgoing soundness for the sake of flexibility. A recent work, Thought of Search (ToS), proposed defining the search space with code, having LLMs produce that code. ToS requires a _human in the loop_, collaboratively producing a sound successor function and goal test. The result, however, is worth the effort: all the tested datasets were solved with 100% accuracy. 
Consequently, there is great potential to automate the ToS process.
We take a first major step towards automating ToS (AutoToS), taking the _human out of the loop_ of interactions with the language model. AutoToS guides the language model step by step towards the generation of sound and complete search components, through feedback from both generic and domain specific unit tests. 
We show that AutoToS is able to achieve 100% accuracy on all the evaluated domains with a small number of LLM calls.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The work builds on previous method Thought of Search. In ToS, the LLM is prompted for code on goal-testing function an a successor-generation function. These funkction can be used via search methods BFS or DFS, etc.  While ToS achieves 100% accuracy, in previous work,  it needs a  human expert in the loops for evaluation. This is applied to planning, games etc where a tree of alternatives can be searched / explored to find a solution. The method is straight forward to implement.

### Strengths
1. It takes out the human in the loop for evaluation which makes it more usefull for practical application.
2. It allows for the use of smaller models since it splits to planning proplen into steps.
3. It is straigh forward to implement to replicated. 
4. Good evaluation with a number of models and good number of problems.

### Weaknesses
1. It is expensive as each step needs a LLM call and this comes also with some latency. 
2.  The paper claims 100% accuracy on tested datasets. This is great but comes with the downside that it does not allow access to the limits too. Benchmarks marks would be preferable that provide some space. Especially, blocksworld that would be possible by using more blocks or needs more plannings steps to solve. 
3. The paper runs a best-of-5 as they have a validator so this seems to make it more expenstive but it can run in parallel, i belive. 
4. Dependency on Test-Case Quality as knowledged by the authors ... also a limit for humans, we miss things too.

### Questions
Could you easily run this on task which are harder and to challenge this method and find problem cases? Where does it fail? One weak point was the test-case quality but if you make a problem really hard how far can you get? What problems occure and what do you think how to fix?

### Soundness
4

### Presentation
4

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
The paper is an extension of prior work, Thought of Search (ToS), in which an LLM is used to generate Python code for a successor function and a goal test function which are then used with standard search algorithms like BFS or DFS to solve the given planning problems. While ToS required human in the loop to iteratively generate a sound successor function and a goal test function using LLMs, the paper aims to automate this step by guiding LLMs to generate a sound and complete successor function and a correct goal test function by using generic and domain specific unit tests for these functions as feedback to LLMs.

### Strengths
- The paper is easy to follow, and the contributions of the paper in the context of existing works are clearly communicated.
- The paper presents an important step towards automation of ToS enabling automated planning without requiring humans in the loop.
- The paper’s focus on the necessity of soundness and completeness in the context of planning (which are often overshadowed in works that tend to use LLMs directly as planners) is well-appreciated, as they are key to achieving good and reliable planning performance.
- The paper presents comprehensive experimental evaluation in different planning problems with varying complexity.

### Weaknesses
- The paper is incremental as it takes one specific step of the prior approach, and automates that step with an LLM-in-the-loop mechanism replacing the human-in-the-loop mechanism of prior approach. While this is an important extension, this limits the novelty and impacts of the paper to the broader research community.
- The claim about 100% accuracy feels underwhelming since it is based on success in any of the 5 trials (using an external validator). It would be interesting to see what average success rates look like for these experiments to get a better idea of the effectiveness of the proposed approach.

### Questions
- In line with my comment above about best-of-five trials when reporting accuracy, wouldn’t it be more meaningful to report average accuracy, since it is a more accurate reflection of the system’s performance when deployed in realistic settings without humans in the loop, as the paper aims to envision?
- In the discussion of results shown in Table 2, the authors claim that the number of LLM calls of their approach is comparable to that of ToS where human gives feedback, but the numbers in the table show that their approach consistently has higher number of calls compared to ToS. On what grounds are the authors making this claim? Are the authors considering feedback by humans in ToS as “pseudo” calls? This point should be clarified.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper focuses on utilizing LLMs to solve planning problems that require search. Previous methods like ToS defined a search space with code, but required a human in the loop to inspect. The paper proposed AuthToS to achieve 100% accuracy on 5 synthetic tasks without human intervention.

### Strengths
The paper is generally written clearly.

This direction of reducing LLM dependence on human feedback for complex reasoning has potential.

### Weaknesses
I don’t see how this method can be really useful in more practical scenarios. Since the experiment setting is very toy, if you only focus on tasks with available `isgoal` and `succ` functions, you can actually replace the LLM call with very simple human-written functions, even simpler than the prompts you write. 

Besides, I don't believe the 100% accuracy in the experiment section is meaningful. It is trivial to write a simple search method to achieve 100\% accuracy on all of them, as long as you have access to the two above functions.

There are no indices for different sections, which I believe does not conform to the ICLR template’s requirement.

### Questions
I don’t find any details in the paper on the human evaluation protocol details, since as the authors mentioned there’re human intervention in ToS method in Table 2. My guess is there’s some predefined functions as ground truth for ToS, but this is unclear from the text.

It is also better to include more baseline methods in Table 2 comparison to show your method’s practical value. For example, include the simplest search pipeline baseline.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper extends the so-caled thought-of-search (ToS) method which basically uses code to structure the search for solutions. Like a generic A* search algorithm this defines a successor function and a goal test function which can then be compared to any other search approach executed on the same environment.

### Strengths
- The main contribution, according to the authors, is that they can leverage all the existing approaches in software engineering and LLM-based code generation to create an automated feedback mechanism thus avoiding need for human intervention. (But it looks like the human still has to generate some unit tests, unless you are using formal specs of the environment?)
- They show that this approach leads to 100% accuracy (just like the manually intervened ToS models).
- The error analysis is good showing distinctions across the different models.

### Weaknesses
- While I like the paper's approach this seems like a very minor contribution and primarily relies on test-based checks to achieve full 100% automation. The choice of problems might have been restricted to those where this is possible. I am not sure how this can be used as a general reasoning mechanism for LLM applications.
- While I commended the error analysis above, this also feels like an opportunity to use this to see what needs to be done so you can get uniform performance across models and domains.

### Questions
- Please address the issues raised in the weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces an adaption of the Thought of Search (ToS) framework: AutoTos, a system that seeks to sideline humans' involvement in the feedback loop for generated coding snippets. The automation relies on domain-specific and generic unit tests to iteratively provide feedback to LLMs on their code outputs, aiming for code that is sound and complete. Experiments are conducted across five benchmark domains and with multiple LLMs.

### Strengths
- The paper clearly motivates the need to automate the iterative feedback and exception handling process within the LLM-driven ToS paradigm, effectively shifting the burden from continuous human involvement to the specification of unit tests by humans.
- Evaluation is conducted across multiple domains of varying complexity, including classic (i.e. with BlocksWorld and Sokoban), logic (i.e. with PrOntoQA), and mini-crosswords, as well as with various LLMs of different sizes.
- AutoToS often achieves 100% accuracy and does so with a manageable number of LLM calls, usually on par with the original ToS approach involving human experts, especially when using larger LLMs.

### Weaknesses
- Even though much of the process is automated, the framework assumes either existing or easily generated unit tests. Consequently, human involvement is not eliminated but rather shifted: from interacting directly with the LLM to designing or generating appropriate unit tests.
- The authors use examples from the 24 Game without providing sufficient explanations of the game itself. Including a brief description of the game in the Background section, ideally expanding the existing explanation on lines $275-277$, would help readers unfamiliar with the game better understand its rules and how states transition through arithmetic operations, thereby clarifying the relevance of the successor states used in the examples.
- The framework relies on soundness and completeness as feedback properties. While the experimental results show the benefits they bring, the paper itself does not include any comparison with other types of feedback or other feedback-guided approaches.

### Questions
- Could you please verify the equation on line 112? Shouldn't it be $f(s_i, a_i) = s_{i+1}$?
- What does $T$ represent on line 115?
- Providing more details on the limitations and challenges of generating the unit tests, as well as possible coverage metrics for those tests, would help clarify the full extent of the remaining human effort.

### Soundness
3

### Presentation
2

### Contribution
2
