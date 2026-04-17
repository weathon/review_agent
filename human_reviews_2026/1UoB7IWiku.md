# Code World Models for General Game Playing

- Decision: Accept (Poster)
- Scores: 2, 8, 8, 6

## Abstract
Large Language Models (LLMs) reasoning abilities are increasingly being applied to classical board and card games, but the dominant approach---involving prompting for direct move generation---has significant drawbacks. It relies on the model's implicit fragile pattern-matching capabilities, leading to frequent illegal moves and strategically shallow play. Here we introduce an alternative approach: We use the LLM to translate natural language rules and game trajectories into a formal, executable world model  represented as Python code. This generated model---comprising functions for state transition, legal move enumeration, and termination checks---serves as a verifiable simulation engine for high-performance planning algorithms like Monte Carlo tree search (MCTS). In addition, we prompt the LLM to generate heuristic value functions (to make MCTS more efficient), and inference functions (to estimate hidden states in imperfect information games). Our method offers three distinct advantages compared to directly using the LLM as a policy: (1) Verifiability: The generated CWM serves as a formal specification of the game's rules, allowing planners to algorithmically enumerate valid actions and avoid illegal moves, contingent on the correctness of the synthesized model; (2) Strategic Depth: We combine LLM semantic understanding with the deep search power of classical planners; and (3) Generalization: We direct the LLM to focus on the meta-task of data-to-code translation, enabling it to adapt to new games more easily. We evaluate our agent on 10 different games,  of which 4 are novel and created for this paper. 5 of the games are  fully observed (perfect information), and 5 are partially observed (imperfect information). We find that our method outperforms or matches Gemini 2.5 Pro in 9 out of the 10 considered games.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a very interesting .

### Strengths
This is a thorough, well-founded work. It is well-motivated and it starts with good rigour and a clever approach to what will eventually become, in my opinion, a large open problem in agentic workflows.

### Weaknesses
While the motivations/contributions are strong, the paper's methodology is not there yet. It has some weaknesses that (unfortunately) do not make it, in my _personal_ opinion, ready for publication. 

1. The paper's writing is inconsistent across the work. Tonal shifts occur often (e.g., L431 'you can check it in...')
2. Only one (closed-source) baseline.
    - While not strictly needed, an ablation on prompts/models would be helpful to ascertain _why_ the results are the way they are.
3. Paper structuring and presentation needs work. This is different from the paper's writing: overall the presentation / scientific methodology needs to be improved.

Here's how the paper can be improved:
1. Make the writing consistent--proof-read and ensure that it adjusts to the expected tonality of a scientific article. 
    - Ensure descriptions of tables/figures are well-done. E.g., what is '# LLM Calls' in Table 2? If it is 'number of LLM calls', why is it a float? Is it an average then?
    - The related work section needs _a bit_ (not a lot) of work. Some known works in both applying Markov processes or getting LLMs to play videogames are missing. I would also suggest adding a small explanation on what they did.
    - Make sure your paper is proofread. For example, L308 introduces 'OOD' before defining what it is. The sentence in L86 does not need parentheses. This indicates a lack of proofreading.
    - Definitions are important. Gemini is not indicated as a reasoning model: since the behaviour of an LLM (next-token predictor) and an RLM (baked-in CoT) are quite distinct, referring to Gemini (an RLM) as an LLM does require some clarification.
2. More open baselines/agents will be beneficial for the robustness, soundness, and longevity of the work. These three are _musts_ for contributions to any conference, let alone ICLR. 
    - Related: the code would be better put in a repository. This will avoid presentation issues like those in App. I.2
3. On scientific writing:
    1. Scientific writing follows a very specific template:
        1. Results/Experiments contain the outcomes of your evaluation of the hypothesis. These should be supported with numerical evidence. Opinions and interpretations of the results (like 5.1.1) are for the discussion.
        2. Discussion contains the discussion _of the results_, not of your paper. 
        3. Conclusion (which is missing) allows you to draw a conclusion (or interpretation) of your hypothesis based on data.
    2. Significant digits must be consistent and should make sense (why is accuracy in T2 reported to five significant digits? Is such precision truly needed?).
    3. Skipping the definition of imperfect info games because they are 'tricky' _would_ make sense if it weren't for the fact that (a) the experimental work does rely on imperfect information; and (b) the appendix to which the reader is referred does not contain a formal definition.

I genuinely think this could be a very strong contribution, but needs more work than might be feasible for this submission.

Minor, and not something that influenced my review: the lack of a reproducibility statement plus (1) the fact that all the code is in an appendix; and (2) the baselines are closed-source, do not indicate good, open science practices. I would encourage the authors to rethink this and add such a statement. Again, not something that can/will impact my assessment of the paper, but always nice to have.

### Questions
In addition from my questions above, I'd like to know if it is possible to know what would be the behaviour of a reasoning model alone in this agentic workflow scenario. This _is_ important since a comparison between a Markov-like optimised prompt/workflow versus the effect of a (behind-an-API) reasoning model _versus_ a RLM using the Markov-optimised approaches would allow an ablation on whether it is the prompt, the workflow, or the model that provides the contributions.

### Soundness
1

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes Code World Models (CWMs), using an LLM to translate natural-language rules plus a few example trajectories into executable Python code that implements a game’s world model (state transitions, legal moves, termination), along with synthesized heuristic value functions and inference functions for imperfect-information games. Experiments cover 5 perfect information games and 5 imperfect information games, and showing promising results while using Gemini 2.5 Pro as the LLM.

### Strengths
1. "data to code" fashion creates a verifiable simulation of games, and both perfect and imperfect information games can be applied in this fashion.
2. Two ways of synthesizing inference functions for imperfect information games to avoid the exponential cost is a valuable contribution. Both hidden history inference and hidden state inference are straightforward yet effective.

### Weaknesses
1. Synthesize quality relies on generated test cases over a limited amount of trajectories. When LLM failed to parse game rules, it might not build the code world model effectively. 
2. It seems like CWM performs worse than Random in the game of Gin rummy.

### Questions
1. In closed-deck learning, can you quantify how the learned state-space size correlates with performance? This corresponds to your hypothesis at Line 455 - 457.
2. Have you tried iterative online refinement (updating the CWM during play), and if so, does it reduce reliance on high-quality initial trajectories? (You note it’s possible but skipped for efficiency.)

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors extend the Code World Model (CWM) framework by considering two-player games, performing value function code synthesis to improve player performance, introducing the concept of "inference as code" to enable state estimation in imperfect information games, and providing a learning algorithm (based on code-based autoencoders) to enable learning in the novel closed deck (strict partial observability) setting. Their results shows the superiority of this approach with respect to LLMs as policies on multiple perfect and imperfect information games, including newly created ones.

### Strengths
- Well written paper.
- Very good related works section
- Good amount of environments/games tested

### Weaknesses
- Less ablation studies performed
- Very few baselines
- Only Gemini 2.5 tested. Other LLMs would bring the variance that is needed to be demonstrated

### Questions
There are many who have shown "Code as policies", working in different settings. Why is it better than just creating actions through LLMs?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a new method for gameplay agents, to use LLMs to generate a code world model, which serves as a verifiable simulation engine for the planning algorithms. The experiments evaluate the gameplay performance on 10 different games, outperforming or rivaling Gemini 2.5 Pro.

### Strengths
1. To use the LLM to generate the code world model is a novel idea to represent a specific game in a verifiable manner.

2. The work is solid. I also read the appendix of the paper. It provides all necessary details and examples.

### Weaknesses
1. The experiments are made on 10 distinct games, generalizing 4 to the other 6 games. It is hard for readers to assess the OOD generalizability of the method.

2. One concern is what kind of games (e.g. poker-like games) can benefit from the proposed method. Can the method work for any types of games? The complexity to generate a code world for card games is somewhat low, for example, the code cannot be very long, so what about more complex games?

### Questions
Please see above.

### Soundness
3

### Presentation
2

### Contribution
3
