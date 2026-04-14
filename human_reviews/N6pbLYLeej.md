# Chain-of-Thought Provably Enables Learning the (Otherwise) Unlearnable

- Decision: Accept (Poster)
- Scores: 6, 6, 10, 6

## Abstract
Modern language models have demonstrated remarkable reasoning capabilities by using chain-of-thought (CoT). One hypothesis about the inner workings of CoT is that it breaks down originally complex tasks into smaller subtasks that are more amenable to learning. We formalize this notion by showing possibility and impossibility results of learning from in-context demonstrations with and without CoT. In particular, with CoT, we examine a family of learning algorithms that learn a task step-by-step, capable of composing simpler functions from individual reasoning steps to form an overall complex function. This process reduces the difficulty of learning a task to that of the hardest reasoning step in the chain. Moreover, we prove Transformers can express this algorithm and thus they can efficiently in-context learn arbitrary tasks as long as these tasks can be decomposed into a finite number of subtasks, each of which are efficiently learnable. In contrast, without CoT, we demonstrate that there exist tasks that are inherently unlearnable by the same algorithm. Overall, our results suggest several provably effective ways for decomposing target problems to instantiate CoT. Empirically, we demonstrate our proposed CoT construction significantly enhances the reasoning capabilities of real-world LLMs in solving challenging arithmetic reasoning tasks, including learning polynomials and Boolean formulas.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the impact of subtask complexity on the reasoning capabilities of Transformers when employing Chain-of-Thought (CoT). Theoretical results demonstrate the expressive power of Transformers to learn from in-context CoT samples through in-context gradient descent. Furthermore, the paper establishes upper and lower bounds for the error bound $\Delta(\mathcal{P}, h)$, highlighting the significance of task decomposition. Experimental results illustrate the effectiveness of CoT in parity and DNF tasks.

### Strengths
This work emphasizes the importance of understanding principles for designing CoT steps, providing several insights for creating more effective CoT formats when presenting in-context samples. The theoretical findings regarding error decomposition at each CoT intermediate step reveal the hardest subtask as a key bottleneck in reasoning.

### Weaknesses
1. While the paper proposes effective strategies for decomposing target problems as claimed in lines 23-25, these strategies appear limited to specific problems like parity or DNF. It remains uncertain whether these approaches can be generalized to CoT format design for a broader range of reasoning problems.

2. The theoretical results are somewhat unconvincing:
   - Lemma 1 indicates that a Transformer with depth linear to the number of reasoning steps $k$ and the number of gradient descent (GD) steps $t$ diverges from practical applications. Current large language models (LLMs) typically possess at most hundreds of layers, yet they can handle thousands of context lengths. Consequently, a construction involving constant-depth Transformers with polynomial hidden dimensions may yield more practical insights.
   - In Lemma 2, the bound becomes less meaningful when $k$ is not treated as a constant. However, reasoning lengths often are not constant; the expressive power is strictly constrained. [1] illustrates that a constant-depth, log-precision Transformer with polynomial hidden dimensions and either no CoT reasoning or $O(\log n)$ CoT reasoning steps cannot solve problems outside the $\mathsf{TC}^0$ class, where $n$ is the problem size. To strengthen the effectiveness of CoT, the reasoning steps $k$ should generally be $\Omega(\log n)$, and in many instances, $k = \text{poly}(n)$. Under these conditions, the bounds become quite loose.

[1] Chain of Thought Empowers Transformers to Solve Inherently Serial Problems, https://arxiv.org/abs/2402.12875

### Questions
1. Can you describe how to design effective CoT formats for more general problems using the strategies proposed in this paper?

2. What are the input and output formats (including CoT steps) for all experiments? What is the length of CoT reasoning steps for each task?

3. Lines 355-357 indicate that increasing reasoning steps $k$ negatively impacts performance. Can you elaborate on how this conclusion is drawn from the experiments in Section 5?

4. Is it feasible to develop a Transformer with constant depth as proposed in Lemma 1? For instance, could the filtering process mentioned in lines 303-305 be applied? A polynomial embedding size is acceptable in the context of constant layers, which aligns more closely with practical applications.

5. In lines 381-397, why is $|\mathcal{H}_1'|$ expected to be large when applying CoT, while $|\mathcal{H}_1'|=1$ without CoT? Given the problem setup, $\mathcal{A}$_CoT is an algorithm that learns from demonstrations, meaning the output predictor $h$ is influenced by the input demonstrations and is not necessarily unique. In this context, the assertion that $|\mathcal{H}_1'| = 1$ without CoT seems questionable.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies how chain-of-thought (CoT) impacts in-context learning (ICL) for transformer models. There are two main results. First, the paper shows that a linear-depth transformer model can implement a learning algorithm A_CoT that chains together k gradient descent learning algorithms A_i that run for t steps each for 1 <= i <= k (Lemma 1). An upper bound on error is given for A_CoT, thereby showing that a composite task is efficiently learnable by a transformer model if each subtask is efficiently learnable by each of the k individual gradient learning algorithms (Lemma 2), where the hardness of learning is bounded by the difficulty of the most difficult subtask (Theorem 3). Second, the authors lower-bound the error of A_CoT in the case of finite hypothesis class for A_1 as a function of the size of that hypothesis class (Theorem 4), which suggests that there may be functions that are learnable with CoT that are not learnable without CoT. A concrete witness is identified for learning parities (Corollary 5).

### Strengths
The paper studies an important problem in trying to theoretically elucidate CoT and ICL for transformer models. There is also preliminary experimental results that corroborate the theoretical results.

### Weaknesses
The presentation is quite dense could be improved to explain the motivation and intuitions behind definitions/results.
- For instance, the definition of in-context learning (Definition 1) makes a distinction between predictors and a learner (around line 195-196). From my understanding, the definition of Model(Data, query) = LearningAlg(Data)(query) = predictor_D(query) is just partial evaluation, and so the distinction that is drawn between predictors and learners is somewhat artificial. Rather, the main idea seems to me to formulate ICL, and so we just define a function that takes a query that is a tuple (demonstrations, x) as opposed to x. Moreover, it would be helpful to explain why PAC learning is chosen and to give pointers (presumably to Theorem 4) on how exactly PAC learning will be used to help you demonstrate your results and why it is sensible.
- Why does Definition 2 use multisets?
- Some of the material in the appendix on how your Transformers implement gradient descent result differs from Cheng et al. and Von Oswald et al.'s would be helpful in the main text to give more context.
- Could you offer more clarification on how to extend the results to handle k > 1 in lines 294-305 which is from what I understand is the main point of innovation of the paper. However, I can't seem to find more details in the main text or the appendix.
- The references for this paper do not use ICLR's reference style ...
- The statement of Lemma 1 might be helped by stating that t steps of gradient descent map to t layers of self-attention which is mentioned in the appendix.
- I find the statement of Theorem 4 to be somewhat strange. If I understood correctly, without CoT, we can essentially run for t gradient steps on a neural network that predicts from a representation that has t self-attention layers. With CoT for k steps, we can essentially run for kt gradient descent steps with kt self-attention layers and k prediction layers. How does Theorem 4 control for the same number of layers across reasoning steps k? Intuitively, I expect a theorem showing that there functions that are learnable with CoT that are not learnable without CoT to (1) show that a kt layer network run for kt gradient descent steps can not learn the task purely with an identity (trivial) task decomposition and (2) that a task decomposition (T in theorem 4) used by CoT can be constructed in polynomial time *uniformly* in the task with a Hypothesis class H_1'. However, there are no limits on the task decomposition T in the theorem and it's currently not clear to me how the size of the Transformer is controlled in the current statement of Theorem 4. Perhaps I misunderstood and would be curious to get more insight into what is actually going on.

### Questions
Please see weaknesses for questions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper analyzes Chain of Thought (CoT) learning in LLMs. The key idea behind CoT is to learn hard tasks by being shown intermediate steps involved in learning. The paper formulates this as in-context learning (ICL) which means that a parametric language model can be trained to generalize a set of demonstrations (with k steps) to predict the final output for a new input. The paper shows that CoT can be implemented by a step-by-step algorithm, where each step is generalized by a learning algorithm A_i from the examples of the i'th stage. The key lemma is to show that a transformer of depth linear in k can express the learning algorithm A_{CoT} = A_1...A_k as a sequence of gradient descent steps. 
The main theorem then shows that CoT can efficiently learn a set of tasks which can be decomposed into subtasks with a sample complexity bounded by the sample complexity of the hardest subtask. Negatively, the paper gives a lower bound on the overall error as a function of hypothesis space and the problem class. 
The concrete results show that the class of parity functions is not polynomially learnable in 1 stage, but can be learned in the CoT framework in 2 stages. 
The paper introduces a new benchmark for CoT with controllable hardness. Each stage of each function manipulates two integers through addition and multiplication. The hardness is controlled by skipping some of the intermediate steps of the function. The success rates on GPT-40 and GPT-3.5 illustrate the predicted result - they degrade gradually with increasing the maximum number of contiguous missing steps.  They also show positive results on an (n,k)-parity task and on 3-term DNF learning using the same LLMs.

### Strengths
The paper gives a strong theoretical justification and analysis of CoT in the context of LLMs, building on prior work on understanding the prior work on viewing LLMs as few shot learners. The analysis is sound and well-presented. 

The paper shows concretely how CoT makes otherwise unlearnable classes learnable through decomposition.   

The empirical results support the theory and clarify the paper. 

The paper presents some interesting benchmarks including a challenging task that can express arbitrary polynomials over integers, which can be controlled for hardness.

### Weaknesses
It appears that this work is somewhat related to `learning from exercises' in the problem solving setting (eg, https://dl.acm.org/doi/10.5555/93335.93344, and https://onlinelibrary.wiley.com/doi/10.1111/j.1467-8640.2008.00330.x) and curriculum learning (http://arxiv.org/abs/2101.10382). It would be good to see a discussion of the relationship.

### Questions
Since this work builds upon a number of previous works in CoT and transformers, please state precisely what was previously known and what are the novel (theoretical) contributions.  

It is not always obvious how to decompose a given task. Do you have any advice on how to approach this on a practical problem?

The importance of the work is diminished by having to manually decompose each new task to generate demonstrations. Do you have any thoughts on how to reduce this burden?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the effectiveness of chain of thought (CoT) in the in-context learning setting. Particularly, it is shown that there is a parametrization of the Transformer architecture that allows in-context learning of certain functions with CoT. It is further claimed that there are functions that are only learnable with CoT and not otherwise in this setting. The paper concludes with experiments with GPT models and their in-context learning abilities based on the granularity of the CoT provided to them.

### Strengths
- The paper is generally well-written and theoretically studies an important problem: ‘effectiveness of CoT in the in-context learning setting’. Moreover, the paper generally proposes rather clear definitions of in-context learning with and without CoT based on the PAC framework. 
- A particularly interesting theoretical prediction is that making CoT more granular and longer does not necessarily improve the performance.

### Weaknesses
- One problem with in-context learning works as done in this paper is that it is only proven that there is a Transformer parametrization suitable for in-context learning (similar to an expressivity result). However, it is not clear whether this parameterization is realistic or not. I.e., if a Transformer pre-training is done in a reasonable manner, it’s not clear if it's possible/likely that the Transformer achieves the needed parameterization. (E.g., the claimed parametrization may not be reachable by GD in practice and there is generally no insight on the pre-training process.)

### Questions
- The depth requirement in Theorem 1, $kt$, seems rather impractical and quite large. Further, I think there are several clarifications required. For start, I think $t$ itself is not a parameter of learning setting but part of the learning algorithm. We are in the PAC setting in this paper. So it would indeed make sense to give the depth of Transformer based on $k, \delta, \epsilon$ instead. In particular, for me it’s not clear if the complexity in $k$ is really linear. Based on eq. (11) we know that as $k$ increases, we have a compounding and increasing error. Thus, probably more GD iterations are also needed. I.e., for a given $\epsilon, \delta$, parameter $t$ is not proven to be independent of $k$. So effectively, I’m not sure if the depth is linear in $k$ as claimed. I think giving the depth in $k, \delta, \epsilon$ would fix all these issues. 
- The negative result for the in-context learning is given for the hypothesis class in eq. (7) and not Transformers. Can we say something about Transformers not being able to learn the functions in question in an in-context fashion?
- In definition 1, part 2, how is $\mathbb{E}_x[l(h(x),f(x))] \leq \epsilon$ with prob. $1-\delta$ equivalent with $\Delta(\mathcal{P}, h) \leq \epsilon$?
- For Theorem 3, it is said that the number of samples we need is equal to the number of samples needed for the hardest subtask. However, the samples used by different subtasks are not independent. Can you further clarify how is this taken into account?
- Given a fixed architecture (with fixed depth), can you please further elaborate on the effect of making CoT more granular and also the effect of increasing the number of in-context samples?
- In the in-context setting, when we go the in-context setting we provide more info by giving each sample. In other words, comparing the sample complexities with and without CoT doesn’t seem to be totally fair. Maybe, can you give a comparison based on the total number of tokens given? 



Suggestions/Minor remarks:
- For me, the notation $\Delta(\mathcal{P}, h)$ used in equation 5 is not clear. $h$ on the right hand side depends on the function being learned however the $h$ is on the left hand side should be independent of it and represent the learning algorithm. Maybe replace it with $\mathcal{A}$?
- I would suggest that the authors include the existence of a hypothesis class in page 2 to increase the clarity.

### Soundness
3

### Presentation
3

### Contribution
3
