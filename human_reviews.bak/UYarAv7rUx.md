# Can Mamba Always Enjoy the "Free Lunch"?

- Decision: Reject
- Scores: 5, 5, 3, 3, 5

## Abstract
Transformers have been the cornerstone of current Large Language Models (LLMs); however, its linear growth in overhead during inference with respect to sequence length poses challenges for modeling long sequences. In this context, Mamba has gradually attracted attention due to its constant-level size during inference and existing empirical results have shown that it can perform comparably to Transformers in sequence modeling while offering significant savings. However, one may ask that, can Mamba always enjoy the ``free lunch"? In this paper, we focus on analyzing the expressive ability of Mamba from a theoretical standpoint. First, inspired by the connection between Mamba and linear attention, we investigate potential shortcomings of the Mamba when performing the COPY operation. Our results indicate that Mamba with constant size may encounter bottlenecks when handling COPY, while it can achieve perfect performance when the size scales linearly with sequence length. Based on this observation, we analyze Mamba's ability to tackle DP problems when equipped with Chain of Thought (CoT). Our findings suggest that to solve arbitrary DP problems, the total cost of Mamba is comparable to standard and efficient Transformers. However, similar to efficient Transformers, when facing DP problems with favorable properties such as locality, Mamba can provide savings in overhead. Our results contribute to a deeper understanding of Mamba.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper provides theoretical analysis to the question of whether Mamba can always enjoy the “free lunch”, that is whether it is able to achieve similar performance as Transformer while remaining significantly lower cost. The paper provides analysis for two tasks: COPY and dynamic programming  (DP), and demonstrates that for the COPY task, fixed-size Mamba may encounter bottlenecks, unless the size of the model grows linearly with sequence length, and for the DP task there is no particular advantage of Mamba unless the problem has a local structure.

### Strengths
This paper provides some in-depth theoretical study which elucidates the limits of the Mamba model architecture. These contributions help researchers better understand the trade-off between performance and computational cost for the two popular model architectures.

### Weaknesses
Existing works already demonstrated similar limitations of Mamba (and SSM in general) in handling tasks like COPY (especially [1,2], which the paper also referred to). The conclusions of this paper did not appear to have provided much additional insight, and the contributions are incremental. What is more, the theoretical derivation and results of the paper (Theorem 1-4, and the assumptions they build upon) seem to be pure complex mathematical constructions, which do not appear to be concise and interpretable enough to be fully convincing or empirically verifiable.

[1] Repeat After Me: Transformers are Better than State Space Models at Copying
[2] The Illusion of State in State-Space Models

### Questions
- The theoretical result in Theorem 1 is honestly a bit difficult to understand intuitively: Is this a loose or tight bound? It appears to be a characterization of the difficulty of a COPY problem, but it is difficult to get a sense of what it indicates in practice as there are many variables introduced to control this bound. It would be great to justify the interpretability of this theoretical result and how this bound is an accurate characterization of the model behavior (e.g. plot curves comparing theoretical bounds with empirical results).

- Similarly, the analysis of the paper depends on critical Assumptions 1 and 2. While I understand these are mathematical constructions needed to make the derivations go through, these assumptions appear to be a bit too convoluted and difficult be measured or justified empirically. In general, there are many ways to derive theoretical upper and lower bounds, depending on the choice of assumptions, however an “optimal” theory should be based on minimal assumptions while producing interpretable outcomes.

- It would be great to also provide some experimental examples to support the claim of Sec. 5 (Mamba does not offer additional cost savings compared to Transformers for DP problems), similar to the COPY problem.

- In Fig. 2, what is the Transformer baseline in the rightmost chart?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper provides a theoretical overview of Mamba by following a reformulation to make a direct comparison to attention in Transformers. Then, the authors present theorems in the context of a “COPY” task (akin to correctly outputting a token from an earlier position). First, they prove that Mamba can indeed perform the COPY task; next, they prove that performing COPY for any token from a length-N sequence would require O(N)-sized Mamba blocks. Empirical results on a small synthetic dataset also show that Mamba still performs worse than Transformers on training dynamics, but otherwise correspond with the proven results. Finally, the authors explore chain-of-thought solutions for dynamics programming (DP) problems which leverage past results, proving that a sufficiently-sized Mamba can generate the correct answer and that actually depending on the type of DP problem, a tighter bound on Mamba size can be found.

### Strengths
1. The main results (Theorems 1 and 2) on the relation between token/recall distance and size of the Mamba block, while overlapping with Jelassi et al., 2024, are still different and appear new. 

2. There is insightful empirical analysis on learning dynamics for Mamba (for COPY synthetic task) that complement the theoretical results.

3. Additional theoretical results (Theorems 3 and 4) relate the more general conclusion from Theorems 1 and 2 to an applied problem setting (DP problems), including a comparison against standard and sparse transformers. The proof/extension to DP/CoT-type problems in particular is nontrivial, although is in the appendix presumably due to space constraints.

### Weaknesses
1. While the inclusion of DP problems feels necessary to relate some of the abstract findings to a real problem (CoT with DP), the findings feel inconclusive because it teaches us about representational power and is an existence proof. However, as we observe in the Phonebook experiments, that doesn’t translate in practice to viability; in particular the bounds are still quite loose. When reading the paper, it felt like Section 5 was a separate paper as it introduces a new set of notation despite the result being ultimately a corollary of the original theorem.

2. It isn’t stated clearly why the assumptions (both Assumption 1 and Assumption 2) are safe to assume. When are these assumptions satisfied?

Assumption 1, first condition – I don’t understand why it is reasonable to assume this. Assume $L=1$, and then we would need $|c_i^\top b_{pos(i)}|$ to be maximal for all $i$. This doesn't seem to be necessarily true for all tokens, like stop words. While this feels more likely true as we increase $L$, it still doesn't feel like a safe assumption. Even the assumption that for large $L$, all $|c_i^\top b_j| > \delta$ for all $j \in S_i$ seems unlikely as there could be one token that ends up with a low score. I guess I don’t understand why we think $|c_i^\top b_j|>\rho$ just because $j \in S_i$. Separately, if we want to ensure a “gap”, shouldn’t there be an $\epsilon > 0 $ so that  $\rho \ge \delta + \epsilon$?

Assumption 1, second condition - this is missing a something. “There exists $S_i$ such that…” or “There exists $j$ such that?

For all the conditions of Assumption 2, it would be helpful to give a title to each condition, like “Existence of expressible solution”, “DP is poly(n)” and “DP steps can be approximated by Mamba”. 

3. The presentation of the core theorems (Section 4.2) is very difficult to understand. It might be easier to present Theorem 1 in terms of $\delta$ (and use $\rho$ only in the proof - the first condition doesn't actually need $\rho$ either). For Theorem 2, the “simple intuition” could be better expressed in relation to the terms of Eq. 7, i.e. as a direct result of the first term increasing.

Some other examples:
$M$ is not defined in equation 7, but it should be mentioned as $M = \max_i||x_i||_\infty$; as mentioned above, the 2nd condition of Assumption 1 is unclear’ “ploy” -> “poly”; we should be targeting output tokens, but it seems like the problem statement has changed to target vectors. 


**In summary** of 1, 2, 3, combining the fact that the assumptions are not obvious or intuitive, the conditions of Eq 7 seems possibly difficult to satisfy, and the middling empirical results, it is hard to know whether the existence proof is a rare special case or something more in general. It is still interesting even if it is a rare special case as it is indicative of representation/expressivity of Mamba, but the lack of discussion around it makes its impact hard to understand.


4. Finally, the empirical results would be significantly more interpretable if instead of the total number of parameters, the actual values of the Mamba block size were reported, since that's what the theoretical results are in reference to.

### Questions
1. “free lunch” is not well-defined, and using it in the title is a bit misleading, since it is specifically referring to the sequence length/COPY task. My one-sentence understanding of all the findings is:

> If the distance of the token or information being recalled is N, then Mamba needs block size O(N) to recall that information.

Do you agree with this generalized statement? It would be an intuitive corollary (which you also have the proof for) then that for an m-local DP problem with T steps, it would take O(m) size and O(mT) cost. I want to make sure I am not misunderstanding the paper.

2. What’s the significance of the $l_\infty$ norm in Theorem 2? Is it a corollary, assumption, or not actually relevant to the results?

3. Is it possible to express the DP conditions of Assumption 2 in terms of a formal problem complexity class? It reminds me of DSPACE(poly(n)) but not exactly either. If there's a better term, that could help clarify the types of problems Theorem 2 targets because as of now, it is only including a (admittedly large) subset of commonly-seen DP problems.

4. For Theorem 2: is it possible to give a tighter bound than poly(M,N)? L817 suggests it is $NM^2$, is that a correct bound?


Minor comments that can be changed before final version:

* The results initially appear quite similar to Jalessi et al., 2024, and I still thought that even though the authors address this in the introduction, stating that Jalessi handles sequences does not represent the difference properly. I now believe the two papers and findings are different: Jalessi et al., 2024 focuses on scaling in terms of sequence length and number of states while this paper is focused distance of a token. Something in **bold** in the introduction could reduce this confusion.

* It would be nice to include an example of a DP problem in the appendix so that we don't need to open a separate paper to find one.

* Fix the BERT citation in L32

* There is some prior work/discussion around length (Ben-Kish et al., 2024 [link](https://arxiv.org/abs/2406.14528v1)) and "no free lunch" (Sec 5 of Mamba paper). 

* Sec 4.1, when we set H_0=O, what is O? Do you mean 0?

* L212 Why is a_i \in [0,1]? I know why it is positive, but it isn’t clear to me why it’s <=1

* L718: le -> \le? 

* L271: “given a input” -> “given an input”

* L279 “and” -> “and” (un-italicize it)

* L305: O(ploy(M, N) -> O(poly(m,n)) ["ploy(m,n)" appears multiple times elsewhere too]

* Sec 4.3 header: “Mamba **is** Empirically Weaker…”

* L347: “copy task finally, Transformer” is missing punctuation

* Update the theorem numbering in the appendix so they correspond with main body

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper explores the Mamba model's expressive power, particularly its performance on COPY tasks and dynamic programming (DP) problems. The authors show that, with linearly scaling size, Mamba can accurately execute COPY operations and, with Chain of Thought (CoT) prompting, can handle DP problems at a complexity similar to Transformers unless the problem has m-locality structure.

### Strengths
Many theoretical analyses are provided.

### Weaknesses
The theoretical results and empirical observations presented in this draft largely reiterate findings already covered in the literature, with minimal novel insights that would interest myself or other readers, leading me to recommend rejection.

Firstly, the theoretical contributions appear trivial. Although I am unsure why the authors define the COPY operation in its current form (see question 1), this task seems more akin to token-level (soft) associative recall. Stronger theoretical results on the state size requirements for RNNs (including Mamba) for recall tasks, such as Theorem 3.1 of Arora et al. (2024) [1] and Theorem 4.6 of Wen et al. (2024) [2], already establish that all RNNs require \(O(L)\) state size to address these tasks. Additionally, as acknowledged by the authors, Section 2.3 of Jelassi et al. (2024) [3] similarly concludes that achieving sentence-level COPY in Mamba (and other RNNs) requires linearly increasing hidden size. Given these precedents, I find the theoretical contribution of Theorems 1-2 here unclear in significance and unsurprising, as the proofs align closely with this existing body of work.

Secondly, regarding the CoT proof, Mamba can be formulated as a (gated) linear attention model, as detailed in Yang et al. (2024) [4]. This draft seems to restate established findings by connecting Mamba to linear attention and applying known results on Efficient Transformers (i.e., standard linear attention) from Wen et al. (2024) without adding unique insights. Thus, the paper’s contribution remains ambiguous.

Finally, this draft requires improvement in writing quality. The notation system is confusing; for example, it is unclear why in line 237, ¥$\alpha_j = \prod_{k=j+1}^i a_k$ is defined without any subscript or superscript in $\alpha$.

### Reference
[1] Simple linear attention language models balance the recall-throughput tradeoff https://arxiv.org/abs/2402.18668

[2] RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval  https://arxiv.org/abs/2402.18510

[3] Repeat After Me: Transformers are Better than State Space Models at Copying Transformers are Better than State Space Models at Copying https://arxiv.org/pdf/2402.01032

[4] Gated Linear Attention Transformers with Hardware-Efficient Training  https://arxiv.org/abs/2312.06635

### Questions
Regarding the COPY task setting, it appears that for each token $i$, a "relevant token" $\text{pos}(i)$ is randomly selected from a set $S_i$, defined such that the absolute value of $c_i^T b_{\text{pos}(i)}$ exceeds a threshold. Could you clarify the rationale behind this definition? I find it unclear why the COPY operation is structured this way and would appreciate an explanation of its intended purpose and relevance.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper studies MAMBA's limitations in copying tasks. The authors start by introducing MAMBA basics and how MAMBA simulates linear attention. Then, the authors argue that, similar to linear attention, the dot-products between the current position and target positions should be larger than other positions in order to copy the target positions. Therefore, scaling MAMBA can improve copying performance as scaling is beneficial to linear attention.  Furthermore, the authors discuss how Chain-of-Thought can improve MAMBA in dynamic programming, but gains are limited by the length of the answer.

### Strengths
The topic is interesting. Models don't (currently) have an explicit blackboard to store variables for copying, only approximately through state vectors based on probabilistic predictions.

### Weaknesses
1. The presentation is not good. Notations are not concise enough for readers. One solution is, the authors can provide a notation list and figure out a way to share notations with literature. Figure 1 is a good idea. The authors could integrate equations into the figure.  I also feel some texts and equations are redundant in the main texts. For example, preliminaries (introduction to MAMBA) are too long and only Eq 1 is closely related to this paper. The paper should be more dense and concise.

2. Motivation is not clear. The authors claim that their work focuses on token-level tasks (line 115) in contrast to existing works. But all the discussions and experiments are based on copying a sequence. This can be seen in definition 1 "Then the output of COPY operation is a sequence of vectors o1,o2,...,oN with oi = vpos(i) where pos(i) ∈ Si is the position we want to copy".  Furthermore, it is not clear how copying connects to dynamic programming. Does the authors study in the setting that the model only do copy for DP?  Can you try some toke-level copying tasks like LAMBADA.

3. To me, the findings are not novel and not distinguished from existing works. Most of the paper focuses on introducing existing works. 

4. The main claim about the bottleneck of MAMBA is not discussed in this paper.  While we can see this from empirical studies, it is not clear how we can take inspiration from the assumption the authors make.  This is important as the authors claim they explain the limitations from theoretical standpoints.

### Questions
Typos:
fI: In →In ? line 397
A(n) should be An. line 420.

I'm happy to discuss the paper in case of misunderstanding.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper is mostly a theoretical study on the computational expressiveness of the Mamba model. 
It studied Mamba's ability on a self-defined, non-standard "COPY" task and the computational cost needed to solve dynamic programming (DP) problems.
Under several important yet *opaque* assumptions, the paper draws several contingent conclusions ---

* Mamba *"may"* have trouble performing the "COPY" task reliably when the context to be copied can have arbitrary length (while the model is of a constant size)
* If the model size is allowed to scale linearly with context length to be copied, it can perform the "COPY" task perfectly
* Mamba requires a computational cost to Transformer to solve DP problems when equipped with CoT
* In DP problems with locality properties, Mamba can save computation

In addition to theoretical analysis, which comprises most of the paper, the authors also conduct empirical experiments that further sheds light on the realistic behaviors of Mamba on the COPY task.

### Strengths
* I appreciate the detailed yet concise background introduction to Mamba in Section 3 and 4.1.  These sections are well-written, set up good notation and summarize the literature on Mamba's architecture and its connection to linear attention well.

* Despite the existing theoretical work on SSM's ability to perform COPY task, the authors attempted to formulate a different kind of COPY task and studies Mamba's ability to solve it.  This formulation goes beyond the discretized formulation of the COPY in prior work and study the numerical approximation aspect (however, to make this possible, the authors introduces several *significant* caveats that I'll detail in Weakness section). 

* Studying the computational complexity of Mamba using the connection between SSM and linear attention is novel.  This angle allows the authors to adapt the proofs in Feng et al. (2024) that were intended for the softmax-attention in the standard Transformer.

* To the extent of the portions that I have checked, the mathematical derivations seem correct (this does not mean that the results are significant).

### Weaknesses
My main issue is that the theoretical results presented in the paper not only comes with significant caveats (in the form of assumptions) that are very hard to interpret, but also have conclusions that are hard to interpret and may be too weak to draw conclusions from.  These issues are somewhat swept under the rug by the authors since they did not discuss the implications. 

Starting with Definition 1, it appears that the set $S_i$ is actually dependent on a specific instance of Mamba model since the definition of the COPY task involves the intermediate hidden states $c_i$ and $b_i$.  Furthermore, the COPY task is limited to the cases where the positions to copy must be in the set that's defined *with respect to* a specific instance of Mamba model.  The fact the task is dependent on the specific model parameters seems circular and hard to interpret, but the authors made no effort in discussing or disclosing this caveat.  

Aside from the previous hidden assumption, the actual Assumption 1 is also opaque.  It's very unclear when an actual Mamba instance will satisfy this assumption, and without such discussion, the theorems derived under this assumption should not be used to draw even more general conclusions, such as "Mamba models may encounter bottlenecks when handling COPY task".  The assumption itself is also written in rather confusing language.  I spent a long time trying to figure out the dependence of the variables.

Finally, theorem 1 states that there is a specific Mamba model that can perform COPY but its lower bound $\rho$ can blow up.  What it *doesn't* say, is that any Mamba model that can perform COPY (up to $\epsilon$) will suffer from this exponential blow up.  The statement of the theorem seems too weak that the later "conclusion" seems misleading.  I interpreted this as -- Given the task and $L$, one can find a Mamba block that approximates the COPY task well; but the theorem says nothing about the rest of the Mamba blocks in the hypothesis space.  This is in contrast to previous work, such as Jelassi et al. (2024), who provided universal error bounds for SSMs on the COPY task.

Overall, the dependence between variables should be made clear (See concrete confusion points in the Questions section).
The authors can perhaps think deeper about the effects of the caveats and incorporate such thinking into the writing. 
Moreover, the authors should more explicitly frame the limitations of their conclusions, so that readers won't think their time is wasted after they decided to dig deeper into paper based on the abstract or introduction.  I'd be glad to update my scores if these concerns are addressed.

### Questions
As written, variables in definitions/theorems/assumptions are so entangled that, as a reader, it is quite hard to understand their logical relationships, e.g.

* the task definition depends on $\delta$, but then the theorem states that $\delta$ observes some constraint; it would be helpful to discuss this circular dependence

* the task definition depends on $b_i, c_i$ which depends on the Mamba block, but then the theorem shows the existence of a Mamba block given the task.  Again, it would be helpful to discuss this circular dependence

Another concrete way to improve these confusion points is to stay close to standard mathematical language, such as "Given ___, there exists ___, s.t. ___ holds" and ensure that the assumptions are actually made clear.  For example, Definition 1 probably should include "Given a Mamba block and ..."; Assumption 1 first part can be written as "There exists $\rho$ such that for any $i \in [N]$, ... holds".

### Soundness
2

### Presentation
2

### Contribution
3
