# Diffusion Language Models are Provably Optimal Parallel Samplers

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 4

## Abstract
Diffusion language models (DLMs) have emerged as a promising alternative to autoregressive models for faster inference via parallel token generation. We provide a rigorous foundation for this advantage by formalizing a model of parallel sampling and showing that DLMs augmented with polynomial-length chain-of-thought (CoT) can simulate any parallel sampling algorithm using an optimal number of sequential steps. 
Consequently, whenever a target distribution 
can be generated using a small number of sequential steps, a DLM can be used to generate the distribution using the same number of optimal sequential steps.
However, without the ability to modify previously revealed tokens, DLMs with CoT can still incur large intermediate footprints. We prove that enabling remasking (converting unmasked tokens to masks or revision (converting unmasked tokens to other unmasked tokens) together with CoT further allows DLMs to simulate any parallel sampling algorithm with optimal space complexity.
We further justify the advantage of revision by establishing a strict expressivity gap: DLMs with revision or remasking are strictly more powerful than those without.
Our results not only provide a theoretical justification for the promise of DLMs as the most efficient sampler, but also 
advocate for why revisions should be enabled in DLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper provides a theoretical justification on why Diffusion language models (DLMs) can simulate any parallel sampling algorithm using an optimal number of sequential steps. It makes use of the theory of complexity theory and boolean circuits. First, it proves that DLM with CoT can generate any distribution (on binary data) with the minimum possible sequential steps. Second, it shows that DLMs empowered with the remasking and revision abilities, can sample from any distribution with the minimum amount of memory. Finally, the authors characterize more precisely the simulation abilities of DLMs with/without remasking or/and revision, showing  that DLMs with remasking or revision are more expressive.

Overall, the paper is theoretical and contains no experiments.

### Strengths
The paper provides a theory of diffusion language models (DLMs) in the context of binary data and boolean circuits, precisely characterising DLMs expressivity and their ability to simulate parallel sampling algorithm with optimality characteristics. To the best of my knowledge, this is the first paper which builds up a theory for DLMs using such an angle.

The paper contains the key results and all the proofs.

### Weaknesses
1. The authors could steer the readers better at the beginning of the paper on how and why the boolean circuits theory is relevant for understanding DLMs. While the authors provide a presentation & background on boolean circuits, what is missing is the step on how such theory could generalize our understanding of DLMs on arbitrary vocabularies.

2. In boolean circuits presentation, the authors introduce class TC with MAJORITY operation, but such operation is never used (nor this class). It seems to be coming out of nowhere, especially for readers who are not familiar with boolean circuits. I think authors could do a better job to guide the reader on why such complexity classes are introduced in the way they are introduced. Moreover, the authors say "By definition, it is known that for all i \in N, ...", but no reference to such a statement is provided.

3. While I understand that the theory the authors propose is about understanding the expressivity and optimality of "ideal DLMs", such theory does not tell us much about what abilities we would have once we train the DLMs. I think the authors should at least include a discussion about that in their paper.

4. The authors introduce CoT as essentially some intermediate sample steps which DLMs could perform, but CoT is more than that -- it is highly structured prompt of very specific type. The theory proposed in the paper does not make this distinction. I think therefore term CoT might not be correct as it is related to a broader literature. Could authors please explain how CoT they describe connects to CoT which is generally used in the LLM litterature?

### Questions
Please see the "Weaknesses" section.

1. Can you provide a more clear connection on the relevance of boolean circuits and your theory angle to a broader understanding of DLMs?

2. Can you explain the complexity classes for boolean circuits and why they are introduced the way they are introduced ? Adding some references would be helpful.

3. How does the proposed theory relate to trained DLMs (which can have some approximation errors?)

4. How does your notion of CoT related to a broadly understood notion of CoT?

### Soundness
3

### Presentation
2

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
This paper studies diffusion language models (DLMs) which recently appeared as an alternative to autoregressive models for their efficiency at inference time (possibility to generate tokens in parallel). It provides theoretical results on the benefits of DLMs for parallel sampling from the perspetive of implementing boolean circuits. This allow to quantify the efficiency of DLMs in terms of circuits classes, depth and bits to implement them. The main results show that for any circuit of depth $d$, there exists a DLM with chain-of-thought (CoT) that can implement it with $d$ decoding steps. This comes with additional large intermediate footprints. Further allowing DLMs to use remasking and revision extends the efficiency and expressivity of DLMs since they become strictly more expressive than DLMs without revision or remasking.

### Strengths
- The paper is well-written and enough context is given for the reader on the notions used in the paper
- The subject is interesting since diffusion LMs are more and more used so a better theoretical understanding is of importance
- The circuit formulation is elegant and the proofs seem sound although I did not check all of the details in appendix
- The theoretical claims are of great interest and the methodology of comparing DLMs with and without remasking and revision is well conducted
- The proofs techniques could be of independent interests to study complex models such as neural networks and language models.

### Weaknesses
I list below what I believe are weaknesses but I would be happy to be corrected if I misunderstood some parts.
- The connection to distribution sampling should be made more explicit
- The succession of theoretical results without much discussion on their insights for diffusion LMs hinders the contributions
- Thm 3.1 is one of the main result but only an existence results. As such, it does not ensure that any DLM can simulate any distribution sampling efficiencly nor does it provide guarantees on the existing DLM that can. As such, I find the presentation of this results in abstract ("We prove that enabling remasking (converting
unmasked tokens to masks or revision (converting unmasked tokens to other unmasked tokens) together with CoT further allows DLMs to simulate any parallel sampling algorithm with optimal space complexity") and the rest of the paper misleading. Could the authors elaborate on that (see questions)?
- Those two weaknessess explained above make unclear the benefits of the current work. While showing the benefits of remasking and revision is interesting, the connection to distribution sampling is unclear and the connection to practical benefits observed in practice is unclear. I believe the paper would be improved with more discussion on that.

Overall, my main issue with the current work is the connection to distribution sampling that is not clear enough and as such, the benefits of the results to explain the current (and potential future) success of DLMs in terms of efficiency and expressivity is unclear beyond the boolean circuit framework. I believe that better connecting that to distribution sampling and practical use of DLMs (still with theoretical results) would improve the current work. This is why I currently lean towards rejecting the paper but a convincing connection would make me lean towards accepting it.

### Questions
- Thm 3.1 is one of the main results and it is an existence result but does not ensure that a given DLM can achieve this low number of decoding steps. As such, its benefits for practial DLMs is unclear. Could the authors elaborate on that point?
- Why not considering prompts in Section 4? Since it echoes results of Section 3, it would be interesting to have similar settings. Could the authors elaborate on that?
- Obtaining meaningful theoretical results often require simplifying assumptions. In line 131, the authors assume that the length with CoT is $L$ while in practice $L$ is larger than q|CoT|o. How would the results change if considering a bigger sequence length $L$?
- Remasking is assumed to be modeled by a random function $G$ ni line 138. What is it in practice? If different than random, how does it impact the theoretical results?

*Typos*
- line 176: "the" should be removed
- line 176: $(x, R)$ should be $(\chi, R)$
- line 200: $\mathbb{Z}^+$ is simply the set of integers $\mathbb{N}$.
- line 202: "unmasked" --> "unmask"

### Soundness
3

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
2

### Summary
In this paper, the authors derive some theoretical results on Diffusion Language Models.
In particular, they  show that random functions realized by a circuit with depth $d$ can be described by a Diffusion Language Model with $d$ decoding steps (when augmented with CoT). 
Interestingly they show that the length of the sequence to be generated with a (masked) diffusion model to generate a circuit of size $N$ is $N$ while in the case of a forward process with non-masked perturbation or in the case where remasking is used at inference time they can generate the circuit with a sequence length of the width of the circuit.
Finally, the authors show that if we restrict the building block circuits for the Diffusion Language Model to be in a certain complexity class then  there is exists an example for which Diffusion Language Models with forward perturbation or remasking can achieve the reconstruction of the examplar circuit in $O(1)$ steps and such a bound is not attainable without the ability to remask tokens.

### Strengths
For me the importance of the results is two fold:

* First, they show the impact of remasking (or using a non-masking forward process) by showing that not only Diffusion Language Models can simulate any circuit but also by showing that the sequence length necessary to generate such circuits is limited by the width of the circuit in the case of Diffusion Language Model.

* Second they highlight an example showing the superiority of Diffusion Language Models with remasking (or using a non-masking forward process) by showing that such Diffusion Language Models can achieve the reconstruction of an examplar circuit in $O(1)$ steps and such a bound is not attainable without the ability to remask tokens.

### Weaknesses
I do not have a lot of complains about the paper. 
I will highlight that I am not an expert on circuits so I did find some parts of the paper hard to follow. 


* I would suggest to clarify the writing especially Section 4. Indeed, in this Section while I understood the results and their consequences I was unable to follow the logical structure. (Again it might be acceptable for experts in circuits but I could not follow).

* Importance of the results: I am not fully convinced that being able to generate circuit is indicative of performance of Diffusion Language Models. In particular, are there real-word scenarios which are well described by the circuit framework the authors describe (sudoku, puzzle?). It is unclear to me that in the case of _text_ this analysis remains relevant. I would appreciate a discussion about this potential limitation. 

* All the results rely on CoT, it would be more impactful to understand what would happen without CoT assumptions. 

* All the proofs and the main points of the paper rely on the fact that we can perform parallel sampling operations within the framework of Diffusion Language Model. One big omission in my opinion is that speculative decoding/sampling are powerful methods for AR models which unlock parallel sampling for AR models. I think those models and the consequences of the analysis presented in the current paper and its extension to spec decoding is key to understand what are the key ingredients of the proof and what is truly specific to DLM

### Questions
* The authors state that "More recently, DLMs with revision (Song et al., 2025) are introduced and exhibit remarkable capability." 
It is not really clear to me, how the "revision" technique differs from a uniform discrete diffusion. In that case, that would mean that we leave the masking world and that strategies such as remasking are not needed. I am wondering if the current framework also allows the authors to understand this scenario?

* Related to this question, I am wondering if the strategy of the authors could be used to understand the benefits of current continuous augmentation strategies such as CCDD [1] .

* What would be the step to extend the results beyond binary vocabulary? What are the blockers?

* Typo line 176 (notation for the input)


[1] Zhou et al., (2025) -- Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper provides a theoretical analysis of Diffusion Language Models (DLMs) by leveraging the framework of  Boolean circuit complexity. DLMs have become very popular recently as they allow faster inference than Autoregressive (AR) models via parallel decoding. The manuscript represents parallel computation using circuit depth (time) and circuit width (space). They analyze DLMs with some (unusual) Chain-of-Thought (CoT), acting as a kind of scratchpad, and two increasingly popular inference-time mechanisms: remasking and updating (modifying already unmasked tokens).

The paper establishes the following results:
Theorem 3.1 shows that DLMs using sufficient CoT can simulate any depth-$d$ circuit in exactly $d$ rounds. This achieves the theoretical minimum for parallel computation, contrary to AR models which require steps proportional to circuit size.
Theorem 3.2 shows that by remasking and updating with CoT, DLMs can achieve optimal time ($O(d)$ steps) and optimal space (memory proportional to circuit width $w$).
In Section 4, the paper shows that standard DLMs with components restricted to the $AC^0$ complexity class cannot sample the uniform parity distribution in constant time, whereas DLMs with updating/revision or remasking can.

### Strengths
Originality: The paper is original as it is (to the best of my knowledge) the first paper that analyzes the parallel sampling capabilities of DLMs through circuit complexity ideas, so it is an original combination of ideas from theoretical CS and generative modeling. 

Quality. The main results are rigorous and the proofs are constructive. I am not an expert on this type of results but could follow (most of) them. Theorem 3.1 provides an equivalence between circuit depth and decoding rounds. The construction in Theorem 3.2 appears non-trivial, it achieves near-optimal space complexity using remasking/updating. The expressivity separation of Section 4 provides a concrete example demonstrating the limitations of standard DLMs.

Clarity. The paper is quite difficult to read but it is also because it brings two literatures together. It provides a  formalization of the DLM inference process (Algorithm 1) which defines quite clearly the roles of the predictor, the scheduler, CoT, remasking, and updating/revision. The mapping between these components and Boolean circuits is clarified in Section 2.2 and is key to the understanding of the theoretical analysis.

Significance. This provides positive results justifying the advantage of DLMs over AR models, as sequential steps scale with circuit size rather than depth. It also shows that heuristic techniques proposed to improve DLMs (updating/remasking) are indeed mechanisms necessary to achieve optimal space efficiency and expressivity.

### Weaknesses
My main concerns are how these theoretical results relate to practical DLM training and inference; i.e. the presented results are interesting but I am not convinced they are telling us much in particular about existing DLMs and whether there is a path to exploit these results to obtain better DLMs.

Expressivity and Learnability. The main results (Thms 3.1, 3.2) are existence proofs. While they show that an optimal predictor $p$ and scheduler $\mathcal{F}$  exist for specific circuit classes, they fall short of addressing whether these can be learned by standard architectures using standard ELBO objective. It would be beneficial to explain (empirically or theoretically) whether standard training technique can be learned via standard stochastic gradient techniques or whether these techniques tend to find less parallelizable solutions. 

Computational Cost. Optimality is defined as minimizing the number of sequential rounds. This does not really take into account the actual computational cost or wall-clock latency of using predictor $p$ and scheduler $\mathcal{F}$ in each round. It would be useful to clarify that the results do not appear to guarantee lower overall FLOPs compared to AR models (correct me if i am wrong), only lower latency for an idealized parallel hardware. 

Complexity of the unmasking schedule. The optimality results appear to rely on an unput-dependent, and dynamic unmasking schedule $\mathcal{F}$ that precisely identifies which tokens (circuit nodes) to compute next. This is much more complex that what is typically used in practice. It would be beneficial to discuss the schedule$\mathcal{F}$ . How robust are the results if you were to modify the $\mathcal{F}$, would one be far off the optimal? how would we learn it practically?

$AC^0$ Constraints. The  results of Section 4 appear to rely heavily on the assumption that the predictor and scheduler are constrained to $AC^0$. However, $AC^0$ is a weak complexity class. How would these results translate to practical scenarios where one  has access to much more powerful predictors, e.g., Transformers. It would be useful to clarify the implications of the  $AC^0$ constraint. Would the  architectural separation outlined in section 4 still hold for much more powerful predictors? 

CoT. I find the reliance on CoT quite disturbing... In particular because the CoT considered in the theoretical arguments appears to differs very significantly from practical CoT, i.e. a chain of reasoning steps. I think this requires very substantial clarification.

### Questions
I think it is accepted that transformers are more powerful than $AC^0$. What happens to Theorem 4.5 in this case? If the separation does not hold anymore, what does this imply about the necessity of updating/revision for expressivity in practice?

Is there any evidence, whether theoretical or empirical, that standard diffusion training objectives encourage the model to discover the clever time/space-optimal strategies  outlined in  Theorems 3.1 and 3.2? If it is not the case, can you think of any objective that would explicitly force he model to utilize revision/remasking effectively for few-round decoding?

How robust is the $O(d)$ round complexity to the optimality of the scheduler $\mathcal{F}$? Do existing suboptimal (unmask random tokens) or adaptive schedules (i.e. greedy decoding) significantly increase the required rounds?

Beyond parity, can you think of other types of distributions exhibit hardness for standard DLMs but become tractable with updating/revision?

### Soundness
3

### Presentation
3

### Contribution
2
