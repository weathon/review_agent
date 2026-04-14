# Causal Order: The Key to Leveraging Imperfect Experts in Causal Inference

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8, 8

## Abstract
Large Language Models (LLMs) have recently been used as experts to infer causal graphs, often by repeatedly applying a pairwise prompt that asks about the causal relationship of each variable pair. However, such experts, including human domain experts, cannot distinguish between direct and indirect effects given a pairwise prompt. Therefore, instead of the graph, we propose that causal order be used as a more stable output interface for utilizing expert knowledge. When querying a perfect expert with a pairwise prompt, we show that the inferred graph can have significant errors whereas the causal order is always correct. In practice, however, LLMs are imperfect experts and we find that pairwise prompts lead to multiple cycles and do not yield a valid order. Hence, we propose a prompting strategy that introduces an auxiliary variable for every variable pair and instructs the LLM to avoid cycles within this triplet. We show, both theoretically and empirically, that such a triplet prompt leads to fewer cycles than the pairwise prompt. Across multiple real-world graphs, the triplet prompt yields a more accurate order using both LLMs and human annotators as experts. By querying the expert with different auxiliary variables for the same variable pair, it also increases robustness---triplet method with much smaller models such as Phi-3 and Llama-3 8B outperforms a pairwise prompt with GPT-4. For practical usage, we show how the estimated causal order from the triplet method  can be used to reduce error in downstream discovery and effect inference tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores the use of large language models (LLMs) for causal inference tasks, particularly causal discovery. The authors present two key arguments: (1) For both LLMs (termed as imperfect experts) or perfect experts, it is more effective to identify the topological order of causal variables rather than directly attempting to discover the full causal graph. They demonstrate that this approach yields more robust results and remains valuable for downstream tasks such as causal discovery or treatment effect estimation. (2) Since LLMs are imperfect experts, relying on existing pairwise prompts to determine causal order may still result in cycles. To address this, the authors propose a triplet-based prompt design. They empirically validate their claims on several datasets.

**Claim**: I have research experience in causality. However, I have limited experience in LLMs or LLMs for causality. Hence, I might miss something especially when it comes to evaluation of novelty or comparison with existing works.

### Strengths
1. Regarding the first key contribution, I find the proposal of identifying causal orders rather than full causal graphs to be both useful and insightful.
2. Overall, the paper is easy to follow. While there are some typos, the writing remains generally clear.

### Weaknesses
I will present my main concerns here. Minor concerns or questions can be found in the Question section.

**Significance of contribution**

I have several concerns regarding the methodology section:
1. 3.1 - While I agree with the intuition that identifying causal order is easier and leads to fewer errors, I’m not sure the result in this subsection is particularly interesting by itself. It seems almost obvious, as knowing the correct causal order is a prerequisite for identifying the causal graph. What matters more is whether the reduced chance of errors still preserves the usefulness of the method.
2. 3.2 - This section should have provided evidence addressing the question above. However, I find it difficult to assess the significance of the method presented. Proposition 3.2 offers a sufficient condition for backdoor adjustment, but applying this condition broadly to include everything that satisfies it seems impractical.
3. 4.1 - Should these be considered as contributions of the paper?
4. 4.2 - The authors try to justify why triplet prompt is better than pairwise prompt here. One thing unclear to me is that, as more variables are provided in the context (as the authors also mentioned), there is a higher chance of LLMs making mistakes. In practice, does that mean an $\epsilon$-expert LLM would become an $\epsilon’$-expert where $\epsilon’>\epsilon$? If so, would this become a trade-off and how should we interpret this tradeoff?
5. 4.2 - Another question regarding this theorem: It seems that the major problem the authors try to resolve is to avoid cycles. However, Proposition 4.1 focuses on the error of edge prediction given the assumption of acyclicity. Hence, I am not sure if this theorem provides insight on how to resolve the key problem of this paper.

**Experiment**

My main concern is the lack of comparison with existing LLM-based methods. For example, while it’s valuable to verify that providing causal order helps with downstream causal tasks, do we know if integrating causal order into causal discovery outperforms directly using LLMs to identify the causal graph or providing other types of inputs to a causal discovery algorithm? Could the authors justify their choice not to compare against existing baselines, such as those mentioned in Section 2?

### Questions
1. Experiment - Missing reference to Table 2 in the main paper.
2. Line 114 - Could the authors provide some justification on why the redundancy leads to a more reliable order?
3. Line 52 - Would using triplet prompt lead to a more efficient method in this sense?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a new technique for causal discovery algorithms by leveraging triplets. The authors argue that traditional causal discovery algorithms based on pairwise variable relationships cannot handle situations involving mediating variables and can lead to circular structures in causal graphs. Thus, they emphasize the importance of utilizing causal order, meaning modeling based on the topological order among variables. Causal order is effective for downstream tasks, and the triplet approach can efficiently identify causal order. The authors discuss the error of estimated causal order from both empirical and theoretical perspectives, especially in the presence of imperfect experts.

### Strengths
The approach introduces the use of causal order as the output for causal discovery algorithms employed by LLMs, with D_{top} topological divergence as a metric. This novel approach brings a fresh perspective and insight into the fields of LLMs and causal discovery.

### Weaknesses
1. Typo error: Lines 333-334 seem to have missed a closing parenthesis in O(V^3).
2. Complexity considerations: According to the paper, the triplet-based method has a complexity of O(V^3), whereas the traditional pairwise method has a complexity of O(V^2). For larger-scale graphs, the time complexity of these methods poses significant limitations for causal discovery.
3. The approach of using LLMs to simply determine causal relationships between variables is challenging to implement effectively in real-world applications. It requires the large model itself to possess a high level of expert knowledge, which is often difficult to achieve, as real causal inference typically demands deep domain-specific expertise and nuanced understanding beyond general-purpose language models.

### Questions
1. The current method produces a causal order rather than a full causal graph. While the authors suggest that, in the absence of confounders, this causal order can offer a superset of backdoor adjustment sets, the presence of “imperfect experts” in real-world applications may lead to additional error accumulation in the causal order. This raises uncertainty about whether the derived causal order is sufficiently reliable for downstream tasks, such as causal effect estimation, and how significantly these biases could impact the results. Experimental validation would be valuable to assess the practicality and risks of this trade-off.
2. In the triplet method, the authors employ a four-step process to determine causal order. Unlike the pairwise approach, each iteration uses a third auxiliary variable \(C\) to determine the causal edge direction between \(A\) and \(B\), followed by aggregation across all variables. My understanding is that the pairwise approach is simply a special case of this method where the number of auxiliary variables equals zero. The authors could further clarify the distinctions and differences between the pairwise approach under this four-step method and the current approach.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
LLMs are often used as an expert for finding causal graphs, but they can’t distinguish between direct and indirect edges. Authors proposed to use causal order as a more stable version of it, along with a triplet prompt approach. 
- Their proposed approach facilitates the recovery of causal order despite imperfect experts. 
- Prompt strategy that introduces auxiliary variables that introduces an auxiliary variable for every variable pair and instructs the LLM to avoid cycles within this triplet. 
- Triplet prompt leads to fewer cycles than pairwise	
- Causal order is a simpler structure that can still encode helpful information for down-stream tasks;

### Strengths
Well-written motivation and discussion of limitations; 

Good discussion on the utility of the causal order, and related works. 

Sound experiments with empirical evidence of the effectiveness of the proposed method.

### Weaknesses
Despite a sound experimental analysis, there a few details missing, i.e., how many repetitions were done in Tables 1 and 2, 

One potential area not explored/mentioned is what happens with a large number of features. As the proposed method decreases the feature space, would the proposed method also facilitate the adoption of causal discovery areas that contains larger graphs?

### Questions
Q1: Can you clarify the experimental setup in Table 4? The title shows ‘ours - < causal discovery > + LMM’. But in line 279 it is described that “causal order is used to reduce search space of causal discovery methods”. In that sense, should it be LLM + < causal discovery > instead? As the LLM with triplet is used to reduce the search space inside the causal discovery method. 

Q2: Can you describe in more details the causal effect component? How is the causal order used in that context? Is it used to drop variables (aka, no informative features or down-stream features)? How are the counterfactuals estimated? 

Q3: Another ablation study that would be helpful is to see the robustness to the number of variables. Most causal discovery methods struggle to scale for larger graphs / number of features. How LLMs + triplet could be adopted to reduce search space and improve this current causal discovery bottleneck?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors aimed to find a better interface to use domain knowledge for causal discovery, including LLM, but not restricted. For LLM, this is done via a pairwise prompting strategy. The authors found that expert knowledge often falls short in distinguishing direct and indirect causal relations between pairwise variables, correspondingly many cycles in causal graphs, but their causal order is relatively precise. Given this finding, the authors suggest using expert knowledge of causal order instead of causal relation. While integrating the strategy of using causal order and LLM prompting, the authors propose to introduce an auxiliary variable to improve the conventional pairwise prompting strategy by avoiding cycles within the triplet. Experimental results support that the proposed prompting methods increase the robustness and performance.

### Strengths
- The paper is well-written. Motivation, method, and experiments are logically straightforward.
- The attempts to utilize LLM for causal discovery have identified certain problems, such as the difficulty in distinguishing between direct and indirect causal relationships, which consequently lead to cycles in the causal graph. The authors convincingly argue that these issues are inherent to the pairwise prompting approach.
- The authors propose using causal order to leverage domain expert knowledge in causal discovery without encountering the issues associated with pairwise prompting. Additionally, it presents a novel prompting strategy to support this approach.
- When the proposed method was utilized, performance improvement was consistently observed across experiments. Additionally, the experimental design accommodates node size scaling from small to large datasets, allowing for a comprehensive understanding of how the method is affected by node size.

### Weaknesses
Overall, I found this work very interesting, though a few questions remain unanswered. Addressing these issues could potentially lead to a higher score.

- It is hard to determine whether triplet prompting is effective for high-performance LLMs (such as GPT-4, Claude3-opus). It appears that the prompting strategy and LLM can be applied orthogonally, so it is weird the corresponding results are omitted. Could you add results for Triplet prompting + high-performance LLMs in Table 2, to clarify this matter?
- In the experimental design related to the downstream causal discovery algorithm, it is difficult to determine how the proposed method's advantages change with varying observations. Could you add corresponding experiments with varying sample sizes to clarify this matter?

### Questions
- While domain expert knowledge is independent of the number of observations, it significantly affects the causal discovery algorithm. Therefore, it raises curiosity about how much performance improvement the proposed method can achieve in situations of data scarcity. Could you conduct an experiment with a very small dataset as a showcase for a data scarcity regime? It would be very interesting.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper considers optimal ways of querying imperfect experts (e.g., LLMs) when the aim is to discover a causal DAG over a set of variables. The key proposed idea is to query imperfect experts about the causal ordering over variables instead of graphical relationships like edges. The paper shows that certain ways of orienting edges based on queries to experts can lead to errors in the overall DAG, while the causal ordering from a perfect expert will never be incorrect. The paper proposes ways of integrating causal ordering into known causal discovery algorithms, and then focuses on extracting orderings from imperfect experts. To minimize errors by imperfect experts, the paper proposes querying for the order over triplets of variables while enforcing acyclicity, proving that this makes fewer errors than querying about pairs of variables (under some assumptions). The work then studies this querying strategy empirically using a variety of hand-curated known causal DAGs. They study both human and LLM imperfect experts, and find evidence that the triplet querying strategy improves causal discovery and effect estimation performance over both the pairwise querying strategy and using no experts at all.

### Strengths
+ The technical as well as written clarity of the paper is excellent. Technically, the notation is clear, all preliminaries and results are well-defined and well-explained. The paper is also written in a way that is easy to read and follow.

+ The technical result about the optimality of the triplet querying strategy over the pairwise querying strategy is novel, to the best of my knowledge, and could have an impact on causal discovery algorithms more broadly.

+ The empirical studies are comprehensive, and the initiative to contrast both humans and LLM imperfect experts is impressive since studies with humans are time-consuming and costly.

### Weaknesses
+ In the broader context of causal discovery, I question the idea that LLMs are $\epsilon$-imperfect experts. Existing results including those in this work consistently consider causal discovery problems that involve standard medical knowledge, or general knowledge about the world that's fairly well-established by now. However, causal discovery as a field ought to be about discovering new knowledge, in domains where we have very only faint hypotheses about variables relate, e.g., consider abstract variables that represent aspects of human behavior like "trust" or "aggression." In these settings, we should expect $\epsilon$ to be so large as to render LLM experts unreliable. I'm curious to see the authors better justify and contextualize the role of LLMs in genuinely challenging causal discovery problems.

+ The premise in this paper is that querying experts with questions of the form "does A cause B" and directing the edge A->B if the expert says yes leads to incorrect graphs. However, the query "does A cause B" isn't by itself inherently flawed -- it's already a question about causal ordering -- the policy to orient edges based on ancestral causal relations is what's flawed. Nevertheless, the introduction and parts of section 3 (e.g., definition 3.1) seem to critique the query itself, when really, the issue is that responses to this query should not be used to directly orient edges (e.g., the perfect expert in 3.1 is defined based on a flawed policy). Therefore, the argumentation strategy in the paper feels like setting up a strawman argument. Note, though, that this is just a critique about the storytelling; the technical results are valid and make sense regardless. One could pivot the story slightly to argue that the flaw isn't with the "does A cause B" style of query, but in ensuring that this is interpreted only as a causal ordering.

+ There is a bit of extra context that I would have liked in the empirical studies: 1) can you contextualize the datasets and graphs that are studied further and discuss whether they are semi-synthetic, purely realistic, etc.? 2) Despite the results in Figure 3 on synthetic data, it would be useful in the broader empirical studies to compare the causal ordering queries to the flawed policy of orienting edges based no responses to "does A cause B"-style questions. (I apologize if this comparison is indeed there and I missed it -- if that ended up being the case, please point me to the correct figure or table in the empirical studies.)

### Questions
+ Why do you think the premise of LLMs as $\epsilon$-imperfect experts makes sense more generally in causal discovery, given that $\epsilon$ *should* large if we're solving a truly interesting causal discovery problem?

+ Can you the details about the empirical studies that I mention above?


Edit: The authors satisfactorily addressed my questions and concerns in their response; I'm happy to thus raise my score and support their work.

### Soundness
4

### Presentation
3

### Contribution
3
