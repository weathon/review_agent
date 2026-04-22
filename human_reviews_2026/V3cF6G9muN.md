# Sample, Align, Synthesize: Graph-Based Response Synthesis with ConGrs

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Language models can be sampled multiple times to access the distribution underlying their responses, but existing methods cannot efficiently synthesize rich epistemic signals across different long-form responses. We introduce Consensus Graphs (ConGrs), a flexible DAG-based data structure that represents shared  information, as well as semantic variation in a set of sampled LM responses to the same prompt. We construct ConGrs using a light-weight lexical sequence alignment algorithm from bioinformatics, supplemented by the targeted usage of a secondary LM judge. Further, we design task-dependent decoding methods to synthesize a single, final response from our ConGr data structure. Our experiments show that synthesizing responses from ConGrs improves factual precision on two biography generation tasks by up to 31\% over an average response and reduces reliance on LM judges by more than 80\% compared to other methods. We also use ConGrs for three refusal-based tasks requiring abstention on unanswerable queries and find that  abstention rate is increased by up to 56\%. We apply our approach to the MATH and AIME reasoning tasks and find an improvement over self-verification and majority vote baselines by up to 6 points of accuracy. We show that ConGrs provide a flexible method for capturing variation in LM responses and using the epistemic signals provided by response variation to synthesize more effective responses.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a consensus-based decoding approach designed to enhance the performance of LLMs. The method first samples multiple responses from an LLM, constructs a DAG to represent their relationships, and then synthesizes them into a final, unified response. By leveraging this structured representation, the LLM can select and integrate complementary information from different outputs to produce a more coherent and accurate result. Experimental results demonstrate the effectiveness and robustness of the proposed method.

### Strengths
1. This paper presents a clear motivation and proposes a simple yet effective method for enhancing the performance of LLMs.

2. The process of constructing and utilizing the DAG is well presented, and the graph can be applied in various ways, such as aggregation and intervention.

3. Extensive experiments are conducted, and the results appear promising.

### Weaknesses
1. There is a considerable body of research on achieving consensus among multiple responses. However, key variants of self-consistency methods are missing from the comparison, which makes it difficult to fully assess the contribution of this work.

2. Although the constructed DAG is relatively simple, the draft response sampled from it essentially represents the shared content among multiple responses. It remains unclear how this approach fundamentally differs from existing methods in the literature, e.g., self-consistency.

3. As mentioned, the DAG-based approach is claimed to be cost-effective, as shown in Table 2. However, without comparisons against additional baselines, this claim is not well supported. Since only one baseline is used as a reference, it is difficult to determine whether the method is indeed cost-effective.

[1] https://aclanthology.org/2024.acl-long.634.pdf

### Questions
See Weakness

### Soundness
3

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
The paper introduces Consensus Graphs (CONGRS), a novel graph-based data structure designed to generate a single, reliable response from multiple long-form outputs produced by a Large Language Model (LLM). The approach proceeds in three stages. First, multiple responses are sampled from the LLM. Second, these responses are aligned by constructing a CONGR, which identifies shared anchor spans (consensus nodes) and uses a secondary language model to assess semantic equivalence within the variable segments (disagreement nodes). Finally, a unified response is synthesized from this graph using one of two task-specific decoding strategies: Consensus Decoding (an aggregation method for factual tasks) or Guided Self-Verification (an intervention method for reasoning tasks).

### Strengths
1. The method is novel.
2. CONGRS-based synthesis provides significant gains in all areas such as: factuality, safety, and reasoning
3. The process is cost-efficient.

### Weaknesses
1. The CONGRS construction process fundamentally relies on the presence of anchor spans—identical lexical subsequences shared across multiple responses. The paper notes that this assumption holds well for aligned models but degrades in more open-ended tasks. In 6 out of 100 biography examples, the responses exhibited "so much variability that no consensus nodes are made."" For creative writing tasks, the consensus nodes contained "much less text," leading the authors to acknowledge that "CONGRS may offer more limited utility in that setting."
2. While the method is cost-effective, it is not cost-free and still relies on a secondary language model for two critical steps. An LM judge is required to "determine which parts of the responses between the anchor spans are semantically equivalent" , and another LM call is needed in the final synthesis step to "fix grammatical errors" and coherence. This introduces external dependencies and potential points of failure, for example if the judge LLM have bias and could be wrong (the entire process fail). 
3. The paper claims that the final LM-based synthesis step, "does not introduce or remove hallucinations". However, this claim seems really weak when supported by a "qualitative analysis" and "manual inspection" of only "25 biography examples".

### Questions
1. The paper’s core alignment method depends on identifying anchor spans (consensus nodes) through lexical alignment. This approach performs well for aligned models but degrades on more open-ended or creative tasks. How robust is it to different kinds of alignments? For instance, if a model is aligned via RLHF techniques that promote lexical diversity, could this disrupt the lexical alignment process and lead to failures in forming consensus nodes, as observed in 6 of the 100 biography examples?
2. How sensitive is the final graph quality and response factuality to the strength of this LM judge? What would happen if you used a less-capable open-source model as the judge?
3. It seems to me that adaptive consensus threshold ($\tau$) requires manual tuning, since the paper shows that the optimal consensus threshold $\tau$ depends on the entity's rarity (rare entities need a higher, more conservative $\tau$). Can this be truely adaptive?
4. Your "Guided Self-Verification" method uses the CONGRS graph to localize errors for the LM to self-check. However, the paper cites that LMs "struggle to localize errors". So, how often does the LM fail to correct an error you've localized?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes consensus graphs to capture semantic variation across multiple LM-generated responses to the same prompt. By leveraging these graphs, the approach aims to synthesize more effective responses and enhance factual accuracy.

### Strengths
The method provides a reasonable way to identify uncertain segments in LM outputs, which can help mitigate some hallucinations and improve reliability.

### Weaknesses
1. The approach primarily addresses cases where the LM exhibits uncertainty. It is less effective when the LM is confidently incorrect, as consensus cannot correct such errors.
2. The method relies on multi-pass generation and semantic equivalence class identification, which introduces significant LM inference cost. Alternative approaches that utilize internal model states for improving truthfulness exist and may offer more efficiency. Including a comparison with such methods would strengthen the paper.

### Questions
1. Can this method be extended to achieve consensus across responses from multiple models, rather than a single LM?
2. How are hyperparameters selected? Do they require task-specific or model-specific tuning?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Consensus Graphs (CONGRS), a graph-based framework to synthesize multiple language models (LM) responses sampled for the same prompt. The authors adapt a multiple sequence alignment (MSA) algorithm (Needleman–Wunsch) from bioinformatics to align lexical sequences across responses, producing a directed acyclic graph (DAG) that encodes consensus and disagreement spans. Two decoding schemes are introduced:

Consensus Decoding, which aggregates high-agreement nodes to enhance factuality.

Guided Self-Verification, which targets high-disagreement regions to refine reasoning accuracy.

Experiments on factual generation (biography, PopQA), refusal tasks (HALoGEN), and reasoning tasks (MATH, AIME) show improved factual precision, abstention rate, and reasoning performance compared to several baselines.

### Strengths
The paper explores how to exploit response variation across LM samples as a reliability signal. This direction aligns with the interest in epistemic calibration, self-verification, and uncertainty-aware decoding for LMs.

The proposed pipeline (Sample to Align to Synthesize) is conceptually clear, and the DAG representation provides an interpretable way to identify consensus and conflict regions across responses.

Compared to prior LM-judge-heavy aggregation methods (e.g., UAD, ASC), the approach achieves similar or better performance with fewer tokens, which makes it practically appealing.

### Weaknesses
The core mechanism of using MSA (Needleman–Wunsch) to align token sequences and representing them as a DAG is a direct adaptation of standard sequence alignment and partial-order graph techniques.
The "bioinformatics inspiration" reads more like narrative packaging than like algorithmic innovation. The key insight (leveraging multi-sample alignment) could likely be realized without this heavy graph formalism.

The paper's framing ("node frequency as reliability") is conceptually appealing but not formally supported. There is no theoretical link between graph topology and uncertainty.

Constructing a lexical DAG across m responses of length L likely has high cost. The paper does not analyze runtime, memory, or trade-offs for larger-scale reasoning settings.

### Questions
How does CONGRS compare to simpler token-level overlap or attention-based voting schemes in both cost and quality?

How does performance scale with m (number of samples) and L (response length)?

### Soundness
2

### Presentation
3

### Contribution
2
