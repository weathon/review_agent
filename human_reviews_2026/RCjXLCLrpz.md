# Fluid Reasoning Representations

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Traditional large language models struggle with abstract reasoning tasks. By generating extended chains of thought, reasoning models such as OpenAI's o1 and o3 show dramatic accuracy improvements. However, the internal transformer mechanisms underlying this superior performance remain poorly understood. This work presents an early mechanistic analysis of how reasoning models process abstract structural information during extended reasoning. We analyze QwQ-32B on Mystery BlocksWorld -- a semantically obfuscated benchmark that measures planning and reasoning capabilities.
We find that QwQ gradually improves its internal understanding of actions and concepts through its extended rollouts, developing abstract representations that focus on structure rather than specific action names. Through steering experiments, we establish causal evidence that these adaptations improve problem solving: injecting refined representations from successful traces enhances accuracy, while symbolic representations can replace many specific Mystery BlocksWorld-obfuscated encodings with minimal performance loss. We therefore find that one of the factors driving reasoning model performance is in-context refinement of token representations -- which we call Fluid Reasoning Representations. This provides early mechanistic interpretability into reasoning models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates how reasoning models develop internal representations during long chain-of-thought (CoT) reasoning. Specifically, the authors analyse the QwQ-32B model on an obfuscated planning task (Mystery BlocksWorld) to test whether reasoning generalises beyond surface-level token identities. The authors hypothesise that *reasoning models progressively refine their internal representations of problem entities during reasoning, developing context-specific semantics that enable abstract structural reasoning independent of surface-level semantics*. 


The study comprises three main analyses:
1. *Representational Dynamics:* The authors track how hidden representations of actions and predicates evolve over reasoning timestamps and across multiple “naming” schemes, showing that representations of the same underlying concepts converge across different surface names.


2. *Causal Steering:* To test whether these learned representations are behaviorally meaningful rather than merely correlational, the authors perform activation steering, i.e., directly injecting or perturbing hidden-state vectors during reasoning, and observe that positive steering improves accuracy while negative steering degrades it.


3. *Symbolic Patching:* To probe abstraction, the authors replace naming-specific embeddings with averaged, naming-invariant “symbolic” representations. Performance remains stable mainly, suggesting the model’s reasoning operates in an abstract representational space.

### Strengths
- The paper addresses an important mechanistic question: how reasoning models internally represent abstract structure during extended reasoning.

- The experimental setup is well-motivated, and the task (Mystery BlocksWorld) provides a clean testbed for analyzing abstraction.

- The authors conduct multiple complementary analyses, including steering and patching interventions.

### Weaknesses
- The paper’s central hypothesis that "**reasoning models** dynamically refine internal representations of problem entities ..." requires validation across more than a single model. Demonstrating the same phenomena in at least one additional reasoning model (and ideally contrasting with a non-reasoning or base variant, such as Qwen-32B) would significantly strengthen the claim.


- The robustness and reproducibility of the results are not yet clear. Some experimental details (e.g., the choice of layers, token windows, or the 40 “solved” reference puzzles) appear somewhat arbitrary or underspecified. Including these in a reproducibility table or appendix would greatly help future replication efforts.

- The writing occasionally over-generalises the findings, implying broader conclusions than the presented evidence supports. The authors should temper claims about “reasoning models” in general and clarify that observations are currently limited to QwQ-32B and the specific task.

- Some results (e.g., steering improvements) are small and would benefit from clearer statistical reporting and effect sizes.

### Questions
Please check the weakness section

### Soundness
2

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
This paper analyzes how QwQ-32B's internal representations evolve when solving Mystery BlocksWorld, a semantically obfuscated planning benchmark. The authors extract representations of actions and predicates at different points during reasoning traces and show that representations converge toward similar encodings across different naming schemes. They conduct steering experiments where refined representations from successful traces are injected into new problems, and symbolic patching experiments where naming-specific representations are replaced with averaged vectors.

The paper does not contain any mathematical properties or theorems. The experimental setup and research methodology appears sound. 

Overall, the paper adapts the framework in Park et al. for planning problems. It showed that the phenomenon discovered in Park et al. can also be observed in planning problems, in particular, in the Mystery Blocks World instances.

### Strengths
1. Clear demonstration of representational convergence: The paper effectively shows that representations become increasingly similar across namings as reasoning progresses, with divergent representations at early timestamps converging around 7k tokens. This temporal progression is well-visualized and makes the adaptation process tangible, providing genuine insight into how models refine their understanding during extended generation.

2. Causal Validation via Steering: The paper successfully uses steering experiments to prove its claims. By injecting the refined representations from successful traces into new problems, the authors show these representations causally improve problem-solving accuracy. The fact that the averaged "cross-naming" representations (which are purely abstract) had the strongest positive effect supports the paper's central hypothesis.

3. Sophisticated cross-naming methodology: Creating 15 diverse naming variants and averaging representations across them to extract symbolic encodings is a thoughtful approach to isolating abstract structural meaning from surface-level lexical information.

4. Insightful base model comparison (Section 3.3): The finding that base models exhibit similar representational adaptation when processing the same traces is valuable. This helps clarify that reasoning models leverage a fundamental capability of transformers through extended generation.

### Weaknesses
1. Insufficient novelty over Park et al. (2025): That models adapt representations during in-context learning is documented in prior work (Park et al., 2025, cited by authors). This paper essentially applies their framework to Blocks World rather than discovering reasoning-specific mechanisms.

2. Single model analysis despite multi-model data: Table 1 shows results for DeepSeek-R1, Llama Nemotron, and QwQ, yet only QwQ is analyzed mechanistically. This makes it unclear if the findings are general to all models or specific to QwQ. Since the authors already ran performance benchmarks, why not apply the same representational analysis to the other models to test for generalization?

3. Single domain with no generalization evidence: BlocksWorld has only a small number of concepts with deterministic rules. Zero evidence that findings extend to other reasoning domains. Testing even one additional domain is essential to demonstrate this isn't domain-specific.

4. Missing explanation of refinement mechanism: The paper documents that representations converge toward symbolic encodings but does not explain the internal mechanism by which this refinement occurs. What computations cause early representations to transform into refined ones? Which attention heads read/write these representations? What information do different layers add during refinement? The paper observes the phenomenon (representations change) without explaining the process (how/why the model performs this transformation internally).

### Questions
Please see the weaknesses session.

### Soundness
3

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
4

### Summary
This paper presents an early mechanistic analysis of how reasoning models process abstract structural information during extended reasoning, which analyzes QwQ-32B on Mystery BlocksWorld. This paper finds that QwQ gradually improves its internal understanding of actions and concepts through its extended rollouts, developing abstract representations that focus on structure rather than specific action names. Through steering experiments, it establishes causal evidence that these adaptations improve problem solving.

### Strengths
1. This paper provides insights for abstract reasoning area.
2. The method is somewhat novel.
3. The discovered theory can be applied for LLMs enhancement.

### Weaknesses
1. The presentation of this paper should be significantly improved.
2. The experiments are limited, the conclusions are not universally applicable.

### Questions
This method about this paper is novel, while the experiments and presetations shoule be significantly improved before acceptance.

1. Figure 1 is placed after the abstract, while there is no details about Figure 1 in the introduction.
2. There is neither a formal definition of the task nor exmples of the task.
3. What does in-naming and cross-naming mean? What does high and low values of them represent?
4. There should have a formal definitation of Mystery BlocksWorld.
5. I know action and predicate in language, are they the same in your paper?
6. The analysis are about actions and predicates, why the hypothese is about entities (such as lines 203-204)?
7. "we first create a set of all possible token sequences that could encode this action". How to understand "token sequences" encode "this action"? There are many such difficult-to-understand sentences in the article.
8. Conducting expeirments on other reasoning LLMs is helpful to make your conclusions universal.
9. More datasets should be considered to further enhance the persuasiveness.
10. A whole workflow is needed to better demonstrate your method, only the text can make confusion.

### Soundness
2

### Presentation
1

### Contribution
3
