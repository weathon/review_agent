# ProofBridge: Auto-Formalization of Natural Language Proofs in Lean via Joint Embeddings

- Decision: Accept (Poster)
- Scores: 6, 2, 2, 6, 6

## Abstract
Translating human-written mathematical theorems and proofs from natural language (NL) into formal languages (FLs) like Lean 4 has long been a significant challenge for AI. Most state-of-the-art methods either focus on theorem-only NL-to-FL auto-formalization or on FL proof synthesis from FL theorems. In practice, auto-formalization of both theorem and proof still requires human intervention, as seen in AlphaProof’s silver-medal performance at the 2024 IMO, where problem statements were manually translated before automated proof synthesis.

We present ProofBridge, a unified framework for automatically translating entire NL theorems and proofs into Lean 4. At its core is a joint embedding model that aligns NL and FL (NL-FL) theorem+proof pairs in a shared semantic space, enabling cross-modal retrieval of semantically relevant FL examples to guide translation.  ProofBridge integrates retrieval-augmented fine-tuning with iterative proof repair, leveraging Lean’s type checker and semantic equivalence feedback to ensure both syntactic correctness and semantic fidelity. Experiments show substantial improvements in proof auto-formalization over strong baselines (including GPT-5, Gemini-2.5, Kimina-Prover, DeepSeek-Prover), with our retrieval-augmented approach yielding significant gains in semantic correctness (SC, via proving bi-directional equivalence) and type correctness (TC, via type-checking theorem+proof) across pass@k metrics on miniF2F-Test-PF, a dataset we curated. In particular, ProofBridge improves cross-modal retrieval quality by up to 3.28x Recall@1 over all-MiniLM-L6-v2, and achieves +31.14% SC and +1.64% TC (pass@32) compared to the baseline Kimina-Prover-RL-1.7B.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
ProofBridge is a unified framework for directly translating both the statement and proof from natural to formal language. It utilizes an embedding system that projects FL and NL statements to a joint space. This is used as a retriever for the autoformalization model which the authors train using SFT. The model is tested with iterative repair and it demonstrates superior performance in type check rate and semantic correctness compared to baselines.

### Strengths
The task of end-to-end proof autoformalization has, as the authors claim, been underexplored and it is an interesting new problem formulation. It is well-motivated by the difficulty manual formalization, such as during the AlphaProof experiment on IMO 2024 problems. Although there has been existing work on statement or proof autoformalization, an end-to-end approach for both statements and proofs has rarely been explored or evaluated. The experiment results rigorously show the proposed method's advantage over the numerous baselines considered. This paper may serve as a solid baseline for this end-to-end autoformalization setting for the community.

The presentation of this paper is clear and easy to follow.

### Weaknesses
My biggest concern is that while the problem setting has been underexplored, the method that the authors consider is largely only piecing together existing approaches. The method largely resembles existing work on retrieval-augmented statement autoformalization (see below), with the added complexity that it starts training from Kimina-Prover-1.7B which has been trained for proving already.

Contrary to what the authors claim, I think NL/FL joint embedding has been explored in previous work, in the form of first informalizing the FL statement and then embedding NL queries and informalized theorems into a joint space, and retrieving similar FL examples from NL queries at test time for the purpose of statement or proof autoformalization. This was proposed in prior work like [1, 2, 3]. In this paper, the authors proposed a slightly different method that trains linear projections from FL and NL embeddings (based on existing models) into a shared space. Although technically different, I don't see how this is advantageous compared to existing approaches based on informalization which have shown to be effective. Either some explanation or quantitative comparison is needed to demonstrate this.

[1] https://arxiv.org/abs/2403.13310v2
[2] https://arxiv.org/abs/2410.10878
[3] https://openreview.net/forum?id=hUb2At2DsQ

The "Semantic Correctness" (SC) evaluation metric is not new. In [1], the BEq metric was introduced which is basically the same as SC. Subsequent work such as RLMEval [2] and StepFun-Formalizer [3] have extended and improved BEq (although I admit the latter should be considered contemporaneous work). Even ignoring BEq, there has been extensive prior work on using an LLM judge as a semantic correctness check. This is a small issue, but I would encourage the authors to recognize the previous work.

[1] https://openreview.net/forum?id=hUb2At2DsQ
[2] https://arxiv.org/abs/2406.07222
[3] https://arxiv.org/abs/2508.04440

The retriever baseline, all-MiniLM-L6-v2, shouldn't be considered a SOTA. That is a small model pretrained on general NL data. I encourage the authors to try a larger pretrained model such as e5-mistral-7b-instruct (which LeanSearch seems to use), or at least not label the current baseline as SOTA (because LeanSearch exists).

### Questions
For the same natural language problem, there might exist different correct ways to state it in Lean, some being easier to prove than others. Do the authors envision ProofBridge to be extensible to optimize for downstream provability of formalized statements?

### Soundness
3

### Presentation
4

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
This paper presents a framework for automatically translating natural language mathematical theorems and proofs into Lean 4 formal language with RAG. The approach consists of three main components: (1) a joint embedding model that aligns natural and formal language theorem-proof pairs, (2) retrieval-augmented fine-tuning of LLMs, and (3) an iterative repair mechanism that uses Lean's type checker and semantic equivalence verification. The authors demonstrate improvements over baselines on some experiments.

### Strengths
1. The use of LeanDojo’s ByT5 encoder to capture the "linearized DAG traversal" of Lean proofs, rather than just treating the code as flat text. It might be a thoughtful method for encoding the structural properties of formal proofs.
2. The integration of RAG and SFT is well-executed. The authors provide corresponding ablation studies which help demonstrate the impact of SFT.

### Weaknesses
1. The core motivation of the task, i.e., translating a known natural language proof into a formal proof, is questionable. The primary value of formal math languages is to verify statements whose correctness is uncertain. In most practical scenarios, a formal statement is the starting point, and the proof is the unknown to be found. If an informal proof is already known to be correct, the motivation for formalization becomes unclear. Of course, you can state that it's useful in some scenarios, like translating textbooks and formalizing existing proofs. But currently, what you are claiming is that the previous 2-stage practice (formalize statement and prove in formal language) is problematic, and yours is correct. It's not the case.
2. The paper claims the superiority of encoding the "linearized DAG traversal" using the ByT5 encoder. However, there is no ablation study to support this. How does this compare to a simpler baseline, such as encoding the raw text of the FL proof?
3. The experimental comparisons in Table 2 are insufficient and potentially misleading: (1) The baseline models were also provided with the full input context, i.e., the NL theorem, the NL proof, and retrieved RAG examples, right? Then the authors should provide ablations that isolate the benefit of the (arguably unrealistic) input. Specifically, how much does adding the NL proof and the RAG component each contribute to the performance of both the baseline and the proposed model? This is crucial to understand if PROOFBRIDGE is genuinely better at using the informal proof, or whatever else. (2) The reported Semantic Correctness for powerful models like Gemini-2.5-Pro and GPT-5-mini seems anomalously low. The authors should provide a case study or a deeper analysis explaining these failures. (3) The paper dismisses Kimina-Autoformalizer for achieving 0% correctness. This is a strawman argument. These models are designed for statement formalization only and to output sorry, not end-to-end proof generation. Stating they "perform poorly" is an unfair characterization of models built for a different task.
4. The work is missing a comparison to REAL-Prover: Retrieval Augmented Lean Prover for Mathematical Reasoning (arXiv:2505.20613, 2025), which is highly relevant to this work.

### Questions
Please answer the question in Weakness.

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
4

### Summary
This paper presents ProofBridge, a unified framework for the autoformalization of NL math proofs into Lean4. It contains three main components: (1) a joint embedding model that aligns NL and FL theorems in a shared semantic space; (2) a retrieval-augmented SFT that conditions proof autoformalization; (3) an iterative proof repair mechanism that combines Lean type-checking and semantic equivalence verification. The authors curate two benchmarks: NuminaMath-Lean-PF and MiniF2F-Test-PF for evaluation. Experiment results show improvements over all-MiniLM-L6-v2 on the retrieval task and Kimina-Prover-RL-1.7B in the proof autoformalization task.

### Strengths
1. This paper presents an interesting proof repair technique by a proof autoformalizer
2. The experiment indicates the framework performs well in the 1.7B level model
3. The NuminaMath-Lean-PF contributes a 40k-level dataset of natural language to a formal language autoformalization dataset.

### Weaknesses
Despite the paper’s mentioned strengths, it requires significant revision to address the following cirtical weaknesses:

1. **Overclaiming SOTA:** My primary concern is that the paper claims to outperform previous SOTA models and position itself as the new SOTA. However, the baselines it uses for comparison are clearly not current SOTA methods in either proof retrieval or proof autoformalization tasks. For the retrieval task, all-MiniLM-L6-V2 is an outdated model in 2020. More recent and competitive baselines should be considered, such as Qwen3-Embedding-8B [1]. For the proof autoformalization task, comparing solely against the 1.7B variant of Kimina-Prover is insufficient, as the same work introduces 8B and 72B versions that significantly outperform the 1.7B variant. The overclaiming of SOTA significantly harms the objectivity of the paper. 
2. **Questionable Evaluation Methodology:** The evaluation procedure for proof retrieval is overly idealized and does not adequately reflect real-world complexity. The task assumes that for each NL input, there exists an exact corresponding FL proof in the dataset, and the goal is to simply retrieve this match. However, realistic scenarios are much more complex. For example, there may be many out-of-distribution queries and hard negative samples that require fine-grained discrimination.
3. **Incomplete experimental analysis:** The experimental section lacks the depth and rigor expected for the conference paper. It misses some important ablation studies, such as the impact of individual components, the effect of retrieval hyperparameters, the analysis of inference hyperparameters, and the impact of training data size and composition. It also lacks qualitative analysis, such as examples of successful and failed retrievals, a qualitative comparison between generated and gold-standard proofs, and case studies on the repair process.
4. **Poor paper organization and clarity:** The paper suffers from significant structural and clarity issues:
    1. Structural problems: The NuminaMath-Lean-PF dataset is first mentioned in the methodology section in line 304 without any introduction to its construction and characteristics. Readers must go to Section 5 to find its details.
    2. Unclear experiment section: The experiment section is difficult to read due to lack of clear subsection organization (such as separate subsections for retrieval results, autoformalization results, ablations, and additional studies). It also mixes the main results and ablation studies, and provides an insufficient description of experimental settings.

[1] Zhang, Y., Li, M., Long, D., Zhang, X., Lin, H., Yang, B., ... & Zhou, J. (2025). Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models. *arXiv preprint arXiv:2506.05176*.

### Questions
1. Why was Kimina-Prover-RL-1.7B chosen as the primary baseline instead of larger variants?
2. What is the out-of-distribution test results or adversarial perturbations results on the retrieval task
3. What is the performance breakdown across different mathematical domains
4. Can the authors provide comprehensive qualitative examples showing when and why the framework helps?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel "lean code generator" that translates from natural language statement-proof pairs to formalized statement-proof pairs. This is achieved by first training a joint embedding space of NL and formal proofs by contrastive learning, and then training a RAG translator by supervised finetuning that retrieves relevant formal proofs in the database according to natural language prompts. The retrieval system reports SOTA performance on the recall rate. The RAG translator achieves SOTA performance on both the semantic and compilation correctness.

### Strengths
1. The paper is well-motivated to address the gap between the power of NL and formal math provers, by translating between natural and formal languages. 
2. The retrieval augmentation has shown compelling performance in the translating accuracy compared to the SFT-only baseline.

### Weaknesses
1. Since existing baselines are limited in both joint embedding learning and FL-NL translation, the paper will benefit from an end-to-end evaluation. It's important to evaluate how SOTA NL prover's ability can be boosted in FL domain by translating their NL proofs to FL proofs. It will be interesting to see if GPT and Gemini-2.5-pro benefits from FL-NL translation and how the resulting performances compare to SOTA FL provers.

2. The translation type correctness for RAG-SFT w/o external repair from Gemini is relatively poor compared to Kimina-prover-RL.  However, the semantic correctness excels by a significant margin.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work presents a framework (ProofBridge) for end-to-end translation of entire natural language (NL) theorem-proof pairs into a formal language (Lean). ProofBridge has 3 modules:  (1) a joint embedding model trained via contrastive learning to align NL and FL representations in a shared semantic space, (2) retrieval-augmented fine-tuned LLM conditioned on retrieved FL examples, and (3) iterative proof repair using Lean's type checker and LLM-based semantic equivalence verification. This work also includes the curation of two new datasets: NUMINAMATH-LEAN-PF (38.9k NL-FL pairs) for training and MINIF2F-TEST-PF for evaluation.

### Strengths
1. Unlike AlphaProof and other systems that require manual theorem translation before proof synthesis, Proofbridge addresses the complete translation pipeline end-to-end.
2. The semantic correctness metric based on bi-directional equivalence.
3. The paper compares against 10 models like GPT-5-mini, Gemini-2.5, and specialized theorem provers (DeepSeek-Prover, Kimina-Prover), demonstrating significant improvements (+31.14% SC, +1.64% TC at pass@32 over Kimina-Prover-RL-1.7B)
4. NuminaMath-Lean-PF (38.9k pairs) provides a substantial training resource.

### Weaknesses
1. The paper dismisses FormL4 [1] as "not released," but it looks like it was released a year ago. The paper should provide a more detailed comparison of technical approaches and, if possible, empirical comparison using their published benchmark.
2. The paper shows that zero-shot actually performs better on SC (40.16% vs. 31.56%), suggesting random few-shot examples hurt performance. This makes the comparison less fair since Proofbridge always uses retrieved (relevant) examples.
3. No analysis of the judge's reliability for SC Metric, as using an LLM introduces a potential source of unreliability.
3. It is unclear how this approach would scale to other, more abstract areas like topology and category theory. These proofs are often much longer, rely on a vast library of existing lemmas, and have more complex dependency structures.

[1] Process-Driven Autoformalization in Lean 4 https://github.com/rookie-joe/PDA

### Questions
1. What quality control measures were applied to the Gemini-generated NL proofs?
2. Breakdown of errors by type (syntax errors, wrong theorem, wrong proof strategy)?
3. Performance in various math domains?
4. How does retrieval quality correlate with final translation quality?

### Soundness
3

### Presentation
3

### Contribution
3
