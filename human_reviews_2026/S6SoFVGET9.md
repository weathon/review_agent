# Tiny Moves: Game-based Hypothesis Refinement

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Scientific discovery is an iterative process, yet most machine learning approaches treat it as an end-to-end prediction task, limiting interpretability and alignment with scientific reasoning workflows. We introduce The Hypothesis Game, a symbolic, game-based framework where a system of agents refines hypotheses through a fixed set of reasoning moves (a reasoning grammar). Inspired by the idea that scientific progress often relies on small, incremental changes, our framework emphasizes “tiny moves” as the building blocks of incremental hypothesis evolution. We evaluate the approach on pathway-level reasoning tasks derived from Reactome, focusing on reconstruction from partial cues and recovery of corrupted hypotheses. Across 820 reconstruction and 2880 corruption experiments, it matches strong prompting baselines on reconstruction and achieves superior precision and error recovery in corruption. Beyond accuracy, it produces concise, interpretable hypotheses and enables controllable reasoning, highlighting the potential of game-based reasoning for accelerating discovery across the sciences.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Hypothesis Game, a symbolic, game‑based framework in which LLM agents refine a shared hypothesis via a fixed grammar of small “tiny moves” (actions). Hypotheses are represented as sets of fragments, and the experiments primarily use structured text. The implemented move set comprises prune, expand, expand_with_corpus (retrieve + integrate), and debate, orchestrated by a central LLM controller called the Game Master. Modes guide move selection through prompt instructions in the prototype, and a scoring formalism is defined but not used to drive control. Two game variants are specified: Simple Hypothesis Refinement (whole‑state edits) and Localized Hypothesis Refinement (fragment‑level edits with consistency enforcement), each given as an algorithm.

The evaluation focuses on pathway‑level reasoning from the Reactome dataset through two tasks: 1. reconstruction from partial cues and 2. corruption recovery. The study samples 100 pathways for reconstruction and 20 for corruption, running 820 and 2880 experiments, respectively. The authors compare the proposed one with baselines: Zero‑Shot, Chain‑of‑Thought, and ReAct; metrics combine entity‑level precision/recall/F1 via Gilda mapping and an LLM‑as‑judge measure of reaction‑level “Detailed Recall.” In reconstruction, all methods struggle; the Hypothesis Game is comparable to ReAct, with text noting ReAct’s slightly higher F1.
In corruption recovery, the Hypothesis Game could remove more errors and achieves the highest precision and F1 while maintaining recall.

### Strengths
- The writing is very clear and very easy to follow. I like the way the authors illustrate the key idea and its association with human science development. 
- The high-level idea makes sense. This work formalizes hypothesis refinement as a compositional reasoning game with a reusable grammar of moves, enabling transparent trajectories and controllable reasoning styles. 
- The approach demonstrates relatively good performance in corruption recovery, combining higher error removal with the high precision and F1 while maintaining recall.
- The paper releases curated datasets on Hugging Face for reproducibility.
- Related works are generally sufficient there.

### Weaknesses
- The high-level idea of employing LLM for action planning given a set of operators is not very novel.  I am not very sure about the significance of the novelty/technical contribution of this paper - currently im not very positive. 
- Consider adding more comparative agent frameworks as baselines. Also, the experiment scope is constrained to Reactome human pathways and an open‑access biomedical corpus, which is a bit limited.
- The performance is not "that" good on such a scope of experiment. In the reconstruction task, ReAct actually still slightly outperforms the Hypothesis Game in F1, and all methods show low precision and recall. 
- Is an LLM‑as‑judge for reaction‑level correctness reliable?
- The method needs \(k_{\max}\) and termination criteria, but the experiments vary only move availability and corpus access. The move‑selection prompt instructs “run at least 20 rounds,” which interacts with traceability and cost but is not ablated. 
- Minor: Fig.1 in the appendix is not clear. In the appendix: "The moves used in this section correspond to the moves presented in table ??.", please fix this..

### Questions
Besides the question raised above.

- The reaction‑level metric uses an LLM‑as‑judge; can you report any calibration or agreement checks to assess the judge’s reliability?

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
2

### Summary
The paper introduces The Hypothesis Game, a symbolic, game-based framework for iterative hypothesis refinement using a small reasoning grammar ("tiny moves") applied to a shared hypothesis state \(H_t\). The formalism includes game modes (via \(\pi_M\)) and an optional scoring vector \(S(H_t)\), with two algorithmic variants for whole-state and localized edits. Experiments target pathway-level reasoning on Reactome: (i) reconstruction from partial cues and (ii) recovery from corrupted hypotheses. Using entity-level metrics (via Gilda) and an LLM-as-judge metric for reaction fidelity, the method matches strong prompting baselines on reconstruction and achieves higher precision and F1 on corruption, across 820 reconstruction and 2,880 corruption trials.

### Strengths
- The paper formalizes hypothesis refinement as iterated transformations under a move budget, providing a crisp, compositional semantics. This clarity aids reuse and analysis. 

- A small set of moves (prune, expand, retrieve, debate) is specified and mapped to agent responsibilities. This fosters generality while remaining practical. The grammar supports composition and repeated application, encouraging modular reasoning and stepwise traceability. Conceptual visualization (Fig. 1) improves readability for non-experts and illustrates extensibility to graph settings.

- The "Game Master" controller and specialized agents are described, with moves and diagnostics summarized in Table 1. Modes are approximated via prompt conditioning (Sec. 3), clarifying how theory is realized in practice.

- Evaluation tasks are well-motivated. Two tasks (reconstruction and corruption recovery) are well aligned with incremental hypothesis evolution (Sec. 4). Entity fidelity uses standardized mapping (Gilda) and reaction fidelity uses an LLM-as-judge checking inputs/outputs/directionality/interaction type. This provides complementary views. Dataset size and split are substantial for a pilot.

- The method achieves the highest precision and F1 in corruption recovery across error types and rates, showing the value of targeted edits. On reconstruction, performance is close to ReAct and better than Zero-Shot/CoT on precision, indicating disciplined expansion helps avoid spurious content.

### Weaknesses
- Scope of baselines and diagnostics is narrow. Baselines are limited to prompting (Zero-Shot, Chain-of-Thought, ReAct), omitting other structured controllers or graph-based planners, which constrains external validity. No ablation on the move grammar (e.g., removing debate or retrieve) or on the move budget to isolate which components drive gains.

- Formal elements are not fully operationalized. Modes \(\pi_M\) are implemented via prompt text rather than as an explicit policy; scoring \(S(H_t)\) and \(U(H_t)\) are defined but not used to drive control. Optimization is handled by the controller’s "Diagnose" component without metric-driven selection, weakening the empirical link between the formalism and gains. No experiments vary \(\beta\) or demonstrate policy learning from scores, leaving the control levers untested.

- LLM-as-judge reliability and human oversight are under-specified. Reaction-level fidelity relies on an LLM judge*with no reported agreement against human adjudication or across multiple judges. Although two experts review corruption generation, inter-annotator agreement is not reported.

- Potential leakage/contamination risks are not controlled. The authors note that some pathways are "relatively well known", which may favor models with prior knowledge or retrieval, but no time-based or contamination controls are described. Retrieval-based moves can draw on external corpora, yet there is no ablation on retrieval sources or temporal cutoffs. No assessment is given for how performance changes when restricting to knowledge post-dates or unseen pathways.

### Questions
See Weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces "The Hypothesis Game", a symbolic, game-based framework for hypothesis refinement in scientific discovery. The method formalizes reasoning as a set of discrete operations ("tiny moves") such as prune, expand, retrieve, and debate, applied iteratively to structured hypotheses. Evaluation on pathway-level reasoning tasks derived from Reactome shows that the approach performs comparably to strong prompting baselines in hypothesis reconstruction and outperforms them in error correction.

### Strengths
1) The paper provides a principled and interpretable alternative to free-form LLM reasoning.
2) The reasoning grammar is well motivated and modular, allowing transparent control of reasoning styles and easy extensibility.
3) The experimental setup is carefully designed, including both reconstruction and corruption tasks on Reactome pathways.

### Weaknesses
In my opinion, the main weakness of the paper is that the scoring formalism proposed in Section 2.4 remains theoretical and untested.
Other than that:
1) The evaluation lacks any assessment of statistical significance across multiple runs. This limits confidence in the reported performance differences.
2) Biological validation of the refined hypotheses is absent. While metrics are informative, there is no verification that reconstructed or corrected pathways remain biologically plausible or mechanistically consistent.

### Questions
I would ask the authors to address the weaknesses highlighted above. Particularly, the authors should either implement one scoring-driven variant to illustrate how metrics could guide reasoning trajectories, or provide more explanations on why  scoring-driven variants are left out from the current work.
Moreover:
1) Please assess the statistical significance of the reported performance differences between The Hypothesis Game and the baselines.
2) Include at least a few biologically grounded evaluation, such as checking whether (a randomly chosen subset of) reconstructed pathways preserve known functional or causal relationships.

### Soundness
3

### Presentation
3

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
The paper introduces The Hypothesis Game, a game-based framework in which LLM agents refine scientific hypotheses through small, structured reasoning steps defined by four operations: prune, expand, retrieve, and debate. It models hypothesis refinement as an iterative process guided by a fixed reasoning grammar. The approach is evaluated on biological pathway reasoning tasks from Reactome, including reconstruction and corruption recovery. In experiments, it performs comparably to strong prompting baselines on reconstruction and achieves higher precision and F1 scores on corruption recovery.

### Strengths
(1) The paper proposes a novel framing of scientific reasoning as a game with explicit, interpretable moves, offering an original perspective on hypothesis refinement.

(2)The paper is well organized and clearly written, making a complex concept easy to follow.

### Weaknesses
* The scope is limited to biological pathways; it is unclear how the framework performs in other scientific domains.
* The controller is still prompt-based, not a learned or metric-driven system, which makes the implementation less autonomous.
* The move set is small (only four operations), which may limit creativity and open-ended discovery.
* The evaluation relies partly on LLM judgments and does not include human or symbolic verification.

### Questions
* The scoring function $U(H_t)$ is defined but unused. It would be useful to clarify what prevented its use and how explicit scoring might affect the controller’s reasoning.
* The paper often refers to 'game', but the implementation appears to be prompt-based rather than a true multi-agent or turn-based game. Clarification on whether it is primarily a conceptual framing or includes actual game dynamics such as turns, scoring, or agent interaction would strengthen understanding.
* How consistent is move selection across runs? An evaluation of the stability of this prompt-based control would be informative.
* The framework employs four core operations (prune, expand, retrieve, debate). Further explanation of why these specific operations were chosen and whether adding or removing move types affects performance would clarify the design space.
* The paper mentions that pathway descriptions were rephrased "to avoid memorization," but the procedure is not detailed. It is unclear who performed this rephrasing (humans or LLMs) and how semantic accuracy was verified would improve clarity and reproducibility.
* How was semantic equivalence verified after rephrasing, and could rewording introduce ambiguity or loss of information?
* Did the two experts agree with each other when reviewing the corruptions?
* Did those same experts also check whether the model’s corrected hypotheses were actually right after the experiments?
* Figure 2 -- Qualitative Example: The example compares incremental vs. single-step edits. How was "minor additional change" quantified? Is it possible to use token-level or entity-level edit distances?
* The framework aims to support hypothesis discovery, but its refinement process prioritizes consistency with known data. Could there be a discussion on whether iterative editing might reduce the novelty of initially creative hypotheses, effectively converging toward familiar knowledge rather than exploring new ideas?

### Soundness
2

### Presentation
2

### Contribution
2
