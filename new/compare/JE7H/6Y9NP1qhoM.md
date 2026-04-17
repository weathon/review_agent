---
job_id: 64321b0c-86a3-4530-8611-173d32ca4270
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 6Y9NP1qhoM.pdf
paper: Goal-Aware Identification and Rectification of Misinformation in Multi-Agent Systems
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper addresses misinformation robustness, dataset design, and defense mechanisms for LLM-based multi-agent systems, squarely within ICLR topics on representation learning, safety, and learning on graphs / multi-agent systems.

## Minimum Quality
Pass ✅.  
All core sections (Abstract, Introduction, Related Work, Method, Experiments, Results, Limitations, Conclusion) are present and reasonably detailed. The methodology is coherent, experimental evaluation is substantial (multiple models, attacks, defenses, ablations), and the paper is written in clear English.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no hidden instructions targeting automated reviewers or other manipulative content beyond the standard system prompts used to generate the dataset and run agents.

---

# Expected Review Outcome:

## Summary

The paper studies misinformation injection in LLM-based multi-agent systems (MAS), arguing that existing work largely focuses on overtly malicious or jailbreak-style attacks. It introduces MISINFOTASK, a task-driven dataset of 108 complex, multi-step tasks with carefully designed misinformation goals and arguments, and proposes ARGUS, a training-free two-stage defense framework that adaptively localizes critical misinformation channels in the MAS graph and rectifies them via a goal-aware corrective agent. Experiments across four strong LLMs, three injection vectors (prompt, RAG poisoning, tool), several MAS topologies, and ablations show that ARGUS substantially reduces “misinformation toxicity” (MT) and partially restores task success rate (TSR) compared to attack-only and two baselines.

## Strengths

1. **Clear problem focus on covert misinformation in MAS, rather than generic jailbreaks.**  
   The paper crisply distinguishes “misinformation” from “malicious information” in the Introduction and Figure 1 (left panel), emphasizing semantically benign but factually incorrect content that can quietly derail multi-agent reasoning. This is a meaningful and timely problem for the MAS community.

2. **Non-trivial dataset design oriented around complex multi-agent tasks.**  
   MISINFOTASK is not just QA pairs; each instance includes a realistic multi-step task, decomposition potential, a specific misinformation goal, multiple plausible misleading arguments, and ground-truth counter-facts. Section 3.1 and Appendix C make clear that tasks are constructed to require planning and collaboration (e.g., the renewable energy planning example in Figure 7 and Figure 9), which is much closer to how MAS are actually used than simple QA benchmarks.

3. **A coherent, graph-based adaptive localization mechanism that is precisely specified.**  
   The method for selecting monitored edges combines structural and semantic signals: edge betweenness centrality (Equations (2)–(4)), semantic relevance to inferred misinformation goals (Equations (5)–(7)), and message frequency (Equation (8)), aggregated into a final score driving Top‑k edge selection (Equation (9)). This part is mathematically clear and reasonably well motivated: the adaptive re-localization over rounds aligns with the goal of tracking where misinformation is actually flowing rather than statically pruning nodes.

4. **Goal-aware corrective agent leveraging CoT, not just surface-level detection.**  
   Section 4.2 describes a multi-step process where the corrective agent performs sentence-level scrutiny, “internal knowledge resonance”, and heuristic persuasive reconstruction. While some parts are prompt-level rather than formal algorithms, it explicitly operationalizes goal inference (feeding into adaptive localization) with Algorithm 1, and ties into the quantitative “goal identification accuracy” in Figure 4. This is more nuanced than simple toxicity filters or rule-based guards.

5. **Substantial empirical evaluation across models, attack types, and topologies.**  
   - **Table 1**: For four different LLMs, ARGUS consistently reduces MT and improves TSR across Prompt Injection, RAG Poisoning, and Tool Injection. For example, with GPT‑4o‑mini under Tool Injection, MT drops from 5.78 (attack-only) to 2.67 and TSR increases from 68.75% to 89.66%. This is a strong signal of robustness across model families and attack vectors.  
   - **Figure 2** quantifies the systematic degradation of MAS under attack, with TSR dropping by roughly 20 points and MT rising from ~1.3 to ~4.7, establishing that the threat is real.  
   - **Figure 6** shows ARGUS working across five MAS topologies (Self-determined, Chain, Full, Circle, Star), indicating some topology-agnostic transfer.

6. **Careful ablations isolating key design choices.**  
   - **Table 2** ablates dynamic localization, CoT-based revision, and multi-turn correction, and also includes an oracle setting with known ground truth. Each removal significantly worsens MT and/or TSR, with “w/o Dynamic Local.” particularly harming performance, suggesting the adaptive graph-based component is indeed doing useful work.  
   - **Table 3** ablates the weighting of topological, relevance, and frequency scores (α, β, γ). The performance deterioration when γ (relevance) or α (topology) is removed supports the design choice of combining these signals.  
   - Additional ablations in Tables 5–8 explore number of agents, hybrid attacks, task-type sensitivity, and cost overhead, which gives a reasonably holistic view of the system’s behavior and tradeoffs.

7. **Evaluation metrics and pipeline are explicitly defined.**  
   Equation (1) defines MT and TSR using an LLM-based semantic Score function and thresholding, and Section 3.3 plus Appendix G provide enough detail to understand how these are computed. The distinction between MT (alignment with misinformation goal) and TSR (alignment with correct task goal) is conceptually clean and discussed again in Section E.2.

8. **Figures help clarify dynamics, not just decoration.**  
   - **Figure 5** shows MT trajectories over rounds with and without ARGUS, making the “contagion” vs “containment” story intuitively clear: with attacks only, MT monotonically increases; with ARGUS, MT decreases each round across all three injection types.  
   - **Figure 3** is a good high-level schematic of the whole system: MISINFOTASK on the left, attack vectors on the right, and the central ARGUS pipeline annotating where topological importance, channel frequency, and information relevance are used, which is helpful for readers stitching together the math in Section 4.

## Weaknesses

1. **Definition and operationalization of “Misinformation Toxicity (MT)” are fragile and under-analyzed.**  
   MT (Equation (1)) is defined as the average LLM-judge semantic similarity between the final conclusion and the “misinformation goal” \(g^{k}_{mis}\). There are several issues:  
   - There is no analysis of how sensitive MT is to judge choice, prompting, or sampling variance. For instance, is MT stable if the judge is changed, or if multiple seeds are used? You repeatedly rely on single-number MT differences (e.g., reductions of ~1–2 points in Table 1) without confidence intervals or agreement checks.  
   - The Score function returns [0, 10] but the semantics are opaque: what does an MT of 4.7 actually mean in practice? Are those outputs strongly endorsing the misinformation, or just partially aligned? A qualitative breakdown or calibration (e.g., threshold-based interpretation or sanity-check examples) is missing.  
   - MT conflates presence of misinformation and its weighting in the response. A conclusion that briefly mentions but refutes the misinformation could still appear semantically similar. Without explicitly instructing the judge to consider polarity of stance, you risk mislabeling mitigated outputs as “toxic”.  
   Given that all main results are driven by MT and TSR, a deeper robustness analysis of this metric is important.

2. **Goal inference and semantic-relevance machinery are somewhat ad hoc and under-specified.**  
   The adaptive re-localization hinges on \(a_{cor}\)’s inferred goals and on embedding similarity computations (Equations (5)–(7)):  
   - It is unclear which embedding model \(\Phi(\cdot)\) is used, whether it is the same for all experiments, and whether its domain (code, math, long instructions) matches MISINFOTASK’s variety. Small changes in embeddings can affect \(\mathcal{G}'_{mis}\) and hence the monitoring set.  
   - The deduplication of inferred goals based on cosine similarity (Section 4.1.2) is conceptually reasonable but algorithmically under-specified: what threshold, what clustering policy, and are there failure modes when goals are heterogeneous?  
   - Equation (6) sets a threshold \(\theta_{sim}\) and uses max over sentences, but there is no systematic justification for \(\theta_{sim}=0.4\) beyond a brief mention in Appendix B.5. A sensitivity analysis (e.g., effect of varying \(\theta_{sim}\) on Table 3 style metrics) is missing.  
   - The “intent inference accuracy” in **Figure 4** is reported, but the exact ground truth mapping, how many goals per instance, and how partially correct inferences are scored are not explained. This makes it hard to interpret that figure or to understand how much the adaptive part really depends on accurate intent modeling.

3. **Validation of ARGUS is confined to a synthetic, model-generated dataset and simulated attacks.**  
   While MISINFOTASK is thoughtfully designed, it is entirely generated with GPT‑4o and then filtered (Appendix C), and attacks are templated Prompt / RAG / Tool injections crafted against that dataset. There are no experiments on:  
   - Human-authored misinformation scenarios or logs from real MAS applications.  
   - Existing benchmarks or red-teaming corpora for misinformation (even single-agent) to test generalization.  
   - Real-world noisy interactions where misinformation is mixed with multiple benign alternative goals.  
   This limits external validity. The claims about “real-world” robustness in the Abstract and Conclusion are therefore somewhat strong relative to the evaluation setting.

4. **Comparison baselines and ablations do not fully separate “more thinking” from “better localization”.**  
   Both Self-Check and G‑Safeguard are arguably weaker baselines for this particular task:  
   - Self-Check is applied as local self-critique, but it is not clear whether you integrated it in a way that is truly competitive (e.g., multi-round self-debate on critical messages or aggregator-level checks).  
   - G‑Safeguard is trained on a separate log dataset, but the paper does not describe how much data is used, whether the classifier capacity is appropriate, and whether hyperparameters or pruning thresholds are tuned for MT rather than generic “risk”.  
   - There is no baseline that uses the *same* number of corrective calls with random or static edge selection (e.g., “Random‑k edges with CoT correction but no adaptive scoring”), which would isolate the effect of the topological + semantic scoring versus just doing more model calls. Some of this is partially probed with “w/o Dynamic Local.” in **Table 2**, but that configuration still uses static initial localization based on betweenness, which already bakes in structure. A random-edge or uniform-edge baseline would help.  
   Without a stronger set of baselines that decouple “where” vs “how” and “how often” you call corrective CoT, it is hard to attribute the gains uniquely to the core claimed contributions.

5. **Dataset scale and diversity may be limited for robust claims.**  
   MISINFOTASK contains 108 tasks, which is modest given the complexity and heterogeneity of MAS scenarios. You report averages across all tasks but do not show variance or per-task distributions of MT/TSR under different defenses (except per-category MT in **Table 7**). Specific issues:  
   - There is no clear training/validation/test split because methods are training-free, but this also means overfitting to the dataset design and attack style is a real concern.  
   - All tasks are single-language, text-only and concentrated in a handful of cognitive categories (Table 4). It is unclear how well ARGUS would handle multimodal or code-heavy interactions where misinformation is embedded in diagrams or full programs.  
   - The dataset is claimed to be “realistic” but still templated by a single generation prompt (Figure 8). While there was human filtering, the process and criteria for acceptance / rejection are described qualitatively; inter-annotator agreement or examples of borderline cases are not provided.

6. **Some important experimental details are missing or too compressed, which hinders reproducibility and interpretation.**  
   A few examples:  
   - The exact prompts for the corrective CoT and goal inference are only referred to in Appendix G and Figure 13, but in the main text the CoT process is described at a high level. Since the method is training-free and heavily prompt-driven, the precise templates and their tuning strategy are central to the contribution and should not be relegated entirely to the appendix.  
   - The computational budget is summarized in **Table 8**, but there is no breakdown of how many calls per round per agent go to \(a_{cor}\), what the average monitored‑edge coverage is, or how these scale with number of agents and edges. This makes it hard to reason about cost–effectiveness or scalability beyond the tested 3–6 agent range.  
   - For Self-Check and G‑Safeguard, there is limited information on hyperparameters or model configurations (e.g., type/size of GNN, training epochs and loss function, exactly how nodes are labeled as high/low risk). This raises the risk that baselines are under-tuned.

7. **Certain mathematical definitions are ambiguous or incomplete.**  
   While the core equations are mostly sound, there are some rough edges:  
   - In Equation (2), the normalization term \(N_{norm}\) is vaguely defined as “a normalization factor” without specifying whether it equals the number of ordered pairs \(i\neq j\), or some other constant. This matters when comparing edge scores across graphs or different sizes.  
   - In Equation (8), \(\texttt{Score}_{freq}^{r-1}(e) = \texttt{count}(m_e(r))\) appears to use \(r\) rather than \(r-1\); this is likely a typo but creates ambiguity in implementation. Consistency across notations \(m_e^{r-1}\), \(m_e(r)\) is not fully maintained.  
   - The final composite score \(\texttt{Score}^r(e)\) is said to be a weighted sum of \(\texttt{Score}_{topo}\), \(\texttt{Score}_{rel}\), \(\texttt{Score}_{freq}\), but there is no explicit equation in the main text, nor upper/lower bound normalization. As a result, it is unclear whether frequencies on edges with many messages could swamp topological or relevance terms.  
   - Equation (4) and Algorithm 1: for \(r=R\), \(\mathcal{E}_{R+1}\) is still computed but never used, which is fine in practice but should be acknowledged.

8. **Claims about generality and “training-free unified shield” are somewhat overstated.**  
   Although ARGUS requires no gradient-based training, it is heavily specialized to the MISINFOTASK threat model via prompt design (Figure 13), choice of \(\theta_{sim}\), and the structure of the scoring function. The paper sometimes implies stronger universality than is warranted by the empirical scope, for example in the Conclusion (“high generalization in countering diverse threats”), while all “diverse threats” are still text misinformation injected in prompt/RAG/tool for similar MAS architectures and LLM families. It would be more accurate to qualify that this is within the tested class of multi-agent LLM settings.

9. **Limited discussion of potential failure modes and robustness against adaptive adversaries.**  
   The attacker in Section 3.3 is restricted to compromising a single agent or knowledge base; there is no exploration of adversaries who (i) adaptively move injection points after observing ARGUS behavior, (ii) craft misinformation to be semantically orthogonal to previous inferred goals but still harmful, or (iii) target the corrective agent itself (e.g., by poisoning messages that feed into goal inference). Hybrid attacks in **Table 6** are a step forward, but they do not represent strategic adaptation. A discussion of how ARGUS might fail or be evaded would make the contribution more credible.

## Potentially Missing Related Work

1. **A. Gautam, “Multi-agent Systems for Misinformation Lifecycle: Detection, Correction And Source Identification”, 2025.**  
   This work reportedly proposes a multi-agent framework covering detection and correction of misinformation, which is directly relevant to MISINFOTASK and ARGUS’s multi-agent correction pipeline. It should be discussed in Section 6 (MAS Information Injection / MAS Defense Strategies) and compared in terms of system architecture and extent of lifecycle coverage (ARGUS focuses on in‑flight rectification rather than full source tracing).

2. **K. Lakara, G. Channing, J. Sock, “LLM-Consensus: Multi-Agent Debate for Visual Misinformation Detection”, 2024.**  
   While focused on visual misinformation, this paper uses multi-agent debate/consensus for misinformation detection, conceptually close to your use of a corrective agent and goal-aware inference. It should be cited next to Chern et al. (2024) in Related Work and briefly contrasted: debate-based vs. topological localization and single-corrector approaches.

3. **Z. Yu, Z. Ying, Y. Dai, “RAMA: Retrieval-Augmented Multi-Agent Framework for Misinformation Detection in Multimodal Fact-Checking”, 2025.**  
   RAMA leverages multi-agent collaboration and retrieval augmentation specifically for misinformation detection. It is important to mention in Section 6 when discussing datasets and systems for misinformation detection, especially since you also use retrieval-based attacks and emphasize retrieval-augmented MAS. A short comparison of RAMA’s retrieval‑centric strategy with ARGUS’s goal-aware graph-localization would strengthen positioning.

## Questions

1. **On MT metric robustness:**  
   Could you provide (in the rebuttal or camera-ready) a calibration or robustness study of the MT metric? For example, report inter-run variance across multiple judge seeds, or sensitivity to the choice of judge model and prompt. Even a small study on, say, 20 tasks would help assess whether observed 0.3–0.5 point differences are meaningful.

2. **On semantic embeddings and thresholds in localization:**  
   Which embedding model is used for \(\Phi(\cdot)\) in Equations (5)–(7)? How sensitive are the results in Table 3 to the similarity threshold \(\theta_{sim}\) and to embeddings choice? Concrete numbers exploring 2–3 alternative configurations would increase confidence that ARGUS’s gains are not hyperparameter brittle.

3. **On localization vs. correction strength:**  
   Have you tried a baseline where \(a_{cor}\) is placed on a random set of \(k\) edges (constant across rounds) but still runs the same CoT + heuristic rectification? This would help isolate how much of Table 1’s improvement is due to your adaptive scoring versus simply having more “thinking checkpoints”.

4. **On generalization beyond MISINFOTASK:**  
   Do you have any preliminary experiments (even small-scale) on human-authored or out-of-distribution tasks, such as existing fact-checking datasets converted into MAS workflows, or logs from any internal multi-agent applications? Evidence that MT reductions transfer, even partially, would significantly strengthen the impact.

5. **On attacker adaptivity and poisoning of \(a_{cor}\):**  
   Given that \(a_{cor}\) relies on LLM knowledge and CoT, how would ARGUS behave if the compromised agent targeted the corrective agent’s input channels directly (e.g., by feeding misleading meta-statements about the supposed goal)? Do you envision additional safeguards or redundancy (e.g., multiple corrective agents) to make ARGUS robust to such second-order attacks?

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The methodology is conceptually sound, equations are mostly consistent, and experiments are broad with reasonable ablations; remaining concerns are about metric robustness, dataset external validity, and somewhat underdeveloped baselines, not fundamental errors.

## Presentation Rating

3: good.  
The paper is generally well written, with helpful figures (especially Figures 1–3, 5–6) and clear problem framing; some crucial implementation specifics (metrics calibration, embedding choices, and prompt details) could be surfaced more prominently from the appendix.

## Contribution Rating

3: good.  
The combination of a task-driven MAS misinformation dataset and a graph-based, goal-aware training-free defense framework is a meaningful contribution to the MAS safety area, though current evidence is limited to synthetic tasks and one threat model.

## Overall Rating

8: Accept, good paper (poster).  
Despite some concerns about metric robustness, dataset realism, and baseline strength, the paper offers a well-executed and timely study of a clearly important problem, with a reasonably principled method and extensive experiments that should be valuable to the MAS and safety communities.

## Reviewer Confidence

4: confident.  
I am familiar with LLM-based MAS, agent safety, and graph-based defenses, and I carefully checked the core equations and experimental design, though I did not attempt to replicate results.