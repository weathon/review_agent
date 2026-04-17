---
job_id: 2049df18-10cc-453b-bf36-ece1a43ccb02
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: lSM6MtjQcM.pdf
paper: AetherCode: Evaluating LLMs’ Ability to Win In Premier Programming Competitions
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper introduces a code-reasoning benchmark for LLMs, which fits squarely under “datasets and benchmarks” and “representation learning for language/programming” within ICLR’s scope.

## Minimum Quality
Pass ✅.  
The paper is in English and has all required components: Abstract, Introduction, Benchmark Curation/Method (Section 2), Experiments / Results (Section 3), Related Work (Section 4), and Conclusion (Section 5). The methodology and experiments are reasonably detailed and technically sound; there are no obvious fatal theoretical or experimental flaws.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no signs of prompt injection, hidden instructions, or manipulation aimed at automated reviewers.

---

# Expected Review Outcome:

## Summary

The paper introduces **AetherCode**, a competitive-programming benchmark designed to evaluate LLM reasoning and coding skills on problems drawn from premier contests (IOI, ICPC, USACO, etc.) from 2024–2025. It details a curation pipeline (Figure 1) that converts contest PDFs to Markdown+LaTeX, assigns rich algorithm/difficulty metadata, and constructs high-coverage test suites via a hybrid Generator–Validator (G‑V) agent plus extensive expert curation, with evaluation framed via TPR/TNR on a large set of human submissions. The authors benchmark 17 state-of-the-art reasoning and non-reasoning LLMs (Table 3, Table 4), show that even top models solve only a minority of problems, and analyze failure modes and performance across algorithmic categories.

## Strengths

1. **Clear, well-executed benchmark construction pipeline**

   - Figure 1 on Page 3 gives a concise end-to-end view of the workflow: statement processing, expert difficulty/algorithm tagging, and test case construction via G‑V agent plus human experts and elite auditors. This figure is genuinely useful: each block corresponds to a concrete step later fleshed out in Sections 2.1–2.3, and the link from “Submissions” to “Accuracy Check” underlines that test-case design is driven by solution coverage, not just random input generation.
   - The benchmark is well scoped: 456 problems, all from high-level OI/ICPC/USACO/CCPC etc. (Table 7; Page 19–20). Table 2 specifies dataset characteristics, including year distribution (400 problems from 2024, 56 from 2025), origin (76 OI vs. 380 ICPC), and average number of test cases (47.15), which supports the claim of recent, challenging problems with reasonably rich evaluation.

2. **Strong test-case quality story with explicit metrics**

   - Section 2.3 is one of the most substantive parts of the paper. The authors explicitly move beyond “more tests is better” and define test-suite quality as its ability to distinguish correct vs. incorrect solutions. Equations (1) and (2) formalize TPR and TNR as:
     \[
     \text{TPR} = \frac{\text{Passed Correct Solutions}}{\text{Correct Solutions}}, \quad
     \text{TNR} = \frac{\text{Rejected Incorrect Solutions}}{\text{Incorrect Solutions}}.
     \]
     This is a clean, concrete mathematical framing, and the subsequent analysis around these two metrics is significantly clearer than the typical vague “stress tests” narrative.
   - The combination of a **G‑V Agent System** (Section 2.3.2) with expert augmentation (Section 2.3.3) is compelling: the automatic phase achieves 89.9% TNR and 100% TPR, and then 67 CF-rated experts plus an elite team of ICPC gold medalists systematically “hack” remaining incorrect solutions until reported TPR/TNR reach 100% on the collected set. This two-stage process is a meaningful methodological contribution on how to *actually* curate high-quality test suites.

3. **Breadth and structure of the benchmark**

   - Table 1 provides a crisp comparison to existing benchmarks (HumanEval, MBPP, APPS, USACO, CodeContests, LiveCodeBench, CodeELO, LiveCodeBench Pro) in terms of difficulty, number of problems, update policy, test case construction, and source. AetherCode is one of the few that is both (i) high difficulty (★★★) and (ii) continuously updateable (✓), while relying on “G‑V Agent & Experts” for tests and sourcing from “Premier Contests” rather than a single platform like CodeForces.
   - The taxonomy in Section 2.2, with 10 primary categories and 144 detailed tags (Table 6), plus difficulty segmentation, is quite thorough. Figure 2 summarizes difficulty and category distributions, revealing that AetherCode is not dominated by one or two categories: for example, Basic (222), DP (110), Math (90), DS (120), Graph (54), Tech (147), Tree (24). This structured coverage is valuable for diagnosing model strengths/weaknesses and for future controlled studies.

4. **Comprehensive and informative experimental evaluation**

   - Table 3 (Page 8) presents Pass@1/2/4 statistics for 11 reasoning and 6 non-reasoning models, broken down by problem difficulty and year. There is a clear and sizable gap between top-tier reasoning models (o4‑mini‑high at 35.5% Pass@1 overall) and non-reasoning models (e.g., GPT‑4.1 at 10.5% Pass@1). The difficulty breakdown shows that even o4‑mini‑high only gets 8.0% on “Hard” and 3.8% on “Extreme”, directly supporting the paper’s core claim that current LLMs are far from IOI/ICPC gold-medalist level.
   - Table 4 drills down by algorithmic category (Basic, Search, DP, Strings, Math, DS, Graph, Geo, Tech, Tree) for Pass@1. The pattern that all models are relatively stronger on Basic and Strings but struggle with Geometry and Trees is evident; for instance, o4‑mini‑high obtains 38.1% on Basic and 35.6% on Strings, but only 27.1% on Geometry and 7.3% on Trees. This category-level view, enabled by the metadata in Figure 2 and Table 6, is one of the main payoffs of the benchmark design.
   - Failure analysis in Table 8 (Page 20) is also useful: it quantifies the distribution of Wrong Answer, Time Limit, Runtime Error, and Compile Error across models. The observation that Claude’s thinking models have a much larger Time Limit fraction (~50%) than others aligns with the qualitative analysis in Section 3.3 and yields concrete, actionable insights for model developers.

5. **Timeliness and decontamination focus**

   - The dataset covers contests from 2024–2025 (Table 7) and explicitly records dates for decontamination analysis (Section 2.1, “Metadata”). This addresses a common criticism that many benchmarks recycle old contest problems that are likely to be in training sets.
   - Splitting performance by year in Table 3 (columns “2024” vs. “2025”) is a simple but important design: it allows future work to track model performance on newly-added, presumably less-contaminated problems.

6. **Clarity and readability**

   - The exposition is generally clear and well-structured. Section 2 is logically ordered (problem collection → categorization → test cases), Section 3 follows with main results, category-wise results, and failure analysis, and the related work section is reasonably comprehensive on benchmarks and competitive-programming evaluations.
   - Figures and tables are consistently referenced and interpreted in the text. Figure 1 and Figure 2 are not mere decoration; they anchor the explanation of the curation process and dataset statistics.

## Weaknesses

1. **Validation of “100% TPR/TNR” is constrained by the collected solution set**

   - The paper repeatedly claims 100% TPR and 100% TNR on test suites (Section 2.3.1, Page 6: “we have achieved a 100% TPR and 100% TNR on our collected solution set”). However, Equations (1)–(2) define TPR/TNR strictly on the *finite* set of curated solutions (at least 5 correct and 20 incorrect per problem). This is essentially a finite-sample empirical estimate; it does not imply that the tests can distinguish *all* future incorrect solutions, especially adversarial LLM outputs with non-human-like error patterns.
   - Because the authors emphasize this as “a high standard for test cases,” they should more clearly acknowledge that the TNR metric is only as good as the diversity of the incorrect solution corpus. The current description does not quantify how representative these incorrect solutions are (e.g., how many unique algorithmic ideas or distinct bug types per problem, or any coverage metrics). This applies especially to the long tail of problems with “fewer than 50” incorrect solutions, where they resort to elite manual audits. Without at least some statistics or examples of distinct failure modes, the 100% TNR can be misleadingly strong.

2. **Risk of overfitting test cases to the known incorrect solutions**

   - The test-case construction procedure (Appendix C, Section 2.3.3) explicitly includes steps where experts “write various incorrect and inefficient solutions to verify the comprehensiveness of the test cases” and then extend tests until every such solution is hacked. This is a valuable practice from a contest-setter perspective, but from a *benchmark* perspective it raises the possibility of overfitting the test suite to a fixed model of error types.
   - In other words, if experts generate a particular family of incorrect patterns and tests are optimized to kill exactly those, the suite might still be fragile to very different failure modes from future LLMs. The paper does not attempt any *out-of-sample* validation, for example: generate new families of incorrect solutions after the test suite is frozen, or hold out some incorrect solutions during case construction and later measure TNR on them. Given how prominently the 100% numbers are advertised, some form of held-out or cross-validation style test would significantly strengthen the claim.

3. **Limited quantitative analysis of test-case robustness beyond aggregate TNR**

   - Apart from the single reported number “G‑V agent phase TNR = 89.9%, TPR = 100%” and then “after human augmentation, 100%/100%” (Section 2.3.2 & 2.3.3), there is no breakdown of test-suite quality across difficulty levels, categories, or contest sources. For example, are Geo/DP/Tree problems systematically harder to cover with the automatic generator? Does the expert augmentation disproportionately affect some categories?
   - Table 5 (Page 17) gives difficulty distributions per category, but there is no analogous table for test coverage quality or for the relative contribution of automatic vs. manual tests by category. A category-wise or difficulty-wise TNR breakdown would both validate the construction pipeline and illuminate where automatic methods are weak. Without such analysis, the “G‑V agent + expert” story reads strong conceptually but is thin on diagnostic evidence.

4. **Positioning against closely related benchmarks that also target competitive programming**

   - The related work section under “Code Reasoning Benchmarks” (Section 4.2) cites several recent efforts (USACO Bench, LLM-Pros, OJBench, ICPC-Eval). However, the comparison is mostly qualitative and at a high level; there is no table analogous to Table 1 that directly contrasts AetherCode to **USACO Bench**, **OJBench**, and **ICPC-Eval** on statistics like time span, number and diversity of contests, test-case accessibility, and decontamination risks.
   - Since these benchmarks also focus on contest problems and in some cases use official judges (ICPC-Eval, OJBench) or USACO’s public tests, it would be helpful to see a more rigorous argument about what is *substantively* new: scale, recency, breadth of contests, and especially test-case self-containment vs. reliance on external judges. Right now the “to our knowledge, AetherCode is the first to comprehensively collect latest problems from premier competitions” claim (Section 4.2) feels somewhat overstated without a quantitative side-by-side.

5. **Experimental protocol details could be clearer, especially around sampling and prompts**

   - The main evaluation prompt is only specified in Appendix A as: “Please solve the following programming problem using {LANGUAGE}. Please place your final answer in a markdown code block. {STATEMENT}”. There is no explicit description in the main text of sampling parameters (temperature, top‑p, stop criteria) for Pass@N, nor a discussion of whether these parameters are optimized per model or fixed globally.
   - Since Table 3’s Pass@2 and Pass@4 results underpin the “Top-Tier Models Exhibit Great Exploration Potential” conclusion (Page 8), the absence of detailed sampling settings affects interpretability. For example, the 13.3% increase in Gemini‑2.5‑Pro’s Pass@4 over Pass@1 could partly be due to using a relatively high temperature; one cannot tell whether some non-reasoning models suffer from overly conservative decoding. These details are important for reproducibility and fair comparison.
  
6. **Limited human-level baselines or connection to contest difficulty metrics**

   - The paper motivates AetherCode as “gap to top-tier human competitors,” but provides no explicit quantitative comparison to human performance on these 456 problems. For instance, using contest scoreboards, one could report the average number of problems solved by finalist teams, or the fraction of problems solved in-contest, and then compare to LLMs at the problem level.
   - Difficulty segmentation (Figure 2 and Table 5) is said to be based on “number of participants who solved them” plus expert judgment, yet no explicit statistics are given: e.g., what median human solve-rate corresponds to Easy vs. Medium vs. Hard? Without at least a couple of illustrative numbers or problem-level examples, the claimed “human-centric” notion of difficulty remains qualitative.

7. **Limited ablations on evaluation choices (Pass@N, time limits, environment)**

   - The environment is fixed to C++17 with O2, 2 cores, 4 GB memory, time limits taken from original contests (Appendix A). However, there is no discussion of how sensitive results are to environment changes or time limits. For some problems, original time limits might have been set assuming significantly faster or slower CPUs than the 3.8 GHz host; this especially matters for Time Limit Exceeded statistics in Table 8.
   - Similarly, while the authors show Pass@1/2/4, it would be informative to see at least one curve of performance vs. number of samples for a top reasoning model and a non-reasoning model. This would clarify whether the apparent “exploration potential” saturates quickly or continues to rise, and whether 4 samples is enough to characterize model behavior.

8. **Some missing directly related work on LLMs and competitive programming**

   - While the paper covers many benchmarks, it omits several recent works that also evaluate LLMs on competitive programming or sophisticated programming tasks, which would strengthen positioning:
     - Wei et al., *Evaluating and Improving Large Language Models for Competitive Program Generation* (2025), studies LLM performance on competitive programming and improvement strategies; it should be compared in Section 4.2 and possibly referenced in Section 3 when discussing failure modes (since that paper likely analyzes similar errors).
     - Dumitran et al., *Evaluating the Performance of Large Language Models in Competitive Programming: A Multi-Year, Multi-Grade Analysis* (2024), presents a multi-year dataset of contest problems and LLM evaluation; it belongs in Section 4.2 as a direct precedent in terms of using contest archives.
     - Raihan et al., *On the Performance of Large Language Models on Introductory Programming Assignments* (2025), while lower-level than AetherCode, is relevant for framing the “from introductory to premier contest problems” difficulty spectrum and could be cited in the introduction (Page 1–2) as part of the broader narrative on LLM programming ability.

9. **Minor clarity issues**

   - Table 3’s “Pass@N” column labels (1, 2, 4) are only explained in the caption; the main text sometimes uses “Puss@4” (typo, Page 7) which should be corrected.
   - There are occasional duplicated references (e.g., “Comanici et al., 2025” cited twice for different works; this is a bit confusing but not critical).
   - The notation in Section 2.2 for difficulty segmentation says “four levels: Easy, Medium, Hard, Extreme” then states later that they divide into “three roughly equal categories: Easy, Medium, and Hard.” Clarifying the exact partitioning and the role of Extreme (which appears separately in Figure 2 and Table 5) would help.

Overall, none of these issues are fatal, but they collectively suggest that the benchmark and methodology, while strong, could be more carefully validated and explained, especially with respect to test-case generalization and positioning.

## Potentially Missing Related Work

1. **Wei, M., Li, Z., Chen, X. (2025). “Evaluating and Improving Large Language Models for Competitive Program Generation.”**  
   - Relevance: Directly evaluates LLMs on competitive-programming tasks and proposes improvement strategies, making it closely related to AetherCode’s evaluation focus.  
   - Suggested incorporation: Discuss in Section 4.2 as part of “Code Reasoning Benchmarks” / competitive-programming evaluations, and briefly contrast their problem sources, test suites, and evaluation methodology with AetherCode’s approach. It may also be worth referencing in Section 3.3 when analyzing failure modes.

2. **Dumitran, A. M., Badea, A. C., Muscalu, S.-G. (2024). “Evaluating the Performance of Large Language Models in Competitive Programming: A Multi-Year, Multi-Grade Analysis.”**  
   - Relevance: Uses contest problems over multiple years to analyze LLM performance, which is thematically very similar to AetherCode’s focus on recent premier contests and year-wise analysis (Table 3).  
   - Suggested incorporation: Add to Section 4.2 with explicit comparison on contest coverage (types of contests, grade levels, years) and on whether they provide open-source test suites; potentially mention in the introduction as prior evidence that competitive-programming remains a challenging setting.

3. **Raihan, N., Goswami, D., Puspo, S. S. C. (2025). “On the Performance of Large Language Models on Introductory Programming Assignments.”**  
   - Relevance: Addresses LLM performance on simpler programming tasks, providing a lower bound of difficulty that contrasts with AetherCode’s premier contests.  
   - Suggested incorporation: Briefly acknowledge this work in Section 4.1 or the introduction, to better situate AetherCode as pushing beyond both basic benchmarks (HumanEval/MBPP) and classroom-level tasks, forming a gradient of difficulty across programming benchmarks.

*(Hossain et al., 2025; LLM-ProS) is already cited in Section 4.2, so it is not missing.)*

## Questions

1. **Representativeness of incorrect solution corpus**  
   - How many *distinct* incorrect solutions per problem do you have on average, and can you categorize them (e.g., wrong algorithm, off-by-one, corner case, TLE due to complexity, memory issues)? Some statistics or examples would help assess how meaningful 100% TNR is.  
   - Did you consider holding out a subset of incorrect solutions during test construction and then reporting TNR on that held-out subset to estimate out-of-sample robustness?

2. **Effect of expert-authored incorrect solutions on test overfitting**  
   - When experts write “various incorrect and inefficient solutions” (Section 2.3.3), do they do so based on prior knowledge of the current test suite (i.e., actively trying to hack it), or is this done before tests are generated? Could you clarify whether there is any explicit procedure to prevent overfitting the tests to a fixed set of failure patterns?

3. **Human-level baselines and difficulty calibration**  
   - Do you have access to per-problem solve counts or scores from contest leaderboards? If so, can you share at least high-level statistics such as the fraction of teams that solved Easy/Medium/Hard/Extreme problems? It would help to quantitatively substantiate the human-centric difficulty labels in Figure 2 and Table 5.  
   - Are there any problems labeled “Extreme” that *no team* solved on-site but that some LLMs can now solve? That would be a particularly compelling data point.

4. **Sampling and decoding parameters for Pass@N**  
   - What are the exact decoding parameters (temperature, top‑p, etc.) for Pass@1/2/4 across all models? Are they fixed globally or tuned per model?  
   - Did you run any sensitivity analysis to confirm that the observed trends (e.g., larger Pass@4 gain for top models) are robust to reasonable changes in these parameters?

5. **Test-suite statistics by category/difficulty**  
   - Can you report the average number of test cases per problem broken down by difficulty and category (e.g., Basic vs. Geometry vs. DP)? This would help understand whether certain categories inherently require more complex tests and whether your automatic generator struggles more on them.

Clarifying these points would increase confidence in the strength and generality of the benchmark.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The benchmark construction and experiments are generally methodologically sound, with a solid test-case generation pipeline and careful evaluation. The main caveat is that the 100% TPR/TNR claim is confined to the collected solution set and would benefit from more explicit out-of-sample robustness checks.

## Presentation Rating

3: good.  
The paper is clearly written, logically structured, and supported by informative figures and tables (notably Figure 1, Figure 2, Table 1, Table 3, Table 4, and Table 8). A few minor clarity and typographical issues remain, but they do not significantly hinder understanding.

## Contribution Rating

3: good.  
AetherCode is a meaningful addition to the ecosystem of code-reasoning benchmarks: it focuses on very recent premier contests, offers rich metadata, and invests heavily in test-case quality. The conceptual novelty is moderate, but the execution and scale make it an impactful resource.

## Overall Rating

8: Accept, good paper (poster).  
The benchmark is timely, well-motivated, and carefully constructed, with extensive expert involvement and informative evaluations across many top LLMs. While I would like to see stronger validation of test-suite generalization and sharper positioning versus other contest-based benchmarks, the work is clearly above the bar as a valuable dataset/benchmark contribution for ICLR.

## Reviewer Confidence

4: confident.  
I am familiar with competitive-programming-based evaluation of LLMs and with recent code benchmarks. I carefully checked the methodology, equations, and tables; while some external contest details are necessarily taken at face value, my overall assessment is unlikely to change dramatically.