# WALT: Web Agents that Learn Tools

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Web agents promise to automate complex browser tasks, but current methods remain brittle -- relying on step-by-step UI interactions and heavy LLM reasoning that break under dynamic layouts and long horizons. Humans, by contrast, exploit website-provided functionality through high-level operations like search, filter, and sort. We introduce WALT (Web Agents that Learn Tools), a framework that reverse-engineers latent website functionality into deterministic, callable tools. Rather than hypothesizing ad-hoc skills, WALT exposes robust implementations of automations already designed into websites, spanning discovery (search, filter, sort), communication (post, comment, upvote), and content management (create, edit, delete). Tools abstract away low-level execution: instead of reasoning about how to click and type, agents simply call search(query) or create(listing). This shifts the computational burden from fragile step-by-step reasoning to reliable tool invocation. On VisualWebArena and WebArena, WALT achieves state-of-the-art success rates (52.9% on VisualWebArena, 50.1% on WebArena) with fewer steps and less LLM-dependent reasoning. On Online-Mind2Web, a benchmark of 139 real-world websites, WALT autonomously discovers 252 tools and improves success rate by 20.5% over a tool-free baseline, establishing a robust and generalizable paradigm for browser automation. Code: https://github.com/SalesforceAIResearch/WALT

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The research paper introduces WALT (Web Agents that Learn Tools), a novel framework designed to overcome the brittleness and inefficiency of web agents. The core problem is that existing agents rely on step-by-step UI interactions, which frequently fail on dynamic websites or over long task horizons. In contrast, WALT proposes a paradigm shift from learning agent-centric "skills" to reverse-engineering a website's functionalities—spanning discovery, communication, and content management—into reusable "tools". The methodology follow the demonstrate-generate-validate loop, where a browser agent first demonstrates a function, a tool-generation agent then maps the interaction trace to a structured tool, and a test agent rigorously validates it offline. Optimize multi-step UI sequences into single, robust URL manipulations. This abstraction change the agent's role from a low-level manipulator to a high-level planner. 
On the VisualWebArena and WebArena benchmarks, WALT achieve (52.9% and 51%, respectively) and fewer action than previous methods, demonstrating a more robust approach to browser automation.

### Strengths
1. Shift from inducing brittle, low-level click/browse action to discovering environment-centric "tools." Address the cause of fragility in web automation. 
2. Change some of the UI interactions to direct URL manipulations, which can directly reduce the long-horizon UI interaction with better robustness.
3. Achieves state-of-the-art performance on two challenging benchmarks, with comprehensive ablation studies on gpt-4.1, gemini-2.5-flash and gpt5-mini that convincingly attribute these gains to the tool-based framework itself, rather than the powerful LLM.

### Weaknesses
1. Significant practical limitations on scalability and long-term maintenance. The discovery process need upfront computational and time cost. It is impractical to apply at the scale of the entire web.
2. It require the server store all the website which have explored before. Also, if the some pre-extract website changed. The agent still need update the tools. Still need a efficient way to detect the website change and update the knowledge.
3. The paper proposed that convert several UI action into one URL operation. It is hard to gurantee all action can change to URL operation. The methods lack some generalization. 
4. The evaluation is limit to a small number of research benchmarks. The framework's effectiveness against real-world adversarial challenges, the sophisticated anti-automation measures in real world still can be improved. 

The main concerns lie in the generalizability, cost-effectiveness, and long-term maintainance of the WALT approach in production environments beyond simulated academic settings.

### Questions
When real use want to delegate a task on general websites, it is hard to predict the performance on unseen website. If focus on a small range of websites, some prepopulate specific api rules in prompt/tools may generate better performance. Hope the author can share some experiments or analysis about the WALT test on any real world senoria.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents WALT, a framework for constructing web “tools” from the functional elements of websites. The central idea is to enable an agent to broadly explore and interact with a website, identify its reusable functionalities (such as search, filtering, or posting), and abstract them into high-level tools that can be invoked directly in subsequent tasks. These tools allow the agent to operate at a more semantic level, avoiding fragile, step-by-step UI manipulations. Experiments on the VisualWebArena and WebArena benchmarks demonstrate that WALT achieves substantial improvements in both success rate and execution efficiency compared with previous web-agent baselines.

### Strengths
1. The core idea is novel and intuitive and abstraction is conceptually sound and potentially impactful.
2. The empirical results show consistent improvement over prior methods across two benchmarks.

### Weaknesses
1. The paper is difficult to follow, particularly in the method section. Many paragraphs are either overly verbose or lack essential details, making it challenging to reconstruct the full workflow. Logical connections between subsections are weak, and the role of each module is not clearly articulated. For example, the implementation and workflow of B_browser, which generates demonstrations, are insufficiently described.
2. The proposed approach requires significant offline effort to explore and build a tool set for each website. And these tools are effective only for one website. If the websites change, or when the agent encounters unseen sites, this method does not work.
3. While the offline stage costs a lot, the experiment section does not analyze the time cost, success rate, or resource consumption of tool construction.
4. The paper does not include experiments that directly evaluate the Tool Construction procedure itself (e.g., URL promotion, schema synthesis, validation loops). The current experiments merely show that “using tools” improves results but do not demonstrate that the proposed construction method is necessary or superior to simpler alternatives. There is no analysis of whether this complex construction pipeline is justified in terms of accuracy, efficiency, or cost.
5. The labels of Figure 4 overlap and the text is unreadable.

### Questions
1. If the system detects or constructs a very large number of tools, how does it manage them efficiently? Would the growing tool set cause degradation in planning or selection performance?
2. It remains unclear how this method was used for the benchmark experiments. Were the same websites used in both the construction and evaluation phases? If so, could there be potential leakage or overfitting to specific site structures?

### Soundness
2

### Presentation
1

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
The paper proposes WALT, a novel framework that reverse-engineers website functionalities into deterministic, invocable tools. This abstract away frafile, step-by-step UI interactions, making browser automation more reliable. One core contribution is to replace brittle click-based sequences with robust URL manipulations through API reverseengineering. Evaluation was run on VisualWebArena and WebAreana. WALT achieves state-of-the-art success rates with fewer steps than the baselines.

### Strengths
- **Originality:** WALT is novel in that it reframing browser automation as demonstrate-generate-validate high-level tools corresponding to website-provided functionalities, which is intuitive and robust. 
- **Significance:** WALT shows SOTA performance on the two evaluaiton benchmarks, with higher efficiency with a controlled baseline approach without tools [Figure 3].
- **Robustness and generalizability:** Tools span multiple categories (search, filter, content creation, communication) and remain reliable under diverse layouts.
- **Great qualitatitive analysis:** Detailed analysis and observation of composition, success rates, and action type of discovered tools [Section 4.5].

### Weaknesses
- **Cost not quantified.** The paper does not specify the cost of offline exploration/validation. Please report discovery time distributions and other costs per validated tool.
- **Generalizability / Practicality beyond benchmarks.** WALT's generalization to live, frequently changing websites (e.g., with CAPTCHA or A/B testing) remains untested; a small study on production sandboxes or WorkArena++ tasks would be better.
- **Presentation**
  - Citation format: Line 299

### Questions
1. Why does the ablation study focus on a single split (VisualWebArena Classifieds) but not on other splits or WebArena?
2. There is a large focus on promoting eligible UI chains to URL operations. What is the frequency of failure for URL promotion or schema inference? How does WALT handle this?
3. **Fairness of comparison.** Some baselines like Claude Computer-Use Agent may operate with different observation spaces/limits/paradigm. How do you ensure comparison with baselines is fair with no hidden advantages for WALT?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces WALT (Web Agents that Learn Tools), a new framework designed to make web agents more robust and efficient. It tackles a core problem: current web agents are "brittle" because they rely on step-by-step UI interactions (e.g., click, type) and heavy LLM reasoning for every single action. This approach easily fails when website layouts change or tasks become complex.

WALT’s solution is inspired by how humans browse the web. Instead of thinking about individual clicks, humans use high-level functions a website provides, like "search," "filter," or "sort". WALT is a framework that reverse-engineers a website's built-in functionality into a set of reusable, callable "tools".

For example, instead of an agent executing a long, fragile sequence of actions to find the "cheapest blue kayak," the WALT agent can invoke a single, robust tool it learned for that site, such as search(query='kayak', category='Boats', sort_by='price', order='asc'). This "abstracts away low-level execution" and shifts the agent's job from fragile, step-by-step reasoning to high-level planning and reliable tool invocation.

### Strengths
1.  It proposes a new paradigm for web automation that shifts from brittle, low-level UI sequences to robust, high-level tool invocation.

2. It introduces a "demonstrate-generate-validate" loop to autonomously create these tools.  A browser agent explores the site to demonstrate its functionality (e.g., using search with all its filters). A "tool generation agent" analyzes these traces to create a structured tool, prioritizing robust URL manipulation (API reverse-engineering) over simple UI replays. A test agent verifies that the newly created tool works correctly before it is registered for use.

3. WALT achieves significantly higher success rates on the Visual WebArena (52.9%) and WebArena (51%) benchmarks, outperforming prior methods.

4. By abstracting complex actions into single tool calls, WALT completes tasks using fewer steps and less LLM-dependent reasoning.

### Weaknesses
1. The paper frames WALT as a paradigm shift from “skills” to “tools,” but the distinction is not always clear. Prior work such as SkillWeaver (Zheng et al., 2025) and Hybrid Agent (Song et al., 2024) already explored higher-level abstractions (skills, APIs) that reduce reliance on brittle UI actions. WALT’s contribution risks being seen as a rebranding of “API induction” or “workflow abstraction” unless the conceptual boundary is sharpened.

2. While ablations are provided, they mostly show aggregate improvements (e.g., +2.6% from multimodal DOM parsing). The analysis does not deeply isolate why certain components matter or how they interact. For example, how much of the gain comes from URL promotion vs. schema validation vs. fallback strategies?

3. WALT excels at deterministic, schema-driven tasks (search, sort, CRUD operations) but struggles with compound reasoning tasks (e.g., “find the most expensive boat with an image showing it on water and then rate it”). These failures highlight that WALT’s abstraction layer may not handle tasks requiring joint optimization across structured and perceptual constraints.

### Questions
Evaluation is limited to WebArena and VisualWebArena, which are simulated benchmarks. While these are standard, they may not capture the full variability of real-world websites (CAPTCHAs, A/B testing, anti-bot measures, dynamic content). The paper acknowledges this but does not empirically test robustness outside controlled environments.

### Soundness
3

### Presentation
3

### Contribution
3
