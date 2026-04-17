# SPILLage: Agentic Oversharing on the Web

- Decision: Reject
- Scores: 2, 4, 2

## Abstract
We present SPILLAGE, a novel framework for analyzing how web agents handle user resources when accomplishing tasks on their behalf across real-world websites. SPILLAGE introduces the problem of Natural Oversharing–the inappropriate disclosure of user information to external parties–and characterizes such disclosures along two axes based on user privacy expectations and agent capabilities: directness (explicit vs. implicit) and mode (textual vs. non-textual actions). While prior work has mostly focused on explicit textual data leakage in
synthetic environments, we study four distinct oversharing categories of agents in the wild: Explicit Content, Implicit Content, Explicit Behavioral, and Implicit Behavioral oversharing. Building on this taxonomy, we introduce the first real-world benchmark for oversharing, evaluated on live e-commerce sites. Our tasks grant agents access to user resources containing a natural blend of information essential
for task completion with inappropriate information. We deploy agents on these real websites and measure oversharing at every execution step using structured, step-level annotations. Our experiments show that agents are much more public about what users expect to be private. Agents overemphasis on task utility leave them blind to distinguishing which information is inappropriate to disclose in their
interactions with websites. For instance, a gpt-4o-based agent produces 1,151 cases of explicit behavioral oversharing on a single site. More interestingly, various categories of oversharing often co-occur within a single agentic step. Finally, we find that oversharing can be substantially reduced without a utility loss, suggesting practical mitigation opportunities.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces SPILLAGE, a framework for analyzing how web agents handle user resources and potentially disclose sensitive information to external parties. The authors identify four categories of oversharing: explicit content, implicit content, explicit behavioral, and implicit behavioral. They developed a benchmark to evaluate web agents on live e-commerce sites and found that agents tend to overshare more information than necessary, compromising user privacy. The study also discovered that reducing unnecessary information in user prompts can improve task success rates.

### Strengths
- Testing agent's oversharing in real-world web environment is certainly important. This real-world benchmark allows doing that (although I do have some concerns on this listed in the next section).
- Taxonomy of oversharing categories is quite interesting and it provides a nuanced understanding of the issue.

### Weaknesses
- I don't think the use of live websites is a good idea here. It makes it challenging to control variables and reproduce results, which is a crucial aspect of scientific experimentation. Whereas environments such as BrowserGym, WebArena allow this.
- By looking into examples, the definition and detection of implicit (both content and behavior) oversharing may be subjective, and it's unclear whether humans would perform better in similar situations. For example, for a repeated search of products in the range of $400-$600, what should AI do to avoid oversharing this? Maybe conducting a human study to compare human and agent performance could provide valuable insights.
- I guess the above concern is also relevant to explicit oversharing. By inspecting prompts in appendices, I noticed that models are not given any hint about the oversharing nor any instructions to preserve sensitive data. Since these models are primarily designed to be instruction-following, I'm wondering if this benchmark is asking too much from the model. For example, there is no mention that email chain in D.3 should be treated with some care and agent generally is not instructed to be sensitive-aware. So, it might be a fair game to include part of it in the product search?
- The evaluation methodology relies on LLM judges to detect oversharing events, which may not be perfect. There's a risk that the judges may miss certain types of oversharing or incorrectly classify benign behavior as oversharing. Was there any study to evaluate accuracy of LLM judge
- The paper doesn't thoroughly discuss straightforward solutions to mitigate oversharing, such as modifying the agent's design or implementing additional safety protocols (e.g. via system prompt).

### Questions
- example in lines 256-260 doesn't make sense to me. How iPhone is related to the "divorce" example?
- did authors use a separate 'tool' role (for GPT models) to interact with Browser? Or everything in the user-context?

### Soundness
3

### Presentation
2

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
This paper presents SPILLAGE, a framework for analyzing the oversharing behaviors of web agents. It categorizes the oversharing behaviors into four categories along two dimensions: directness (explicit vs. implicit) and mode (textual vs. non-textual actions). Based on the four categories, it crafts a benchmark consisting of prompts at three levels based on synthetic personas, and then evaluated two agentic frameworks: Browser-Use and AutoGen, along with three LLMs: o3, o4-mini, and gpt-4o. They found the oversharing behaviors were prevalent during the interactions of the agents with the real-world websites and removing inappropriate information doesn't reduce utility, showing a possible mitigation direction to improve both privacy and utility.

### Strengths
- The study on contextual privacy leakage in the context of web agents is important and much needed. The coverage of explicit and implicit leakage and leakage through text disclosure or actions is comprehensive.
- The evaluation on real-world websites provides further insights into how such privacy leakage can occur with current web agents in a realistic environment.
- The ablation study showed removing inappropriate information preserves privacy and improves utility at the same time, suggesting that there's still room for data minimization to achieve a balance between privacy and utility.

### Weaknesses
- The definition of "inappropriate information" feels vague and anecdotal. I didn't find a systematic process for judging what is appropriate or inappropriate to share and reasonable justifications for the process. In many examples presented in the paper, I feel the disclosed information is borderline sensitive, and some even seems necessary for the task (e.g., selecting a price range is a common action to do when the user is on a budget, while this is considered an implicit behavioral leakage as shown in Figure 1). This can reduce the credibility of the benchmark.
- It's not clear to me what's the novel contribution of this benchmark as compared to prior art. For example, it seems to have the same evaluation targets as AgentDAM (web agents), similar angle (oversharing vs. data minimization), both assuming a benign setting (no malicious actors), and AgentDAM also evaluates the privacy leakage of the agents *in action*. In the Table 1 of this paper, these attributes of prior work seems to be misrepresented which conveys misleading messages about the novelty of this work. In addition, I'm a bit skeptical about how much more helpful it is for using real-world websites to conduct the experiments than using a sandboxed environment. The sandboxed environments are also developed based on the real-world websites, and have the benefits of being more stable for continuous testing and replicating the results. The interactions displayed in the examples (e.g., typing text into a search box and  dragging a slider) don't seem to be too complicated to be emulated in the sandboxed environments.
- For the privacy-utility evaluation, it was not clear how a successful task completion is determined. For example, in the e-commerce setting, there's not only the goal of successfully placing an order, but also the requirement of selecting the right product that fits the user's needs --- this is actually the part that could potentially benefit from the additional personal information which I think is the more interesting part of privacy-utility tradeoffs. It was not clear whether this is captured in the evaluation, and if so, how it was measured.

### Questions
- For the utility measurement, what's the task success criteria, and how is the accuracy measured?
- What's the definition of inappropriate information? Is there any theoretical basis or empirical validation for justifying that they are indeed data that should not be shared, or would it be more appropriate to rely on more direct and objective signals (e.g., clear specification of privacy rules in the prompt, or completely irrelevant yet sensitive information)

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
5

### Summary
The paper defines a framework and a benchmark for studying information oversharing in LLM agents. They generate a synthetic dataset of 30 personas using an LLM (Claude 3.7 Sonnet) with scenarios and extra private information. They evaluate the oversharing as explicit/implicit and content/behavioral, with 3 LLMs (o3, o4-mini, GPT-4o), 2 websites (Amazon, eBay), 2 frameworks (browseruse, autogen), and an LLM as a judge (GPT-4.1 Mini). Their results show considerable oversharing, which varies by model and framework.

### Strengths
Studying whether LLM agents overshare private information conveyed by the user is very relevant. I particularly like the behavioral approach, where the authors don’t set up the problem in an adversarial setting, but in a naturally occurring way. These cases are pervasive, and we’re not able to control for them (e.g., another model can detect the issue before the agent acts).

The writing is simple, clear, and easy to follow.

### Weaknesses
As I said in the strengths, I really like the idea and the approach, but what you have so far can only be considered a pilot. I encourage the authors to improve this work, as I’d like to see it published at a top venue.

- The paper is very LLM-heavy, meaning that you use them for absolutely everything. You generate a synthetic dataset using one (Claude 3.7 Sonnet), run experiments with three (o3, o4-mini, GPT-4o), and evaluate the results with an LLM-as-judge (GPT-4.1 Mini). I think this shows a lack of depth in exploring the behavior of these agents when it comes to oversharing.
- The most important issue I see with this work is the fact that all you have is a baseline. You generated some synthetic data, collected behavioral data from the agents, and reported their oversharing results. However, the models don’t even know (although they could infer it) that data is private and should not be shared. What happens if you prompt them to be careful? What happens if you label part of the information as private? That is crucial, in my opinion, to show that this is a real problem.
- While your setup is potentially more realistic than other related work (as you point out), your synthetic dataset is not necessarily realistic. To make these results more robust and generalizable, I think you’d need to use real datasets and detect their private information directly. You can still use a synthetic dataset, but as I said, it looks more like a pilot.
- Considering the limited methodology and a framework that heavily relies on existing ones like Browser-Use and AutoGen, the experiments fall short. You only test 3 models from the same family (OpenAI): o3, o4-mini, and GPT-4o. I think it’s important to expand this list to other providers (e.g., Anthropic, Gemini) and focus on either frontier models (e.g., Claude 4.5 Sonnet, Gemini 2.5 Pro, GPT-5) or fast/economic (e.g., Claude 4.5 Haiku, Gemini 2.5 Flash, GPT-5 Mini/Nano). The scale of your experiments right now is quite low, so I don’t think this would be too much to ask for.
- You explore the amount of information overshared, but beyond the LLM-as-judge, it’d be important to see what kind of information they’re oversharing. Some of it is more critical, but also, we would also like to see if there’s a certain bias. Your ablation is great to show that the missing information wasn’t necessary to complete the task, but it doesn’t say anything about why these models are oversharing (if it’s unnecessary for the task) in the first place. For example, reasoning models seem to overshare less, and that could indicate a trend towards no oversharing. If you believe this is a relevant problem that won’t go away by itself, you should show why.
- I’d like to see a lot more information about your synthetic dataset. I can barely see some examples, and you end up with 30 personas. While I’m not sure that’s enough, your dataset is not validated by other work, so you should let us know much more about it.

### Minor Details

- There’s an isolated reference at the end of Section 2.2
- Line 174: Table* 2

### Questions
- What is your intuition for the oversharing? Why do they do it if it’s not necessary?
- Reasoning models seem to overshare less. What do you expect to see from frontier reasoning models like Claude 4.5 Sonnet, Gemini 2.5 Pro, or GPT-5? Do you think they will still overshare?

### Soundness
2

### Presentation
2

### Contribution
2
