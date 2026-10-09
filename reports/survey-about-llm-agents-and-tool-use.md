# LLM Agents and Tool Use: A Thematic Survey

## TL;DR
- Early work framed tool use as modular routing and browser-assisted QA: MRKL routes queries to specialized experts, while WebGPT adds browsing plus human feedback to improve factual answering [1][2].
- Tool-use agents became more explicit with interleaved reasoning and acting: ReAct combines chain-of-thought-style reasoning traces with environment actions, and Toolformer self-supervises API selection and argument filling from a small set of demonstrations [3][4].
- Benchmarking has emphasized grounded, long-horizon interaction in text, web, and embodied settings; WebShop, ALFWorld, and BabyAI each exposed different bottlenecks in compositional instructions, transfer, and sample efficiency [5][6][7].
- More recent agent designs broaden the action space: CodeAct consolidates actions as executable Python code, while API-focused benchmarks such as ShortcutsBench stress real-world parameter filling and user-input handling [8][9][10].
- Surveys now converge on evaluation gaps: tool-use agents are increasingly judged by end-to-end task success and interaction quality, but multi-tool credit assignment, autonomous tool discovery, and safety/compliance remain open problems [11][12].

## Background
LLM agents with tool use arise from a simple limitation: a standalone language model is static, whereas many real tasks require current information, symbolic computation, external APIs, or multi-step interaction. MRKL proposed an extendable router over expert modules, explicitly naming tools such as calculators, currency converters, and database APIs as the right response when an LM alone is insufficient [1]. WebGPT then showed a different pattern: a model can browse the web, collect references, and use human feedback to improve long-form QA [2]. Together, these works establish two foundational ideas for the field: routing to specialized capabilities and grounding generation in external evidence [1][2].

Benchmarks also shaped the field early. BabyAI was designed to study sample efficiency in grounded language learning and included an extensible suite of 19 levels [7]. ALFWorld aligned text-based and embodied environments so that agents could learn abstract policies before acting in a visual simulator [6]. WebShop brought web navigation into a realistic shopping setting with 1.18 million products and 12,087 crowd-sourced instructions, making strategic exploration and query reformulation central challenges [5].

## Modular routing, browsing, and grounded environments
MRKL is best read as a systems architecture rather than a single policy: it combines a router with specialized experts and external knowledge sources [1]. Its importance is conceptual. Instead of forcing one LM to do everything, it treats tool use as selection among modules, which later work extends into APIs, search, code execution, and multi-agent systems [1].

WebGPT is the first clear bridge from retrieval to browsing-based interaction. It fine-tunes GPT-3 in a text-based browsing environment, first from demonstrations and then with a reward model and reinforcement learning or rejection sampling [2]. The paper’s emphasis on citations while browsing is important for evaluation: the model is not just asked for answers, but to gather supporting evidence that humans can judge [2].

ALFWorld and WebShop anchor the grounded-agent perspective from different angles. ALFWorld emphasizes transfer between text and embodied tasks, using aligned TextWorld and ALFRED worlds so the agent can learn in one representation and act in another [6]. WebShop instead stresses realistic webpage interaction, where the agent must choose actions, customize items, and deal with noisy text and long-horizon exploration [5]. These environments helped move the field from single-turn tool calls to sequential decision making [5][6].

## Reasoning, acting, and self-supervised tool learning
ReAct made the reasoning-action loop explicit. Its key contribution is interleaving reasoning traces with actions, so that the model can revise plans after new observations instead of generating a full plan in isolation [3]. The paper’s use of Wikipedia API calls for knowledge-heavy tasks shows how tool use can reduce hallucination and error propagation when the model needs external facts [3].

Toolformer pushed tool selection closer to the pretraining stage. Rather than hand-crafting prompting policies, it trains models to decide which APIs to call, when to call them, what arguments to pass, and how to incorporate results into next-token prediction [4]. The significance is less about a specific API set than about the learning signal: a model can self-supervise its own tool-use traces from a small number of demonstrations [4].

The shift from browsing and routing to reasoning-plus-action matters because it changes what “agent quality” means. Success is no longer just answer correctness; it also includes whether the agent chooses a useful tool, invokes it at the right time, and integrates the result without derailing the task [2][3][4].

## Code, API calls, and real-world tool interfaces
Recent work broadens tool use beyond textual search and into executable actions. CodeAct proposes using executable Python code as a unified action space for agents, integrated with a Python interpreter so that the agent can revise prior actions or emit new ones in multi-turn interaction [8]. Its reported improvements on API-Bank and a new benchmark suggest that code-as-action can be a practical control layer when tasks span many tools [8]. The Hugging Face paper entry for CodeAct makes the same design point in a compact form, and also highlights the 7k multi-turn instruction-tuning dataset [10].

API-centric benchmarks expose a different class of difficulty. ShortcutsBench uses real APIs from Apple, refined user queries, human-annotated action sequences, and parameter-filling values, making agent performance depend on selecting the right API and supplying the right arguments [9][11]. The benchmark reports weak recognition of needed input across both open and closed models, which suggests that “can the model call a tool?” is now less interesting than “can it identify when external input is required and fill it correctly?” [9].

These results point to a broader trend: tool use is moving from a single external search or calculator toward heterogeneous, composable interfaces. In that setting, the main design question becomes how to represent actions so that planning, argument filling, and recovery from mistakes all fit into one loop [8][9].

## Evaluation, comparison, and failure modes
Across surveys and benchmarks, evaluation has become more structured. The survey on agentic tool use reviews the evaluation landscape and highlights unresolved issues such as multi-tool credit assignment, autonomous tool creation, and integrating safety and compliance [11]. The separate evaluation survey proposes a taxonomy based on evaluation objectives and evaluation process, indicating that the field now treats reliability, safety, datasets, metric computation, and tooling as distinct concerns [12].

The benchmark papers show why this shift matters. WebShop and ALFWorld measure success in long-horizon settings rather than isolated tool calls, and WebGPT requires reference collection for human judgment rather than simple output matching [2][5][6]. ShortcutsBench adds real APIs and explicit parameter filling, exposing failures that would be invisible in standard QA benchmarks [9]. In other words, tool-use agents fail not only by hallucinating facts, but by choosing the wrong tool, skipping required user input, or failing to maintain state over multiple steps [2][5][9].

## Trends and open problems
Three trends stand out. First, action spaces are becoming richer: from routers and web browsing to code execution and API orchestration [1][3][8][9]. Second, benchmarks are moving toward realism, with real APIs, long-horizon interactions, and explicit handling of user-supplied parameters [5][9][12]. Third, evaluation is broadening from isolated correctness to reliability, safety, and operational compliance [11][12].

Open problems remain substantial. Long-horizon credit assignment across multiple tools is still unresolved, especially when the agent must decide whether an error came from planning, tool choice, argument filling, or state tracking [11]. Autonomous tool discovery and self-extension are attractive but underdeveloped, especially compared with manually specified tool sets [11]. Safety and compliance are increasingly important in enterprise and public-facing agents, but the current literature still offers limited end-to-end evidence on robust controls under realistic tool access [11][12]. Finally, the field still lacks consensus on how to compare heterogeneous systems fairly when some use browsing, some use APIs, and some use executable code as a unifying interface [3][8][9].

## References
[1] MRKL Systems: A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning. arxiv. https://arxiv.org/abs/2205.00445 (2022-05-01)
[2] WebGPT: Browser-assisted question-answering with human feedback. arxiv. https://arxiv.org/abs/2112.09332 (2021-12-16)
[3] BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning. web. https://openreview.net/forum?id=rJeXCo0cYX (unknown)
[4] ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. web. https://arxiv.org/abs/2010.03768 (unknown)
[5] WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents. arxiv. https://arxiv.org/abs/2207.01206 (unknown)
[6] ReAct: Synergizing Reasoning and Acting in Language Models. web. https://arxiv.org/abs/2210.03629 (2023-03-10)
[7] Toolformer: Language Models Can Teach Themselves to Use Tools. arxiv. https://arxiv.org/abs/2302.04761 (2023-02-09)
[8] Executable Code Actions Elicit Better LLM Agents. web. https://arxiv.org/abs/2402.01030 (2024-06-07)
[9] ShortcutsBench: A Large-Scale Real-world Benchmark for API-based Agents. web. https://arxiv.org/abs/2407.00132 (2024-06-28)
[10] CodeAct: Executable Code Actions Elicit Better LLM Agents. hf-search. https://huggingface.co/papers/2402.01030 (2024-02-01)
[11] ShortcutsBench: A Large-Scale Real-world Benchmark for API-based Agents. hf-search. https://huggingface.co/papers/2407.00132 (2024-06-28)
[12] Agentic Tool Use in Large Language Models: A Survey. arxiv. https://arxiv.org/abs/2604.00835 (2026-06-29)
