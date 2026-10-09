# Survey of Video and Multimodal Generation

## TL;DR
- Early video generation methods largely built on GANs, VAEs, and recurrent or attention-based architectures, with repeated structural ideas such as separating content from motion or background from foreground [1][2][3][4].
- Text-conditioned video generation emerged by adapting these ideas to captions, first through hybrid VAE-GAN or temporal GAN designs and then through more compositional conditioning mechanisms [5][6][7].
- Recent work has shifted from only synthesizing videos to evaluating them more rigorously, with benchmarks emphasizing compositional faithfulness, intrinsic trustworthiness, and human-aligned judgments [8][9][10][11][12].
- Surveys and benchmark papers now frame long-video generation as a multi-objective problem involving controllability, temporal consistency, fidelity, and safety rather than a single visual-quality target [13][14][15][16].

## Background
The earliest foundations for video and multimodal generation came from models that iteratively built structured outputs rather than generating everything in one pass. DRAW introduced a recurrent variational framework with attention for image generation [17], and caption-conditioned extensions later showed how the same idea could support text-to-image generation [7]. These ideas matter for video because they established the use of latent recurrence, attention, and incremental canvas construction as tools for multimodal synthesis.

For video generation specifically, two foundational design pressures appear repeatedly. One is spatiotemporal realism: Generating Videos with Scene Dynamics used a spatio-temporal GAN and foreground/background separation to model scene dynamics from unlabeled video [1]. The other is decomposition of latent factors: TGAN split generation into temporal and image generators, while MoCoGAN explicitly separated content from motion [2][4]. Both strategies were early attempts to make video generation more controllable and more trainable by reducing the burden on a single monolithic generator.

A third foundational thread is conditional generation from language. Video Generation From Text introduced a hybrid VAE-GAN in which a conditional VAE produced a gist for static layout and the remaining components generated motion and content conditioned on text [5]. Generating Videos from Captions then used a caption-conditioned temporal GAN with separate video, frame, and motion discriminators [6]. Together, these papers show how text-to-video inherits the core multimodal problem of aligning language with visual structure and temporal change.

## Foundations of video generation
The earliest video generators tried to exploit unlabeled video while keeping training stable. Generating Videos with Scene Dynamics emphasized scene dynamics and foreground/background disentanglement, reporting that the approach could generate short videos and learn features useful for action recognition [1]. TGAN similarly worked with unlabeled video but restructured the generator into temporal and image components and used Wasserstein GAN training with singular value clipping to stabilize deeper models [2].

MoCoGAN made the factorization idea more explicit by positing that a clip consists of content that stays fixed and motion that evolves stochastically [4]. This decomposition is important because it made video generation easier to control: the same content could be rendered with different motion, and vice versa [4]. In practice, this line of work helped establish the now-common intuition that video generation needs separate handles for identity, appearance, and temporal dynamics.

## Early multimodal and text-conditioned generation
A multimodal precursor to modern video generation is DRAW and its caption-conditioned descendant [17][7]. These models used sequential generation with attention to link language and image structure, showing that generative models can align output parts with relevant words [7]. Although image-based, the design pattern influenced later video work because it offered a way to inject semantic conditioning without forcing the whole output to be generated from a single global latent code.

Video Generation From Text extended this to video by separating static and dynamic information in a conditional VAE-GAN framework [5]. It generated a gist for background color and object layout, then conditioned motion/content generation on both the gist and the text [5]. Generating Videos from Captions pushed the idea toward a temporal GAN with multiple discriminators, explicitly targeting video realism, frame realism, and motion coherence at once [6].

The broad multimodal literature has since absorbed these lessons into larger survey framing. The 2024 survey on multimodal generation and editing connects earlier generation/editing work to LLM-based and CLIP/T5-based methods, and it highlights the recent rise of transformer-based autoregressive generation with discrete codebooks [18]. That framing is useful for video because it clarifies that video generation is not one architecture family but a moving frontier across multiple generative paradigms.

## Recent benchmarks and evaluation
Recent progress is increasingly benchmark-driven. VBench++ decomposes video generation quality into 16 dimensions, including subject identity inconsistency, motion smoothness, temporal flickering, and spatial relationship, and extends evaluation to both text-to-video and image-to-video [8]. Its design reflects the field’s move from a single aggregate quality score to a multi-axis diagnostic view.

T2V-CompBench focuses on compositional text-to-video generation and evaluates seven compositional categories, including attribute binding, spatial relationships, motion binding, object interactions, and generative numeracy [9]. It uses MLLM-based, detection-based, and tracking-based metrics [9], showing that no single evaluator captures all relevant aspects of text-video faithfulness. VBench-2.0 pushes further toward intrinsic faithfulness by evaluating human fidelity, controllability, creativity, physics, and commonsense [10].

Video-Bench and related survey work argue that standard image-style metrics and embedding-based scores often fail to match human judgments of video quality, especially for temporal and cross-modal consistency [11][12]. This is a notable shift: the community is no longer only asking whether generated videos look realistic, but whether they are coherent over time, faithful to prompts, and aligned with human preference under more demanding prompt suites [11][12].

## Controllability, fidelity, and long-range generation
A major current theme is the tension between controllability and visual consistency. VideoComposer uses motion vectors and a spatio-temporal condition encoder to improve controllable synthesis while preserving spatial and temporal consistency [19]. More recent survey and benchmark work treats controllability as conditioning on structured signals rather than only text [15], which broadens the interface from language to trajectories, layouts, and other controls [14][15].

Long-video generation makes the tradeoff sharper. A review of video diffusion generation notes that long-video methods must balance computational cost, quality, and temporal coherence, and that multimodal diffusion can improve semantic fidelity and controllability while still suffering from frame inconsistencies in complex scenes [13]. LongVie extends this concern to ultra-long generation, emphasizing degradation-aware training and multimodal control to preserve consistency over longer sequences [16]. LongVie 2 further frames the problem as a video world-model setting in which controllability, visual quality, and temporal consistency must be balanced together [20].

These results suggest that the field has moved from isolated generation tasks to system-level design. The core challenge is no longer only whether a model can make a plausible clip, but whether it can maintain identity, obey structured controls, preserve physics and commonsense, and remain stable over long horizons [10][11][13][14][16][20].

## Trends and open problems
The dominant trend is toward evaluation-aware generation: benchmarks increasingly dissect quality into finer dimensions and use richer evaluators, especially multimodal ones [8][9][10][11][12]. Another clear trend is multimodal control beyond text, with video generation becoming a problem of jointly following language, motion, layout, and other structured signals [14][15][19].

Open problems remain substantial. Long-horizon temporal coherence is still difficult, especially when models must preserve multiple objects and handle scene changes without drift [11][13][16][20]. Evaluation also remains unstable: different benchmarks prioritize realism, compositional faithfulness, or human preference, and the literature still lacks one universally trusted protocol [8][9][11][12]. Finally, the shift toward multimodal generation raises alignment questions about how to trade off fidelity, controllability, creativity, and safety when outputs become more complex and more interactive [10][12][14].

## References
[1] Generating Videos with Scene Dynamics. arxiv. https://arxiv.org/abs/1609.02612 (2016-09-08)
[2] Temporal Generative Adversarial Nets with Singular Value Clipping. arxiv. https://arxiv.org/abs/1611.06624 (2016-11-21)
[3] Sync-DRAW: Automatic Video Generation using Deep Recurrent Attentive Architectures. arxiv. https://arxiv.org/abs/1611.10314 (2017-10-21)
[4] MoCoGAN: Decomposing Motion and Content for Video Generation. arxiv. https://arxiv.org/abs/1707.04993 (2017-07-16)
[5] Video Generation From Text. arxiv. https://arxiv.org/abs/1710.00421 (2017-10-01)
[6] Generating Videos from Captions. arxiv. https://arxiv.org/abs/1804.08264 (2018-04-22)
[7] Generating Images from Captions with Attention. arxiv. https://arxiv.org/abs/1511.02793 (2016-02-29)
[8] VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models. arxiv. https://arxiv.org/abs/2411.13503 (2024-11-20)
[9] T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation. hf-search. https://huggingface.co/papers/2407.14505 (2024-07-19)
[10] VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness. hf-search. https://huggingface.co/papers/2503.21755 (2025-03-27)
[11] Video-Bench: Human-Aligned Video Generation Benchmark. web. https://arxiv.org/html/2504.04907 (2025-04-29)
[12] Generative AI Video Evaluation: Survey of Metrics, Benchmarks, and Trustworthiness. web. https://openaccess.thecvf.com/content/CVPR2026W/VGBE/papers/Safavigerdini_Generative_AI_Video_Evaluation_Survey_of_Metrics_Benchmarks_and_Trustworthiness_CVPRW_2026_paper.pdf (unknown)
[13] Video diffusion generation: comprehensive review and open .... web. https://link.springer.com/article/10.1007/s10462-025-11331-6 (2025-08-20)
[14] Video Generation Models: A Survey of Post-Training and Alignment. arxiv. https://arxiv.org/abs/2610.00812 (2026-09-30)
[15] Controllable Video Generation: A Survey. hf-search. https://huggingface.co/papers/2507.16869 (2025-07-22)
[16] LongVie: Multimodal-Guided Controllable Ultra-Long Video Generation. hf-search. https://huggingface.co/papers/2508.03694 (2025-08-05)
[17] DRAW: A Recurrent Neural Network For Image Generation. arxiv. https://arxiv.org/abs/1502.04623 (2015-02-20)
[18] LLMs Meet Multimodal Generation and Editing: A Survey. arxiv. https://arxiv.org/abs/2405.19334 (2024-05-29)
[19] VideoComposer: Compositional Video Synthesis with Motion Controllability. hf-search. https://huggingface.co/papers/2306.02018 (2023-06-03)
[20] LongVie 2: Multimodal Controllable Ultra-Long Video World Model. hf-search. https://huggingface.co/papers/2512.13604 (2025-12-15)
