// app/data/projects.ts
export type Project = {
    slug: string;
    title: string;
    tag: string;
    img?: string;                 // used for grid cards
    github?: string;
    demo?: string;
    blurb?: string;               // short card copy
    video?: string;               // optional demo for modal
    fit?: 'cover' | 'contain';    // card/modal media fit
    bg?: string;                  // letterbox color
    longIntro?: string;           // 1–2 sentence intro in modal
    sections?: { title: string; items: string[] }[];
    hideMediaInModal?: boolean;   // text-first modal when true
  };
  
  export const projects: Project[] = [
    /* ——— SCOUT ——— */
    {
      slug: 'scout',
      title: 'Scout: Local Agent Platform (macOS)',
      tag: 'Agents · macOS · Local-first',
      img: '/assets/scout-logo.jpg',
      github: 'https://github.com/layanneelassaad/scout-app',
      demo: 'https://scout.store/',
      blurb:
        '',
      video: '/assets/scout-demo.mp4',
      fit: 'contain', bg: '#000',
      hideMediaInModal: true,
      longIntro:
        'A macOS app to discover and run AI agents locally. It ships with an Agent Store, a single place to manage capabilities, and an optional on-device knowledge graph for faster retrieval. Everything is opt-in and stays on your machine.',
      sections: [
        {
            title: 'What it does',
            items: [
              'Agent Store with clear, per-agent permission prompts and reviews.',
              'Agent Manager to enable/disable agents and grant/revoke access at any time.',
              'Permissioned Search Agent limited to just the folders you choose.',
              'Optional local Knowledge Graph for faster, better retrieval and on-device reasoning.',
              'Local-first privacy: nothing leaves your machine unless you opt in.',
              'Everything runs offline unless a user enables an integration.'
            ]
          },
          {
            title: 'Tech',
            items: [
              'App: Swift (macOS) UI with an embedded Python backend.',
              'LLM runtime: local Ollama.',
              'Knowledge Graph: networkx prototype; migrating to Neo4j.',
              'Ingestion: Unstructured, python email library, aiohttp, PyYAML, python-docx, pytesseract.',
              'NLP pipeline: spaCy tokenization; transformers for embeddings; entity recognition (GliNer) and relation extraction (GliREL).',
              'Graph validation & traversal: SHACL checks plus utilities to walk the graph.',
              'Permissions: sandboxed, file-scoped access with explicit per-agent prompts.',
              'Payments demo: optional Node service.',
              'Dev workflow: ChatGPT and Cursor for AI-assisted coding.'
            ]
          }
      ]
    },
  
    /* ——— CLOUD9 ——— */
    {
      slug: 'cloud9',
      title: 'Cloud9 Event Management',
      tag: 'Microservices · FastAPI · React · AWS',
      img: '/assets/cloud9-logo.jpg',
      github: 'https://github.com/layanneelassaad/cloud9',
      blurb:
        'Columbia-focused events/course engagement built as small services with a composite GraphQL layer and React UI.',
      fit: 'cover',
      longIntro:
        'A microservices platform for campus events and course engagement. Students discover events and RSVP; organizers publish and manage profiles. Services are small and independent, with a composite service that joins data and exposes a GraphQL API. Dev runs on Docker Compose; prod runs on AWS EC2 + RDS.',
      sections: [
        {
          title: 'Services',
          items: [
            '**event-service** (FastAPI + MySQL) for CRUD on events.',
            '**organization-service** (FastAPI + SQLAlchemy + MySQL) for org profiles.',
            '**rsvp-service** (FastAPI + SQLAlchemy + MySQL) for per-event responses.',
            '**composite-service** (FastAPI + GraphQL) fans out to services and joins results.',
            '**frontend** (React).'
          ]
        },
        {
          title: 'Infra & Tech',
          items: [
            'FastAPI, Uvicorn, SQLAlchemy; MySQL 8 (RDS)',
            'GraphQL via graphql-core in the composite service',
            'React frontend; Docker Compose locally; AWS EC2 + RDS in prod.'
          ]
        }
      ]
    },
  
    /* ——— TRANSFER LEARNING (HIERARCHICAL) ——— */
    {
      slug: 'transfer-hierarchical',
      title: 'Transfer Learning for Hierarchical Image Classification',
      tag: 'Vision · Transfer Learning · Open-set',
      img: '/assets/class-logo.jpg',
      github: 'https://github.com/layanneelassaad/transfer-learning-for-multilabel-img-classification',
      blurb:
        'Dual-head classifier (superclass + subclass) at 64×64; compares scratch CNN vs. ResNet-18/EfficientNet with linear/partial/full fine-tuning.',
      fit: 'cover',
      // app/data/projects.ts — slug === 'transfer-hierarchical'
longIntro:
'Hierarchical image classification at 64×64 with a shared backbone and two heads (superclass, subclass). Compares scratch CNN vs ResNet-18/EfficientNet-B0 under linear/partial/full fine-tuning, with Mixup and confidence-based novelty routing.',

sections: [
{
  title: 'Task & framing',
  items: [
    'Coarse to fine labels (superclass → subclass) modeled as multi-task: shared backbone, two classification heads.',
    'Focus on low-resolution 64×64 images and open-set conditions at test time.',
    'Loss is CE_super + CE_sub; evaluation tracks both heads.'
  ]
},
{
  title: 'Models & transfer regimes',
  items: [
    'Scratch CNN baseline (small 3-block convnet).',
    'ResNet-18 and EfficientNet-B0 pretrained on ImageNet.',
    'Linear probe, partial unfreeze (last block), and full fine-tune.'
  ]
},
{
  title: 'Data & training details',
  items: [
    'Light aug at 64×64 (resize, flip, rotate, jitter); 90/10 train/val split.',
    'Adam with weight decay; Mixup for regularization.',
    'Class-name ↔ index utilities; metrics per head across epochs.'
  ]
},
{
  title: 'Open-set handling',
  items: [
    'Confidence threshold at inference to route low-confidence predictions to a novel class.',
    'Validation novel accuracy near zero by design; primary signal is routing behavior under shifts.'
  ]
},
{
  title: 'Results (validation)',
  items: [
    'Subclass accuracy: ResNet-18 full ≈ 88.7 percent; EfficientNet-B0 full ≈ 80.6 percent; scratch CNN ≈ 76.2 percent.',
    'Linear probes below roughly 40 percent subclass accuracy, showing feature adaptation is necessary at 64×64.',
    'Superclass accuracy at or above 96 percent across models.'
  ]
},
{
  title: 'Takeaways',
  items: [
    'Full fine-tuning of pretrained backbones is the clear winner for fine-grained subclasses.',
    'Mixup and weight decay stabilize training on small, low-res data.',
    'Simple confidence routing provides a practical open-set fallback without extra heads.'
  ]
},
{
  title: 'What I built',
  items: [
    'Dual-head training notebook with switchable backbones and fine-tuning regimes.',
    'Plots and comparison tables for head-level metrics and transfer settings.',
    'Reusable utilities for label maps, augmentations, and evaluation.'
  ]
},
{
  title: 'Tech',
  items: [
    'PyTorch and torchvision; Jupyter for experiments.',
    'Lightweight 64×64 augmentation pipeline and result visualizations.'
  ]
}
]

    },
  
    /* ——— PIPELINE PARALLELISM ——— */
    {
      slug: 'pipeline-parallel',
      title: 'Pipeline Parallelism: GPipe vs 1F1B vs Interleaved',
      tag: 'Systems · Distributed Training',
      img: '/assets/pipeline-logo.jpg',
      blurb:
        'Benchmarks PyTorch pipeline schedules on CPU across GPipe, 1F1B, and Interleaved; analyzes bubbles, throughput, and scaling.',
      github: 'https://github.com/layanneelassaad/pipeline-parallelism',
      fit: 'cover',
      // app/data/projects.ts — update where slug === 'pipeline-parallel'
longIntro:
'Benchmarks GPipe, 1F1B, and Interleaved schedules on multi-process CPU (gloo). Measures tokens/s, scaling efficiency, and bubble behavior; includes a CLI runner, grid sweep, and CSV→plot analysis.',

sections: [
{
  title: 'What’s measured',
  items: [
    'Throughput (tokens/s) and speedup vs GPipe across models and process counts.',
    'Scaling efficiency when moving from 2 → 4 processes on CPU/gloo.',
    'Bubble dynamics: fill/drain costs and how overlap and virtual stages change utilization.',
    'Activation memory footprint by schedule (O(M) for GPipe vs O(K) for 1F1B/Interleaved).'
  ]
},
{
  title: 'Schedules compared',
  items: [
    'GPipe: many micro-batches to amortize bubbles; simple but holds activations for all micro-batches.',
    '1F1B: alternates forward/backward after warm-up; trims middle idle regions; activation memory bounded by pipeline depth.',
    'Interleaved 1F1B: splits each physical stage into v virtual chunks; shorter hops reduce fill/drain time roughly by 1/v but add hand-offs.'
  ]
},
{
  title: 'CPU/gloo context',
  items: [
    'Communication and barriers are relatively expensive on CPU; per-stage compute is smaller.',
    'Overlap helps at small process counts; at larger P, extra hand-offs can erase gains.',
    'On GPUs with NCCL and larger compute per stage, crossover points shift (1F1B/interleaving win more broadly).'
  ]
},
{
  title: 'Results (selected)',
  items: [
    'Model 4L-4H, P=2: 1F1B ≈ 1.20× GPipe; Interleaved ≈ 1.13×.',
    'Model 4L-4H, P=4: 1F1B ≈ 1.04×; Interleaved ≈ 1.00×; GPipe competitive.',
    'Model 4L-8H, P=2: Interleaved ≈ 1.03× GPipe; 1F1B ≈ 0.96×.',
    'Model 4L-8H, P=4: GPipe ≈ best; 1F1B ≈ 0.99×; Interleaved ≈ 0.97×.'
  ]
},
{
  title: 'Scaling efficiency (CPU)',
  items: [
    'P=2: roughly 50–60% efficiency for the winners.',
    'P=4: roughly 24–27% efficiency across schedules due to barrier and context-switch overhead.',
    'Implication: tokens/s rises with P, but communication grows faster than useful compute on CPU.'
  ]
},
{
  title: 'Takeaways',
  items: [
    'If memory is tight, 1F1B’s O(K) activations are a clear win; at small P it also boosts throughput.',
    'Interleaving helps when stages are wide and P is small; at higher P on CPU, extra transfers negate gains.',
    'When P grows on CPU, GPipe’s simpler coordination can match or beat overlap-heavy schedules.'
  ]
},
{
  title: 'What I built',
  items: [
    'CLI runner for GPipe/1F1B/Interleaved with grid sweeps; emits per-run CSV.',
    'Analysis script that aggregates CSVs and plots speedup and efficiency.',
    'Repro paths: local torchrun or Docker image for consistent CPU runs.'
  ]
},
{
  title: 'Tech',
  items: [
    'PyTorch 2.x torch.distributed (gloo) and torchrun multi-process CPU.',
    'torch.distributed.pipelining to define stages and schedules.',
    'CSV aggregation + Matplotlib plots; optional Docker image for reproducibility.'
  ]
}
]

    },
  
    /* ——— ESG MONITOR ——— */
    {
      slug: 'esg-monitor',
      title: 'ESG Monitor',
      tag: 'Full-stack ML · FastAPI + React',
      img: '/assets/esg.jpg',
      github: 'https://github.com/layanneelassaad/ESGMonitor',
      blurb:
        'Upload ESG reports and get summaries, tags, explainable E/S/G sub-scores, location roll-ups, and a JSON report.',
      fit: 'cover',
      // app/data/projects.ts — slug === 'esg-monitor'
longIntro:
'Upload ESG reports and get summaries, topic tags, explainable E/S/G sub-scores, location roll-ups, and a JSON report. Built as a FastAPI service with token-aware chunking, Transformers pipelines, and Postgres persistence.',

sections: [
{
  title: 'What it does',
  items: [
    'Ingests PDF or text sustainability reports and produces an analyst-ready summary package.',
    'Outputs include abstractive summary, ESG topic tags, explainable E/S/G sub-scores, branch and location summaries, and a JSON artifact.'
  ]
},
{
  title: 'Pipeline',
  items: [
    'Ingestion and parsing with token-aware chunking and overlap so models see clean segments.',
    'Summarization per chunk then stitched; zero-shot labeling for ESG topics; NER for locations and entities; sentiment signal as a small prior.',
    'Scores are aggregated across chunks with contribution weights and stored alongside source text.'
  ]
},
{
  title: 'Explainability',
  items: [
    'E, S, and G sub-scores include contribution breakdowns by section and label.',
    'NER-driven location roll-ups show which branches drive scores and why.',
    'All outputs are returned as a typed JSON schema for downstream use.'
  ]
},
{
  title: 'Models',
  items: [
    'Summarization: BART large CNN.',
    'Zero-shot classification: BART MNLI.',
    'Named-entity recognition: dslim bert base NER with simple aggregation.',
    'Sentiment prior: DistilBERT SST-2.'
  ]
},
{
  title: 'Tech and infra',
  items: [
    'FastAPI with typed Pydantic schemas and CORS.',
    'Transformers pipelines with token-aware chunking and an offline fallback.',
    'Postgres for persistence; tables auto-created on startup.',
    'Docker Compose for one-command spin-up; React frontend for upload and review.'
  ]
},
{
  title: 'What I built',
  items: [
    'End-to-end API with chunking, batching, and backpressure for larger PDFs.',
    'Attribution logic that ties labels and sentiment to sub-score contributions.',
    'Typed JSON schema, DB models, and a simple React UI to explore results.'
  ]
}
]

    },
  
    /* ——— CONTRASTIVE DECODING ——— */
    {
      slug: 'contrastive-decoding',
      title: 'Contrastive Decoding (Temperature & Context Ablations)',
      tag: 'NLP · Decoding Research',
      img: '/assets/contrastive.jpg',
      github: 'https://github.com/layanneelassaad/contrastive-decoding',
      blurb:
        'Implements expert–amateur contrastive decoding; ablates amateur temperature and context window; reports diversity/MAUVE/perplexity.',
      fit: 'cover',
      longIntro:
  'Two-model decoding that downweights tokens the small “amateur” over-prefers and lifts expert-preferred ones. On WikiText-103, it cuts repetition and off-topic drift vs greedy/top-p/beam.',

sections: [
  {
    title: 'What it is',
    items: [
      'Implements contrastive decoding with an expert and an amateur from the same family (e.g., gpt2-large vs gpt2).',
      'Uses an expert plausibility mask (alpha threshold) before contrastive scoring to avoid ratio blowups and missed good tokens.',
      'You tune beta (contrast strength), amateur temperature T, and window W.',
      'Based on Li et al., 2022 (see paper link in repo).'
    ]
  },
  {
    title: 'Ablation',
    items: [
      'Grid: T in {0.5, 1.0, 1.5}; W in {one, half, max}.',
      'Dataset: WikiText-103; generation lengths 64/128; beam 1.',
      'Metrics: distinct-1/2, MAUVE, perplexity computed with the expert.'
    ]
  },
  {
    title: 'Results',
    items: [
      'Best diversity and MAUVE at T=0.5, W=max (distinct-2 0.563, MAUVE 0.044).',
      'Lowest perplexity at T=1.5, W=max (PPL 3.27) with noticeably lower diversity.',
      'Larger W consistently improves coherence; tiny windows can loop or drift.',
      'CD reduces bland fillers and repetition chains that greedy/top-p often select.'
    ]
  },
  {
    title: 'Why it helps',
    items: [
      'Amateur tends to over-score generic tokens; the contrastive term suppresses them.',
      'Expert mask keeps ranking inside a reasonable candidate set at each step.',
      'Net effect: more specific, on-topic continuations without hand-tuned penalties.'
    ]
  },
  {
    title: 'Qualitative',
    items: [
      'With T=0.5, W=max: continuations introduce concrete entities and plausible timelines.',
      'With T=1.5, W=max: text collapses into repetitive clauses like “he was appointed…”',
      'More context in the amateur helps flag repetition loops earlier.'
    ]
  },
  {
    title: 'Tech',
    items: [
      'PyTorch and Transformers for expert/amateur scoring and batching.',
      'Scripted runs (`run.py`, `eval.py`) write JSONL generations and CSV metrics.',
      'Plots for diversity vs MAUVE and PPL trade-offs included.'
    ]
  }
]

    },
  
    /* ——— MULTI-LABEL EMOTION DETECTION ——— */
    {
      slug: 'emotion-multilabel',
      title: 'Multi-Label Emotion Detection (Transformers)',
      tag: 'NLP · Multi-label',
      img: '/assets/nlp.jpg',
      github: 'https://github.com/layanneelassaad/emotion-detection-transformers',
      blurb:
        'Compares BERT, BART, and GPT-2 on 5-label multi-emotion classification with per-label thresholding.',
      fit: 'cover',
      // app/data/projects.ts → inside projects[] where slug === 'emotion-multilabel'
longIntro:
'Multi-label emotion classification over five labels with per-label thresholding and transfer from GoEmotions; focus on calibration and error analysis. BERT is the strongest overall in this setup.',

sections: [
{
  title: 'Task & data',
  items: [
    'Five emotions: Anger, Fear, Joy, Sadness, Surprise; multiple labels can be active per text.',
    'Train/test CSVs with binary indicators; an extra transfer setting maps GoEmotions (27 labels) into the 5-label taxonomy for comparison.',
    'Analysis scripts produce label frequencies, co-occurrences, and length distributions.'
  ]
},
{
  title: 'Models & training',
  items: [
    'BERT (bert-base-uncased), BART (facebook/bart-base), GPT-2 (gpt2) adapted for classification.',
    'Loss: BCEWithLogits over one-vs-rest targets; AdamW with linear warmup; small, comparable sweeps for each model.',
    'Evaluation: accuracy, micro/macro F1; class reports and confusion matrices.'
  ]
},
{
  title: 'Thresholding & calibration',
  items: [
    'Sigmoid probabilities with per-label threshold search from 0.10 to 0.90 (step 0.05) on dev.',
    'Per-label thresholds beat a fixed 0.5 and help minority classes.',
    'Notes on simple calibration: raising Fear threshold trims false positives; lowering Anger and Surprise thresholds recovers recall.'
  ]
},
{
  title: 'Results (test snapshot)',
  items: [
    'BERT: accuracy 0.397, micro F1 0.613, macro F1 0.608.',
    'BART: accuracy 0.345, micro F1 0.573, macro F1 0.562.',
    'GPT-2: accuracy 0.362, micro F1 0.572, macro F1 0.545.',
    'GoEmotions→5-label transfer baseline: accuracy 0.070, micro F1 0.257, macro F1 0.254 (label mismatch + domain shift).'
  ]
},
{
  title: 'Class-level patterns',
  items: [
    'Joy: most stable; clear lexical/context cues.',
    'Fear: over-predicted; high recall but precision drops on disbelief/safety phrases.',
    'Surprise: fragile; often confused with Fear or Joy, especially on short or ambiguous texts.',
    'Anger: under-detected in very short, abrupt utterances; benefits from lower threshold or class-balanced loss.',
    'Sadness: moderate and stable; errors split between misses and spurious activations.'
  ]
},
{
  title: 'What I built',
  items: [
    'Training/eval scripts for all three architectures with shared metrics and plots.',
    'Per-label threshold search utility and aggregate evaluator across models.',
    'Confusion-matrix reports and error buckets that tie directly to threshold recommendations.'
  ]
},
{
  title: 'Future work',
  items: [
    'Larger sweeps and class-balanced or focal loss for minority classes.',
    'Better transfer via schema-aligned training or multi-task setups.',
    'Lightweight calibration and cost-sensitive thresholds for product use.'
  ]
},
{
  title: 'Tech',
  items: [
    'Hugging Face Transformers and PyTorch for fine-tuning and scoring.',
    'Pandas/NumPy for data and metrics; Matplotlib plots for reports.',
    'CLI entrypoints for reproducible runs; outputs include JSON/CSV predictions and summaries.'
  ]
}
]

    },
  
    /* ——— LION LIFT (iOS) ——— */
    {
      slug: 'lion-lift',
      title: 'Lion Lift — Airport Carpool Matching (iOS)',
      tag: 'iOS · SwiftUI · Firebase',
      img: '/assets/lion.jpg',
      github: 'https://github.com/layanneelassaad/lion-lift',
      blurb:
        'Campus-verified ride sharing to/from NYC airports with flight-aware matching and in-app chat.',
      fit: 'cover',
      longIntro:
        'Enter your flight, get matched with peers on similar schedules, coordinate in-app, and split costs.',
      sections: [
        {
          title: 'Features',
          items: [
            'Flight-windowed matching by airport/time.',
            'In-app chat and pickup/drop-off coordination.',
            'Campus-only access; MapKit flows.'
          ]
        },
        {
            title: 'How matching works',
            items: [
              'Users add flight with airport, date/time, and seats needed, which are then bucketed by airport and scored using time gap and group size fit.',
              'Top matches form a ride group; members can chat and pin pickup/drop-off on a map.'
            ]
          },
          {
            title: 'Tech',
            items: [
              'SwiftUI (iOS 16+) with MVVM and async/await.',
              'Firebase Auth for campus-only access (domain allowlist or invite gating).',
              'Firestore for data and realtime listeners; messages as a subcollection per ride.',
              'MapKit for place search, routing hints, and coordinate storage.',
              'Optional server integrations on roadmap: flight status and weather APIs, LLM assist for coordination tips.'
            ]
          },
          {
            title: 'Data model',
            items: [
              'users: profile, school verification flag.',
              'flights: ownerId, airport, departure time, seats.',
              'rides: airport, window start/end, memberIds, status.',
              'rides/{rideId}/messages: senderId, text, timestamp.'
            ]
          },
          {
            title: 'Security and privacy',
            items: [
              'Firestore rules restrict reads/writes to ride members and owners.',
              'PII kept minimal; only campus-verified accounts can view matches.',
              'Rate limits and input validation on flight creation and chat.'
            ]
          }
      ]
    },
  
    /* ——— LL97 (HACKATHON) ——— */
    {
      slug: 'll97',
      title: 'Piercing the Sky (Columbia HackathonWinner)',
      tag: 'Data Science · ML',
      img: '/assets/hack.jpg',
      github: 'https://github.com/layanneelassaad/data-science-hackathon',
      blurb:
        'Predicts NYC skyscraper emissions and 2030 LL97 compliance; surfaces policy-relevant drivers and targets.',
      fit: 'cover',
      longIntro:
        'Manhattan-focused study estimating skyscraper emissions and 2030 LL97 compliance, with clear drivers and policy signals.',
      sections: [
        {
            title: 'Why it matters',
            items: [
              'NYC Local Law 97 sets steep 2030 emission limits; high-rises dominate urban energy use.',
              'We target Manhattan for its tall-building density and higher GHG intensity.',
              'Goals: model emissions and a 2030 compliance proxy, then explain what actually drives both.'
            ]
          },
          {
            title: 'Data and features',
            items: [
              'Building-level features: floor area, height, unit/bedroom density, primary use type, borough, energy-intensity proxies.',
              'Imputation via KNN for numerics; explicit Unknown bins for categoricals.',
              'Standardization for linear models; selective log transforms only when they reduced error.'
            ]
          },
          {
            title: 'Splits and leakage control',
            items: [
              'Stratified train/val/test by use type and size bands to preserve label balance.',
              'Design avoids near-duplicate leakage among highly similar assets.',
              'Compliance label built from policy-aligned intensity thresholds.'
            ]
          },
          {
            title: 'Modeling',
            items: [
              'Regression: OLS, Ridge, Lasso, and KNN; nested cross-validation for hyperparameters.',
              'Classification: Logistic Regression with standardized numerics and one-hot categories.',
              'Probability calibration with temperature scaling; operating point chosen for balanced precision/recall.'
            ]
          },
          {
            title: 'Results',
            items: [
              'Ridge (scaled and tuned) achieved about 68% accuracy with the best bias-variance tradeoff.',
              'Logistic classifier reached about 62% accuracy with balanced precision and recall; AUROC and PR curves clearly above chance.',
              'Residuals show mild heteroscedasticity on very large assets, suggesting segment-specific models as a next step.'
            ]
          },
          {
            title: 'Interpretation',
            items: [
              'Floor area and energy-use intensity are the strongest positive contributors.',
              'Use type remains predictive even after controlling for size and intensity.',
              'Coefficient plots and error slices by borough/use type guided recommendations.'
            ]
          },
          {
            title: 'Policy takeaways',
            items: [
              'Concentration risk: a small set of ultra-emitters drives totals; target them first.',
              'Prioritize multifamily and large commercial for incentives and financing.',
              'Pair penalties with rebates and low-interest retrofit programs to move the needle before 2030.'
            ]
          },
          {
            title: 'Tech',
            items: [
              'Python stack: pandas, numpy, scikit-learn; plotting and reporting via notebooks.',
              'Deterministic seeds, function-based preprocessing, and saved metrics/figures for reproducibility.'
            ]
          }
      ]
    },
  
    /* ——— CFD DISEASE TRANSMISSION (MIT) ——— */
    {
      slug: 'cfd-disease-transmission',
      title: 'CFD of Disease Transmission (MIT)',
      tag: 'Research · CFD · Sensing',
      img: '/assets/mit.jpg',
      github: undefined,
      blurb:
        'CFD + instrumented chamber to study droplet transport, persistence, and re-aerosolization under varying humidity/temperature.',
      fit: 'cover',
      // app/data/projects.ts — slug === 'cfd-disease-transmission'
longIntro:
'MIT research combining an instrumented chamber with CFD to study how pathogen-laden droplets move, transform, settle, and get re-aerosolized indoors. We quantify how humidity and temperature shape survivability and aerosol risk to inform practical controls in real spaces.',

sections: [
{
  title: 'Study focus',
  items: [
    'End-to-end view of droplet transmission: ejection, transport, persistence on surfaces, and reintroduction to air.',
    'Link controlled experiments with CFD to explain when and why aerosols persist or re-enter airflow.'
  ]
},
{
  title: 'Experimental setup',
  items: [
    'Arduino-controlled chamber with calibrated temperature and specific-humidity sensors for repeatable environments.',
    'Protocol to expose textiles and surfaces, then measure survivability and re-aerosolization under flow.',
    'Parameter sweeps across humidity and temperature bands to isolate environmental effects.'
  ]
},
{
  title: 'CFD modeling',
  items: [
    'Turbulent multiphase transport of respiratory droplets with size distributions, evaporation, and breakup.',
    'Surface interaction modeling for deposition, resuspension, and secondary aerosol generation from bursting bubbles.',
    'Simulation-to-bench alignment to validate trends seen in chamber measurements.'
  ]
},
{
  title: 'Four-phase framework',
  items: [
    'Ejection: initial speeds, angles, and size spectrum from cough/sneeze surrogates.',
    'Transport: advection and dispersion in indoor airflow, modulated by humidity and temperature.',
    'Persistence: survival on surfaces as a function of material and environment.',
    'Reintroduction: re-aerosolization via disturbance and bubble-burst mechanisms.'
  ]
},
{
  title: 'Findings',
  items: [
    'Lower humidity and higher temperature reduced pathogen survivability on textiles in our chamber tests.',
    'CFD and experiments agreed on conditions that shorten airborne lifetime and limit secondary aerosol formation.',
    'Implication: targeted control of humidity and temperature can materially lower indoor transmission risk.'
  ]
},
{
  title: 'Impact',
  items: [
    'Guidance for hospitals, schools, and high-traffic spaces on environmental setpoints that mitigate spread.',
    'A reproducible lab-plus-simulation workflow for evaluating interventions.'
  ]
},
{
  title: 'My role',
  items: [
    'Built and calibrated the chamber hardware and data-collection scripts.',
    'Ran environmental sweeps, analyzed survivability and re-aerosolization data, and cross-validated with CFD runs.',
    'Authored the summary of design trade-offs and operational recommendations.'
  ]
}
]

    },
  ];
  