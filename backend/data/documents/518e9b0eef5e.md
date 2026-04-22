# Building PrismDocs: An Intelligent Document Generator for Making Complex Content Accessible

_How we built a production-ready system that transforms research papers, web articles, and documents into multiple accessible formats—PDFs, presentations, mind maps, podcasts, and FAQs—using LangGraph and modern LLMs_

---

![PrismDocs Banner](https://raw.githubusercontent.com/nitishkmr005/PrismDocs/main/.github/banner.svg)

*AI-Powered Document Generation • Transform content into professional PDFs, presentations, mind maps, and podcasts*

---

## Table of Contents

1. [The Problem We're Solving](#the-problem-were-solving)
2. [Business Value & Use Cases](#business-value--use-cases)
3. [System Architecture Overview](#system-architecture-overview)
4. [The LangGraph Workflow: Step by Step](#the-langgraph-workflow-step-by-step)
5. [Deliverable Formats: Architecture & Assembly](#deliverable-formats-architecture--assembly)
6. [Technical Deep Dive](#technical-deep-dive)
7. [Intelligent Caching Strategy](#intelligent-caching-strategy)
8. [API Design & Integration](#api-design--integration)
9. [Production Considerations](#production-considerations)
10. [Future Improvements & Roadmap](#future-improvements--roadmap)
11. [Lessons Learned](#lessons-learned)

---

## The Problem We're Solving

In today's knowledge economy, organizations face a dual challenge: **content is everywhere, but it's rarely in the right format—and complex ideas often remain inaccessible**. Teams deal with:

- **Scattered knowledge**: PDFs, slide decks, markdown files, web articles, research papers, Word documents
- **Complex content that's hard to digest**: Dense research papers, technical documentation, and lengthy articles that require significant effort to understand
- **Manual conversion**: Hours spent reformatting content for different audiences
- **Inconsistent presentation**: No unified visual language across documents
- **Lost context**: Important information buried in poorly structured files
- **One-size-fits-all output**: No way to adapt content for different learning styles (visual, auditory, interactive)

### Making Complex Content Accessible

Not everyone learns the same way. A 50-page research paper might be impenetrable to some readers but becomes crystal clear when:

- **Visualized as a mind map** showing concept relationships
- **Explained through a podcast** with conversational dialogue
- **Broken into FAQs** answering the questions readers actually have
- **Enhanced with diagrams** illustrating key processes and architectures

This is what PrismDocs delivers: **transforming complex content into multiple accessible formats** that match how people actually consume information.

### The Real Cost

Consider a typical scenario:

- A research team publishes a dense 30-page technical paper
- They need to share findings with executives, developers, and external stakeholders—each with different needs
- Manual process: Create slides for executives, diagrams for developers, FAQs for support teams
- **Our solution**: One input, multiple outputs—each optimized for its audience

This isn't just about saving time—it's about **making knowledge accessible to everyone** regardless of their learning preferences or time constraints.

---

## Business Value & Use Cases

### Primary Use Cases

#### 1. **Research Paper Comprehension**

**Problem**: Research papers and technical articles are dense, time-consuming, and difficult to share with non-experts.

**Solution**: Our system transforms complex research into multiple accessible formats:

- **Mind maps** showing concept hierarchies and relationships at a glance
- **Podcasts** explaining findings through natural, conversational dialogue
- **FAQs** answering the questions readers actually have
- **Visual summaries** with AI-generated diagrams for key concepts
- **Executive presentations** highlighting actionable insights

**Impact**: A 30-page research paper becomes a 10-minute podcast, a visual mind map, and a 5-slide deck—each serving different audiences.

#### 2. **Technical Documentation Consolidation**

**Problem**: Engineering teams have documentation scattered across PDFs, markdown files, and wikis.

**Solution**: Our system:

- Ingests multiple file formats simultaneously
- Merges content intelligently while preserving structure
- Generates both PDF documentation and PPTX presentations
- Adds AI-generated executive summaries

**Impact**: Reduced documentation preparation time from days to minutes.

#### 3. **Web Article Synthesis**

**Problem**: Long-form articles and blog posts require significant time investment to consume and understand.

**Solution**:

- Extracts and normalizes web content
- Creates structured mind maps for quick comprehension
- Generates podcast versions for hands-free learning
- Builds FAQs for interactive exploration
- Produces visual summaries with relevant diagrams

**Impact**: Users choose how they consume content—read, listen, explore, or visualize.

#### 4. **Educational Content Transformation**

**Problem**: Educational materials often exist in a single format that doesn't suit all learners.

**Solution**:

- Transforms lectures and course materials into visual mind maps
- Creates podcast versions for auditory learners
- Generates FAQ documents for self-paced review
- Builds presentation decks for instructors
- Produces enhanced PDFs with diagrams and visual explanations

**Impact**: One piece of content serves visual, auditory, and reading learners equally well.

---

## System Architecture Overview

Our document generator is built on **Hybrid Clean Architecture** principles, combining domain-driven design with practical infrastructure needs.

### Architectural Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                           │
│  Upload → Generate (SSE Stream) → Download                       │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│              Application Layer (LangGraph)                       │
│  Workflow Orchestration & State Management                       │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────┬──────────────────┬───────────────────┐
│   Domain Layer   │  Infrastructure  │   External APIs   │
│  Business Logic  │   File System    │   Gemini/Claude   │
│  Models & Rules  │   Parsers        │   OpenAI          │
└──────────────────┴──────────────────┴───────────────────┘
```

### Project Structure

```
backend/doc_generator/
├── domain/                   # Pure business logic (zero dependencies)
│   ├── models.py             # Pydantic models (WorkflowState, GeneratorConfig)
│   └── prompts/              # All LLM prompt templates
│       ├── text/             # Content generation prompts
│       ├── image/            # Image detection/generation prompts
│       ├── podcast/          # Podcast script prompts
│       └── mindmap/          # Mind map generation prompts
├── application/              # Use case orchestration
│   ├── unified_workflow.py   # Main LangGraph workflow
│   ├── unified_state.py      # Unified workflow state definition
│   ├── checkpoint_manager.py # Session-based checkpointing
│   └── nodes/                # 27+ workflow nodes
│       ├── validate_sources.py
│       ├── transform_content.py
│       ├── generate_images.py
│       ├── podcast_script.py
│       └── ... (27 total)
├── infrastructure/           # External integrations
│   ├── api/routes/           # FastAPI endpoints
│   ├── generators/           # PDF/PPTX/Markdown generators
│   ├── llm/                  # Multi-provider LLM service
│   └── image/                # Image generation services
└── utils/                    # Utility functions
```

### Key Design Decisions

1. **LangGraph for Workflow**: Provides state management, retry logic, and observability
2. **Multi-Provider LLM Support**: Gemini, Claude, and OpenAI with intelligent fallbacks
3. **Pure Python Stack**: No Node.js dependencies—easier deployment and maintenance
4. **Docker-First**: Containerized from day one for consistent environments
5. **Three-Layer Caching**: Request-level, content-level, and image-level caching

---

## The LangGraph Workflow: Step by Step

The heart of our system is a **unified LangGraph workflow** that transforms raw inputs into polished outputs. Each node is a pure function that mutates shared state.

### Complete Workflow Architecture

Here's how the unified workflow is built:

```python
def build_unified_workflow(checkpointer: Any = None) -> StateGraph:
    """
    Build the unified LangGraph workflow for all content types.

    Workflow Structure:
    1. COMMON: validate_sources -> resolve_sources -> extract_sources
       -> merge_sources -> summarize_sources -> (Route by output_type)

    2a. DOCUMENT BRANCH:
        detect_format -> parse_document_content -> transform_content -> enhance_content
        -> generate_images -> describe_images -> persist_images
        -> generate_output -> validate_output

    2b. PODCAST BRANCH:
        generate_podcast_script -> synthesize_podcast_audio

    2c. MINDMAP BRANCH:
        generate_mindmap

    2d. IMAGE_GENERATE BRANCH:
        build_image_prompt -> generate_image

    2e. IMAGE_EDIT BRANCH:
        edit_image
    """
    workflow = StateGraph(UnifiedWorkflowState)

    # ==========================================
    # COMMON NODES
    # ==========================================
    workflow.add_node("validate_sources", validate_sources_node)
    workflow.add_node("resolve_sources", resolve_sources_node)
    workflow.add_node("extract_sources", extract_sources_node)
    workflow.add_node("merge_sources", merge_sources_node)
    workflow.add_node("summarize_sources", summarize_sources_node)

    # ==========================================
    # DOCUMENT BRANCH NODES
    # ==========================================
    workflow.add_node("doc_detect_format", _wrap_document_node(detect_format_node))
    workflow.add_node("doc_transform_content", _wrap_document_node(transform_content_node))
    workflow.add_node("doc_enhance_content", _wrap_document_node(enhance_content_node))
    workflow.add_node("doc_generate_images", _wrap_document_node(generate_images_node))
    workflow.add_node("doc_describe_images", _wrap_document_node(describe_images_node))
    workflow.add_node("doc_persist_images", _wrap_document_node(persist_image_manifest_node))
    workflow.add_node("doc_generate_output", _wrap_document_node(generate_output_node))

    # ==========================================
    # PODCAST BRANCH NODES
    # ==========================================
    workflow.add_node("podcast_generate_script", generate_podcast_script_node)
    workflow.add_node("podcast_synthesize_audio", synthesize_podcast_audio_node)

    # ==========================================
    # WORKFLOW EDGES
    # ==========================================
    workflow.set_entry_point("validate_sources")
    workflow.add_edge("validate_sources", "resolve_sources")
    workflow.add_edge("resolve_sources", "extract_sources")
    workflow.add_edge("extract_sources", "merge_sources")
    workflow.add_edge("merge_sources", "summarize_sources")

    # Route after summarization based on output_type
    workflow.add_conditional_edges(
        "summarize_sources",
        route_by_output_type,
        {
            "document": "doc_detect_format",
            "podcast": "podcast_generate_script",
            "mindmap": "mindmap_generate",
            "faq": "generate_faq",
            "image_generate": "build_image_prompt",
            "image_edit": "edit_image",
        },
    )

    return workflow.compile(checkpointer=checkpointer) if checkpointer else workflow.compile()
```

### Chunked Summarization (No Truncation)

We run a **chunked map-reduce summarizer** before routing so large inputs are never truncated. This produces a stable `summary_content` field that downstream branches can safely consume without token overflow.

**What it does**:
- Splits raw content into chunks
- Summarizes each chunk
- Reduces the summaries into a single cohesive summary

**When it’s used**:
- Always runs after `merge_sources`
- **Used directly** by Podcast, Mindmap, FAQ, and Image (standalone) branches
- **Optional context** for Article and Presentation (which still use full `raw_content`)

**When it’s not used**:
- If summarization fails or LLM is unavailable, branches fall back to `raw_content`
- For document outputs, the full content is preserved for higher-fidelity rendering

**Why this is required**: It prevents token overflow on long or multi-source inputs while keeping coverage high.

**Used in**: `backend/doc_generator/application/nodes/summarize_sources.py`, `backend/doc_generator/utils/chunked_summary.py`

### Workflow Visual Representation

```
validate_sources
      ↓
resolve_sources
      ↓
extract_sources
      ↓
merge_sources
      ↓
summarize_sources
      ↓
  [Route by output_type]
      ├── document → detect_format → parse_document_content → transform_content
      │                              → enhance_content → generate_images
      │                              → describe_images → persist_images
      │                              → generate_output → validate_output
      ├── podcast → generate_script → synthesize_audio
      ├── mindmap → generate_mindmap
      ├── faq → generate_faq → generate_output → validate_output
      ├── image_generate → build_prompt → generate_image
      └── image_edit → edit_image
```

### Node-by-Node Breakdown

The table below explains **every workflow node**, why it exists, and what it does.

| Node | Branch | What it does | Why it is needed |
| --- | --- | --- | --- |
| `validate_sources` | Common | Verifies sources and decides skip/reuse behavior. | Avoids wasted work and fails fast on bad input. |
| `resolve_sources` | Common | Resolves file IDs/URLs/text into canonical paths and normalized inputs. | Ensures downstream parsers receive consistent inputs. |
| Parse Sources (`extract_sources`) | Common | Parses each source (file/URL/text) into `content_blocks` (includes OCR for images). | Centralizes parsing across all output types. |
| `merge_sources` | Common | Merges content blocks into `raw_content` and writes a temp markdown file for docs. | Creates a single, ordered source of truth before summarization/transform. |
| `summarize_sources` | Common | Chunked map-reduce summarization to produce `summary_content`. | Prevents token overflow while preserving coverage. |
| `doc_detect_format` | Document | Detects the format of the **merged document input** (temp markdown or source file). | Required because this step operates on the merged document file, not the original sources. |
| Parse Document Input (`doc_parse_document_content`) | Document | Parses the merged doc input into `raw_content` + metadata (title/pages/hash). | Required to generate canonical content and metadata for rendering and caching. |
| `doc_transform_content` | Document | LLM transforms `raw_content` into structured markdown/sections. | Creates a stable schema for PDF/Markdown/PPTX renderers. |
| `doc_enhance_content` | Document | Adds executive summary and (optionally) slide structure. | Enables executive-ready slides and summaries. |
| Generate Section Images (`doc_generate_images`) | Document | Per-section image decision + raster generation. | Adds visuals where they improve comprehension. |
| `doc_describe_images` | Document | Generates short captions/alt text and embeds base64 when needed. | Improves accessibility and PDF embedding. |
| `doc_persist_images` | Document | Writes an image manifest to support cache reuse. | Avoids regenerating images on reruns. |
| Render Output (`doc_generate_output`) | Document/FAQ | Renders final PDF/PPTX/Markdown/FAQ output. | Produces the deliverable artifact. |
| `doc_validate_output` | Document/FAQ | Validates output file and applies retry rules. | Prevents returning incomplete or corrupt files. |
| `podcast_generate_script` | Podcast | Generates structured dialogue JSON. | Provides TTS-ready script format. |
| `podcast_synthesize_audio` | Podcast | Converts dialogue to audio using TTS. | Produces the final podcast deliverable. |
| `mindmap_generate` | Mindmap | Generates hierarchical mindmap JSON. | Produces a renderable tree structure. |
| `generate_faq` | FAQ | Extracts Q&A JSON from content. | Produces structured FAQ output. |
| Build Standalone Image Prompt (`build_image_prompt`) | Image | Builds a single prompt from mindmap summary or user input. | Creates a scoped prompt for standalone image generation. |
| `image_generate` | Image | Renders raster/SVG from the prompt. | Produces the final image asset. |
| `image_edit` | Image Edit | Applies edits (style/region) to an existing image. | Enables iterative refinement on generated images. |

---

### 1️⃣ **Detect Format**

**Purpose**: Identify the input type and route to the appropriate parser.
**Why this is required**: The document branch re-parses the **merged input file**, which may be different from individual sources. This step ensures the correct parser is used for rendering and metadata extraction.

**Logic**:

- File extensions: `.pdf`, `.pptx`, `.docx`, `.md`, `.txt`, `.xlsx`
- URL detection: `http://` or `https://`
- Fallback: Treat as inline markdown text

```python
def detect_format(state: WorkflowState) -> WorkflowState:
    input_path = state.input_path

    if input_path.startswith(('http://', 'https://')):
        state.content_format = ContentFormat.URL
    elif input_path.endswith('.pdf'):
        state.content_format = ContentFormat.PDF
    elif input_path.endswith(('.md', '.txt')):
        state.content_format = ContentFormat.MARKDOWN
    # ... more formats

    return state
```

---

### 2️⃣ **Parse Document Input**

**Purpose**: Parse the merged document input and normalize it to markdown.
**Why this is required**: This is the **document-branch parser** (not the initial source extraction). It produces canonical `raw_content`, title, page count, and content hash needed for rendering and caching.

**Parsers**:

1. **UnifiedParser (MarkItDown + fallbacks)**: For PDF, DOCX, PPTX, XLSX
2. **WebParser (MarkItDown)**: For URLs and HTML
3. **MarkdownParser**: For `.md` and `.txt`

**Output**:

- Normalized markdown content
- Metadata (title, source, page count)
- **Content hash** (SHA-256) for caching

---

### 3️⃣ **Transform Content**

**Purpose**: Convert raw markdown into a structured, blog-style narrative using LLM.

This is where the magic happens. The `transform_content_node` uses sophisticated LLM prompts to restructure content:

```python
def transform_content_node(state: WorkflowState) -> WorkflowState:
    """
    Transform raw content into structured blog-style format for generators.

    Uses LLM to:
    - Transform raw content (transcripts, docs) into well-structured blog posts
    - Insert visual markers where diagrams should appear
    - Generate inline mermaid diagrams for simple concepts
    - Create executive summaries and slide structures
    """
    content = state.get("raw_content", "")
    metadata = state.get("metadata", {})

    # Detect content type for appropriate transformation
    content_type = _detect_content_type(input_format, content)

    # Get content generator
    content_generator = get_content_generator(
        api_key=content_api_key,
        provider=provider,
        model=model,
    )

    if content_generator.is_available():
        generated = content_generator.generate_blog_content(
            raw_content=content,
            content_type=content_type,
            topic=topic,
            max_tokens=max_tokens,
            include_visual_markers=True,
            audience=metadata.get("audience"),
        )

        # Store generated content
        structured["markdown"] = generated.markdown
        structured["title"] = generated.title
        structured["sections"] = generated.sections
        structured["visual_markers"] = [
            {
                "marker_id": m.marker_id,
                "type": m.visual_type,
                "title": m.title,
                "description": m.description,
            }
            for m in generated.visual_markers
        ]

    state["structured_content"] = structured
    return state
```

---

### 4️⃣ **Enhance Content**

**Purpose**: Add executive summaries and slide structures.

**Enhancements**:

1. **Executive Summary** (always generated): 3-5 key takeaways in bullet-point format
2. **Slide Structure** (only for PPTX output): Slide titles and talking points

---

### 5️⃣ **Generate Section Images**

**Purpose**: Create relevant, high-quality images for each section.

**Decision Logic**:

```
For each section:
  LLM analyzes content →
  Returns image prompt OR "none" →
  If prompt: Generate image via Gemini Imagen →
  Store in images/ folder
```

---

### 6️⃣ **Describe Images**

**Purpose**: Add captions/alt text and embed images when required by the output.

**Why this is required**: It improves accessibility and ensures images render consistently in PDFs.

---

### 7️⃣ **Persist Image Manifest**

**Purpose**: Save a manifest of generated images and their prompts.

**Why this is required**: It enables cache reuse across runs and keeps image paths stable.

---

### 8️⃣ **Render Output**

**Purpose**: Render final PDF or PPTX with all content and images.

The render step takes structured content (title, sections, images, metadata) and assembles it into the final deliverable. See [Output Formats by Deliverable](#output-formats-by-deliverable) for detailed assembly diagrams showing how sections, images, and descriptions are aligned.

---

### 9️⃣ **Validate Output**

**Purpose**: Ensure the generated file is valid and complete.

**Retry Logic**:

```python
def _should_retry_document(state: UnifiedWorkflowState) -> str:
    """
    Decide whether to retry document generation or end workflow.
    """
    errors = state.get("errors", [])

    if not errors:
        return "end"

    max_retries = get_settings().generator.max_retries
    retry_count = state.get("metadata", {}).get("_retry_count", 0)

    if retry_count >= max_retries:
        logger.warning(f"Max retries reached ({retry_count}), ending workflow")
        return "end"

    last_error = errors[-1] if errors else ""
    if "Generation failed" in last_error or "Validation failed" in last_error:
        state["metadata"]["_retry_count"] = retry_count + 1
        logger.warning(f"Retrying generation (attempt {retry_count + 1}/{max_retries})")
        return "retry"

    return "end"
```

---

## Deliverable Formats: Architecture & Assembly

After summarization, the workflow routes by `output_type`. Each branch shares the same ingestion and normalization steps, then applies a format-specific renderer powered by carefully orchestrated LLM prompts.

This section explains **how each output format is assembled**—from the prompts that drive content generation to the final rendering pipeline. Each format follows a consistent pattern: **prompt → transform → assemble → validate**.

> **See it in action**: [Sample Outputs on GitHub](https://github.com/nitishkmr005/PrismDocs/tree/main/sampleOutputs) | [Generated Documents](https://github.com/nitishkmr005/PrismDocs/tree/main/sampleOutputs/Generated%20Documents)

### 1. Article (PDF + Markdown)

**Best for**: executive briefs, formal reports, print-ready deliverables, and version-controlled knowledge bases.

**What it produces**: Professional documents with structured sections, AI-generated images, captions, and consistent typography.

#### How Article Generation Works: The Complete Flow

The article pipeline transforms raw content into a structured, publication-ready document through a carefully orchestrated sequence of prompts and rendering steps:

```
Raw Content
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. CONTENT TRANSFORMATION (LLM Prompt #1)                   │
│    System: "You are an expert technical writer..."          │
│    Input: Raw markdown, content type, audience              │
│    Output: Structured JSON (title, sections, takeaways)     │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. IMAGE DECISION (LLM Prompt #2 - Per Section)             │
│    System: "Decide whether a section needs a visual..."     │
│    Input: Section title + content preview                   │
│    Output: {needs_image, image_type, prompt, confidence}    │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. IMAGE GENERATION (Gemini Imagen)                         │
│    Input: Image prompt + style guidance                     │
│    Output: Generated PNG/SVG image                          │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. IMAGE DESCRIPTION (LLM Prompt #3)                        │
│    System: "Write a concise blog-style description..."      │
│    Input: Section content + generated image                 │
│    Output: 2-4 sentence caption                             │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. DOCUMENT ASSEMBLY (ReportLab or Markdown Writer)         │
│    Input: Structured content + images + descriptions        │
│    Process: Align sections → embed images → add captions    │
│    Output: Final PDF or Markdown file                       │
└─────────────────────────────────────────────────────────────┘
```

#### The Prompts: How Elements Connect

**Prompt #1: Content Transformation** transforms raw input into structured JSON that downstream renderers depend on. Without this schema, the PDF/Markdown generators wouldn't know how to organize sections or where to place images.

```python
def get_content_system_prompt() -> str:
    return """You are an expert technical writer who transforms raw educational content
(like lecture transcripts, slides, and documents) into polished, comprehensive blog posts.

Hard constraints:
- Use ONLY the provided content; do not add new facts, examples, metrics, or external context
- Do not guess or fill gaps with invented details
- If a detail is missing in the source, omit it

Your writing style:
- Clear, professional, and suitable for the target audience
- Educational with detailed explanations
- Well-organized with numbered sections (1., 1.1, etc.)

Output format:
- Respond with valid JSON only (no extra text)
- Follow the JSON schema in the user prompt
- Preserve ALL technical content - do not skip topics"""
```

This produces a structured JSON that defines the document's skeleton:
```json
{
  "title": "Document Title",
  "introduction": "Opening paragraphs...",
  "sections": [
    {
      "heading": "1. Section Name",
      "content": "Section content...",
      "subsections": [...]
    }
  ],
  "key_takeaways": "Summary..."
}
```

**Prompt #2: Image Decision** analyzes each section to determine if a visual would add value. It's deliberately selective—only sections with explicit visualizable structure (workflows, comparisons, hierarchies) receive images.

```python
def build_prompt_generator_prompt(section_title: str, content_preview: str) -> str:
    return f"""You are deciding whether a section needs a visual. Be selective.

Section Title: {section_title}
Section Content: {content_preview}

Decision rules:
- Only say an image is needed when the section contains explicit visualizable structure
- Use ONLY concepts and labels explicitly present in the section content
- Avoid generic visuals

Return ONLY valid JSON:
{{
  "needs_image": true|false,
  "image_type": "infographic|diagram|chart|mermaid|decorative|none",
  "prompt": "Concise visual prompt using only section concepts",
  "confidence": 0.0 to 1.0
}}"""
```

This connects to **Prompt #3** (Image Generation), which takes the prompt and converts it into Gemini-specific instructions with style guidance.

**Prompt #3: Image Description** runs after image generation to create accessibility-friendly captions that connect the visual back to the section content.

These three prompts work together: **Transform → Decide → Generate → Describe**. The structured JSON from Prompt #1 tells us what sections exist. Prompt #2 determines which sections need images. Prompt #3 ensures images are properly captioned. All of this feeds into the assembly step.

#### How PDF Assembly Works

The PDF generator takes structured content and assembles it into a cohesive document through careful alignment of sections, images, and metadata:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. COVER PAGE                                                   │
│     ├── Accent bar (visual branding)                            │
│     ├── Content type label (e.g., "TECHNICAL REPORT")           │
│     ├── Main title (large, impactful typography)                │
│     ├── Subtitle or source description                          │
│     ├── Cover image (if generated)                              │
│     └── Metadata (author, date, generator attribution)          │
├─────────────────────────────────────────────────────────────────┤
│  2. TABLE OF CONTENTS                                            │
│     ├── Auto-extracted headings from markdown                   │
│     ├── Reading time estimate (based on word count)             │
│     └── Clickable navigation to sections                        │
├─────────────────────────────────────────────────────────────────┤
│  3. EXECUTIVE SUMMARY (if available)                             │
│     └── Key takeaways in bullet format                          │
├─────────────────────────────────────────────────────────────────┤
│  4. CONTENT SECTIONS (repeated for each section)                 │
│     ├── Section banner (styled heading with divider)            │
│     ├── Section image (if generated for this section)           │
│     │   ├── Image rendered at optimal width                     │
│     │   ├── Figure caption with auto-numbering                  │
│     │   └── Image description (2-4 sentences from LLM)          │
│     ├── Lead paragraph (slightly larger, sets the tone)         │
│     ├── Body paragraphs (consistent typography)                 │
│     ├── Bullet lists (with custom bullet styling)               │
│     ├── Code blocks (syntax highlighted, line numbers)          │
│     ├── Tables (formatted with alternating row colors)          │
│     └── Mermaid diagrams (rendered as images)                   │
├─────────────────────────────────────────────────────────────────┤
│  5. KEY TAKEAWAYS                                                │
│     └── Closing summary and conclusions                         │
└─────────────────────────────────────────────────────────────────┘
```

**Section-Image Alignment Logic**:

1. Each section (H2 heading) is assigned a numeric ID based on its position or explicit numbering (e.g., "1. Introduction" → section_id=1)
2. Generated images are stored with their section_id as the key
3. When rendering, the PDF generator looks up images by section_id
4. Images are placed immediately after the section banner, before body content
5. Image descriptions (generated by vision LLM) are rendered as captions below the image

**Typography Hierarchy**:
- Title: 28pt, bold, primary color
- Section banners: 18pt, bold with accent background
- Subsection headings: 14pt, bold
- Body text: 11pt, regular
- Captions: 9pt, italic, muted color

**Sample Output**:

![Article PDF Sample](https://raw.githubusercontent.com/nitishkmr005/PrismDocs/main/sampleOutputs/Screenshots/Article%5BPDF%5D.png)

*[Download sample PDF](https://github.com/nitishkmr005/PrismDocs/blob/main/sampleOutputs/Generated%20Documents/Article%5BPDF%5D.pdf)*

### 2. Article Markdown

**Best for**: docs sites, wikis, and version-controlled knowledge bases.

**Pipeline**:
- Structured content -> Markdown generator
- Preserves headings, lists, code blocks, and links

**Output**: a clean `.md` document that drops directly into Git workflows.

**Sample Output**:

![Article Markdown Sample](https://raw.githubusercontent.com/nitishkmr005/PrismDocs/main/sampleOutputs/Screenshots/Article%5BMarkdown%5D.png)

*[View sample Markdown](https://github.com/nitishkmr005/PrismDocs/blob/main/sampleOutputs/Generated%20Documents/Article%5BMarkdown%5D.md)*

### 3. Presentation (PPTX)

**Best for**: stakeholder updates, project reviews, and pitch decks.

**What it produces**: Executive-ready PowerPoint decks with title slides, agenda, executive summary, content slides, and embedded visuals.

#### How Presentation Generation Works: The Complete Flow

The presentation pipeline converts content into slide-optimized bullets and narrative flow through specialized prompts:

```
Structured Content (from Article workflow)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. EXECUTIVE SUMMARY (LLM Prompt #1)                        │
│    System: "You are an executive communication specialist..." │
│    Input: Full content (truncated to 8000 chars)            │
│    Output: 3-5 bullet points for leadership                 │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. SECTION EXTRACTION                                        │
│    Input: Markdown content                                   │
│    Process: Parse H2 headings → deduplicate titles          │
│    Output: List of sections with title, content, section_id │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. SLIDE STRUCTURE (LLM Prompt #2)                          │
│    System: "You are a presentation designer..."             │
│    Input: Sections + image hints                            │
│    Output: Slide titles, bullets (7-10 words), speaker notes│
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. IMAGE EMBEDDING (Reuse from Article workflow)            │
│    Input: Section images (already generated)                │
│    Process: Match section_id → place before content slides  │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. PPTX ASSEMBLY (python-pptx)                              │
│    Process: Title → Agenda → Summary → Section Slides       │
│    Output: Final PowerPoint file                            │
└─────────────────────────────────────────────────────────────┘
```

#### The Prompts: How Elements Connect

**Prompt #1: Executive Summary** creates the opening narrative for the deck. This isn't just a summary—it sets the tone for what executives need to know before diving into details.

```python
def executive_summary_prompt(content: str, max_points: int) -> str:
    return f"""Analyze the following content and create an executive summary suitable for senior leadership.

Requirements:
- Maximum {max_points} key points
- Focus on strategic insights, outcomes, and business impact
- Use clear, concise language
- Each point should be 1-2 sentences max
- Use ONLY information present in the content

Content: {content[:8000]}

Respond with ONLY the bullet points, no introduction or conclusion."""
```

This connects to **Prompt #2** by providing context: the summary highlights what matters most, then the slide structure expands those points into individual slides.

**Prompt #2: Slide Structure** converts sections into slide-ready format. It's section-aligned, meaning slide titles must match section headings exactly to preserve narrative coherence.

```python
def section_slide_structure_prompt(sections: list[dict], max_slides: int) -> str:
    return f"""Create a presentation outline aligned to the sections below.

Requirements:
- One slide per section (maximum {max_slides})
- Title must match the section title exactly
- 3-4 bullet points per slide, 7-10 words max each
- Bullets should be parallel, action-led, and slide-ready
- Do NOT include numeric prefixes like "1." or "2.1"
- Provide 1-2 sentence speaker notes per slide
- Use ONLY information from each section

Sections: {sections}

Respond in JSON format:
{{
  "slides": [
    {{
      "section_title": "Exact Section Title",
      "bullets": ["Point 1", "Point 2"],
      "speaker_notes": "Brief speaker notes"
    }}
  ]
}}"""
```

The connection: **Extract Sections → Generate Slide JSON → Match Images by section_id → Assemble PPTX**. The section_id acts as the glue—it links content blocks to their corresponding images and slides.

#### How Presentation Assembly Works

The PPTX generator transforms content into a structured slide deck by extracting sections, generating slide-specific bullets, and embedding visuals:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. TITLE SLIDE                                                  │
│     ├── Presentation title (from content or metadata)           │
│     └── Subtitle (author, date, or source)                      │
├─────────────────────────────────────────────────────────────────┤
│  2. COVER IMAGE SLIDE (if generated)                             │
│     └── Full-slide visual representing the topic                │
├─────────────────────────────────────────────────────────────────┤
│  3. AGENDA SLIDE                                                 │
│     └── Auto-extracted section headings (max 6 items)           │
├─────────────────────────────────────────────────────────────────┤
│  4. EXECUTIVE SUMMARY SLIDE (if available)                       │
│     └── Key takeaways in bullet format                          │
├─────────────────────────────────────────────────────────────────┤
│  5. SECTION SLIDES (repeated for each section)                   │
│     ├── IMAGE SLIDE (if section has generated image)            │
│     │   ├── Section title                                       │
│     │   ├── Full-width section image                            │
│     │   └── Caption (image type + section name)                 │
│     └── CONTENT SLIDE(S)                                        │
│         ├── Section title (stripped of numbering)               │
│         ├── Bullet points (max 5 per slide)                     │
│         │   └── If >5 bullets: continuation slides created      │
│         └── Speaker notes (from LLM or extracted content)       │
└─────────────────────────────────────────────────────────────────┘
```

**Section-to-Slide Mapping**:

1. **Extract sections**: Parse markdown for H2 headings, capturing title and content
2. **Deduplicate**: Skip sections with identical normalized titles (e.g., "1. Introduction" and "Introduction")
3. **Generate slide structure**: LLM converts section content into slide-friendly bullets (7-10 words max each)
4. **Match images**: Section images are looked up by section_id and embedded before content slides
5. **Handle overflow**: Long bullet lists are split across continuation slides (minimum 3 bullets per continuation)

**Bullet Point Generation**:
- LLM is prompted to create 3-4 bullets per slide
- Each bullet: parallel structure, action-oriented, no markdown formatting
- Speaker notes provide 1-2 sentences of context per slide

**Image Integration**:
- Section images (infographics, diagrams) are placed on dedicated slides before content
- Cover images appear after the title slide
- Introduction sections intentionally skip images to maintain focus

**Sample Output**:

![Slides PDF Sample](https://raw.githubusercontent.com/nitishkmr005/PrismDocs/main/sampleOutputs/Screenshots/Slides%5BPDF%5D.png)

*[Download sample PPTX](https://github.com/nitishkmr005/PrismDocs/blob/main/sampleOutputs/Generated%20Documents/Slides%5BPPTX%5D.pptx) | [Download as PDF](https://github.com/nitishkmr005/PrismDocs/blob/main/sampleOutputs/Generated%20Documents/Slides%5BPDF%5D.pdf)*

### 4. Mindmap

**Best for**: planning sessions, discovery workshops, concept overviews, and rapid understanding of complex topics.

**What it produces**: Hierarchical JSON tree structures (up to 20 levels deep) with central concepts and nested branches.

#### How Mindmap Generation Works

```
Summary Content (from chunked summarization)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ MINDMAP GENERATION (Single LLM Prompt)                      │
│    System: "You are an expert at creating hierarchical      │
│             mind maps... [mode: summarize/brainstorm/...]"  │
│    Input: Content + depth guidelines + source count         │
│    Output: JSON tree {title, summary, nodes: {...}}         │
└─────────────────────────────────────────────────────────────┘
```

**The Prompt: Mode-Aware Tree Building**

The mindmap prompt adapts based on the `mode`:
- **Summarize** (default): Extract ONLY what's explicitly in the content
- **Brainstorm**: Expand creatively beyond the source
- **Goal Planning**: Transform ideas into execution plans
- **Pros & Cons**: Structure decision analysis

```python
def mindmap_system_prompt(mode: str) -> str:
    base_prompt = """Your task is to analyze the provided content and generate a mind map structure as JSON.

The JSON structure must follow this exact format:
{
  "title": "Main Topic",
  "summary": "Brief 1-2 sentence summary",
  "nodes": {
    "id": "root",
    "label": "Central Concept",
    "children": [
      {
        "id": "1",
        "label": "Main Branch 1",
        "children": [...]
      }
    ]
  }
}

Rules:
1. Each node must have unique "id", concise "label" (max 50 chars), and "children" array
2. IDs follow hierarchical pattern: "root", "1", "1.1", "1.1.1"
3. Balance the tree - avoid lopsided branches
4. Return ONLY the JSON object"""

    mode_instructions = {
        "summarize": "Extract ONLY information EXPLICITLY stated in the content",
        "brainstorm": "Branch into creative extensions and possibilities",
        # ... other modes
    }

    return base_prompt + mode_instructions.get(mode, mode_instructions["summarize"])
```

**Depth Scaling**: The prompt includes content-aware depth guidelines:
- Simple topics: 2-4 levels
- Moderate topics: 4-8 levels
- Complex technical topics: 6-12 levels
- Very detailed content: 10-20 levels

This single-prompt approach produces the entire tree in one LLM call, which is then returned as JSON ready for frontend rendering.

**Sample Output**:

![Mindmap Sample](https://raw.githubusercontent.com/nitishkmr005/PrismDocs/main/sampleOutputs/Screenshots/Mindmap.png)

*[View rendered mindmap](https://github.com/nitishkmr005/PrismDocs/blob/main/sampleOutputs/Generated%20Documents/Mindmap.png)*

### 5. Image

**Best for**: single visual assets like headers, diagrams, or infographics.

**Pipeline**:
- Content -> image prompt builder -> image generation
- Context-aware prompts to match section intent

**Output**: a generated image asset ready for reuse across documents.

**Sample Output** (with editing):

| Original Generated Image | Area Selection for Edit | Edited Result |
|:---:|:---:|:---:|
| ![Original](https://raw.githubusercontent.com/nitishkmr005/PrismDocs/main/sampleOutputs/Screenshots/Original%20Generated%20Image.png) | ![Selection](https://raw.githubusercontent.com/nitishkmr005/PrismDocs/main/sampleOutputs/Screenshots/Image%20Editing%20By%20Selecting%20area%20to%20edit.png) | ![Edited](https://raw.githubusercontent.com/nitishkmr005/PrismDocs/main/sampleOutputs/Screenshots/Edited%20Image.png) |

*[View original image](https://github.com/nitishkmr005/PrismDocs/blob/main/sampleOutputs/Generated%20Documents/Original%20Image.png) | [View edited image](https://github.com/nitishkmr005/PrismDocs/blob/main/sampleOutputs/Generated%20Documents/Edited%20Image.png)*

### 6. Podcast

**Best for**: audio-first audiences, hands-free learning, commuters, and auditory learners.

**What it produces**: Multi-speaker conversational scripts (JSON) + synthesized audio files (WAV) with natural dialogue flow.

#### How Podcast Generation Works

```
Summary Content (from chunked summarization)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ PODCAST SCRIPT GENERATION (Single LLM Prompt)               │
│    System: "Generate a podcast script..."                   │
│    Input: Content + style + duration + speakers             │
│    Output: JSON {title, description, dialogue: [...]}       │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ TTS SYNTHESIS (Text-to-Speech)                              │
│    Input: Dialogue JSON (speaker + text pairs)              │
│    Process: Concatenate speaker lines → synthesize audio    │
│    Output: WAV audio file                                   │
└─────────────────────────────────────────────────────────────┘
```

**The Prompt: Natural Dialogue Generation**

The podcast prompt creates conversational, engaging dialogue that covers key content points without sounding like a transcript:

```python
prompt = f"""Generate a podcast script about the following content.

CONTENT: {raw_content}

REQUIREMENTS:
- Style: {style}  # conversational, educational, interview, etc.
- Target duration: {duration_minutes} minutes
- Speakers: {speaker_list}  # e.g., "Host, Expert"
- Based on {source_count} source document(s)

OUTPUT FORMAT (JSON):
{{
  "title": "Episode title",
  "description": "Brief episode description",
  "dialogue": [
    {{"speaker": "Host", "text": "Welcome to..."}},
    {{"speaker": "Expert", "text": "Thanks for having me..."}}
  ]
}}

Create an engaging dialogue that covers the key points from the content.
The dialogue should feel natural and conversational."""
```

**Connection to TTS**: The dialogue JSON is parsed into speaker/text pairs, then each line is passed to TTS. The speaker name determines the voice profile used for synthesis. The concatenated audio becomes the final podcast.

**Sample Output**:

![Podcast Sample](https://raw.githubusercontent.com/nitishkmr005/PrismDocs/main/sampleOutputs/Screenshots/Podcast.png)

*[Listen to sample podcast](https://github.com/nitishkmr005/PrismDocs/blob/main/sampleOutputs/Generated%20Documents/Podcast.wav)*

### 7. FAQ

**Best for**: onboarding, troubleshooting, self-serve help pages, and knowledge base articles.

**What it produces**: Structured Q&A documents (JSON) with configurable detail levels, answer formats, and audience-specific language.

#### How FAQ Generation Works

```
Summary Content (from chunked summarization)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ FAQ EXTRACTION (Single LLM Prompt)                          │
│    System: "You are an expert at creating FAQ documents..." │
│    Input: Content + config (count, format, detail, mode)    │
│    Output: JSON {title, description, items: [...]}          │
└─────────────────────────────────────────────────────────────┘
```

**The Prompt: Configurable Q&A Extraction**

The FAQ prompt is highly parameterized—it adapts to different use cases by adjusting detail level, answer format, and audience:

```python
def build_faq_extraction_prompt(
    content: str,
    faq_count: int,           # How many questions to generate
    answer_format: str,       # "concise" or "bulleted"
    detail_level: str,        # "short" (1-2 sentences), "medium" (2-4), "deep" (4-8)
    mode: str,                # "balanced", "onboarding", "troubleshooting"
    audience: str,            # "general_reader", "developer", "business"
) -> str:
    return f"""You are an expert at creating FAQ documents from source content.

Analyze the content and extract relevant frequently asked questions.

**Configuration:**
- Generate exactly {faq_count} FAQ items
- Answer format: {answer_format}
- Detail level: {detail_level}
- FAQ mode: {mode}
- Audience: {audience}

**Output JSON format:**
{{
  "title": "FAQ title based on content topic",
  "description": "Brief description of what these FAQs cover",
  "items": [
    {{
      "id": "faq-1",
      "question": "What is X?",
      "answer": "X is... **Key points:**\\n- Point 1\\n- Point 2",
      "tags": ["topic1", "topic2"]
    }}
  ]
}}

Return ONLY valid JSON, no other text."""
```

**Configuration Impact**:
- **Mode: "onboarding"** → focuses on "What is...?" and "How do I...?" questions
- **Mode: "troubleshooting"** → focuses on "Why doesn't...?" and "How to fix...?" questions
- **Audience: "developer"** → uses technical terminology
- **Audience: "business"** → emphasizes outcomes and ROI

This single-prompt approach produces the entire FAQ structure, which can then be rendered into web pages, markdown, or help docs.

**Sample Output**:

![FAQ Sample](https://raw.githubusercontent.com/nitishkmr005/PrismDocs/main/sampleOutputs/Screenshots/FAQ.png)

*[View sample FAQ](https://github.com/nitishkmr005/PrismDocs/blob/main/sampleOutputs/Generated%20Documents/FAQ.md)*

---

## Technical Deep Dive

### Technology Stack

| Component           | Technology             | Why We Chose It                              |
| ------------------- | ---------------------- | -------------------------------------------- |
| **Workflow Engine** | LangGraph 0.2.55       | State management, retry logic, observability |
| **PDF Parsing**     | MarkItDown + fallbacks | Clean markdown with graceful degradation     |
| **Web Scraping**    | MarkItDown 0.0.1a2     | Clean markdown from HTML                     |
| **PDF Generation**  | ReportLab 4.2.5        | Full layout control, production-ready        |
| **PPTX Generation** | python-pptx 1.0.2      | Native PowerPoint format                     |
| **LLM Providers**   | Gemini, Claude, OpenAI | Multi-provider flexibility                   |
| **API Framework**   | FastAPI                | Async support, SSE streaming                 |
| **Validation**      | Pydantic 2.10.5        | Type safety and config validation            |
| **Logging**         | Loguru 0.7.3           | Beautiful, structured logs                   |
| **Package Manager** | uv                     | 10-100x faster than pip                      |

### LLM Service: Multi-Provider Architecture

The `LLMService` class provides a unified interface for multiple LLM providers with automatic fallbacks:

```python
class LLMService:
    """
    LLM service for intelligent content processing.
    Uses OpenAI GPT or Claude models for content summarization,
    slide generation, and executive presentation enhancement.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
    ):
        self.requested_provider = provider

        if provider == "gemini":
            if self.gemini_api_key and GENAI_AVAILABLE:
                self.client = genai.Client(api_key=self.gemini_api_key)
                self.provider = "gemini"
        elif provider == "openai" and self.openai_api_key:
            self.client = OpenAI(api_key=self.openai_api_key)
            self.provider = "openai"
        elif provider in ("anthropic", "claude") and self.claude_api_key:
            self.client = Anthropic(api_key=self.claude_api_key)
            self.provider = "claude"

    def _call_llm(
        self,
        system_msg: str,
        user_msg: str,
        max_tokens: int,
        temperature: float,
        json_mode: bool = False,
    ) -> str:
        """Call LLM provider with automatic fallback on errors."""

        if self.provider == "gemini":
            prompt = f"System: {system_msg}\n\nUser: {user_msg}"
            if json_mode:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    ),
                )
            else:
                response = self.client.models.generate_content(
                    model=self.model, contents=prompt
                )
            return response.text.strip()

        elif self.provider == "claude":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_msg,
                messages=[{"role": "user", "content": user_msg}],
            )
            return response.content[0].text

        else:  # openai
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
```

**Why Multiple Providers**:

1. **Cost optimization**: Gemini is 10x cheaper than GPT-4 for content
2. **Capability matching**: Claude excels at visual reasoning
3. **Redundancy**: Fallback if one provider has issues
4. **Flexibility**: Easy to swap providers per use case

## Intelligent Caching Strategy

Our three-layer caching system dramatically reduces costs and latency.

### Layer 1: API Request Cache

**What**: Caches entire API responses based on request fingerprint.

**Cache Key**:

```python
cache_key = hash({
    "output_format": "pdf",
    "sources": ["file_id_123", "https://example.com"],
    "provider": "gemini",
    "model": "gemini-2.5-pro",
    "preferences": {"temperature": 0.7, "max_tokens": 8000}
})
```

**Hit Rate**: ~40% for repeated requests

**Savings**: Full workflow skip—saves 30-60 seconds and $0.20-2.00

---

### Layer 2: Structured Content Cache

**What**: Caches transformed markdown based on content hash.

**Cache Key**: SHA-256 of normalized markdown

**Hit Rate**: ~25% when content hasn't changed but output format differs

**Savings**: Skips LLM transformation—saves 10-30 seconds and $0.10-0.50

---

### Layer 3: Image Cache + Manifest

**What**: Reuses generated images when content hash matches.

**Manifest**:

```json
{
  "content_hash": "sha256:abc123",
  "sections": [{ "title": "Intro", "image_path": "section_1.png" }]
}
```

**Hit Rate**: ~60% for documents with stable content

**Savings**: Skips image generation—saves 20-40 seconds and $0.50-1.50

---

### Combined Impact

For a document processed 5 times with minor edits:

| Run | Request Cache        | Content Cache | Image Cache | Time | Cost  |
| --- | -------------------- | ------------- | ----------- | ---- | ----- |
| 1   | Miss                 | Miss          | Miss        | 60s  | $2.00 |
| 2   | Hit                  | -             | -           | 1s   | $0.00 |
| 3   | Miss (new format)    | Hit           | Hit         | 15s  | $0.20 |
| 4   | Miss (edit)          | Miss          | Hit         | 35s  | $0.50 |
| 5   | Hit                  | -             | -           | 1s   | $0.00 |

**Total**: 112s and $2.70 vs. 300s and $10.00 without caching (63% time saved, 73% cost saved)

---

## API Design & Integration

### FastAPI Endpoints

**1. Upload File**

```bash
POST /api/upload
Content-Type: multipart/form-data

Response:
{
  "file_id": "f_abc123",
  "filename": "document.pdf",
  "size": 1234567,
  "mime_type": "application/pdf",
  "expires_in": 3600
}
```

**2. Generate Document (SSE Stream)**

```bash
POST /api/generate
Content-Type: application/json
X-Google-Key: <gemini_api_key>

{
  "output_format": "pdf",
  "provider": "gemini",
  "model": "gemini-2.5-pro",
  "image_model": "gemini-2.5-flash-image",
  "sources": [
    {"type": "file", "file_id": "f_abc123"},
    {"type": "url", "url": "https://example.com/article"},
    {"type": "text", "content": "Raw markdown content"}
  ],
  "cache": {"reuse": true},
  "preferences": {
    "temperature": 0.7,
    "max_tokens": 8000,
    "max_slides": 10
  }
}

Response (SSE stream):
event: progress
data: {"step": "parse_content", "status": "running"}

event: progress
data: {"step": "transform_content", "status": "complete"}

event: complete
data: {
  "download_url": "/api/download/f_abc123/pdf/document.pdf",
  "file_path": "f_abc123/pdf/document.pdf"
}
```

**3. Download Generated File**

```bash
GET /api/download/{file_id}/{format}/{filename}

Response: Binary file (PDF or PPTX)
```

### Server-Sent Events (SSE)

**Why SSE Over WebSockets**:

1. Simpler protocol (HTTP)
2. Automatic reconnection
3. Better for one-way streaming
4. Works through most proxies

**Event Types**:

- `progress`: Workflow step updates
- `cache_hit`: Request served from cache
- `complete`: Final result with download URL
- `error`: Failure with error message

---

## Production Considerations

### Docker Deployment

**Multi-Stage Build**:

```dockerfile
# Stage 1: Build dependencies
FROM python:3.11-slim as builder
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv pip install --system -r pyproject.toml

# Stage 2: Runtime
FROM python:3.11-slim
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY src/ /app/src/
WORKDIR /app
CMD ["python", "scripts/run_generator.py"]
```

### Observability with Opik

```python
from opik import log_llm_call

log_llm_call(
    name="content_transform",
    prompt=prompt,
    response=response,
    provider="gemini",
    model="gemini-2.5-pro",
    input_tokens=1500,
    output_tokens=3000,
    duration_ms=2500
)
```

**Metrics Tracked**:

- LLM call count and latency
- Token usage per step
- Cache hit rates
- Error rates and retry counts
- End-to-end workflow duration

### Security

**Input Validation**:

- File size limits (100MB max)
- MIME type checking
- Content sanitization
- URL validation (prevent SSRF)

**API Key Handling**:

- Never logged or stored
- Passed via headers only
- Validated before use

---

## Future Improvements & Roadmap

### Phase 1: Enhanced Intelligence (Q1 2026)

1. **Multi-Modal Input**: Audio transcription, video frame extraction
2. **Advanced Image Generation**: Diagram type detection, consistent visual style
3. **Collaborative Editing**: Real-time preview, version control

### Phase 2: Enterprise Features (Q2 2026)

1. **Template System**: Custom PDF/PPTX templates, brand kit integration
2. **Advanced Caching**: Distributed cache (Redis), semantic similarity matching
3. **Batch Processing**: Queue system, priority scheduling

### Phase 3: AI Enhancements (Q3 2026)

1. **Intelligent Content Analysis**: Fact-checking, citation verification
2. **Personalization**: Audience-specific content, language translation

### Phase 4: Platform Expansion (Q4 2026)

1. **Additional Output Formats**: HTML, EPUB, LaTeX, Notion export
2. **Integration Ecosystem**: Slack bot, Google Drive, Zapier connectors

---

## Lessons Learned

### What Worked Well

**1. LangGraph for Workflow**

- State management is clean and debuggable
- Retry logic is built-in and reliable
- Easy to add new nodes without breaking existing flow

**2. Multi-Provider LLM Strategy**

- Cost savings: 70% cheaper than GPT-4 only
- Flexibility: Can switch providers per use case
- Resilience: Fallback when one provider is down

**3. Content Hash for Caching**

- Simple yet powerful
- Deterministic and collision-resistant
- Enables all downstream caching

**4. Prompt Engineering with Source Fidelity**

- The "Hard constraints" section prevents hallucination
- Explicit "Use ONLY information present" instructions work
- JSON schema in prompts ensures structured output

### Challenges & Solutions

**Challenge 1: Image Generation Consistency**

- **Problem**: Generated images sometimes didn't match section content
- **Solution**: Added image description step that analyzes the actual generated image
- **Result**: 90% relevance improvement

**Challenge 2: Section Numbering Across Chunks**

- **Problem**: Chunks had inconsistent numbering (1, 2, 1, 2 instead of 1, 2, 3, 4)
- **Solution**: Pass section counter and outline to each chunk
- **Result**: Perfect numbering consistency

**Challenge 3: LLM Hallucination**

- **Problem**: LLM occasionally added facts not in source content
- **Solution**: Explicit system prompt: "No new facts—only restructure"
- **Result**: 95% fidelity to source content

**Challenge 4: Rate Limiting**

- **Problem**: Gemini Imagen has 20 images/minute limit
- **Solution**: Added 3-second delay between requests + retry logic
- **Result**: Zero rate limit errors

### Key Insights

**1. Separation of Concerns is Critical**

- Each node does ONE thing
- Easy to test, debug, and extend
- Clear responsibility boundaries

**2. Caching is Not Optional**

- X% cost reduction
- Y% time reduction
- Better user experience

**3. Observability from Day One**

- Opik tracing saved hours of debugging
- Structured logging made issues obvious
- Metrics guided optimization efforts

**4. Prompt Engineering is an Art**

- System prompts set the tone and constraints
- User prompts provide specific instructions
- JSON schemas ensure structured, parseable output

---

## Conclusion

Building PrismDocs taught me that **making knowledge accessible is as important as creating it**. A brilliant research paper that nobody can understand has limited impact. The same paper transformed into a mind map, a podcast, and an FAQ can reach executives, commuters, and support teams alike.

The system doesn't try to be a human writer—it's a tool that handles the tedious parts (formatting, image creation, structure, format conversion) so humans can focus on the creative parts (ideas, strategy, storytelling).

### Core Principles

1. **Accessibility over Exclusivity**: One input, multiple outputs for different audiences
2. **Fidelity over Creativity**: Restructure, don't reinvent
3. **Caching over Computation**: Reuse whenever possible
4. **Flexibility over Lock-in**: Multi-provider, multi-format

### Try It Yourself

The system is open-source and production-ready:

```bash
# Clone the repo
git clone https://github.com/nitishkmr005/PrismDocs

# Install dependencies
make setup-prismdocs

# Generate your first document
python scripts/run_generator.py input.pdf --output pdf
```

**Resources**:

- [GitHub Repository](https://github.com/nitishkmr005/PrismDocs)
- [Example Outputs](https://github.com/nitishkmr005/PrismDocs/tree/main/sampleOutputs/Generated%20Documents)

---

## About the Author

PrismDocs was built to make knowledge accessible to everyone—regardless of how they prefer to consume information. Great ideas shouldn't be locked away in formats that don't match how people learn.

**Questions? Feedback?** Open an issue on GitHub.

---

_Last updated: February 1, 2026_
