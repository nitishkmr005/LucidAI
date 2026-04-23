"use client";

import { Fragment, type CSSProperties, type ReactNode, useEffect, useRef, useState, useCallback } from "react";
import { DocumentUploader } from "./document-uploader";
import { SearchResultPopup } from "./search-result-popup";

const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

type DocumentMeta = {
  doc_id: string;
  filename: string;
  title: string;
  word_count: number;
  sentence_count: number;
  uploaded_at: string;
};

type Annotation = {
  id: string;
  sentence_idx: number;
  color?: string;
  created_at: string;
};

type Snippet = {
  id: string;
  term: string;
  explanation: string;
  sentence_idx: number;
  word_idx: number;
  search_results?: { title: string; snippet: string; url: string }[];
  created_at: string;
};

type SearchResult = { title: string; snippet: string; url: string };

export type DocumentEvent =
  | { type: "doc_list"; documents: DocumentMeta[] }
  | { type: "doc_read_start"; doc_id: string; sentences: string[]; title: string }
  | { type: "doc_opened"; doc_id: string; title: string; raw_markdown: string; sentences: string[]; annotations: { highlights: Annotation[]; snippets: Snippet[] } }
  | { type: "doc_highlight"; sentence_idx: number; word_count: number }
  | { type: "doc_save_snippet"; term: string; sentence_idx?: number }
  | { type: "doc_note_saved"; snippet: Snippet }
  | { type: "doc_highlight_saved"; sentence_idx: number; color?: string }
  | { type: "doc_search_start"; query: string }
  | { type: "doc_search_result"; query: string; results: SearchResult[] }
  | { type: "doc_export"; format: string; download_url: string }
  | { type: "doc_reading_pause" }
  | { type: "doc_reading_resume" }
  | { type: "tts_word_tick"; sentence_idx: number; word_idx: number }
  | { type: "tts_interrupted" };

type DocumentPanelProps = {
  event: DocumentEvent | null;
  pendingSnippetTerm?: string | null;
  pendingSnippetExplanation?: string | null;
  onSelectionChange?: (docId: string | null) => void;
};

function renderInlineMarkdown(text: string, keyPrefix: string): ReactNode[] {
  const tokens = text.split(/(`[^`]+`|\*\*[^*]+\*\*|__[^_]+__|\*[^*]+\*|_[^_]+_|\[[^\]]+\]\([^)]+\))/g);

  return tokens.filter(Boolean).map((token, index) => {
    const key = `${keyPrefix}-${index}`;

    if (/^`[^`]+`$/.test(token)) {
      return <code key={key} className="doc-md-inline-code">{token.slice(1, -1)}</code>;
    }
    if (/^(\*\*|__)[\s\S]+(\*\*|__)$/.test(token)) {
      return <strong key={key}>{token.slice(2, -2)}</strong>;
    }
    if (/^(\*|_)[\s\S]+(\*|_)$/.test(token)) {
      return <em key={key}>{token.slice(1, -1)}</em>;
    }

    const linkMatch = token.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (linkMatch) {
      return (
        <a key={key} href={linkMatch[2]} target="_blank" rel="noreferrer">
          {linkMatch[1]}
        </a>
      );
    }

    return <Fragment key={key}>{token}</Fragment>;
  });
}

function stripInlineMarkdown(text: string): string {
  return text
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/(\*\*|__)(.*?)\1/g, "$2")
    .replace(/(\*|_)(.*?)\1/g, "$2");
}

function normalizeForSentenceMatch(text: string): string {
  return stripInlineMarkdown(text)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function splitLongFragment(fragment: string): string[] {
  const normalized = fragment.replace(/\s+/g, " ").trim().replace(/^[\s\-•|]+|[\s\-•|]+$/g, "");
  if (normalized.length <= 220) {
    return normalized.length > 10 ? [normalized] : [];
  }

  const splitPatterns = [
    /\s+[•|]\s+/,
    /\s+[-–—]\s+/,
    /:\s+(?=[A-Z])/,
    /(?<=[a-z])\s+(?=[A-Z][a-z]{3,}\b)/,
  ];

  let pending = [normalized];
  for (const pattern of splitPatterns) {
    const nextPending: string[] = [];
    let changed = false;
    for (const item of pending) {
      if (item.length <= 220) {
        nextPending.push(item);
        continue;
      }
      const parts = item.split(pattern).map((part) => part.trim().replace(/^[\s\-•|]+|[\s\-•|]+$/g, "")).filter((part) => part.length > 10);
      if (parts.length > 1) {
        nextPending.push(...parts);
        changed = true;
      } else {
        nextPending.push(item);
      }
    }
    pending = nextPending;
    if (changed) break;
  }

  const finalParts: string[] = [];
  for (const item of pending) {
    const compact = item.replace(/\s+/g, " ").trim();
    if (compact.length <= 240) {
      if (compact.length > 10) finalParts.push(compact);
      continue;
    }
    const words = compact.split(/\s+/);
    let chunk: string[] = [];
    for (const word of words) {
      chunk.push(word);
      const candidate = chunk.join(" ");
      if (candidate.length >= 180 && /[,:;.]$/.test(word)) {
        finalParts.push(candidate.trim());
        chunk = [];
      }
    }
    if (chunk.length) {
      finalParts.push(chunk.join(" ").trim());
    }
  }

  return finalParts;
}

function splitSpeakableSegments(text: string): string[] {
  const plain = stripInlineMarkdown(text);
  const rawFragments = plain.split(/(?<=[.!?])\s+|\s+[•|]\s+/);
  const segments: string[] = [];
  for (const fragment of rawFragments) {
    const cleaned = fragment.trim();
    if (cleaned.length <= 10) continue;
    segments.push(...splitLongFragment(cleaned));
  }
  return segments;
}

type MarkdownRendererOptions = {
  sentenceTexts: string[];
  activeSentenceIdx: number | null;
  activeWordIdx: number | null;
  snippetsBySentence: Map<number, Snippet[]>;
  highlightsBySentence: Map<number, string>;
  openSnippetId: string | null;
  onToggleSnippet: (snippetId: string) => void;
  registerSentenceRef: (sentenceIdx: number, element: HTMLElement | null) => void;
};

function renderMarkdownDocument(rawMarkdown: string, options: MarkdownRendererOptions): ReactNode[] {
  const {
    sentenceTexts,
    activeSentenceIdx,
    activeWordIdx,
    snippetsBySentence,
    highlightsBySentence,
    openSnippetId,
    onToggleSnippet,
    registerSentenceRef,
  } = options;
  const lines = rawMarkdown.replace(/\r\n/g, "\n").split("\n");
  const nodes: ReactNode[] = [];
  let sentenceCursor = 0;
  let paragraphLines: string[] = [];
  let listItems: { ordered: boolean; content: string }[] = [];
  let quoteLines: string[] = [];
  let codeFence: { language: string; lines: string[] } | null = null;

  const renderSpeakableText = (text: string, keyPrefix: string): ReactNode[] => {
    const normalizedBlock = normalizeForSentenceMatch(text);
    const matchedSegments: { sentenceIdx: number; text: string }[] = [];
    let matchCursor = 0;
    let probe = sentenceCursor;

    while (probe < sentenceTexts.length) {
      const candidate = sentenceTexts[probe];
      const normalizedCandidate = normalizeForSentenceMatch(candidate);
      if (!normalizedCandidate) {
        probe += 1;
        continue;
      }
      const foundAt = normalizedBlock.indexOf(normalizedCandidate, matchCursor);
      if (foundAt === -1) {
        break;
      }
      matchedSegments.push({ sentenceIdx: probe, text: candidate });
      matchCursor = foundAt + normalizedCandidate.length;
      probe += 1;
    }

    const segments = matchedSegments.length
      ? matchedSegments
      : splitSpeakableSegments(text).map((segment, index) => ({
          sentenceIdx: sentenceCursor + index,
          text: segment,
        }));

    if (!segments.length) {
      return renderInlineMarkdown(text, keyPrefix);
    }

    sentenceCursor += matchedSegments.length || segments.length;

    return segments.map((segment, index) => {
      const sentenceIdx = segment.sentenceIdx < sentenceTexts.length ? segment.sentenceIdx : null;
      const isActive = sentenceIdx !== null && sentenceIdx === activeSentenceIdx;
      const sentenceSnippets = sentenceIdx !== null ? snippetsBySentence.get(sentenceIdx) ?? [] : [];
      const highlightColor = sentenceIdx !== null ? highlightsBySentence.get(sentenceIdx) : undefined;
      const segmentWords = segment.text.split(/\s+/);
      const key = `${keyPrefix}-segment-${index}`;

      return (
        <Fragment key={key}>
          <span
            ref={sentenceIdx !== null ? (element) => registerSentenceRef(sentenceIdx, element) : undefined}
            className={[
              "doc-md-segment",
              isActive ? "doc-md-segment--active" : "",
              highlightColor ? "doc-md-segment--highlighted" : "",
              sentenceSnippets.length ? "doc-md-segment--noted" : "",
            ].filter(Boolean).join(" ")}
            style={highlightColor ? { "--highlight-color": highlightColor } as CSSProperties : undefined}
            data-sentence-idx={sentenceIdx ?? undefined}
          >
            {isActive
              ? segmentWords.map((word, wordIndex) => (
                  <span
                    key={`${key}-word-${wordIndex}`}
                    className={activeWordIdx === wordIndex ? "doc-word--active" : undefined}
                  >
                    {word}
                    {wordIndex < segmentWords.length - 1 ? " " : ""}
                  </span>
                ))
              : renderInlineMarkdown(segment.text, `${key}-inline`)}
            {sentenceSnippets.length ? (
              <span className="doc-note-anchor">
                {sentenceSnippets.map((snippet) => (
                  <button
                    key={snippet.id}
                    type="button"
                    className="doc-snippet-indicator"
                    onClick={() => onToggleSnippet(snippet.id)}
                    aria-label="Open saved note"
                  >
                    note
                  </button>
                ))}
                {sentenceSnippets.map((snippet) => (
                  openSnippetId === snippet.id ? (
                    <span key={`${snippet.id}-tooltip`} className="doc-snippet-tooltip" role="dialog">
                      <strong>{snippet.term}</strong>
                      <span>{snippet.explanation}</span>
                    </span>
                  ) : null
                ))}
              </span>
            ) : null}
          </span>
          {index < segments.length - 1 ? " " : ""}
        </Fragment>
      );
    });
  };

  const flushParagraph = () => {
    if (!paragraphLines.length) return;
    const text = paragraphLines.join(" ").trim();
    if (text) {
      nodes.push(
        <p key={`p-${nodes.length}`} className="doc-md-paragraph">
          {renderSpeakableText(text, `p-${nodes.length}`)}
        </p>,
      );
    }
    paragraphLines = [];
  };

  const flushList = () => {
    if (!listItems.length) return;
    const ordered = listItems[0].ordered;
    const Tag = ordered ? "ol" : "ul";
    nodes.push(
      <Tag key={`list-${nodes.length}`} className={ordered ? "doc-md-list doc-md-list--ordered" : "doc-md-list"}>
        {listItems.map((item, index) => (
          <li key={`list-item-${nodes.length}-${index}`} className="doc-md-list-item">
            {renderSpeakableText(item.content, `list-item-${nodes.length}-${index}`)}
          </li>
        ))}
      </Tag>,
    );
    listItems = [];
  };

  const flushQuote = () => {
    if (!quoteLines.length) return;
    nodes.push(
      <blockquote key={`quote-${nodes.length}`} className="doc-md-quote">
        {quoteLines.map((line, index) => (
          <p key={`quote-line-${nodes.length}-${index}`}>
            {renderSpeakableText(line, `quote-line-${nodes.length}-${index}`)}
          </p>
        ))}
      </blockquote>,
    );
    quoteLines = [];
  };

  const flushCode = () => {
    if (!codeFence) return;
    nodes.push(
      <pre key={`code-${nodes.length}`} className="doc-md-code-block">
        {codeFence.language ? <span className="doc-md-code-lang">{codeFence.language}</span> : null}
        <code>{codeFence.lines.join("\n")}</code>
      </pre>,
    );
    codeFence = null;
  };

  for (const line of lines) {
    if (codeFence) {
      if (line.trim().startsWith("```")) {
        flushCode();
      } else {
        codeFence.lines.push(line);
      }
      continue;
    }

    const trimmed = line.trim();

    if (!trimmed) {
      flushParagraph();
      flushList();
      flushQuote();
      continue;
    }

    const codeStart = trimmed.match(/^```(\w+)?$/);
    if (codeStart) {
      flushParagraph();
      flushList();
      flushQuote();
      codeFence = { language: codeStart[1] ?? "", lines: [] };
      continue;
    }

    const heading = trimmed.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      flushList();
      flushQuote();
      const level = Math.min(heading[1].length, 6);
      const content = renderSpeakableText(heading[2], `h-${nodes.length}`);
      if (level === 1) {
        nodes.push(<h1 key={`h-${nodes.length}`} className="doc-md-heading doc-md-heading--h1">{content}</h1>);
      } else if (level === 2) {
        nodes.push(<h2 key={`h-${nodes.length}`} className="doc-md-heading doc-md-heading--h2">{content}</h2>);
      } else if (level === 3) {
        nodes.push(<h3 key={`h-${nodes.length}`} className="doc-md-heading doc-md-heading--h3">{content}</h3>);
      } else if (level === 4) {
        nodes.push(<h4 key={`h-${nodes.length}`} className="doc-md-heading doc-md-heading--h4">{content}</h4>);
      } else if (level === 5) {
        nodes.push(<h5 key={`h-${nodes.length}`} className="doc-md-heading doc-md-heading--h5">{content}</h5>);
      } else {
        nodes.push(<h6 key={`h-${nodes.length}`} className="doc-md-heading doc-md-heading--h6">{content}</h6>);
      }
      continue;
    }

    if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
      flushParagraph();
      flushList();
      flushQuote();
      nodes.push(<hr key={`hr-${nodes.length}`} className="doc-md-rule" />);
      continue;
    }

    const quote = trimmed.match(/^>\s?(.*)$/);
    if (quote) {
      flushParagraph();
      flushList();
      quoteLines.push(quote[1]);
      continue;
    }

    const orderedItem = trimmed.match(/^\d+\.\s+(.+)$/);
    if (orderedItem) {
      flushParagraph();
      flushQuote();
      listItems.push({ ordered: true, content: orderedItem[1] });
      continue;
    }

    const unorderedItem = trimmed.match(/^[-*+]\s+(.+)$/);
    if (unorderedItem) {
      flushParagraph();
      flushQuote();
      listItems.push({ ordered: false, content: unorderedItem[1] });
      continue;
    }

    flushList();
    flushQuote();
    paragraphLines.push(trimmed);
  }

  flushParagraph();
  flushList();
  flushQuote();
  flushCode();

  return nodes;
}

export function DocumentPanel({ event, pendingSnippetTerm, pendingSnippetExplanation, onSelectionChange }: DocumentPanelProps) {
  const [documents, setDocuments] = useState<DocumentMeta[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [sentences, setSentences] = useState<string[]>([]);
  const [rawMarkdown, setRawMarkdown] = useState<string>("");
  const [docTitle, setDocTitle] = useState<string>("");
  const [activeSentenceIdx, setActiveSentenceIdx] = useState<number | null>(null);
  const [activeWordIdx, setActiveWordIdx] = useState<number | null>(null);
  const [snippets, setSnippets] = useState<Map<number, Snippet[]>>(new Map());
  const [highlights, setHighlights] = useState<Map<number, string>>(new Map());
  const [openSnippetId, setOpenSnippetId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState<string | null>(null);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearchLoading, setIsSearchLoading] = useState(false);
  const [isReading, setIsReading] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const sentenceRefs = useRef(new Map<number, HTMLElement | null>());

  const loadDocuments = useCallback(async () => {
    try {
      const res = await fetch(`${backendUrl}/documents/`);
      if (res.ok) setDocuments(await res.json());
    } catch {}
  }, []);

  const loadDocumentDetail = useCallback(async (docId: string) => {
    const res = await fetch(`${backendUrl}/documents/${docId}`);
    if (!res.ok) return null;
    return res.json();
  }, []);

  useEffect(() => { loadDocuments(); }, [loadDocuments]);

  useEffect(() => {
    sentenceRefs.current.clear();
    setOpenSnippetId(null);
  }, [rawMarkdown, selectedDocId]);

  useEffect(() => {
    if (activeSentenceIdx === null) return;
    sentenceRefs.current.get(activeSentenceIdx)?.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  }, [activeSentenceIdx]);

  // React to events from voice agent
  useEffect(() => {
    if (!event) return;

    if (event.type === "doc_list") {
      setDocuments(event.documents);
    } else if (event.type === "doc_read_start") {
      setSentences(event.sentences);
      setDocTitle(event.title);
      setSelectedDocId(event.doc_id);
      onSelectionChange?.(event.doc_id);
      setActiveSentenceIdx(null);
      setActiveWordIdx(null);
      setIsReading(true);
      if (!rawMarkdown || selectedDocId !== event.doc_id) {
        void loadDocumentDetail(event.doc_id).then((data) => {
          if (!data) return;
          setRawMarkdown(data.raw_markdown ?? "");
          setHighlights(new Map((data.annotations?.highlights ?? []).map((item: Annotation) => [item.sentence_idx, item.color ?? "yellow"])));
          const nextSnippets = new Map<number, Snippet[]>();
          for (const snippet of data.annotations?.snippets ?? []) {
            const list = nextSnippets.get(snippet.sentence_idx) ?? [];
            list.push(snippet);
            nextSnippets.set(snippet.sentence_idx, list);
          }
          setSnippets(nextSnippets);
        });
      }
    } else if (event.type === "doc_opened") {
      setRawMarkdown(event.raw_markdown);
      setSentences(event.sentences);
      setDocTitle(event.title);
      setSelectedDocId(event.doc_id);
      onSelectionChange?.(event.doc_id);
      setHighlights(new Map((event.annotations?.highlights ?? []).map((item) => [item.sentence_idx, item.color ?? "yellow"])));
      const nextSnippets = new Map<number, Snippet[]>();
      for (const snippet of event.annotations?.snippets ?? []) {
        const list = nextSnippets.get(snippet.sentence_idx) ?? [];
        list.push(snippet);
        nextSnippets.set(snippet.sentence_idx, list);
      }
      setSnippets(nextSnippets);
    } else if (event.type === "doc_highlight") {
      // `doc_highlight` is emitted when the server is about to stream a sentence.
      // The actual visible reading cursor should follow playback start via
      // `tts_word_tick` so the document doesn't jump ahead of the audio.
    } else if (event.type === "tts_word_tick") {
      setActiveSentenceIdx(event.sentence_idx);
      setActiveWordIdx(event.word_idx);
    } else if (event.type === "tts_interrupted") {
      setActiveSentenceIdx(null);
      setActiveWordIdx(null);
      setIsReading(false);
    } else if (event.type === "doc_reading_pause") {
      setIsReading(false);
    } else if (event.type === "doc_reading_resume") {
      setIsReading(true);
    } else if (event.type === "doc_note_saved") {
      setSnippets((prev) => {
        const next = new Map(prev);
        const list = next.get(event.snippet.sentence_idx) ?? [];
        next.set(event.snippet.sentence_idx, [...list, event.snippet]);
        return next;
      });
      setOpenSnippetId(event.snippet.id);
    } else if (event.type === "doc_highlight_saved") {
      setHighlights((prev) => {
        const next = new Map(prev);
        next.set(event.sentence_idx, event.color ?? "yellow");
        return next;
      });
    } else if (event.type === "doc_search_start") {
      setSearchQuery(event.query);
      setSearchResults([]);
      setIsSearchLoading(true);
    } else if (event.type === "doc_search_result") {
      setSearchResults(event.results);
      setIsSearchLoading(false);
    } else if (event.type === "doc_export") {
      const a = document.createElement("a");
      a.href = `${backendUrl}${event.download_url}`;
      a.download = "";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  }, [event, loadDocumentDetail, onSelectionChange, rawMarkdown, selectedDocId]);

  // Save snippet when agent signals save
  useEffect(() => {
    if (!pendingSnippetTerm || !pendingSnippetExplanation || !selectedDocId || activeSentenceIdx === null) return;
    const term = pendingSnippetTerm;
    const explanation = pendingSnippetExplanation;
    const idx = activeSentenceIdx;
    fetch(`${backendUrl}/documents/${selectedDocId}/annotations/snippet`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ term, explanation, sentence_idx: idx, word_idx: activeWordIdx ?? -1 }),
    })
      .then((r) => r.json())
      .then((sn: Snippet) => {
        setSnippets((prev) => {
          const next = new Map(prev);
          const list = next.get(idx) ?? [];
          next.set(idx, [...list, sn]);
          return next;
        });
      })
      .catch(() => {});
  }, [pendingSnippetTerm, pendingSnippetExplanation, selectedDocId, activeSentenceIdx, activeWordIdx]);

  const handleUpload = useCallback(async (file: File) => {
    setIsUploading(true);
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${backendUrl}/documents/upload`, { method: "POST", body: form });
    if (!res.ok) throw new Error((await res.json()).detail ?? "Upload failed");
    await loadDocuments();
    setIsUploading(false);
  }, [loadDocuments]);

  const handleSelectDoc = useCallback(async (docId: string) => {
    const data = await loadDocumentDetail(docId);
    if (!data) return;
    setSelectedDocId(docId);
    onSelectionChange?.(docId);
    setRawMarkdown(data.raw_markdown ?? "");
    setSentences(data.sentences);
    setDocTitle(data.title);
    setActiveSentenceIdx(null);
    setActiveWordIdx(null);
    setHighlights(new Map((data.annotations?.highlights ?? []).map((item: Annotation) => [item.sentence_idx, item.color ?? "yellow"])));
    const nextSnippets = new Map<number, Snippet[]>();
    for (const snippet of data.annotations?.snippets ?? []) {
      const list = nextSnippets.get(snippet.sentence_idx) ?? [];
      list.push(snippet);
      nextSnippets.set(snippet.sentence_idx, list);
    }
    setSnippets(nextSnippets);
  }, [loadDocumentDetail, onSelectionChange]);

  const handleDeleteDoc = useCallback(async (docId: string) => {
    await fetch(`${backendUrl}/documents/${docId}`, { method: "DELETE" });
    if (selectedDocId === docId) {
      setSelectedDocId(null);
      onSelectionChange?.(null);
      setSentences([]);
      setRawMarkdown("");
      setDocTitle("");
    }
    await loadDocuments();
  }, [selectedDocId, loadDocuments, onSelectionChange]);

  const activeSentenceText =
    activeSentenceIdx !== null && activeSentenceIdx >= 0 && activeSentenceIdx < sentences.length
      ? sentences[activeSentenceIdx]
      : null;
  const activeWordText =
    activeSentenceText && activeWordIdx !== null
      ? activeSentenceText.split(/\s+/)[activeWordIdx] ?? null
      : null;

  const renderWords = useCallback((sentence: string, isActive: boolean) => {
    const words = sentence.split(/\s+/);

    return words.map((word, wi) => (
      <span
        key={`${word}-${wi}`}
        className={isActive && activeWordIdx === wi ? "doc-word--active" : undefined}
      >
        {word}
        {wi < words.length - 1 ? " " : ""}
      </span>
    ));
  }, [activeWordIdx]);

  const hasDocument = sentences.length > 0;
  const registerSentenceRef = useCallback((sentenceIdx: number, element: HTMLElement | null) => {
    sentenceRefs.current.set(sentenceIdx, element);
  }, []);
  const toggleSnippet = useCallback((snippetId: string) => {
    setOpenSnippetId((current) => current === snippetId ? null : snippetId);
  }, []);
  const markdownNodes = rawMarkdown
    ? renderMarkdownDocument(rawMarkdown, {
        sentenceTexts: sentences,
        activeSentenceIdx,
        activeWordIdx,
        snippetsBySentence: snippets,
        highlightsBySentence: highlights,
        openSnippetId,
        onToggleSnippet: toggleSnippet,
        registerSentenceRef,
      })
    : null;

  return (
    <section className="doc-panel surface">
      <div className="doc-panel-main">
        <header className="doc-panel-header">
          <div>
            <p className="kicker">Document Workspace</p>
            <h2 className="doc-panel-title">{docTitle || "Document library"}</h2>
          </div>
          <div className="doc-panel-status">
            <span className="status-pill is-ghost">{documents.length.toLocaleString()} documents</span>
            {isReading ? (
              <span className="mode-chip active-listening">
                <span className="mode-chip-dot" aria-hidden="true" /> Reading
              </span>
            ) : null}
            {activeWordText ? (
              <span className="doc-word-pill">Live word: {activeWordText}</span>
            ) : null}
          </div>
        </header>

        {searchQuery && (
          <SearchResultPopup
            query={searchQuery}
            results={searchResults}
            isLoading={isSearchLoading}
            onDismiss={() => { setSearchQuery(null); setIsSearchLoading(false); }}
          />
        )}

        <div className="doc-panel-body" ref={containerRef}>
          {!hasDocument ? (
            <DocumentUploader
              documents={documents}
              selectedDocId={selectedDocId}
              isUploading={isUploading}
              onUpload={handleUpload}
              onSelect={handleSelectDoc}
              onDelete={handleDeleteDoc}
              onRefresh={loadDocuments}
            />
          ) : (
            <div className="doc-content">
              <div className="doc-content-toolbar">
                <button
                  type="button"
                  className="doc-back-btn"
                  onClick={() => {
                    setSentences([]);
                    setDocTitle("");
                    setSelectedDocId(null);
                    onSelectionChange?.(null);
                    loadDocuments();
                  }}
                >
                  ← Back to library
                </button>
                <span className="doc-reading-status">
                  {isReading ? (
                    <span className="mode-chip active-listening">
                      <span className="mode-chip-dot" aria-hidden="true" /> Reading
                    </span>
                  ) : null}
                </span>
              </div>
              {activeSentenceText ? (
                <div className="doc-live-status">
                  <span className="doc-live-label">Reading now</span>
                  <p className="doc-live-text">{renderWords(activeSentenceText, true)}</p>
                </div>
              ) : null}
              <div className="doc-markdown">
                {markdownNodes}
              </div>
            </div>
          )}
        </div>
      </div>
      <aside className="doc-rail" aria-label="Document tools">
        <button type="button" className="doc-rail-item is-active" aria-label="Library">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M7 3h7l4 4v14H7z" />
            <path d="M14 3v5h5" />
            <path d="M10 13h6M10 17h4" />
          </svg>
          <span>Library</span>
        </button>
        <button type="button" className="doc-rail-item" aria-label="Notes">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M5 4h14v16H5z" />
            <path d="M8 8h8M8 12h8M8 16h5" />
          </svg>
          <span>Notes</span>
        </button>
        <button type="button" className="doc-rail-item" aria-label="History">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M12 8v5l3 2" />
            <path d="M4 12a8 8 0 1 0 2.35-5.65L4 8" />
            <path d="M4 4v4h4" />
          </svg>
          <span>History</span>
        </button>
        <button type="button" className="doc-rail-item" aria-label="Settings">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M12 15.5A3.5 3.5 0 1 0 12 8a3.5 3.5 0 0 0 0 7.5z" />
            <path d="M19.4 15a1.8 1.8 0 0 0 .36 1.98l.04.04a2 2 0 1 1-2.83 2.83l-.04-.04A1.8 1.8 0 0 0 15 19.4a1.8 1.8 0 0 0-1 .6 1.8 1.8 0 0 0-.5 1.27V21a2 2 0 1 1-4 0v-.06A1.8 1.8 0 0 0 8 19.4a1.8 1.8 0 0 0-1.98.36l-.04.04a2 2 0 1 1-2.83-2.83l.04-.04A1.8 1.8 0 0 0 4.6 15a1.8 1.8 0 0 0-.6-1 1.8 1.8 0 0 0-1.27-.5H2.7a2 2 0 1 1 0-4h.06A1.8 1.8 0 0 0 4.6 8a1.8 1.8 0 0 0-.36-1.98l-.04-.04a2 2 0 1 1 2.83-2.83l.04.04A1.8 1.8 0 0 0 9 4.6a1.8 1.8 0 0 0 1-.6A1.8 1.8 0 0 0 10.5 2.73V2.7a2 2 0 1 1 4 0v.06A1.8 1.8 0 0 0 15 4.6a1.8 1.8 0 0 0 1.98-.36l.04-.04a2 2 0 1 1 2.83 2.83l-.04.04A1.8 1.8 0 0 0 19.4 9c.3.33.6.66 1 .6h.06a2 2 0 1 1 0 4h-.06a1.8 1.8 0 0 0-1 .6z" />
          </svg>
          <span>Settings</span>
        </button>
      </aside>
    </section>
  );
}
