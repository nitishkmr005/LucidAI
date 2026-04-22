"use client";

import { useEffect, useRef, useState, useCallback } from "react";
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

export function DocumentPanel({ event, pendingSnippetTerm, pendingSnippetExplanation, onSelectionChange }: DocumentPanelProps) {
  const [documents, setDocuments] = useState<DocumentMeta[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [sentences, setSentences] = useState<string[]>([]);
  const [docTitle, setDocTitle] = useState<string>("");
  const [activeSentenceIdx, setActiveSentenceIdx] = useState<number | null>(null);
  const [activeWordIdx, setActiveWordIdx] = useState<number | null>(null);
  const [highlights, setHighlights] = useState<Set<number>>(new Set());
  const [snippets, setSnippets] = useState<Map<number, Snippet>>(new Map());
  const [openSnippetIdx, setOpenSnippetIdx] = useState<number | null>(null);
  const [searchQuery, setSearchQuery] = useState<string | null>(null);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearchLoading, setIsSearchLoading] = useState(false);
  const [isReading, setIsReading] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const activeRowRef = useRef<HTMLParagraphElement | null>(null);

  const loadDocuments = useCallback(async () => {
    try {
      const res = await fetch(`${backendUrl}/documents/`);
      if (res.ok) setDocuments(await res.json());
    } catch {}
  }, []);

  useEffect(() => { loadDocuments(); }, [loadDocuments]);

  // Scroll active sentence into view
  useEffect(() => {
    activeRowRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
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
      setActiveSentenceIdx(null);
      setActiveWordIdx(null);
      setIsReading(true);
    } else if (event.type === "doc_opened") {
      setSentences(event.sentences);
      setDocTitle(event.title);
      setSelectedDocId(event.doc_id);
      const hlSet = new Set<number>(event.annotations.highlights.map((h) => h.sentence_idx));
      setHighlights(hlSet);
      const snMap = new Map<number, Snippet>();
      for (const sn of event.annotations.snippets) snMap.set(sn.sentence_idx, sn);
      setSnippets(snMap);
    } else if (event.type === "doc_highlight") {
      setActiveSentenceIdx(event.sentence_idx);
      setActiveWordIdx(null);
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
  }, [event]);

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
        setSnippets((prev) => new Map(prev).set(idx, sn));
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
    const res = await fetch(`${backendUrl}/documents/${docId}`);
    if (!res.ok) return;
    const data = await res.json();
    setSelectedDocId(docId);
    onSelectionChange?.(docId);
    setSentences(data.sentences);
    setDocTitle(data.title);
    setActiveSentenceIdx(null);
    setActiveWordIdx(null);
    const hlSet = new Set<number>((data.annotations?.highlights ?? []).map((h: Annotation) => h.sentence_idx));
    setHighlights(hlSet);
    const snMap = new Map<number, Snippet>();
    for (const sn of (data.annotations?.snippets ?? [])) snMap.set(sn.sentence_idx, sn);
    setSnippets(snMap);
  }, [onSelectionChange]);

  const handleDeleteDoc = useCallback(async (docId: string) => {
    await fetch(`${backendUrl}/documents/${docId}`, { method: "DELETE" });
    if (selectedDocId === docId) {
      setSelectedDocId(null);
      onSelectionChange?.(null);
      setSentences([]);
      setDocTitle("");
    }
    await loadDocuments();
  }, [selectedDocId, loadDocuments, onSelectionChange]);

  const handleHighlightSentence = useCallback((idx: number) => {
    if (!selectedDocId) return;
    fetch(`${backendUrl}/documents/${selectedDocId}/annotations/highlight`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentence_idx: idx }),
    }).then(() => setHighlights((prev) => new Set(prev).add(idx))).catch(() => {});
  }, [selectedDocId]);

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

  const renderSentence = (sentence: string, idx: number) => {
    const isActive = activeSentenceIdx === idx;
    const isHighlighted = highlights.has(idx);
    const hasSnippet = snippets.has(idx);

    return (
      <p
        key={idx}
        ref={isActive ? activeRowRef : null}
        className={[
          "doc-sentence",
          isActive ? "doc-sentence--reading" : "",
          isHighlighted ? "doc-sentence--highlighted" : "",
        ].filter(Boolean).join(" ")}
        data-sentence-idx={idx}
      >
        {renderWords(sentence, isActive)}
        {" "}
        <button
          type="button"
          className="doc-sentence-highlight-btn"
          onClick={() => handleHighlightSentence(idx)}
          title="Highlight sentence"
          aria-label="Highlight this sentence"
        >
          ✦
        </button>
        {hasSnippet && (
          <button
            type="button"
            className="doc-snippet-indicator"
            onClick={() => setOpenSnippetIdx(openSnippetIdx === idx ? null : idx)}
            title="View saved note"
            aria-label="View saved note"
          >
            💡
          </button>
        )}
        {openSnippetIdx === idx && hasSnippet && (
          <span className="doc-snippet-tooltip">
            <strong>{snippets.get(idx)?.term}:</strong> {snippets.get(idx)?.explanation}
          </span>
        )}
      </p>
    );
  };

  const hasDocument = sentences.length > 0;

  return (
    <section className="doc-panel surface">
      <header className="doc-panel-header">
        <div>
          <p className="kicker">Document Reader</p>
          {docTitle && <h2 className="doc-panel-title">{docTitle}</h2>}
        </div>
        <div className="doc-panel-status">
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
            <div className="doc-sentences">
              {sentences.map((s, i) => renderSentence(s, i))}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
