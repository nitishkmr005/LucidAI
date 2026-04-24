"use client";

import { useRef, useState, useCallback } from "react";

type DocumentMeta = {
  doc_id: string;
  filename: string;
  title: string;
  word_count: number;
  sentence_count: number;
  uploaded_at: string;
};

type DocumentUploaderProps = {
  documents: DocumentMeta[];
  selectedDocId: string | null;
  isUploading: boolean;
  readingDocId?: string | null;
  readingSentenceIdx?: number | null;
  onUpload: (file: File) => Promise<void>;
  onSelect: (docId: string) => void;
  onDelete: (docId: string) => void;
  onRefresh: () => Promise<void>;
};

export function DocumentUploader({
  documents,
  selectedDocId,
  isUploading,
  readingDocId = null,
  readingSentenceIdx = null,
  onUpload,
  onSelect,
  onDelete,
}: DocumentUploaderProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const featuredDocument = documents.find((doc) => doc.doc_id === selectedDocId) ?? documents[0] ?? null;

  const handleFile = useCallback(
    async (file: File) => {
      if (!file.name.endsWith(".md")) {
        setUploadError("Only .md (Markdown) files are supported.");
        return;
      }
      setUploadError(null);
      try {
        await onUpload(file);
      } catch (err) {
        setUploadError(err instanceof Error ? err.message : "Upload failed.");
      }
    },
    [onUpload]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const featuredIsReading = Boolean(featuredDocument && readingDocId === featuredDocument.doc_id);
  const estimatedPages = featuredDocument ? Math.max(1, Math.round(featuredDocument.word_count / 260)) : 1;
  const progressPercent = featuredDocument && featuredIsReading && readingSentenceIdx !== null
    ? Math.max(4, Math.min(100, Math.round(((readingSentenceIdx + 1) / Math.max(featuredDocument.sentence_count, 1)) * 100)))
    : 0;
  const featuredStatusLabel = featuredIsReading
    ? "Reading aloud"
    : featuredDocument?.doc_id === selectedDocId
      ? "Selected"
      : "Ready to open";
  const featuredStatusMeta = featuredIsReading && featuredDocument
    ? `${Math.min(readingSentenceIdx ?? 0, featuredDocument.sentence_count - 1) + 1} / ${featuredDocument.sentence_count} sentences`
    : `${estimatedPages.toLocaleString()} ${estimatedPages === 1 ? "page" : "pages"}`;

  const openFilePicker = () => fileInputRef.current?.click();

  return (
    <div className={`doc-uploader${documents.length > 0 ? " has-documents" : ""}`}>
      <input
        ref={fileInputRef}
        type="file"
        accept=".md"
        style={{ display: "none" }}
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); e.target.value = ""; }}
      />

      {/* ── Empty state ── */}
      {documents.length === 0 && (
        <div className="doc-empty-state">
          <div
            className={`doc-drop-zone${isDragOver ? " is-over" : ""}${isUploading ? " is-uploading" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            onClick={openFilePicker}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") openFilePicker(); }}
            aria-label="Upload markdown file"
          >
            <div className="doc-drop-icon" aria-hidden="true">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
              </svg>
            </div>
            <p className="doc-drop-label">
              {isUploading ? "Uploading…" : isDragOver ? "Drop to upload" : "Drop a .md file or click to browse"}
            </p>
            <p className="doc-drop-hint">Markdown files only</p>
            {uploadError && <p className="doc-drop-error">{uploadError}</p>}
          </div>
        </div>
      )}

      {/* ── Has documents ── */}
      {documents.length > 0 && (
        <>
          {/* Featured / selected document */}
          {featuredDocument && (
            <button
              type="button"
              className={`doc-feature-card${selectedDocId === featuredDocument.doc_id ? " is-selected" : ""}${featuredIsReading ? " is-reading" : ""}`}
              onClick={() => onSelect(featuredDocument.doc_id)}
            >
              <span className="doc-feature-icon" aria-hidden="true">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M7 3h7l4 4v14H7z"/>
                  <path d="M14 3v5h5"/>
                  <path d="M10 13h6M10 17h4"/>
                </svg>
              </span>
              <span className="doc-feature-content">
                <strong className="doc-feature-title">{featuredDocument.title}</strong>
                <span className="doc-feature-stats">
                  {featuredDocument.word_count.toLocaleString()} words · {estimatedPages} {estimatedPages === 1 ? "page" : "pages"}
                </span>
                {featuredIsReading && (
                  <span className="doc-progress-track">
                    <span className="doc-progress-fill" style={{ width: `${progressPercent}%` }} />
                  </span>
                )}
              </span>
              <span className={`doc-feature-badge${featuredIsReading ? " is-live" : ""}`}>
                {featuredIsReading && <span className="doc-feature-badge-dot" aria-hidden="true" />}
                {featuredStatusLabel}
              </span>
            </button>
          )}

          {/* Library section header */}
          <div className="doc-list-header">
            <h3 className="doc-list-heading">
              Library
              <span className="doc-list-count">{documents.length}</span>
            </h3>
            <button
              type="button"
              className="doc-add-button"
              onClick={openFilePicker}
              disabled={isUploading}
              title="Upload a new document"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
              </svg>
              {isUploading ? "Uploading…" : "Upload"}
            </button>
          </div>

          <ul className="doc-list">
            {documents.map((doc) => {
              const isThisReading = readingDocId === doc.doc_id;
              const pages = Math.max(1, Math.round(doc.word_count / 260));
              return (
                <li
                  key={doc.doc_id}
                  className={[
                    "doc-list-item",
                    selectedDocId === doc.doc_id ? "is-selected" : "",
                    isThisReading ? "is-reading" : "",
                  ].filter(Boolean).join(" ")}
                >
                  <button
                    type="button"
                    className="doc-list-select"
                    onClick={() => onSelect(doc.doc_id)}
                    title={`${doc.filename} — ${doc.word_count.toLocaleString()} words`}
                  >
                    <span className="doc-list-icon" aria-hidden="true">
                      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M7 3h7l4 4v14H7z"/>
                        <path d="M14 3v5h5"/>
                        <path d="M10 13h6M10 17h4"/>
                      </svg>
                    </span>
                    <span className="doc-list-info">
                      <span className="doc-list-title">{doc.title}</span>
                      <span className="doc-list-meta">
                        {doc.word_count.toLocaleString()} words · {pages} {pages === 1 ? "page" : "pages"}
                      </span>
                    </span>
                    {isThisReading && (
                      <span className="doc-list-reading-badge" aria-label="Currently reading">
                        <span className="doc-list-reading-dot" aria-hidden="true" />
                        Reading
                      </span>
                    )}
                  </button>
                  <button
                    type="button"
                    className="doc-list-delete"
                    onClick={(e) => { e.stopPropagation(); onDelete(doc.doc_id); }}
                    aria-label={`Delete ${doc.title}`}
                    title="Delete"
                  >
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="3 6 5 6 21 6"/>
                      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
                      <path d="M10 11v6M14 11v6M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
                    </svg>
                  </button>
                </li>
              );
            })}
          </ul>

          {uploadError && <p className="doc-drop-error" style={{ marginTop: 10 }}>{uploadError}</p>}
        </>
      )}
    </div>
  );
}
