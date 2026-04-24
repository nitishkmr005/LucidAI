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
  onRefresh,
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
      ? "Selected document"
      : "Ready to open";
  const featuredStatusMeta = featuredIsReading && featuredDocument
    ? `Sentence ${Math.min(readingSentenceIdx ?? 0, featuredDocument.sentence_count - 1) + 1} of ${featuredDocument.sentence_count}`
    : `${estimatedPages.toLocaleString()} pages`;

  return (
    <div className={`doc-uploader${documents.length > 0 ? " has-documents" : ""}`}>
      <input
        ref={fileInputRef}
        type="file"
        accept=".md"
        style={{ display: "none" }}
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); e.target.value = ""; }}
      />
      {documents.length > 0 && (
        <button
          type="button"
          className="doc-add-button"
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading}
        >
          <span aria-hidden="true">+</span>
          {isUploading ? "Uploading..." : "Add document"}
        </button>
      )}

      {featuredDocument && (
        <button
          type="button"
          className={`doc-feature-card${selectedDocId === featuredDocument.doc_id ? " is-selected" : ""}`}
          onClick={() => onSelect(featuredDocument.doc_id)}
        >
          <span className="doc-feature-icon" aria-hidden="true">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
              <path d="M7 3h7l4 4v14H7z" />
              <path d="M14 3v5h5" />
              <path d="M10 13h6M10 17h4" />
            </svg>
          </span>
          <span className="doc-feature-content">
            <strong>{featuredDocument.title}</strong>
            <span>{featuredDocument.word_count.toLocaleString()} words · {Math.max(1, Math.round(featuredDocument.word_count / 260)).toLocaleString()} pages</span>
            {featuredIsReading ? (
              <span className="doc-progress-track"><span style={{ width: `${progressPercent}%` }} /></span>
            ) : null}
            <span className={`doc-feature-footer${featuredIsReading ? " is-live" : ""}`}>
              <span>{featuredStatusLabel}</span>
              <span>{featuredStatusMeta}</span>
            </span>
          </span>
        </button>
      )}

      <div
        className={`doc-drop-zone${isDragOver ? " is-over" : ""}${isUploading ? " is-uploading" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") fileInputRef.current?.click(); }}
        aria-label="Upload markdown file"
      >
        <div className="doc-drop-icon" aria-hidden="true">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
          </svg>
        </div>
        <p className="doc-drop-label">
          {isUploading ? "Uploading…" : isDragOver ? "Drop to upload" : "Drop .md file or click to browse"}
        </p>
        {uploadError && <p className="doc-drop-error">{uploadError}</p>}
      </div>

      {documents.length > 0 && (
        <>
        <h3 className="doc-list-heading">All documents</h3>
        <ul className="doc-list">
          {documents.map((doc) => (
            <li
              key={doc.doc_id}
              className={`doc-list-item${selectedDocId === doc.doc_id ? " is-selected" : ""}`}
            >
              <button
                type="button"
                className="doc-list-select"
                onClick={() => onSelect(doc.doc_id)}
                title={`${doc.filename} — ${doc.word_count} words`}
              >
                <span className="doc-list-title">{doc.title}</span>
                <span className="doc-list-meta">{doc.word_count.toLocaleString()} words</span>
              </button>
              <button
                type="button"
                className="doc-list-delete"
                onClick={(e) => { e.stopPropagation(); onDelete(doc.doc_id); }}
                aria-label={`Delete ${doc.title}`}
                title="Delete document"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
                </svg>
              </button>
            </li>
          ))}
        </ul>
        <button type="button" className="doc-view-all-button" onClick={onRefresh}>View all</button>
        </>
      )}
    </div>
  );
}
