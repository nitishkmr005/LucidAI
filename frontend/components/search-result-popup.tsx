"use client";

import { useEffect } from "react";

type SearchResult = { title: string; snippet: string; url: string };

type SearchResultPopupProps = {
  query: string;
  results: SearchResult[];
  isLoading: boolean;
  onDismiss: () => void;
};

export function SearchResultPopup({ query, results, isLoading, onDismiss }: SearchResultPopupProps) {
  useEffect(() => {
    const timer = setTimeout(onDismiss, 30_000);
    return () => clearTimeout(timer);
  }, [query, onDismiss]);

  return (
    <div className="search-popup surface">
      <div className="search-popup-header">
        <span className="kicker">Web Search</span>
        <button type="button" className="search-popup-close" onClick={onDismiss} aria-label="Dismiss">
          ×
        </button>
      </div>
      <p className="search-popup-query">&ldquo;{query}&rdquo;</p>
      {isLoading ? (
        <p className="search-popup-loading">Searching…</p>
      ) : results.length === 0 ? (
        <p className="search-popup-empty">No results found.</p>
      ) : (
        <ul className="search-popup-list">
          {results.map((r, i) => (
            <li key={i} className="search-popup-item">
              <a href={r.url} target="_blank" rel="noopener noreferrer" className="search-popup-title">
                {r.title}
              </a>
              <p className="search-popup-snippet">{r.snippet}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
