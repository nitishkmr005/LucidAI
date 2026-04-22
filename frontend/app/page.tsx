"use client";

import { useState, useCallback } from "react";
import { VoiceAgentConsole } from "../components/voice-agent-console";
import { DocumentPanel } from "../components/document-panel";
import type { DocumentEvent } from "../components/document-panel";

export default function Home() {
  const [latestDocEvent, setLatestDocEvent] = useState<DocumentEvent | null>(null);
  const [pendingSnippetTerm, setPendingSnippetTerm] = useState<string | null>(null);
  const [pendingSnippetExplanation, setPendingSnippetExplanation] = useState<string | null>(null);
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null);

  const handleDocumentEvent = useCallback((event: DocumentEvent) => {
    setLatestDocEvent(event);
    if (event.type === "doc_save_snippet") {
      setPendingSnippetTerm(event.term);
    }
  }, []);

  const handleSnippetExplanation = useCallback((explanation: string) => {
    setPendingSnippetExplanation(explanation);
  }, []);

  return (
    <div className="app-shell">
      <VoiceAgentConsole
        selectedDocumentId={selectedDocumentId}
        onDocumentEvent={handleDocumentEvent}
        onSnippetExplanation={handleSnippetExplanation}
      >
        <DocumentPanel
          event={latestDocEvent}
          pendingSnippetTerm={pendingSnippetTerm}
          pendingSnippetExplanation={pendingSnippetExplanation}
          onSelectionChange={setSelectedDocumentId}
        />
      </VoiceAgentConsole>
    </div>
  );
}
