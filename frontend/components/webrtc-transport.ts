/**
 * WebRTC transport for NeuroTalk.
 *
 * Browser mic audio is sent to the backend as an RTP audio track (Opus codec,
 * handled transparently by the browser).  All JSON signalling messages travel
 * over an ordered RTCDataChannel named "signaling" — the same schema used by
 * the existing WebSocket path, so the frontend message-handler is reusable.
 *
 * ICE candidates are gathered fully before the offer is sent (vanilla ICE) so
 * no trickle-ICE endpoint is needed.  Works well for localhost and STUN-reachable
 * peers; for production behind strict NAT, add TURN servers to STUN_SERVERS.
 */

const STUN_SERVERS: RTCIceServer[] = [
  { urls: "stun:stun.l.google.com:19302" },
  { urls: "stun:stun1.l.google.com:19302" },
];

// Matches the StreamMessage type in voice-agent-console.tsx.
export type RTCStreamMessage = {
  type: string;
  [key: string]: unknown;
};

export class WebRTCTransport {
  private pc: RTCPeerConnection | null = null;
  private dc: RTCDataChannel | null = null;
  private sessionId: string | null = null;
  private readonly backendUrl: string;

  /** Called for every JSON message arriving on the data channel. */
  onMessage: ((msg: RTCStreamMessage) => void) | null = null;

  /** Called when the data channel or peer connection closes. */
  onClose: (() => void) | null = null;

  constructor(backendUrl: string) {
    this.backendUrl = backendUrl;
  }

  /** ``true`` when the data channel is open and ready to send. */
  get isOpen(): boolean {
    return this.dc?.readyState === "open";
  }

  get currentSessionId(): string | null {
    return this.sessionId;
  }

  /**
   * Connect to the backend:
   * 1. Create RTCPeerConnection and add the mic audio track.
   * 2. Create the "signaling" data channel.
   * 3. Wait for ICE gathering to finish, then POST the offer SDP.
   * 4. Apply the answer SDP.
   * 5. Wait for the data channel to open.
   *
   * @param stream  MediaStream obtained from ``getUserMedia``.  Its first audio
   *                track is added to the peer connection and sent via RTP.
   */
  async connect(stream: MediaStream, options?: { voice?: string }): Promise<void> {
    this.pc = new RTCPeerConnection({ iceServers: STUN_SERVERS });

    // Add mic track → browser will negotiate and send Opus RTP to the server.
    for (const track of stream.getAudioTracks()) {
      this.pc.addTrack(track, stream);
    }

    // Ordered data channel for JSON signalling (same schema as WebSocket).
    this.dc = this.pc.createDataChannel("signaling", { ordered: true });

    this.dc.onmessage = (event: MessageEvent) => {
      try {
        const msg = JSON.parse(event.data as string) as RTCStreamMessage;
        this.onMessage?.(msg);
      } catch {
        // ignore malformed frames
      }
    };

    this.dc.onclose = () => {
      this.onClose?.();
    };

    this.pc.onconnectionstatechange = () => {
      const state = this.pc?.connectionState;
      if (state === "failed" || state === "disconnected" || state === "closed") {
        this.onClose?.();
      }
    };

    // Create offer
    const offer = await this.pc.createOffer();
    await this.pc.setLocalDescription(offer);

    // Wait for ICE gathering to complete (all candidates are embedded in the SDP).
    // Timeout after 4 s in case a candidate stalls (common in strict NAT environments).
    await this._waitForIceGathering(4000);

    const resp = await fetch(`${this.backendUrl}/webrtc/offer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sdp: this.pc.localDescription!.sdp,
        type: this.pc.localDescription!.type,
        voice: options?.voice,
      }),
    });

    if (!resp.ok) {
      throw new Error(`WebRTC offer rejected by server: HTTP ${resp.status}`);
    }

    const { sdp, type, session_id } = (await resp.json()) as {
      sdp: string;
      type: RTCSdpType;
      session_id: string;
    };
    this.sessionId = session_id;

    await this.pc.setRemoteDescription(new RTCSessionDescription({ sdp, type }));

    // Block until the data channel is open (server sends "ready" shortly after).
    await this._waitForDcOpen(10_000);
  }

  /**
   * Send a JSON-serialisable message via the data channel.
   *
   * @param message  Object to send.  Silently dropped if the channel is not open.
   */
  send(message: object): void {
    if (this.dc?.readyState === "open") {
      this.dc.send(JSON.stringify(message));
    }
  }

  /**
   * Close the data channel and peer connection.
   *
   * Triggers a ``DELETE /webrtc/session/{id}`` to let the server tear down its
   * side cleanly.  Fire-and-forget: the UI does not wait for this.
   */
  close(): void {
    if (this.sessionId) {
      const id = this.sessionId;
      const url = `${this.backendUrl}/webrtc/session/${id}`;
      // best-effort; don't await
      fetch(url, { method: "DELETE", keepalive: true }).catch(() => undefined);
    }
    try { this.dc?.close(); } catch { /* ignore */ }
    try { this.pc?.close(); } catch { /* ignore */ }
    this.dc = null;
    this.pc = null;
    this.sessionId = null;
  }

  // ── Private helpers ───────────────────────────────────────────────────────

  private _waitForIceGathering(timeoutMs: number): Promise<void> {
    return new Promise<void>((resolve) => {
      if (this.pc!.iceGatheringState === "complete") {
        resolve();
        return;
      }
      const timer = setTimeout(resolve, timeoutMs);
      this.pc!.onicegatheringstatechange = () => {
        if (this.pc!.iceGatheringState === "complete") {
          clearTimeout(timer);
          resolve();
        }
      };
    });
  }

  private _waitForDcOpen(timeoutMs: number): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      if (this.dc!.readyState === "open") {
        resolve();
        return;
      }
      const timer = setTimeout(
        () => reject(new Error("Timed out waiting for data channel to open")),
        timeoutMs,
      );
      this.dc!.onopen = () => {
        clearTimeout(timer);
        resolve();
      };
    });
  }
}
