"""FastAPI router for WebRTC signalling (SDP offer/answer)."""

from uuid import uuid4

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from app.webrtc.session import WebRTCSession

router = APIRouter(prefix="/webrtc", tags=["webrtc"])

# In-process session registry.  Sessions are cleaned up when the peer connection
# transitions to "failed", "closed", or "disconnected" (see WebRTCSession._cleanup).
_sessions: dict[str, WebRTCSession] = {}


class OfferBody(BaseModel):
    sdp: str
    type: str


class OfferResponse(BaseModel):
    sdp: str
    type: str
    session_id: str


@router.post("/offer", response_model=OfferResponse)
async def webrtc_offer(body: OfferBody) -> OfferResponse:
    """
    Exchange SDP offer for answer and create a new WebRTC session.

    The browser creates an ``RTCPeerConnection``, adds the mic audio track,
    creates a data channel named ``"signaling"``, generates an offer and POSTs
    it here.  The returned answer SDP is set as the remote description on the
    browser side, completing the JSEP handshake.

    Args:
        body: JSON body with ``sdp`` (offer SDP string) and ``type``
              (``"offer"``).

    Returns:
        JSON with ``sdp`` (answer), ``type`` (``"answer"``) and ``session_id``.
    """
    session_id = uuid4().hex[:8]
    logger.info("session_id={} event=webrtc_offer_received", session_id)
    session = WebRTCSession(session_id)
    answer = await session.setup(body.sdp, body.type)
    _sessions[session_id] = session
    logger.info("session_id={} event=webrtc_answer_sent", session_id)
    return OfferResponse(sdp=answer.sdp, type=answer.type, session_id=session_id)


@router.delete("/session/{session_id}", status_code=204)
async def close_session(session_id: str) -> None:
    """
    Explicitly close a WebRTC session.

    The browser should call this when the user presses Stop so the server-side
    peer connection is torn down cleanly even if ICE/DTLS signalling lags.

    Args:
        session_id: The ID returned by ``POST /webrtc/offer``.
    """
    session = _sessions.pop(session_id, None)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    await session.pc.close()
    logger.info("session_id={} event=webrtc_session_deleted", session_id)
