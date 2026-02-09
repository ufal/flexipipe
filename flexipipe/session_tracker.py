"""
Session tracking for flexipipe training and tagging operations.

This module tracks active training and tagging sessions, allowing users to
monitor long-running operations and see what flexipipe processes are currently active.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .model_storage import get_flexipipe_config_dir


@dataclass
class SessionInfo:
    """Information about an active flexipipe session."""
    
    session_id: str
    command: str  # "train" or "process" (tag)
    pid: int
    start_time: float
    backend: Optional[str] = None
    model: Optional[str] = None
    language: Optional[str] = None
    input_file: Optional[str] = None
    output_file: Optional[str] = None
    output_dir: Optional[str] = None
    status: str = "running"  # "running", "completed", "failed"
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "SessionInfo":
        """Create from dictionary."""
        return cls(**data)
    
    @property
    def duration(self) -> float:
        """Get session duration in seconds."""
        return time.time() - self.start_time
    
    @property
    def is_running(self) -> bool:
        """Check if the process is still running."""
        try:
            # Check if process exists (works on Unix and Windows)
            os.kill(self.pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def get_sessions_dir() -> Path:
    """Get the directory where session files are stored."""
    config_dir = get_flexipipe_config_dir(create=True)
    sessions_dir = config_dir / "sessions"
    sessions_dir.mkdir(exist_ok=True)
    return sessions_dir


def create_session(
    command: str,
    *,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    language: Optional[str] = None,
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> SessionInfo:
    """
    Create a new session record.
    
    Args:
        command: Command type ("train" or "process")
        backend: Backend being used
        model: Model being used
        language: Language code
        input_file: Input file path
        output_file: Output file path
        output_dir: Output directory (for training)
        
    Returns:
        SessionInfo object
    """
    session_id = f"{command}_{int(time.time())}_{os.getpid()}"
    session = SessionInfo(
        session_id=session_id,
        command=command,
        pid=os.getpid(),
        start_time=time.time(),
        backend=backend,
        model=model,
        language=language,
        input_file=str(input_file) if input_file else None,
        output_file=str(output_file) if output_file else None,
        output_dir=str(output_dir) if output_dir else None,
    )
    
    sessions_dir = get_sessions_dir()
    session_file = sessions_dir / f"{session_id}.json"
    
    try:
        with session_file.open("w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, indent=2)
    except (OSError, PermissionError):
        # If we can't write session file, that's okay - just continue
        pass
    
    return session


def update_session(
    session_id: str,
    *,
    status: Optional[str] = None,
    error: Optional[str] = None,
) -> bool:
    """
    Update an existing session.
    
    Args:
        session_id: Session ID to update
        status: New status (optional)
        error: Error message if failed (optional)
        
    Returns:
        True if session was updated, False if not found
    """
    sessions_dir = get_sessions_dir()
    session_file = sessions_dir / f"{session_id}.json"
    
    if not session_file.exists():
        return False
    
    try:
        with session_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        session = SessionInfo.from_dict(data)
        
        if status:
            session.status = status
        if error:
            session.error = error
        
        with session_file.open("w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, indent=2)
        
        return True
    except (OSError, PermissionError, json.JSONDecodeError):
        return False


def delete_session(session_id: str) -> bool:
    """
    Delete a session file.
    
    Args:
        session_id: Session ID to delete
        
    Returns:
        True if deleted, False if not found
    """
    sessions_dir = get_sessions_dir()
    session_file = sessions_dir / f"{session_id}.json"
    
    try:
        if session_file.exists():
            session_file.unlink()
            return True
        return False
    except (OSError, PermissionError):
        return False


def list_sessions(*, include_completed: bool = False, cleanup_stale: bool = True) -> List[SessionInfo]:
    """
    List all active sessions.
    
    Args:
        include_completed: If True, include completed/failed sessions
        cleanup_stale: If True, automatically remove stale session files (processes that no longer exist)
        
    Returns:
        List of SessionInfo objects
    """
    sessions_dir = get_sessions_dir()
    if not sessions_dir.exists():
        return []
    
    sessions: List[SessionInfo] = []
    stale_sessions: List[str] = []
    
    for session_file in sessions_dir.glob("*.json"):
        try:
            with session_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            session = SessionInfo.from_dict(data)
            
            # Check if process is still running
            if not session.is_running:
                if cleanup_stale:
                    stale_sessions.append(session.session_id)
                elif include_completed:
                    # Mark as completed if not already marked
                    if session.status == "running":
                        session.status = "completed"
                    sessions.append(session)
                continue
            
            # Only include running sessions unless include_completed is True
            if session.status == "running" or include_completed:
                sessions.append(session)
        except (OSError, PermissionError, json.JSONDecodeError, KeyError):
            # Corrupted or invalid session file - mark for cleanup
            if cleanup_stale:
                try:
                    session_file.unlink()
                except (OSError, PermissionError):
                    pass
            continue
    
    # Clean up stale sessions
    if cleanup_stale and stale_sessions:
        for session_id in stale_sessions:
            delete_session(session_id)
    
    # Sort by start time (newest first)
    sessions.sort(key=lambda s: s.start_time, reverse=True)
    
    return sessions


def cleanup_stale_sessions() -> int:
    """
    Clean up all stale session files (processes that no longer exist).
    
    Returns:
        Number of sessions cleaned up
    """
    sessions = list_sessions(include_completed=True, cleanup_stale=True)
    # The cleanup happens in list_sessions when cleanup_stale=True
    # Count how many were actually removed by checking what's left
    remaining = list_sessions(include_completed=True, cleanup_stale=False)
    return len(sessions) - len(remaining)
