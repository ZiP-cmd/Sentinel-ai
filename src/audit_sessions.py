
import secrets

from audit_access import get_audit_record


# Demo users and server-maintained roles.
DEMO_USERS = {
    "james": "investigator",
    "sarah": "security_auditor",
    "michelle": "compliance_officer",
}

# Sessions are maintained on the application side.
_active_sessions = {}

def lookup_current_role(username):
    """
    Demo authoritative role lookup.

    In production, this would query a trusted
    authorization service or database.
    """
    return DEMO_USERS.get(username)


def create_demo_session(username):
    if username not in DEMO_USERS:
        raise ValueError("Unknown demo user.")

    session_token = secrets.token_urlsafe(32)
    _active_sessions[session_token] = username

    return session_token


def end_demo_session(session_token):
    _active_sessions.pop(session_token, None)


def get_authenticated_role(session_token):
    username = _active_sessions.get(session_token)

    if username is None:
        raise PermissionError(
            "Invalid or expired session."
        )

    try:
        role = lookup_current_role(username)
    except (ConnectionError, TimeoutError) as exc:
        raise PermissionError(
            "Authorization temporarily unavailable."
        ) from exc

    if role is None:
        raise PermissionError(
            "Current permissions could not be verified."
        )

    return role



def get_authenticated_audit_record(result, session_token):
    role = get_authenticated_role(session_token)

    return get_audit_record(result, role)
