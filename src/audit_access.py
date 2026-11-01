
AUTHORIZED_AUDIT_ROLES = frozenset({
    "security_auditor",
    "compliance_officer",
})


def get_audit_record(result, user_role):
    """Return audit data only to authorized internal roles."""

    if not isinstance(user_role, str):
        raise PermissionError("Audit access denied.")

    if user_role not in AUTHORIZED_AUDIT_ROLES:
        raise PermissionError("Audit access denied.")

    if not isinstance(result, dict):
        raise ValueError("Invalid audit result.")

    if "audit_record" not in result:
        raise ValueError("Audit record is missing.")

    return result["audit_record"]
