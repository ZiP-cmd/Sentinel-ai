
import sys
from pathlib import Path

import pytest

# Tell Python where our source files are located.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Import our audit session module.
import audit_sessions

from audit_sessions import (
    create_demo_session,
    end_demo_session,
    get_authenticated_role,
    get_authenticated_audit_record,
)


@pytest.fixture
def audit_result():
    return {
        "investigator_view": {
            "status": "filtered",
        },
        "audit_record": {
            "event_id": "audit-test-001",
        },
    }


def test_security_auditor_session_allowed(audit_result):
    token = create_demo_session("sarah")

    try:
        record = get_authenticated_audit_record(
            audit_result, token
        )
        assert record["event_id"] == "audit-test-001"
    finally:
        end_demo_session(token)


def test_investigator_session_denied(audit_result):
    token = create_demo_session("james")

    try:
        with pytest.raises(PermissionError):
            get_authenticated_audit_record(
                audit_result, token
            )
    finally:
        end_demo_session(token)


def test_compliance_officer_session_allowed(audit_result):
    token = create_demo_session("michelle")

    try:
        record = get_authenticated_audit_record(
            audit_result, token
        )
        assert record["event_id"] == "audit-test-001"
    finally:
        end_demo_session(token)


def test_ended_session_denied(audit_result):
    token = create_demo_session("sarah")
    end_demo_session(token)

    with pytest.raises(PermissionError):
        get_authenticated_audit_record(
            audit_result, token
        )


def test_unknown_user_rejected():
    with pytest.raises(ValueError):
        create_demo_session("unknown_user")


def test_session_resolves_server_side_role():
    token = create_demo_session("james")

    try:
        assert get_authenticated_role(token) == "investigator"
    finally:
        end_demo_session(token)

def test_revoked_auditor_loses_access_immediately(
    audit_result,
    monkeypatch,
):
    # Sarah initially has auditing permissions.
    token = create_demo_session("sarah")

    try:
        # Confirm Sarah can access the audit record.
        record = get_authenticated_audit_record(
            audit_result,
            token,
        )

        assert record["event_id"] == "audit-test-001"

        # Simulate an authoritative role change.
        # monkeypatch restores the original value
        # automatically after this test.
        from audit_sessions import DEMO_USERS

        monkeypatch.setitem(
            DEMO_USERS,
            "sarah",
            "investigator",
        )

        # Sarah is still logged in.
        assert get_authenticated_role(token) == "investigator"

        # But the existing session must no longer
        # provide access to restricted audit records.
        with pytest.raises(PermissionError):
            get_authenticated_audit_record(
                audit_result,
                token,
            )

    finally:
        end_demo_session(token)

def test_authorization_outage_denies_audit_access(
    audit_result,
    monkeypatch,
):
    token = create_demo_session("sarah")

    def unavailable_lookup(username):
        raise ConnectionError(
            "Authorization database unavailable"
        )

    monkeypatch.setattr(
        audit_sessions,
        "lookup_current_role",
        unavailable_lookup,
    )

    try:
        with pytest.raises(
            PermissionError,
            match="Authorization temporarily unavailable",
        ):
            get_authenticated_audit_record(
                audit_result,
                token,
            )
    finally:
        end_demo_session(token)


def test_access_restored_after_authorization_recovers(
    audit_result,
    monkeypatch,
):
    token = create_demo_session("sarah")

    def unavailable_lookup(username):
        raise ConnectionError(
            "Authorization database unavailable"
        )

    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                audit_sessions,
                "lookup_current_role",
                unavailable_lookup,
            )

            with pytest.raises(PermissionError):
                get_authenticated_audit_record(
                    audit_result,
                    token,
                )

        # The temporary outage is over.
        # Sarah's existing session is still active.
        record = get_authenticated_audit_record(
            audit_result,
            token,
        )

        assert record["event_id"] == "audit-test-001"

    finally:
        end_demo_session(token)
