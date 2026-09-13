"""Layer 2 corpus: 500 *distinct* knowledge-base documents + eval queries.

Design goals:
  • 25 topic families × 20 variants each = 500 documents. Each variant is a
    genuinely different article on the same topic: unique title, unique body
    angle, same family + concepts + source. No near-duplicate embedding flood.
  • Each document carries `concepts` (knowledge-graph nodes) and `source`.
  • Concept registry maps concepts → aliases so the graph layer can match a
    natural-language query to concept nodes (word-boundary alias matching).
  • Eval queries are phrased *differently* from the document titles
    (rephrasing), unambiguously pointing at one family.

Relevance for evaluation: a test query's gold set == every document of its
family. precision@5 = gold docs inside the returned top-5 (ratio of 5).
"""
from __future__ import annotations

import random
from itertools import product
from typing import Any, Dict, List

# concept name → aliases used to match natural-language queries to graph nodes
CONCEPT_REGISTRY: Dict[str, List[str]] = {
    "password":    ["reset", "password", "credential", "login", "locked"],
    "billing":     ["billing", "invoice", "charge", "payment", "card"],
    "refund":      ["refund", "refunds", "money back", "return"],
    "api":         ["api", "apikey", "key", "rate limit", "token"],
    "plan":        ["plan", "upgrade", "downgrade", "pricing", "tier", "subscription"],
    "shipping":    ["shipping", "delivery", "tracking", "order"],
    "sso":         ["sso", "identity", "saml", "oidc", "single sign-on"],
    "export":      ["export", "download", "backup", "data"],
    "security":    ["security", "2fa", "two-factor", "multifactor", "totp"],
    "sla":         ["sla", "uptime", "availability", "latency", "outage", "downtime", "compensation", "credit"],
    "webhook":     ["webhook", "integration", "event", "callback"],
    "team":        ["team", "role", "member", "invite", "permission", "seat"],
    "compliance":  ["compliance", "gdpr", "audit", "regulation"],
    "retention":   ["retention", "delete", "purge", "erase"],
    "deployment":  ["deployment", "cluster", "helm", "k8s", "kubernetes"],
}

SERVICES = ["groq", "gemini", "redis", "docker", "mysql", "neo4j", "qdrant", "fastembed", "bifrost", "portal"]
PLANS = ["Starter", "Pro"]


def _title_variants(fam: Dict[str, Any]) -> List[str]:
    """Several genuinely different phrasings of the family's article titles."""
    base = fam["title"]  # e.g. "How to reset the {svc} password"
    # deterministically generate alternate phrasings from a per-family list
    alternates = fam.get("title_alternates", [])
    if not alternates:
        return [base]
    return [base] + alternates


def _detail_variants(fam: Dict[str, Any]) -> List[str]:
    """Angles appended to the body so every variant is a distinct article."""
    return fam.get("details", [fam["body"]])


def _variants(fam: Dict[str, Any]) -> List[Dict[str, str]]:
    """Exactly 20 slot combos per family (10 services × 2 plans)."""
    titles = _title_variants(fam)
    details = _detail_variants(fam)
    combos = list(product(SERVICES, PLANS))
    out = []
    for i, (s, p) in enumerate(combos):
        t = titles[i % len(titles)].format(svc=s, plan=p, plan2=("Enterprise" if p == "Pro" else "Pro"))
        body = fam["body"].format(svc=s, plan=p, plan2=("Enterprise" if p == "Pro" else "Pro"))
        detail = details[i % len(details)]
        d_txt = detail.format(svc=s, plan=p, plan2=("Enterprise" if p == "Pro" else "Pro")) \
            if "{" in detail else detail
        out.append({"svc": s, "plan": p, "title": t, "body": f"{body} {d_txt}".strip()})
    return out


def build_corpus(seed: int = 11) -> Dict[str, Any]:
    """Return {documents: [...], registry: family→doc_ids} — deterministic."""
    rng = random.Random(seed)
    documents: List[Dict[str, Any]] = []
    registry: Dict[str, List[str]] = {}
    idx = 0
    for fam in FAMILIES:
        ids: List[str] = []
        for slots in _variants(fam):
            doc_id = f"DOC-{idx:03d}"
            idx += 1
            documents.append({
                "doc_id": doc_id,
                "title": slots["title"],
                "body": slots["body"],
                "family": fam["family"],
                "concepts": list(fam["concepts"]),
                "source": fam["source"],
                "tags": list(fam["concepts"]),
            })
            ids.append(doc_id)
        registry[fam["family"]] = ids

    queries = _build_eval_queries()
    return {"documents": documents, "registry": registry, "queries": queries}


FAMILIES: List[Dict[str, Any]] = [
    {
        "family": "password-reset", "concepts": ["password", "security"], "source": "kb",
        "title": "How to reset the {svc} password",
        "title_alternates": [
            "Recovering access to your locked {svc} account",
            "Password recovery options on {svc}",
        ],
        "body": ("To reset the {svc} password, open the account menu, choose Security, and click "
                 "Reset password. A verification link is emailed to the account owner and expires "
                 "after 15 minutes."),
        "details": [
            "If the link has expired, request a fresh one from the login page — support cannot "
            "recover it for you.",
            "Support can force a reset for locked accounts only after identity verification via a "
            "video call.",
            "Reset links are single-use and invalidated as soon as you set the new password.",
            "Password history is enforced: the last 5 passwords cannot be reused on this account.",
            "Team owners see a reset-requests audit entry; members never see each other's tokens.",
            "For SSO-enabled workspaces, reset the upstream identity provider password instead.",
            "Password managers work with the reset flow as long as the new password is saved before "
            "the session expires.",
            "Resets invalidate all existing sessions and API tokens issued before the change.",
        ],
    },
    {
        "family": "billing-invoices", "concepts": ["billing"], "source": "faq",
        "title": "Where do I find the {svc} monthly invoice?",
        "title_alternates": ["Accessing billing statements on {svc}"],
        "body": ("Monthly invoices for {svc} appear in Billing > Invoices within 24 hours after the "
                 "billing cycle closes. Download PDF or CSV copies for the last 24 months."),
        "details": [
            "Invoices include line items per seat, per storage GB, and per API call batch.",
            "Discrepancies can be disputed within 90 days of the invoice date via the billing portal.",
            "Tax receipts are regenerated at the address stored on the account profile.",
            "CSV exports include the usage ledger so finance can reconcile against their systems.",
            "Enterprise contracts receive a consolidated invoice covering all linked workspaces.",
        ],
    },
    {
        "family": "refund-policy", "concepts": ["refund", "billing"], "source": "faq",
        "title": "{svc} refund policy for {plan} plans",
        "title_alternates": ["Getting a subscription refund on {svc}"],
        "body": ("{svc} offers full refunds within 14 days of purchase on {plan} plans, and pro-rata "
                 "refunds for unused time afterwards. Refund requests are processed in 5-7 business "
                 "days to the original payment method."),
        "details": [
            "Annual plans refund only the unused portion of the prepaid term.",
            "Refunds cancel the subscription immediately and revoke all seats.",
            "Usage overrides: if you consumed more than 10% of the quota, the refund is pro-rated.",
            "Promotional purchases are refunded at the discounted amount actually paid.",
            "Chargebacks bypass the refund policy and can suspend the account during review.",
        ],
    },
    {
        "family": "api-rate-limits", "concepts": ["api"], "source": "docs",
        "title": "{svc} API rate limits on the {plan} tier",
        "title_alternates": ["API throttling rules for {svc}", "Understanding {svc} request quotas"],
        "body": ("The {svc} API enforces 60 requests per minute on {plan} tiers and 600 on higher "
                 "tiers. Exceeding the limit returns HTTP 429 with a Retry-After header."),
        "details": [
            "Batch endpoints count each item inside the batch against the quota.",
            "Limits reset every minute at the top of the UTC minute boundary.",
            "The X-RateLimit-Remaining header lets clients pre-empt throttling.",
            "Server-side retries must honor Retry-After; ignoring it can ban the key.",
            "Rate limit overrides are configurable in the developer console for approved apps.",
        ],
    },
    {
        "family": "plan-upgrades", "concepts": ["plan", "billing"], "source": "faq",
        "title": "How to upgrade from {plan} to {plan2} on {svc}",
        "title_alternates": ["Switching to a higher {svc} tier", "Moving up a {svc} plan mid-cycle"],
        "body": ("Go to Billing > Plans on {svc}, choose {plan2}, and confirm the proration amount. "
                 "Upgrades apply immediately with no downtime; downgrades take effect at the next "
                 "billing cycle."),
        "details": [
            "Seat-based plans bill the upgraded rate for new seats at once, not at renewal.",
            "Feature gates unlock within a minute of the upgrade completing.",
            "Downgrading to {plan} keeps existing data for 30 days before archiving.",
            "Upgrades made mid-cycle credit the remaining value of the old plan.",
            "Contract customers should route tier changes through their account manager.",
        ],
    },
    {
        "family": "shipping-tracking", "concepts": ["shipping"], "source": "faq",
        "title": "Track a {svc} hardware order",
        "title_alternates": ["Where is my {svc} hardware delivery?"],
        "body": ("After checkout, {svc} emails a tracking link within 2 hours. The carrier updates "
                 "status on delivery attempts."),
        "details": [
            "If tracking shows no movement for 5 days, open a ticket with the order number.",
            "Signature-on-delivery is required for orders above $500.",
            "International orders add customs handling time visible in the tracker.",
            "You can reschedule a delivery from the carrier link up to 10 pm local time.",
            "Missing items must be reported within 48 hours of the delivered timestamp.",
        ],
    },
    {
        "family": "sso-setup", "concepts": ["sso", "security"], "source": "docs",
        "title": "Configure SSO for {svc} workspaces",
        "title_alternates": ["Single sign-on setup guide for {svc}"],
        "body": ("Under Workspace Settings > Authentication, {svc} supports SAML 2.0 and OIDC. "
                 "Provide the IdP metadata URL, map the email claim to the primary identity, and "
                 "test with a small pilot group before enforcing SSO-only access."),
        "details": [
            "SCIM provisioning keeps groups in sync between the IdP and {svc}.",
            "SAML logout bindings are supported for Okta, Entra ID, and Google Workspace.",
            "Enforce SSO-only after the pilot: local passwords are disabled for everyone.",
            "Break-glass access: two designated admins keep an emergency local account.",
            "IdP issuer changes require re-uploading the metadata within 7 days.",
        ],
    },
    {
        "family": "data-export", "concepts": ["export", "compliance"], "source": "kb",
        "title": "Export all data from {svc}",
        "title_alternates": ["Porting data out of {svc}"],
        "body": ("Admin > Data > Export on {svc} produces a downloadable archive of all workspace "
                 "content within 12 hours. Exports include audit metadata and attachments."),
        "details": [
            "Formats: JSON, CSV, and a full HTML bundle for static archiving.",
            "Export requests carry a per-user quota of five active exports.",
            "File attachments are archived to durable object storage before packaging.",
            "GDPR erasure requests use the same page's delete flow after export.",
            "Export status is sent to the requesting admin's email when complete.",
        ],
    },
    {
        "family": "two-factor-auth", "concepts": ["security"], "source": "kb",
        "title": "Enable two-factor authentication on {svc}",
        "title_alternates": ["Turning on 2FA for your {svc} account"],
        "body": ("Security > Two-factor on {svc} supports TOTP authenticator apps and hardware "
                 "security keys. Enabling 2FA signs out all other sessions and generates 10 "
                 "single-use recovery codes."),
        "details": [
            "Store recovery codes offline — support cannot circumvent them.",
            "Hardware keys work with WebAuthn on Chrome, Edge, and Safari.",
            "2FA is enforced for admin roles on enterprise workspaces.",
            "Re-authentication is required every 30 days for security-sensitive actions.",
            "Lost devices: use a recovery code, then enroll a new authenticator.",
        ],
    },
    {
        "family": "sla-uptime", "concepts": ["sla"], "source": "docs",
        "title": "{svc} uptime SLA on {plan} plans",
        "title_alternates": ["Availability commitment on {svc}"],
        "body": ("{svc} guarantees 99.9% monthly uptime on {plan}. Downtime credits accrue at 5% "
                 "of the monthly fee per full hour above 27 minutes of unavailability."),
        "details": [
            "Maintenance windows are announced 72 hours ahead and excluded from the SLA.",
            "Credits are issued automatically on the next invoice; no claim needed.",
            "Response-time SLAs apply separately to support, not to the control plane.",
            "The status page publishes incident reports with post-mortem links.",
            "Multi-region failover targets raise the guarantee to 99.99% on enterprise.",
        ],
    },
    {
        "family": "webhooks-integration", "concepts": ["webhook", "api"], "source": "docs",
        "title": "Webhook event delivery on {svc}",
        "title_alternates": ["Subscribing to {svc} events"],
        "body": ("{svc} delivers webhook events over HTTPS with HMAC-SHA256 signatures. Events are "
                 "sent with a single at-least-once delivery and 10 retries with exponential backoff."),
        "details": [
            "Verify the signature header before processing, and respond 2xx within 5 seconds.",
            "Retry intervals start at 30 seconds and double up to a 24-hour cap.",
            "Event payloads include a unique event id for idempotent handling.",
            "Dead-letter queues catch endpoints that fail all retries.",
            "Filter by event type using topic subscriptions in the developer console.",
        ],
    },
    {
        "family": "team-role-permissions", "concepts": ["team"], "source": "kb",
        "title": "Roles and seats on {svc} {plan}",
        "title_alternates": ["Workspace permission levels on {svc}"],
        "body": ("{svc} workspaces define Owner, Admin, Editor, and Viewer roles. Seats on {plan} "
                 "are billed per unique member. Custom roles allow scoping editors to specific "
                 "projects."),
        "details": [
            "Role changes apply instantly and appear in the audit log.",
            "Owners can transfer ownership; only the last owner cannot be demoted.",
            "Viewers see read-only copies with no export permission.",
            "Guest seats are free but limited to a single workspace.",
            "Custom roles support per-project allowlists and deny rules.",
        ],
    },
    {
        "family": "gdpr-compliance", "concepts": ["compliance", "retention", "export"], "source": "docs",
        "title": "GDPR compliance package on {svc}",
        "title_alternates": ["Data protection commitments from {svc}"],
        "body": ("{svc} offers DPA, data processing records, and sub-processor lists on {plan}. "
                 "Data subject requests are answered within 30 days via the privacy portal."),
        "details": [
            "EU data residency is included on enterprise contracts.",
            "Right-to-erasure deletes hot data, backups, and analytics copies.",
            "DPAs are signed electronically and versioned for audit.",
            "Sub-processor changes are announced 30 days in advance.",
            "Data protection impact assessments are available on request.",
        ],
    },
    {
        "family": "data-retention", "concepts": ["retention", "compliance"], "source": "docs",
        "title": "Data retention policy on {svc}",
        "title_alternates": ["How long {svc} keeps your data"],
        "body": ("{svc} retains workspace content for the contract term plus 180 days. Logs are kept "
                 "30 days by default and purge via the retention schedule."),
        "details": [
            "Configure per-bucket retention windows in Admin > Data > Retention.",
            "Deletions are irreversible after the grace period ends.",
            "Legal hold freezes retention for workspaces under investigation.",
            "Audit logs follow a separate, stricter retention ladder.",
            "Backups inherit the most restrictive retention of any contained bucket.",
        ],
    },
    {
        "family": "kubernetes-deployment", "concepts": ["deployment", "api"], "source": "docs",
        "title": "Deploy {svc} on Kubernetes",
        "title_alternates": ["Installing {svc} into a cluster"],
        "body": ("The {svc} Helm chart deploys into an existing cluster with a single values.yaml. "
                 "Recommended resources are 4 vCPU / 8 GiB for the control plane."),
        "details": [
            "Horizontal pod autoscaling on CPU at 70% utilization is the default.",
            "Mounted secrets must not be committed to Git; use a SealedSecret controller.",
            "Rolling updates keep the API available during upgrades.",
            "Priority classes prevent the control plane from being evicted.",
            "Cluster-scoped RBAC is documented for multicluster installs.",
        ],
    },
    {
        "family": "mobile-app", "concepts": ["password", "api"], "source": "faq",
        "title": "Using the {svc} mobile app",
        "title_alternates": ["The {svc} app for phones and tablets"],
        "body": ("The {svc} mobile app mirrors web features: push notifications for events, offline "
                 "drafts, and biometric unlock. Install from the stores; version 4.0 requires "
                 "iOS 16+ or Android 13+."),
        "details": [
            "Biometric unlock uses device-level keys only — no password stored on the phone.",
            "Offline drafts sync when connectivity returns; conflicts keep both copies.",
            "Push notifications can be scoped per project from the notification settings.",
            "The app supports QR login by scanning the web session code.",
            "Corporate MDM policies can disable cloud backup of drafts.",
        ],
    },
    {
        "family": "cli-install", "concepts": ["api", "deployment"], "source": "docs",
        "title": "Install the {svc} CLI",
        "title_alternates": ["Getting started with the {svc} command line tool"],
        "body": ("Install the {svc} CLI with the one-line installer or standalone binaries. "
                 "Authenticate with an API token created in the dashboard."),
        "details": [
            "The CLI supports shell completion for bash, zsh, and fish.",
            "Configuration profiles keep separate contexts for dev and prod.",
            "Tokens are stored in the OS keychain, never in plaintext dotfiles.",
            "The CLI validates schema for plan files before push.",
            "Upgrade with the same installer; version checks happen on every run.",
        ],
    },
    {
        "family": "pricing-tiers", "concepts": ["plan", "billing"], "source": "faq",
        "title": "{svc} pricing tiers compared",
        "title_alternates": ["What each {svc} plan includes"],
        "body": ("{svc} pricing: Starter is free for individuals, Pro adds advanced analytics and "
                 "SSO, Enterprise adds audit logs and dedicated support. Usage above included "
                 "quotas bills per unit."),
        "details": [
            "See the pricing page for current per-unit rates and commit discounts.",
            "Annual billing saves 20% versus monthly on Pro and Enterprise.",
            "Education and non-profit discounts apply to all tiers.",
            "Usage overage caps prevent surprise overbilling.",
            "Migrating tiers preserves projects, integrations, and webhooks.",
        ],
    },
    {
        "family": "backup-restore", "concepts": ["export", "retention"], "source": "kb",
        "title": "Back up and restore a {svc} workspace",
        "title_alternates": ["Restoring a {svc} workspace from backup"],
        "body": ("Workspace admin can schedule daily backups in Admin > Data > Backup. Restores "
                 "target a new workspace and complete in under 4 hours."),
        "details": [
            "Point-in-time restore is available on {plan} for the last 14 days.",
            "Backups are encrypted at rest and replicated across two regions.",
            "Restoring merges into an empty workspace; existing data is untouched.",
            "Test restores are recommended quarterly to validate the pipeline.",
            "Backup retention follows the workspace retention policy.",
        ],
    },
    {
        "family": "usage-limits", "concepts": ["api", "plan"], "source": "faq",
        "title": "Usage limits on {svc} {plan}",
        "title_alternates": ["Included quotas on {svc}"],
        "body": ("{svc} {plan} includes 10,000 API calls and 50 GB storage per month. Shared "
                 "projects pool usage across members."),
        "details": [
            "Over-limit requests fail with 429 unless overage billing is enabled.",
            "Overage billing caps at 2× the included quota before automatic blocking.",
            "Usage meters update within 5 minutes on the dashboard.",
            "Unused quota does not roll over between months.",
            "Enterprise quotas are custom-committed in the contract.",
        ],
    },
    {
        "family": "audit-logs", "concepts": ["compliance", "team"], "source": "docs",
        "title": "Audit logs on {svc}",
        "title_alternates": ["Security event history in {svc}"],
        "body": ("{svc} records admin actions, authentication events, and data exports in the audit "
                 "log. Logs are immutable and exportable as JSON."),
        "details": [
            "Retention follows the workspace policy; enterprise keeps 365 days.",
            "Log entries include actor IP, timestamp, and a per-event digest.",
            "Hash chaining makes tampering detectable across the whole log.",
            "Streaming export pushes entries into your SIEM via webhook.",
            "Only owners can query the audit log via the API.",
        ],
    },
    {
        "family": "feature-flags", "concepts": ["api", "webhook"], "source": "docs",
        "title": "Feature flags via the {svc} API",
        "title_alternates": ["Controlled rollouts with {svc} flags"],
        "body": ("Manage feature flags through the {svc} API: create, evaluate, and update flags. "
                 "Flag evaluation is cached at the edge for 30 seconds."),
        "details": [
            "Rollout changes emit webhook events so clients react immediately.",
            "Targeting supports percentage rollouts, cohorts, and user lists.",
            "Flag archives keep history for rollback and analysis.",
            "SDKs cache evaluated flags locally and refresh in the background.",
            "Kill switches can be flipped from the dashboard in under a second.",
        ],
    },
    {
        "family": "error-codes", "concepts": ["api"], "source": "docs",
        "title": "{svc} common error codes",
        "title_alternates": ["Troubleshooting {svc} API errors"],
        "body": ("{svc} errors: 400 invalid payload, 401 missing or expired token, 403 permission "
                 "denied, 404 unknown resource, 409 version conflict, 429 rate limited, 5xx "
                 "internal."),
        "details": [
            "Each error body includes a request id for support tracing.",
            "4xx errors never consume rate-limit quota.",
            "409 conflicts should be handled with a read-modify-write retry.",
            "5xx responses may be retried with exponential backoff up to 5 attempts.",
            "Error catalog docs list remediation steps per code.",
        ],
    },
    {
        "family": "payment-methods", "concepts": ["billing"], "source": "faq",
        "title": "Manage payment methods on {svc}",
        "title_alternates": ["Updating your saved card on {svc}"],
        "body": ("Add, remove, or reorder payment methods in Billing > Payment methods. Cards are "
                 "tokenized by the processor; {svc} never stores raw card data."),
        "details": [
            "Failed renewals retry for 5 days before the plan is suspended.",
            "ACH and wire transfers are supported for enterprise invoicing.",
            "The default method can be set per workspace, not per user.",
            "Card updates notify all admins by email.",
            "Payment method history is kept for 24 months for chargeback defense.",
        ],
    },
    {
        "family": "seat-management", "concepts": ["team", "plan"], "source": "kb",
        "title": "Manage seats on {svc} {plan}",
        "title_alternates": ["Inviting and removing members on {svc}"],
        "body": ("Admins can invite, suspend, and remove members in Members. Suspended members keep "
                 "their data; removed members release a seat immediately."),
        "details": [
            "Seats are prorated on mid-cycle changes and reflected on the next {svc} invoice.",
            "Re-inviting a removed member creates a fresh seat at the current price.",
            "Bulk operations accept CSV uploads of up to 10,000 rows.",
            "Suspended members cannot sign in but their content stays searchable.",
            "Role-based seat caps can be enforced workspace-wide.",
        ],
    },
]


def _build_eval_queries() -> List[Dict[str, Any]]:
    """~40 test queries, rephrased relative to document titles. gold = family.

    Each query is *unambiguous*: the gold family is genuinely the best answer,
    but wording differs from the doc titles so retrieval must be semantic.
    """
    template = [
        ("password-reset", "I locked myself out of my account, how do I regain access?"),
        ("password-reset", "My credentials stopped working after a password change — what do I do?"),
        ("billing-invoices", "How do I download the invoice from the previous cycle?"),
        ("billing-invoices", "I never received the receipt for the last billing cycle."),
        ("refund-policy", "We bought a subscription we no longer need. Can we request a refund?"),
        ("refund-policy", "What are the terms for returning a subscription I just bought?"),
        ("api-rate-limits", "How many requests am I allowed per minute on the free API key?"),
        ("api-rate-limits", "My integration keeps hitting request caps. What are the quotas?"),
        ("plan-upgrades", "We want to move from Starter to Pro mid-cycle. Is that prorated?"),
        ("plan-upgrades", "We are changing tiers mid-subscription. When does the new price apply?"),
        ("shipping-tracking", "My order's tracking shows no updates in days. Who can help?"),
        ("shipping-tracking", "When will my order arrive and can I follow it live?"),
        ("sso-setup", "We want everyone to sign in with our corporate identity provider."),
        ("sso-setup", "How do I connect our SAML directory to the workspace?"),
        ("data-export", "I need a copy of everything my organization has stored here."),
        ("data-export", "Is it possible to download all our content including attachments?"),
        ("two-factor-auth", "I want a one-time code from an authenticator app when signing in."),
        ("two-factor-auth", "I heard I can use a hardware security key to sign in. Is that supported?"),
        ("sla-uptime", "What compensation do we get when the service is down for hours?"),
        ("sla-uptime", "Does the paid plan guarantee availability? What is the percentage?"),
        ("webhooks-integration", "Can our backend subscribe to event notifications when records change?"),
        ("webhooks-integration", "How do I make sure incoming event callbacks come from you?"),
        ("team-role-permissions", "Who can see what inside the workspace? Explain the roles."),
        ("team-role-permissions", "I want a colleague to only edit one project. Custom access possible?"),
        ("gdpr-compliance", "Where can I exercise my data rights as a European customer?"),
        ("gdpr-compliance", "Do you have a data processing agreement we can sign?"),
        ("data-retention", "How long do you keep our data after we cancel?"),
        ("data-retention", "Can I configure automatic deletion schedules for old records?"),
        ("kubernetes-deployment", "Can this service be deployed inside our Kubernetes cluster with a Helm chart?"),
        ("kubernetes-deployment", "What resources should we allocate for the control plane in a cluster?"),
        ("mobile-app", "Is there a phone app so I can check events on the go?"),
        ("mobile-app", "Can I unlock the mobile application with my fingerprint?"),
        ("cli-install", "I prefer the terminal. How do I get the command line tool up and running?"),
        ("cli-install", "What do I need to authenticate against the service from a terminal session?"),
        ("pricing-tiers", "What comes free and what costs money in your offering?"),
        ("pricing-tiers", "Compare the paid tiers and what each unlocks."),
        ("backup-restore", "If we delete something by accident, can we get it back?"),
        ("backup-restore", "How often are our workspaces snapshotted automatically?"),
        ("usage-limits", "Our workspace is close to the monthly included usage. What happens if we exceed it?"),
        ("usage-limits", "Are there ceilings on monthly API calls for small teams?"),
        ("audit-logs", "Can we see a chronological record of admin actions in the workspace?"),
        ("audit-logs", "For compliance we must show who performed administrative actions. Is that recorded?"),
        ("feature-flags", "How do we toggle features per customer segment via the API?"),
        ("error-codes", "What does the 429 response mean and how should our code handle it?"),
        ("payment-methods", "My team's billing card was reissued with new numbers. How do we update it?"),
        ("seat-management", "Someone left the team—how do we free up their license?"),
        ("seat-management", "Can guests be given limited workspace access?"),
    ]
    return [{"query": q, "family": f} for f, q in template]


if __name__ == "__main__":
    data = build_corpus()
    print(f"corpus: {len(data['documents'])} docs | {len(data['registry'])} families | "
          f"{len(data['queries'])} eval queries")