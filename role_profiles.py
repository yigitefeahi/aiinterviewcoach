from __future__ import annotations

from typing import Any


ROLE_INTERVIEW_PROFILES: dict[str, dict[str, Any]] = {
    "Software Engineer": {
        "themes": ["debugging", "architecture tradeoffs", "testing", "maintainability"],
        "evaluation_focus": ["problem solving", "code quality", "validation", "impact"],
    },
    "Full Stack Developer": {
        "themes": ["end-to-end feature ownership", "API contracts", "frontend/backend tradeoffs", "deployment"],
        "evaluation_focus": ["full-stack reasoning", "integration quality", "product impact", "testing"],
    },
    "Backend Developer": {
        "themes": ["API design", "data modeling", "reliability", "performance", "distributed systems"],
        "evaluation_focus": ["system boundaries", "database reasoning", "observability", "failure handling"],
    },
    "Frontend Developer": {
        "themes": ["UI architecture", "state management", "performance", "accessibility", "design collaboration"],
        "evaluation_focus": ["user experience", "component design", "browser performance", "testing"],
    },
    "Mobile Developer": {
        "themes": ["offline behavior", "app performance", "platform constraints", "release quality"],
        "evaluation_focus": ["mobile UX", "device constraints", "crash handling", "store release readiness"],
    },
    "Data Scientist": {
        "themes": ["experimentation", "model evaluation", "statistical reasoning", "business impact"],
        "evaluation_focus": ["metric choice", "data quality", "model validity", "communication"],
    },
    "Machine Learning Engineer": {
        "themes": ["feature engineering", "model serving", "model monitoring", "ML pipeline reliability"],
        "evaluation_focus": ["production ML", "offline/online metrics", "deployment tradeoffs", "drift handling"],
    },
    "AI Engineer": {
        "themes": ["RAG quality", "LLM evaluation", "prompting", "guardrails", "latency and cost"],
        "evaluation_focus": ["grounding", "hallucination mitigation", "AI product integration", "measurement"],
    },
    "Data Engineer": {
        "themes": ["ETL pipelines", "data quality", "orchestration", "warehousing", "streaming"],
        "evaluation_focus": ["pipeline reliability", "schema evolution", "data validation", "scalability"],
    },
    "Data Analyst": {
        "themes": ["SQL analysis", "dashboards", "KPI definition", "business insight", "data cleaning"],
        "evaluation_focus": ["metric clarity", "stakeholder communication", "insight quality", "recommendations"],
    },
    "Business Analyst": {
        "themes": ["requirements gathering", "process analysis", "stakeholder alignment", "UAT", "acceptance criteria"],
        "evaluation_focus": ["business clarity", "documentation", "prioritization", "validation"],
    },
    "Cloud Engineer": {
        "themes": ["cloud architecture", "IAM", "networking", "infrastructure as code", "cost control"],
        "evaluation_focus": ["security", "reliability", "cost awareness", "operational readiness"],
    },
    "Site Reliability Engineer": {
        "themes": ["SLOs", "incident response", "observability", "toil reduction", "capacity planning"],
        "evaluation_focus": ["reliability thinking", "postmortem quality", "automation", "risk management"],
    },
    "MLOps Engineer": {
        "themes": ["model registry", "CI/CD for ML", "feature stores", "model drift", "reproducibility"],
        "evaluation_focus": ["ML operations", "monitoring", "governance", "deployment safety"],
    },
    "Database Administrator": {
        "themes": ["backup and recovery", "query tuning", "replication", "indexing", "access control"],
        "evaluation_focus": ["data integrity", "performance", "availability", "restore confidence"],
    },
    "System Administrator": {
        "themes": ["server operations", "identity management", "patching", "monitoring", "automation"],
        "evaluation_focus": ["operational discipline", "security hygiene", "incident handling", "documentation"],
    },
    "Solutions Architect": {
        "themes": ["customer requirements", "solution design", "integration", "cost/risk tradeoffs"],
        "evaluation_focus": ["architecture fit", "stakeholder communication", "constraints", "roadmapping"],
    },
    "Software Architect": {
        "themes": ["domain boundaries", "design patterns", "technical direction", "migration strategy"],
        "evaluation_focus": ["architectural judgment", "maintainability", "leadership", "tradeoffs"],
    },
    "Game Developer": {
        "themes": ["gameplay systems", "engine performance", "multiplayer", "debugging", "player experience"],
        "evaluation_focus": ["performance", "iteration", "technical creativity", "user experience"],
    },
    "Embedded Software Engineer": {
        "themes": ["firmware", "RTOS", "hardware interfaces", "timing constraints", "debugging"],
        "evaluation_focus": ["resource constraints", "reliability", "hardware validation", "C/C++ reasoning"],
    },
    "Blockchain Developer": {
        "themes": ["smart contracts", "gas optimization", "security audits", "Web3 integration"],
        "evaluation_focus": ["security", "correctness", "transaction cost", "testing"],
    },
    "Product Manager": {
        "themes": ["roadmap tradeoffs", "metrics", "user research", "prioritization", "go-to-market"],
        "evaluation_focus": ["product judgment", "impact", "stakeholder alignment", "measurement"],
    },
    "UX/UI Designer": {
        "themes": ["user research", "prototyping", "interaction design", "accessibility", "design systems"],
        "evaluation_focus": ["user empathy", "design rationale", "iteration", "communication"],
    },
    "DevOps Engineer": {
        "themes": ["CI/CD", "containers", "infrastructure as code", "release reliability", "monitoring"],
        "evaluation_focus": ["automation", "deployment safety", "observability", "collaboration"],
    },
    "Cybersecurity Analyst": {
        "themes": ["threat analysis", "SIEM", "incident triage", "vulnerability risk", "security monitoring"],
        "evaluation_focus": ["risk reasoning", "investigation quality", "prioritization", "communication"],
    },
    "Security Engineer": {
        "themes": ["threat modeling", "application security", "cloud security", "secure coding"],
        "evaluation_focus": ["risk reduction", "preventive controls", "architecture", "remediation quality"],
    },
    "Penetration Tester": {
        "themes": ["reconnaissance", "exploitation", "OWASP", "reporting", "remediation advice"],
        "evaluation_focus": ["methodology", "evidence", "impact explanation", "ethics"],
    },
    "SOC Analyst": {
        "themes": ["alert triage", "log analysis", "threat hunting", "incident escalation"],
        "evaluation_focus": ["investigation discipline", "false positive reduction", "response speed", "documentation"],
    },
    "QA Engineer": {
        "themes": ["test strategy", "regression risk", "bug reporting", "quality gates", "manual and automated coverage"],
        "evaluation_focus": ["risk-based testing", "clarity", "coverage", "collaboration"],
    },
    "Automation Engineer": {
        "themes": ["workflow automation", "scripting", "RPA", "integration", "time savings"],
        "evaluation_focus": ["process understanding", "reliability", "maintainability", "impact"],
    },
    "Test Automation Engineer": {
        "themes": ["test frameworks", "flaky test reduction", "CI integration", "test data", "coverage"],
        "evaluation_focus": ["automation strategy", "maintainability", "quality signal", "debugging"],
    },
    "Scrum Master": {
        "themes": ["facilitation", "team impediments", "retrospectives", "sprint planning", "continuous improvement"],
        "evaluation_focus": ["servant leadership", "team health", "process improvement", "conflict handling"],
    },
    "Technical Project Manager": {
        "themes": ["delivery planning", "dependency tracking", "risk management", "stakeholder communication"],
        "evaluation_focus": ["execution", "technical coordination", "tradeoffs", "visibility"],
    },
    "Technical Support Engineer": {
        "themes": ["troubleshooting", "logs", "root cause analysis", "customer communication", "escalation"],
        "evaluation_focus": ["diagnosis", "communication", "resolution quality", "prevention"],
    },
    "IT Support Specialist": {
        "themes": ["helpdesk", "device setup", "Active Directory", "ticket handling", "basic networking"],
        "evaluation_focus": ["user empathy", "troubleshooting", "documentation", "reliability"],
    },
    "Network Engineer": {
        "themes": ["routing", "switching", "firewalls", "VPNs", "packet flow", "network troubleshooting"],
        "evaluation_focus": ["network fundamentals", "isolation strategy", "security", "uptime"],
    },
    "ERP/CRM Consultant": {
        "themes": ["requirements", "configuration", "business process", "data migration", "user training"],
        "evaluation_focus": ["business fit", "stakeholder management", "implementation quality", "adoption"],
    },
    "SAP Consultant": {
        "themes": ["SAP modules", "configuration", "implementation", "data migration", "process fit"],
        "evaluation_focus": ["module knowledge", "business process alignment", "testing", "change management"],
    },
}


def get_role_profile(profession: str) -> dict[str, Any]:
    return ROLE_INTERVIEW_PROFILES.get(profession, ROLE_INTERVIEW_PROFILES["Software Engineer"])
