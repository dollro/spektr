DENSE_COLLECTION = "documents_dense"
MULTIVEC_COLLECTION = "documents_multivec"
DENSE_DIM = 512
MULTIVEC_DIM = 128

ENTITY_TYPES: dict[str, str] = {
    "person": "A named individual, author, speaker, or public figure",
    "organization": "A company, institution, government body, or team",
    "location": "A physical place, address, region, country, or facility",
    "date_time": "A specific date, time period, deadline, or schedule reference",
    "monetary_value": "An amount of money, price, fee, budget, or financial figure",
    "document": "A named contract, agreement, report, policy, regulation, or standard",
    "product": "A named product, service, platform, or deliverable",
    "technology": "A programming language, framework, tool, protocol, or system",
    "metric": "A quantitative measure, KPI, percentage, statistic, or benchmark",
    "event": "A named meeting, conference, milestone, incident, or release",
    "legal_term": "A clause, obligation, right, liability, warranty, or legal concept",
    "role": "A job title, department, committee, or functional responsibility",
    "concept": "An abstract idea, methodology, strategy, design pattern, or practice",
    "requirement": "A specification, condition, constraint, or criterion",
}

RELATIONSHIP_TYPES: dict[str, str] = {
    "created_by": "X was created, authored, or produced by Y",
    "owned_by": "X is owned, managed, or governed by Y",
    "uses": "X uses, depends on, or integrates Y",
    "part_of": "X is a component, section, or subset of Y",
    "related_to": "X is associated with or relevant to Y",
    "measured_by": "X is measured, evaluated, or quantified by Y",
    "requires": "X requires, mandates, or depends on Y",
    "applies_to": "X applies to, governs, or regulates Y",
    "succeeds": "X replaces, supersedes, or follows Y",
    "conflicts_with": "X contradicts, opposes, or is incompatible with Y",
    "valued_at": "X has a monetary value, cost, or price of Y",
    "scheduled_for": "X is planned, due, or scheduled for Y",
}
