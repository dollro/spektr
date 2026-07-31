DENSE_COLLECTION = "documents_dense"
MULTIVEC_COLLECTION = "documents_multivec"
DENSE_DIM = 512
MULTIVEC_DIM = 128

ENTITY_TYPES: dict[str, str] = {
    "person": "A named individual, author, executive, or public figure",
    "organization": "A company, institution, government body, or team",
    "location": "A physical place, address, region, country, or facility",
    "date_time": "A specific date, time period, deadline, or schedule reference",
    "monetary_value": "An amount of money, price, fee, budget, or financial figure",
    "document": "A named contract, agreement, report, policy, regulation, or standard",
    "product": "A named product, service, platform, or deliverable",
    "technology": "A programming language, framework, tool, protocol, or system",
    "metric": "A quantitative measure, KPI, percentage, statistic, or benchmark",
    "event": "A named conference, milestone, incident, or release",
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
    "measured_by": "X is measured, evaluated, or quantified by Y",
    "requires": "X requires, mandates, or depends on Y",
    "applies_to": "X applies to, governs, or regulates Y",
    "succeeds": "X replaces, supersedes, or follows Y",
    "conflicts_with": "X contradicts, opposes, or is incompatible with Y",
    "valued_at": "X has a monetary value, cost, or price of Y",
    "scheduled_for": "X is planned, due, or scheduled for Y",
    "mentions": "X references, cites, or names Y in text",
}

_ALL_TYPES = frozenset(ENTITY_TYPES)

# Domain/range constraints: (allowed_source_types, allowed_target_types).
# Triples violating these are dropped during GLiNER ingestion.
RELATION_CONSTRAINTS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "created_by": (
        frozenset({"person", "organization", "product", "technology", "document", "event"}),
        frozenset({"person", "organization"}),
    ),
    "owned_by": (
        frozenset({"product", "organization", "document", "technology", "location"}),
        frozenset({"person", "organization"}),
    ),
    "uses": (
        frozenset({"person", "organization", "product", "technology", "role"}),
        frozenset({"technology", "product", "concept", "document"}),
    ),
    "part_of": (
        frozenset(
            {
                "person",
                "role",
                "technology",
                "product",
                "location",
                "requirement",
                "legal_term",
            }
        ),
        frozenset({"organization", "product", "technology", "document", "location"}),
    ),
    "measured_by": (
        frozenset({"product", "organization", "technology", "event", "concept"}),
        frozenset({"metric"}),
    ),
    "requires": (
        frozenset({"product", "technology", "document", "requirement", "role", "event"}),
        frozenset({"technology", "product", "requirement", "role", "concept"}),
    ),
    "applies_to": (
        frozenset({"document", "legal_term", "requirement", "concept"}),
        frozenset({"person", "organization", "product", "technology", "event"}),
    ),
    "succeeds": (
        frozenset({"product", "technology", "document", "event"}),
        frozenset({"product", "technology", "document", "event"}),
    ),
    "conflicts_with": (
        frozenset({"document", "requirement", "legal_term", "technology"}),
        frozenset({"document", "requirement", "legal_term", "technology"}),
    ),
    "valued_at": (
        frozenset({"product", "organization", "document", "event"}),
        frozenset({"monetary_value"}),
    ),
    "scheduled_for": (
        frozenset({"event", "product", "document", "requirement"}),
        frozenset({"date_time"}),
    ),
    "mentions": (
        frozenset({"person", "organization", "document", "event", "product", "technology"}),
        _ALL_TYPES,
    ),
}

# Named vectors on DENSE_COLLECTION. Sparse vectors must be named in Qdrant,
# and named/unnamed vectors cannot coexist — this is why the collection is
# recreated rather than updated in place.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# miniCOIL length normalisation. Average chunk length in tokens, derived from
# the 512-character chunk target (~80 tokens). Index-time only; not used when
# encoding queries. Revisit if chunk sizing changes.
MINICOIL_AVG_LEN = 80
