DENSE_COLLECTION = "documents_dense"
MULTIVEC_COLLECTION = "documents_multivec"
DENSE_DIM = 512
MULTIVEC_DIM = 128

ENTITY_TYPES: dict[str, str] = {
    "person": "A named individual, author, developer, researcher, or public figure",
    "organization": "A company, institution, open-source project, or team",
    "technology": "A programming language, framework, library, tool, CLI command, or protocol",
    "concept": "An idea, methodology, design pattern, workflow, or practice",
    "metric": "A quantitative measure, statistic, benchmark, or KPI",
    "location": "A physical place, region, or URL/domain",
    "event": "A named event, release, conference, or incident",
}

RELATIONSHIP_TYPES: dict[str, str] = {
    "created_by": "X was created, authored, or developed by Y",
    "uses": "X uses, depends on, or integrates Y",
    "part_of": "X is a component, feature, or subset of Y",
    "related_to": "X is conceptually related to or associated with Y",
    "improves": "X improves, enhances, or optimizes Y",
    "measured_by": "X is measured or quantified by Y",
    "located_in": "X is geographically or organizationally located in Y",
    "describes": "X describes, documents, or explains Y",
}
