CREATE TABLE long_context_repositories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_id UUID NOT NULL REFERENCES github_repositories(id),
    full_name TEXT NOT NULL,
    url TEXT NOT NULL,
    github_size_kb INTEGER NOT NULL,
    primary_language TEXT,
    default_branch TEXT,
    total_code_tokens INTEGER NOT NULL,
    total_code_files INTEGER NOT NULL,
    total_code_bytes BIGINT NOT NULL,
    tokens_by_extension JSONB,
    dependency_score FLOAT NOT NULL,
    internal_import_count INTEGER NOT NULL,
    unique_internal_imports INTEGER NOT NULL,
    graph_density FLOAT,
    avg_in_degree FLOAT,
    max_in_degree INTEGER,
    num_connected_components INTEGER,
    largest_component_fraction FLOAT,
    conference TEXT,
    year INTEGER,
    paper_title TEXT,
    processing_status TEXT NOT NULL DEFAULT 'completed',
    error_message TEXT,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(repo_id)
);

CREATE INDEX idx_lcr_dependency_score ON long_context_repositories(dependency_score DESC);
CREATE INDEX idx_lcr_total_code_tokens ON long_context_repositories(total_code_tokens DESC);
CREATE INDEX idx_lcr_processing_status ON long_context_repositories(processing_status);
