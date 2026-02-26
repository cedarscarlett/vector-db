-- codesem/storage/schema.sql
-- CodeSem pgvector schema (PostgreSQL 14+)

-- Enable pgvector extension
create extension if not exists vector;

-- Main table: embedded code chunks
create table if not exists code_chunks (
    id uuid primary key default gen_random_uuid(),

    -- Repo-relative (or absolute) file path as provided by indexer.
    -- For single-repo usage, this is sufficient. If multi-repo is later
    -- added, introduce repo_id/repo_root and index/filter by it.
    file_path text not null,

    start_line int not null,
    end_line int not null,

    content text not null,
    content_hash text not null,

    -- Embedding and metadata
    -- NOTE:
    -- Do NOT hardcode vector(1536).
    -- The embedding dimension may be overridden via --dimensions or
    -- EMBEDDING_DIMENSIONS. Using unbounded `vector` allows variable
    -- dimensions at the DB level. The application layer is responsible
    -- for keeping embedding dimensions consistent within a deployment.
    embedding vector not null,
    embedding_model text not null,
    embedding_dimensions int null,

    created_at timestamptz not null default now()
);

-- Helpful btree indexes
create index if not exists idx_code_chunks_file_path on code_chunks (file_path);
create index if not exists idx_code_chunks_content_hash on code_chunks (content_hash);

-- Ensure we don't store duplicate chunks for the same file/line-range with identical content.
-- This is optional but useful when reindexing.
--
-- IMPORTANT LIMITATION:
-- This constraint only prevents exact duplicates of the same
-- (file_path, start_line, end_line, content_hash).
--
-- If chunking parameters change (e.g., --chunk-tokens),
-- start_line/end_line ranges will differ and old chunks will
-- not conflict with new ones.
--
-- Stale data cleanup is handled by the application layer
-- (--delete-stale flag). Do not rely on this unique index
-- as full protection against chunk parameter changes.
create unique index if not exists uq_code_chunks_file_range_hash
on code_chunks (file_path, start_line, end_line, content_hash);

-- Vector index (HNSW)
-- Works well across small and large datasets without training.
create index if not exists idx_code_chunks_embedding_hnsw
on code_chunks using hnsw (embedding vector_cosine_ops);
