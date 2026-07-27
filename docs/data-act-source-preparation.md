# EU Data Act Source Preparation

## 1. Purpose and boundary

This document defines the source acceptance contract for preparing Regulation (EU) 2023/2854 for a future metadata-v2 corpus build. It covers source format, catalog identity, preprocessing, expected structural output, and pre-build validation.

It does not authorize downloading a source into the repository, modifying the ingestion pipeline, generating embeddings, or updating `vector_store.json`.

The intended future flow is:

```text
verified official source
    -> normalized legal text
    -> structure-preserving chunks
    -> metadata-v2 enrichment
    -> candidate corpus artifact
```

## 2. Expected source format

### 2.1 Authoritative text

The source must be the English Official Journal text of **Regulation (EU) 2023/2854 on harmonised rules on fair access to and use of data and amending Regulation (EU) 2017/2394 and Directive (EU) 2020/1828 (Data Act)**.

Authoritative location:

- ELI: `http://data.europa.eu/eli/reg/2023/2854/oj`
- EUR-Lex: [Regulation (EU) 2023/2854](https://eur-lex.europa.eu/eli/reg/2023/2854/oj)

The initial corpus must use the Official Journal version, not a summary, commentary, guidance document, unofficial copy, or silently updated consolidated text. A later consolidated version requires a distinct, deliberate `document_version` decision.

### 2.2 Accepted working formats

The current ingestion layer accepts only:

- UTF-8 `.txt`; or
- text-based `.pdf` readable by `pypdf`.

For source preparation, UTF-8 plain text is preferred because it permits direct inspection of legal boundaries and avoids PDF column, table, and page-header extraction errors. A PDF may be retained as the provenance source, but its extracted text must be reviewed before it becomes the build input.

The prepared filename should be:

```text
EU Data Act Regulation 2023-2854.txt
```

This matches the current `EU_DATA_ACT` catalog alias. The filename is a lookup and provenance aid only; it must not become the stable legal identity or Evidence ID.

### 2.3 Source package

Before a corpus build, the source package should provide:

```text
official source file or retrieval reference
prepared UTF-8 text
source preparation manifest
manual validation record
```

The manifest should record:

- official URI;
- acquisition timestamp;
- language (`en`);
- source format and filename;
- source-file SHA-256 checksum;
- prepared-text SHA-256 checksum;
- catalog schema version;
- instrument ID;
- document version;
- preparation method; and
- reviewer/status fields.

Generated files must remain candidate artifacts until all validation checks pass.

## 3. Required legal source catalog mapping

The prepared source must resolve through `config/legal_sources.json` to exactly one legal source:

| Field | Required value | Purpose |
|---|---|---|
| `instrument_id` | `EU_DATA_ACT` | Stable, filename-independent instrument identity |
| `document_version` | `Regulation (EU) 2023/2854` | Exact catalog `version` copied into metadata v2 |
| `authority_level` | `binding_legislation` | Authority assigned to the Official Journal regulation |

Supporting catalog values should remain:

| Field | Expected value |
|---|---|
| title | `EU Data Act` |
| regulation number | `Regulation (EU) 2023/2854` |
| official URI | `http://data.europa.eu/eli/reg/2023/2854/oj` |
| jurisdiction | `EU` |
| language | `en` |
| source alias | `EU Data Act Regulation 2023-2854.txt` |

### 3.1 Mapping rules

- Resolve the prepared filename using `LegalSourceCatalog.resolve_alias()` before chunking.
- Fail preparation if the alias is missing, ambiguous, or resolves to another instrument.
- Obtain version and authority from the resolved `LegalSource`; do not duplicate handwritten values in a build script.
- Do not map Commission guidance, recitals extracted from another version, commentary, or national implementation material to the same source alias.
- Do not change `document_version` without recording a source/version migration because it participates in every stable Evidence ID.

## 4. Required preprocessing

### 4.1 Text normalization

Preprocessing may normalize presentation artifacts, but it must not alter legal meaning or legal hierarchy.

Permitted normalization includes:

- convert line endings to `\n`;
- normalize Unicode consistently;
- remove verified repeating Official Journal headers, footers, page numbers, and ELI lines;
- collapse repeated horizontal whitespace where it is not structural;
- repair verified word breaks caused by PDF extraction;
- remove empty-page artifacts; and
- reduce excessive blank lines while retaining legal-unit separation.

The preparation process must not:

- summarize, paraphrase, translate, or complete the legal text;
- rewrite defined terms;
- remove legally meaningful punctuation or quotation marks;
- merge separate paragraphs or points;
- silently repair uncertain OCR text;
- insert LLM-generated text; or
- mix the regulation with guidance or commentary.

Every non-mechanical correction should be traceable to the official source and recorded in the preparation notes.

### 4.2 Citation preservation

Preserve visible markers needed to generate atomic citations:

```text
Article 2
Article 2(5)
Article 4(1)(a)
Recital 12
Annex III
```

The prepared text itself should retain the source's hierarchy rather than embedding generated canonical citations into the legal prose. Canonical citations are derived during chunking/enrichment.

Citation preparation rules:

- keep every Article number attached to the correct Article heading;
- keep numbered paragraphs distinguishable from footnotes and cross-references;
- keep lettered points attached to their parent paragraph;
- keep numbered definitions as separate recognizable legal units;
- preserve Recital numbers;
- preserve Annex identifiers and Annex boundaries;
- do not turn requested ranges such as `Article 2(5)-(6)` into one chunk identity; and
- do not treat a cross-reference within body text as the current chunk's citation.

The existing structured citation parser accepts only complete, atomic references. Unsupported ranges or trailing prose must be resolved into separate canonical references before enrichment.

### 4.3 Structural markers

The prepared text should preserve one structural marker per line where possible:

```text
CHAPTER II

Article 3
Obligation to make data accessible to the user

1. [paragraph text]

(a) [point text]
```

Required markers are:

- `CHAPTER` plus Roman numeral;
- `SECTION` plus identifier, where present;
- `Article` plus Article number on its own line;
- Article title on the following line;
- numbered paragraph markers;
- lettered point markers;
- numbered definitions within the definitions Article;
- Recital numbers before the operative provisions; and
- `ANNEX` plus identifier on its own line, where present.

The current chunker recognizes Article and Annex headings most reliably when they appear on their own lines. It recognizes operative paragraph lines in `1. ` form and lettered points in `(a) ` form. If the official extraction represents these differently, any normalization to these forms must be mechanical, consistent, and verified against the source.

### 4.4 Definition preparation

Article 2 definitions need special preparation because each definition must be individually retrievable.

For each definition:

- preserve its number;
- preserve the defined term and quotation marks;
- keep the full definition together;
- separate it from adjacent definitions;
- retain `Article 2` as the parent Article; and
- do not use the term itself as the canonical citation.

The expected canonical identity is the legal location, for example:

```text
Article 2(5) -> connected product
Article 2(6) -> related service
```

The defined term should be retained as supplemental `definition_term` metadata once definition-aware chunking is enabled.

## 5. Expected structural output metadata

Source preparation is acceptable only if the later chunking step can produce the following structure without guessing.

### 5.1 Article records

Example:

```json
{
  "canonical_citation": "Article 2",
  "article_number": "2",
  "paragraph_number": null,
  "point_label": null
}
```

An Article-level record is appropriate for an Article heading, title, or provision that cannot be assigned more precisely. It must not combine text from another Article.

### 5.2 Paragraph records

Example:

```json
{
  "canonical_citation": "Article 2(5)",
  "article_number": "2",
  "paragraph_number": "5",
  "point_label": null,
  "parent_citation": "Article 2"
}
```

Every paragraph fragment must retain the paragraph citation even when its text is split for size.

### 5.3 Point records

Example:

```json
{
  "canonical_citation": "Article 4(1)(a)",
  "article_number": "4",
  "paragraph_number": "1",
  "point_label": "a",
  "parent_citation": "Article 4(1)"
}
```

A point record must not include sibling points under the identity of the first point.

### 5.4 Definition records

Example:

```json
{
  "canonical_citation": "Article 2(5)",
  "article_number": "2",
  "paragraph_number": "5",
  "point_label": null,
  "parent_citation": "Article 2",
  "definition_term": "connected product"
}
```

`definition_term` is supplemental retrieval metadata. The stable identity continues to use `Article 2(5)` and the normalized authoritative excerpt.

### 5.5 Recital and Annex records

Expected forms are:

```json
{
  "canonical_citation": "Recital 12",
  "recital_ref": "12"
}
```

```json
{
  "canonical_citation": "Annex III",
  "annex_ref": "III"
}
```

### 5.6 Metadata-v2 enrichment output

After structure is final, every Data Act chunk must be enriched with:

```text
metadata_schema_version = 2.0.0
instrument_id           = EU_DATA_ACT
document_version        = Regulation (EU) 2023/2854
canonical_citation      = normalized atomic citation
authority_level         = binding_legislation
source_record_id        = content-derived v2 source record ID
stable_evidence_id      = content-derived v2 Evidence ID
```

IDs must be generated only after excerpt text and citation are final. Any later text, version, or citation change must trigger ID regeneration.

## 6. Validation checklist before corpus build

### 6.1 Source authority and provenance

- [ ] Source is the English Official Journal text from the official ELI/EUR-Lex record.
- [ ] Full regulation title and number match Regulation (EU) 2023/2854.
- [ ] Source is not a summary, guidance document, unofficial copy, or unrecorded consolidated version.
- [ ] Official URI, acquisition time, language, and source format are recorded.
- [ ] Source-file and prepared-text SHA-256 checksums are recorded.
- [ ] Preparation steps and any manual corrections are documented.

### 6.2 Catalog mapping

- [ ] Prepared filename is `EU Data Act Regulation 2023-2854.txt` or another explicitly catalogued alias.
- [ ] Alias resolves uniquely to `EU_DATA_ACT`.
- [ ] Catalog version resolves to `Regulation (EU) 2023/2854`.
- [ ] Catalog authority resolves to `binding_legislation`.
- [ ] Catalog language is `en` and jurisdiction is `EU`.
- [ ] No secondary material shares the primary-legislation alias.

### 6.3 Text integrity

- [ ] Text is valid UTF-8 and uses normalized line endings.
- [ ] Title, preamble, Recitals, Chapters, Articles, and any Annexes remain in source order.
- [ ] Repeating headers, footers, page numbers, and ELI lines are removed without deleting legal text.
- [ ] Article numbers and Article titles are complete.
- [ ] Paragraph and point markers remain attached to their text.
- [ ] Defined terms and quotation marks are preserved.
- [ ] Footnotes are not misidentified as paragraphs or Recitals.
- [ ] No content was summarized, translated, or generated.

### 6.4 Structural sampling

- [ ] At least one Recital parses to `Recital N`.
- [ ] At least one Article parses to `Article N`.
- [ ] At least one paragraph parses to `Article N(P)`.
- [ ] At least one point parses to `Article N(P)(a)`.
- [ ] Each selected Article 2 definition is isolated and retains its defined term.
- [ ] Any Annex heading parses to `Annex N`.
- [ ] No sample chunk crosses an Article, Recital, or Annex boundary.
- [ ] Citation parser rejects ranges and trailing prose rather than silently normalizing them.

### 6.5 Metadata-v2 dry run

- [ ] A representative chunk enriches successfully through `enrich_chunk_metadata_v2()`.
- [ ] Enriched `instrument_id`, version, citation, and authority match the catalog.
- [ ] Both `source_record_id` and `stable_evidence_id` are present.
- [ ] Repeating enrichment with identical input produces identical IDs.
- [ ] Changing the citation or authoritative excerpt produces a different stable ID.
- [ ] Enrichment does not modify the original chunk object.
- [ ] No candidate record declares schema version `2.0.0` with incomplete v2 metadata.

### 6.6 Build authorization gate

Do not begin the final corpus build until:

1. source provenance is complete;
2. catalog resolution succeeds;
3. structural samples pass;
4. Article 2 definitions can be represented atomically;
5. metadata-v2 dry-run records validate; and
6. a reviewer confirms that prepared text remains faithful to the official source.

Passing this checklist prepares the Data Act source for a candidate build. It does not by itself authorize modification of the active `vector_store.json` or other existing corpus artifacts.
