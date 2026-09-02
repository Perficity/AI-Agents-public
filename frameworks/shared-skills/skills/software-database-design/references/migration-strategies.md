# Migration Strategies

Use this file when the request is about changing schema in production.

## Table of Contents

- [Default Safe Pattern](#default-safe-pattern)
- [Rules](#rules)
- [PostgreSQL Lock Levels](#postgresql-lock-levels-verify-against-the-target-version-before-relying-on-this)
- [High-Risk Changes](#high-risk-changes)
- [Hyrum's Law and the Adds-vs-Drops Rule](#hyrums-law-and-the-adds-vs-drops-rule)
- [Major-Version Upgrade Regressions](#major-version-upgrade-regressions)

## Default Safe Pattern

1. Expand: add nullable column, new table, or compatible index.
2. Backfill: migrate data in batches.
3. Dual-write or dual-read if needed.
4. Contract: remove old path only after cutover is verified.

## Rules

- Separate schema migration from long-running backfill.
- Make migrations idempotent where possible.
- Avoid destructive changes in the same deploy as application cutover.
- For large tables, prefer online/index-concurrent strategies the engine supports.
- Run migration DDL through a direct (non-pooled) connection where the tool supports it — session-level settings like `statement_timeout` and `lock_timeout` are not guaranteed to survive a transaction-mode pooler reassigning the backend connection mid-migration.

## PostgreSQL Lock Levels (verify against the target version before relying on this)

Every plain `ALTER TABLE` sub-command takes at least some table-level lock; the risk is duration, not the lock's existence:

| Operation | Lock | Duration |
|---|---|---|
| `ADD COLUMN` (nullable, no default, or constant default) | `ACCESS EXCLUSIVE` | Milliseconds — metadata-only since Postgres 11, no table rewrite |
| `ADD COLUMN ... DEFAULT <volatile/expression>` | `ACCESS EXCLUSIVE` | Full table rewrite — scales with table size, treat as high-risk |
| `ALTER COLUMN TYPE` (incompatible type) | `ACCESS EXCLUSIVE` | Full table rewrite |
| `SET NOT NULL` (no pre-existing validated constraint) | `ACCESS EXCLUSIVE` | Full table scan to verify no NULLs — scales with table size |
| `ADD CONSTRAINT ... CHECK (...) NOT VALID` | `ACCESS EXCLUSIVE` | Milliseconds — skips the verification scan |
| `VALIDATE CONSTRAINT` (the deferred scan for the above) | `SHARE UPDATE EXCLUSIVE` | Scan runs, but concurrent reads/writes proceed |
| `CREATE INDEX` | `SHARE` | Blocks writes for the build duration |
| `CREATE INDEX CONCURRENTLY` | `SHARE UPDATE EXCLUSIVE` | Slower build, but reads/writes proceed |
| `DROP TABLE`, `TRUNCATE`, `VACUUM FULL`, `CLUSTER` | `ACCESS EXCLUSIVE` | Full duration of the operation |

**The `SET NOT NULL` skip-scan pattern (Postgres 12+):** add the constraint as `NOT VALID` first (instant), `VALIDATE CONSTRAINT` separately (lighter lock, concurrent-safe), then `SET NOT NULL` — Postgres reuses the validated constraint and skips its own redundant scan. This is the standard way to add `NOT NULL` to a populated column on a large table without a blocking full-table-scan window.

## High-Risk Changes

- Column type changes on hot tables (triggers a full rewrite and `ACCESS EXCLUSIVE` for the duration)
- `ADD COLUMN` with a volatile or expression default (also a full rewrite — a constant default is metadata-only and safe)
- Renames without compatibility layer
- Dropping columns still referenced by old code paths
- Backfills that lock large tables or saturate replicas
- `SET NOT NULL` without the `NOT VALID` → `VALIDATE CONSTRAINT` → `SET NOT NULL` sequence on a large, populated table

## Hyrum's Law and the Adds-vs-Drops Rule

Hyrum's Law: with enough consumers of a schema, every observable property — a column's existence, its name, its nullability, even its NULL-vs-empty-string convention — becomes something someone depends on, whether or not it was ever a documented contract. This is why expand/contract is a *deploy-sequencing* discipline, not just a naming convention for the three phases:

- **Adds are safe in any deploy.** A new nullable column or new table changes nothing an existing consumer already observes — nothing to violate.
- **Drops and renames are never safe in the same deploy as the add they pair with.** A rename is a drop wearing an add's clothes: `name` → `full_name` is not one step, it's expand (add `full_name`, dual-write both), migrate (backfill, cut reads over, verify), then contract (stop writing `name`, drop `name` in a *separate, later* deploy) — five numbered steps across at least two deploys, never collapsed into one.
- **The gate for contract is "no code references the old shape," not "the new shape looks done."** Verify via a grep/query audit of read paths (application code, reporting queries, downstream consumers) before dropping — the old column looking unused in your own codebase doesn't mean an out-of-band consumer isn't depending on it.
- Treat a migration with no tested `down` path the same as a deploy that can't be rolled back — don't ship expand or migrate steps you haven't verified you can reverse.

Source: adapted from addyosmani/agent-skills, `skills/deprecation-and-migration.md` (MIT), commit `7676817`, 2026-08-09.

## Major-Version Upgrade Regressions

Source: Jimmy Angelakos, *PostgreSQL Mistakes and How to Avoid Them* (Manning, 2025), §10.1–10.2.

Schema migration is not the only kind of migration that breaks production. A major-version engine upgrade can change the behavior of code you did not touch. Both worked examples below are PostgreSQL, but the failure shape generalizes: the upgrade "succeeds," nothing errors, and a behavior change surfaces days later as a business-logic bug or a latency regression.

### The meta-rule: read *all* intervening release notes, not just the target's

PostgreSQL's backward compatibility is good enough that skipping versions is normal and supported — "you can usually skip versions provided there is no chasm between them," e.g. 13 → 16 directly. The mistake is reading only the release notes for the version you are landing on.

Angelakos' case: a subscription expiry computed as `expiry + '0.333 years'` returned 2025-04-01 on PostgreSQL 13 and 2025-05-01 on PostgreSQL 16, silently granting customers an extra free month. The application code had not changed. The cause was a one-word documentation change — PG 13 said fractional parts of units greater than months "are truncated to be an integer number of months," PG 16 said they "are rounded." `'1.333 years'::interval` is `1 year 3 mons` on 13 and `1 year 4 mons` on 16.

The DBA *had* read the PostgreSQL 16 release notes. The change was not in them, because it shipped in **15**: "When interval input provides a fractional value for a unit greater than months, round to the nearest month (Bruce Momjian)" — PostgreSQL 15.0 release notes, 2022-10-13.

Angelakos' rule, quoted:

> "What they should have done is read all the intervening major release notes to identify the changes made between Postgres 13 and 14, 14 and 15, and finally 15 and 16. Even though skip-upgrading is possible and even desirable, due diligence dictates that you should go through the entire set of release notes."

And the bottom line: "if you have not read all the release notes between the original and target versions, you have committed a serious upgrading mistake. Make sure to pay attention to function deprecations or changes in name, changes to configuration and default settings, and SQL syntax updates."

Practically: the release-notes read is a checklist item with N entries, where N is the number of major versions you are crossing — not one entry. Skipping intermediate notes "is not harmless because one of the releases in between may have introduced a breaking change that you will be unaware of."

### Worked example: CTE inlining changed in PostgreSQL 12

Before PG 12, a `WITH` clause was an *optimization fence* — the CTE was materialized, and its result fed the rest of the query. Starting with PG 12 that stopped being guaranteed: "Starting with Postgres version 12, CTEs began to be automatically inlined when they are referenced just once in the query, as long as they are not recursive and have no side effects."

Angelakos' example query went from 50 ms on PG 11 to 85 ms on PG 12 with no code change. The PG 11 plan showed both CTEs producing parallel sequential scans feeding hash joins and a final `HashAggregate`. The PG 12 plan had "no mention of CTEs, and there are two levels of nested loops. The inner nested loop performs an index scan, which means that the index is scanned multiple times because of the outer loop. The filtering also happens after the joins, which means that what we don't need is eliminated later, after processing more data."

Note the honest framing — this is a regression *for this query*, not a bad change: "This doesn't necessarily mean that this query plan is worse in all cases, but it so happens that, in this case, it's slower."

**The escape hatch is the `MATERIALIZED` keyword.** Writing `WITH unp AS MATERIALIZED (...)` restores the pre-12 fencing behavior for that CTE explicitly. `NOT MATERIALIZED` forces inlining. Treat these as the deliberate way to express intent when the planner's default choice is wrong for a specific query — not as something to sprinkle across every CTE in the codebase, which just relitigates the pre-12 fence and forfeits the cases where inlining wins.

### Worked example: JIT compilation defaulted on in PostgreSQL 12

The second regression in the same 11 → 12 jump: "just-in-time (JIT) compilation was enabled by default. This caused some already fast and optimal queries to become slower by default, as Postgres needlessly spent CPU time trying to optimize them further with JIT."

The trap is that this is invisible to synthetic testing. It fires per-query based on estimated cost — "they may happen to cross the `jit_above_cost` threshold that triggers the automatic JIT optimizer." A query whose plan cost sits just over the threshold pays JIT compilation time it never recoups; one just under does not. Angelakos: "If regression tests are not performed with the actual queries, and using version 11 as the baseline, this sort of slowdown is hard to catch."

### Upgrade testing checklist

Following from both examples — a clean `pg_upgrade` run is not evidence the upgrade is safe:

- Test with **your actual queries**, not representative ones. Both regressions above are query-specific: plan shape and JIT cost thresholds do not generalize from a benchmark.
- **Baseline against the old version.** A latency number from the new version alone tells you nothing; you need the before/after pair on the same data.
- Use **real-world data, or as realistic as is feasible** — plan choice depends on statistics, so a small test dataset can pick a different plan than production will.
- Run it in a **staging environment before production**, covering "all your critical queries and known edge cases." `pg_upgrade` is useful for these dry runs.
- **Test under load.** "a stress test can reveal how overall performance can be affected by the upgrade."
- Cover **business-logic outputs, not just latency and errors**. The interval-rounding bug produced correct-looking data that was wrong by a month; no performance test or error log would have caught it. Assertions on computed business values are what catch this class.
