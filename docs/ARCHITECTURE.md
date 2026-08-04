# Architecture and research rules

## Components

1. GitHub Actions runs short, idempotent jobs.
2. Official exchange archives remain the raw source of truth.
3. Normalized derived events and reports may be stored in a private Hugging Face dataset.
4. Statistical search is deterministic. Optional LLM review may suggest bounded configurations, but cannot promote a model.

## Coverage tiers

- Long history: trades, aggressor side, price, volume, CVD.
- Derivatives enriched: funding and open interest where available.
- Full microstructure: exact opposite-side depth within 10 bp only when a synchronized L2 source is present.

No report may describe a depth result as covering periods where depth is absent. Liquidations are `confirmed`, `probable`, or `unknown`; proxies are never called confirmed.

## No-lookahead contract

- Feature windows end before the event trade.
- Percentile distributions are fitted on earlier observations only.
- Entry is the next observable trade, never the signal price.
- Walk-forward folds are chronological and embargoed.
- Candidate selection cannot read the sealed holdout.

## Hyperliquid boundary

Only public information endpoints may be added. The codebase intentionally contains no exchange action client, signer, wallet secret, or withdrawal functionality. Any future trading integration requires explicit user approval and a separate security review.

