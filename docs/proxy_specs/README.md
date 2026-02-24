# Proxy Specifications by Sector v0.1.0

Sector-specific proxy ranges, sources, and validation rules.

## Structure

```
proxy_specs/
├── README.md (this file)
├── finance/proxies.yaml
├── ai/proxies.yaml
├── critical_infrastructure/proxies.yaml
└── MANIFEST.json
```

## Manifest

Each sector spec is versioned and hashed:

```json
{
  "version": "v0.1.0",
  "sectors": {
    "finance": {
      "version": "v0.1.0",
      "hash_sha256": "abc123...",
      "last_updated": "2026-02-24",
      "audit_outputs": "reports/calibration/finance_audit.json"
    },
    ...
  }
}
```

## Adding a New Sector

1. Create `docs/proxy_specs/<sector>/proxies.yaml`
2. Specify ranges, sources, and validation rules
3. Run `python3 tools/validate_proxy_specs.py --sector <sector>`
4. Update MANIFEST.json with hash and version

## Sectors Included

- **finance/** – Banking, trading, payments systems
- **ai/** – AI/ML systems, LLMs, autonomous agents
- **critical_infrastructure/** – Power, telecom, transportation

