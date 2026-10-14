# WebAgent operations runbook

Short procedures for production and uat. Expand in release 3.4.0 (deploy, roll back, key rotation).

## Backup and restore (`/app/data`)

The backend keeps task results, triage overrides, and security-guard collection data on the
`webagent-data` volume (`/app/data` in the pod).

### Option A — volume snapshots (preferred)

If the cluster has a CSI driver with `VolumeSnapshot`:

1. Create a `VolumeSnapshotClass` and a daily `VolumeSnapshot` of `webagent-data`, retaining 7.
2. **Restore:** scale the Deployment to 0, restore the PVC from the snapshot (cluster-specific), scale to 1, check `GET /health` and one old `GET /api/task/{id}`.

### Option B — CronJob archive

When snapshots are not available, apply `backend/kubernetes/backup-cronjob.yaml` after setting
`BACKUP_URI` on the CronJob to a destination your environment can write (for example an S3
prefix). The job tars `/app/data` read-only at 03:00 UTC. Copy the archive off the node or
wire an uploader in your fork of the manifest.

**Restore:**

1. `kubectl scale deployment/webagent-backend --replicas=0 -n <namespace>`
2. Replace the PVC contents from the tar (or attach a restored volume).
3. `kubectl scale deployment/webagent-backend --replicas=1 -n <namespace>`
4. Verify `/health` and an older task status.

Record the date and outcome of at least one restore drill in uat.

## Spending caps (Together AI and Tavily)

Set monthly limits in each provider dashboard for every key used in uat and prod (suggested:
twice the highest month so far, with alerts at 50% and 80%). Record the key **name** (not the
secret), limit, and date in `backend/PRODUCTION_READINESS.md`. Review quarterly.

## Metrics exposure

- Prometheus in the `monitoring` namespace scrapes port 8000 inside the cluster.
- The public Ingress answers **404** for `/metrics` (`deployment.yaml` server-snippet).
- `networkpolicy.yaml` limits port 8000 to the ingress controller namespace and `monitoring`.
  If your CNI does not enforce NetworkPolicies, rely on the Ingress rule and document that in
  this file.
