# Deployment Guide

## Production Deployment Procedures

This document provides step-by-step instructions for deploying the Rizal Thematic Exploration System to production.

---

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Frontend Deployment (Vercel)](#frontend-deployment-vercel)
3. [Backend Deployment (Railway)](#backend-deployment-railway)
4. [Database Setup (Supabase)](#database-setup-supabase)
5. [CDN Configuration (Cloudflare)](#cdn-configuration-cloudflare)
6. [Environment Variables](#environment-variables)
7. [CI/CD Pipeline](#cicd-pipeline)
8. [Monitoring & Alerts](#monitoring--alerts)
9. [Rollback Procedures](#rollback-procedures)
10. [Maintenance](#maintenance)

---

## Pre-Deployment Checklist

Before deploying to production, ensure:

### Code Quality
- [ ] All tests passing (`npm test`, `pytest`)
- [ ] No TypeScript errors (`npm run type-check`)
- [ ] Linting passed (`npm run lint`, `ruff check .`)
- [ ] Build succeeds locally (`npm run build`)

### Performance
- [ ] Lighthouse score >90 (mobile and desktop)
- [ ] Bundle size <500KB (check with `npm run analyze`)
- [ ] Images optimized (WebP format, proper sizes)
- [ ] Fonts subsetted (only Latin + Latin Extended)

### Security
- [ ] No API keys in code (use environment variables)
- [ ] CORS configured for production domain only
- [ ] Rate limiting enabled on backend
- [ ] SQL injection prevention (parameterized queries)
- [ ] XSS protection (React auto-escapes, but verify `dangerouslySetInnerHTML`)

### Accessibility
- [ ] WCAG 2.1 AA compliant (WAVE scan: 0 errors)
- [ ] Screen reader tested (NVDA/VoiceOver)
- [ ] Keyboard navigation works (no mouse required)
- [ ] Color contrast verified (all text >4.5:1)

### Content
- [ ] All text spell-checked (English and Filipino)
- [ ] Methodology page accurate
- [ ] Theme descriptions correct
- [ ] Loading tips reviewed

---

## Frontend Deployment (Vercel)

### Step 1: Connect GitHub Repository

1. **Sign up/Log in to Vercel**: https://vercel.com
2. **Import Project**:
   ```
   Dashboard → Add New → Project → Import Git Repository
   ```
3. **Select repository**: `your-username/rizal-explorer`
4. **Configure project**:
   - Framework Preset: Next.js
   - Root Directory: `./` (or `frontend/` if monorepo)
   - Build Command: `npm run build` (default)
   - Output Directory: `.next` (default)

### Step 2: Environment Variables

In Vercel dashboard → Settings → Environment Variables:

```bash
# Production
NEXT_PUBLIC_API_URL=https://api.rizal-explorer.com
NEXT_PUBLIC_ENV=production

# Optional: Analytics
NEXT_PUBLIC_SENTRY_DSN=https://your-sentry-dsn
NEXT_PUBLIC_VERCEL_ANALYTICS_ID=auto
```

**Important**: Check "All environments" or select "Production" only.

### Step 3: Deploy

```bash
# Automatic deployment
git add .
git commit -m "feat: ready for production"
git push origin main

# Vercel automatically:
# 1. Detects push to main
# 2. Runs build
# 3. Deploys to production
# 4. Invalidates CDN cache
```

**Deployment URL**: `https://rizal-explorer.vercel.app` (auto-generated)

### Step 4: Custom Domain

1. **Buy domain** (Namecheap, Google Domains, etc.): `rizal-explorer.com`
2. **Add to Vercel**:
   ```
   Project Settings → Domains → Add Domain
   ```
3. **Configure DNS**:
   - Add A record: `@` → `76.76.21.21` (Vercel IP)
   - Add CNAME: `www` → `cname.vercel-dns.com`
4. **Wait for SSL** (automatic, ~5 minutes)

**Verify**: Visit `https://rizal-explorer.com` (should work)

### Step 5: Production Optimizations

In `next.config.js`:

```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  // Compression
  compress: true,
  
  // Image optimization
  images: {
    formats: ['image/webp'],
    deviceSizes: [640, 750, 828, 1080, 1200],
  },
  
  // Security headers
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin',
          },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
```

**Re-deploy** after config changes.

---

## Backend Deployment (Railway)

### Option A: Railway (Recommended)

#### Step 1: Install Railway CLI

```bash
npm install -g @railway/cli

# Or use npx (no install)
npx @railway/cli
```

#### Step 2: Login & Initialize

```bash
cd backend

railway login
railway init

# Follow prompts:
# - Select "Create new project"
# - Project name: "rizal-backend"
# - Environment: "production"
```

#### Step 3: Add Services

```bash
# Add PostgreSQL
railway add postgresql

# Add Redis
railway add redis

# Railway automatically sets DATABASE_URL and REDIS_URL
```

#### Step 4: Configure Build

Create `railway.toml`:

```toml
[build]
builder = "nixpacks"
buildCommand = "pip install -r requirements.txt"

[deploy]
startCommand = "uvicorn app.main:app --host 0.0.0.0 --port $PORT"
restartPolicyType = "on_failure"
restartPolicyMaxRetries = 10
```

#### Step 5: Deploy

```bash
railway up

# Railway will:
# 1. Build Docker image
# 2. Install dependencies
# 3. Start Uvicorn
# 4. Expose on https://your-app.railway.app
```

#### Step 6: Run Migrations

```bash
# Connect to Railway PostgreSQL
railway run alembic upgrade head

# Seed data
railway run python scripts/seed_data.py
```

#### Step 7: Custom Domain for API

```bash
railway domain

# Follow prompts to add: api.rizal-explorer.com

# Configure DNS:
# CNAME: api → your-app.railway.app
```

**Verify**: Visit `https://api.rizal-explorer.com/health` (should return 200)

---

### Option B: Fly.io (Alternative)

#### Step 1: Install Fly CLI

```bash
curl -L https://fly.io/install.sh | sh
fly auth login
```

#### Step 2: Create Fly App

```bash
cd backend
fly launch

# Prompts:
# - App name: rizal-backend
# - Region: Manila (ams for Philippines edge)
# - PostgreSQL: Yes (create)
# - Redis: Yes (Upstash Redis add-on)
```

#### Step 3: Configure fly.toml

```toml
app = "rizal-backend"
primary_region = "sin" # Singapore (closest to Philippines)

[build]
  dockerfile = "Dockerfile"

[env]
  PORT = "8000"

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 1

[[vm]]
  cpu_kind = "shared"
  cpus = 1
  memory_mb = 1024
```

#### Step 4: Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download BERT model at build time (caching)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('xlm-roberta-base')"

# Copy application
COPY . .

# Run migrations
RUN alembic upgrade head

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Step 5: Deploy

```bash
fly deploy

# Access logs
fly logs
```

---

## Database Setup (Supabase)

### Why Supabase?
- Free tier: 500MB database
- pgvector support out-of-box
- Automatic backups
- Dashboard for debugging

### Step 1: Create Project

1. Sign up: https://supabase.com
2. **New Project**:
   - Name: rizal-explorer
   - Database Password: (save securely)
   - Region: Singapore (closest to Philippines)
3. Wait ~2 minutes for provisioning

### Step 2: Enable pgvector

```sql
-- In Supabase SQL Editor
CREATE EXTENSION IF NOT EXISTS vector;
```

### Step 3: Run Migrations

```bash
# Get connection string from Supabase dashboard
# Settings → Database → Connection string (Direct)

export DATABASE_URL="postgresql://postgres:password@db.xxx.supabase.co:5432/postgres"

# Run migrations locally
alembic upgrade head
```

### Step 4: Seed Data

```bash
python scripts/seed_data.py

# This will:
# 1. Load CSV files
# 2. Generate BERT embeddings
# 3. Insert into Supabase
# (Takes ~1-2 hours for full novels)
```

### Step 5: Connection Pooling (Production)

Use PgBouncer for connection pooling:

```bash
# Supabase provides this automatically
# Use "Transaction" mode for best performance

# Connection string for production:
DATABASE_URL="postgresql://postgres:password@db.xxx.supabase.co:6543/postgres?pgbouncer=true"
```

**Set in Railway/Fly.io environment variables.**

---

## CDN Configuration (Cloudflare)

### Step 1: Add Site to Cloudflare

1. Sign up: https://cloudflare.com
2. **Add a Site**: `rizal-explorer.com`
3. **Select Plan**: Free (sufficient for MVP)
4. **Update Nameservers** at your domain registrar:
   ```
   ns1.cloudflare.com
   ns2.cloudflare.com
   ```

### Step 2: DNS Configuration

In Cloudflare DNS settings:

```
Type  | Name | Content                      | Proxy
------|------|------------------------------|-------
A     | @    | 76.76.21.21 (Vercel IP)      | ✅ Yes
CNAME | www  | cname.vercel-dns.com         | ✅ Yes
CNAME | api  | your-app.railway.app         | ❌ No (DNS only)
```

**Why "Proxy" for @ and www?**
- Enables Cloudflare CDN
- Caches static assets
- DDoS protection

**Why "DNS only" for api?**
- Railway handles caching
- Avoid double-proxying

### Step 3: SSL/TLS Settings

```
SSL/TLS → Overview → Full (strict)
```

This ensures:
- Cloudflare ↔ Vercel: HTTPS
- User ↔ Cloudflare: HTTPS

### Step 4: Caching Rules

**Page Rules** (free tier: 3 rules):

1. **Cache static assets**:
   ```
   URL: *rizal-explorer.com/*.js
   Cache Level: Cache Everything
   Edge Cache TTL: 1 week
   ```

2. **Cache fonts**:
   ```
   URL: *rizal-explorer.com/fonts/*
   Cache Level: Cache Everything
   Edge Cache TTL: 1 month
   ```

3. **Don't cache API**:
   ```
   URL: api.rizal-explorer.com/*
   Cache Level: Bypass
   ```

### Step 5: Performance Optimizations

**Speed → Optimization**:
- [x] Auto Minify: JavaScript, CSS
- [x] Brotli compression
- [x] Rocket Loader: Off (breaks React)

**Verify**: Check headers with `curl -I https://rizal-explorer.com`

---

## Environment Variables

### Frontend (.env.production)

```bash
# API endpoint
NEXT_PUBLIC_API_URL=https://api.rizal-explorer.com

# Analytics
NEXT_PUBLIC_SENTRY_DSN=https://your-sentry-dsn
NEXT_PUBLIC_VERCEL_ANALYTICS_ID=auto

# Environment flag
NEXT_PUBLIC_ENV=production
```

### Backend (.env)

```bash
# Database (Supabase)
DATABASE_URL=postgresql://postgres:password@db.xxx.supabase.co:6543/postgres?pgbouncer=true

# Redis (Railway or Upstash)
REDIS_URL=redis://default:password@redis.railway.app:6379

# ML Model
BERT_MODEL_PATH=/app/models/xlm-roberta-base

# Security
SECRET_KEY=your-secret-key-here
ALLOWED_ORIGINS=https://rizal-explorer.com,https://www.rizal-explorer.com

# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
```

**Important**: Never commit `.env` to Git. Use `.env.example` instead.

---

## CI/CD Pipeline

### GitHub Actions Workflow

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  # Frontend Tests
  frontend-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '20'
      - run: npm ci
      - run: npm run lint
      - run: npm run type-check
      - run: npm test
      - run: npm run build

  # Backend Tests
  backend-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: cd backend && pip install -r requirements.txt
      - run: cd backend && ruff check .
      - run: cd backend && pytest

  # Deploy Frontend (only on main)
  deploy-frontend:
    needs: [frontend-test, backend-test]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: amondnet/vercel-action@v20
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
          vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}
          vercel-args: '--prod'

  # Deploy Backend (only on main)
  deploy-backend:
    needs: [frontend-test, backend-test]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: railway-deploy/action@v1
        with:
          railway-token: ${{ secrets.RAILWAY_TOKEN }}
```

### Setup GitHub Secrets

```bash
# Get Vercel token
npx vercel login
npx vercel --token

# Add to GitHub:
# Settings → Secrets → Actions → New repository secret
# - VERCEL_TOKEN
# - VERCEL_ORG_ID (from .vercel/project.json)
# - VERCEL_PROJECT_ID (from .vercel/project.json)

# Get Railway token
railway login
railway token

# Add to GitHub:
# - RAILWAY_TOKEN
```

---

## Monitoring & Alerts

### Sentry (Error Tracking)

#### Frontend Setup

```bash
npm install @sentry/nextjs
npx @sentry/wizard -i nextjs
```

**Configuration** (`sentry.client.config.ts`):

```typescript
import * as Sentry from '@sentry/nextjs';

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  environment: process.env.NEXT_PUBLIC_ENV,
  tracesSampleRate: 0.1, // 10% of transactions
  beforeSend(event) {
    // Don't send errors in development
    if (process.env.NODE_ENV === 'development') return null;
    return event;
  },
});
```

#### Backend Setup

```bash
pip install sentry-sdk
```

**Configuration** (`app/main.py`):

```python
import sentry_sdk

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    environment=os.getenv("ENVIRONMENT"),
    traces_sample_rate=0.1,
)
```

### Uptime Monitoring (UptimeRobot)

1. Sign up: https://uptimerobot.com (free)
2. **Add Monitor**:
   - Type: HTTPS
   - URL: `https://rizal-explorer.com`
   - Interval: 5 minutes
   - Alert Contacts: your-email@example.com
3. **Add API Monitor**:
   - URL: `https://api.rizal-explorer.com/health`
   - Expected: 200 OK

**Alerts**: Email/SMS when site is down >2 minutes

### Performance Monitoring (Vercel Analytics)

**Enable** in Vercel dashboard:
```
Project → Analytics → Enable
```

**Metrics tracked**:
- Real User Monitoring (RUM)
- Core Web Vitals (LCP, FID, CLS)
- Page load times
- Geographic distribution

**Access**: Vercel dashboard → Analytics tab

---

## Rollback Procedures

### Frontend Rollback (Vercel)

**Option 1: Vercel Dashboard**
```
Deployments → [Previous Deployment] → Promote to Production
```

**Option 2: Git Revert**
```bash
# Find last good commit
git log --oneline

# Revert to it
git revert abc123

# Push
git push origin main
# Vercel auto-deploys reverted version
```

**Time to rollback**: ~2 minutes

### Backend Rollback (Railway)

**Option 1: Railway Dashboard**
```
Deployments → [Previous Deployment] → Redeploy
```

**Option 2: Docker Tag**
```bash
railway rollback <deployment-id>
```

**Time to rollback**: ~5 minutes (rebuild + redeploy)

### Database Rollback (Supabase)

**Restore from backup**:
```bash
# Download backup from Supabase dashboard
# Settings → Database → Backups → Download

# Restore locally
psql $DATABASE_URL < backup.sql

# Or use Supabase dashboard:
# Settings → Database → Restore
```

**Time to rollback**: ~10-30 minutes (depending on DB size)

---

## Maintenance

### Weekly Tasks

**Monday**:
- [ ] Review Sentry errors (any critical?)
- [ ] Check uptime report (>99.5%?)
- [ ] Review analytics (traffic patterns)

**Friday**:
- [ ] Update dependencies (security patches only)
  ```bash
  npm audit
  npm update
  pip list --outdated
  ```

### Monthly Tasks

**Database**:
```sql
-- Vacuum (reclaim space)
VACUUM ANALYZE sentences;

-- Reindex (optimize queries)
REINDEX INDEX idx_sentences_embedding;
```

**Backup Verification**:
```bash
# Download latest backup
# Restore to staging environment
# Run smoke tests
```

**Performance Review**:
- [ ] Lighthouse audit (still >90?)
- [ ] Check bundle size (still <500KB?)
- [ ] Review slow queries (PostgreSQL logs)

### Quarterly Tasks

**Security Audit**:
- [ ] Update all dependencies (major versions)
- [ ] Review CORS settings
- [ ] Check for leaked secrets (git-secrets)
- [ ] Penetration testing (optional)

**Cost Review**:
- [ ] Vercel usage (still in free tier?)
- [ ] Railway usage (upgrade needed?)
- [ ] Cloudflare bandwidth

---

## Troubleshooting

### "502 Bad Gateway" on API

**Cause**: Backend not responding
**Fix**:
```bash
# Check Railway logs
railway logs

# Common issue: BERT model not loaded
# Solution: Increase memory limit (Railway dashboard)
```

### "504 Gateway Timeout"

**Cause**: Request taking >30s (Railway limit)
**Fix**:
```python
# Add timeout to slow queries
@router.post("/search", timeout=25.0)
async def search(...):
    # Ensure completes in <25s
```

### Images Not Loading

**Cause**: Cloudflare cache issue
**Fix**:
```bash
# Purge Cloudflare cache
# Dashboard → Caching → Purge Everything
```

### Database Connection Errors

**Cause**: Too many connections
**Fix**:
```python
# Use connection pooling (PgBouncer)
# Already enabled in Supabase connection string
# Verify: ?pgbouncer=true in DATABASE_URL
```

---

## Costs (Monthly)

| Service | Plan | Cost |
|---------|------|------|
| **Vercel** | Hobby | $0 (100GB bandwidth) |
| **Railway** | Starter | $5 (500 hrs, 1 instance) |
| **Supabase** | Free | $0 (500MB DB) |
| **Cloudflare** | Free | $0 (unlimited bandwidth) |
| **Domain** | Namecheap | $1/month ($12/year) |
| **Sentry** | Developer | $0 (5K events/month) |
| **UptimeRobot** | Free | $0 (50 monitors) |
| **Total** | | **~$6/month** |

**Upgrade path** (if traffic grows 10x):
- Vercel Pro: $20/month (1TB bandwidth)
- Railway Pro: $20/month (more compute)
- Supabase Pro: $25/month (8GB DB)
- **Total**: ~$65/month

---

## Post-Deployment Checklist

After deployment, verify:

**Functionality**:
- [ ] Homepage loads
- [ ] Search works (try 3+ queries)
- [ ] Results display correctly
- [ ] Mobile responsive
- [ ] All pages accessible

**Performance**:
- [ ] Lighthouse score >90 (mobile)
- [ ] Search response <2s
- [ ] Images load fast (WebP)

**Security**:
- [ ] HTTPS working (padlock icon)
- [ ] CORS only allows production domain
- [ ] Rate limiting active (test with >10 rapid requests)

**Monitoring**:
- [ ] Sentry capturing errors (trigger test error)
- [ ] Uptime monitor active (check email alerts)
- [ ] Analytics tracking (verify in dashboard)

**Backups**:
- [ ] Database backup successful
- [ ] Can restore from backup (test in staging)

---

## Support

**For deployment issues**:
- Vercel: https://vercel.com/support
- Railway: https://railway.app/help
- Supabase: https://supabase.com/support

**For code issues**:
- GitHub Issues: https://github.com/mematello/Rizal-Thematic-Exploration/issues

**Emergency contact**:
- Your email/phone for critical outages

---

**You're now live in production!** 🚀

Monitor closely for the first 48 hours, then settle into weekly maintenance rhythm.