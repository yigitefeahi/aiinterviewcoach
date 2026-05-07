# AI Coach — Frontend

Next.js (App Router) UI for the AI Interview Coach: registration, interview setup (text/audio/video), live session, and results.

## Run locally

```bash
npm install
npm run dev
```

App: [http://localhost:3000](http://localhost:3000) with default settings.

## API base URL

The browser calls the backend defined by **`NEXT_PUBLIC_API_BASE`**. If unset, client code defaults to `http://<hostname>:8000` (see `src/lib/api.ts`).

## Auth

- Login/register responses set an **HttpOnly** cookie (`access_token`) on the API origin.
- `apiFetch` uses **`credentials: "include"`** so that cookie is sent on same-site / cross-port localhost setups.
- Optional legacy JWT in **`localStorage`** is still read for `Authorization` if present (e.g. old tabs).

## Security headers

`next.config.ts` sets basic headers (`X-Content-Type-Options`, `Referrer-Policy`, `Permissions-Policy`). Tight **Content-Security-Policy** is left to your hosting profile (Next dev often needs relaxed script rules).

## Scripts

| Command | Purpose |
|--------|---------|
| `npm run dev` | Development server |
| `npm run build` | Production build |
| `npm run lint` | ESLint |

Project-wide docs: **`../README.md`**, **`../backend/QUALITY.md`**, **`../backend/docs/EVALUATION_METHODOLOGY.md`**.
