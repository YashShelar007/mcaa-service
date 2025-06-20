# Web User Interface

**Folder:** `src/ui/`  
**Purpose:** Single-page application for model upload, pipeline tracking, and download.

---

## Overview

Chapter 5 (“User Interface & Experience”) describes the UI design goals:

- **Single-Page Flow**: No full-page reloads; inline error banners; live progress updates
- **Responsive Layout**: Tailwind CSS ensures mobile and desktop support
- **Progress Polling**: Periodic `GET /status` calls to display Step Functions history
- **Download**: Final compressed model fetched via pre-signed URL

---

## File Structure

```

src/ui/
├── index.html        ← Main page (vanilla JS + Tailwind)
└── README.md         ← (You are here)

```

---

## Configuration

1. **API Endpoint**  
   In `index.html`, set:
   ```js
   const API_BASE = "<YOUR_API_INVOKE_URL>";
   ```

Example: `https://abcd1234.execute-api.us-west-2.amazonaws.com`

2. **CORS**
   Ensure your API Gateway CORS settings allow origin `*` (or your custom domain).

---

## Deployment

- **Local Testing**

  ```bash
  # Serve on localhost:8080
  npx http-server src/ui -c-1
  ```

- **Production**

  ```bash
  # If hosting in S3 + CloudFront:
  aws s3 sync src/ui/ s3://<YOUR_UI_BUCKET>/ --acl public-read
  ```

---

## Research Notes

- **Usability Evaluation** (Chapter 6): Inline error banners and spinners reduce user confusion during long-running backend steps.
- **Performance**: Tailwind via CDN is sufficient for prototype; for production, recommend a PostCSS build to minimize CSS size.
