# Deployment Guide: Vercel Frontend + Render Backend

## Backend (Render)

1. Create a new Web Service on Render
   - Select **Docker** and point to `render.yaml` in repo root
   - Or manually choose your repo and set Dockerfile path to `120 transfer now/120 transfer now/Dockerfile`

2. Set environment variables:
   - `PORT=8080`
   - `STORAGE_BACKEND=s3`
   - `AWS_S3_BUCKET=your-bucket`
   - `AWS_REGION=ap-southeast-1`
   - `AWS_S3_PREFIX=pet-id`
   - `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` (as **Secret Files/Env**)

3. Deploy and note the Service URL (e.g., `https://pet-id-malaysia.onrender.com`)

## Frontend (Vercel)

1. In Vercel, import the repository and select the `frontend` folder
2. Set Project Environment Variable:
   - `API_BASE_URL=https://pet-id-malaysia.onrender.com`
3. Deploy

## How It Works

- Frontend routes are proxied by `frontend/api/proxy.js` to your backend using `API_BASE_URL`
- Images are stored on S3 when configured; otherwise local storage is used (not recommended for serverless)

## Optional migration

- Call `GET /api/fix_reunited` on backend to migrate any lingering reunited images to the `reunited` store

