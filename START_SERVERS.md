# How to Start Frontend and Backend Servers

## Quick Start

### Option 1: Start Both Separately (Recommended)

**Terminal 1 - Backend (FastAPI):**
```bash
cd /Users/s20/Enesy-Dev
python3 -m uvicorn api_server:app --host 0.0.0.0 --port 8001 --reload
```

**Terminal 2 - Frontend (Node.js):**
```bash
cd /Users/s20/Enesy-Dev/RandDKnowledgeGraph
npm run dev
```

### Option 2: Start Backend in Background

**Start backend:**
```bash
cd /Users/s20/Enesy-Dev
python3 -m uvicorn api_server:app --host 0.0.0.0 --port 8001 --reload &
```

**Then start frontend:**
```bash
cd /Users/s20/Enesy-Dev/RandDKnowledgeGraph
npm run dev
```

## Server Ports

- **Backend (FastAPI)**: `http://localhost:8001`
- **Frontend (Node.js/Vite)**: `http://localhost:5006` (or auto-detected port)

## Verify Servers Are Running

**Check backend:**
```bash
curl http://localhost:8001/api/health
```

**Check frontend:**
```bash
curl http://localhost:5006
```

## Stop Servers

**Stop backend:**
```bash
lsof -ti:8001 | xargs kill -9
```

**Stop frontend:**
```bash
lsof -ti:5006 | xargs kill -9
```

**Or stop both:**
```bash
lsof -ti:8001 | xargs kill -9 && lsof -ti:5006 | xargs kill -9
```

## Troubleshooting

### Port Already in Use

If you get "Address already in use" error:

1. **Find the process:**
   ```bash
   lsof -ti:8001  # for backend
   lsof -ti:5006  # for frontend
   ```

2. **Kill it:**
   ```bash
   lsof -ti:8001 | xargs kill -9
   lsof -ti:5006 | xargs kill -9
   ```

### Backend Not Responding

1. Check if it's running:
   ```bash
   lsof -ti:8001
   ```

2. Check logs for errors

3. Restart:
   ```bash
   python3 -m uvicorn api_server:app --host 0.0.0.0 --port 8001 --reload
   ```

### Frontend Not Connecting to Backend

1. Make sure backend is running on port 8001
2. Check frontend config in `RandDKnowledgeGraph/client/src/lib/api-client.ts`
3. Backend URL should be: `http://localhost:8001`

## Development Workflow

1. **Start backend first** (port 8001)
2. **Then start frontend** (port 5006)
3. **Access frontend** at `http://localhost:5006`
4. Frontend will automatically connect to backend at `http://localhost:8001`

## Notes

- Backend uses **auto-reload** - restarts on code changes
- Frontend uses **Vite HMR** - hot module replacement for instant updates
- Both servers watch for file changes automatically

