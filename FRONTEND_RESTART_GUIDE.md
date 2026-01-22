# How to Restart the Frontend

## Quick Steps

### 1. Stop the Frontend (if running)

```bash
# Kill any process on port 5006
lsof -ti:5006 | xargs kill -9
```

### 2. Navigate to Frontend Directory

```bash
cd /Users/s20/Enesy-Dev/RandDKnowledgeGraph
```

### 3. Start the Frontend

```bash
npm run dev
```

### 4. Access the Frontend

Open your browser and go to:
```
http://localhost:5006
```

## Complete Restart (Both Backend & Frontend)

### Stop Everything First

```bash
# Stop backend (port 8001)
lsof -ti:8001 | xargs kill -9

# Stop frontend (port 5006)
lsof -ti:5006 | xargs kill -9
```

### Start Backend (Terminal 1)

```bash
cd /Users/s20/Enesy-Dev
python3 -m uvicorn api_server:app --host 0.0.0.0 --port 8001 --reload
```

### Start Frontend (Terminal 2)

```bash
cd /Users/s20/Enesy-Dev/RandDKnowledgeGraph
npm run dev
```

## Verify Servers Are Running

### Check Backend
```bash
curl http://localhost:8001/api/health
```
Should return: `{"status":"healthy",...}`

### Check Frontend
```bash
curl http://localhost:5006
```
Should return HTML content (or just open in browser)

## Troubleshooting

### Port Already in Use

If you get "Port 5006 is already in use":

```bash
# Find what's using the port
lsof -ti:5006

# Kill it
lsof -ti:5006 | xargs kill -9

# Then restart
cd /Users/s20/Enesy-Dev/RandDKnowledgeGraph
npm run dev
```

### Frontend Not Connecting to Backend

1. Make sure backend is running:
   ```bash
   curl http://localhost:8001/api/health
   ```

2. Check that frontend is configured to use port 8001:
   - File: `RandDKnowledgeGraph/client/src/lib/api-client.ts`
   - Should have: `http://localhost:8001`

### npm Dependencies Missing

If you get module errors:

```bash
cd /Users/s20/Enesy-Dev/RandDKnowledgeGraph
npm install
```

### Frontend Shows "Cannot connect to backend"

1. Verify backend is running on port 8001
2. Check browser console for errors
3. Make sure no firewall is blocking localhost connections

## Quick Reference

| Service | Port | URL | Command |
|---------|------|-----|---------|
| Backend | 8001 | http://localhost:8001 | `python3 -m uvicorn api_server:app --host 0.0.0.0 --port 8001 --reload` |
| Frontend | 5006 | http://localhost:5006 | `npm run dev` (from RandDKnowledgeGraph/) |

## Notes

- **Backend must be running first** before starting frontend
- Frontend auto-reloads on code changes (Hot Module Replacement)
- Backend auto-reloads on code changes (uvicorn --reload)
- Both servers watch for file changes automatically

