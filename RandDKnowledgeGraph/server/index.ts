import express, { type Request, Response, NextFunction } from "express";
import { registerRoutes } from "./routes";
import { setupVite, serveStatic, log } from "./vite";
import * as net from "net";
import { exec } from "child_process";

const app = express();

// Set environment mode based on NODE_ENV
// Trim whitespace in case of trailing spaces
const env = (process.env.NODE_ENV || "development").trim();
app.set("env", env);

// Debug: Log the environment
console.log("Environment detected:", env);
console.log("NODE_ENV:", process.env.NODE_ENV);
console.log("app.get('env'):", app.get("env"));

declare module 'http' {
  interface IncomingMessage {
    rawBody: unknown
  }
}
app.use(express.json({
  verify: (req, _res, buf) => {
    req.rawBody = buf;
  }
}));
app.use(express.urlencoded({ extended: false }));

app.use((req, res, next) => {
  const start = Date.now();
  const path = req.path;
  let capturedJsonResponse: Record<string, any> | undefined = undefined;

  const originalResJson = res.json;
  res.json = function (bodyJson, ...args) {
    capturedJsonResponse = bodyJson;
    return originalResJson.apply(res, [bodyJson, ...args]);
  };

  res.on("finish", () => {
    const duration = Date.now() - start;
    if (path.startsWith("/api")) {
      let logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;
      if (capturedJsonResponse) {
        logLine += ` :: ${JSON.stringify(capturedJsonResponse)}`;
      }

      if (logLine.length > 80) {
        logLine = logLine.slice(0, 79) + "…";
      }

      log(logLine);
    }
  });

  next();
});

(async () => {
  const server = await registerRoutes(app);

  app.use((err: any, _req: Request, res: Response, _next: NextFunction) => {
    const status = err.status || err.statusCode || 500;
    const message = err.message || "Internal Server Error";

    res.status(status).json({ message });
    throw err;
  });

  // importantly only setup vite in development and after
  // setting up all the other routes so the catch-all route
  // doesn't interfere with the other routes
  // Check multiple ways to detect development mode
  // Trim whitespace to handle trailing spaces
  const nodeEnv = process.env.NODE_ENV ? process.env.NODE_ENV.trim() : null;
  const appEnv = app.get("env") ? String(app.get("env")).trim() : null;
  // Default to development if not explicitly set to production
  const isDevelopment = !nodeEnv || nodeEnv === "development" || appEnv === "development" || (!nodeEnv && !appEnv);
  
  console.log("Development check:", { nodeEnv, appEnv, isDevelopment });
  
  if (isDevelopment) {
    console.log("Setting up Vite dev server...");
    await setupVite(app, server);
  } else {
    console.log("Serving static files...");
    serveStatic(app);
  }

  // ALWAYS serve the app on the port specified in the environment variable PORT
  // Other ports are firewalled. Default to 5005 if not specified.
  // this serves both the API and the client.
  // It is the only port that is not firewalled.
  let port = parseInt(process.env.PORT || '5006', 10);

  // Try to find an available port if the default is busy
  const findAvailablePort = (startPort: number): Promise<number> => {
    return new Promise((resolve, reject) => {
      const server = net.createServer();
      server.listen(startPort, () => {
        const address = server.address();
        const foundPort = (address && typeof address === 'object' && 'port' in address) 
          ? address.port 
          : startPort;
        server.close(() => resolve(foundPort));
      });
      server.on('error', (err: any) => {
        if (err.code === 'EADDRINUSE') {
          // Try next port
          findAvailablePort(startPort + 1).then(resolve).catch(reject);
        } else {
          reject(err);
        }
      });
    });
  };

  // Use 0.0.0.0 universally; it's safer and works on all platforms
  // 'localhost' or '127.0.0.1' can cause ENOTSUP on some Windows setups
  const host = '0.0.0.0';

  try {
    // Try to find available port BEFORE starting the server
    try {
      const requestedPort = port;
      port = await findAvailablePort(port);
      if (port !== requestedPort) {
        console.log(`⚠️  Port ${requestedPort} was busy, using port ${port} instead`);
      }
    } catch (err) {
      console.error('⚠️  Could not find available port, trying default:', err);
      // Continue with default port - uvicorn will handle the error
    }

    // Start server with error handling for port conflicts
    server.listen({ port, host }, () => {
      log(`✅ Server running at http://localhost:${port}`);
      
      // Automatically open Safari browser
      const url = `http://localhost:${port}`;
      exec(`open -a Safari "${url}"`, (error: any) => {
        if (error) {
          console.log(`⚠️  Could not open Safari automatically: ${error.message}`);
          console.log(`   Please open ${url} manually in your browser`);
        } else {
          console.log(`🌐 Opened ${url} in Safari`);
        }
      });
    });
    
    server.on('error', (err: any) => {
      if (err.code === 'EADDRINUSE') {
        console.error(`❌ Port ${port} is already in use.`);
        console.error(`💡 Try one of these solutions:`);
        console.error(`   1. Kill the process using port ${port}: lsof -ti:${port} | xargs kill -9`);
        console.error(`   2. Use a different port: PORT=5006 npm run dev`);
        console.error(`   3. The server should auto-detect an available port, but it failed.`);
        process.exit(1);
      } else {
        console.error('❌ Failed to start server:', err);
        process.exit(1);
      }
    });
  } catch (err: any) {
    console.error('❌ Failed to start server:', err);
    process.exit(1);
  }
})();
