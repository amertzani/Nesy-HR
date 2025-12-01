import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { Client } from "@gradio/client";

// HuggingFace Space configuration
const HF_SPACE_URL = process.env.VITE_HF_API_URL || 'https://asiminam-xnesy.hf.space';

// Cache the Gradio client
let gradioClient: any = null;

/**
 * Get or create Gradio client connection
 */
async function getGradioClient() {
  if (!gradioClient) {
    gradioClient = await Client.connect(HF_SPACE_URL);
  }
  return gradioClient;
}

/**
 * Call a Gradio API function on HuggingFace Space
 * Uses the official @gradio/client library
 */
async function callGradioApi(apiName: string, params: any[] = []): Promise<any> {
  const client = await getGradioClient();
  const result = await client.predict(`/${apiName}`, params);
  // Gradio returns result.data as an array, we want the first element
  return Array.isArray(result.data) ? result.data[0] : result.data;
}

export async function registerRoutes(app: Express): Promise<Server> {
  // Test endpoint to check environment variables
  app.get("/api/test-env", (req, res) => {
    res.json({
      hasViteHfApiUrl: !!process.env.VITE_HF_API_URL,
      value: process.env.VITE_HF_API_URL || "not set"
    });
  });

  // ==========================================================
  // HuggingFace API Proxy Routes
  // ==========================================================

  // Get knowledge base
  app.get("/api/hf/knowledge-base", async (req, res) => {
    try {
      const data = await callGradioApi('get_knowledge_base', []);
      res.json({ success: true, data });
    } catch (error) {
      console.error('HF API error:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  });

  // Create fact
  app.post("/api/hf/facts", async (req, res) => {
    try {
      const { subject, predicate, object, source } = req.body;
      const data = await callGradioApi('create_fact', [
        subject || '',
        predicate || '',
        object || '',
        source || 'API'
      ]);
      res.json({ success: true, data });
    } catch (error) {
      console.error('HF API error:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  });

  // Update fact
  app.patch("/api/hf/facts/:id", async (req, res) => {
    try {
      const { id } = req.params;
      const { subject, predicate, object } = req.body;
      const data = await callGradioApi('update_fact', [
        id,
        subject || '',
        predicate || '',
        object || ''
      ]);
      res.json({ success: true, data });
    } catch (error) {
      console.error('HF API error:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  });

  // Delete fact
  app.delete("/api/hf/facts/:id", async (req, res) => {
    try {
      const { id } = req.params;
      const data = await callGradioApi('delete_fact', [id]);
      res.json({ success: true, data });
    } catch (error) {
      console.error('HF API error:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  });

  // Get knowledge graph
  app.get("/api/hf/graph", async (req, res) => {
    try {
      const data = await callGradioApi('get_graph', []);
      res.json({ success: true, data });
    } catch (error) {
      console.error('HF API error:', error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  });

  const httpServer = createServer(app);

  return httpServer;
}
