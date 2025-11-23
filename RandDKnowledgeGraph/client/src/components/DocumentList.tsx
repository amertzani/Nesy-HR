import { FileText, X, Loader2, CheckCircle2, AlertCircle, Bot } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import type { Document } from "@shared/schema";

interface DocumentListProps {
  documents: Document[];
  onRemove: (id: string) => void;
}

const statusIcons = {
  pending: Loader2,
  processing: Loader2,
  completed: CheckCircle2,
  error: AlertCircle,
};

const statusColors = {
  pending: "text-muted-foreground",
  processing: "text-primary",
  completed: "text-green-600",
  error: "text-destructive",
};

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

export function DocumentList({ documents, onRemove }: DocumentListProps) {
  if (documents.length === 0) {
    return null;
  }

  return (
    <Card className="p-6">
      <h3 className="text-lg font-semibold mb-4">Uploaded Documents</h3>
      <div className="space-y-3">
        {documents.map((doc) => {
          const Icon = statusIcons[doc.status];
          const isProcessing = doc.status === "processing";
          
          return (
            <div
              key={doc.id}
              className="flex items-center gap-4 p-3 rounded-md border"
              data-testid={`document-${doc.id}`}
            >
              <FileText className="h-5 w-5 text-muted-foreground flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between gap-2 mb-1">
                  <p className="text-sm font-medium truncate">{doc.name}</p>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    <span className="text-xs text-muted-foreground font-mono">
                      {formatFileSize(doc.size)}
                    </span>
                    <Icon
                      className={`h-4 w-4 ${statusColors[doc.status]} ${
                        isProcessing ? "animate-spin" : ""
                      }`}
                    />
                  </div>
                </div>
                <div className="flex items-center gap-2 mt-1">
                  {doc.agent_id && (
                    <Badge variant="outline" className="text-xs">
                      <Bot className="h-3 w-3 mr-1" />
                      {doc.agent_id}
                    </Badge>
                  )}
                  {doc.facts_extracted !== undefined && (
                    <span className="text-xs text-muted-foreground">
                      {doc.facts_extracted} facts
                    </span>
                  )}
                </div>
                {isProcessing && (
                  <Progress value={65} className="h-1 mt-2" />
                )}
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => onRemove(doc.id)}
                data-testid={`button-remove-${doc.id}`}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          );
        })}
      </div>
    </Card>
  );
}
