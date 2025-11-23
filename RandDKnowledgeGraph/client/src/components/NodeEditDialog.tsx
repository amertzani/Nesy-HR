import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useForm } from "react-hook-form";
import type { GraphNode } from "@shared/schema";

interface NodeEditDialogProps {
  node: GraphNode | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSave: (node: GraphNode) => void;
}

export function NodeEditDialog({ node, open, onOpenChange, onSave }: NodeEditDialogProps) {
  const { register, handleSubmit, reset, setValue, watch } = useForm<GraphNode>({
    defaultValues: node || undefined,
  });

  const nodeType = watch("type");

  useEffect(() => {
    if (node) {
      reset(node);
    }
  }, [node, reset]);

  const onSubmit = (data: GraphNode) => {
    onSave(data);
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent data-testid="dialog-edit-node">
        <DialogHeader>
          <DialogTitle>Edit Graph Node</DialogTitle>
          <DialogDescription>
            Update the node label or type. Changes will reflect in the knowledge base.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit(onSubmit)}>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="label">Node Label</Label>
              <Input
                id="label"
                {...register("label")}
                className="font-medium"
                data-testid="input-node-label"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="type">Node Type</Label>
              <Select
                value={nodeType}
                onValueChange={(value) => setValue("type", value)}
              >
                <SelectTrigger data-testid="select-node-type">
                  <SelectValue placeholder="Select type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="concept">Concept</SelectItem>
                  <SelectItem value="entity">Entity</SelectItem>
                  <SelectItem value="process">Process</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Connections</Label>
              <div className="text-sm text-muted-foreground">
                {node?.connections || 0} connections in the graph
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
              data-testid="button-cancel"
            >
              Cancel
            </Button>
            <Button type="submit" data-testid="button-save-node">
              Save Changes
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
