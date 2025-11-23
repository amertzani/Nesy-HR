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
import { useForm } from "react-hook-form";
import type { GraphEdge } from "@shared/schema";
import { Trash2 } from "lucide-react";

interface EdgeEditDialogProps {
  edge: GraphEdge | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSave: (edge: GraphEdge) => void;
  onDelete: (edgeId: string) => void;
}

export function EdgeEditDialog({ edge, open, onOpenChange, onSave, onDelete }: EdgeEditDialogProps) {
  const { register, handleSubmit, reset } = useForm<GraphEdge>({
    defaultValues: edge || undefined,
  });

  useEffect(() => {
    if (edge) {
      reset(edge);
    }
  }, [edge, reset]);

  const onSubmit = (data: GraphEdge) => {
    onSave(data);
    onOpenChange(false);
  };

  const handleDelete = () => {
    if (edge) {
      onDelete(edge.id);
      onOpenChange(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent data-testid="dialog-edit-edge">
        <DialogHeader>
          <DialogTitle>Edit Connection</DialogTitle>
          <DialogDescription>
            Modify the relationship label or delete this connection
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit(onSubmit)}>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="label">Relationship Label</Label>
              <Input
                id="label"
                {...register("label")}
                placeholder="e.g., uses, requires, is_part_of"
                data-testid="input-edge-label"
              />
            </div>
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>Connection: <span className="font-mono">{edge?.source} → {edge?.target}</span></p>
            </div>
          </div>
          <DialogFooter className="gap-2">
            <Button
              type="button"
              variant="destructive"
              onClick={handleDelete}
              data-testid="button-delete-edge"
            >
              <Trash2 className="h-4 w-4 mr-2" />
              Delete Connection
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
              data-testid="button-cancel"
            >
              Cancel
            </Button>
            <Button type="submit" data-testid="button-save-edge">
              Save Changes
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
