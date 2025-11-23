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
import { Textarea } from "@/components/ui/textarea";
import { useForm } from "react-hook-form";
import type { Fact } from "@shared/schema";

interface FactEditDialogProps {
  fact: Fact | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSave: (fact: Fact) => void;
}

export function FactEditDialog({ fact, open, onOpenChange, onSave }: FactEditDialogProps) {
  const { register, handleSubmit, reset } = useForm<Fact>({
    defaultValues: {
      id: fact?.id || "",
      subject: fact?.subject || "",
      predicate: fact?.predicate || "",
      object: fact?.object || "",
      source: fact?.source || "",
      details: fact?.details || "",
    },
  });

  useEffect(() => {
    if (fact) {
      console.log('FactEditDialog: Loading fact:', fact);
      console.log('FactEditDialog: Details value:', fact.details, 'Type:', typeof fact.details);
      reset({
        id: fact.id,
        subject: fact.subject,
        predicate: fact.predicate,
        object: fact.object,
        source: fact.source || "",
        details: fact.details ?? "", // Use nullish coalescing to handle null/undefined
      });
    }
  }, [fact, reset]);

  const onSubmit = (data: Fact) => {
    onSave(data);
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent data-testid="dialog-edit-fact">
        <DialogHeader>
          <DialogTitle>Edit Fact</DialogTitle>
          <DialogDescription>
            Update the subject, predicate, or object of this knowledge fact.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit(onSubmit)}>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="subject">Subject</Label>
              <Input
                id="subject"
                {...register("subject")}
                className="font-mono"
                data-testid="input-subject"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="predicate">Predicate</Label>
              <Input
                id="predicate"
                {...register("predicate")}
                className="font-mono"
                data-testid="input-predicate"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="object">Object</Label>
              <Input
                id="object"
                {...register("object")}
                className="font-mono"
                data-testid="input-object"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="details">Details / Comment (Optional)</Label>
              <Textarea
                id="details"
                {...register("details")}
                placeholder="Add additional context, details, or comments about this fact..."
                rows={4}
                className="resize-none"
                data-testid="input-details"
              />
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
            <Button type="submit" data-testid="button-save">
              Save Changes
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
